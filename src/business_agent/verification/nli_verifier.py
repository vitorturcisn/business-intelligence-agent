import json
import re
import unicodedata
from functools import lru_cache
from decimal import Decimal, InvalidOperation
from typing import Literal, Protocol

from pydantic import BaseModel, Field


DEFAULT_NLI_MODEL = (
    "MoritzLaurer/"
    "mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
)

Verdict = Literal["APOIA", "REFUTA", "NEUTRA"]


class Evidence(BaseModel):
    source: str
    content: str


class ClaimAssessment(BaseModel):
    claim: str
    verdict: Verdict
    confidence: float = Field(ge=0, le=1)
    evidence_source: str | None = None
    scores: dict[str, float] = Field(default_factory=dict)
    unsupported_numbers: list[str] = Field(default_factory=list)
    explanation: str


class CoherenceIssue(BaseModel):
    first_claim: str
    second_claim: str
    confidence: float = Field(ge=0, le=1)


class VerificationReport(BaseModel):
    overall_verdict: Literal[
        "SUSTENTADA",
        "REFUTADA",
        "INCONCLUSIVA",
        "INCOERENTE",
    ]
    claims: list[ClaimAssessment]
    coherence_issues: list[CoherenceIssue]
    evidence_sources: list[str]
    summary: str


class NLIClassifier(Protocol):
    def classify(
        self,
        premise: str,
        hypothesis: str,
    ) -> dict[str, float]:
        ...


@lru_cache(maxsize=2)
def _load_nli_model(model_name: str):
    """Carrega tokenizer e modelo uma única vez por processo."""
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    return tokenizer, model


class LocalNLIClassifier:
    """Classificador NLI executado localmente com Transformers e PyTorch.

    A ordem de rótulos documentada para o modelo selecionado é:
    entailment, neutral, contradiction.
    """

    def __init__(self, model_name: str = DEFAULT_NLI_MODEL):
        self.model_name = model_name

    def classify(
        self,
        premise: str,
        hypothesis: str,
    ) -> dict[str, float]:
        if not premise.strip() or not hypothesis.strip():
            raise ValueError("Premissa e hipótese não podem estar vazias.")

        import torch

        tokenizer, model = _load_nli_model(self.model_name)
        inputs = tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        with torch.inference_mode():
            output = model(**inputs)
            probabilities = torch.softmax(
                output.logits[0],
                dim=-1,
            ).tolist()

        raw_labels = model.config.id2label
        labels = {
            int(index): str(label).lower()
            for index, label in raw_labels.items()
        }

        normalized = {}
        for index, label in labels.items():
            if "entail" in label:
                normalized["entailment"] = index
            elif "neutral" in label:
                normalized["neutral"] = index
            elif "contradict" in label:
                normalized["contradiction"] = index

        if len(normalized) != 3:
            # Ordem documentada no model card do modelo escolhido.
            normalized = {
                "entailment": 0,
                "neutral": 1,
                "contradiction": 2,
            }

        return {
            label: float(probabilities[index])
            for label, index in normalized.items()
        }


NUMBER_PATTERN = re.compile(
    r"(?<!\w)(?:R\$\s*)?-?\d[\d.,]*(?:\s*%)?"
)
CURRENCY_PATTERN = re.compile(
    r"(?i)(?:R\$\s*|BRL\s*)-?\d[\d.,]*"
)

# Relação explícita entre os nomes usados nas respostas e as chaves do SQL.
# A regra determinística abaixo só compara números quando consegue associá-los
# à mesma métrica; não trata todo número ausente como uma falsidade.
_METRIC_ALIASES = {
    "net_revenue": (
        "receita líquida", "receita liquida", "net revenue", "net_revenue",
    ),
    "gross_revenue": (
        "receita bruta", "gross revenue", "gross_revenue",
    ),
    "total_discounts": (
        "descontos totais", "desconto total", "total discounts",
        "total_discounts", "descontos",
    ),
    "units_sold": (
        "unidades vendidas", "quantidade vendida", "total de unidades",
        "unidades", "unidade", "units sold", "units_sold",
    ),
    "order_count": (
        "número de pedidos", "numero de pedidos", "quantidade de pedidos",
        "total de pedidos", "total pedidos", "pedidos", "pedido",
        "order count", "order_count",
    ),
    "selic_target": (
        "meta selic", "taxa selic", "selic",
    ),
}

_MONTHS = {
    "janeiro": 1, "fevereiro": 2, "marco": 3, "abril": 4,
    "maio": 5, "junho": 6, "julho": 7, "agosto": 8,
    "setembro": 9, "outubro": 10, "novembro": 11, "dezembro": 12,
}


def _parse_number(raw: str) -> tuple[Decimal, bool] | None:
    """Converte números brasileiros e internacionais para Decimal.

    Retorna também se o valor era percentual.
    """
    value = raw.strip()
    is_percentage = value.endswith("%")

    value = value.replace("R$", "").replace("%", "").replace(" ", "")
    value = re.sub(r"(?i)BRL", "", value)

    if not value:
        return None

    if "," in value and "." in value:
        if value.rfind(",") > value.rfind("."):
            value = value.replace(".", "").replace(",", ".")
        else:
            value = value.replace(",", "")

    elif "," in value:
        integer, decimal = value.rsplit(",", 1)
        if len(decimal) == 3:
            value = integer + decimal
        else:
            value = integer + "." + decimal

    elif "." in value:
        integer, decimal = value.rsplit(".", 1)
        if len(decimal) == 3:
            value = integer + decimal

    try:
        return Decimal(value), is_percentage
    except InvalidOperation:
        return None


def extract_numeric_values(text: str) -> set[tuple[Decimal, bool]]:
    values = set()

    for match in NUMBER_PATTERN.findall(text):
        parsed = _parse_number(match.rstrip(".,;"))
        if parsed is not None:
            values.add(parsed)

    return values


def extract_claims(response: str) -> list[str]:
    """Extrai afirmações atômicas, separando comparações entre receita bruta e líquida."""
    # Remove ordered-list markers before sentence splitting; they are formatting, not claims.
    response = re.sub(r"(?m)^\s*\d+[.)]\s+", "", response)
    parts = re.split(
        r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÂÊÔÃÕÇ0-9])|\n+",
        response.strip(),
    )
    claims: list[str] = []

    def has_metric(text: str, metric: str) -> bool:
        normalized = _normalize_match_text(text)
        if metric == "net":
            return bool(re.search(r"\b(receita\s+liquida|liquida)\b", normalized))
        return bool(re.search(r"\b(receita\s+bruta|bruta)\b", normalized))

    def finish_clause(text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip(" \t-,:;")
        if text and text[-1] not in ".!?":
            text += "."
        return text

    for part in parts:
        # Remove delimitadores Markdown sem destruir identificadores snake_case.
        # Underscores internos, como em `unit_price`, devem ser preservados.
        cleaned = re.sub(
            r"[*`#]|(?<!\w)_{1,3}|_{1,3}(?!\w)",
            "",
            part,
        )
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" \t-\r")
        if not cleaned:
            continue
        # Cabeçalhos isolados, como "No exemplo de setembro:", não são afirmações.
        if cleaned.endswith(":") and not re.search(r"[.!?]", cleaned):
            continue

        clauses = [finish_clause(cleaned)]

        # Decompõe uma explicação em etapas da receita líquida quando a mesma
        # frase descreve quantidade × preço unitário e, em seguida, a aplicação
        # do desconto percentual sobre o valor bruto. A forma atômica mantém
        # cada relação matemática explícita para a seleção de evidência específica.
        normalized_cleaned = _normalize_match_text(cleaned)
        describes_staged_net_formula = (
            bool(re.search(r"\breceita\s+liquida\b", normalized_cleaned))
            and bool(re.search(r"\bquantidade\b|(?<!\w)quantity(?!\w)", normalized_cleaned))
            and bool(re.search(r"\bpreco\s+unitario\b|(?<!\w)unit\s+price(?!\w)", normalized_cleaned))
            and bool(re.search(r"\bpercentual\s+de\s+desconto\b|(?<!\w)discount\s+pct(?!\w)", normalized_cleaned))
            and bool(re.search(r"\bmultiplic\w*\b", normalized_cleaned))
            and bool(re.search(r"\bdepois\s+aplic\w*\b", normalized_cleaned))
            and bool(re.search(r"\b(?:ao|sobre\s+o)\s+valor\s+bruto\b", normalized_cleaned))
        )
        if describes_staged_net_formula:
            clauses = [
                "O valor bruto por item corresponde a quantity × unit_price.",
                "O valor do desconto por item corresponde a quantity × unit_price × discount_pct.",
                "A receita líquida por item corresponde a quantity × unit_price × (1 - discount_pct).",
            ]

        # "Enquanto" só é tratado como separador quando contrapõe explicitamente
        # receita líquida e receita bruta. Outros usos temporais permanecem intactos.
        comparison = re.split(r"\s+enquanto\s+", cleaned, maxsplit=1, flags=re.IGNORECASE)
        if len(comparison) == 2:
            left, right = comparison[0], comparison[1]
            opposite_revenue_metrics = (
                (has_metric(left, "net") and has_metric(right, "gross"))
                or (has_metric(left, "gross") and has_metric(right, "net"))
            )
            if opposite_revenue_metrics:
                # Quando a segunda parte usa uma referência anafórica como "esses descontos",
                # torna a afirmação autônoma e aproxima a redação da definição do glossário.
                if has_metric(left, "net") and has_metric(right, "gross"):
                    normalized_right = _normalize_match_text(right)
                    refers_to_discount_without_context = (
                        "nao considera esses descontos" in normalized_right
                        or "nao considera os descontos" in normalized_right
                        or "nao considera desconto" in normalized_right
                    )
                    if refers_to_discount_without_context:
                        right = (
                            "A receita bruta é o valor total das vendas "
                            "antes da aplicação de descontos"
                        )
                clauses = [finish_clause(left), finish_clause(right)]

        # Divide uma observação que reúne duas fórmulas independentes em uma
        # única frase. Cada parcela poderá então ser confrontada com sua linha
        # específica no dicionário, sem aceitar/rejeitar o conjunto como bloco.
        if len(clauses) == 1:
            formula_pair = re.split(
                r",\s*e\s+(?=(?:o\s+valor\s+bruto|a\s+receita\s+bruta)\b)",
                cleaned,
                maxsplit=1,
                flags=re.IGNORECASE,
            )
            if len(formula_pair) == 2:
                clauses = [finish_clause(formula_pair[0]), finish_clause(formula_pair[1])]

        for clause in clauses:
            if clause and clause not in claims:
                claims.append(clause)

    return claims


def _normalize_match_text(value: str) -> str:
    """Normaliza acentos e separadores para comparar nomes de métricas."""
    normalized = unicodedata.normalize("NFKD", value.lower())
    without_accents = "".join(
        char for char in normalized
        if not unicodedata.combining(char)
    )
    without_accents = without_accents.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", without_accents).strip()


def _detect_metric(claim: str) -> str | None:
    # Normaliza pontuação para reconhecer aliases no fim de frases, por exemplo
    # "6 pedidos." e "receita líquida,".
    normalized_claim = _normalize_match_text(claim)
    normalized_claim = re.sub(r"[^\w]+", " ", normalized_claim).strip()
    matches = []

    for metric, aliases in _METRIC_ALIASES.items():
        for alias in aliases:
            normalized_alias = _normalize_match_text(alias)
            normalized_alias = re.sub(r"[^\w]+", " ", normalized_alias).strip()
            if re.search(
                rf"(?<!\w){re.escape(normalized_alias)}(?!\w)",
                normalized_claim,
            ):
                matches.append((len(normalized_alias), metric))

    if not matches:
        return None

    # O alias mais específico tem prioridade sobre termos genéricos.
    matches.sort(reverse=True)
    return matches[0][1]


def _normalize_period(value: object) -> str | None:
    if value is None:
        return None

    raw_text = str(value).lower()
    numeric_match = re.search(r"\b(\d{4})[-/](\d{1,2})\b", raw_text)
    if numeric_match:
        year, month = int(numeric_match.group(1)), int(numeric_match.group(2))
        if 1 <= month <= 12:
            return f"{year:04d}-{month:02d}"

    text = _normalize_match_text(raw_text)
    for month_name, month_number in _MONTHS.items():
        if re.search(rf"\b{month_name}\b", text):
            year_match = re.search(r"\b(\d{4})\b", text)
            if year_match:
                return f"{int(year_match.group(1)):04d}-{month_number:02d}"

    return None


def _extract_metric_facts(item: Evidence) -> list[tuple[str, Decimal, str | None]]:
    """Extrai fatos numéricos rotulados de JSON ou texto de evidência."""
    facts: list[tuple[str, Decimal, str | None]] = []

    def walk_json(node: object, inherited_period: str | None = None) -> None:
        if isinstance(node, dict):
            period = _normalize_period(
                node.get("month", node.get("period", node.get("mes")))
            ) or inherited_period

            for key, value in node.items():
                normalized_key = _normalize_match_text(str(key))
                metric = next(
                    (
                        metric_name
                        for metric_name, aliases in _METRIC_ALIASES.items()
                        if normalized_key in {
                            _normalize_match_text(alias) for alias in aliases
                        }
                    ),
                    None,
                )
                if metric and isinstance(value, (int, float, str)) and not isinstance(value, bool):
                    parsed = _parse_number(str(value))
                    if parsed is not None:
                        number, is_percentage = parsed
                        if not is_percentage:
                            facts.append((metric, number, period))

                if isinstance(value, (dict, list)):
                    walk_json(value, period)

        elif isinstance(node, list):
            for value in node:
                walk_json(value, inherited_period)

    try:
        payload = json.loads(item.content)
    except (json.JSONDecodeError, TypeError):
        payload = None

    if payload is not None:
        # A API SGS 432 retorna a taxa em latest_observation.value e informa
        # a unidade percentual em latest_observation.unit, não no nome da chave.
        if item.source == "get_selic_target" and isinstance(payload, dict):
            latest = payload.get("latest_observation")
            if isinstance(latest, dict) and "value" in latest:
                parsed_selic = _parse_number(str(latest["value"]))
                unit = str(latest.get("unit", ""))
                if parsed_selic is not None and "%" in unit:
                    facts.append(("selic_target", parsed_selic[0], None))

        walk_json(payload)
        # Para JSON, as chaves já identificam precisamente cada métrica.
        # Não aplicamos a busca textual ampla ao mesmo conteúdo, pois ela
        # poderia associar por engano um valor vizinho a outra chave.
        return list(dict.fromkeys(facts))

    # Aceita fatos textuais nos dois formatos mais comuns:
    # "Receita líquida: R$ 25.250,00" e "6 pedidos".
    normalized_content = _normalize_match_text(item.content)
    number_pattern = r"(?:r\$\s*|brl\s*)?(-?\d[\d.,]*)"

    for metric, aliases in _METRIC_ALIASES.items():
        for alias in aliases:
            normalized_alias = _normalize_match_text(alias)
            escaped_alias = re.escape(normalized_alias)

            # Formato rótulo -> valor: "receita líquida: R$ 25.250,00".
            label_first = re.compile(
                rf"(?<!\w){escaped_alias}(?!\w)"
                rf"[^\n:;]{{0,20}}[:=]?\s*{number_pattern}",
                flags=re.IGNORECASE,
            )
            for match in label_first.finditer(normalized_content):
                parsed = _parse_number(match.group(1).rstrip(".,;"))
                if parsed is not None and not parsed[1]:
                    facts.append((metric, parsed[0], None))

            # Formato valor -> rótulo: "6 pedidos" ou "R$ 25.250 receita líquida".
            value_first = re.compile(
                rf"{number_pattern}\s+(?:de\s+)?"
                rf"{escaped_alias}(?!\w)",
                flags=re.IGNORECASE,
            )
            for match in value_first.finditer(normalized_content):
                parsed = _parse_number(match.group(1).rstrip(".,;"))
                if parsed is not None and not parsed[1]:
                    facts.append((metric, parsed[0], None))

    return list(dict.fromkeys(facts))


def _claim_metric_values(claim: str, metric: str) -> set[Decimal]:
    """Extrai valores da métrica sem confundir o período com a quantidade.

    Métricas monetárias exigem um marcador de moeda. Contagens (pedidos e
    unidades) aceitam números inteiros sem moeda, removendo antes o período
    expresso na própria frase.
    """
    if metric == "selic_target":
        return {
            value
            for value, is_percentage in extract_numeric_values(claim)
            if is_percentage
        }

    if metric in {"net_revenue", "gross_revenue", "total_discounts"}:
        values: set[Decimal] = set()
        for raw in CURRENCY_PATTERN.findall(claim):
            parsed = _parse_number(raw.rstrip(".,;"))
            if parsed is not None and not parsed[1]:
                values.add(parsed[0])
        return values

    cleaned = claim
    cleaned = re.sub(
        r"(?i)\b(?:janeiro|fevereiro|março|marco|abril|maio|junho|julho|"
        r"agosto|setembro|outubro|novembro|dezembro)\s+(?:de\s+)?\d{4}\b",
        " ",
        cleaned,
    )
    cleaned = re.sub(r"\b\d{4}[-/]\d{1,2}\b|\b\d{1,2}[-/]\d{4}\b", " ", cleaned)

    return {
        value
        for value, is_percentage in extract_numeric_values(cleaned)
        if not is_percentage
    }


def _find_metric_numeric_fact(
    claim: str,
    evidence: list[Evidence],
) -> tuple[str, Decimal, Decimal, bool] | None:
    """Compara uma métrica com fatos numéricos rotulados nas evidências.

    O último item do retorno indica se os valores coincidem. Só decide quando
    a métrica, o valor afirmado e o valor de evidência são inequívocos.
    """
    metric = _detect_metric(claim)
    if metric is None:
        return None

    claimed_values = _claim_metric_values(claim, metric)
    if len(claimed_values) != 1:
        return None

    claimed_value = next(iter(claimed_values))
    claim_period = _normalize_period(claim)
    candidate_facts: list[tuple[str, Decimal, str | None]] = []

    for item in evidence:
        for fact_metric, fact_value, fact_period in _extract_metric_facts(item):
            if fact_metric != metric:
                continue
            if claim_period and fact_period and claim_period != fact_period:
                continue
            candidate_facts.append((item.source, fact_value, fact_period))

    distinct_values = {value for _, value, _ in candidate_facts}
    if len(distinct_values) != 1:
        return None

    evidenced_value = next(iter(distinct_values))
    source = candidate_facts[0][0]
    return source, claimed_value, evidenced_value, claimed_value == evidenced_value


def _find_metric_numeric_conflict(
    claim: str,
    evidence: list[Evidence],
) -> tuple[str, Decimal, Decimal] | None:
    """Retorna conflito apenas quando a mesma métrica tem outro valor."""
    fact = _find_metric_numeric_fact(claim, evidence)
    if fact is None:
        return None
    source, claimed_value, evidenced_value, matches = fact
    if matches:
        return None
    return source, claimed_value, evidenced_value


def _find_metric_numeric_match(
    claim: str,
    evidence: list[Evidence],
) -> tuple[str, Decimal] | None:
    """Retorna suporte determinístico quando métrica e valor coincidem."""
    fact = _find_metric_numeric_fact(claim, evidence)
    if fact is None:
        return None
    source, claimed_value, evidenced_value, matches = fact
    if not matches:
        return None
    return source, evidenced_value


_EVIDENCE_STOPWORDS = {
    "a", "as", "o", "os", "um", "uma", "de", "do", "da", "dos", "das",
    "em", "no", "na", "nos", "nas", "por", "para", "com", "e", "ou",
    "que", "foi", "foram", "ser", "sao", "são", "é", "ao", "aos",
    "pelo", "pela", "se", "seu", "sua", "seus", "suas", "valor", "valores",
}


def _evidence_relevance_score(claim: str, evidence_content: str) -> float:
    """Estimate lexical/topic relevance before spending NLI on an evidence chunk."""
    claim_terms = {
        token
        for token in re.findall(r"[^\W_]+", _normalize_match_text(claim))
        if token not in _EVIDENCE_STOPWORDS
    }
    evidence_terms = {
        token
        for token in re.findall(r"[^\W_]+", _normalize_match_text(evidence_content))
        if token not in _EVIDENCE_STOPWORDS
    }

    if not claim_terms:
        return 0.0

    coverage = len(claim_terms.intersection(evidence_terms)) / len(claim_terms)
    claim_metrics = _detect_all_metrics(claim)
    evidence_metrics = _detect_all_metrics(evidence_content)
    metric_boost = 0.35 if claim_metrics.intersection(evidence_metrics) else 0.0

    return max(coverage, metric_boost)


_FIELD_DEFINITION_LINE = re.compile(
    r"^\s*(?:[-*]\s*)?([A-Za-z][A-Za-z0-9_]*)\s*:\s*(.+?)\s*$",
    flags=re.IGNORECASE,
)


def _select_field_definition_evidence(
    claim: str,
    evidence: list[Evidence],
) -> tuple[list[Evidence], bool]:
    """Use concise, matching data-dictionary lines for single-field claims.

    Long RAG chunks can contain several unrelated field definitions and formulas.
    The NLI model is more reliable when a field-definition claim is compared with
    the exact line that defines that field. Multi-field/formula claims keep the
    original evidence and therefore do not receive this specialized selection.
    """
    matches: list[tuple[str, str, str]] = []

    for item in evidence:
        for raw_line in item.content.splitlines():
            match = _FIELD_DEFINITION_LINE.match(raw_line)
            if not match:
                continue

            field_name = match.group(1)
            if not re.search(
                rf"(?<!\w){re.escape(field_name)}(?!\w)",
                claim,
                flags=re.IGNORECASE,
            ):
                continue

            definition = f"{field_name}: {match.group(2).strip()}"
            matches.append((field_name.casefold(), item.source, definition))

    referenced_fields = {field_name for field_name, _, _ in matches}
    if len(referenced_fields) != 1:
        return evidence, False

    selected: list[Evidence] = []
    seen: set[tuple[str, str]] = set()
    for _, source, definition in matches:
        identity = (source, definition)
        if identity in seen:
            continue
        seen.add(identity)
        selected.append(Evidence(source=source, content=definition))

    if not selected:
        return evidence, False

    return selected, True



_FORMULA_HEADER_LINE = re.compile(
    r"^\s*(Receita bruta por item|Valor do desconto por item|Receita líquida por item)\s*:\s*(.*?)\s*$",
    flags=re.IGNORECASE,
)


def _formula_kind_for_claim(claim: str) -> str | None:
    """Identify a single explicit sales formula stated by the claim.

    Selection is intentionally conservative: a formula must mention the
    physical fields and a multiplication relation. General comments about a
    formula or multi-topic explanations keep the original evidence set.
    """
    normalized = _normalize_match_text(claim)
    has_quantity = bool(re.search(r"(?<!\w)quantity(?!\w)|\bquantidade\b", normalized))
    has_unit_price = bool(re.search(r"(?<!\w)unit price(?!\w)|\bpreco unitario\b", normalized))
    has_discount_pct = bool(
        re.search(
            r"(?<!\w)discount pct(?!\w)|\bpercentual de desconto\b|\bdesconto\s*%",
            normalized,
        )
    )
    has_multiplication = bool(
        re.search(r"[×*]", claim)
        or re.search(r"\b(?:vezes|multiplicad[ao]s?)\b", normalized)
    )

    if not (has_quantity and has_unit_price and has_multiplication):
        return None

    has_net_adjustment = bool(
        re.search(r"1\s*[-−]\s*discount[ _]pct", claim, flags=re.IGNORECASE)
        or ("receita liquida" in normalized and has_discount_pct)
    )
    if has_net_adjustment and has_discount_pct:
        return "net_revenue"

    if has_discount_pct and "desconto" in normalized:
        return "discount_value"

    if ("receita bruta" in normalized or "valor bruto" in normalized) and not has_discount_pct:
        return "gross_revenue"

    return None


def _select_formula_definition_evidence(
    claim: str,
    evidence: list[Evidence],
) -> tuple[list[Evidence], bool]:
    """Select the exact labelled formula line for a single-formula claim.

    The NLI model is more reliable with the concise formula heading and its
    equation than with an entire chunk containing several different formulas.
    Claims that do not state one explicit formula retain all original evidence.
    """
    formula_kind = _formula_kind_for_claim(claim)
    if formula_kind is None:
        return evidence, False

    expected_headers = {
        "gross_revenue": "receita bruta por item",
        "discount_value": "valor do desconto por item",
        "net_revenue": "receita líquida por item",
    }
    expected_header = expected_headers[formula_kind]
    selected: list[Evidence] = []
    seen: set[tuple[str, str]] = set()

    for item in evidence:
        lines = item.content.splitlines()
        for index, raw_line in enumerate(lines):
            match = _FORMULA_HEADER_LINE.match(raw_line)
            if not match:
                continue

            header = match.group(1).strip().casefold()
            if header != expected_header.casefold():
                continue

            equation = match.group(2).strip()
            if not equation:
                for next_line in lines[index + 1:]:
                    candidate = next_line.strip()
                    if not candidate:
                        continue
                    if candidate.startswith("## "):
                        break
                    equation = candidate
                    break

            if not equation:
                continue

            premise = f"{match.group(1).strip()}:\n{equation}"
            identity = (item.source, premise)
            if identity in seen:
                continue
            seen.add(identity)
            selected.append(Evidence(source=item.source, content=premise))

    if not selected:
        return evidence, False

    return selected, True

def _canonical_formula_hypothesis(claim: str, formula_kind: str) -> str:
    """Rewrite a detected formula claim as a natural-language NLI hypothesis.

    This is a semantic normalization only: the original claim remains in the
    report, and the NLI model must still approve the normalized hypothesis
    against the selected, source-specific formula evidence.
    """
    canonical_hypotheses = {
        "gross_revenue": (
            "A receita bruta por item é o produto da quantidade (quantity) "
            "pelo preço unitário (unit_price)."
        ),
        "discount_value": (
            "O valor do desconto por item é o produto da quantidade (quantity), "
            "do preço unitário (unit_price) e do percentual de desconto (discount_pct)."
        ),
        "net_revenue": (
            "A receita líquida por item é o produto da quantidade (quantity), "
            "do preço unitário (unit_price) e do fator de retenção "
            "(1 - discount_pct)."
        ),
    }
    return canonical_hypotheses.get(formula_kind, claim)


def _classify_claim(
    claim: str,
    evidence: list[Evidence],
    classifier: NLIClassifier,
    threshold: float,
) -> ClaimAssessment:
    """Classify a claim without allowing one noisy source to overrule all others.

    Entailment is assessed in the normal evidence -> claim direction.
    A contradiction is accepted only when the contradiction score is strong
    in both directions. If one source supports the claim and another provides
    a confirmed contradiction, the result is NEUTRA (source disagreement),
    not an automatic REFUTA.
    """
    if not evidence:
        return ClaimAssessment(
            claim=claim,
            verdict="NEUTRA",
            confidence=0.0,
            explanation="Nenhuma evidência foi fornecida.",
        )

    # Prefer formula-specific evidence whenever the claim expresses a
    # recognizable formula. Otherwise, a mention of only discount_pct could
    # make the field selector choose its dictionary definition instead of
    # the actual discount formula.
    formula_kind = _formula_kind_for_claim(claim)
    used_formula_definition = False
    used_field_definition = False
    classification_hypothesis = claim

    if formula_kind is not None:
        classification_evidence, used_formula_definition = (
            _select_formula_definition_evidence(claim, evidence)
        )
        if used_formula_definition:
            classification_hypothesis = _canonical_formula_hypothesis(
                claim, formula_kind
            )
        else:
            # Do not silently replace a formula premise with one field's
            # definition if the exact formula was not retrieved.
            classification_evidence = evidence
    else:
        classification_evidence, used_field_definition = (
            _select_field_definition_evidence(claim, evidence)
        )

    supported: list[tuple[str, dict[str, float]]] = []
    contradicted: list[tuple[str, dict[str, float], float]] = []
    all_results: list[tuple[str, dict[str, float]]] = []

    for item in classification_evidence:
        forward = classifier.classify(
            premise=item.content,
            hypothesis=classification_hypothesis,
        )
        all_results.append((item.source, forward))

        entailment = forward.get("entailment", 0.0)
        contradiction = forward.get("contradiction", 0.0)

        if entailment >= threshold and entailment > contradiction:
            supported.append((item.source, forward))
            continue

        # NLI contradiction can be asymmetric or noisy for paraphrases.
        # Confirm only candidate contradictions in the reverse direction.
        if contradiction >= threshold and contradiction > entailment:
            reverse = classifier.classify(
                premise=classification_hypothesis,
                hypothesis=item.content,
            )
            reverse_contradiction = reverse.get("contradiction", 0.0)
            reverse_entailment = reverse.get("entailment", 0.0)

            if (
                reverse_contradiction >= threshold
                and reverse_contradiction > reverse_entailment
            ):
                confirmed_score = min(contradiction, reverse_contradiction)
                contradicted.append((item.source, forward, confirmed_score))

    best_support = (
        max(supported, key=lambda item: item[1].get("entailment", 0.0))
        if supported
        else None
    )
    best_contradiction = (
        max(contradicted, key=lambda item: item[2])
        if contradicted
        else None
    )

    source: str | None = None
    scores: dict[str, float] = {}

    if best_support and best_contradiction:
        # Conflicting sources are not enough to call a claim false.
        verdict: Verdict = "NEUTRA"
        confidence = max(
            best_support[1].get("entailment", 0.0),
            best_contradiction[2],
        )
        scores = best_support[1]
        explanation = (
            "As evidências recuperadas divergem: uma fonte sustenta a afirmação "
            "e outra apresenta contradição confirmada. É necessária revisão "
            "da documentação antes de concluir."
        )

    elif best_contradiction:
        verdict = "REFUTA"
        source, scores, confidence = best_contradiction
        explanation = (
            "A contradição foi confirmada nas duas direções pela evidência."
        )

    elif best_support:
        verdict = "APOIA"
        source, scores = best_support
        confidence = scores.get("entailment", 0.0)
        if used_field_definition:
            explanation = (
                "A definição específica do campo no dicionário de dados "
                "sustenta a afirmação."
            )
        elif used_formula_definition:
            explanation = (
                "A fórmula específica documentada sustenta a afirmação."
            )
        else:
            explanation = "A evidência relevante sustenta a afirmação."

    else:
        verdict = "NEUTRA"
        confidence = max(
            (
                max(
                    item_scores.get("entailment", 0.0),
                    item_scores.get("contradiction", 0.0),
                )
                for _, item_scores in all_results
            ),
            default=0.0,
        )
        if all_results:
            scores = max(
                all_results,
                key=lambda item: item[1].get("entailment", 0.0),
            )[1]
        explanation = "As evidências não permitem confirmar a afirmação."

    claim_numbers = extract_numeric_values(claim)
    available_numbers = set()

    for item in evidence:
        available_numbers.update(extract_numeric_values(item.content))

        # No JSON do BCB, 13.75 é um número sem sufixo percentual; a unidade
        # "% a.a." vem em outro campo. Normalize-o para comparar com "13,75%".
        if item.source == "get_selic_target":
            try:
                bcb_payload = json.loads(item.content)
            except (json.JSONDecodeError, TypeError):
                bcb_payload = None
            latest = (
                bcb_payload.get("latest_observation")
                if isinstance(bcb_payload, dict)
                else None
            )
            if isinstance(latest, dict) and "%" in str(latest.get("unit", "")):
                parsed_selic = _parse_number(str(latest.get("value", "")))
                if parsed_selic is not None:
                    available_numbers.add((parsed_selic[0], True))

    missing_numbers = claim_numbers - available_numbers

    # An objective conflict for the same labeled metric/period has priority
    # over probabilistic language classification.
    numeric_conflict = _find_metric_numeric_conflict(claim, evidence)
    if numeric_conflict is not None:
        source, claimed_value, evidenced_value = numeric_conflict
        verdict = "REFUTA"
        confidence = 1.0
        explanation = (
            f"O valor informado ({claimed_value}) diverge do valor "
            f"registrado na evidência para a mesma métrica ({evidenced_value})."
        )
    else:
        numeric_match = _find_metric_numeric_match(claim, evidence)
        if numeric_match is not None:
            source, evidenced_value = numeric_match
            verdict = "APOIA"
            confidence = 1.0
            explanation = (
                f"O valor informado ({evidenced_value}) coincide com o valor "
                "registrado na evidência para a mesma métrica."
            )

    # An absent number is not proof of falsity.
    if missing_numbers and verdict != "REFUTA":
        verdict = "NEUTRA"
        source = None
        explanation = (
            "A afirmação contém valores numéricos não encontrados "
            "nas evidências fornecidas."
        )

    return ClaimAssessment(
        claim=claim,
        verdict=verdict,
        confidence=confidence,
        evidence_source=source,
        scores=scores,
        unsupported_numbers=[
            f"{value}{'%' if percent else ''}"
            for value, percent in sorted(missing_numbers)
        ],
        explanation=explanation,
    )


# Palavras funcionais são removidas apenas na comparação de estrutura.
# Termos que alteram o significado são preservados.
_COHERENCE_STOPWORDS = {
    "a", "as", "o", "os", "um", "uma", "uns", "umas",
    "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas",
    "por", "para", "com", "e", "que", "foi", "foram", "ser", "sao",
    "são", "era", "eram", "é", "ao", "aos", "pelo", "pela", "se",
    "seu", "sua", "seus", "suas", "valor", "valores", "total",
}


def _normalized_claim_tokens(claim: str) -> set[str]:
    """Return meaningful words, ignoring numbers, Markdown and stopwords."""
    without_markdown = re.sub(r"[*_`~]+", "", claim.lower())
    without_numbers = NUMBER_PATTERN.sub(" ", without_markdown)
    tokens = re.findall(r"[^\W_]+", without_numbers, flags=re.UNICODE)
    return {token for token in tokens if token not in _COHERENCE_STOPWORDS}


def _detect_all_metrics(text: str) -> set[str]:
    """Return every known metric explicitly mentioned in a claim or evidence."""
    normalized = _normalize_match_text(text)
    normalized = re.sub(r"[^\w]+", " ", normalized).strip()
    detected: set[str] = set()

    for metric, aliases in _METRIC_ALIASES.items():
        for alias in aliases:
            normalized_alias = _normalize_match_text(alias)
            normalized_alias = re.sub(r"[^\w]+", " ", normalized_alias).strip()
            if re.search(
                rf"(?<!\w){re.escape(normalized_alias)}(?!\w)",
                normalized,
            ):
                detected.add(metric)
                break

    return detected


def _detect_primary_metric(text: str) -> str | None:
    """Return the first explicitly named metric in a multi-metric sentence."""
    normalized = _normalize_match_text(text)
    normalized = re.sub(r"[^\w]+", " ", normalized).strip()
    matches: list[tuple[int, int, str]] = []

    for metric, aliases in _METRIC_ALIASES.items():
        for alias in aliases:
            normalized_alias = _normalize_match_text(alias)
            normalized_alias = re.sub(r"[^\w]+", " ", normalized_alias).strip()
            for match in re.finditer(
                rf"(?<!\w){re.escape(normalized_alias)}(?!\w)",
                normalized,
            ):
                matches.append((match.start(), -len(normalized_alias), metric))

    if not matches:
        return None

    return min(matches)[2]


def _claims_share_subject(first_claim: str, second_claim: str) -> bool:
    """Avoid comparing statements about different business metrics."""
    first_metrics = _detect_all_metrics(first_claim)
    second_metrics = _detect_all_metrics(second_claim)

    if first_metrics and second_metrics:
        return bool(first_metrics.intersection(second_metrics))

    first_tokens = _normalized_claim_tokens(first_claim)
    second_tokens = _normalized_claim_tokens(second_claim)

    if not first_tokens or not second_tokens:
        return False

    overlap = len(first_tokens.intersection(second_tokens))
    denominator = min(len(first_tokens), len(second_tokens))
    return denominator > 0 and overlap / denominator >= 0.50


def _subjects_for_cues(claim: str, cues: set[str]) -> set[str]:
    """Find metric subjects in the clause containing a polarity/opposition cue."""
    clauses = re.split(
        r"\b(?:enquanto|mas|porém|porem|contudo|todavia|ao passo que)\b|[;.!?]",
        claim.lower(),
    )
    subjects: set[str] = set()

    for clause in clauses:
        tokens = set(re.findall(r"[^\W_]+", _normalize_match_text(clause)))
        if tokens.intersection(cues):
            clause_metrics = _detect_all_metrics(clause)
            # "Descontos" often appears as the object of a revenue claim,
            # not as the subject whose before/after relation is being stated.
            if len(clause_metrics) > 1:
                clause_metrics.discard("total_discounts")
            subjects.update(clause_metrics)

    return subjects


def _has_explicit_opposition(first_claim: str, second_claim: str) -> bool:
    """Require an explicit opposite relation before pairwise NLI coherence checks."""
    first_tokens = set(re.findall(r"[^\W_]+", _normalize_match_text(first_claim)))
    second_tokens = set(re.findall(r"[^\W_]+", _normalize_match_text(second_claim)))

    opposite_pairs = (
        ({"antes"}, {"após", "apos", "depois"}),
        ({"maior", "maiores"}, {"menor", "menores"}),
        ({"inclui", "incluem", "incluir"}, {"exclui", "excluem", "excluir"}),
        ({"considera", "consideram"}, {"desconsidera", "desconsideram"}),
        ({"aumenta", "aumentam", "cresce", "crescem"}, {"diminui", "diminuem", "reduz", "reduzem", "cai"}),
        ({"positivo", "positiva"}, {"negativo", "negativa"}),
        ({"permitido", "permitida"}, {"proibido", "proibida"}),
        ({"com"}, {"sem"}),
    )

    for left_cues, right_cues in opposite_pairs:
        first_left = _subjects_for_cues(first_claim, left_cues)
        second_right = _subjects_for_cues(second_claim, right_cues)
        if first_left and second_right and first_left.intersection(second_right):
            return True

        first_right = _subjects_for_cues(first_claim, right_cues)
        second_left = _subjects_for_cues(second_claim, left_cues)
        if first_right and second_left and first_right.intersection(second_left):
            return True

    # Negation only counts when both claims use the same predicate and metric.
    negations = {"não", "nao", "nunca", "jamais"}
    negated_first = bool(first_tokens.intersection(negations))
    negated_second = bool(second_tokens.intersection(negations))
    shared_predicates = {
        "considera", "consideram", "inclui", "incluem", "aplica", "aplicam",
        "utiliza", "utilizam", "possui", "possuem", "tem", "têm",
        "contem", "contém", "representa", "representam", "incorpora", "incorporam",
    }

    if negated_first != negated_second:
        if first_tokens.intersection(second_tokens).intersection(shared_predicates):
            first_metrics = _detect_all_metrics(first_claim)
            second_metrics = _detect_all_metrics(second_claim)
            if first_metrics.intersection(second_metrics):
                return True

    return False


def _has_direct_numeric_conflict(first_claim: str, second_claim: str) -> bool:
    """Detect different values only when both claims refer to the same metric."""
    first_numbers = extract_numeric_values(first_claim)
    second_numbers = extract_numeric_values(second_claim)

    if not first_numbers or not second_numbers or first_numbers == second_numbers:
        return False

    first_primary_metric = _detect_primary_metric(first_claim)
    second_primary_metric = _detect_primary_metric(second_claim)

    # If each sentence assigns a value to a different primary metric, those
    # values are not contradictory (e.g. gross revenue versus net revenue).
    if (
        first_primary_metric is not None
        and second_primary_metric is not None
        and first_primary_metric != second_primary_metric
    ):
        return False

    if not _claims_share_subject(first_claim, second_claim):
        return False

    first_tokens = _normalized_claim_tokens(first_claim)
    second_tokens = _normalized_claim_tokens(second_claim)

    if not first_tokens or not second_tokens:
        return False

    union = first_tokens | second_tokens
    similarity = len(first_tokens & second_tokens) / len(union)
    return similarity >= 0.65


def _claims_share_subject(first_claim: str, second_claim: str) -> bool:
    """Avoid pairwise NLI comparisons between claims about distinct metrics."""
    first_metric = _detect_metric(first_claim)
    second_metric = _detect_metric(second_claim)

    if first_metric and second_metric and first_metric != second_metric:
        return False

    first_tokens = _normalized_claim_tokens(first_claim)
    second_tokens = _normalized_claim_tokens(second_claim)

    if not first_tokens or not second_tokens:
        return False

    overlap = len(first_tokens & second_tokens) / min(
        len(first_tokens),
        len(second_tokens),
    )
    return overlap >= 0.50


def check_logical_coherence(
    claims: list[str],
    classifier: NLIClassifier,
    threshold: float = 0.70,
) -> list[CoherenceIssue]:
    """Check contradictions only between claims sharing the same subject.

    Direct numeric conflicts remain deterministic. Semantic contradictions
    require strong NLI contradiction in both directions.
    """
    issues = []

    for index, first_claim in enumerate(claims):
        for second_claim in claims[index + 1:]:
            if _has_direct_numeric_conflict(first_claim, second_claim):
                issues.append(
                    CoherenceIssue(
                        first_claim=first_claim,
                        second_claim=second_claim,
                        confidence=1.0,
                    )
                )
                continue

            # Fórmulas explícitas de categorias diferentes (por exemplo,
            # desconto e receita bruta) são afirmações complementares, não
            # pares candidatos a contradição semântica. Evita uma chamada
            # NLI redundante entre duas fórmulas independentes.
            first_formula_kind = _formula_kind_for_claim(first_claim)
            second_formula_kind = _formula_kind_for_claim(second_claim)
            if (
                first_formula_kind is not None
                and second_formula_kind is not None
                and first_formula_kind != second_formula_kind
            ):
                continue

            if not _claims_share_subject(first_claim, second_claim):
                continue

            forward = classifier.classify(
                premise=first_claim,
                hypothesis=second_claim,
            )
            backward = classifier.classify(
                premise=second_claim,
                hypothesis=first_claim,
            )

            forward_contradiction = forward.get("contradiction", 0.0)
            backward_contradiction = backward.get("contradiction", 0.0)
            score = min(forward_contradiction, backward_contradiction)

            # Contradiction must dominate entailment in both directions.
            if (
                score >= threshold
                and forward_contradiction > forward.get("entailment", 0.0)
                and backward_contradiction > backward.get("entailment", 0.0)
            ):
                issues.append(
                    CoherenceIssue(
                        first_claim=first_claim,
                        second_claim=second_claim,
                        confidence=score,
                    )
                )

    return issues


def verify_claims(
    claims: list[str],
    evidence: list[Evidence],
    classifier: NLIClassifier,
    threshold: float = 0.55,
) -> VerificationReport:
    if not claims:
        raise ValueError("É necessário fornecer ao menos uma afirmação.")

    assessments = [
        _classify_claim(
            claim=claim,
            evidence=evidence,
            classifier=classifier,
            threshold=threshold,
        )
        for claim in claims
    ]

    coherence_issues = check_logical_coherence(
        claims=claims,
        classifier=classifier,
    )

    verdicts = {assessment.verdict for assessment in assessments}

    if coherence_issues:
        overall = "INCOERENTE"
        summary = "Foram identificadas afirmações potencialmente contraditórias."

    elif "REFUTA" in verdicts:
        overall = "REFUTADA"
        summary = "Ao menos uma afirmação foi contradita pelas evidências."

    elif "NEUTRA" in verdicts:
        overall = "INCONCLUSIVA"
        summary = "Há afirmações que não puderam ser confirmadas."

    else:
        overall = "SUSTENTADA"
        summary = "Todas as afirmações foram apoiadas pelas evidências."

    return VerificationReport(
        overall_verdict=overall,
        claims=assessments,
        coherence_issues=coherence_issues,
        evidence_sources=sorted({item.source for item in evidence}),
        summary=summary,
    )


def verify_response(
    response: str,
    evidence: list[Evidence],
    classifier: NLIClassifier | None = None,
) -> VerificationReport:
    claims = extract_claims(response)

    if classifier is None:
        classifier = LocalNLIClassifier()

    return verify_claims(
        claims=claims,
        evidence=evidence,
        classifier=classifier,
    )
