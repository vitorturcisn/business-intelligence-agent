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
    """Separa a resposta em afirmações para avaliação individual."""
    if not response or not response.strip():
        raise ValueError("A resposta não pode estar vazia.")

    parts = re.split(
        r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÂÊÔÃÕÇ0-9])|\n+",
        response.strip(),
    )

    claims = [
        part.strip(" \t-*\r")
        for part in parts
        if part.strip(" \t-*\r")
    ]

    if not claims:
        raise ValueError("Nenhuma afirmação foi identificada.")

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
        walk_json(payload)
        # Para JSON, as chaves já identificam precisamente cada métrica.
        # Não aplicamos a busca textual ampla ao mesmo conteúdo, pois ela
        # poderia associar por engano um valor vizinho a outra chave.
        return list(dict.fromkeys(facts))

    # Também aceita evidências textuais rotuladas, por exemplo:
    # "Receita líquida: R$ 25.250,00".
    normalized_content = _normalize_match_text(item.content)
    for metric, aliases in _METRIC_ALIASES.items():
        for alias in aliases:
            normalized_alias = _normalize_match_text(alias)
            pattern = re.compile(
                rf"{re.escape(normalized_alias)}[^\n]{{0,70}}?"
                rf"(?:r\$\s*|brl\s*)?(-?\d[\d.,]*)",
                flags=re.IGNORECASE,
            )
            match = pattern.search(normalized_content)
            if match:
                parsed = _parse_number(match.group(1).rstrip(".,;"))
                if parsed is not None and not parsed[1]:
                    facts.append((metric, parsed[0], None))
                break

    # Remove fatos repetidos (JSON + trecho textual podem descrever o mesmo valor).
    return list(dict.fromkeys(facts))


def _claim_metric_values(claim: str, metric: str) -> set[Decimal]:
    """Extrai valores da métrica sem confundir o período com a quantidade.

    Métricas monetárias exigem um marcador de moeda. Contagens (pedidos e
    unidades) aceitam números inteiros sem moeda, removendo antes o período
    expresso na própria frase.
    """
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


def _classify_claim(
    claim: str,
    evidence: list[Evidence],
    classifier: NLIClassifier,
    threshold: float,
) -> ClaimAssessment:
    if not evidence:
        return ClaimAssessment(
            claim=claim,
            verdict="NEUTRA",
            confidence=0.0,
            explanation="Nenhuma evidência foi fornecida.",
        )

    results = []
    for item in evidence:
        scores = classifier.classify(
            premise=item.content,
            hypothesis=claim,
        )
        results.append((item.source, scores))

    best_support = max(
        results,
        key=lambda item: item[1].get("entailment", 0.0),
    )
    best_contradiction = max(
        results,
        key=lambda item: item[1].get("contradiction", 0.0),
    )

    support_score = best_support[1].get("entailment", 0.0)
    contradiction_score = best_contradiction[1].get("contradiction", 0.0)

    if (
        contradiction_score >= threshold
        and contradiction_score > support_score
    ):
        verdict: Verdict = "REFUTA"
        confidence = contradiction_score
        source = best_contradiction[0]
        scores = best_contradiction[1]
        explanation = "A evidência apresenta uma relação de contradição."

    elif support_score >= threshold:
        verdict = "APOIA"
        confidence = support_score
        source = best_support[0]
        scores = best_support[1]
        explanation = "A evidência sustenta a afirmação."

    else:
        verdict = "NEUTRA"
        confidence = max(support_score, contradiction_score)
        source = None
        scores = best_support[1]
        explanation = "As evidências não permitem confirmar a afirmação."

    claim_numbers = extract_numeric_values(claim)
    available_numbers = set()

    for item in evidence:
        available_numbers.update(extract_numeric_values(item.content))

    missing_numbers = claim_numbers - available_numbers

    # Quando a afirmação atribui um valor monetário a uma métrica identificável
    # e o SQL fornece outro valor para a mesma métrica/período, há contradição
    # direta. Essa regra evita depender de uma pontuação NLI que pode ficar abaixo
    # do threshold mesmo diante de um conflito numérico objetivo.
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

    # A ausência de um valor na evidência não prova, sozinha,
    # que ele é falso. Sem uma comparação direta para a mesma métrica,
    # classificamos o número não encontrado como NEUTRA.
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
# Termos que alteram o significado (por exemplo, "não", "bruta" e
# "líquida") são preservados.
_COHERENCE_STOPWORDS = {
    "a", "as", "o", "os", "um", "uma", "uns", "umas",
    "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas",
    "por", "para", "com", "e", "que", "foi", "foram", "ser", "sao",
    "são", "era", "eram", "é", "ao", "aos", "pelo", "pela", "se",
    "seu", "sua", "seus", "suas", "valor", "valores",
}


def _normalized_claim_tokens(claim: str) -> set[str]:
    """Retorna os termos da afirmação, ignorando números e palavras funcionais."""
    without_numbers = NUMBER_PATTERN.sub(" ", claim.lower())
    tokens = re.findall(r"[^\W_]+", without_numbers, flags=re.UNICODE)
    return {token for token in tokens if token not in _COHERENCE_STOPWORDS}


def _has_direct_numeric_conflict(first_claim: str, second_claim: str) -> bool:
    """Detecta valores diferentes em afirmações com a mesma estrutura textual.

    Esta regra determinística complementa o NLI. Ela cobre especialmente
    divergências de métricas numéricas e não depende do comportamento do
    classificador, que pode variar entre o modelo real e doubles de teste.
    """
    first_numbers = extract_numeric_values(first_claim)
    second_numbers = extract_numeric_values(second_claim)

    if not first_numbers or not second_numbers or first_numbers == second_numbers:
        return False

    first_tokens = _normalized_claim_tokens(first_claim)
    second_tokens = _normalized_claim_tokens(second_claim)

    if not first_tokens or not second_tokens:
        return False

    union = first_tokens | second_tokens
    similarity = len(first_tokens & second_tokens) / len(union)

    # Exige estrutura muito semelhante para não confundir métricas distintas
    # (por exemplo, receita bruta e receita líquida) como conflito numérico.
    return similarity >= 0.80


def check_logical_coherence(
    claims: list[str],
    classifier: NLIClassifier,
    threshold: float = 0.70,
) -> list[CoherenceIssue]:
    """Identifica afirmações contraditórias usando regras numéricas e NLI.

    Valores diferentes em afirmações de estrutura praticamente idêntica são
    tratados como conflito direto. Nos demais casos, exige-se contradição NLI
    forte nas duas direções para reduzir falsos positivos de paráfrases.
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

            forward = classifier.classify(
                premise=first_claim,
                hypothesis=second_claim,
            )
            backward = classifier.classify(
                premise=second_claim,
                hypothesis=first_claim,
            )

            score = min(
                forward.get("contradiction", 0.0),
                backward.get("contradiction", 0.0),
            )

            if score >= threshold:
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
