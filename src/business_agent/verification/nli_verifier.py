import re
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
    """
    Carrega tokenizer e modelo uma única vez por processo.
    """
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name
    )

    model.eval()

    return tokenizer, model


class LocalNLIClassifier:
    """
    Classificador NLI executado localmente com Transformers e PyTorch.

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


def _parse_number(raw: str) -> tuple[Decimal, bool] | None:
    """
    Converte números brasileiros e internacionais para Decimal.
    Retorna também se o valor era percentual.
    """
    value = raw.strip()
    is_percentage = value.endswith("%")

    value = value.replace("R$", "").replace("%", "").replace(" ", "")

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
    """
    Separa a resposta em afirmações para avaliação individual.
    """
    if not response or not response.strip():
        raise ValueError("A resposta não pode estar vazia.")

    parts = re.split(
        r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÂÊÔÃÕÇ0-9])|\n+",
        response.strip(),
    )

    claims = [
        part.strip(" \t-*")
        for part in parts
        if part.strip(" \t-*")
    ]

    if not claims:
        raise ValueError("Nenhuma afirmação foi identificada.")

    return claims


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
    contradiction_score = best_contradiction[1].get(
        "contradiction", 0.0
    )

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
        available_numbers.update(
            extract_numeric_values(item.content)
        )

    missing_numbers = claim_numbers - available_numbers

    # A ausência de um valor na evidência não prova, sozinha,
    # que ele é falso. Por isso, classificamos como NEUTRA.
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


def check_logical_coherence(
    claims: list[str],
    classifier: NLIClassifier,
    threshold: float = 0.70,
) -> list[CoherenceIssue]:

    issues = []

    for index, first_claim in enumerate(claims):
        for second_claim in claims[index + 1:]:
            forward = classifier.classify(
                premise=first_claim,
                hypothesis=second_claim,
            )

            backward = classifier.classify(
                premise=second_claim,
                hypothesis=first_claim,
            )

            score = max(
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
        evidence_sources=sorted({
            item.source for item in evidence
        }),
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
