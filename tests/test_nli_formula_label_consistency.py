import pytest

from business_agent.verification.nli_verifier import Evidence, verify_claims


class NeutralNLIClassifier:
    def __init__(self):
        self.calls = []

    def classify(self, premise: str, hypothesis: str):
        self.calls.append((premise, hypothesis))
        return {
            "entailment": 0.20,
            "neutral": 0.75,
            "contradiction": 0.05,
        }


@pytest.fixture
def sales_formula_evidence():
    return [
        Evidence(
            source="business_knowledge_search:02_dicionario_dados.md",
            content=(
                "Receita bruta por item:\n"
                "quantity × unit_price\n\n"
                "Valor do desconto por item:\n"
                "quantity × unit_price × discount_pct\n\n"
                "Receita líquida por item:\n"
                "quantity × unit_price × (1 - discount_pct)"
            ),
        )
    ]


def test_gross_revenue_formula_cannot_include_discount_deduction(sales_formula_evidence):
    claim = (
        "receita bruta = (quantidade × preço unitário) - "
        "(quantidade × preço unitário × discount_pct)."
    )
    classifier = NeutralNLIClassifier()

    report = verify_claims(
        claims=[claim],
        evidence=sales_formula_evidence,
        classifier=classifier,
    )

    assessment = report.claims[0]
    assert assessment.verdict == "REFUTA"
    assert assessment.confidence == 1.0
    assert assessment.evidence_source == "business_knowledge_search:02_dicionario_dados.md"
    assert assessment.scores == {"formula_label_mismatch": 1.0}
    assert "receita bruta" in assessment.explanation
    assert report.overall_verdict == "REFUTADA"
    assert classifier.calls == []


def test_correct_gross_revenue_formula_is_not_flagged(sales_formula_evidence):
    claim = "Receita bruta por item = quantity × unit_price."
    classifier = NeutralNLIClassifier()

    report = verify_claims(
        claims=[claim],
        evidence=sales_formula_evidence,
        classifier=classifier,
    )

    assert report.claims[0].verdict == "NEUTRA"
    assert report.claims[0].scores != {"formula_label_mismatch": 1.0}
    assert len(classifier.calls) >= 1


def test_ordinary_explanation_of_gross_revenue_and_discount_is_not_flagged(sales_formula_evidence):
    claim = (
        "A receita bruta representa o valor antes dos descontos, "
        "e discount_pct indica o percentual de desconto aplicado."
    )
    classifier = NeutralNLIClassifier()

    report = verify_claims(
        claims=[claim],
        evidence=sales_formula_evidence,
        classifier=classifier,
    )

    assert report.claims[0].verdict == "NEUTRA"
    assert report.claims[0].scores != {"formula_label_mismatch": 1.0}
    assert len(classifier.calls) >= 1


def test_mislabeled_formula_requires_documented_gross_formula():
    claim = (
        "receita bruta = (quantidade × preço unitário) - "
        "(quantidade × preço unitário × discount_pct)."
    )
    evidence_without_gross_formula = [
        Evidence(
            source="glossario",
            content="Receita líquida é o valor das vendas depois dos descontos.",
        )
    ]
    classifier = NeutralNLIClassifier()

    report = verify_claims(
        claims=[claim],
        evidence=evidence_without_gross_formula,
        classifier=classifier,
    )

    assert report.claims[0].verdict == "NEUTRA"
    assert report.claims[0].scores != {"formula_label_mismatch": 1.0}
    assert len(classifier.calls) >= 1
