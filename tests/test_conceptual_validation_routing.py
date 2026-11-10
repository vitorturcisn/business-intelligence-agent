from business_agent.agents.text_to_query import is_dimensional_sales_query
from business_agent.verification.nli_verifier import Evidence, verify_claims
from scripts.run_local_openai_tools_agent import is_conceptual_rag_query


ADVERSARIAL_QUESTION = (
    "A receita bruta por item \u00e9 quantity \u00d7 unit_price \u00d7 "
    "(1 - discount_pct), correto?"
)


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


def sales_formula_evidence():
    return [
        Evidence(
            source="business_knowledge_search:02_dicionario_dados.md",
            content=(
                "Receita bruta por item:\n"
                "quantity \u00d7 unit_price\n\n"
                "Valor do desconto por item:\n"
                "quantity \u00d7 unit_price \u00d7 discount_pct\n\n"
                "Receita lÃ­quida por item:\n"
                "quantity \u00d7 unit_price \u00d7 (1 - discount_pct)"
            ),
        )
    ]


def test_formula_correctness_question_is_not_routed_to_text_to_query():
    assert not is_dimensional_sales_query(ADVERSARIAL_QUESTION)


def test_formula_correctness_question_is_routed_to_conceptual_rag():
    assert is_conceptual_rag_query(ADVERSARIAL_QUESTION)


def test_explicit_equation_without_correctness_word_is_not_text_to_query():
    question = "Receita bruta por item = quantity \u00d7 unit_price \u00d7 (1 - discount_pct)."
    assert not is_dimensional_sales_query(question)
    assert is_conceptual_rag_query(question)


def test_real_dimensional_queries_still_use_text_to_query():
    assert is_dimensional_sales_query("Compare a receita lÃ­quida por categoria")
    assert is_dimensional_sales_query("Qual regi\u00e3o teve mais vendas?")
    assert is_dimensional_sales_query("Quais produtos foram mais vendidos?")


def test_gross_formula_written_with_natural_copula_is_rejected():
    claim = (
        "A receita bruta por item \u00e9 quantity \u00d7 unit_price \u00d7 "
        "(1 - discount_pct)."
    )
    classifier = NeutralNLIClassifier()
    report = verify_claims(
        claims=[claim],
        evidence=sales_formula_evidence(),
        classifier=classifier,
    )

    assessment = report.claims[0]
    assert assessment.verdict == "REFUTA"
    assert assessment.scores == {"formula_label_mismatch": 1.0}
    assert report.overall_verdict == "REFUTADA"
    assert classifier.calls == []


def test_correct_gross_formula_is_not_flagged_by_deterministic_guard():
    claim = "A receita bruta por item \u00e9 quantity \u00d7 unit_price."
    classifier = NeutralNLIClassifier()
    report = verify_claims(
        claims=[claim],
        evidence=sales_formula_evidence(),
        classifier=classifier,
    )

    assert report.claims[0].scores != {"formula_label_mismatch": 1.0}
    assert classifier.calls