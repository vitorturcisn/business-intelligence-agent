from business_agent.verification.nli_verifier import (
    Evidence,
    extract_claims,
    extract_numeric_values,
    verify_claims,
    verify_response,
)


class FakeNLIClassifier:
    def classify(self, premise: str, hypothesis: str):
        normalized = hypothesis.lower()

        if "12 pedidos" in normalized or "30.000" in normalized:
            return {
                "entailment": 0.01,
                "neutral": 0.02,
                "contradiction": 0.97,
            }

        if (
            "6 pedidos" in normalized
            or "25.250" in normalized
            or "receita líquida é receita bruta menos descontos" in normalized
        ):
            return {
                "entailment": 0.96,
                "neutral": 0.03,
                "contradiction": 0.01,
            }

        if (
            "25.250" in premise
            and "30.000" in hypothesis
        ):
            return {
                "entailment": 0.01,
                "neutral": 0.02,
                "contradiction": 0.97,
            }

        return {
            "entailment": 0.10,
            "neutral": 0.85,
            "contradiction": 0.05,
        }


def test_extract_claims_splits_response():
    claims = extract_claims(
        "Foram registrados 6 pedidos. "
        "A receita líquida foi R$ 25.250,00."
    )

    assert len(claims) == 2


def test_extract_numeric_values_handles_brazilian_currency():
    values = extract_numeric_values(
        "Receita líquida de R$ 25.250,00 e desconto de R$ 1.400,00."
    )

    assert (25000 + 250, False) in values
    assert (1400, False) in values


def test_supported_claim_is_approved():
    report = verify_claims(
        claims=["Em setembro foram registrados 6 pedidos."],
        evidence=[
            Evidence(
                source="sales_sql",
                content="Em setembro foram registrados 6 pedidos.",
            )
        ],
        classifier=FakeNLIClassifier(),
    )

    assert report.claims[0].verdict == "APOIA"
    assert report.overall_verdict == "SUSTENTADA"


def test_contradicted_claim_is_rejected():
    report = verify_claims(
        claims=["Em setembro foram registrados 12 pedidos."],
        evidence=[
            Evidence(
                source="sales_sql",
                content="Em setembro foram registrados 6 pedidos.",
            )
        ],
        classifier=FakeNLIClassifier(),
    )

    assert report.claims[0].verdict == "REFUTA"
    assert report.overall_verdict == "REFUTADA"


def test_unsupported_numeric_value_is_flagged():
    report = verify_claims(
        claims=["A receita líquida foi R$ 30.000,00."],
        evidence=[
            Evidence(
                source="sales_sql",
                content="A receita líquida foi R$ 25.250,00.",
            )
        ],
        classifier=FakeNLIClassifier(),
    )

    assert report.claims[0].unsupported_numbers
    assert report.claims[0].verdict == "REFUTA"


def test_missing_evidence_produces_neutral_verdict():
    report = verify_claims(
        claims=["A empresa vendeu 100 unidades."],
        evidence=[],
        classifier=FakeNLIClassifier(),
    )

    assert report.claims[0].verdict == "NEUTRA"
    assert report.overall_verdict == "INCONCLUSIVA"


class SourceAwareNLIClassifier:
    """Fake classifier that reproduces a noisy source and a clear supporting source."""

    def classify(self, premise: str, hypothesis: str):
        if premise.startswith("Glossario:"):
            return {
                "entailment": 0.01,
                "neutral": 0.02,
                "contradiction": 0.97,
            }

        if premise.startswith("Regras:"):
            return {
                "entailment": 0.93,
                "neutral": 0.06,
                "contradiction": 0.01,
            }

        # Reverse direction for the noisy glossary source: not contradictory.
        if hypothesis.startswith("Glossario:"):
            return {
                "entailment": 0.02,
                "neutral": 0.93,
                "contradiction": 0.05,
            }

        return {
            "entailment": 0.05,
            "neutral": 0.90,
            "contradiction": 0.05,
        }


class BidirectionalConflictNLIClassifier:
    def classify(self, premise: str, hypothesis: str):
        if premise.startswith("Fonte contraditoria:"):
            return {
                "entailment": 0.01,
                "neutral": 0.02,
                "contradiction": 0.97,
            }
        if hypothesis.startswith("Fonte contraditoria:"):
            return {
                "entailment": 0.01,
                "neutral": 0.02,
                "contradiction": 0.96,
            }
        if premise.startswith("Fonte favoravel:"):
            return {
                "entailment": 0.96,
                "neutral": 0.03,
                "contradiction": 0.01,
            }
        return {
            "entailment": 0.05,
            "neutral": 0.90,
            "contradiction": 0.05,
        }


def test_strong_support_is_not_overruled_by_one_asymmetric_source():
    report = verify_claims(
        claims=[
            "A receita líquida corresponde à receita bruta menos os descontos."
        ],
        evidence=[
            Evidence(
                source="glossario",
                content="Glossario: trecho ambíguo sobre receita bruta e líquida.",
            ),
            Evidence(
                source="regras",
                content="Regras: a receita líquida corresponde à receita bruta menos o desconto.",
            ),
        ],
        classifier=SourceAwareNLIClassifier(),
    )

    assert report.claims[0].verdict == "APOIA"
    assert report.claims[0].evidence_source == "regras"
    assert report.overall_verdict == "SUSTENTADA"


def test_conflicting_sources_produce_inconclusive_verdict():
    report = verify_claims(
        claims=["A receita líquida corresponde à receita bruta menos os descontos."],
        evidence=[
            Evidence(
                source="fonte_contraditoria",
                content="Fonte contraditoria: receita líquida não corresponde à receita bruta menos descontos.",
            ),
            Evidence(
                source="fonte_favoravel",
                content="Fonte favoravel: receita líquida corresponde à receita bruta menos descontos.",
            ),
        ],
        classifier=BidirectionalConflictNLIClassifier(),
    )

    assert report.claims[0].verdict == "NEUTRA"
    assert report.overall_verdict == "INCONCLUSIVA"


def test_coherence_does_not_compare_gross_and_net_revenue_as_same_metric():
    class AlwaysContradictoryClassifier:
        def classify(self, premise: str, hypothesis: str):
            return {
                "entailment": 0.01,
                "neutral": 0.01,
                "contradiction": 0.98,
            }

    report = verify_claims(
        claims=[
            "A receita bruta é o valor antes dos descontos.",
            "A receita líquida é o valor depois dos descontos.",
        ],
        evidence=[
            Evidence(
                source="glossario",
                content=(
                    "Receita bruta é o valor antes dos descontos. "
                    "Receita líquida é o valor depois dos descontos."
                ),
            )
        ],
        classifier=AlwaysContradictoryClassifier(),
    )

    assert report.coherence_issues == []
    assert report.overall_verdict != "INCOERENTE"


def test_extract_claims_removes_markdown_emphasis():
    claims = extract_claims(
        "**Receita bruta** é o valor antes dos descontos.\n"
        "- **Receita líquida** é o valor após descontos."
    )

    assert len(claims) == 2
    assert all("*" not in claim for claim in claims)


def test_empty_response_is_rejected():
    try:
        verify_response(
            response="",
            evidence=[],
            classifier=FakeNLIClassifier(),
        )
    except ValueError:
        return

    raise AssertionError("Resposta vazia deveria gerar ValueError.")


def test_conflicting_claims_are_detected():
    report = verify_claims(
        claims=[
            "A receita líquida foi R$ 25.250,00.",
            "A receita líquida foi R$ 30.000,00.",
        ],
        evidence=[
            Evidence(
                source="sales_sql",
                content="A receita líquida foi R$ 25.250,00.",
            )
        ],
        classifier=FakeNLIClassifier(),
    )

    assert report.coherence_issues
    assert report.overall_verdict == "INCOERENTE"



class AlwaysContradictionClassifier:
    def classify(self, premise: str, hypothesis: str):
        return {
            "entailment": 0.01,
            "neutral": 0.01,
            "contradiction": 0.98,
        }


def test_extract_claims_ignores_section_heading_and_markdown():
    claims = extract_claims(
        "No exemplo de setembro:\n"
        "- **Receita bruta**: R$ 26.650,00.\n"
        "- **Receita líquida**: R$ 25.250,00."
    )

    assert len(claims) == 2
    assert all("**" not in claim for claim in claims)
    assert all("No exemplo" not in claim for claim in claims)


def test_coherence_does_not_flag_compatible_gross_net_explanation():
    report = verify_claims(
        claims=[
            "A receita bruta é o valor total das vendas antes de descontos.",
            "A receita líquida é o valor após os descontos, enquanto a receita bruta não considera os descontos.",
        ],
        evidence=[],
        classifier=AlwaysContradictionClassifier(),
    )

    assert report.coherence_issues == []
    assert report.overall_verdict != "INCOERENTE"


def test_coherence_checks_opposite_relations_for_same_metric():
    report = verify_claims(
        claims=[
            "A receita bruta é calculada antes dos descontos.",
            "A receita bruta é calculada após os descontos.",
        ],
        evidence=[],
        classifier=AlwaysContradictionClassifier(),
    )

    assert len(report.coherence_issues) == 1
    assert report.overall_verdict == "INCOERENTE"


def test_different_metric_values_are_not_a_logical_conflict():
    report = verify_claims(
        claims=[
            "A receita líquida foi R$ 25.250,00.",
            "A receita bruta foi R$ 26.650,00.",
        ],
        evidence=[],
        classifier=AlwaysContradictionClassifier(),
    )

    assert report.coherence_issues == []
    assert report.overall_verdict != "INCOERENTE"


def test_numeric_conflict_detected_when_value_precedes_metric_label():
    report = verify_claims(
        claims=["Em setembro foram registrados 12 pedidos."],
        evidence=[
            Evidence(
                source="sales_sql",
                content="Resumo de setembro: 6 pedidos.",
            )
        ],
        classifier=FakeNLIClassifier(),
    )

    assert report.claims[0].verdict == "REFUTA"
    assert report.overall_verdict == "REFUTADA"



def test_extract_claims_splits_gross_net_comparison_into_atomic_claims():
    claims = extract_claims(
        "O cálculo de receita líquida reflete o valor final após a aplicação de descontos, "
        "enquanto a receita bruta não considera esses descontos."
    )

    assert len(claims) == 2
    assert "receita líquida" in claims[0].lower()
    assert "receita bruta" in claims[1].lower()
    assert claims[0].endswith(".")
    assert claims[1].endswith(".")



def test_gross_revenue_comparison_rewrites_anaphoric_discount_reference():
    claims = extract_claims(
        "O cálculo de receita líquida reflete o valor final após a aplicação de descontos, "
        "enquanto a receita bruta não considera esses descontos."
    )

    assert len(claims) == 2
    assert claims[1] == (
        "A receita bruta é o valor total das vendas antes da aplicação de descontos."
    )
