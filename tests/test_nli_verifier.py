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
