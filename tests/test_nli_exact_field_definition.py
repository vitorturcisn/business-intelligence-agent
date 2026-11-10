from business_agent.verification.nli_verifier import Evidence, verify_claims


class NeutralNLIClassifier:
    def __init__(self):
        self.calls = []

    def classify(self, premise: str, hypothesis: str):
        self.calls.append((premise, hypothesis))
        return {
            "entailment": 0.5439,
            "neutral": 0.1838,
            "contradiction": 0.2723,
        }


def test_literal_dictionary_field_definition_is_supported_deterministically():
    evidence = [
        Evidence(
            source="business_knowledge_search:02_dicionario_dados.md",
            content="- product: nome do produto vendido.",
        )
    ]
    classifier = NeutralNLIClassifier()

    report = verify_claims(
        claims=["product: nome do produto vendido."],
        evidence=evidence,
        classifier=classifier,
    )

    assessment = report.claims[0]
    assert assessment.verdict == "APOIA"
    assert assessment.confidence == 1.0
    assert assessment.evidence_source == "business_knowledge_search:02_dicionario_dados.md"
    assert assessment.scores == {"exact_match": 1.0}
    assert "literalmente" in assessment.explanation
    assert report.overall_verdict == "SUSTENTADA"
    assert classifier.calls == []


def test_exact_definition_match_ignores_case_and_accents_only():
    evidence = [
        Evidence(
            source="business_knowledge_search:02_dicionario_dados.md",
            content="- PRODUCT: Nome do produto vendido.",
        )
    ]
    classifier = NeutralNLIClassifier()

    report = verify_claims(
        claims=["product: nome do produto vendido."],
        evidence=evidence,
        classifier=classifier,
    )

    assert report.claims[0].verdict == "APOIA"
    assert report.claims[0].scores == {"exact_match": 1.0}
    assert classifier.calls == []


def test_additional_unverified_detail_still_requires_nli_support():
    evidence = [
        Evidence(
            source="business_knowledge_search:02_dicionario_dados.md",
            content="- product: nome do produto vendido.",
        )
    ]
    classifier = NeutralNLIClassifier()
    claim = "product: nome do produto vendido e cÃ³digo SKU Ãºnico."

    report = verify_claims(
        claims=[claim],
        evidence=evidence,
        classifier=classifier,
    )

    assert report.claims[0].verdict == "NEUTRA"
    assert report.overall_verdict == "INCONCLUSIVA"
    assert classifier.calls == [("product: nome do produto vendido.", claim)]