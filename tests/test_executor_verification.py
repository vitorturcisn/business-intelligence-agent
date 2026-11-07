from types import SimpleNamespace

from business_agent.agents.executor_verification import (
    extract_executor_evidence,
    verify_executor_result,
    render_verified_answer,
)


class FakeNLIClassifier:
    def classify(self, premise: str, hypothesis: str):
        if "12 pedidos" in hypothesis:
            return {
                "entailment": 0.01,
                "neutral": 0.01,
                "contradiction": 0.98,
            }

        if "6 pedidos" in hypothesis:
            return {
                "entailment": 0.98,
                "neutral": 0.01,
                "contradiction": 0.01,
            }

        return {
            "entailment": 0.05,
            "neutral": 0.90,
            "contradiction": 0.05,
        }


def build_result(answer: str):
    action = SimpleNamespace(tool="sales_monthly_summary")

    return {
        "output": answer,
        "intermediate_steps": [
            (
                action,
                "Resumo de setembro: 6 pedidos.",
            )
        ],
    }


def test_executor_evidence_is_extracted():
    result = build_result("Foram registrados 6 pedidos.")

    evidence = extract_executor_evidence(result)

    assert len(evidence) == 1
    assert evidence[0].source == "sales_monthly_summary"
    assert "6 pedidos" in evidence[0].content


def test_supported_executor_answer_is_approved():
    result = build_result("Em setembro foram registrados 6 pedidos.")

    report = verify_executor_result(
        result,
        classifier=FakeNLIClassifier(),
    )

    assert report.overall_verdict == "SUSTENTADA"
    assert render_verified_answer(result, report) == result["output"]


def test_contradicted_executor_answer_is_blocked():
    result = build_result("Em setembro foram registrados 12 pedidos.")

    report = verify_executor_result(
        result,
        classifier=FakeNLIClassifier(),
    )

    rendered = render_verified_answer(result, report)

    assert report.overall_verdict == "REFUTADA"
    assert "NAO LIBERADA" in rendered
    assert "12 pedidos" in rendered
    assert rendered != result["output"]
