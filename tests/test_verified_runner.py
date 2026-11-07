from langchain_core.messages import ToolMessage

from business_agent.agents.verified_runner import (
    extract_agent_evidence,
    verify_agent_result,
)
from business_agent.verification.nli_verifier import Evidence


class FakeNLIClassifier:
    def classify(self, premise, hypothesis):
        if "6 pedidos" in hypothesis:
            return {
                "entailment": 0.96,
                "neutral": 0.03,
                "contradiction": 0.01,
            }

        return {
            "entailment": 0.10,
            "neutral": 0.85,
            "contradiction": 0.05,
        }


def test_extract_agent_evidence_uses_tool_names():
    messages = [
        ToolMessage(
            content='{"order_count": 6}',
            tool_call_id="call_123",
            name="sales_monthly_summary",
        )
    ]

    evidence = extract_agent_evidence(messages)

    assert len(evidence) == 1
    assert evidence[0].source == "sales_monthly_summary"
    assert "order_count" in evidence[0].content


def test_agent_result_can_be_verified():
    result = {
        "messages": [
            ToolMessage(
                content="O mês de setembro teve 6 pedidos.",
                tool_call_id="call_456",
                name="sales_monthly_summary",
            ),
            type(
                "AssistantMessage",
                (),
                {
                    "content": (
                        "Em setembro foram registrados 6 pedidos."
                    )
                },
            )(),
        ]
    }

    report = verify_agent_result(
        result,
        classifier=FakeNLIClassifier(),
    )

    assert report.evidence_sources == ["sales_monthly_summary"]
    assert report.claims[0].verdict == "APOIA"


def test_verifier_rejects_empty_agent_result():
    try:
        verify_agent_result(
            {},
            classifier=FakeNLIClassifier(),
        )
    except ValueError:
        return

    raise AssertionError("Resultado vazio deveria ser rejeitado.")
