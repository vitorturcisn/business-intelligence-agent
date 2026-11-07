from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)

from business_agent.agents.verified_runner import (
    extract_agent_evidence,
    verify_agent_result,
)


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
            HumanMessage(content="Quantos pedidos tivemos?"),
            ToolMessage(
                content="O mês de setembro teve 6 pedidos.",
                tool_call_id="call_456",
                name="sales_monthly_summary",
            ),
            AIMessage(
                content="Em setembro foram registrados 6 pedidos."
            ),
        ]
    }

    report = verify_agent_result(
        result,
        classifier=FakeNLIClassifier(),
    )

    assert report.evidence_sources == ["sales_monthly_summary"]
    assert report.claims[0].verdict == "APOIA"


def test_old_tool_messages_are_not_used_as_current_evidence():
    messages = [
        HumanMessage(content="Pergunta anterior."),
        ToolMessage(
            content="O mês anterior teve 6 pedidos.",
            tool_call_id="old_call",
            name="old_sales_tool",
        ),
        AIMessage(content="Resposta anterior."),
        HumanMessage(content="Nova pergunta."),
        AIMessage(content="Não tenho dados suficientes."),
    ]

    evidence = extract_agent_evidence(messages)

    assert evidence == []


def test_only_current_turn_evidence_is_used():
    messages = [
        HumanMessage(content="Pergunta anterior."),
        ToolMessage(
            content="Resultado antigo.",
            tool_call_id="old_call",
            name="old_tool",
        ),
        AIMessage(content="Resposta anterior."),
        HumanMessage(content="Nova pergunta."),
        ToolMessage(
            content="O mês atual teve 6 pedidos.",
            tool_call_id="new_call",
            name="sales_monthly_summary",
        ),
        AIMessage(
            content="O mês atual teve 6 pedidos."
        ),
    ]

    evidence = extract_agent_evidence(messages)

    assert len(evidence) == 1
    assert evidence[0].source == "sales_monthly_summary"


def test_verifier_rejects_empty_agent_result():
    try:
        verify_agent_result(
            {},
            classifier=FakeNLIClassifier(),
        )
    except ValueError:
        return

    raise AssertionError("Resultado vazio deveria ser rejeitado.")
