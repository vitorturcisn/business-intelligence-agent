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
    assert "LIBERADA PELO VERIFICADOR" in rendered
    assert "12 pedidos" in rendered
    assert rendered != result["output"]



def test_rag_tool_observation_is_split_per_document():
    action = SimpleNamespace(tool="business_knowledge_search")
    observation = (
        "Resultado 1\n"
        "Fonte: 01_glossario_kpis.md\n"
        "Posição no documento: 0\n"
        "Conteúdo:\n"
        "## Receita bruta\nValor total das vendas antes de descontos.\n"
        "\n---\n\n"
        "Resultado 2\n"
        "Fonte: 02_dicionario_dados.md\n"
        "Posição no documento: 455\n"
        "Conteúdo:\n"
        "Receita bruta por item: quantity × unit_price."
    )

    evidence = extract_executor_evidence(
        {
            "output": "A receita bruta é calculada antes dos descontos.",
            "intermediate_steps": [(action, observation)],
        }
    )

    assert len(evidence) == 2
    assert evidence[0].source == "business_knowledge_search:01_glossario_kpis.md"
    assert evidence[1].source == "business_knowledge_search:02_dicionario_dados.md"
    assert "Resultado 2" not in evidence[0].content
    assert "quantity × unit_price" in evidence[1].content
