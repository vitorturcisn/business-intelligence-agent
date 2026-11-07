from business_agent.agents.legacy_openai_tools import (
    build_legacy_openai_tools_executor,
)


def test_legacy_agent_executor_runs_sales_tool():
    executor = build_legacy_openai_tools_executor()

    result = executor.invoke(
        {
            "input": (
                "Apresente as vendas de setembro de 2026."
            )
        }
    )

    assert "output" in result
    assert "intermediate_steps" in result
    assert len(result["intermediate_steps"]) >= 1

    first_action = result["intermediate_steps"][0][0]

    assert first_action.tool == "sales_monthly_summary"
    assert "25250" in result["output"]


def test_legacy_agent_executor_returns_expected_month():
    executor = build_legacy_openai_tools_executor()

    result = executor.invoke(
        {
            "input": (
                "Apresente as vendas de setembro de 2026."
            )
        }
    )

    observations = [
        str(observation)
        for _, observation in result["intermediate_steps"]
    ]

    combined_observations = " ".join(observations)

    assert "2026-09" in combined_observations
