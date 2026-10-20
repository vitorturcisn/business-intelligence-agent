
"""Testes de integração do executor ReAct demonstrativo."""

import json

from business_agent.agents import legacy_react


def test_react_executes_sales_and_selic_tools(monkeypatch):
    """Verifica a trajetória sem depender da rede ou de créditos de LLM."""

    # Simula somente o retorno da API externa.
    # A consulta SQL de vendas continua sendo executada de verdade.
    monkeypatch.setattr(
        legacy_react,
        "fetch_selic_observations",
        lambda: {
            "series_code": 432,
            "observations_count": 1,
            "latest_observation": {
                "date": "2026-10-20",
                "value": 13.75,
                "unit": "% a.a.",
            },
            "source": (
                "https://api.bcb.gov.br/dados/"
                "serie/bcdata.sgs.432/dados"
            ),
        },
    )

    executor = legacy_react.build_react_demo()

    result = executor.invoke(
        {
            "input": (
                "Consulte as vendas de setembro e a meta Selic recente."
            )
        }
    )

    steps = result["intermediate_steps"]
    tool_names = [action.tool for action, _ in steps]

    assert tool_names == [
        "sales_monthly_summary",
        "get_selic_recent",
    ]

    sales_observation = steps[0][1]
    sales_data = json.loads(sales_observation)

    assert sales_data["month"] == "2026-09"
    assert sales_data["net_revenue"] == 25250.00

    selic_observation = json.loads(steps[1][1])

    assert selic_observation["series_code"] == 432
    assert selic_observation["latest_observation"]["value"] == 13.75

    assert "[MODO SIMULADO]" in result["output"]
