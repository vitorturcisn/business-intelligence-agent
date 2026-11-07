
"""Testes das chains sequenciais, paralelas e do roteador."""

import pytest

from business_agent.chains.advanced_chains import (
    build_router,
    parallel_analysis_chain,
    sequential_analysis_chain,
)


def test_sequential_chain_extracts_periods_and_sources():
    result = sequential_analysis_chain.invoke(
        {
            "question": (
                "Compare o faturamento de 2026-09 com 2026-08."
            )
        }
    )

    assert result["intent"] == "sales_analysis"
    assert result["periods"] == ["2026-09", "2026-08"]
    assert result["required_sources"] == ["sql"]
    assert len(result["steps"]) == 1


def test_parallel_chain_analyzes_multiple_sources():
    result = parallel_analysis_chain.invoke(
        {
            "question": (
                "Compare o faturamento de 2026-09 "
                "com a meta Selic do Banco Central."
            )
        }
    )

    assert result["intent"] == "multi_source_analysis"
    assert result["periods"] == ["2026-09"]
    assert result["requested_sources"] == [
        "sql",
        "api_banco_central",
    ]


def test_router_selects_sales_destination():
    result = build_router().invoke(
        {"question": "Qual foi o faturamento de 2026-09?"}
    )

    assert result["destination"] == "sales_analysis"
    assert result["next_inputs"]["question"].endswith("2026-09?")


def test_router_selects_external_indicator_destination():
    result = build_router().invoke(
        {"question": "Qual é a meta Selic recente?"}
    )

    assert result["destination"] == "external_indicator"


def test_router_selects_metric_definition_destination():
    result = build_router().invoke(
        {"question": "Como calcular a receita líquida?"}
    )

    assert result["destination"] == "metric_definition"


def test_empty_question_is_rejected():
    with pytest.raises(ValueError, match="não vazia"):
        sequential_analysis_chain.invoke({"question": "  "})
