import json

import pytest

from business_agent.agents.text_to_query import (
    build_query_answer,
    is_dimensional_sales_query,
    parse_query_plan,
)
from business_agent.tools.sales_tool import get_sales_breakdown


def test_detects_category_sales_query_with_accents():
    assert is_dimensional_sales_query("Compare a receita líquida por categoria")


def test_detects_region_sales_query_without_accents():
    assert is_dimensional_sales_query("Qual regiao teve mais vendas?")


def test_detects_top_products_query():
    assert is_dimensional_sales_query("Quais produtos foram mais vendidos?")


def test_does_not_route_conceptual_question_to_text_to_query():
    assert not is_dimensional_sales_query("Explique o que é receita líquida por categoria")


def test_does_not_route_unrelated_question():
    assert not is_dimensional_sales_query("Qual é a meta Selic mais recente?")


def test_parses_valid_query_plan():
    plan = parse_query_plan(
        '{"metric":"net_revenue","dimension":"category","sort":"desc","limit":3}'
    )
    assert plan == {
        "metric": "net_revenue",
        "dimension": "category",
        "sort": "desc",
        "limit": 3,
    }


def test_parses_json_surrounded_by_model_text():
    plan = parse_query_plan(
        'Plano selecionado:\n{"metric":"units_sold","dimension":"product","sort":"desc","limit":5}'
    )
    assert plan["metric"] == "units_sold"


def test_rejects_unknown_metric():
    with pytest.raises(ValueError, match="Métrica não permitida"):
        parse_query_plan(
            '{"metric":"salary","dimension":"category","sort":"desc","limit":5}'
        )


def test_rejects_unknown_dimension():
    with pytest.raises(ValueError, match="Dimensão não permitida"):
        parse_query_plan(
            '{"metric":"net_revenue","dimension":"customer_email","sort":"desc","limit":5}'
        )


def test_rejects_extra_sql_field():
    with pytest.raises(ValueError, match="exatamente os campos"):
        parse_query_plan(
            '{"metric":"net_revenue","dimension":"category","sort":"desc","limit":5,"sql":"DROP TABLE sales"}'
        )


def test_rejects_limit_above_maximum():
    with pytest.raises(ValueError, match="entre 1 e 10"):
        parse_query_plan(
            '{"metric":"net_revenue","dimension":"category","sort":"desc","limit":100}'
        )


def test_rejects_non_json_response():
    with pytest.raises(ValueError, match="objeto JSON"):
        parse_query_plan("Não sei montar o plano.")


def test_category_breakdown_uses_parameterized_read_only_query():
    rows = get_sales_breakdown(
        metric="net_revenue",
        dimension="category",
        month="2026-09",
        sort="desc",
        limit=5,
    )
    assert rows[0]["dimension"] == "Eletrônicos"
    assert rows[0]["value"] == 18335.0
    assert len(rows) == 3


def test_region_breakdown_returns_regions():
    rows = get_sales_breakdown(
        metric="units_sold",
        dimension="region",
        month="2026-09",
        sort="desc",
        limit=10,
    )
    assert rows
    assert all(row["dimension"] in {"Sul", "Sudeste", "Nordeste"} for row in rows)


def test_rejects_invalid_breakdown_parameters():
    with pytest.raises(ValueError, match="Métrica"):
        get_sales_breakdown("arbitrary_sql", "category", "2026-09", "desc", 5)


def test_builds_grounded_answer_from_rows():
    answer = build_query_answer(
        {"metric": "net_revenue", "dimension": "category", "sort": "desc", "limit": 2},
        [{"dimension": "Eletrônicos", "value": 18335.0}],
        "2026-09",
    )
    assert "Eletrônicos" in answer
    assert "R$ 18.335,00" in answer
    assert "setembro" not in answer  # período informado é preservado sem inferir mês por extenso
