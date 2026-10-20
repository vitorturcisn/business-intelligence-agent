
"""Testes das ferramentas de consulta de vendas."""

import pytest

from business_agent.tools.sales_tool import (
    get_category_ranking,
    get_monthly_summary,
    sales_category_ranking,
    sales_monthly_summary,
)


def test_august_monthly_summary():
    result = get_monthly_summary("2026-08")

    assert result["month"] == "2026-08"
    assert result["order_count"] == 6
    assert result["units_sold"] == 29
    assert result["gross_revenue"] == 19000.00
    assert result["total_discounts"] == 980.00
    assert result["net_revenue"] == 18020.00


def test_september_monthly_summary():
    result = get_monthly_summary("2026-09")

    assert result["month"] == "2026-09"
    assert result["order_count"] == 6
    assert result["units_sold"] == 40
    assert result["gross_revenue"] == 26650.00
    assert result["total_discounts"] == 1400.00
    assert result["net_revenue"] == 25250.00


def test_category_ranking_is_descending():
    result = get_category_ranking("2026-09")

    assert len(result) == 3
    assert result[0]["category"] == "Eletrônicos"
    assert result[0]["net_revenue"] == 18335.00

    revenues = [item["net_revenue"] for item in result]
    assert revenues == sorted(revenues, reverse=True)


def test_invalid_month_is_rejected():
    with pytest.raises(ValueError):
        get_monthly_summary("setembro de 2026")


def test_missing_month_is_rejected():
    with pytest.raises(ValueError, match="Não existem vendas"):
        get_monthly_summary("2026-10")


def test_langchain_tools_are_registered():
    assert sales_monthly_summary.name == "sales_monthly_summary"
    assert sales_category_ranking.name == "sales_category_ranking"

    assert "YYYY-MM" in sales_monthly_summary.description
    assert "YYYY-MM" in sales_category_ranking.description
