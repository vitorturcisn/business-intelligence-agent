
"""Ferramentas de consulta aos dados de vendas em SQLite.

As consultas são fixas e parametrizadas. O usuário não fornece SQL
arbitrário, e as ferramentas não permitem modificar o banco de dados.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from langchain_core.tools import tool


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DB_PATH = PROJECT_ROOT / "data" / "sample" / "sales.db"


def _validate_month(month: str) -> str:
    """Valida o formato YYYY-MM."""

    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", month):
        raise ValueError(
            "O mês deve estar no formato YYYY-MM, por exemplo 2026-08."
        )

    return month


def _query_database(
    query: str,
    parameters: tuple[Any, ...],
) -> list[dict[str, Any]]:
    """Executa uma consulta controlada e devolve linhas como dicionários."""

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Banco de dados não encontrado: {DB_PATH}. "
            "Execute python scripts/setup_data.py primeiro."
        )

    # Conexão read-only: as ferramentas não podem alterar os dados.
    connection_uri = DB_PATH.as_uri() + "?mode=ro"

    with sqlite3.connect(connection_uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(query, parameters).fetchall()

    return [dict(row) for row in rows]


def get_monthly_summary(month: str) -> dict[str, Any]:
    """Calcula indicadores consolidados de um mês."""

    month = _validate_month(month)

    query = """
        SELECT
            COUNT(*) AS order_count,
            COALESCE(SUM(quantity), 0) AS units_sold,
            COALESCE(SUM(quantity * unit_price), 0) AS gross_revenue,
            COALESCE(
                SUM(quantity * unit_price * discount_pct), 0
            ) AS total_discounts,
            COALESCE(
                SUM(quantity * unit_price * (1 - discount_pct)), 0
            ) AS net_revenue
        FROM sales
        WHERE substr(sale_date, 1, 7) = ?
    """

    rows = _query_database(query, (month,))
    result = rows[0]

    if result["order_count"] == 0:
        raise ValueError(f"Não existem vendas cadastradas para {month}.")

    for key in ("gross_revenue", "total_discounts", "net_revenue"):
        result[key] = round(float(result[key]), 2)

    result["month"] = month
    result["currency"] = "BRL"

    return result


def get_category_ranking(month: str) -> list[dict[str, Any]]:
    """Ordena as categorias por receita líquida decrescente."""

    month = _validate_month(month)

    query = """
        SELECT
            category,
            SUM(quantity * unit_price * (1 - discount_pct))
                AS net_revenue,
            SUM(quantity) AS units_sold,
            COUNT(*) AS order_count
        FROM sales
        WHERE substr(sale_date, 1, 7) = ?
        GROUP BY category
        ORDER BY net_revenue DESC, category ASC
    """

    rows = _query_database(query, (month,))

    if not rows:
        raise ValueError(f"Não existem vendas cadastradas para {month}.")

    results = []

    for row in rows:
        results.append(
            {
                "category": row["category"],
                "net_revenue": round(float(row["net_revenue"]), 2),
                "units_sold": row["units_sold"],
                "order_count": row["order_count"],
                "month": month,
                "currency": "BRL",
            }
        )

    return results


# Expressões e dimensões permitidas para o Text-to-query. O modelo escolhe
# identificadores deste catálogo; nunca fornece SQL executável.
_BREAKDOWN_METRICS = {
    "net_revenue": "SUM(quantity * unit_price * (1 - discount_pct))",
    "gross_revenue": "SUM(quantity * unit_price)",
    "total_discounts": "SUM(quantity * unit_price * discount_pct)",
    "order_count": "COUNT(*)",
    "units_sold": "SUM(quantity)",
}
_BREAKDOWN_DIMENSIONS = {
    "category": "category",
    "region": "region",
    "product": "product",
}


def get_sales_breakdown(
    metric: str,
    dimension: str,
    month: str | None = None,
    sort: str = "desc",
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Executa agregação por dimensão com catálogo fechado e conexão read-only."""
    if metric not in _BREAKDOWN_METRICS:
        raise ValueError("Métrica não permitida para consulta dimensional.")
    if dimension not in _BREAKDOWN_DIMENSIONS:
        raise ValueError("Dimensão não permitida para consulta dimensional.")
    if sort not in {"asc", "desc"}:
        raise ValueError("Ordenação deve ser 'asc' ou 'desc'.")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10:
        raise ValueError("O limite deve ser um inteiro entre 1 e 10.")

    parameters: tuple[Any, ...]
    where_clause = ""
    if month is not None:
        month = _validate_month(month)
        where_clause = "WHERE substr(sale_date, 1, 7) = ?"
        parameters = (month, limit)
    else:
        parameters = (limit,)

    # Expressão métrica, coluna dimensional e direção vêm exclusivamente
    # de allow-lists; mês e limite são enviados como parâmetros SQL.
    query = f"""
        SELECT
            {_BREAKDOWN_DIMENSIONS[dimension]} AS dimension,
            {_BREAKDOWN_METRICS[metric]} AS value
        FROM sales
        {where_clause}
        GROUP BY {_BREAKDOWN_DIMENSIONS[dimension]}
        ORDER BY value {sort.upper()}, dimension ASC
        LIMIT ?
    """
    rows = _query_database(query, parameters)

    if not rows:
        period_label = f" para {month}" if month else ""
        raise ValueError(f"Não existem vendas cadastradas{period_label}.")

    for row in rows:
        if metric in {"net_revenue", "gross_revenue", "total_discounts"}:
            row["value"] = round(float(row["value"]), 2)
        else:
            row["value"] = int(row["value"])
    return rows


@tool("sales_monthly_summary")
def sales_monthly_summary(month: str) -> str:
    """Consulta o resumo de vendas de um mês no formato YYYY-MM.

    Retorna quantidade de pedidos, unidades vendidas, faturamento bruto,
    descontos totais e receita líquida em reais.
    Use esta ferramenta para perguntas sobre o desempenho mensal.
    """

    return json.dumps(
        get_monthly_summary(month),
        ensure_ascii=False,
        indent=2,
    )


@tool("sales_category_ranking")
def sales_category_ranking(month: str) -> str:
    """Classifica as categorias por receita líquida de um mês YYYY-MM.

    Retorna as categorias ordenadas da maior para a menor receita líquida.
    Use esta ferramenta para comparar categorias de produtos.
    """

    return json.dumps(
        get_category_ranking(month),
        ensure_ascii=False,
        indent=2,
    )
