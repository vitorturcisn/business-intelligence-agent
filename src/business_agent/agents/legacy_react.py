
"""Demonstração do executor ReAct clássico com ferramentas reais.

O LLM é simulado para validar o ciclo de ações sem consumir créditos.
As ferramentas SQL e Banco Central são executadas de verdade.
"""

from __future__ import annotations

import json

from langchain_classic.agents import AgentType, initialize_agent
from langchain_core.language_models.fake_llm import FakeListLLM
from langchain_core.tools import tool

from business_agent.tools.sales_tool import sales_monthly_summary
from business_agent.tools.bcb_api_tool import fetch_selic_observations


@tool("get_selic_recent")
def get_selic_recent(period: str) -> str:
    """Consulta a API real do Banco Central para obter observações recentes da Selic.

    O argumento period deve ser exatamente 'last_30_days'.
    Use essa ferramenta quando a pergunta exigir um indicador econômico atual.
    """

    if period.strip() != "last_30_days":
        raise ValueError(
            "O único período permitido nesta demonstração é last_30_days."
        )

    result = fetch_selic_observations()

    return json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
    )


def build_react_demo():
    """Constrói um executor ReAct com respostas de modelo predeterminadas."""

    fake_llm = FakeListLLM(
        responses=[
            (
                "Thought: Preciso consultar o resumo de vendas de setembro.\n"
                "Action: sales_monthly_summary\n"
                "Action Input: 2026-09"
            ),
            (
                "Thought: Agora preciso consultar a API externa da Selic.\n"
                "Action: get_selic_recent\n"
                "Action Input: last_30_days"
            ),
            (
                "Thought: As duas ferramentas foram executadas.\n"
                "Final Answer: [DEMONSTRAÇÃO COM LLM SIMULADO] "
                "As consultas de vendas e Selic foram executadas. "
                "Consulte as observações das ferramentas para conferir "
                "os dados retornados."
            ),
        ]
    )

    tools = [
        sales_monthly_summary,
        get_selic_recent,
    ]

    agent_executor = initialize_agent(
        tools=tools,
        llm=fake_llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
        return_intermediate_steps=True,
    )

    return agent_executor
