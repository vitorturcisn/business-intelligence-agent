
"""Demonstração do executor ReAct clássico.

O modelo é simulado para testar o ciclo de execução sem créditos.
A ferramenta SQL e a chamada à API do Banco Central são reais.
"""

from __future__ import annotations

import json

from langchain_classic.agents import AgentType, initialize_agent
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.tools import tool

from business_agent.tools.sales_tool import sales_monthly_summary
from business_agent.tools.bcb_api_tool import fetch_selic_observations


@tool("get_selic_recent")
def get_selic_recent(period: str) -> str:
    """Consulta a API real do Banco Central para obter observações recentes da Selic.

    O argumento deve ser 'last_30_days'.
    Use para consultar um indicador econômico externo recente.
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
    """Cria o executor ReAct com respostas de teste predeterminadas."""

    fake_llm = FakeListLLM(
        responses=[
            (
                "Thought: Preciso consultar as vendas de setembro.\n"
                "Action: sales_monthly_summary\n"
                "Action Input: 2026-09"
            ),
            (
                "Thought: Agora preciso consultar a API da Selic.\n"
                "Action: get_selic_recent\n"
                "Action Input: last_30_days"
            ),
            (
                "Thought: As duas ferramentas foram executadas.\n"
                "Final Answer: [MODO SIMULADO] As consultas de vendas "
                "e Selic foram executadas. Confira os resultados "
                "registrados na trajetória das ferramentas."
            ),
        ]
    )

    tools = [
        sales_monthly_summary,
        get_selic_recent,
    ]

    return initialize_agent(
        tools=tools,
        llm=fake_llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
        return_intermediate_steps=True,
    )
