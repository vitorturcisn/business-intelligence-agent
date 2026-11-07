import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from business_agent.tools.sales_tool import (
    sales_monthly_summary,
    sales_category_ranking,
)
from business_agent.tools.bcb_api_tool import get_selic_target
from business_agent.tools.rag_tool import business_knowledge_search


SYSTEM_PROMPT = """
# PERSONA
Você é um analista sênior de Business Intelligence especializado em
análise de vendas, indicadores de negócio e indicadores econômicos.

# TAREFA
Responder perguntas de negócio utilizando ferramentas especializadas.
Você deve decidir quais fontes consultar antes de formular a resposta.

# FONTES E FERRAMENTAS

1. sales_monthly_summary:
Use para consultar indicadores consolidados de vendas por mês.

2. sales_category_ranking:
Use para identificar o desempenho das categorias de produtos.

3. business_knowledge_search:
Use para consultar definições de KPIs, regras de negócio,
fórmulas, dicionário de dados e documentação.

4. get_selic_target:
Use para consultar os dados da Selic na API do Banco Central.

# REGRAS DE DECISÃO

- Perguntas quantitativas sobre vendas devem consultar o SQL.
- Perguntas sobre conceitos e fórmulas devem consultar o RAG.
- Perguntas sobre Selic devem consultar a API do Banco Central.
- Perguntas que envolvam mais de uma fonte devem combinar as
  ferramentas necessárias.
- Não invente valores, períodos ou resultados.
- Não utilize o RAG como fonte de valores quantitativos de vendas.
- Diferencie receita bruta de receita líquida.
- Diferencie número de pedidos de unidades vendidas.
- Informe limitações quando uma fonte não puder ser consultada.

# FORMATO
Responda em português brasileiro.
Seja claro e objetivo.
Apresente os valores, períodos e fontes consultadas.
Explique brevemente os cálculos relevantes.
Nunca apresente resultados que não tenham sido retornados pelas fontes.
"""


def get_business_tools():
    return [
        sales_monthly_summary,
        sales_category_ranking,
        business_knowledge_search,
        get_selic_target,
    ]


def build_business_agent(model=None, checkpointer=None):
    """
    Constrói o agente principal com SQL, RAG e API externa.

    O modelo pode ser injetado para testes.
    Sem injeção, utiliza ChatOpenAI quando a chave está configurada.
    """
    if model is None:
        load_dotenv()

        api_key = os.getenv("OPENAI_API_KEY", "").strip()

        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY não configurada. "
                "Configure a chave ou injete um modelo de teste."
            )

        model = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
            api_key=api_key,
        )

    return create_agent(
        model=model,
        tools=get_business_tools(),
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer or InMemorySaver(),
        name="business_intelligence_agent",
    )
