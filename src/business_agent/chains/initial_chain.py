
"""Chain inicial do Business Intelligence Agent.

Demonstra:
- PromptTemplate e ChatPromptTemplate
- Saída estruturada com PydanticOutputParser
- Composição declarativa com LCEL
- Composição imperativa com @chain
- Interface Runnable: invoke, batch e stream
- Modo simulado para testes sem chamadas externas
"""

from __future__ import annotations

import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import chain
from langchain_openai import ChatOpenAI


class RequestAnalysis(BaseModel):
    """Representa a análise estruturada de uma solicitação."""

    intent: Literal[
        "sales_analysis",
        "metric_definition",
        "external_indicator",
        "multi_source_analysis",
        "other",
    ] = Field(description="Intenção principal do usuário.")

    metric: str | None = Field(
        default=None,
        description="Indicador solicitado, se identificado.",
    )

    period: str | None = Field(
        default=None,
        description="Período solicitado, se identificado.",
    )

    requires_tools: bool = Field(
        description=(
            "Indica se a solicitação exige consultar "
            "uma fonte ou ferramenta."
        )
    )

    missing_information: list[str] = Field(
        default_factory=list,
        description=(
            "Informações necessárias que ainda não foram fornecidas."
        ),
    )


def build_chains():
    """Cria as chains do MVP.

    LLM_MODE=mock:
        Utiliza respostas fixas para testar o fluxo localmente.

    LLM_MODE=openai:
        Utiliza o modelo configurado e requer OPENAI_API_KEY.
    """

    load_dotenv()

    mode = os.getenv("LLM_MODE", "mock").strip().lower()

    # 1. Seleciona o modelo conforme o modo de execução.
    if mode == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()

        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY não configurada. "
                "Configure a chave no arquivo .env."
            )

        model_name = os.getenv("OPENAI_MODEL", "gpt-5-mini")

        llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
        )

        simple_llm = llm
        structured_llm = llm

    elif mode == "mock":
        # Resposta fixa para testar o parser e as chains sem API.
        mock_json_response = """
        {
            "intent": "sales_analysis",
            "metric": "faturamento",
            "period": "2026-09 vs 2026-08",
            "requires_tools": true,
            "missing_information": []
        }
        """

        structured_llm = FakeListChatModel(
            responses=[mock_json_response]
        )

        simple_llm = FakeListChatModel(
            responses=[
                (
                    "[MODO SIMULADO] A chain executou localmente. "
                    "Nenhum modelo externo foi consultado."
                )
            ]
        )

        llm = structured_llm

    else:
        raise ValueError(
            f"LLM_MODE inválido: {mode!r}. "
            "Utilize 'mock' ou 'openai'."
        )

    # 2. Parser que converte a resposta em um objeto Pydantic.
    output_parser = PydanticOutputParser(
        pydantic_object=RequestAnalysis
    )

    # 3. PromptTemplate: chamada simples de texto.
    simple_prompt = PromptTemplate.from_template(
        """
Você é um assistente de inteligência de negócios.
Responda em português, de forma clara e concisa.

Não invente números nem afirme ter consultado dados que não recebeu.

Pergunta do usuário:
{question}
"""
    )

    simple_chain = simple_prompt | simple_llm | StrOutputParser()

    # 4. ChatPromptTemplate: classificação estruturada.
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
Você analisa solicitações de inteligência de negócios.

Identifique a intenção principal:
- sales_analysis: análise de vendas ou faturamento;
- metric_definition: definição ou regra de cálculo de uma métrica;
- external_indicator: indicador obtido em uma fonte externa;
- multi_source_analysis: solicitação que combina diferentes fontes;
- other: demais casos.

Extraia a métrica e o período quando estiverem explícitos.
Não invente informações ausentes.
Use missing_information para registrar o que falta.
Defina requires_tools como true quando for necessário consultar
dados, documentos ou uma fonte externa.

Retorne apenas o conteúdo no formato solicitado abaixo.

{format_instructions}
""",
            ),
            (
                "human",
                "Solicitação: {question}",
            ),
        ]
    ).partial(
        format_instructions=output_parser.get_format_instructions()
    )

    # 5. Composição declarativa usando LCEL.
    json_chain = chat_prompt | structured_llm | output_parser

    # 6. Composição imperativa usando @chain.
    # Reutiliza os mesmos componentes para permitir comparação.
    @chain
    def imperative_json_chain(inputs: dict) -> RequestAnalysis:
        prompt_value = chat_prompt.invoke(inputs)
        response = structured_llm.invoke(prompt_value)
        return output_parser.invoke(response)

    # 7. Devolve as chains para os scripts e notebooks.
    return {
        "model": llm,
        "mode": mode,
        "simple_chain": simple_chain,
        "json_chain": json_chain,
        "imperative_json_chain": imperative_json_chain,
    }
