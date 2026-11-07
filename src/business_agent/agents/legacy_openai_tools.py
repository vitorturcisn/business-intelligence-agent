from typing import Any

from langchain_classic.agents import (
    AgentExecutor,
    create_openai_tools_agent,
)
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from business_agent.tools.sales_tool import sales_monthly_summary


class LegacyOpenAIToolsMockModel(FakeMessagesListChatModel):
    """
    Modelo determinístico para demonstrar o ciclo clássico
    create_openai_tools_agent + AgentExecutor.

    Não utiliza API paga ou LLM externo.
    """

    def bind_tools(self, tools: Any, **kwargs: Any):
        # O modelo simulado já conhece a chamada que deve emitir.
        # O AgentExecutor fará a execução da ferramenta real.
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:

        tool_results = [
            message
            for message in messages
            if (
                isinstance(message, ToolMessage)
                or getattr(message, "type", None) == "function"
            )
        ]

        if tool_results:
            response = AIMessage(
                content=(
                    "Análise de vendas executada com sucesso "
                    "pela ferramenta sales_monthly_summary.\n\n"
                    "Resultado SQL:\n"
                    f"{tool_results[-1].content}"
                )
            )
        else:
            response = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "sales_monthly_summary",
                        "args": {"month": "2026-09"},
                        "id": "legacy-sales-monthly-001",
                    }
                ],
            )

        return ChatResult(
            generations=[
                ChatGeneration(message=response)
            ]
        )


def build_legacy_openai_tools_executor(
    model=None,
    tools=None,
    verbose=False,
):
    """
    Constrói o agente clássico do LangChain.

    A implementação principal do projeto continua utilizando
    create_agent (LangChain moderno).
    """

    selected_tools = (
        tools if tools is not None else [sales_monthly_summary]
    )

    if not any(
        item.name == "sales_monthly_summary"
        for item in selected_tools
    ):
        raise ValueError(
            "A ferramenta sales_monthly_summary é obrigatória "
            "para esta demonstração."
        )

    selected_model = (
        model
        if model is not None
        else LegacyOpenAIToolsMockModel(responses=[])
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Você é um agente de Business Intelligence.

                Sua função é responder perguntas sobre indicadores
                comerciais utilizando as ferramentas disponíveis.

                Regras:
                - Utilize ferramentas para obter dados quantitativos.
                - Não invente valores.
                - Baseie suas respostas nos resultados retornados.
                - Responda em português.
                - Seja objetivo e organizado.
                """,
            ),
            MessagesPlaceholder(
                variable_name="chat_history",
                optional=True,
            ),
            ("human", "{input}"),
            MessagesPlaceholder(
                variable_name="agent_scratchpad",
            ),
        ]
    )

    agent = create_openai_tools_agent(
        selected_model,
        selected_tools,
        prompt,
    )

    return AgentExecutor(
        agent=agent,
        tools=selected_tools,
        verbose=verbose,
        return_intermediate_steps=True,
        max_iterations=4,
        handle_parsing_errors=True,
    )
