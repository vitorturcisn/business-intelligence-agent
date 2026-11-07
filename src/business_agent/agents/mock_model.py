from typing import Any

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult


class FakeSalesToolCallingModel(FakeMessagesListChatModel):
    """
    Simula o ciclo de tool calling para validar o agente sem API paga.

    Primeiro turno: chama a ferramenta SQL.
    Após a execução da ferramenta: produz uma resposta.
    Turnos seguintes: reutiliza o resultado anterior da mesma sessão.
    """

    def bind_tools(self, tools: Any, **kwargs: Any):
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:

        last_user_index = max(
            (
                index
                for index, message in enumerate(messages)
                if isinstance(message, HumanMessage)
            ),
            default=-1,
        )

        messages_after_last_user = messages[last_user_index + 1:]

        recent_tool_messages = [
            message
            for message in messages_after_last_user
            if isinstance(message, ToolMessage)
        ]

        previous_tool_messages = [
            message
            for message in messages[:last_user_index]
            if isinstance(message, ToolMessage)
        ]

        if recent_tool_messages:
            latest_tool = recent_tool_messages[-1]

            response = AIMessage(
                content=(
                    "Consulta executada pela ferramenta SQL.\n\n"
                    f"Resultado retornado:\n{latest_tool.content}"
                )
            )

        elif previous_tool_messages:
            latest_tool = previous_tool_messages[-1]

            response = AIMessage(
                content=(
                    "Com base no resultado da consulta anterior, "
                    "podemos continuar a análise sem repetir a "
                    "consulta SQL.\n\n"
                    f"Resultado anterior:\n{latest_tool.content}"
                )
            )

        else:
            number_of_users = sum(
                isinstance(message, HumanMessage)
                for message in messages
            )

            response = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "sales_monthly_summary",
                        "args": {"month": "2026-09"},
                        "id": f"sales_call_{number_of_users}",
                    }
                ],
            )

        return ChatResult(
            generations=[ChatGeneration(message=response)]
        )
