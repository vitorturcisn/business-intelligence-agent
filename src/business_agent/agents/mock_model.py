from typing import Any

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


class FakeSalesToolCallingModel(FakeMessagesListChatModel):
    """
    Modelo simulado para validar o ciclo de tool calling.

    Emite uma chamada para sales_monthly_summary e depois
    produz uma resposta utilizando o resultado da ferramenta.

    Não representa autonomia real de um LLM.
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

        tool_messages = [
            message
            for message in messages_after_last_user
            if isinstance(message, ToolMessage)
        ]

        if tool_messages:
            latest_tool = tool_messages[-1]

            response = AIMessage(
                content=(
                    "Consulta executada pela ferramenta SQL.\n\n"
                    f"Resultado retornado:\n{latest_tool.content}"
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
            generations=[
                ChatGeneration(message=response)
            ]
        )
