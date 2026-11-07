from pydantic import PrivateAttr
import re
import unicodedata
import uuid
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


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())

    return "".join(
        char for char in text
        if not unicodedata.combining(char)
    )


class RoutingEvaluationMockModel(FakeMessagesListChatModel):
    _available_tools: set[str] = PrivateAttr(default_factory=set)
    """
    Modelo determinístico para avaliação de tool calling.

    Regras:
    - Selic: API simulada do Banco Central.
    - Categorias: ranking SQL.
    - Conceitos e fórmulas: RAG.
    - Métricas de vendas: SQL.
    - Perguntas que pedem valor e definição: SQL + RAG.

    Não representa autonomia de um LLM.
    """

    def bind_tools(self, tools: Any, **kwargs: Any):
        self._available_tools = {item.name for item in tools}
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

        recent_messages = messages[last_user_index + 1:]

        recent_tool_messages = [
            message
            for message in recent_messages
            if isinstance(message, ToolMessage)
        ]

        if recent_tool_messages:
            response_content = "\n\n".join(
                f"Fonte: {message.name}\n"
                f"Resultado: {message.content}"
                for message in recent_tool_messages
            )

            response = AIMessage(
                content=(
                    "Resultado da análise com base nas ferramentas "
                    "executadas:\n\n"
                    f"{response_content}"
                )
            )

        else:
            question = str(messages[last_user_index].content)
            normalized = normalize(question)

            month_match = re.search(
                r"\b20\d{2}-(0[1-9]|1[0-2])\b",
                question,
            )

            month_names = {
                "janeiro": "01",
                "fevereiro": "02",
                "marco": "03",
                "abril": "04",
                "maio": "05",
                "junho": "06",
                "julho": "07",
                "agosto": "08",
                "setembro": "09",
                "outubro": "10",
                "novembro": "11",
                "dezembro": "12",
            }

            portuguese_month_match = re.search(
                r"\b(janeiro|fevereiro|marco|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)\s+(?:de\s+)?(20\d{2})\b",
                normalized,
            )

            if month_match:
                month = month_match.group(0)
            elif portuguese_month_match:
                month_match = portuguese_month_match
                month = (
                    f"{portuguese_month_match.group(2)}-"
                    f"{month_names[portuguese_month_match.group(1)]}"
                )
            else:
                month = "2026-09"

            calls = []

            def add_call(name: str, args: dict):
                if name not in self._available_tools:
                    raise RuntimeError(
                        f"Ferramenta de avaliação não encontrada: {name}"
                    )

                calls.append({
                    "name": name,
                    "args": args,
                    "id": str(uuid.uuid4()),
                })

            if "selic" in normalized or "juros basicos" in normalized:
                add_call(
                    "get_selic_target",
                    {
                        "start_date": "2026-09-01",
                        "end_date": "2026-09-30",
                    },
                )

            elif "categorias" in normalized or "ranking" in normalized:
                add_call(
                    "sales_category_ranking",
                    {"month": month},
                )

            else:
                wants_sales = bool(month_match) or any(
                    term in normalized
                    for term in [
                        "vendas",
                        "faturamento",
                        "pedidos",
                        "quanto vendeu",
                        "valor vendido",
                    ]
                )

                wants_rag = any(
                    term in normalized
                    for term in [
                        "definicao",
                        "defina",
                        "formula",
                        "calculado",
                        "calculada",
                        "conceito",
                    ]
                )

                if wants_sales:
                    add_call(
                        "sales_monthly_summary",
                        {"month": month},
                    )

                if wants_rag:
                    add_call(
                        "business_knowledge_search",
                        {"query": question},
                    )

                if not calls:
                    add_call(
                        "business_knowledge_search",
                        {"query": question},
                    )

            response = AIMessage(
                content="",
                tool_calls=calls,
            )

        return ChatResult(
            generations=[
                ChatGeneration(message=response)
            ]
        )



