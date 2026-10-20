
"""Memória de curto prazo e histórico por sessão.

Inclui:
- ConversationBufferMemory para a demonstração didática.
- RunnableWithMessageHistory para histórico associado a session_id.

A resposta é determinística nesta etapa, sem chamadas a LLM.
"""

from __future__ import annotations

from typing import Any

from langchain_classic.memory import ConversationBufferMemory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory


def build_conversation_buffer_memory() -> ConversationBufferMemory:
    """Cria a memória de buffer exigida pela disciplina."""

    return ConversationBufferMemory(
        memory_key="chat_history",
        input_key="input",
        output_key="output",
        return_messages=True,
    )


def build_session_history_demo():
    """Cria um Runnable com histórico separado por sessão.

    O armazenamento fica em memória RAM e existe apenas durante
    a execução do processo. Persistência em disco será implementada
    separadamente na memória de longo prazo.
    """

    histories: dict[str, InMemoryChatMessageHistory] = {}

    def get_session_history(
        session_id: str,
    ) -> InMemoryChatMessageHistory:
        if not session_id.strip():
            raise ValueError("session_id não pode ser vazio.")

        if session_id not in histories:
            histories[session_id] = InMemoryChatMessageHistory()

        return histories[session_id]

    def respond(inputs: dict[str, Any]) -> dict[str, str]:
        question = str(inputs["input"]).strip()
        history = inputs.get("chat_history", [])

        previous_user_messages = [
            message.content
            for message in history
            if isinstance(message, HumanMessage)
        ]

        if not question:
            raise ValueError("A pergunta não pode estar vazia.")

        normalized = question.casefold()

        recall_phrases = (
            "o que eu disse antes",
            "o que eu falei antes",
            "qual foi minha pergunta anterior",
            "o que eu perguntei antes",
        )

        asks_about_history = any(
            phrase in normalized for phrase in recall_phrases
        )

        if asks_about_history and previous_user_messages:
            output = (
                "Na mesma sessão, você disse anteriormente: "
                f"{previous_user_messages[-1]}"
            )
        elif asks_about_history:
            output = (
                "Não há uma pergunta anterior registrada "
                "nesta sessão."
            )
        else:
            output = (
                "Mensagem recebida nesta sessão: "
                f"{question}"
            )

        return {"output": output}

    runnable = RunnableLambda(respond)

    runnable_with_history = RunnableWithMessageHistory(
        runnable,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="output",
    )

    return runnable_with_history, histories
