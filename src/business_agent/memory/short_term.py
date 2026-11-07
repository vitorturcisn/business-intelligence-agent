"""Memória de curto prazo e histórico conversacional por sessão.

Inclui:
- ConversationBufferMemory para a demonstração didática exigida pela disciplina.
- RunnableWithMessageHistory para histórico associado a session_id.
- Integração de histórico com um AgentExecutor ou outro Runnable compatível.
- Janela deslizante que mantém as últimas interações completas no contexto.

O histórico criado aqui fica em memória RAM e dura enquanto o processo estiver
ativo. Persistência e recuperação de longo prazo são responsabilidades de outros
módulos do projeto.
"""

from __future__ import annotations

from typing import Any

from langchain_classic.memory import ConversationBufferMemory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory


def build_conversation_buffer_memory() -> ConversationBufferMemory:
    """Cria a memória de buffer clássica usada na demonstração da disciplina."""
    return ConversationBufferMemory(
        memory_key="chat_history",
        input_key="input",
        output_key="output",
        return_messages=True,
    )


def build_session_history_demo():
    """Cria uma demonstração determinística de histórico separado por sessão.

    O armazenamento fica em memória RAM e existe apenas durante a execução
    do processo. A função é mantida para preservar os testes e a demonstração
    didática existentes.
    """
    histories: dict[str, InMemoryChatMessageHistory] = {}

    def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
        if not isinstance(session_id, str) or not session_id.strip():
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
            output = "Não há uma pergunta anterior registrada nesta sessão."
        else:
            output = f"Mensagem recebida nesta sessão: {question}"

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


def build_agent_with_session_history(
    agent_executor: Any,
    max_messages: int = 8,
):
    """Envolve um AgentExecutor com histórico conversacional por sessão.

    Args:
        agent_executor: AgentExecutor/Runnable cuja entrada usa ``input``,
            cujo prompt aceita ``chat_history`` e cuja saída contém ``output``.
        max_messages: Quantidade máxima de mensagens anteriores mantidas no
            contexto. Deve ser um inteiro positivo par, pois cada interação
            completa normalmente contém uma mensagem humana e uma do agente.
            O padrão 8 corresponde a aproximadamente quatro turnos anteriores.

    Returns:
        Tupla ``(agent_with_history, histories)``. O primeiro item recebe
        ``config={"configurable": {"session_id": "..."}}`` em cada invoke.
        O segundo permite inspeção e testes do histórico durante a execução.

    Observação:
        O histórico é volátil: não sobrevive ao encerramento do processo.
        A função limita o contexto enviado em cada novo turno; a integração
        com memória persistente de longo prazo é feita separadamente.
    """
    if (
        isinstance(max_messages, bool)
        or not isinstance(max_messages, int)
        or max_messages < 2
        or max_messages % 2 != 0
    ):
        raise ValueError(
            "max_messages deve ser um inteiro par maior ou igual a 2."
        )

    histories: dict[str, InMemoryChatMessageHistory] = {}

    def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id não pode ser vazio.")

        if session_id not in histories:
            histories[session_id] = InMemoryChatMessageHistory()

        history = histories[session_id]
        messages = list(history.messages)

        # Remove interações antigas antes de montar o contexto do próximo turno.
        if len(messages) > max_messages:
            retained = messages[-max_messages:]

            # Mantém o histórico começando por uma mensagem do usuário,
            # evitando iniciar a janela com uma resposta órfã do agente.
            while retained and not isinstance(retained[0], HumanMessage):
                retained = retained[1:]

            history.clear()
            if retained:
                history.add_messages(retained)

        return history

    agent_with_history = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="output",
    )

    return agent_with_history, histories
