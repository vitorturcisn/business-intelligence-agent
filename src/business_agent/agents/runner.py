import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from business_agent.memory.long_term import LongTermMemoryStore


@dataclass
class BusinessRunOutcome:
    agent_result: dict[str, Any]
    retrieved_memories: list[dict]
    tool_names: list[str]
    saved_episode_key: str | None


def _current_turn(messages: list[Any]) -> list[Any]:
    last_user_index = max(
        (
            index
            for index, message in enumerate(messages)
            if isinstance(message, HumanMessage)
        ),
        default=-1,
    )

    return messages[last_user_index + 1:]


def _build_memory_context(memories: list[dict]) -> str:
    if not memories:
        return ""

    sections = []

    for memory in memories:
        sections.append(
            f"Tipo: {memory['kind']}\n"
            f"Chave: {memory['memory_key']}\n"
            f"Informação: {memory['content'][:1200]}"
        )

    return (
        "\n\n[MEMÓRIAS PERSISTENTES RECUPERADAS]\n"
        "As informações abaixo são registros históricos. "
        "Não as trate automaticamente como valores atuais. "
        "Para indicadores atuais, consulte as ferramentas de dados.\n\n"
        + "\n\n---\n\n".join(sections)
        + "\n[FIM DAS MEMÓRIAS]\n"
    )


def _message_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    return str(content)


def run_business_query(
    agent,
    question: str,
    thread_id: str,
    memory_store: LongTermMemoryStore | None = None,
    persist_episode: bool = False,
) -> BusinessRunOutcome:
    """
    Executa uma pergunta com recuperação de memória persistente.

    1. Recupera memórias relacionadas usando busca lexical.
    2. Injeta o contexto recuperado na pergunta.
    3. Executa o agente com seu checkpoint de sessão.
    4. Registra um episódio quando solicitado e quando ferramentas
       foram efetivamente executadas.

    persist_episode é opt-in para evitar armazenamento involuntário.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("A pergunta não pode estar vazia.")

    if not isinstance(thread_id, str) or not thread_id.strip():
        raise ValueError("thread_id não pode estar vazio.")

    store = memory_store or LongTermMemoryStore()

    retrieved_memories = store.search_memories(
        query=question,
        limit=3,
    )

    memory_context = _build_memory_context(retrieved_memories)

    augmented_question = (
        f"Pergunta original do usuário:\n{question.strip()}"
        f"{memory_context}"
    )

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(content=augmented_question)
            ]
        },
        config={
            "configurable": {
                "thread_id": thread_id,
            }
        },
    )

    messages = result.get("messages", [])
    current_messages = _current_turn(messages)

    tool_messages = [
        message
        for message in current_messages
        if isinstance(message, ToolMessage)
    ]

    tool_names = [
        message.name or "unknown_tool"
        for message in tool_messages
    ]

    saved_episode_key = None

    if persist_episode and tool_messages:
        final_messages = [
            message
            for message in current_messages
            if isinstance(message, AIMessage)
            and not message.tool_calls
            and message.content
        ]

        final_answer = (
            _message_to_text(final_messages[-1].content)
            if final_messages
            else "Resposta final não identificada."
        )

        tool_evidence = "\n\n".join(
            f"Fonte: {message.name or 'unknown_tool'}\n"
            f"Resultado: {_message_to_text(message.content)[:2000]}"
            for message in tool_messages
        )

        timestamp = datetime.now(timezone.utc).isoformat()

        episode_content = (
            f"Data da execução: {timestamp}\n"
            f"Pergunta: {question.strip()}\n"
            f"Ferramentas executadas: {', '.join(tool_names)}\n\n"
            f"Evidências:\n{tool_evidence}\n\n"
            f"Resposta final:\n{final_answer[:2000]}"
        )

        saved_episode_key = f"run_{uuid.uuid4().hex}"

        store.save_memory(
            kind="episodic",
            memory_key=saved_episode_key,
            content=episode_content,
            tags=[
                "business-agent",
                "execucao",
                *tool_names,
            ],
        )

    return BusinessRunOutcome(
        agent_result=result,
        retrieved_memories=retrieved_memories,
        tool_names=tool_names,
        saved_episode_key=saved_episode_key,
    )
