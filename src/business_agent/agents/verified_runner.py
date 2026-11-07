import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from business_agent.verification.nli_verifier import (
    Evidence,
    LocalNLIClassifier,
    NLIClassifier,
    VerificationReport,
    verify_response,
)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    return json.dumps(
        content,
        ensure_ascii=False,
        default=str,
    )


def extract_agent_evidence(messages: list[Any]) -> list[Evidence]:
    """
    Extrai somente as evidências geradas depois da última
    mensagem do usuário, evitando utilizar resultados antigos.
    """
    last_user_index = max(
        (
            index
            for index, message in enumerate(messages)
            if isinstance(message, HumanMessage)
        ),
        default=-1,
    )

    current_turn = messages[last_user_index + 1:]

    evidence = []

    for message in current_turn:
        if isinstance(message, ToolMessage):
            evidence.append(
                Evidence(
                    source=message.name or "unknown_tool",
                    content=_content_to_text(message.content),
                )
            )

    return evidence


def verify_agent_result(
    agent_result: dict[str, Any],
    classifier: NLIClassifier | None = None,
) -> VerificationReport:

    messages = agent_result.get("messages", [])

    if not messages:
        raise ValueError("O resultado do agente não possui mensagens.")

    final_messages = [
        message
        for message in messages
        if isinstance(message, AIMessage)
        and not message.tool_calls
        and message.content
    ]

    if not final_messages:
        raise ValueError("Nenhuma resposta final foi encontrada.")

    answer = _content_to_text(final_messages[-1].content)

    evidence = extract_agent_evidence(messages)

    return verify_response(
        response=answer,
        evidence=evidence,
        classifier=classifier or LocalNLIClassifier(),
    )
