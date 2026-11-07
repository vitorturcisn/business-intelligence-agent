import json
from typing import Any

from langchain_core.messages import ToolMessage

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
    Extrai as respostas das ferramentas executadas pelo agente.
    Cada ferramenta é tratada como uma fonte independente.
    """
    evidence = []

    for message in messages:
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
    """
    Verifica a resposta final do agente utilizando as evidências
    retornadas pelas ferramentas executadas durante a interação.
    """
    messages = agent_result.get("messages", [])

    if not messages:
        raise ValueError("O resultado do agente não possui mensagens.")

    answer = _content_to_text(messages[-1].content)

    evidence = extract_agent_evidence(messages)

    return verify_response(
        response=answer,
        evidence=evidence,
        classifier=classifier or LocalNLIClassifier(),
    )
