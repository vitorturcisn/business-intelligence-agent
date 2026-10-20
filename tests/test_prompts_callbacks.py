
"""Testes do prompt PTCF, few-shot e callbacks."""

import logging

from langchain_core.tools import tool

from business_agent.agents.prompts import (
    FEW_SHOT_EXAMPLES,
    PTCF_SYSTEM_PROMPT,
    build_ptcf_agent_prompt,
)
from business_agent.observability.callbacks import (
    AgentLoggingCallbackHandler,
)


def test_ptcf_prompt_has_required_sections():
    for section in ("Persona", "Task", "Context", "Format"):
        assert section in PTCF_SYSTEM_PROMPT


def test_prompt_contains_few_shot_examples():
    prompt = build_ptcf_agent_prompt()

    assert len(FEW_SHOT_EXAMPLES) >= 3
    assert "input" in prompt.input_variables
    assert "agent_scratchpad" in prompt.input_variables



def test_prompt_formats_with_history_and_scratchpad():
    prompt = build_ptcf_agent_prompt()

    messages = prompt.format_messages(
        input="Qual foi o faturamento de setembro?",
        chat_history=[],
        agent_scratchpad=[],
    )

    assert messages[0].type == "system"

    # A última mensagem humana contém a pergunta atual do usuário.
    assert messages[-1].type == "human"
    assert "Qual foi o faturamento" in messages[-1].content

    # Os exemplos few-shot anteriores continuam presentes.
    assert any(
        message.type == "human"
        and "meta Selic" in message.content
        for message in messages
    )


def test_callback_logs_tool_events_without_logging_output(caplog):
    @tool("test_logging_tool")
    def test_logging_tool(value: str) -> str:
        """Ferramenta simples usada apenas neste teste."""

        return f"resultado-secreto-{value}"

    handler = AgentLoggingCallbackHandler()

    with caplog.at_level(
        logging.INFO,
        logger="business_agent.execution",
    ):
        result = test_logging_tool.invoke(
            {"value": "123"},
            config={"callbacks": [handler]},
        )

    assert result == "resultado-secreto-123"
    assert "tool.start name=test_logging_tool" in caplog.text
    assert "tool.end output_type=str" in caplog.text
    assert "resultado-secreto-123" not in caplog.text
