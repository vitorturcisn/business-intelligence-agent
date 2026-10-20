
"""Testes da memória conversacional."""

import pytest

from business_agent.memory.short_term import (
    build_conversation_buffer_memory,
    build_session_history_demo,
)


def test_conversation_buffer_memory_saves_messages():
    memory = build_conversation_buffer_memory()

    memory.save_context(
        {"input": "Quero analisar as vendas de setembro."},
        {"output": "Vamos considerar setembro de 2026."},
    )

    result = memory.load_memory_variables({"input": "E agosto?"})
    history = result["chat_history"]

    assert len(history) == 2
    assert history[0].content == (
        "Quero analisar as vendas de setembro."
    )
    assert history[1].content == (
        "Vamos considerar setembro de 2026."
    )


def test_runnable_history_recalls_previous_turn():
    runnable, histories = build_session_history_demo()
    config = {"configurable": {"session_id": "session-test-1"}}

    runnable.invoke(
        {"input": "Quero analisar as vendas de setembro."},
        config=config,
    )

    result = runnable.invoke(
        {"input": "O que eu disse antes?"},
        config=config,
    )

    assert "Quero analisar as vendas de setembro." in result["output"]
    assert len(histories["session-test-1"].messages) == 4


def test_sessions_do_not_share_history():
    runnable, histories = build_session_history_demo()

    runnable.invoke(
        {"input": "Minha categoria preferida é eletrônicos."},
        config={"configurable": {"session_id": "session-A"}},
    )

    result = runnable.invoke(
        {"input": "O que eu disse antes?"},
        config={"configurable": {"session_id": "session-B"}},
    )

    assert "Não há uma pergunta anterior" in result["output"]
    assert len(histories["session-A"].messages) == 2
    assert len(histories["session-B"].messages) == 2


def test_empty_question_is_rejected():
    runnable, _ = build_session_history_demo()

    with pytest.raises(ValueError, match="não pode estar vazia"):
        runnable.invoke(
            {"input": "   "},
            config={"configurable": {"session_id": "session-test"}},
        )


def test_empty_session_id_is_rejected():
    runnable, _ = build_session_history_demo()

    with pytest.raises(ValueError, match="session_id"):
        runnable.invoke(
            {"input": "Olá"},
            config={"configurable": {"session_id": ""}},
        )
