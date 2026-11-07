import pytest

from langgraph.checkpoint.memory import InMemorySaver

from business_agent.agents.orchestrator import build_business_agent
from business_agent.agents.mock_model import FakeSalesToolCallingModel
from business_agent.agents.runner import run_business_query
from business_agent.memory.long_term import LongTermMemoryStore


@pytest.fixture
def memory_store(tmp_path):
    return LongTermMemoryStore(
        tmp_path / "agent_memory_test.db"
    )


def create_agent():
    return build_business_agent(
        model=FakeSalesToolCallingModel(responses=[]),
        checkpointer=InMemorySaver(),
    )


def test_runner_executes_agent_and_records_episode(memory_store):
    outcome = run_business_query(
        agent=create_agent(),
        question="Consulte as vendas de setembro de 2026.",
        thread_id="runner-test-001",
        memory_store=memory_store,
        persist_episode=True,
    )

    assert outcome.tool_names == ["sales_monthly_summary"]
    assert outcome.saved_episode_key is not None

    saved = memory_store.get_memory(
        "episodic",
        outcome.saved_episode_key,
    )

    assert saved is not None
    assert "2026-09" in saved["content"]
    assert "25250" in saved["content"]


def test_runner_retrieves_episode_in_new_session(memory_store):
    first = run_business_query(
        agent=create_agent(),
        question="Consulte as vendas de setembro de 2026.",
        thread_id="runner-session-a",
        memory_store=memory_store,
        persist_episode=True,
    )

    reopened_store = LongTermMemoryStore(memory_store.db_path)

    second = run_business_query(
        agent=create_agent(),
        question="Retome a análise de vendas de setembro de 2026.",
        thread_id="runner-session-b",
        memory_store=reopened_store,
        persist_episode=False,
    )

    recovered_keys = {
        memory["memory_key"]
        for memory in second.retrieved_memories
    }

    assert first.saved_episode_key in recovered_keys


def test_runner_does_not_save_episode_without_opt_in(memory_store):
    outcome = run_business_query(
        agent=create_agent(),
        question="Consulte as vendas de setembro de 2026.",
        thread_id="runner-test-no-save",
        memory_store=memory_store,
        persist_episode=False,
    )

    assert outcome.saved_episode_key is None

    memories = memory_store.search_memories(
        "runner-test-no-save",
        kind="episodic",
    )

    assert memories == []


def test_runner_rejects_empty_question(memory_store):
    with pytest.raises(ValueError):
        run_business_query(
            agent=create_agent(),
            question="",
            thread_id="invalid-session",
            memory_store=memory_store,
        )


def test_runner_rejects_empty_thread_id(memory_store):
    with pytest.raises(ValueError):
        run_business_query(
            agent=create_agent(),
            question="Consulte vendas.",
            thread_id="",
            memory_store=memory_store,
        )
