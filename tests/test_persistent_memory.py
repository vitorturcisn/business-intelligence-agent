from business_agent.memory.long_term import LongTermMemoryStore
from business_agent.memory.persistent_memory import (
    PersistentMemoryManager,
    extract_explicit_memory_request,
)


def test_extract_explicit_memory_request_with_lembre_se():
    assert extract_explicit_memory_request(
        "Lembre-se que nosso KPI principal é receita líquida."
    ) == "nosso KPI principal é receita líquida."


def test_extract_explicit_memory_request_with_guarde_na_memoria():
    assert extract_explicit_memory_request(
        "Guarde na memória: usamos dbt nos modelos Gold."
    ) == "usamos dbt nos modelos Gold."


def test_normal_question_is_not_treated_as_memory_write():
    assert extract_explicit_memory_request(
        "Qual foi a receita líquida de setembro?"
    ) is None


def test_explicit_memory_is_persisted_and_can_be_reopened(tmp_path):
    db_path = tmp_path / "persistent_memory.db"
    manager = PersistentMemoryManager(LongTermMemoryStore(db_path))

    saved = manager.save_explicit_memory(
        "Nosso KPI principal é receita líquida."
    )

    reopened = PersistentMemoryManager(LongTermMemoryStore(db_path))
    recalled = reopened.store.get_memory(
        kind="semantic",
        memory_key=saved["memory_key"],
    )

    assert recalled is not None
    assert recalled["content"] == "Nosso KPI principal é receita líquida."
    assert "user-provided" in recalled["tags"]


def test_same_explicit_memory_is_upserted_not_duplicated(tmp_path):
    manager = PersistentMemoryManager(
        LongTermMemoryStore(tmp_path / "persistent_memory.db")
    )

    first = manager.save_explicit_memory("O KPI principal é receita líquida.")
    second = manager.save_explicit_memory("O KPI principal é receita líquida.")

    records = manager.store.list_memories(kind="semantic")

    assert first["memory_key"] == second["memory_key"]
    assert len(records) == 1


def test_format_context_marks_memories_as_user_provided_context():
    context = PersistentMemoryManager.format_context(
        [
            {
                "kind": "semantic",
                "memory_key": "kpi_principal",
                "content": "O KPI principal é receita líquida.",
            }
        ]
    )

    assert "MEMÓRIAS PERSISTENTES RECUPERADAS" in context
    assert "KPI principal" in context
    assert "não as trate como valores atuais" in context
