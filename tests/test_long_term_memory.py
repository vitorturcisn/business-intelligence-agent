
"""Testes da memória de longo prazo persistente."""

import pytest

from business_agent.memory.long_term import LongTermMemoryStore


def test_memory_persists_across_store_instances(tmp_path):
    database = tmp_path / "memory.db"

    store_1 = LongTermMemoryStore(database)
    store_1.save_memory(
        kind="semantic",
        memory_key="receita_liquida",
        content="Receita líquida é o faturamento após os descontos.",
        tags=["métricas", "vendas"],
    )

    # Abre outra instância, simulando a reinicialização do componente.
    store_2 = LongTermMemoryStore(database)

    result = store_2.get_memory(
        "semantic",
        "receita_liquida",
    )

    assert result is not None
    assert "após os descontos" in result["content"]
    assert result["kind"] == "semantic"
    assert "vendas" in result["tags"]


def test_saving_same_key_updates_existing_memory(tmp_path):
    store = LongTermMemoryStore(tmp_path / "memory.db")

    store.save_memory(
        "procedural",
        "sales_rule",
        "Consultar as vendas mensais.",
    )
    store.save_memory(
        "procedural",
        "sales_rule",
        "Consultar as vendas e retornar receita líquida.",
    )

    results = store.search_memories(
        "receita líquida",
        kind="procedural",
    )

    assert len(results) == 1
    assert "receita líquida" in results[0]["content"]


def test_search_finds_memory_using_tags(tmp_path):
    store = LongTermMemoryStore(tmp_path / "memory.db")

    store.save_memory(
        "episodic",
        "analysis_2026_09",
        "Análise de vendas concluída para setembro.",
        tags=["faturamento", "setembro"],
    )

    results = store.search_memories("faturamento")

    assert len(results) == 1
    assert results[0]["memory_key"] == "analysis_2026_09"


def test_search_can_filter_by_memory_kind(tmp_path):
    store = LongTermMemoryStore(tmp_path / "memory.db")

    store.save_memory(
        "semantic",
        "revenue_definition",
        "Definição de receita líquida.",
    )
    store.save_memory(
        "episodic",
        "revenue_analysis",
        "Análise de receita líquida de setembro.",
    )

    results = store.search_memories(
        "receita líquida",
        kind="episodic",
    )

    assert len(results) == 1
    assert results[0]["kind"] == "episodic"


def test_invalid_kind_is_rejected(tmp_path):
    store = LongTermMemoryStore(tmp_path / "memory.db")

    with pytest.raises(ValueError, match="Tipo de memória inválido"):
        store.save_memory(
            "unknown",
            "key",
            "Conteúdo.",
        )


def test_empty_content_is_rejected(tmp_path):
    store = LongTermMemoryStore(tmp_path / "memory.db")

    with pytest.raises(ValueError, match="content não pode ficar vazio"):
        store.save_memory(
            "semantic",
            "key",
            "   ",
        )


def test_missing_memory_returns_none(tmp_path):
    store = LongTermMemoryStore(tmp_path / "memory.db")

    assert store.get_memory(
        "semantic",
        "inexistente",
    ) is None
