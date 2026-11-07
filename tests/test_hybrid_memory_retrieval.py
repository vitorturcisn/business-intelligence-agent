import pytest

from business_agent.memory.long_term import LongTermMemoryStore
from business_agent.memory.hybrid_retrieval import HybridMemoryRetriever


def test_semantic_search_finds_paraphrased_memory(tmp_path):
    store = LongTermMemoryStore(tmp_path / "memory.db")

    store.save_memory(
        kind="semantic",
        memory_key="receita_liquida",
        content=(
            "Receita líquida corresponde ao faturamento bruto "
            "após a dedução dos descontos comerciais e devoluções."
        ),
        tags=["kpi", "vendas"],
    )

    store.save_memory(
        kind="semantic",
        memory_key="taxa_conversao",
        content=(
            "Taxa de conversão representa a proporção de visitantes "
            "que realizaram compras durante determinado período."
        ),
        tags=["marketing", "conversao"],
    )

    retriever = HybridMemoryRetriever(
        store,
        index_dir=tmp_path / "faiss_index",
    )

    results = retriever.search(
        "Quanto sobra do valor vendido depois de retirar os abatimentos?",
        limit=1,
    )

    assert len(results) == 1
    assert results[0]["memory_key"] == "receita_liquida"
    assert "semantic" in results[0]["retrieval_methods"]


def test_hybrid_search_combines_lexical_and_semantic_results(tmp_path):
    store = LongTermMemoryStore(tmp_path / "memory.db")

    store.save_memory(
        kind="semantic",
        memory_key="revenue_definition",
        content=(
            "Receita líquida é o faturamento após descontos "
            "e devoluções."
        ),
        tags=["receita", "kpi"],
    )

    results = store.search_memories_hybrid(
        "receita líquida descontos",
        limit=3,
    )

    assert len(results) >= 1
    assert results[0]["memory_key"] == "revenue_definition"
    assert set(results[0]["retrieval_methods"]) == {
        "lexical",
        "semantic",
    }
    assert results[0]["hybrid_score"] > 0


def test_semantic_search_filters_memory_kind(tmp_path):
    store = LongTermMemoryStore(tmp_path / "memory.db")

    store.save_memory(
        kind="semantic",
        memory_key="revenue",
        content="Receita líquida considera os descontos comerciais.",
    )

    store.save_memory(
        kind="procedural",
        memory_key="revenue_process",
        content="Consultar vendas no banco SQL e validar os descontos.",
    )

    retriever = HybridMemoryRetriever(
        store,
        index_dir=tmp_path / "faiss_index",
    )

    results = retriever.search(
        "Como analisar receita e descontos?",
        kind="semantic",
        limit=5,
    )

    assert results
    assert all(item["kind"] == "semantic" for item in results)


def test_vector_index_refreshes_after_memory_update(tmp_path):
    store = LongTermMemoryStore(tmp_path / "memory.db")

    store.save_memory(
        kind="semantic",
        memory_key="revenue_definition",
        content="Receita líquida é faturamento menos descontos.",
    )

    retriever = HybridMemoryRetriever(
        store,
        index_dir=tmp_path / "faiss_index",
    )

    first_results = retriever.search(
        "Qual é a receita líquida?",
        limit=1,
    )

    assert first_results[0]["content"] == (
        "Receita líquida é faturamento menos descontos."
    )

    store.save_memory(
        kind="semantic",
        memory_key="revenue_definition",
        content="Receita líquida é faturamento menos impostos.",
    )

    updated_results = retriever.search(
        "Qual é a receita líquida?",
        limit=1,
    )

    assert updated_results[0]["content"] == (
        "Receita líquida é faturamento menos impostos."
    )


def test_hybrid_search_rejects_invalid_limit(tmp_path):
    store = LongTermMemoryStore(tmp_path / "memory.db")

    with pytest.raises(ValueError, match="limit"):
        store.search_memories_hybrid(
            "receita",
            limit=0,
        )


def test_empty_memory_store_returns_no_semantic_results(tmp_path):
    store = LongTermMemoryStore(tmp_path / "memory.db")

    retriever = HybridMemoryRetriever(
        store,
        index_dir=tmp_path / "faiss_index",
    )

    assert retriever.search("receita líquida") == []


def test_semantic_search_recovers_without_lexical_overlap(tmp_path):
    store = LongTermMemoryStore(tmp_path / "memory.db")

    store.save_memory(
        kind="semantic",
        memory_key="receita_liquida",
        content=(
            "Receita l?quida corresponde ao faturamento bruto "
            "ap?s a dedu??o dos descontos comerciais e devolu??es."
        ),
        tags=["kpi", "receita", "vendas"],
    )

    store.save_memory(
        kind="semantic",
        memory_key="taxa_conversao",
        content=(
            "Taxa de convers?o representa a propor??o de visitantes "
            "que realizaram compras."
        ),
        tags=["marketing"],
    )

    question = (
        "Quanto sobra do valor vendido depois de retirar "
        "os abatimentos?"
    )

    # A busca lexical n?o deve encontrar correspond?ncias artificiais.
    lexical_results = store.search_memories(
        question,
        limit=5,
    )

    assert lexical_results == []

    retriever = HybridMemoryRetriever(
        store,
        index_dir=tmp_path / "faiss_index",
    )

    semantic_results = retriever.search(
        question,
        limit=1,
    )

    assert semantic_results
    assert semantic_results[0]["memory_key"] == "receita_liquida"

    hybrid_results = retriever.search_hybrid(
        question,
        limit=1,
    )

    assert hybrid_results[0]["memory_key"] == "receita_liquida"
    assert hybrid_results[0]["retrieval_methods"] == ["semantic"]
