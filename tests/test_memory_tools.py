import json

import pytest

from business_agent.memory.long_term import LongTermMemoryStore
from business_agent.memory import tools as memory_tools


@pytest.fixture
def isolated_memory(tmp_path, monkeypatch):
    database = tmp_path / "test_agent_memory.db"
    store = LongTermMemoryStore(database)

    monkeypatch.setattr(
        memory_tools,
        "get_memory_store",
        lambda: store,
    )

    return store


def test_save_semantic_memory(isolated_memory):
    result = memory_tools.save_business_memory.invoke({
        "kind": "semantic",
        "memory_key": "kpi_receita_liquida",
        "content": "Receita líquida é receita bruta menos descontos.",
        "tags": ["kpi", "receita", "vendas"],
    })

    saved = json.loads(result)

    assert saved["kind"] == "semantic"
    assert saved["memory_key"] == "kpi_receita_liquida"


def test_save_episodic_memory(isolated_memory):
    result = memory_tools.save_business_memory.invoke({
        "kind": "episodic",
        "memory_key": "analise_vendas_2026_09",
        "content": (
            "Análise do dataset sintético de setembro de 2026: "
            "6 pedidos e receita líquida de R$ 25.250,00."
        ),
        "tags": ["vendas", "setembro", "2026"],
    })

    saved = json.loads(result)

    assert saved["kind"] == "episodic"


def test_save_procedural_memory(isolated_memory):
    result = memory_tools.save_business_memory.invoke({
        "kind": "procedural",
        "memory_key": "processo_analise_mensal",
        "content": (
            "Consultar SQL, validar os KPIs no glossário "
            "e registrar período e fonte."
        ),
        "tags": ["processo", "analise", "sql"],
    })

    saved = json.loads(result)

    assert saved["kind"] == "procedural"


def test_search_business_memory(isolated_memory):
    memory_tools.save_business_memory.invoke({
        "kind": "semantic",
        "memory_key": "kpi_receita",
        "content": "Receita líquida considera os descontos.",
        "tags": ["receita", "kpi"],
    })

    result = memory_tools.search_business_memory.invoke({
        "query": "receita líquida",
        "kind": "semantic",
        "limit": 5,
    })

    results = json.loads(result)

    assert len(results) == 1
    assert results[0]["memory_key"] == "kpi_receita"


def test_get_memory_by_exact_key(isolated_memory):
    memory_tools.save_business_memory.invoke({
        "kind": "procedural",
        "memory_key": "processo_sql",
        "content": "Consultar a tabela sales.",
    })

    result = memory_tools.get_business_memory.invoke({
        "kind": "procedural",
        "memory_key": "processo_sql",
    })

    parsed = json.loads(result)

    assert parsed["found"] is True
    assert parsed["memory"]["memory_key"] == "processo_sql"


def test_missing_memory_returns_not_found(isolated_memory):
    result = memory_tools.get_business_memory.invoke({
        "kind": "semantic",
        "memory_key": "inexistente",
    })

    assert json.loads(result)["found"] is False


def test_memory_survives_store_reinitialization(
    isolated_memory,
    tmp_path,
):
    memory_tools.save_business_memory.invoke({
        "kind": "episodic",
        "memory_key": "analise_persistente",
        "content": "Foi executada uma análise mensal de vendas.",
        "tags": ["vendas"],
    })

    reopened_store = LongTermMemoryStore(isolated_memory.db_path)

    result = reopened_store.get_memory(
        "episodic",
        "analise_persistente",
    )

    assert result is not None
    assert "análise mensal" in result["content"]


def test_empty_search_query_is_rejected(isolated_memory):
    with pytest.raises(Exception):
        memory_tools.search_business_memory.invoke({
            "query": "",
        })
