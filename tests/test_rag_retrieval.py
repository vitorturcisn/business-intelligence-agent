import pytest

from langchain_core.documents import Document

from business_agent.tools.rag_tool import (
    business_knowledge_search,
    search_knowledge_base,
    transform_query,
)


class FakeVectorStore:
    def __init__(self, documents):
        self.documents = documents
        self.calls = []

    def similarity_search_with_score(self, query, k=4):
        self.calls.append((query, k))

        return [
            (document, float(index + 1))
            for index, document in enumerate(self.documents[:k])
        ]


def test_search_returns_document_content_and_source():
    fake_store = FakeVectorStore([
        Document(
            page_content="Receita líquida é a receita bruta menos descontos.",
            metadata={"source": "01_glossario_kpis.md"},
        )
    ])

    result = search_knowledge_base(
        "Como calcular receita líquida?",
        vector_store=fake_store,
    )

    assert "receita bruta menos descontos" in result
    assert "01_glossario_kpis.md" in result
    assert "Distância L2" in result


def test_search_uses_original_query_and_transformed_query():
    fake_store = FakeVectorStore([
        Document(
            page_content="A Selic é consultada na API do Banco Central.",
            metadata={"source": "04_fontes_e_limitacoes.md"},
        )
    ])

    search_knowledge_base(
        "Qual é a fonte da taxa Selic?",
        vector_store=fake_store,
        k=3,
    )

    assert len(fake_store.calls) == 2
    assert fake_store.calls[0][0] == "Qual é a fonte da taxa Selic?"
    assert fake_store.calls[0][1] == 3
    assert "SGS" in fake_store.calls[1][0]


def test_transform_query_expands_selic_terms():
    result = transform_query("Qual é a fonte da taxa Selic?")

    assert "Banco Central" in result
    assert "SGS" in result
    assert "série 432" in result


def test_transform_query_expands_revenue_terms():
    result = transform_query("Como calcular a receita líquida?")

    assert "receita bruta" in result
    assert "discount_pct" in result


def test_transform_query_expands_order_and_quantity_terms():
    result = transform_query(
        "Qual a diferença entre unidades vendidas e número de pedidos?"
    )

    assert "quantity" in result
    assert "sale_id" in result


def test_transform_query_preserves_unmatched_query():
    query = "Quais são os produtos disponíveis?"

    assert transform_query(query) == query


def test_search_rejects_empty_query():
    fake_store = FakeVectorStore([])

    with pytest.raises(ValueError):
        search_knowledge_base("", vector_store=fake_store)


def test_search_rejects_invalid_k():
    fake_store = FakeVectorStore([])

    with pytest.raises(ValueError):
        search_knowledge_base(
            "Consulta de teste",
            vector_store=fake_store,
            k=0,
        )


def test_search_handles_empty_results():
    fake_store = FakeVectorStore([])

    result = search_knowledge_base(
        "Informação inexistente",
        vector_store=fake_store,
    )

    assert "Nenhum documento relevante" in result


def test_tool_has_expected_name():
    assert business_knowledge_search.name == "business_knowledge_search"


def test_transform_query_expands_brazilian_base_interest_synonyms():
    query = "De onde vêm os dados dos juros básicos da economia brasileira?"
    result = transform_query(query)

    assert result != query
    assert "Banco Central" in result
    assert "SGS" in result
    assert "série 432" in result


def test_transform_query_expands_faturamento_after_discounts():
    query = "Como descobrir o faturamento após os descontos?"
    result = transform_query(query)

    assert result != query
    assert "receita líquida" in result
    assert "discount_pct" in result


def test_transform_query_expands_sales_after_discounts_synonym():
    query = "Como calcular as vendas depois dos descontos?"
    result = transform_query(query)

    assert result != query
    assert "receita bruta" in result
    assert "discount_pct" in result

