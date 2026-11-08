import unicodedata
from typing import Any

from langchain_core.documents import Document
from langchain_core.tools import tool

from business_agent.retrieval.faiss_store import load_faiss_index


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(
        char for char in text
        if not unicodedata.combining(char)
    )


FIELD_DEFINITION_MARKERS = (
    "quais campos", "quais colunas", "campos fisicos", "colunas fisicas",
    "dicionario de dados", "campos da tabela", "colunas da tabela",
    "estrutura da tabela",
)

FIELD_DEFINITION_SEARCH_QUERY = (
    "DicionÃ¡rio de dados da tabela sales seÃ§Ã£o Colunas: "
    "quantity: quantidade de unidades vendidas; "
    "unit_price: preÃ§o unitÃ¡rio do produto; "
    "discount_pct: percentual de desconto aplicado Ã  venda"
)


def is_field_definition_query(query: str) -> bool:
    normalized = normalize_text(query)
    return any(marker in normalized for marker in FIELD_DEFINITION_MARKERS)


def _is_field_definition_chunk(document: Document) -> bool:
    """Recognize explicit field-definition lines from the data dictionary."""
    source = normalize_text(str(document.metadata.get("source", "")))
    if "dicionario_dados" not in source:
        return False
    known_fields = {"quantity", "unit_price", "discount_pct"}
    for line in document.page_content.splitlines():
        candidate = line.strip().lstrip("-* ").strip().strip("`")
        field_name, separator, description = candidate.partition(":")
        if separator and field_name.strip().strip("`") in known_fields and description.strip():
            return True
    return False


def transform_query(query: str) -> str:
    """
    Expande perguntas com termos técnicos e conceitos relacionados.
    A transformação é determinística e não depende de LLM.
    """
    normalized = normalize_text(query)
    expansions = []

    if any(term in normalized for term in [
        "selic",
        "taxa basica de juros",
        "taxa de juros",
        "juros basicos",
        "juros basicos da economia",
        "juros da economia brasileira",
        "juro basico",
        "dados dos juros",
        "fonte dos juros",
    ]):
        expansions.append(
            "taxa Selic Banco Central SGS série 432 "
            "fonte externa API observações e limitações"
        )

    if any(term in normalized for term in [
        "receita liquida",
        "faturamento liquido",
        "valor liquido",
        "faturamento apos descontos",
        "faturamento apos os descontos",
        "vendas apos descontos",
        "vendas apos os descontos",
        "vendas depois dos descontos",
        "faturamento depois dos descontos",
        "valor das vendas depois dos descontos",
    ]):
        expansions.append(
            "receita líquida receita bruta descontos "
            "quantity unit_price discount_pct fórmula"
        )

    if is_field_definition_query(query):
        expansions.append(
            "dicionÃ¡rio de dados tabela sales campos fÃ­sicos colunas "
            "descriÃ§Ã£o dos campos quantity unit_price discount_pct"
        )

    if any(term in normalized for term in [
        "unidades vendidas",
        "quantidade vendida",
        "volume vendido",
        "numero de pedidos",
        "quantidade de pedidos",
    ]):
        expansions.append(
            "soma quantity unidades vendidas "
            "sale_id distintos número de pedidos"
        )

    if not expansions:
        return query

    return (
        f"{query}\n\n"
        f"Conceitos relacionados: {'; '.join(expansions)}"
    )


def format_documents(documents: list[Document]) -> str:
    if not documents:
        return "Nenhum documento relevante foi encontrado."

    results = []

    for position, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "Fonte desconhecida")
        start_index = document.metadata.get("start_index")
        distance = document.metadata.get("retrieval_distance")

        distance_text = (
            f"Distância L2: {distance:.4f}\n"
            if distance is not None
            else ""
        )

        results.append(
            f"Resultado {position}\n"
            f"Fonte: {source}\n"
            f"Posição no documento: {start_index}\n"
            f"{distance_text}"
            f"Conteúdo:\n{document.page_content}"
        )

    return "\n\n---\n\n".join(results)


def search_knowledge_base(
    query: str,
    vector_store: Any | None = None,
    k: int = 4,
) -> str:
    """
    Busca no FAISS usando consulta original e consulta expandida.
    Combina os candidatos sem duplicar documentos e mantém
    a menor distância L2 encontrada para cada documento.
    """
    if not query or not query.strip():
        raise ValueError("A consulta não pode estar vazia.")

    if k < 1:
        raise ValueError("O número de resultados deve ser positivo.")

    store = vector_store if vector_store is not None else load_faiss_index()

    expanded_query = transform_query(query)

    queries: list[tuple[str, bool]] = [(query, False)]

    if expanded_query != query:
        queries.append((expanded_query, False))

    if is_field_definition_query(query):
        queries.append((FIELD_DEFINITION_SEARCH_QUERY, True))

    best_candidates = {}
    field_definition_candidates = set()

    for search_query, is_targeted_search in queries:
        matches = store.similarity_search_with_score(
            search_query,
            k=k,
        )

        for document, distance in matches:
            identity = (
                document.metadata.get("source"),
                document.metadata.get("start_index"),
                document.page_content,
            )

            if is_targeted_search and _is_field_definition_chunk(document):
                field_definition_candidates.add(identity)

            numeric_distance = float(distance)

            if (
                identity not in best_candidates
                or numeric_distance < best_candidates[identity][0]
            ):
                best_candidates[identity] = (
                    numeric_distance,
                    document,
                )

    ranked_candidates = list(best_candidates.items())
    ranked_candidates.sort(
        key=lambda item: (
            item[0] not in field_definition_candidates,
            item[1][0],
        )
    )
    ranked = [candidate for _, candidate in ranked_candidates[:k]]

    selected_documents = []

    for distance, document in ranked:
        document.metadata["retrieval_distance"] = distance
        selected_documents.append(document)

    return format_documents(selected_documents)


@tool
def business_knowledge_search(query: str) -> str:
    """
    Busca informações na base de conhecimento do negócio.
    Utilize para consultar definições de KPIs, fórmulas de vendas,
    dicionário de dados, regras de negócio, fontes e limitações.
    """
    return search_knowledge_base(query, k=4)
