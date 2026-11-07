import json
from typing import Literal

from langchain_core.tools import tool

from business_agent.memory.long_term import LongTermMemoryStore


MemoryKind = Literal["semantic", "episodic", "procedural"]


def get_memory_store() -> LongTermMemoryStore:
    return LongTermMemoryStore()


@tool
def save_business_memory(
    kind: MemoryKind,
    memory_key: str,
    content: str,
    tags: list[str] | None = None,
) -> str:
    """
    Salva uma memória persistente do projeto Business Intelligence.

    Tipos:
    - semantic: fatos, conceitos e informações estáveis do negócio.
    - episodic: análises, consultas e acontecimentos relevantes.
    - procedural: processos, instruções e procedimentos recorrentes.

    Salve apenas informações úteis para interações futuras.
    """
    store = get_memory_store()

    result = store.save_memory(
        kind=kind,
        memory_key=memory_key,
        content=content,
        tags=tags,
    )

    return json.dumps(result, ensure_ascii=False, default=str)


@tool
def search_business_memory(
    query: str,
    kind: MemoryKind | None = None,
    limit: int = 5,
) -> str:
    """
    Pesquisa memórias persistentes usando palavras-chave.

    Utilize quando o usuário mencionar decisões anteriores,
    análises passadas, fatos conhecidos ou procedimentos já definidos.

    A busca atual é lexical, não vetorial.
    """
    store = get_memory_store()

    results = store.search_memories_hybrid(
        query=query,
        kind=kind,
        limit=limit,
    )

    return json.dumps(results, ensure_ascii=False, default=str)


@tool
def get_business_memory(
    kind: MemoryKind,
    memory_key: str,
) -> str:
    """
    Recupera uma memória persistente utilizando seu tipo e chave exata.
    """
    store = get_memory_store()

    result = store.get_memory(
        kind=kind,
        memory_key=memory_key,
    )

    if result is None:
        return json.dumps(
            {"found": False, "memory_key": memory_key},
            ensure_ascii=False,
        )

    return json.dumps(
        {"found": True, "memory": result},
        ensure_ascii=False,
        default=str,
    )
