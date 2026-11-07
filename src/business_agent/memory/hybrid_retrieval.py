from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

if TYPE_CHECKING:
    from business_agent.memory.long_term import LongTermMemoryStore


BASE_DIR = Path(__file__).resolve().parents[3]

DEFAULT_INDEX_ROOT = (
    BASE_DIR / "data" / "indexes" / "agent_memory_faiss"
)

_VECTOR_CACHE: dict[tuple[str, str], FAISS] = {}


@lru_cache(maxsize=1)
def _get_embeddings():
    from business_agent.retrieval.faiss_store import build_embeddings

    return build_embeddings()


def _fingerprint(memories: list[dict]) -> str:
    payload = [
        {
            "id": item["id"],
            "kind": item["kind"],
            "memory_key": item["memory_key"],
            "content": item["content"],
            "tags": item["tags"],
            "updated_at": item["updated_at"],
        }
        for item in memories
    ]

    serialized = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()


def _build_documents(memories: list[dict]) -> list[Document]:
    documents = []

    for memory in memories:
        page_content = (
            f"Tipo de memória: {memory['kind']}\n"
            f"Chave: {memory['memory_key']}\n"
            f"Conteúdo: {memory['content']}\n"
            f"Tags: {memory['tags']}"
        )

        documents.append(
            Document(
                page_content=page_content,
                metadata={
                    "id": memory["id"],
                    "kind": memory["kind"],
                    "memory_key": memory["memory_key"],
                    "content": memory["content"],
                    "tags": memory["tags"],
                    "created_at": memory["created_at"],
                    "updated_at": memory["updated_at"],
                },
            )
        )

    return documents


class HybridMemoryRetriever:
    """
    Recuperação híbrida de memórias.

    SQLite permanece como fonte persistente de verdade.
    FAISS é um índice vetorial derivado e reconstruível.

    A recuperação combina:
    - Busca lexical do SQLite.
    - Similaridade semântica do FAISS.
    - Reciprocal Rank Fusion (RRF) para combinar rankings.
    """

    def __init__(
        self,
        store: LongTermMemoryStore,
        index_dir: str | Path | None = None,
    ) -> None:
        self.store = store

        if index_dir is None:
            database_id = hashlib.sha256(
                str(store.db_path.resolve()).encode("utf-8")
            ).hexdigest()[:16]

            self.index_dir = DEFAULT_INDEX_ROOT / database_id
        else:
            self.index_dir = Path(index_dir)

    def _load_or_build(
        self,
        memories: list[dict],
    ) -> FAISS:
        fingerprint = _fingerprint(memories)
        cache_key = (
            str(self.index_dir.resolve()),
            fingerprint,
        )

        if cache_key in _VECTOR_CACHE:
            return _VECTOR_CACHE[cache_key]

        manifest_path = self.index_dir / "index_manifest.json"
        faiss_path = self.index_dir / "index.faiss"
        pickle_path = self.index_dir / "index.pkl"

        if (
            manifest_path.is_file()
            and faiss_path.is_file()
            and pickle_path.is_file()
        ):
            try:
                manifest = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )

                if manifest.get("fingerprint") == fingerprint:
                    vector_store = FAISS.load_local(
                        folder_path=str(self.index_dir),
                        embeddings=_get_embeddings(),
                        allow_dangerous_deserialization=True,
                    )

                    _VECTOR_CACHE[cache_key] = vector_store
                    return vector_store

            except Exception:
                # Índice ausente, inválido ou incompatível:
                # será reconstruído com os registros do SQLite.
                pass

        documents = _build_documents(memories)

        vector_store = FAISS.from_documents(
            documents,
            _get_embeddings(),
        )

        self.index_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        vector_store.save_local(str(self.index_dir))

        manifest_path.write_text(
            json.dumps(
                {
                    "fingerprint": fingerprint,
                    "memory_count": len(memories),
                    "source": str(self.store.db_path.resolve()),
                    "embedding_model": (
                        "sentence-transformers/"
                        "paraphrase-multilingual-MiniLM-L12-v2"
                    ),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        # Remove do cache versões antigas do mesmo índice.
        for old_key in list(_VECTOR_CACHE):
            if old_key[0] == cache_key[0]:
                del _VECTOR_CACHE[old_key]

        _VECTOR_CACHE[cache_key] = vector_store
        return vector_store

    def search(
        self,
        query: str,
        kind: str | None = None,
        limit: int = 5,
    ) -> list[dict]:
        """Busca memórias por similaridade semântica."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query não pode ficar vazia.")

        if not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit deve estar entre 1 e 100.")

        memories = self.store.list_memories(
            kind=kind,
            limit=10000,
        )

        if not memories:
            return []

        vector_store = self._load_or_build(memories)

        search_args = {
            "query": query.strip(),
            "k": min(limit, len(memories)),
            "fetch_k": len(memories),
        }

        if kind is not None:
            search_args["filter"] = {"kind": kind}

        matches = vector_store.similarity_search_with_score(
            **search_args
        )

        results = []

        for document, distance in matches:
            memory = dict(document.metadata)
            memory["semantic_distance"] = float(distance)
            memory["retrieval_methods"] = ["semantic"]
            results.append(memory)

        return results

    def search_hybrid(
        self,
        query: str,
        kind: str | None = None,
        limit: int = 5,
    ) -> list[dict]:
        """
        Combina ranking lexical e semântico com Reciprocal Rank Fusion.

        A pontuação RRF combina posições nos rankings, não probabilidades.
        """

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query não pode ficar vazia.")

        if not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit deve estar entre 1 e 100.")

        candidate_limit = min(100, max(limit * 5, limit))

        lexical_results = self.store.search_memories(
            query=query,
            kind=kind,
            limit=candidate_limit,
        )

        semantic_results = self.search(
            query=query,
            kind=kind,
            limit=candidate_limit,
        )

        combined: dict[int, dict] = {}

        def add_ranked_results(
            records: list[dict],
            method: str,
        ) -> None:
            for rank, memory in enumerate(records, start=1):
                memory_id = int(memory["id"])

                if memory_id not in combined:
                    combined[memory_id] = dict(memory)
                    combined[memory_id]["hybrid_score"] = 0.0
                    combined[memory_id]["retrieval_methods"] = []

                item = combined[memory_id]

                item["hybrid_score"] += 1.0 / (60 + rank)

                if method not in item["retrieval_methods"]:
                    item["retrieval_methods"].append(method)

                item[f"{method}_rank"] = rank

                if method == "semantic":
                    item["semantic_distance"] = memory[
                        "semantic_distance"
                    ]

        add_ranked_results(
            lexical_results,
            "lexical",
        )

        add_ranked_results(
            semantic_results,
            "semantic",
        )

        results = sorted(
            combined.values(),
            key=lambda item: (
                item["hybrid_score"],
                item["updated_at"],
            ),
            reverse=True,
        )

        return results[:limit]
