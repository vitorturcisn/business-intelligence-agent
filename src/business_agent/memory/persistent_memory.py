
from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any

from business_agent.memory.long_term import LongTermMemoryStore


_EXPLICIT_MEMORY_PATTERN = re.compile(
    r"^\s*(?:lembre-se|lembre|guarde|salve|registre)"
    r"(?:\s+na\s+mem[oó]ria)?\s*(?:que|:)\s*(?P<content>.+?)\s*$",
    flags=re.IGNORECASE | re.DOTALL,
)


_PERSONAL_RECALL_MARKERS = (
    "eu mencionei",
    "mencionei",
    "eu disse",
    "eu falei",
    "que eu mencionei",
    "que mencionei",
    "eu pedi para lembrar",
    "o que pedi para guardar",
    "minha preferencia",
    "minhas preferencias",
)

_PERSONAL_RECALL_STOPWORDS = {
    "a", "as", "o", "os", "um", "uma", "de", "do", "da", "dos", "das",
    "em", "no", "na", "nos", "nas", "para", "por", "com", "e", "ou",
    "que", "qual", "quais", "como", "quando", "onde", "foi", "era",
    "eu", "meu", "minha", "meus", "minhas", "voce", "seu", "sua",
    "mencionei", "mencionar", "disse", "falei", "pedi", "guardar",
    "lembra", "lembre", "lembrar", "principalmente",
}


def _normalize_for_key(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    normalized = "".join(
        char for char in normalized if not unicodedata.combining(char)
    )
    return re.sub(r"\s+", " ", normalized).strip()


def _tokenize(text: str) -> set[str]:
    normalized = _normalize_for_key(text)
    tokens = set(re.findall(r"[a-z0-9]+", normalized))
    return tokens - _PERSONAL_RECALL_STOPWORDS


def extract_explicit_memory_request(text: str) -> str | None:
    """Extract content only when the user explicitly requests memory storage."""
    if not isinstance(text, str) or not text.strip():
        return None

    match = _EXPLICIT_MEMORY_PATTERN.match(text)
    if not match:
        return None

    content = match.group("content").strip()
    return content or None


def _is_personal_recall_query(query: str) -> bool:
    normalized = _normalize_for_key(query)
    return any(
        marker in normalized
        for marker in _PERSONAL_RECALL_MARKERS
    )


def _is_explicit_user_memory(memory: dict[str, Any]) -> bool:
    memory_key = str(memory.get("memory_key", ""))

    if memory_key.startswith("explicit_user_memory_"):
        return True

    tags = memory.get("tags", [])

    if isinstance(tags, str):
        tags = [tags]

    if isinstance(tags, (list, tuple, set)):
        normalized_tags = {
            _normalize_for_key(str(tag))
            for tag in tags
        }
        return "explicit-user-memory" in normalized_tags

    return False


class PersistentMemoryManager:
    """Coordinates explicit writes and persistent memory retrieval.

    SQLite remains the persistent source of truth. FAISS remains available
    through the hybrid retrieval method for ordinary contextual questions.

    Personal-recall questions prioritize only memories explicitly saved
    at the user's request.
    """

    def __init__(
        self,
        store: LongTermMemoryStore | None = None,
    ) -> None:
        self.store = store or LongTermMemoryStore()

    def save_explicit_memory(
        self,
        content: str,
    ) -> dict[str, Any]:
        if not isinstance(content, str) or not content.strip():
            raise ValueError(
                "O conteudo da memoria nao pode ficar vazio."
            )

        cleaned = content.strip()

        digest = hashlib.sha256(
            _normalize_for_key(cleaned).encode("utf-8")
        ).hexdigest()[:16]

        memory_key = f"explicit_user_memory_{digest}"

        return self.store.save_memory(
            kind="semantic",
            memory_key=memory_key,
            content=cleaned,
            tags=[
                "explicit-user-memory",
                "user-provided",
                "business-agent",
            ],
        )

    def _retrieve_explicit_user_memories(
        self,
        query: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        memories = self.store.list_memories(
            kind="semantic",
            limit=10000,
        )

        explicit_memories = [
            memory
            for memory in memories
            if _is_explicit_user_memory(memory)
        ]

        if not explicit_memories:
            return []

        query_terms = _tokenize(query)

        ranked: list[tuple[int, str, dict[str, Any]]] = []

        for memory in explicit_memories:
            content = str(memory.get("content", ""))
            memory_key = str(memory.get("memory_key", ""))

            memory_terms = _tokenize(
                f"{content} {memory_key}"
            )

            overlap = len(query_terms.intersection(memory_terms))

            # If the query has meaningful terms, do not return
            # memories that have no relation to those terms.
            if query_terms and overlap == 0:
                continue

            updated_at = str(
                memory.get("updated_at")
                or memory.get("created_at")
                or ""
            )

            ranked.append((overlap, updated_at, memory))

        ranked.sort(
            key=lambda item: (item[0], item[1]),
            reverse=True,
        )

        results = []

        for _, _, memory in ranked[:limit]:
            result = dict(memory)
            result["retrieval_methods"] = [
                "explicit-user-memory"
            ]
            results.append(result)

        return results

    def retrieve(
        self,
        query: str,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            return []

        cleaned_query = query.strip()

        # Personal-recall questions must use explicit user memories.
        # Do not mix them with generic semantic or episodic records.
        if _is_personal_recall_query(cleaned_query):
            return self._retrieve_explicit_user_memories(
                query=cleaned_query,
                limit=limit,
            )

        # Ordinary contextual questions continue using hybrid retrieval.
        return self.store.search_memories_hybrid(
            query=cleaned_query,
            limit=limit,
        )

    @staticmethod
    def format_context(
        memories: list[dict[str, Any]],
    ) -> str:
        """Format retrieved memories as context for the language model."""
        if not memories:
            return ""

        sections = []

        for memory in memories:
            content = str(memory.get("content", ""))[:1200]
            memory_key = str(
                memory.get("memory_key", "unknown")
            )

            if _is_explicit_user_memory(memory):
                source_description = (
                    "Mem\u00f3ria explicitamente solicitada pelo usu\u00e1rio"
                )
            else:
                source_description = "Registro contextual recuperado"

            sections.append(
                "- Origem: {origin}\n"
                "  Tipo: {kind}\n"
                "  Chave: {key}\n"
                "  Informa\u00e7\u00e3o: {content}".format(
                    origin=source_description,
                    kind=memory.get("kind", "unknown"),
                    key=memory_key,
                    content=content,
                )
            )

        return (
            "\n\n[MEM\u00d3RIAS PERSISTENTES RECUPERADAS]\n"
            "As informa\u00e7\u00f5es abaixo foram registradas anteriormente "
            "a pedido do usu\u00e1rio ou recuperadas como contexto. "
            "Elas servem como contexto; n\u00e3o as trate como valores "
            "atuais do banco. Para m\u00e9tricas quantitativas atuais, "
            "consulte as ferramentas de dados.\n"
            + "\n".join(sections)
            + "\n[FIM DAS MEM\u00d3RIAS PERSISTENTES]\n"
        )
