
"""Memória de longo prazo persistente para o agente."""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal


MemoryKind = Literal["semantic", "episodic", "procedural"]

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "sample" / "agent_memory.db"

VALID_KINDS = {"semantic", "episodic", "procedural"}


class LongTermMemoryStore:
    """Armazena e recupera memórias persistentes em SQLite."""

    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB_PATH,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_database(self) -> None:
        """Cria a tabela caso ela ainda não exista."""

        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS long_term_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL CHECK (
                        kind IN ('semantic', 'episodic', 'procedural')
                    ),
                    memory_key TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tags TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(kind, memory_key)
                )
                """
            )
            connection.commit()

    @staticmethod
    def _validate_kind(kind: str) -> None:
        if kind not in VALID_KINDS:
            raise ValueError(
                f"Tipo de memória inválido: {kind!r}. "
                f"Utilize um destes: {sorted(VALID_KINDS)}."
            )

    @staticmethod
    def _validate_text(value: str, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} não pode ficar vazio.")

        return value.strip()

    def save_memory(
        self,
        kind: MemoryKind,
        memory_key: str,
        content: str,
        tags: list[str] | None = None,
    ) -> dict:
        """Cria ou atualiza uma memória identificada por tipo e chave."""

        self._validate_kind(kind)
        memory_key = self._validate_text(memory_key, "memory_key")
        content = self._validate_text(content, "content")

        normalized_tags = sorted(
            {
                tag.strip().casefold()
                for tag in (tags or [])
                if isinstance(tag, str) and tag.strip()
            }
        )
        tags_text = " ".join(normalized_tags)
        now = datetime.now(timezone.utc).isoformat()

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO long_term_memories (
                    kind,
                    memory_key,
                    content,
                    tags,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(kind, memory_key) DO UPDATE SET
                    content = excluded.content,
                    tags = excluded.tags,
                    updated_at = excluded.updated_at
                """,
                (
                    kind,
                    memory_key,
                    content,
                    tags_text,
                    now,
                    now,
                ),
            )
            connection.commit()

        return self.get_memory(kind, memory_key)

    def get_memory(
        self,
        kind: MemoryKind,
        memory_key: str,
    ) -> dict | None:
        """Busca uma memória pela categoria e pela chave exata."""

        self._validate_kind(kind)
        memory_key = self._validate_text(memory_key, "memory_key")

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM long_term_memories
                WHERE kind = ? AND memory_key = ?
                """,
                (kind, memory_key),
            ).fetchone()

        return dict(row) if row else None

    def search_memories(
        self,
        query: str,
        kind: MemoryKind | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Busca memórias por correspondência de palavras-chave.

        A consulta é lexical, não vetorial. Os resultados são
        ordenados pelo número de termos encontrados e pela atualização.
        """

        query = self._validate_text(query, "query")

        if kind is not None:
            self._validate_kind(kind)

        if not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit deve ser um inteiro entre 1 e 100.")

        terms = list(
            dict.fromkeys(re.findall(r"\w+", query.casefold()))
        )

        if not terms:
            return []

        where_clauses = []
        parameters: list[str | int] = []

        if kind is not None:
            where_clauses.append("kind = ?")
            parameters.append(kind)

        # Filtra no banco por pelo menos um dos termos.
        term_conditions = []

        for term in terms:
            term_conditions.append(
                """
                (
                    lower(memory_key) LIKE ?
                    OR lower(content) LIKE ?
                    OR lower(tags) LIKE ?
                )
                """
            )

            pattern = f"%{term}%"
            parameters.extend([pattern, pattern, pattern])

        where_clauses.append(
            "(" + " OR ".join(term_conditions) + ")"
        )

        sql = f"""
            SELECT *
            FROM long_term_memories
            WHERE {" AND ".join(where_clauses)}
            ORDER BY updated_at DESC
            LIMIT ?
        """

        parameters.append(limit)

        with self._connect() as connection:
            rows = connection.execute(
                sql,
                parameters,
            ).fetchall()

        results = [dict(row) for row in rows]

        # Reordena os candidatos pelo número de termos correspondentes.
        def score(memory: dict) -> int:
            searchable = " ".join(
                [
                    memory["memory_key"],
                    memory["content"],
                    memory["tags"],
                ]
            ).casefold()

            return sum(term in searchable for term in terms)

        results.sort(
            key=lambda item: (
                score(item),
                item["updated_at"],
            ),
            reverse=True,
        )

        return results

    def list_memories(
        self,
        kind: MemoryKind | None = None,
        limit: int = 10000,
    ) -> list[dict]:
        """Lista mem?rias para reconstru??o do ?ndice vetorial."""

        if kind is not None:
            self._validate_kind(kind)

        if not isinstance(limit, int) or not 1 <= limit <= 10000:
            raise ValueError("limit deve estar entre 1 e 10000.")

        with self._connect() as connection:
            if kind is None:
                rows = connection.execute(
                    """
                    SELECT *
                    FROM long_term_memories
                    ORDER BY id ASC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT *
                    FROM long_term_memories
                    WHERE kind = ?
                    ORDER BY id ASC
                    LIMIT ?
                    """,
                    (kind, limit),
                ).fetchall()

        return [dict(row) for row in rows]

    def search_memories_hybrid(
        self,
        query: str,
        kind: MemoryKind | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Combina busca lexical e sem?ntica com FAISS."""

        from business_agent.memory.hybrid_retrieval import (
            HybridMemoryRetriever,
        )

        retriever = HybridMemoryRetriever(self)

        return retriever.search_hybrid(
            query=query,
            kind=kind,
            limit=limit,
        )

