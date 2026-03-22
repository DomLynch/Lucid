"""
lucid/store.py — SQLite persistence for the memory system.

Implements MemoryStore interface using SQLite with numpy-based
vector similarity search. Replaces Hindsight's PostgreSQL + pgvector.

Tables:
- banks: Memory bank metadata
- facts: Stored facts with embeddings (as JSON arrays)
- entities: Named entities with fact associations
- links: Memory links (semantic/temporal/entity/causal)
- observations: Consolidated insights

For <10K facts, SQLite + numpy cosine similarity is sufficient.
For larger scale, swap to PostgreSQL + pgvector via the same interface.
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from lucid.types import (
    Bank,
    Entity,
    EntityType,
    Fact,
    FactKind,
    FactType,
    MemoryLink,
    Observation,
)

_log = logging.getLogger("lucid.store")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS banks (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT '',
    mission TEXT NOT NULL DEFAULT '',
    background TEXT NOT NULL DEFAULT '',
    disposition TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY,
    bank_id TEXT NOT NULL,
    text TEXT NOT NULL DEFAULT '',
    what TEXT NOT NULL DEFAULT '',
    "when" TEXT NOT NULL DEFAULT 'N/A',
    "where" TEXT NOT NULL DEFAULT 'N/A',
    who TEXT NOT NULL DEFAULT 'N/A',
    why TEXT NOT NULL DEFAULT 'N/A',
    fact_type TEXT NOT NULL DEFAULT 'world',
    fact_kind TEXT NOT NULL DEFAULT 'conversation',
    event_date TEXT,
    occurred_start TEXT,
    occurred_end TEXT,
    confidence REAL NOT NULL DEFAULT 1.0,
    embedding TEXT NOT NULL DEFAULT '[]',
    source_context TEXT NOT NULL DEFAULT '',
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    bank_id TEXT NOT NULL,
    text TEXT NOT NULL,
    entity_type TEXT NOT NULL DEFAULT 'other',
    embedding TEXT NOT NULL DEFAULT '[]',
    fact_ids TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS links (
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    link_type TEXT NOT NULL DEFAULT 'semantic',
    strength REAL NOT NULL DEFAULT 0.5,
    PRIMARY KEY (source_id, target_id, link_type)
);

CREATE TABLE IF NOT EXISTS observations (
    id TEXT PRIMARY KEY,
    bank_id TEXT NOT NULL,
    text TEXT NOT NULL DEFAULT '',
    entity_id TEXT,
    source_fact_ids TEXT NOT NULL DEFAULT '[]',
    embedding TEXT NOT NULL DEFAULT '[]',
    confidence REAL NOT NULL DEFAULT 0.8,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_facts_bank ON facts(bank_id);
CREATE INDEX IF NOT EXISTS idx_entities_bank ON entities(bank_id);
CREATE INDEX IF NOT EXISTS idx_observations_bank ON observations(bank_id);
"""


class SQLiteMemoryStore:
    """SQLite-based memory store with numpy vector search.

    Implements the MemoryStore Protocol. Embeddings stored as JSON
    arrays, similarity computed via numpy cosine distance.
    """

    def __init__(self, db_path: str | Path = "lucid.db") -> None:
        self._db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None
        self._ensure_schema()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def _ensure_schema(self) -> None:
        conn = self._get_conn()
        conn.executescript(_SCHEMA)
        conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Bank operations
    # ------------------------------------------------------------------

    async def create_bank(self, bank: Bank) -> None:
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO banks (id, name, mission, background, disposition) VALUES (?, ?, ?, ?, ?)",
            (bank.id, bank.name, bank.mission, bank.background, json.dumps(bank.disposition)),
        )
        conn.commit()

    async def get_bank(self, bank_id: str) -> Bank:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM banks WHERE id = ?", (bank_id,)).fetchone()
        if row is None:
            raise KeyError(f"Bank '{bank_id}' not found")
        return Bank(
            id=row["id"], name=row["name"], mission=row["mission"],
            background=row["background"], disposition=json.loads(row["disposition"]),
        )

    # ------------------------------------------------------------------
    # Fact operations
    # ------------------------------------------------------------------

    async def save_facts(self, facts: list[Fact]) -> None:
        conn = self._get_conn()
        for f in facts:
            conn.execute(
                """INSERT OR REPLACE INTO facts
                   (id, bank_id, text, what, "when", "where", who, why,
                    fact_type, fact_kind, event_date, occurred_start, occurred_end,
                    confidence, embedding, source_context, metadata, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (f.id, f.bank_id, f.text, f.what, f.when, f.where, f.who, f.why,
                 f.fact_type.value, f.fact_kind.value, f.event_date,
                 f.occurred_start, f.occurred_end, f.confidence,
                 json.dumps(f.embedding), f.source_context,
                 json.dumps(f.metadata), f.created_at),
            )
        conn.commit()

    async def get_facts(self, bank_id: str, fact_ids: list[str] | None = None) -> list[Fact]:
        conn = self._get_conn()
        if fact_ids:
            placeholders = ",".join("?" * len(fact_ids))
            rows = conn.execute(
                f"SELECT * FROM facts WHERE bank_id = ? AND id IN ({placeholders})",
                [bank_id] + fact_ids,
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM facts WHERE bank_id = ?", (bank_id,)).fetchall()
        return [self._row_to_fact(row) for row in rows]

    async def search_facts_by_embedding(
        self, bank_id: str, query_embedding: list[float], limit: int = 50,
    ) -> list[Fact]:
        """Cosine similarity search using numpy."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM facts WHERE bank_id = ? AND embedding != '[]'", (bank_id,)
        ).fetchall()

        if not rows:
            return []

        query_vec = np.array(query_embedding)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []

        scored: list[tuple[float, Any]] = []
        for row in rows:
            emb = json.loads(row["embedding"])
            if not emb:
                continue
            fact_vec = np.array(emb)
            fact_norm = np.linalg.norm(fact_vec)
            if fact_norm == 0:
                continue
            similarity = float(np.dot(query_vec, fact_vec) / (query_norm * fact_norm))
            scored.append((similarity, row))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, row in scored[:limit]:
            fact = self._row_to_fact(row)
            fact.relevance = score
            results.append(fact)

        return results

    async def search_facts_by_text(
        self, bank_id: str, query: str, limit: int = 50,
    ) -> list[Fact]:
        """Simple substring search (BM25 replacement for SQLite)."""
        conn = self._get_conn()
        query_lower = f"%{query.lower()}%"
        rows = conn.execute(
            "SELECT * FROM facts WHERE bank_id = ? AND LOWER(text) LIKE ? LIMIT ?",
            (bank_id, query_lower, limit),
        ).fetchall()

        results = []
        for i, row in enumerate(rows):
            fact = self._row_to_fact(row)
            fact.relevance = 1.0 - (i * 0.05)  # Simple rank-based score
            results.append(fact)

        return results

    # ------------------------------------------------------------------
    # Entity operations
    # ------------------------------------------------------------------

    async def save_entities(self, entities: list[Entity]) -> None:
        conn = self._get_conn()
        for e in entities:
            conn.execute(
                """INSERT OR REPLACE INTO entities
                   (id, bank_id, text, entity_type, embedding, fact_ids, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (e.id, e.bank_id, e.text, e.entity_type.value,
                 json.dumps(e.embedding), json.dumps(e.fact_ids), e.created_at),
            )
        conn.commit()

    async def get_entities(self, bank_id: str) -> list[Entity]:
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM entities WHERE bank_id = ?", (bank_id,)).fetchall()
        return [
            Entity(
                id=row["id"], bank_id=row["bank_id"], text=row["text"],
                entity_type=EntityType(row["entity_type"]),
                embedding=json.loads(row["embedding"]),
                fact_ids=json.loads(row["fact_ids"]),
            )
            for row in rows
        ]

    async def get_entity_by_text(self, bank_id: str, text: str) -> Entity | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM entities WHERE bank_id = ? AND LOWER(text) = LOWER(?)",
            (bank_id, text),
        ).fetchone()
        if row is None:
            return None
        return Entity(
            id=row["id"], bank_id=row["bank_id"], text=row["text"],
            entity_type=EntityType(row["entity_type"]),
            embedding=json.loads(row["embedding"]),
            fact_ids=json.loads(row["fact_ids"]),
        )

    # ------------------------------------------------------------------
    # Link operations
    # ------------------------------------------------------------------

    async def save_links(self, links: list[MemoryLink]) -> None:
        conn = self._get_conn()
        for link in links:
            conn.execute(
                "INSERT OR REPLACE INTO links (source_id, target_id, link_type, strength) VALUES (?, ?, ?, ?)",
                (link.source_id, link.target_id, link.link_type.value, link.strength),
            )
        conn.commit()

    async def get_linked_facts(
        self, bank_id: str, fact_id: str, link_type: str | None = None, limit: int = 20,
    ) -> list[Fact]:
        conn = self._get_conn()
        if link_type:
            rows = conn.execute(
                """SELECT f.* FROM facts f
                   JOIN links l ON (l.target_id = f.id OR l.source_id = f.id)
                   WHERE (l.source_id = ? OR l.target_id = ?)
                   AND l.link_type = ?
                   AND f.id != ?
                   AND f.bank_id = ?
                   LIMIT ?""",
                (fact_id, fact_id, link_type, fact_id, bank_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT f.* FROM facts f
                   JOIN links l ON (l.target_id = f.id OR l.source_id = f.id)
                   WHERE (l.source_id = ? OR l.target_id = ?)
                   AND f.id != ?
                   AND f.bank_id = ?
                   LIMIT ?""",
                (fact_id, fact_id, fact_id, bank_id, limit),
            ).fetchall()
        return [self._row_to_fact(row) for row in rows]

    # ------------------------------------------------------------------
    # Observation operations
    # ------------------------------------------------------------------

    async def save_observations(self, observations: list[Observation]) -> None:
        conn = self._get_conn()
        for obs in observations:
            conn.execute(
                """INSERT OR REPLACE INTO observations
                   (id, bank_id, text, entity_id, source_fact_ids, embedding, confidence, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (obs.id, obs.bank_id, obs.text, obs.entity_id,
                 json.dumps(obs.source_fact_ids), json.dumps(obs.embedding),
                 obs.confidence, obs.created_at),
            )
        conn.commit()

    async def get_observations(self, bank_id: str, limit: int = 20) -> list[Observation]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM observations WHERE bank_id = ? LIMIT ?", (bank_id, limit)
        ).fetchall()
        return [
            Observation(
                id=row["id"], bank_id=row["bank_id"], text=row["text"],
                entity_id=row["entity_id"],
                source_fact_ids=json.loads(row["source_fact_ids"]),
                embedding=json.loads(row["embedding"]),
                confidence=row["confidence"],
            )
            for row in rows
        ]

    async def search_observations_by_embedding(
        self, bank_id: str, query_embedding: list[float], limit: int = 10,
    ) -> list[Observation]:
        """Cosine similarity search on observations."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM observations WHERE bank_id = ? AND embedding != '[]'", (bank_id,)
        ).fetchall()

        if not rows:
            return []

        query_vec = np.array(query_embedding)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []

        scored = []
        for row in rows:
            emb = json.loads(row["embedding"])
            if not emb:
                continue
            obs_vec = np.array(emb)
            obs_norm = np.linalg.norm(obs_vec)
            if obs_norm == 0:
                continue
            similarity = float(np.dot(query_vec, obs_vec) / (query_norm * obs_norm))
            scored.append((similarity, row))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            Observation(
                id=row["id"], bank_id=row["bank_id"], text=row["text"],
                entity_id=row["entity_id"],
                source_fact_ids=json.loads(row["source_fact_ids"]),
                embedding=json.loads(row["embedding"]),
                confidence=row["confidence"],
            )
            for _, row in scored[:limit]
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _row_to_fact(self, row: sqlite3.Row) -> Fact:
        return Fact(
            id=row["id"],
            bank_id=row["bank_id"],
            text=row["text"],
            what=row["what"],
            when=row["when"],
            where=row["where"],
            who=row["who"],
            why=row["why"],
            fact_type=FactType(row["fact_type"]),
            fact_kind=FactKind(row["fact_kind"]),
            event_date=row["event_date"],
            occurred_start=row["occurred_start"],
            occurred_end=row["occurred_end"],
            confidence=row["confidence"],
            embedding=json.loads(row["embedding"]),
            source_context=row["source_context"],
            metadata=json.loads(row["metadata"]),
            created_at=row["created_at"],
        )
