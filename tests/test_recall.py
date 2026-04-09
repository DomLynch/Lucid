"""
Tests for lucid/recall.py — 4-way retrieval with RRF merge.

Covers:
1. RRF merge formula correctness
2. Semantic search integration
3. BM25 search integration
4. Entity graph search
5. Temporal search with date proximity
6. Token budget filtering
7. Fact type filtering
8. Reranking integration
9. Empty query handling
10. Full recall pipeline
"""

import math
from datetime import datetime, timezone
from typing import Any

import pytest

from lucid.recall import (
    _apply_token_budget,
    _rrf_merge,
    _temporal_search,
    recall,
)
from lucid.types import Budget, Entity, EntityType, Fact, FactType, RecallResult


# ---------------------------------------------------------------------------
# Mock implementations
# ---------------------------------------------------------------------------

class MockEmbedder:
    def __init__(self, dim=4):
        self._dim = dim

    async def embed(self, texts):
        return [[0.1] * self._dim for _ in texts]

    @property
    def dimension(self):
        return self._dim


class MockReranker:
    async def rerank(self, query, documents, top_k=10):
        # Reverse order (last becomes first) to prove reranking happened
        return [(i, 1.0 - i * 0.1) for i in reversed(range(min(top_k, len(documents))))]


class MockStore:
    def __init__(self, facts=None, entities=None):
        self._facts = {f.id: f for f in (facts or [])}
        self._entities = entities or []

    async def search_facts_by_embedding(self, bank_id, query_emb, limit=50):
        facts = list(self._facts.values())[:limit]
        for i, f in enumerate(facts):
            f.relevance = 1.0 - i * 0.1
        return facts

    async def search_facts_by_text(self, bank_id, query, limit=50):
        query_lower = query.lower()
        matches = [f for f in self._facts.values() if query_lower in f.text.lower()]
        for i, f in enumerate(matches):
            f.relevance = 1.0 - i * 0.1
        return matches[:limit]

    async def get_facts(self, bank_id, fact_ids=None):
        if fact_ids:
            return [self._facts[fid] for fid in fact_ids if fid in self._facts]
        return list(self._facts.values())

    async def get_entities(self, bank_id):
        return self._entities

    async def save_facts(self, facts): pass
    async def save_entities(self, entities): pass
    async def save_links(self, links): pass


def _make_facts(n, prefix="fact"):
    return [
        Fact(id=f"{prefix}_{i}", text=f"{prefix} number {i}", bank_id="test")
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Tests: RRF merge
# ---------------------------------------------------------------------------

class TestRRFMerge:
    def test_single_list(self):
        scores = _rrf_merge(
            semantic=[("a", 1.0), ("b", 0.5)],
            bm25=[],
            entity=[],
            temporal=[],
        )
        assert scores["a"] > scores["b"]

    def test_two_lists_boost(self):
        """Item appearing in both lists should score higher."""
        scores = _rrf_merge(
            semantic=[("a", 1.0), ("b", 0.5)],
            bm25=[("b", 1.0), ("c", 0.5)],
            entity=[],
            temporal=[],
        )
        # "b" appears in both → higher score
        assert scores["b"] > scores["a"]
        assert scores["b"] > scores["c"]

    def test_four_way_boost(self):
        """Item in all 4 lists should have highest score."""
        scores = _rrf_merge(
            semantic=[("x", 1.0)],
            bm25=[("x", 1.0)],
            entity=[("x", 1.0)],
            temporal=[("x", 1.0)],
        )
        # "x" gets contribution from all 4 lists
        expected = 4 * (1.0 / (60 + 1))  # rank 0 in all lists
        assert abs(scores["x"] - expected) < 0.001

    def test_empty_lists(self):
        scores = _rrf_merge([], [], [], [])
        assert scores == {}

    def test_rrf_formula_correct(self):
        """Verify exact RRF formula: 1/(k+rank+1) where k=60."""
        scores = _rrf_merge(
            semantic=[("a", 1.0), ("b", 0.5), ("c", 0.3)],
            bm25=[],
            entity=[],
            temporal=[],
        )
        assert abs(scores["a"] - 1.0 / 61) < 0.0001  # rank 0
        assert abs(scores["b"] - 1.0 / 62) < 0.0001  # rank 1
        assert abs(scores["c"] - 1.0 / 63) < 0.0001  # rank 2


# ---------------------------------------------------------------------------
# Tests: Temporal search
# ---------------------------------------------------------------------------

class TestTemporalSearch:
    def test_closer_dates_score_higher(self):
        now = datetime(2026, 3, 22, tzinfo=timezone.utc)
        facts = [
            Fact(id="recent", occurred_start="2026-03-21T00:00:00+00:00"),
            Fact(id="old", occurred_start="2026-01-01T00:00:00+00:00"),
        ]
        results = _temporal_search(facts, now, limit=10)
        assert len(results) == 2
        recent_score = next(s for fid, s in results if fid == "recent")
        old_score = next(s for fid, s in results if fid == "old")
        assert recent_score > old_score

    def test_same_day_score_near_one(self):
        now = datetime(2026, 3, 22, tzinfo=timezone.utc)
        facts = [Fact(id="today", occurred_start="2026-03-22T00:00:00+00:00")]
        results = _temporal_search(facts, now, limit=10)
        assert len(results) == 1
        assert results[0][1] > 0.95  # Very close to 1.0

    def test_no_dated_facts(self):
        now = datetime(2026, 3, 22, tzinfo=timezone.utc)
        facts = [Fact(id="undated")]
        results = _temporal_search(facts, now, limit=10)
        assert len(results) == 0

    def test_respects_limit(self):
        now = datetime(2026, 3, 22, tzinfo=timezone.utc)
        facts = [
            Fact(id=f"f{i}", occurred_start=f"2026-03-{22-i:02d}T00:00:00+00:00")
            for i in range(10)
        ]
        results = _temporal_search(facts, now, limit=3)
        assert len(results) == 3


# ---------------------------------------------------------------------------
# Tests: Token budget
# ---------------------------------------------------------------------------

class TestTokenBudget:
    def test_filters_by_budget(self):
        facts = [Fact(text="x" * 400) for _ in range(10)]  # 400 chars = ~100 tokens each
        result = _apply_token_budget(facts, max_tokens=250)
        assert len(result) < 10
        assert len(result) >= 2  # At least 2 should fit

    def test_empty_facts(self):
        result = _apply_token_budget([], max_tokens=1000)
        assert result == []

    def test_single_fact_within_budget(self):
        facts = [Fact(text="short")]
        result = _apply_token_budget(facts, max_tokens=100)
        assert len(result) == 1

    def test_preserves_order(self):
        facts = [Fact(id=f"f{i}", text=f"fact {i}") for i in range(5)]
        result = _apply_token_budget(facts, max_tokens=10000)
        assert [f.id for f in result] == [f"f{i}" for i in range(5)]


# ---------------------------------------------------------------------------
# Tests: Full recall pipeline
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestRecall:
    async def test_basic_recall(self):
        facts = _make_facts(5)
        store = MockStore(facts=facts)
        embedder = MockEmbedder()

        result = await recall(
            bank_id="test",
            query="test query",
            embedder=embedder,
            store=store,
        )

        assert isinstance(result, RecallResult)
        assert len(result.results) > 0
        assert result.query == "test query"

    async def test_empty_query(self):
        result = await recall(bank_id="test", query="", store=MockStore())
        assert result.results == []

    async def test_no_store(self):
        result = await recall(bank_id="test", query="test")
        assert result.results == []

    async def test_with_reranker(self):
        facts = _make_facts(5)
        store = MockStore(facts=facts)
        embedder = MockEmbedder()
        reranker = MockReranker()

        result = await recall(
            bank_id="test",
            query="test",
            embedder=embedder,
            store=store,
            reranker=reranker,
        )

        # Reranker reverses order, so results should be different from input
        assert len(result.results) > 0

    async def test_fact_type_filter(self):
        facts = [
            Fact(id="world_1", text="world fact", fact_type=FactType.WORLD),
            Fact(id="exp_1", text="experience fact", fact_type=FactType.EXPERIENCE),
        ]
        store = MockStore(facts=facts)
        embedder = MockEmbedder()

        result = await recall(
            bank_id="test",
            query="fact",
            embedder=embedder,
            store=store,
            fact_types=["world"],
        )

        for f in result.results:
            assert f.fact_type == FactType.WORLD

    async def test_entity_search_contributes(self):
        facts = [
            Fact(id="f1", text="Alice lives in London", bank_id="test"),
            Fact(id="f2", text="The weather is nice", bank_id="test"),
        ]
        entities = [
            Entity(text="Alice", entity_type=EntityType.PERSON, bank_id="test", fact_ids=["f1"]),
        ]
        store = MockStore(facts=facts, entities=entities)

        result = await recall(
            bank_id="test",
            query="Alice",
            store=store,
        )

        # "f1" should appear because it's linked to entity "Alice"
        result_ids = [f.id for f in result.results]
        assert "f1" in result_ids

    async def test_budget_levels(self):
        facts = _make_facts(10)
        store = MockStore(facts=facts)
        embedder = MockEmbedder()

        for budget in [Budget.LOW, Budget.MID, Budget.HIGH]:
            result = await recall(
                bank_id="test",
                query="test",
                embedder=embedder,
                store=store,
                budget=budget,
            )
            assert isinstance(result, RecallResult)
