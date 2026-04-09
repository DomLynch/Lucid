"""
Tests for lucid/store.py — SQLite memory store with vector search.

Covers:
1. Bank CRUD
2. Fact persistence and retrieval
3. Embedding-based vector search (cosine similarity)
4. Text search (substring)
5. Entity persistence and lookup
6. Link persistence and graph traversal
7. Observation persistence and search
8. Full retain → store → recall round-trip
"""

import json

import numpy as np
import pytest

from lucid.types import (
    Bank,
    Entity,
    EntityType,
    Fact,
    FactType,
    LinkType,
    MemoryLink,
    Observation,
)


@pytest.fixture
def store(tmp_path):
    from lucid.store import SQLiteMemoryStore
    s = SQLiteMemoryStore(db_path=tmp_path / "test.db")
    yield s
    s.close()


@pytest.mark.asyncio
class TestBanks:
    async def test_create_and_get(self, store):
        bank = Bank(id="b1", name="Test Bank", mission="Remember everything")
        await store.create_bank(bank)

        loaded = await store.get_bank("b1")
        assert loaded.name == "Test Bank"
        assert loaded.mission == "Remember everything"

    async def test_get_nonexistent_raises(self, store):
        with pytest.raises(KeyError):
            await store.get_bank("nonexistent")

    async def test_disposition_round_trip(self, store):
        bank = Bank(id="b1", disposition={"skepticism": 5, "empathy": 1})
        await store.create_bank(bank)

        loaded = await store.get_bank("b1")
        assert loaded.disposition["skepticism"] == 5
        assert loaded.disposition["empathy"] == 1


@pytest.mark.asyncio
class TestFacts:
    async def test_save_and_get(self, store):
        facts = [
            Fact(id="f1", bank_id="b1", text="Fact one"),
            Fact(id="f2", bank_id="b1", text="Fact two"),
        ]
        await store.save_facts(facts)

        loaded = await store.get_facts("b1")
        assert len(loaded) == 2
        texts = {f.text for f in loaded}
        assert "Fact one" in texts
        assert "Fact two" in texts

    async def test_get_by_ids(self, store):
        facts = [
            Fact(id="f1", bank_id="b1", text="First"),
            Fact(id="f2", bank_id="b1", text="Second"),
            Fact(id="f3", bank_id="b1", text="Third"),
        ]
        await store.save_facts(facts)

        loaded = await store.get_facts("b1", fact_ids=["f1", "f3"])
        assert len(loaded) == 2
        ids = {f.id for f in loaded}
        assert "f1" in ids
        assert "f3" in ids

    async def test_embedding_round_trip(self, store):
        emb = [0.1, 0.2, 0.3, 0.4]
        fact = Fact(id="f1", bank_id="b1", text="test", embedding=emb)
        await store.save_facts([fact])

        loaded = await store.get_facts("b1")
        assert loaded[0].embedding == emb
        assert loaded[0].has_embedding

    async def test_all_fields_round_trip(self, store):
        fact = Fact(
            id="f1", bank_id="b1", text="Full fact",
            what="what", when="when", where="where", who="who", why="why",
            fact_type=FactType.EXPERIENCE, fact_kind=Fact.__dataclass_fields__["fact_kind"].default,
            event_date="2026-03-22", occurred_start="2026-03-22T00:00:00",
            confidence=0.9, source_context="test context",
        )
        await store.save_facts([fact])

        loaded = (await store.get_facts("b1"))[0]
        assert loaded.what == "what"
        assert loaded.who == "who"
        assert loaded.fact_type == FactType.EXPERIENCE
        assert loaded.confidence == 0.9
        assert loaded.source_context == "test context"


@pytest.mark.asyncio
class TestVectorSearch:
    async def test_cosine_similarity_ranking(self, store):
        """Facts with closer embeddings should rank higher."""
        query_emb = [1.0, 0.0, 0.0, 0.0]

        facts = [
            Fact(id="close", bank_id="b1", text="Close match", embedding=[0.9, 0.1, 0.0, 0.0]),
            Fact(id="far", bank_id="b1", text="Far match", embedding=[0.0, 0.0, 0.0, 1.0]),
            Fact(id="mid", bank_id="b1", text="Mid match", embedding=[0.5, 0.5, 0.0, 0.0]),
        ]
        await store.save_facts(facts)

        results = await store.search_facts_by_embedding("b1", query_emb, limit=3)
        assert len(results) == 3
        assert results[0].id == "close"
        assert results[-1].id == "far"
        assert results[0].relevance > results[-1].relevance

    async def test_empty_store(self, store):
        results = await store.search_facts_by_embedding("b1", [1.0, 0.0], limit=10)
        assert results == []

    async def test_respects_limit(self, store):
        facts = [
            Fact(id=f"f{i}", bank_id="b1", text=f"fact {i}", embedding=[float(i)] * 4)
            for i in range(10)
        ]
        await store.save_facts(facts)

        results = await store.search_facts_by_embedding("b1", [5.0] * 4, limit=3)
        assert len(results) == 3


@pytest.mark.asyncio
class TestTextSearch:
    async def test_substring_match(self, store):
        facts = [
            Fact(id="f1", bank_id="b1", text="Alice lives in London"),
            Fact(id="f2", bank_id="b1", text="The weather is nice"),
        ]
        await store.save_facts(facts)

        results = await store.search_facts_by_text("b1", "London")
        assert len(results) == 1
        assert results[0].id == "f1"

    async def test_case_insensitive(self, store):
        facts = [Fact(id="f1", bank_id="b1", text="ALICE")]
        await store.save_facts(facts)

        results = await store.search_facts_by_text("b1", "alice")
        assert len(results) == 1

    async def test_no_match(self, store):
        facts = [Fact(id="f1", bank_id="b1", text="test")]
        await store.save_facts(facts)

        results = await store.search_facts_by_text("b1", "nonexistent")
        assert len(results) == 0


@pytest.mark.asyncio
class TestEntities:
    async def test_save_and_get(self, store):
        entities = [
            Entity(id="e1", bank_id="b1", text="Alice", entity_type=EntityType.PERSON, fact_ids=["f1"]),
            Entity(id="e2", bank_id="b1", text="London", entity_type=EntityType.LOCATION, fact_ids=["f1"]),
        ]
        await store.save_entities(entities)

        loaded = await store.get_entities("b1")
        assert len(loaded) == 2

    async def test_get_by_text(self, store):
        await store.save_entities([
            Entity(id="e1", bank_id="b1", text="Alice", entity_type=EntityType.PERSON),
        ])

        found = await store.get_entity_by_text("b1", "Alice")
        assert found is not None
        assert found.id == "e1"

    async def test_get_by_text_case_insensitive(self, store):
        await store.save_entities([
            Entity(id="e1", bank_id="b1", text="Alice", entity_type=EntityType.PERSON),
        ])

        found = await store.get_entity_by_text("b1", "alice")
        assert found is not None

    async def test_fact_ids_round_trip(self, store):
        await store.save_entities([
            Entity(id="e1", bank_id="b1", text="test", fact_ids=["f1", "f2", "f3"]),
        ])

        loaded = (await store.get_entities("b1"))[0]
        assert loaded.fact_ids == ["f1", "f2", "f3"]


@pytest.mark.asyncio
class TestLinks:
    async def test_save_and_traverse(self, store):
        facts = [
            Fact(id="f1", bank_id="b1", text="Fact 1"),
            Fact(id="f2", bank_id="b1", text="Fact 2"),
        ]
        await store.save_facts(facts)

        links = [MemoryLink(source_id="f1", target_id="f2", link_type=LinkType.ENTITY, strength=0.8)]
        await store.save_links(links)

        linked = await store.get_linked_facts("b1", "f1")
        assert len(linked) == 1
        assert linked[0].id == "f2"

    async def test_traverse_by_type(self, store):
        facts = [
            Fact(id="f1", bank_id="b1", text="Fact 1"),
            Fact(id="f2", bank_id="b1", text="Fact 2"),
            Fact(id="f3", bank_id="b1", text="Fact 3"),
        ]
        await store.save_facts(facts)

        links = [
            MemoryLink(source_id="f1", target_id="f2", link_type=LinkType.ENTITY),
            MemoryLink(source_id="f1", target_id="f3", link_type=LinkType.TEMPORAL),
        ]
        await store.save_links(links)

        entity_linked = await store.get_linked_facts("b1", "f1", link_type="entity")
        assert len(entity_linked) == 1
        assert entity_linked[0].id == "f2"


@pytest.mark.asyncio
class TestObservations:
    async def test_save_and_get(self, store):
        obs = [
            Observation(id="o1", bank_id="b1", text="Alice prefers direct communication"),
            Observation(id="o2", bank_id="b1", text="Nexus uses a modular architecture"),
        ]
        await store.save_observations(obs)

        loaded = await store.get_observations("b1")
        assert len(loaded) == 2

    async def test_embedding_search(self, store):
        obs = [
            Observation(id="o1", bank_id="b1", text="close", embedding=[0.9, 0.1, 0.0, 0.0]),
            Observation(id="o2", bank_id="b1", text="far", embedding=[0.0, 0.0, 0.0, 1.0]),
        ]
        await store.save_observations(obs)

        results = await store.search_observations_by_embedding("b1", [1.0, 0.0, 0.0, 0.0], limit=2)
        assert len(results) == 2
        assert results[0].id == "o1"
