"""
Tests for lucid/types.py — Core data types.

Covers:
1. Fact creation and serialization
2. Entity equality and hashing
3. Bank defaults and disposition
4. Enum values match Hindsight's expected strings
5. Result types have correct defaults
6. Memory hierarchy relationships
7. ID generation uniqueness
"""

import pytest

from lucid.types import (
    Bank,
    Budget,
    Entity,
    EntityType,
    Fact,
    FactKind,
    FactType,
    LinkType,
    MemoryLink,
    MentalModel,
    Observation,
    RecallResult,
    ReflectResult,
    RetainResult,
    _new_id,
    _now_iso,
)


class TestHelpers:
    def test_new_id_is_16_chars(self):
        assert len(_new_id()) == 16

    def test_new_id_is_unique(self):
        ids = {_new_id() for _ in range(100)}
        assert len(ids) == 100

    def test_now_iso_has_timezone(self):
        ts = _now_iso()
        assert "+" in ts or "Z" in ts


class TestFact:
    def test_create_default(self):
        f = Fact()
        assert f.id
        assert f.text == ""
        assert f.fact_type == FactType.WORLD
        assert f.fact_kind == FactKind.CONVERSATION
        assert f.confidence == 1.0
        assert f.relevance == 0.0
        assert not f.has_embedding

    def test_create_with_content(self):
        f = Fact(
            text="Alice lives in London",
            what="Alice lives in London",
            where="London",
            who="Alice",
            fact_type=FactType.WORLD,
            fact_kind=FactKind.CONVERSATION,
        )
        assert f.text == "Alice lives in London"
        assert f.where == "London"
        assert f.who == "Alice"

    def test_has_embedding(self):
        f = Fact(embedding=[0.1, 0.2, 0.3])
        assert f.has_embedding

    def test_to_dict_excludes_embedding(self):
        f = Fact(text="test", embedding=[0.1, 0.2])
        d = f.to_dict()
        assert "embedding" not in d
        assert d["text"] == "test"
        assert d["fact_type"] == "world"

    def test_to_dict_has_all_fields(self):
        f = Fact(
            text="test",
            what="what",
            when="when",
            where="where",
            who="who",
            why="why",
        )
        d = f.to_dict()
        for key in ["id", "text", "what", "when", "where", "who", "why",
                     "fact_type", "fact_kind", "confidence", "created_at"]:
            assert key in d

    def test_event_fact(self):
        f = Fact(
            fact_kind=FactKind.EVENT,
            occurred_start="2026-03-22",
            occurred_end="2026-03-22",
        )
        assert f.fact_kind == FactKind.EVENT
        assert f.occurred_start == "2026-03-22"


class TestEntity:
    def test_create_default(self):
        e = Entity(text="Alice", entity_type=EntityType.PERSON)
        assert e.text == "Alice"
        assert e.entity_type == EntityType.PERSON

    def test_equality_case_insensitive(self):
        e1 = Entity(text="Alice", entity_type=EntityType.PERSON, bank_id="b1")
        e2 = Entity(text="alice", entity_type=EntityType.PERSON, bank_id="b1")
        assert e1 == e2

    def test_inequality_different_type(self):
        e1 = Entity(text="Apple", entity_type=EntityType.ORGANIZATION, bank_id="b1")
        e2 = Entity(text="Apple", entity_type=EntityType.LOCATION, bank_id="b1")
        assert e1 != e2

    def test_hashable(self):
        e1 = Entity(text="Alice", entity_type=EntityType.PERSON, bank_id="b1")
        e2 = Entity(text="alice", entity_type=EntityType.PERSON, bank_id="b1")
        assert hash(e1) == hash(e2)
        assert len({e1, e2}) == 1  # Dedup in set

    def test_different_bank_not_equal(self):
        e1 = Entity(text="Alice", entity_type=EntityType.PERSON, bank_id="b1")
        e2 = Entity(text="Alice", entity_type=EntityType.PERSON, bank_id="b2")
        assert e1 != e2


class TestMemoryLink:
    def test_create(self):
        link = MemoryLink(source_id="a", target_id="b", link_type=LinkType.SEMANTIC, strength=0.8)
        assert link.source_id == "a"
        assert link.link_type == LinkType.SEMANTIC
        assert link.strength == 0.8


class TestObservation:
    def test_create(self):
        obs = Observation(
            text="Alice prefers direct communication",
            entity_id="ent_1",
            source_fact_ids=["f1", "f2", "f3"],
            confidence=0.9,
        )
        assert obs.text == "Alice prefers direct communication"
        assert len(obs.source_fact_ids) == 3
        assert obs.entity_id == "ent_1"


class TestMentalModel:
    def test_create(self):
        mm = MentalModel(
            text="Alice is a high-intensity founder",
            observation_ids=["obs_1", "obs_2"],
        )
        assert mm.text == "Alice is a high-intensity founder"
        assert len(mm.observation_ids) == 2


class TestBank:
    def test_default_disposition(self):
        bank = Bank(name="test")
        assert bank.disposition["skepticism"] == 3
        assert bank.disposition["literalism"] == 3
        assert bank.disposition["empathy"] == 3

    def test_custom_disposition(self):
        bank = Bank(
            name="critical",
            disposition={"skepticism": 5, "literalism": 1, "empathy": 2},
        )
        assert bank.disposition["skepticism"] == 5


class TestEnums:
    """Verify enum values match Hindsight's expected strings."""

    def test_fact_type_values(self):
        assert FactType.WORLD.value == "world"
        assert FactType.EXPERIENCE.value == "experience"
        assert FactType.ASSISTANT.value == "assistant"

    def test_fact_kind_values(self):
        assert FactKind.EVENT.value == "event"
        assert FactKind.CONVERSATION.value == "conversation"

    def test_entity_type_values(self):
        assert EntityType.PERSON.value == "person"
        assert EntityType.ORGANIZATION.value == "organization"
        assert EntityType.LOCATION.value == "location"

    def test_link_type_values(self):
        assert LinkType.SEMANTIC.value == "semantic"
        assert LinkType.TEMPORAL.value == "temporal"
        assert LinkType.ENTITY.value == "entity"
        assert LinkType.CAUSAL.value == "causal"

    def test_budget_values(self):
        assert Budget.LOW.value == "low"
        assert Budget.MID.value == "mid"
        assert Budget.HIGH.value == "high"


class TestRetainResult:
    def test_defaults(self):
        r = RetainResult()
        assert r.success is True
        assert r.fact_ids == []
        assert r.facts_count == 0
        assert r.token_usage["total_tokens"] == 0


class TestRecallResult:
    def test_defaults(self):
        r = RecallResult()
        assert r.results == []
        assert r.entities == {}
        assert r.query == ""

    def test_with_facts(self):
        facts = [Fact(text="fact 1"), Fact(text="fact 2")]
        r = RecallResult(results=facts, query="test query", total_candidates=50)
        assert len(r.results) == 2
        assert r.total_candidates == 50


class TestReflectResult:
    def test_defaults(self):
        r = ReflectResult()
        assert r.text == ""
        assert r.based_on == {}
        assert r.tool_calls == 0

    def test_with_content(self):
        r = ReflectResult(
            text="Alice is a founder...",
            based_on={"f1": "fact text 1", "f2": "fact text 2"},
            tool_calls=3,
        )
        assert len(r.based_on) == 2
        assert r.tool_calls == 3
