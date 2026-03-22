"""
Tests for lucid/retain.py — Memory retention with fact extraction.

Covers:
1. Fact extraction from LLM response
2. JSON response parsing (valid, malformed, nested)
3. Temporal inference from relative dates
4. Entity extraction and deduplication
5. Entity link creation
6. Fact text building from 5 dimensions
7. Fallback when no LLM provided
8. Empty input handling
"""

import json
from datetime import datetime, timezone
from typing import Any

import pytest

from lucid.retain import (
    _build_fact_text,
    _classify_entity_type,
    _create_entity_links,
    _dedup_entities,
    _infer_temporal_date,
    _parse_extraction_response,
    retain,
)
from lucid.types import Entity, EntityType, Fact, FactType, LinkType


# ---------------------------------------------------------------------------
# Mock LLM client
# ---------------------------------------------------------------------------

class MockLLM:
    def __init__(self, response: dict[str, Any] | None = None):
        self._response = response or {
            "content": json.dumps({"facts": [
                {
                    "what": "Dominic lives in Dubai",
                    "when": "N/A",
                    "where": "Dubai",
                    "who": "Dominic",
                    "why": "N/A",
                    "fact_type": "world",
                    "fact_kind": "conversation",
                    "entities": [{"text": "Dominic"}, {"text": "Dubai"}],
                }
            ]}),
            "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        }
        self.calls: list[dict] = []

    async def complete(self, messages, temperature=0.0, max_tokens=4096, response_format=None):
        self.calls.append({"messages": messages, "temperature": temperature})
        return self._response


class MockEmbedder:
    def __init__(self, dim: int = 4):
        self._dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1 * (i + 1)] * self._dim for i in range(len(texts))]

    @property
    def dimension(self) -> int:
        return self._dim


class MockStore:
    def __init__(self):
        self.facts: list[Fact] = []
        self.entities: list[Entity] = []
        self.links: list = []

    async def save_facts(self, facts):
        self.facts.extend(facts)

    async def save_entities(self, entities):
        self.entities.extend(entities)

    async def save_links(self, links):
        self.links.extend(links)


# ---------------------------------------------------------------------------
# Tests: JSON parsing
# ---------------------------------------------------------------------------

class TestParseExtractionResponse:
    def test_valid_json_with_facts_key(self):
        content = json.dumps({"facts": [{"what": "test", "fact_type": "world"}]})
        result = _parse_extraction_response(content)
        assert len(result) == 1
        assert result[0]["what"] == "test"

    def test_valid_json_array(self):
        content = json.dumps([{"what": "test1"}, {"what": "test2"}])
        result = _parse_extraction_response(content)
        assert len(result) == 2

    def test_empty_string(self):
        assert _parse_extraction_response("") == []

    def test_malformed_json(self):
        assert _parse_extraction_response("not json at all") == []

    def test_json_embedded_in_text(self):
        content = 'Here are the facts: {"facts": [{"what": "found it"}]}'
        result = _parse_extraction_response(content)
        assert len(result) == 1
        assert result[0]["what"] == "found it"

    def test_empty_facts_array(self):
        content = json.dumps({"facts": []})
        result = _parse_extraction_response(content)
        assert result == []


# ---------------------------------------------------------------------------
# Tests: Temporal inference
# ---------------------------------------------------------------------------

class TestTemporalInference:
    def test_yesterday(self):
        event_date = datetime(2026, 3, 22, tzinfo=timezone.utc)
        result = _infer_temporal_date("I saw them yesterday", event_date)
        assert result is not None
        assert "2026-03-21" in result

    def test_today(self):
        event_date = datetime(2026, 3, 22, tzinfo=timezone.utc)
        result = _infer_temporal_date("I went today", event_date)
        assert result is not None
        assert "2026-03-22" in result

    def test_tomorrow(self):
        event_date = datetime(2026, 3, 22, tzinfo=timezone.utc)
        result = _infer_temporal_date("Meeting tomorrow", event_date)
        assert result is not None
        assert "2026-03-23" in result

    def test_no_temporal_expression(self):
        event_date = datetime(2026, 3, 22, tzinfo=timezone.utc)
        result = _infer_temporal_date("Dominic likes coffee", event_date)
        assert result is None

    def test_no_event_date(self):
        result = _infer_temporal_date("yesterday was good", None)
        assert result is None

    def test_last_week(self):
        event_date = datetime(2026, 3, 22, tzinfo=timezone.utc)
        result = _infer_temporal_date("last week we met", event_date)
        assert result is not None
        assert "2026-03-15" in result


# ---------------------------------------------------------------------------
# Tests: Entity classification
# ---------------------------------------------------------------------------

class TestEntityClassification:
    def test_user_is_person(self):
        assert _classify_entity_type("user") == EntityType.PERSON

    def test_company_suffix(self):
        assert _classify_entity_type("Apple Inc") == EntityType.ORGANIZATION

    def test_unknown(self):
        assert _classify_entity_type("Brain") == EntityType.OTHER


# ---------------------------------------------------------------------------
# Tests: Fact text building
# ---------------------------------------------------------------------------

class TestBuildFactText:
    def test_all_dimensions(self):
        text = _build_fact_text({
            "what": "Dominic lives in Dubai",
            "who": "Dominic",
            "where": "Dubai",
            "when": "2024",
            "why": "Work opportunities",
        })
        assert "Dominic lives in Dubai" in text
        assert "Involving: Dominic" in text
        assert "Location: Dubai" in text
        assert "When: 2024" in text
        assert "Context: Work opportunities" in text

    def test_na_fields_excluded(self):
        text = _build_fact_text({
            "what": "Simple fact",
            "who": "N/A",
            "where": "N/A",
            "when": "N/A",
            "why": "N/A",
        })
        assert text == "Simple fact"
        assert "Involving" not in text
        assert "Location" not in text


# ---------------------------------------------------------------------------
# Tests: Entity deduplication
# ---------------------------------------------------------------------------

class TestEntityDedup:
    def test_dedup_same_entity(self):
        e1 = Entity(text="Dominic", entity_type=EntityType.PERSON, bank_id="b1", fact_ids=["f1"])
        e2 = Entity(text="dominic", entity_type=EntityType.PERSON, bank_id="b1", fact_ids=["f2"])
        result = _dedup_entities([e1, e2])
        assert len(result) == 1
        assert "f1" in result[0].fact_ids
        assert "f2" in result[0].fact_ids

    def test_different_entities_kept(self):
        e1 = Entity(text="Dominic", entity_type=EntityType.PERSON, bank_id="b1")
        e2 = Entity(text="Dubai", entity_type=EntityType.LOCATION, bank_id="b1")
        result = _dedup_entities([e1, e2])
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Tests: Entity link creation
# ---------------------------------------------------------------------------

class TestEntityLinks:
    def test_creates_links_for_shared_entity(self):
        facts = [Fact(id="f1"), Fact(id="f2"), Fact(id="f3")]
        entities = [Entity(text="Dominic", fact_ids=["f1", "f2", "f3"])]
        links = _create_entity_links(facts, entities)
        # 3 facts sharing 1 entity = 3 links (f1-f2, f1-f3, f2-f3)
        assert len(links) == 3
        assert all(link.link_type == LinkType.ENTITY for link in links)

    def test_no_links_for_single_fact(self):
        entities = [Entity(text="Solo", fact_ids=["f1"])]
        links = _create_entity_links([], entities)
        assert len(links) == 0


# ---------------------------------------------------------------------------
# Tests: Full retain pipeline
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestRetain:
    async def test_basic_retain(self):
        llm = MockLLM()
        embedder = MockEmbedder()
        store = MockStore()

        result = await retain(
            bank_id="test-bank",
            content="Dominic lives in Dubai",
            llm=llm,
            embedder=embedder,
            store=store,
        )

        assert result.success
        assert result.facts_count == 1
        assert len(result.fact_ids) == 1
        assert len(store.facts) == 1
        assert store.facts[0].text
        assert store.facts[0].has_embedding

    async def test_retain_without_llm(self):
        """Without LLM, content is stored as a single fact."""
        store = MockStore()

        result = await retain(
            bank_id="test-bank",
            content="Simple text to remember",
            store=store,
        )

        assert result.success
        assert result.facts_count == 1
        assert store.facts[0].what == "Simple text to remember"

    async def test_retain_empty_content(self):
        result = await retain(bank_id="test", content="")
        assert not result.success
        assert result.facts_count == 0

    async def test_retain_with_entities(self):
        llm = MockLLM()
        store = MockStore()

        result = await retain(
            bank_id="test-bank",
            content="test",
            llm=llm,
            store=store,
        )

        assert len(store.entities) >= 1  # "Dominic" and "Dubai" from mock

    async def test_retain_token_usage_tracked(self):
        llm = MockLLM()

        result = await retain(
            bank_id="test-bank",
            content="test",
            llm=llm,
        )

        assert result.token_usage["total_tokens"] == 150

    async def test_retain_with_context(self):
        llm = MockLLM()

        await retain(
            bank_id="test-bank",
            content="test content",
            context="Personal profile",
            llm=llm,
        )

        # Check context was passed to LLM
        user_msg = llm.calls[0]["messages"][1]["content"]
        assert "Personal profile" in user_msg

    async def test_retain_with_event_date(self):
        llm = MockLLM()
        event_date = datetime(2026, 3, 22, tzinfo=timezone.utc)

        await retain(
            bank_id="test-bank",
            content="test",
            event_date=event_date,
            llm=llm,
        )

        user_msg = llm.calls[0]["messages"][1]["content"]
        assert "March 22, 2026" in user_msg
