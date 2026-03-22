"""
Tests for lucid/reflect.py — Agentic synthesis loop.

Covers:
1. Basic reflect with done() tool
2. Multi-turn reasoning (recall then done)
3. Max iterations forced stop
4. Citation tracking (based_on)
5. Token usage accumulation
6. Empty query handling
7. No LLM handling
8. Observation search
9. Tool execution dispatch
"""

import json
from typing import Any

import pytest

from lucid.reflect import _execute_tool, reflect
from lucid.types import Budget, Fact, Observation, ReflectResult


# ---------------------------------------------------------------------------
# Mock implementations
# ---------------------------------------------------------------------------

class MockReflectLLM:
    """Mock LLM that follows a script of responses."""

    def __init__(self, script: list[dict[str, Any]]):
        self._script = list(script)
        self._call_count = 0

    async def complete(self, messages, temperature=0.0, max_tokens=4096, response_format=None):
        if self._call_count < len(self._script):
            response = self._script[self._call_count]
        else:
            # Default: return done
            response = {
                "content": "",
                "tool_calls": [{"id": "call_default", "name": "done",
                               "arguments": {"answer": "Default answer"}}],
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            }
        self._call_count += 1
        return response


class MockReflectStore:
    def __init__(self, facts=None, observations=None):
        self._facts = facts or []
        self._observations = observations or []

    async def search_facts_by_embedding(self, bank_id, query_emb, limit=50):
        return self._facts[:limit]

    async def search_facts_by_text(self, bank_id, query, limit=50):
        return [f for f in self._facts if query.lower() in f.text.lower()][:limit]

    async def get_facts(self, bank_id, fact_ids=None):
        if fact_ids:
            return [f for f in self._facts if f.id in fact_ids]
        return self._facts

    async def get_entities(self, bank_id):
        return []

    async def get_observations(self, bank_id, limit=20):
        return self._observations[:limit]

    async def save_facts(self, facts): pass
    async def save_entities(self, entities): pass
    async def save_links(self, links): pass


class MockReflectEmbedder:
    async def embed(self, texts):
        return [[0.1] * 4 for _ in texts]

    @property
    def dimension(self):
        return 4


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestReflect:
    async def test_basic_done(self):
        """LLM immediately calls done() with an answer."""
        llm = MockReflectLLM([
            {
                "content": "",
                "tool_calls": [{"id": "call_1", "name": "done",
                               "arguments": {"answer": "Dominic is a founder in Dubai."}}],
                "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
            },
        ])

        result = await reflect(
            bank_id="test",
            query="Who is Dominic?",
            llm=llm,
        )

        assert result.text == "Dominic is a founder in Dubai."
        assert result.tool_calls >= 1

    async def test_recall_then_done(self):
        """LLM first recalls facts, then synthesizes with done()."""
        facts = [
            Fact(id="f1", text="Dominic lives in Dubai", bank_id="test"),
            Fact(id="f2", text="Dominic is building Brain", bank_id="test"),
        ]

        llm = MockReflectLLM([
            # Step 1: recall
            {
                "content": "Let me search for information about Dominic.",
                "tool_calls": [{"id": "call_1", "name": "recall",
                               "arguments": {"query": "Dominic"}}],
                "usage": {"input_tokens": 100, "output_tokens": 30, "total_tokens": 130},
            },
            # Step 2: done
            {
                "content": "",
                "tool_calls": [{"id": "call_2", "name": "done",
                               "arguments": {"answer": "Based on the facts, Dominic lives in Dubai and is building Brain."}}],
                "usage": {"input_tokens": 200, "output_tokens": 50, "total_tokens": 250},
            },
        ])

        store = MockReflectStore(facts=facts)

        result = await reflect(
            bank_id="test",
            query="Tell me about Dominic",
            llm=llm,
            store=store,
        )

        assert "Dominic" in result.text
        assert result.tool_calls == 2

    async def test_citations_tracked(self):
        """Facts retrieved via recall() are tracked in based_on."""
        facts = [
            Fact(id="f1", text="Dominic lives in Dubai", bank_id="test"),
        ]

        llm = MockReflectLLM([
            {
                "content": "",
                "tool_calls": [{"id": "call_1", "name": "recall",
                               "arguments": {"query": "Dominic"}}],
                "usage": {"input_tokens": 50, "output_tokens": 20, "total_tokens": 70},
            },
            {
                "content": "",
                "tool_calls": [{"id": "call_2", "name": "done",
                               "arguments": {"answer": "He lives in Dubai."}}],
                "usage": {"input_tokens": 50, "output_tokens": 20, "total_tokens": 70},
            },
        ])

        store = MockReflectStore(facts=facts)

        result = await reflect(
            bank_id="test",
            query="Where does Dominic live?",
            llm=llm,
            store=store,
        )

        assert "f1" in result.based_on
        assert "Dubai" in result.based_on["f1"]

    async def test_max_iterations(self):
        """Agent stops after max iterations even without done()."""
        # LLM keeps calling recall forever
        llm = MockReflectLLM([
            {
                "content": f"Searching more...",
                "tool_calls": [{"id": f"call_{i}", "name": "recall",
                               "arguments": {"query": f"query {i}"}}],
                "usage": {"input_tokens": 50, "output_tokens": 20, "total_tokens": 70},
            }
            for i in range(20)
        ] + [
            # Final text response when tools are removed
            {
                "content": "I could not find enough information.",
                "tool_calls": [],
                "usage": {"input_tokens": 50, "output_tokens": 20, "total_tokens": 70},
            },
        ])

        store = MockReflectStore()

        result = await reflect(
            bank_id="test",
            query="test",
            llm=llm,
            store=store,
            max_iterations=3,
        )

        # Should stop at max_iterations
        assert result.tool_calls <= 3

    async def test_token_usage_accumulated(self):
        llm = MockReflectLLM([
            {
                "content": "",
                "tool_calls": [{"id": "call_1", "name": "done",
                               "arguments": {"answer": "answer"}}],
                "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
            },
        ])

        result = await reflect(bank_id="test", query="test", llm=llm)
        assert result.token_usage["total_tokens"] == 150

    async def test_empty_query(self):
        result = await reflect(bank_id="test", query="")
        assert result.text == ""

    async def test_no_llm(self):
        result = await reflect(bank_id="test", query="test")
        assert result.text == ""

    async def test_text_response_without_tools(self):
        """LLM returns text directly without calling any tools."""
        llm = MockReflectLLM([
            {
                "content": "Here is my synthesis of the available information.",
                "tool_calls": [],
                "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
            },
        ])

        result = await reflect(bank_id="test", query="test", llm=llm)
        assert result.text == "Here is my synthesis of the available information."


@pytest.mark.asyncio
class TestToolExecution:
    async def test_recall_tool(self):
        facts = [Fact(id="f1", text="test fact", bank_id="test")]
        store = MockReflectStore(facts=facts)
        based_on: dict[str, str] = {}

        result = await _execute_tool(
            "recall", {"query": "test"}, "test",
            store=store, based_on=based_on,
        )

        assert "test fact" in result
        assert "f1" in based_on

    async def test_search_observations_tool(self):
        obs = [Observation(text="Dominic prefers direct communication")]
        store = MockReflectStore(observations=obs)

        result = await _execute_tool(
            "search_observations", {"query": "communication"}, "test",
            store=store,
        )

        assert "direct communication" in result

    async def test_done_tool(self):
        result = await _execute_tool(
            "done", {"answer": "Final answer here"}, "test",
        )
        assert result == "Final answer here"

    async def test_unknown_tool(self):
        result = await _execute_tool(
            "nonexistent", {}, "test",
        )
        assert "Unknown tool" in result
