"""
lucid/reflect.py — Agentic synthesis via multi-turn reasoning.

The reflect pipeline:
1. Load bank profile (mission, disposition)
2. Build system prompt with personality traits
3. Enter agentic loop with tools:
   - recall: search facts by query
   - search_observations: find consolidated insights
   - done: return final synthesis
4. LLM reasons, calls tools, builds understanding
5. On done() or max iterations, return synthesis

Transplanted from Hindsight's reflect/agent.py (1,161 LOC) +
reflect/prompts.py (~400 LOC). Target: ~250 LOC.
"""

import json
import logging
from typing import Any

from lucid.interfaces import Embedder, LLMClient, MemoryStore, Reranker
from lucid.recall import recall
from lucid.types import Budget, Observation, ReflectResult

_log = logging.getLogger("lucid.reflect")

# Max iterations before forced stop
DEFAULT_MAX_ITERATIONS = 8

# System prompt for the reflect agent
REFLECT_SYSTEM_PROMPT = """You are a memory synthesis agent. Your job is to answer questions by searching through stored memories and observations.

You have access to these tools:
- recall(query): Search stored facts and memories. Returns relevant facts.
- search_observations(query): Search consolidated observations and insights.
- done(answer): Return your final answer. You MUST call this when you're ready.

Process:
1. Think about what information you need
2. Use recall() and search_observations() to gather evidence
3. Synthesize your findings
4. Call done() with your final answer

Rules:
- Base your answer ONLY on retrieved facts and observations
- If you can't find relevant information, say so honestly
- Cite specific facts when making claims
- Be comprehensive but concise"""


REFLECT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "Search stored facts and memories by query",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_observations",
            "description": "Search consolidated observations and insights",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": "Return your final synthesized answer",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "description": "Your comprehensive answer"},
                },
                "required": ["answer"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Main reflect function
# ---------------------------------------------------------------------------

async def reflect(
    bank_id: str,
    query: str,
    llm: LLMClient | None = None,
    embedder: Embedder | None = None,
    store: MemoryStore | None = None,
    reranker: Reranker | None = None,
    budget: Budget = Budget.MID,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> ReflectResult:
    """Synthesize knowledge via multi-turn agentic reasoning.

    Args:
        bank_id: Memory bank ID.
        query: What to reflect on.
        llm: LLM client for agent reasoning.
        embedder: For recall queries.
        store: Memory store.
        reranker: For recall reranking.
        budget: Recall search breadth.
        max_iterations: Max agent turns before forced stop.

    Returns:
        ReflectResult with synthesized answer and citations.
    """
    if not query.strip() or llm is None:
        return ReflectResult()

    result = ReflectResult()
    based_on: dict[str, str] = {}

    # Build system prompt with bank context
    system_prompt = _build_system_prompt(bank_id, store)

    # Initialize conversation
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    for iteration in range(max_iterations):
        is_last = iteration == max_iterations - 1

        # On last iteration, remove tools to force text response
        tools = [] if is_last else REFLECT_TOOLS

        _log.debug("Reflect iteration %d/%d", iteration + 1, max_iterations)

        # Call LLM with tools (empty list on last iteration forces text response)
        response = await llm.complete(
            messages=messages,
            temperature=0.3,
            max_tokens=2048,
            tools=tools if tools else None,
        )

        content = response.get("content", "")
        usage = response.get("usage", {})
        total_usage["input_tokens"] += usage.get("input_tokens", 0)
        total_usage["output_tokens"] += usage.get("output_tokens", 0)
        total_usage["total_tokens"] += usage.get("total_tokens", 0)
        result.tool_calls += 1

        # Check for tool calls in response
        tool_calls = response.get("tool_calls", [])

        if not tool_calls:
            # No tool calls — LLM returned text (final answer or last iteration)
            result.text = content
            break

        # Process tool call
        tc = tool_calls[0] if isinstance(tool_calls, list) else tool_calls
        tool_name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
        tool_args = tc.get("arguments", {}) if isinstance(tc, dict) else getattr(tc, "arguments", {})
        tool_id = tc.get("id", f"call_{iteration}") if isinstance(tc, dict) else getattr(tc, "id", f"call_{iteration}")

        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except json.JSONDecodeError:
                tool_args = {"query": tool_args}

        # Add assistant message with tool call
        messages.append({
            "role": "assistant",
            "content": content or "",
            "tool_calls": [{
                "id": tool_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(tool_args),
                },
            }],
        })

        # Execute tool
        tool_result = await _execute_tool(
            tool_name, tool_args, bank_id,
            embedder=embedder, store=store, reranker=reranker,
            budget=budget, based_on=based_on,
        )

        # Check if done
        if tool_name == "done":
            result.text = tool_args.get("answer", tool_result)
            # Add tool response to close the conversation cleanly
            messages.append({
                "role": "tool",
                "content": tool_result,
                "tool_call_id": tool_id,
                "name": tool_name,
            })
            break

        # Add tool response
        messages.append({
            "role": "tool",
            "content": tool_result,
            "tool_call_id": tool_id,
            "name": tool_name,
        })

    result.based_on = based_on
    result.token_usage = total_usage

    _log.info(
        "Reflect for bank %s: query='%s' iterations=%d facts_cited=%d",
        bank_id, query[:50], result.tool_calls, len(based_on),
    )

    return result


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

def _build_system_prompt(bank_id: str, store: MemoryStore | None) -> str:
    """Build system prompt with bank context."""
    # For now, use the base prompt. Bank-specific context can be added
    # when the store provides bank profile data.
    return REFLECT_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

async def _execute_tool(
    tool_name: str,
    tool_args: dict[str, Any],
    bank_id: str,
    embedder: Embedder | None = None,
    store: MemoryStore | None = None,
    reranker: Reranker | None = None,
    budget: Budget = Budget.MID,
    based_on: dict[str, str] | None = None,
) -> str:
    """Execute a reflect agent tool and return result as string."""
    if tool_name == "recall":
        query = tool_args.get("query", "")
        if not query or store is None:
            return "No results found."

        recall_result = await recall(
            bank_id=bank_id,
            query=query,
            embedder=embedder,
            store=store,
            reranker=reranker,
            budget=budget,
        )

        if not recall_result.results:
            return "No facts found matching this query."

        # Format results and track citations
        lines = []
        for fact in recall_result.results[:15]:  # Limit to top 15
            lines.append(f"- {fact.text}")
            if based_on is not None:
                based_on[fact.id] = fact.text

        return "\n".join(lines)

    if tool_name == "search_observations":
        query = tool_args.get("query", "")
        if not query or store is None:
            return "No observations found."

        try:
            observations = await store.get_observations(bank_id, limit=10)
            if not observations:
                return "No observations available yet."

            lines = []
            for obs in observations:
                if query.lower() in obs.text.lower():
                    lines.append(f"- {obs.text}")

            return "\n".join(lines) if lines else f"No observations matching '{query}'."
        except Exception:
            return "Observation search not available."

    if tool_name == "done":
        return tool_args.get("answer", "No answer provided.")

    return f"Unknown tool: {tool_name}"
