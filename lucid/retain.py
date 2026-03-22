"""
lucid/retain.py — Store memories with LLM-powered fact extraction.

The retain pipeline:
1. Send input text + extraction prompt to LLM
2. Parse structured JSON response into Facts
3. Extract and resolve entities
4. Generate embeddings for each fact
5. Create links (semantic, entity, temporal)
6. Persist everything to MemoryStore

Transplanted from Hindsight's retain/fact_extraction.py (2,231 LOC)
and memory_engine.py retain path (~500 LOC). Target: ~400 LOC.
"""

import json
import logging
import re
from datetime import datetime, timedelta
from typing import Any

from lucid.interfaces import Embedder, LLMClient, MemoryStore
from lucid.types import (
    Entity,
    EntityType,
    Fact,
    FactKind,
    FactType,
    LinkType,
    MemoryLink,
    RetainResult,
)

_log = logging.getLogger("lucid.retain")


# ---------------------------------------------------------------------------
# Extraction prompt (transplanted from Hindsight fact_extraction.py)
# ---------------------------------------------------------------------------

FACT_EXTRACTION_PROMPT = """Extract SIGNIFICANT facts from text. Be SELECTIVE - only extract facts worth remembering long-term.

LANGUAGE: Detect the language of the input text and produce ALL output in that EXACT same language.

FACT FORMAT - BE CONCISE

1. **what**: Core fact - concise but complete (1-2 sentences max)
2. **when**: Temporal info if mentioned. "N/A" if none.
3. **where**: Location if relevant. "N/A" if none.
4. **who**: People involved with relationships. "N/A" if just general info.
5. **why**: Context/significance ONLY if important. "N/A" if obvious.

COREFERENCE RESOLUTION
Link generic references to names when both appear:
- "my roommate" + "Emily" → use "Emily (user's roommate)"
- "the manager" + "Sarah" → use "Sarah (the manager)"

CLASSIFICATION
fact_kind:
- "event": Specific datable occurrence (set occurred_start/end)
- "conversation": Ongoing state, preference, trait (no dates)

fact_type:
- "world": About user's life, other people, external events
- "assistant": Interactions with assistant (requests, recommendations)

TEMPORAL HANDLING
Use "Event Date" from input as reference for relative dates.
- Convert ALL relative temporal expressions to absolute dates.
- For events: set occurred_start AND occurred_end
- For conversation facts: NO occurred dates

ENTITIES
Include: people names, organizations, places, key objects, abstract concepts.
Always include "user" when fact is about the user.

SELECTIVITY
ONLY extract facts that are:
- Personal info: names, relationships, roles, background
- Preferences: likes, dislikes, habits, interests
- Significant events: milestones, decisions, achievements
- Plans/goals: future intentions, deadlines, commitments
- Expertise: skills, knowledge, certifications
- Important context: projects, problems, constraints

DO NOT extract:
- Generic greetings or pleasantries
- Pure filler ("thanks", "sounds good", "ok")
- Process chatter ("let me check", "one moment")
- Repeated information

QUALITY OVER QUANTITY
Ask: "Would this be useful to recall in 6 months?" If no, skip it.
CONSOLIDATE related statements into ONE fact when possible."""


EXTRACTION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "what": {"type": "string"},
                    "when": {"type": "string"},
                    "where": {"type": "string"},
                    "who": {"type": "string"},
                    "why": {"type": "string"},
                    "fact_kind": {"type": "string", "enum": ["event", "conversation"]},
                    "fact_type": {"type": "string", "enum": ["world", "assistant"]},
                    "occurred_start": {"type": "string"},
                    "occurred_end": {"type": "string"},
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                            },
                            "required": ["text"],
                        },
                    },
                },
                "required": ["what", "when", "where", "who", "why", "fact_type"],
            },
        },
    },
    "required": ["facts"],
}


# ---------------------------------------------------------------------------
# Temporal inference (from Hindsight)
# ---------------------------------------------------------------------------

_TEMPORAL_PATTERNS: dict[str, int] = {
    r"\blast night\b": -1,
    r"\byesterday\b": -1,
    r"\btoday\b": 0,
    r"\bthis morning\b": 0,
    r"\bthis afternoon\b": 0,
    r"\bthis evening\b": 0,
    r"\btonigh?t\b": 0,
    r"\btomorrow\b": 1,
    r"\blast week\b": -7,
    r"\bthis week\b": 0,
    r"\bnext week\b": 7,
    r"\blast month\b": -30,
    r"\bthis month\b": 0,
    r"\bnext month\b": 30,
}


def _infer_temporal_date(fact_text: str, event_date: datetime | None) -> str | None:
    """Infer a temporal date from relative expressions in fact text."""
    if event_date is None:
        return None

    fact_lower = fact_text.lower()
    for pattern, offset_days in _TEMPORAL_PATTERNS.items():
        if re.search(pattern, fact_lower):
            target = event_date + timedelta(days=offset_days)
            return target.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()

    return None


# ---------------------------------------------------------------------------
# Entity extraction helpers
# ---------------------------------------------------------------------------

_ENTITY_TYPE_MAP: dict[str, EntityType] = {
    "person": EntityType.PERSON,
    "organization": EntityType.ORGANIZATION,
    "org": EntityType.ORGANIZATION,
    "location": EntityType.LOCATION,
    "place": EntityType.LOCATION,
    "date": EntityType.DATE,
    "concept": EntityType.CONCEPT,
}


def _classify_entity_type(text: str) -> EntityType:
    """Simple heuristic entity type classification."""
    text_lower = text.lower()

    # "user" is always a person
    if text_lower == "user":
        return EntityType.PERSON

    # Known entity patterns
    if any(word in text_lower for word in ["inc", "corp", "ltd", "llc", "company", "fund"]):
        return EntityType.ORGANIZATION

    # Default to OTHER — entity resolution can refine later
    return EntityType.OTHER


def _build_fact_text(extracted: dict[str, Any]) -> str:
    """Combine the 5 dimensions into a single fact string."""
    parts = [extracted.get("what", "")]

    who = extracted.get("who", "N/A")
    if who and who.upper() != "N/A":
        parts.append(f"| Involving: {who}")

    where = extracted.get("where", "N/A")
    if where and where.upper() != "N/A":
        parts.append(f"| Location: {where}")

    when = extracted.get("when", "N/A")
    if when and when.upper() != "N/A":
        parts.append(f"| When: {when}")

    why = extracted.get("why", "N/A")
    if why and why.upper() != "N/A":
        parts.append(f"| Context: {why}")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Main retain function
# ---------------------------------------------------------------------------

async def retain(
    bank_id: str,
    content: str,
    context: str = "",
    event_date: datetime | None = None,
    llm: LLMClient | None = None,
    embedder: Embedder | None = None,
    store: MemoryStore | None = None,
) -> RetainResult:
    """Store content as memory facts with extraction, embedding, and linking.

    Args:
        bank_id: Memory bank ID.
        content: Text to extract facts from.
        context: Context about when/why this memory was formed.
        event_date: When the event occurred (for temporal inference).
        llm: LLM client for fact extraction.
        embedder: Embedding generator.
        store: Memory store for persistence.

    Returns:
        RetainResult with fact IDs and token usage.
    """
    if not content.strip():
        return RetainResult(success=False, facts_count=0)

    result = RetainResult()

    # Step 1: Extract facts via LLM
    _log.debug("Extracting facts from %d chars", len(content))
    extracted_facts, usage = await _extract_facts(content, context, event_date, llm)
    result.token_usage = usage

    if not extracted_facts:
        _log.debug("No facts extracted")
        result.facts_count = 0
        return result

    # Step 2: Convert to Fact objects
    facts: list[Fact] = []
    all_entities: list[Entity] = []
    links: list[MemoryLink] = []

    for i, ef in enumerate(extracted_facts):
        fact_text = _build_fact_text(ef)

        fact = Fact(
            bank_id=bank_id,
            text=fact_text,
            what=ef.get("what", ""),
            when=ef.get("when", "N/A"),
            where=ef.get("where", "N/A"),
            who=ef.get("who", "N/A"),
            why=ef.get("why", "N/A"),
            fact_type=FactType(ef.get("fact_type", "world")),
            fact_kind=FactKind(ef.get("fact_kind", "conversation")),
            occurred_start=ef.get("occurred_start"),
            occurred_end=ef.get("occurred_end"),
            source_context=context,
            event_date=event_date.isoformat() if event_date else None,
        )

        # Temporal inference fallback
        if not fact.occurred_start and fact.fact_kind == FactKind.EVENT:
            inferred = _infer_temporal_date(fact.text, event_date)
            if inferred:
                fact.occurred_start = inferred
                fact.occurred_end = inferred

        facts.append(fact)

        # Extract entities from this fact
        raw_entities = ef.get("entities", [])
        for re_ent in raw_entities:
            ent_text = re_ent.get("text", "").strip()
            if not ent_text:
                continue
            entity = Entity(
                bank_id=bank_id,
                text=ent_text,
                entity_type=_classify_entity_type(ent_text),
                fact_ids=[fact.id],
            )
            all_entities.append(entity)

        # Create causal links
        causal = ef.get("causal_relations", [])
        if causal:
            for cr in causal:
                target_idx = cr.get("target_index", cr.get("target_fact_index", -1))
                if 0 <= target_idx < i:
                    links.append(MemoryLink(
                        source_id=fact.id,
                        target_id=facts[target_idx].id,
                        link_type=LinkType.CAUSAL,
                        strength=cr.get("strength", 0.8),
                    ))

    # Step 3: Generate embeddings
    if embedder and facts:
        texts_to_embed = [f.text for f in facts]
        embeddings = await embedder.embed(texts_to_embed)
        for fact, emb in zip(facts, embeddings):
            fact.embedding = emb

    # Step 4: Deduplicate entities (within this batch)
    unique_entities = _dedup_entities(all_entities)

    # Step 4b: Cross-retain entity resolution (merge with existing entities)
    if store:
        unique_entities = await _resolve_cross_retain_entities(
            bank_id, unique_entities, store
        )

    # Step 5: Create entity links (facts sharing the same entity)
    entity_links = _create_entity_links(facts, unique_entities)
    links.extend(entity_links)

    # Step 6: Persist
    if store:
        await store.save_facts(facts)
        if unique_entities:
            await store.save_entities(unique_entities)
        if links:
            await store.save_links(links)

    result.fact_ids = [f.id for f in facts]
    result.entity_ids = [e.id for e in unique_entities]
    result.facts_count = len(facts)
    result.success = True

    _log.info("Retained %d facts, %d entities, %d links for bank %s",
              len(facts), len(unique_entities), len(links), bank_id)

    return result


# ---------------------------------------------------------------------------
# LLM fact extraction
# ---------------------------------------------------------------------------

async def _extract_facts(
    content: str,
    context: str,
    event_date: datetime | None,
    llm: LLMClient | None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Extract facts from content using LLM.

    Returns:
        (list of extracted fact dicts, token usage dict)
    """
    if llm is None:
        # No LLM — return content as a single fact (fallback)
        return [{"what": content, "when": "N/A", "where": "N/A",
                 "who": "N/A", "why": context or "N/A",
                 "fact_type": "world", "fact_kind": "conversation",
                 "entities": []}], {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    # Build user message
    user_parts = []
    if context:
        user_parts.append(f"Context: {context}")
    if event_date:
        user_parts.append(f"Event Date: {event_date.strftime('%B %d, %Y')}")
    user_parts.append(f"Text:\n{content}")
    user_message = "\n\n".join(user_parts)

    messages = [
        {"role": "system", "content": FACT_EXTRACTION_PROMPT},
        {"role": "user", "content": user_message},
    ]

    try:
        response = await llm.complete(
            messages=messages,
            temperature=0.0,
            max_tokens=4096,
            response_format={"type": "json_object"},
        )

        content_str = response.get("content", "")
        usage = response.get("usage", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})

        # Parse JSON response
        parsed = _parse_extraction_response(content_str)
        return parsed, usage

    except Exception as exc:
        _log.warning("Fact extraction failed: %s", exc)
        # Fallback: store as single fact
        return [{"what": content, "when": "N/A", "where": "N/A",
                 "who": "N/A", "why": context or "N/A",
                 "fact_type": "world", "fact_kind": "conversation",
                 "entities": []}], {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def _parse_extraction_response(content: str) -> list[dict[str, Any]]:
    """Parse LLM JSON response into list of fact dicts."""
    if not content:
        return []

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Try to find JSON in the response
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                _log.warning("Could not parse extraction response as JSON")
                return []
        else:
            return []

    # Handle both {"facts": [...]} and direct [...]
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("facts", [])

    return []


# ---------------------------------------------------------------------------
# Entity deduplication
# ---------------------------------------------------------------------------

def _dedup_entities(entities: list[Entity]) -> list[Entity]:
    """Deduplicate entities by (bank_id, text, type) using Entity.__hash__."""
    seen: dict[int, Entity] = {}
    for e in entities:
        h = hash(e)
        if h in seen:
            # Merge fact_ids
            existing = seen[h]
            for fid in e.fact_ids:
                if fid not in existing.fact_ids:
                    existing.fact_ids.append(fid)
        else:
            seen[h] = e
    return list(seen.values())


async def _resolve_cross_retain_entities(
    bank_id: str,
    new_entities: list[Entity],
    store: Any,
) -> list[Entity]:
    """Resolve new entities against existing ones in the store.

    If an entity with the same text and type already exists, merge
    the new fact_ids into the existing entity instead of creating
    a duplicate. This ensures a canonical entity graph across
    multiple retain() calls.
    """
    resolved: list[Entity] = []

    try:
        existing_entities = await store.get_entities(bank_id)
    except Exception:
        return new_entities

    # Build lookup by (text_lower, entity_type)
    existing_map: dict[tuple[str, str], Entity] = {}
    for e in existing_entities:
        key = (e.text.lower(), e.entity_type.value)
        existing_map[key] = e

    for new_ent in new_entities:
        key = (new_ent.text.lower(), new_ent.entity_type.value)
        if key in existing_map:
            # Merge fact_ids into existing entity
            existing = existing_map[key]
            for fid in new_ent.fact_ids:
                if fid not in existing.fact_ids:
                    existing.fact_ids.append(fid)
            resolved.append(existing)
        else:
            resolved.append(new_ent)
            existing_map[key] = new_ent

    return resolved


def _create_entity_links(facts: list[Fact], entities: list[Entity]) -> list[MemoryLink]:
    """Create entity links between facts that share the same entity."""
    links: list[MemoryLink] = []

    for entity in entities:
        fids = entity.fact_ids
        for i in range(len(fids)):
            for j in range(i + 1, len(fids)):
                links.append(MemoryLink(
                    source_id=fids[i],
                    target_id=fids[j],
                    link_type=LinkType.ENTITY,
                    strength=0.7,
                ))

    return links
