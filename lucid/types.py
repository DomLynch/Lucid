"""
lucid/types.py — Core data types for the memory system.

Models the memory hierarchy:
  Raw input → Facts → Entities → Observations → Mental Models

Each type maps directly to Hindsight's data model so golden corpus
outputs can be compared 1:1.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
import hashlib
import uuid


def _new_id() -> str:
    """Generate a short unique ID."""
    return uuid.uuid4().hex[:16]


def _now_iso() -> str:
    """Current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class FactType(str, Enum):
    """Type of fact — what domain it belongs to."""
    WORLD = "world"          # About user's life, other people, events
    EXPERIENCE = "experience"  # Agent's own observations
    ASSISTANT = "assistant"   # Interactions with assistant


class FactKind(str, Enum):
    """Kind of fact — temporal classification."""
    EVENT = "event"           # Specific datable occurrence
    CONVERSATION = "conversation"  # Ongoing state, preference, trait


class EntityType(str, Enum):
    """Type of named entity."""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    DATE = "date"
    CONCEPT = "concept"
    OTHER = "other"


class LinkType(str, Enum):
    """Type of connection between facts."""
    SEMANTIC = "semantic"     # Meaning similarity
    TEMPORAL = "temporal"     # Time proximity
    ENTITY = "entity"        # Shared entities
    CAUSAL = "causal"        # Cause-effect relationship


class Budget(str, Enum):
    """Retrieval budget — controls search breadth."""
    LOW = "low"       # Fast, narrow (100 candidates)
    MID = "mid"       # Balanced (300 candidates)
    HIGH = "high"     # Thorough (600 candidates)


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------

@dataclass
class Fact:
    """One extracted memory unit.

    Maps to Hindsight's memory_units table. The fundamental
    unit of memory — everything else is built on top of facts.

    The 5 dimensions (what/when/where/who/why) come directly
    from Hindsight's fact extraction prompt.
    """

    id: str = field(default_factory=_new_id)
    bank_id: str = ""

    # Content
    text: str = ""              # Combined fact text
    what: str = ""              # Core fact (1-2 sentences)
    when: str = "N/A"           # Temporal info
    where: str = "N/A"         # Location
    who: str = "N/A"           # People involved
    why: str = "N/A"           # Context/significance

    # Classification
    fact_type: FactType = FactType.WORLD
    fact_kind: FactKind = FactKind.CONVERSATION

    # Temporal
    event_date: str | None = None       # When the event occurred
    occurred_start: str | None = None   # Start of event window
    occurred_end: str | None = None     # End of event window

    # Scoring
    confidence: float = 1.0    # 0.0-1.0
    relevance: float = 0.0     # Set during recall, not stored

    # Embedding (stored as list of floats)
    embedding: list[float] = field(default_factory=list)

    # Metadata
    source_context: str = ""    # Context provided at retain time
    created_at: str = field(default_factory=_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_embedding(self) -> bool:
        return len(self.embedding) > 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict (excluding embedding for readability)."""
        return {
            "id": self.id,
            "bank_id": self.bank_id,
            "text": self.text,
            "what": self.what,
            "when": self.when,
            "where": self.where,
            "who": self.who,
            "why": self.why,
            "fact_type": self.fact_type.value,
            "fact_kind": self.fact_kind.value,
            "event_date": self.event_date,
            "confidence": self.confidence,
            "source_context": self.source_context,
            "created_at": self.created_at,
        }


@dataclass
class Entity:
    """A named entity extracted from facts.

    Maps to Hindsight's entities table. Entities connect facts
    that mention the same person, org, place, etc.
    """

    id: str = field(default_factory=_new_id)
    bank_id: str = ""
    text: str = ""              # Entity name (e.g., "Alice")
    entity_type: EntityType = EntityType.OTHER
    embedding: list[float] = field(default_factory=list)
    fact_ids: list[str] = field(default_factory=list)  # Facts that mention this entity
    created_at: str = field(default_factory=_now_iso)

    def __hash__(self) -> int:
        return hash((self.bank_id, self.text.lower(), self.entity_type.value))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Entity):
            return False
        return (
            self.bank_id == other.bank_id
            and self.text.lower() == other.text.lower()
            and self.entity_type == other.entity_type
        )


@dataclass
class MemoryLink:
    """A connection between two facts.

    Maps to Hindsight's memory_links table. Links form the
    graph that spreading activation traverses during recall.
    """

    source_id: str = ""
    target_id: str = ""
    link_type: LinkType = LinkType.SEMANTIC
    strength: float = 0.5      # 0.0-1.0


@dataclass
class Observation:
    """A consolidated insight derived from multiple facts.

    Maps to Hindsight's observations table. Observations are
    higher-level than raw facts — they represent understood
    patterns or entity-specific knowledge.

    Example: "Alice prefers direct communication" (derived
    from multiple facts about communication style).
    """

    id: str = field(default_factory=_new_id)
    bank_id: str = ""
    text: str = ""
    entity_id: str | None = None  # If entity-specific
    source_fact_ids: list[str] = field(default_factory=list)
    embedding: list[float] = field(default_factory=list)
    confidence: float = 0.8
    created_at: str = field(default_factory=_now_iso)


@dataclass
class MentalModel:
    """A synthesized understanding — highest level of the hierarchy.

    Created by reflect() when patterns emerge across observations.
    Example: "Alice is a high-intensity founder who values quality
    over speed and manages multiple AI agents simultaneously."
    """

    id: str = field(default_factory=_new_id)
    bank_id: str = ""
    text: str = ""
    observation_ids: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)


# ---------------------------------------------------------------------------
# Bank (memory namespace)
# ---------------------------------------------------------------------------

@dataclass
class Bank:
    """A memory bank — namespace for a set of memories.

    One bank per agent/user. Contains all facts, entities,
    observations, and mental models for that context.
    """

    id: str = field(default_factory=_new_id)
    name: str = ""
    mission: str = ""          # What to prioritize remembering
    background: str = ""       # Context about the bank's subject
    disposition: dict[str, int] = field(default_factory=lambda: {
        "skepticism": 3,       # 1-5, how critically to evaluate
        "literalism": 3,       # 1-5, how literally to interpret
        "empathy": 3,          # 1-5, how much to weight emotional context
    })
    created_at: str = field(default_factory=_now_iso)


# ---------------------------------------------------------------------------
# Operation results
# ---------------------------------------------------------------------------

@dataclass
class RetainResult:
    """Result from a retain() operation."""

    success: bool = True
    fact_ids: list[str] = field(default_factory=list)
    entity_ids: list[str] = field(default_factory=list)
    facts_count: int = 0
    token_usage: dict[str, int] = field(default_factory=lambda: {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    })


@dataclass
class RecallResult:
    """Result from a recall() operation."""

    results: list[Fact] = field(default_factory=list)
    entities: dict[str, Entity] = field(default_factory=dict)
    query: str = ""
    total_candidates: int = 0


@dataclass
class ReflectResult:
    """Result from a reflect() operation."""

    text: str = ""
    based_on: dict[str, str] = field(default_factory=dict)  # fact_id → fact_text
    tool_calls: int = 0
    token_usage: dict[str, int] = field(default_factory=lambda: {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    })
