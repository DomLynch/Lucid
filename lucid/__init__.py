"""
Lucid — Clear memory for cognitive agents.

Extracted from Hindsight (239K LOC → ~2K LOC).
Three operations: retain(), recall(), reflect().
"""

from lucid.interfaces import Embedder, EntityResolver, LLMClient, MemoryStore, Reranker
from lucid.recall import recall
from lucid.reflect import reflect
from lucid.retain import retain
from lucid.store import SQLiteMemoryStore
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
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "retain",
    "recall",
    "reflect",
    "SQLiteMemoryStore",
    "LLMClient",
    "Embedder",
    "Reranker",
    "MemoryStore",
    "EntityResolver",
    "Bank",
    "Budget",
    "Fact",
    "FactType",
    "FactKind",
    "Entity",
    "EntityType",
    "MemoryLink",
    "LinkType",
    "Observation",
    "MentalModel",
    "RetainResult",
    "RecallResult",
    "ReflectResult",
]
