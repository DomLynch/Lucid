"""
Lucid — Clear memory for cognitive agents.

Extracted from Hindsight (239K LOC → ~2K LOC).
Three operations: retain(), recall(), reflect().
"""

__version__ = "0.1.0"

from lucid.retain import retain
from lucid.recall import recall
from lucid.reflect import reflect
from lucid.store import SQLiteMemoryStore
from lucid.types import (
    Fact,
    Entity,
    Observation,
    MentalModel,
    Bank,
    MemoryLink,
    RetainResult,
    RecallResult,
    ReflectResult,
)
from lucid.interfaces import LLMClient, Embedder, Reranker, MemoryStore, EntityResolver

__all__ = [
    "retain",
    "recall",
    "reflect",
    "SQLiteMemoryStore",
    "Fact",
    "Entity",
    "Observation",
    "MentalModel",
    "Bank",
    "MemoryLink",
    "RetainResult",
    "RecallResult",
    "ReflectResult",
    "LLMClient",
    "Embedder",
    "Reranker",
    "MemoryStore",
    "EntityResolver",
]
