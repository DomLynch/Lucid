"""
lucid/interfaces.py — Protocol interfaces for pluggable backends.

Same pattern as NanoLetta: Protocol-based structural subtyping.
Implementations don't need to inherit, just match the signature.

Five interfaces:
1. LLMClient — for fact extraction and reflect agent
2. Embedder — for vector generation
3. Reranker — for cross-encoder reranking
4. MemoryStore — for fact/entity/observation persistence
5. EntityResolver — for entity linking and deduplication
"""

from typing import Any, Protocol, runtime_checkable

from lucid.types import (
    Bank,
    Entity,
    Fact,
    MemoryLink,
    Observation,
)


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------

@runtime_checkable
class LLMClient(Protocol):
    """LLM client for fact extraction and reflect agent.

    Can reuse NanoLetta's OpenAICompatibleClient.
    """

    async def complete(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Send a completion request.

        Args:
            messages: Chat messages in OpenAI format.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            response_format: Optional JSON schema for structured output.
            tools: Optional tool schemas in OpenAI format (for agentic loops).

        Returns:
            {"content": str, "tool_calls": list | None,
             "usage": {"input_tokens": int, "output_tokens": int, "total_tokens": int}}
        """
        ...


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

@runtime_checkable
class Embedder(Protocol):
    """Vector embedding generator.

    Converts text to dense vectors for semantic search.
    """

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors (same length as input).
        """
        ...

    @property
    def dimension(self) -> int:
        """Embedding vector dimension (e.g., 384 for bge-small)."""
        ...


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------

@runtime_checkable
class Reranker(Protocol):
    """Cross-encoder reranker for scoring query-document pairs."""

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """Rerank documents by relevance to query.

        Args:
            query: The search query.
            documents: List of document texts to rerank.
            top_k: Number of top results to return.

        Returns:
            List of (original_index, score) tuples, sorted by score descending.
        """
        ...


# ---------------------------------------------------------------------------
# Memory Store
# ---------------------------------------------------------------------------

@runtime_checkable
class MemoryStore(Protocol):
    """Persistence for facts, entities, observations, and banks.

    SQLite implementation provided by default.
    PostgreSQL + pgvector implementation for scale.
    """

    # Bank operations
    async def create_bank(self, bank: Bank) -> None: ...
    async def get_bank(self, bank_id: str) -> Bank: ...

    # Fact operations
    async def save_facts(self, facts: list[Fact]) -> None: ...
    async def get_facts(self, bank_id: str, fact_ids: list[str] | None = None) -> list[Fact]: ...
    async def search_facts_by_embedding(
        self, bank_id: str, query_embedding: list[float], limit: int = 50
    ) -> list[Fact]: ...
    async def search_facts_by_text(
        self, bank_id: str, query: str, limit: int = 50
    ) -> list[Fact]: ...

    # Entity operations
    async def save_entities(self, entities: list[Entity]) -> None: ...
    async def get_entities(self, bank_id: str) -> list[Entity]: ...
    async def get_entity_by_text(self, bank_id: str, text: str) -> Entity | None: ...

    # Link operations
    async def save_links(self, links: list[MemoryLink]) -> None: ...
    async def get_linked_facts(
        self, bank_id: str, fact_id: str, link_type: str | None = None, limit: int = 20
    ) -> list[Fact]: ...

    # Observation operations
    async def save_observations(self, observations: list[Observation]) -> None: ...
    async def get_observations(self, bank_id: str, limit: int = 20) -> list[Observation]: ...
    async def search_observations_by_embedding(
        self, bank_id: str, query_embedding: list[float], limit: int = 10
    ) -> list[Observation]: ...


# ---------------------------------------------------------------------------
# Entity Resolver
# ---------------------------------------------------------------------------

@runtime_checkable
class EntityResolver(Protocol):
    """Resolves and deduplicates entities across facts.

    "Alice", "alice@company.com", and "the account owner" should
    resolve to the same entity when context allows.
    """

    async def resolve(
        self,
        texts: list[str],
        entity_types: list[str],
        existing_entities: list[Entity],
    ) -> list[Entity]:
        """Resolve entity mentions to canonical entities.

        Args:
            texts: Raw entity mention strings.
            entity_types: Corresponding entity type for each text.
            existing_entities: Already-known entities for this bank.

        Returns:
            List of resolved Entity objects (may reuse existing IDs).
        """
        ...
