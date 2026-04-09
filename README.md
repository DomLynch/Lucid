# lucid

> Hindsight's memory runtime — stripped from 239,000 lines to 2,000.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-110%20passing-brightgreen.svg)](#tests)

Lucid is a memory runtime for AI agents. It extracts facts from conversations, stores them in a SQLite knowledge base, retrieves the most relevant ones on demand, and synthesises answers from memory using multi-turn reasoning.

Three operations. That's the whole API:

```python
await retain(text, ...)   # extract + store facts
await recall(query, ...)  # retrieve relevant facts
await reflect(query, ...) # synthesise an answer from memory
```

No Postgres. No Redis. No Kafka. No Docker. Just SQLite and an LLM.

---

## Size comparison

| Component | Hindsight | Lucid | Reduction |
|-----------|-----------|-------|-----------|
| API surface (FastAPI) | ~8,000 LOC | — | 100% |
| Storage backends (Postgres, Redis, S3) | ~15,000 LOC | ~430 LOC (SQLite) | 97% |
| Cloud control plane | ~20,000 LOC | — | 100% |
| Ingestion pipelines | ~12,000 LOC | 479 LOC | 96% |
| Retrieval engine | ~10,000 LOC | 317 LOC | 97% |
| Synthesis layer | ~8,000 LOC | 310 LOC | 96% |
| **Total** | **~239,000 LOC** | **~2,000 LOC** | **99%** |

What was cut: REST API, database migrations, multi-tenancy, cloud infrastructure, Kafka ingestion, S3 storage, Postgres/Redis backends, authentication, billing, admin dashboards.

What remains: the memory logic.

---

## Install

```bash
# No pip package yet — copy the folder directly
cp -r lucid/ your-project/lucid/

# Dependencies
pip install httpx  # for the built-in LLM client
```

```
lucid/
├── types.py      # Data model (Fact, Entity, Bank, Budget, etc.)
├── interfaces.py # Protocol definitions (LLMClient, Embedder, MemoryStore, Reranker)
├── retain.py     # Fact extraction + storage pipeline
├── recall.py     # 4-strategy retrieval with RRF fusion
├── reflect.py    # Multi-turn synthesis over memory
└── store.py      # SQLite-backed MemoryStore
```

---

## Quick start

```python
import asyncio
from lucid import retain, recall, reflect, SQLiteMemoryStore, Budget

# Minimal implementations of the protocol interfaces
class MyLLM:
    async def complete(self, messages, tools=None, **kwargs):
        # Use OpenAI, Ollama, Anthropic, etc.
        ...

class MyEmbedder:
    async def embed(self, text: str) -> list[float]:
        # Use nomic-embed-text, text-embedding-3-small, etc.
        ...

async def main():
    store = SQLiteMemoryStore("memory.db")
    llm = MyLLM()
    embedder = MyEmbedder()
    bank_id = "user-alice"  # partition per user

    # --- RETAIN ---
    # Extract facts from a conversation turn and store them
    result = await retain(
        text="Alice mentioned she's allergic to shellfish and prefers morning meetings.",
        store=store,
        llm=llm,
        embedder=embedder,
        bank_id=bank_id,
    )
    print(f"Stored {len(result.facts)} facts, resolved {len(result.entities)} entities")

    # --- RECALL ---
    # Retrieve the most relevant facts for a query
    results = await recall(
        query="What are Alice's dietary restrictions?",
        store=store,
        embedder=embedder,
        bank_id=bank_id,
        budget=Budget(max_facts=5, max_tokens=800),
    )
    for fact in results.facts:
        print(f"  [{fact.score:.2f}] {fact.text}")

    # --- REFLECT ---
    # Synthesise an answer from memory (multi-turn reasoning)
    answer = await reflect(
        query="What should I know before scheduling a lunch with Alice?",
        store=store,
        llm=llm,
        embedder=embedder,
        bank_id=bank_id,
    )
    print(answer.response)

asyncio.run(main())
```

---

## How it works

### retain()

Extracts structured facts from free text using an LLM, resolves entities, embeds facts for vector search, and persists everything:

```
input text
    ↓
LLM extracts Fact objects (text, kind, type, entities, temporal markers)
    ↓
Resolve entities: match against existing entities in store
    ↓
Link facts to resolved entities (cross-retain continuity)
    ↓
Embed fact text for vector retrieval
    ↓
Save Facts, Entities, and MemoryLinks to SQLite
```

**Fact kinds:**
- `EPISODIC` — things that happened ("Alice visited Paris")
- `SEMANTIC` — things that are true ("Alice is allergic to shellfish")
- `PROCEDURAL` — how to do something ("Alice prefers email over Slack")

**Fact types:**
- `preference`, `belief`, `fact`, `instruction`, `event`, `identity`, `relationship`

### recall()

Four retrieval strategies fused with Reciprocal Rank Fusion (RRF):

| Strategy | What it does |
|----------|-------------|
| Semantic vector search | Cosine similarity on embedded fact text |
| Keyword search | BM25-style text matching |
| Entity lookup | Facts linked to entities matching query |
| Recency boost | Recent facts ranked higher |

Results are filtered by a `Budget` (max facts, max tokens) to keep context windows manageable.

### reflect()

Multi-turn agentic synthesis: the LLM reasons over recalled facts to answer a question, calling back into memory for follow-up retrievals when needed:

```
query
  ↓
recall() → top-k facts
  ↓
LLM synthesises answer (with optional tool call: search_memory)
  ↓
If LLM calls search_memory → recall() again with new query
  ↓
Continue until answer is complete or max_turns reached
  ↓
Return ReflectResult(response, facts_used, turns)
```

---

## Data model

```python
@dataclass
class Fact:
    id: str
    bank_id: str          # partition key
    text: str             # the fact as a natural language sentence
    kind: FactKind        # EPISODIC | SEMANTIC | PROCEDURAL
    fact_type: FactType   # preference | belief | event | instruction | ...
    embedding: list[float]
    created_at: str
    entity_ids: list[str]

@dataclass
class Entity:
    id: str
    bank_id: str
    name: str
    entity_type: EntityType   # person | place | organization | concept | other
    summary: str
    fact_ids: list[str]

@dataclass
class Budget:
    max_facts: int = 10
    max_tokens: int = 2000
```

---

## SQLite schema

```sql
facts        -- extracted facts with embeddings
entities     -- resolved named entities
memory_links -- fact ↔ entity associations
observations -- raw input log (provenance)
```

WAL mode enabled. Vector similarity computed in Python — suitable for up to ~50k facts. Swap the `MemoryStore` protocol for a vector DB backend when you need more scale.

---

## Bring your own components

Every component is a protocol — swap in your preferred implementation:

```python
from lucid.interfaces import Embedder, Reranker, MemoryStore

class ChromaStore(MemoryStore):
    """Use ChromaDB as the backend instead of SQLite."""
    async def save_fact(self, fact): ...
    async def search_by_embedding(self, embedding, bank_id, limit): ...
    # ... implement the other methods

class CohereReranker(Reranker):
    """Re-rank results with Cohere's rerank endpoint."""
    async def rerank(self, query, facts, limit): ...
```

---

## Tests

```bash
# Unit tests (no LLM required)
python3 -m pytest tests/ -q --ignore=tests/test_e2e.py

# End-to-end test (requires LLM + embedder)
OPENAI_API_KEY=sk-... python3 -m pytest tests/test_e2e.py -v -s
```

110 tests covering fact extraction, entity resolution, recall strategies, RRF fusion, synthesis, SQLite store.

---

## What was removed from Hindsight

Lucid is a targeted extraction of Hindsight's memory kernel:

- **REST API** (FastAPI + auth + rate limiting) — not included
- **Postgres + Redis backends** — replaced with a single SQLite store
- **S3 / object storage** — removed
- **Kafka ingestion pipeline** — removed
- **Multi-tenancy + billing** — removed (use `bank_id` for partitioning)
- **Cloud control plane** — removed
- **Helm charts / Docker Compose** — removed
- **TypeScript client SDK** — removed (Python only)

The fact extraction logic, entity resolution, four-strategy retrieval, RRF fusion, and multi-turn reflection are preserved.

---

## Part of a suite

Lucid pairs naturally with:

- **[NanoLetta](https://github.com/domininclynch/nanoletta)** — cognitive agent loop (Letta → 1.9k LOC). Wire Lucid in via the memory tool interface.
- **[Temporal](https://github.com/domininclynch/temporal)** — temporal knowledge graph (Graphiti → 2.8k LOC). Knows *when* facts changed.

---

## Requirements

- Python 3.11+
- `httpx` (for the built-in LLM client — omit if you bring your own)
- Any OpenAI-compatible LLM endpoint
- Any embedding function returning `list[float]`

---

## License

MIT. See [LICENSE](LICENSE).

---

## Acknowledgments

The memory architecture, fact taxonomy, and retrieval design are inspired by [Hindsight](https://github.com/getzep/hindsight) by Zep AI (MIT). Lucid is an independent extraction — not affiliated with Zep.
