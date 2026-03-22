# Lucid

Lucid is a compact memory runtime extracted from Hindsight.

It provides three core operations:

- `retain()` to extract and store facts from input text
- `recall()` to retrieve relevant facts with multi-strategy search
- `reflect()` to synthesize answers over stored memory with tool-assisted reasoning

## Scope

Lucid keeps the memory core small and auditable:

- `types.py` and `interfaces.py` define the public contracts
- `retain.py` handles fact extraction, entity handling, embeddings, and persistence
- `recall.py` handles retrieval, rank fusion, reranking, and budget filtering
- `reflect.py` handles multi-turn synthesis over memory
- `store.py` provides the default SQLite-backed store

Current runtime size is approximately **2,010 LOC** across the core modules.

## Minimal usage

```python
from lucid import Budget, SQLiteMemoryStore, recall, reflect, retain

store = SQLiteMemoryStore("lucid.db")

# retain(...)
# recall(...)
# reflect(...)
```

You provide concrete implementations for the protocol interfaces in `lucid.interfaces`:

- `LLMClient`
- `Embedder`
- `Reranker`
- `MemoryStore`

## Running tests

```bash
python3 -m pytest -q
```

## Status

Lucid is currently a focused extraction intended for internal use and auditability first. Packaging and distribution metadata can be expanded later if it becomes a standalone published package.
