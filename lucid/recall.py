"""
lucid/recall.py — 4-way parallel retrieval with RRF merge and reranking.

The recall pipeline:
1. Embed the query
2. Run 4 parallel retrieval strategies:
   a) Semantic search (cosine similarity on embeddings)
   b) BM25 keyword search (term frequency matching)
   c) Entity graph traversal (spreading activation)
   d) Temporal search (date proximity)
3. Merge results via Reciprocal Rank Fusion (RRF)
4. Rerank top candidates with cross-encoder
5. Apply token budget filtering
6. Return ranked facts

Transplanted from Hindsight's search/retrieval.py (690 LOC) +
memory_engine.py recall path (~400 LOC). Target: ~350 LOC.
"""

import logging
import math
from collections import defaultdict
from datetime import datetime
from lucid.interfaces import Embedder, MemoryStore, Reranker
from lucid.types import Budget, Entity, Fact, RecallResult

_log = logging.getLogger("lucid.recall")

# RRF constant (standard value from the paper)
_RRF_K = 60

# Budget → candidate limits
_BUDGET_LIMITS = {
    Budget.LOW: 100,
    Budget.MID: 300,
    Budget.HIGH: 600,
}

# Default max tokens for result filtering
_DEFAULT_MAX_TOKENS = 4096

# Approximate tokens per character
_CHARS_PER_TOKEN = 4


# ---------------------------------------------------------------------------
# Main recall function
# ---------------------------------------------------------------------------

async def recall(
    bank_id: str,
    query: str,
    embedder: Embedder | None = None,
    store: MemoryStore | None = None,
    reranker: Reranker | None = None,
    budget: Budget = Budget.MID,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    fact_types: list[str] | None = None,
    question_date: datetime | None = None,
) -> RecallResult:
    """Recall memories using 4-way parallel retrieval.

    Args:
        bank_id: Memory bank ID.
        query: Search query.
        embedder: For query embedding.
        store: For fact retrieval.
        reranker: For cross-encoder reranking.
        budget: Search breadth (LOW/MID/HIGH).
        max_tokens: Token budget for results.
        fact_types: Filter by fact types (e.g., ["world", "experience"]).
        question_date: For temporal proximity scoring.

    Returns:
        RecallResult with ranked facts.
    """
    if not query.strip() or store is None:
        return RecallResult(query=query)

    candidate_limit = _BUDGET_LIMITS.get(budget, 300)

    # Step 1: Run retrieval strategies
    semantic_results: list[tuple[str, float]] = []
    bm25_results: list[tuple[str, float]] = []
    entity_results: list[tuple[str, float]] = []
    temporal_results: list[tuple[str, float]] = []

    # 1a: Semantic search
    if embedder:
        query_embeddings = await embedder.embed([query])
        if query_embeddings:
            query_emb = query_embeddings[0]
            semantic_facts = await store.search_facts_by_embedding(
                bank_id, query_emb, limit=candidate_limit
            )
            for i, fact in enumerate(semantic_facts):
                semantic_results.append((fact.id, fact.relevance))

    # 1b: BM25 keyword search
    bm25_facts = await store.search_facts_by_text(
        bank_id, query, limit=candidate_limit
    )
    for i, fact in enumerate(bm25_facts):
        bm25_results.append((fact.id, fact.relevance))

    # 1c: Entity graph traversal
    entity_results = await _entity_graph_search(bank_id, query, store, candidate_limit)

    # 1d: Temporal search
    if question_date:
        all_facts = await store.get_facts(bank_id)
        temporal_results = _temporal_search(all_facts, question_date, candidate_limit)

    # Step 2: Collect all candidate fact IDs
    all_fact_ids: set[str] = set()
    for results in [semantic_results, bm25_results, entity_results, temporal_results]:
        all_fact_ids.update(fid for fid, _ in results)

    if not all_fact_ids:
        return RecallResult(query=query, total_candidates=0)

    # Step 3: RRF merge
    rrf_scores = _rrf_merge(
        semantic=semantic_results,
        bm25=bm25_results,
        entity=entity_results,
        temporal=temporal_results,
    )

    # Sort by RRF score
    sorted_ids = sorted(rrf_scores.keys(), key=lambda fid: rrf_scores[fid], reverse=True)

    # Step 4: Load facts
    candidate_facts = await store.get_facts(bank_id, fact_ids=sorted_ids)
    fact_map = {f.id: f for f in candidate_facts}

    # Order by RRF score
    ordered_facts = [fact_map[fid] for fid in sorted_ids if fid in fact_map]

    # Step 5: Filter by fact type
    if fact_types:
        ordered_facts = [f for f in ordered_facts if f.fact_type.value in fact_types]

    # Step 6: Rerank top candidates
    if reranker and ordered_facts:
        rerank_limit = min(50, len(ordered_facts))  # Rerank top 50 max
        to_rerank = ordered_facts[:rerank_limit]
        rest = ordered_facts[rerank_limit:]

        docs = [f.text for f in to_rerank]
        reranked = await reranker.rerank(query, docs, top_k=rerank_limit)

        reranked_facts = [to_rerank[idx] for idx, score in reranked]
        # Update relevance scores
        for i, (idx, score) in enumerate(reranked):
            reranked_facts[i].relevance = score

        ordered_facts = reranked_facts + rest

    # Step 7: Apply token budget
    filtered = _apply_token_budget(ordered_facts, max_tokens)

    # Step 8: Load relevant entities
    entities: dict[str, Entity] = {}
    try:
        all_entities = await store.get_entities(bank_id)
        for entity in all_entities:
            entities[entity.id] = entity
    except Exception:
        pass

    _log.info(
        "Recall for bank %s: query='%s' candidates=%d returned=%d",
        bank_id, query[:50], len(all_fact_ids), len(filtered),
    )

    return RecallResult(
        results=filtered,
        entities=entities,
        query=query,
        total_candidates=len(all_fact_ids),
    )


# ---------------------------------------------------------------------------
# RRF (Reciprocal Rank Fusion)
# ---------------------------------------------------------------------------

def _rrf_merge(
    semantic: list[tuple[str, float]],
    bm25: list[tuple[str, float]],
    entity: list[tuple[str, float]],
    temporal: list[tuple[str, float]],
) -> dict[str, float]:
    """Merge ranked lists using Reciprocal Rank Fusion.

    RRF formula: score(d) = sum(1 / (k + rank_i(d))) for each list i

    This is the standard fusion method — no tuning needed,
    works well without score normalization across lists.
    """
    scores: dict[str, float] = defaultdict(float)

    for ranked_list in [semantic, bm25, entity, temporal]:
        for rank, (fact_id, _) in enumerate(ranked_list):
            scores[fact_id] += 1.0 / (_RRF_K + rank + 1)

    return dict(scores)


# ---------------------------------------------------------------------------
# Entity graph search
# ---------------------------------------------------------------------------

async def _entity_graph_search(
    bank_id: str,
    query: str,
    store: MemoryStore,
    limit: int,
) -> list[tuple[str, float]]:
    """Search via entity graph — find facts connected to query entities.

    Simple spreading activation:
    1. Find entities mentioned in the query
    2. Find all facts connected to those entities
    3. Score by number of shared entities (more = higher)
    """
    results: list[tuple[str, float]] = []

    try:
        all_entities = await store.get_entities(bank_id)
    except Exception:
        return results

    # Find entities mentioned in query
    query_lower = query.lower()
    matching_entities = [
        e for e in all_entities
        if e.text.lower() in query_lower or query_lower in e.text.lower()
    ]

    if not matching_entities:
        return results

    # Count how many query-entities each fact shares
    fact_scores: dict[str, float] = defaultdict(float)
    for entity in matching_entities:
        for fact_id in entity.fact_ids:
            fact_scores[fact_id] += 1.0

    # Normalize and sort
    max_score = max(fact_scores.values()) if fact_scores else 1.0
    sorted_facts = sorted(
        fact_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:limit]

    return [(fid, score / max_score) for fid, score in sorted_facts]


# ---------------------------------------------------------------------------
# Temporal search
# ---------------------------------------------------------------------------

def _temporal_search(
    facts: list[Fact],
    question_date: datetime,
    limit: int,
) -> list[tuple[str, float]]:
    """Score facts by temporal proximity to question_date.

    Closer dates score higher. Uses exponential decay.
    """
    scored: list[tuple[str, float]] = []

    for fact in facts:
        date_str = fact.occurred_start or fact.event_date
        if not date_str:
            continue

        try:
            fact_date = datetime.fromisoformat(date_str)
            # Days between dates
            delta_days = abs((question_date - fact_date).total_seconds()) / 86400
            # Exponential decay: score = e^(-delta/30)
            # 0 days = 1.0, 30 days = 0.37, 90 days = 0.05
            score = math.exp(-delta_days / 30.0)
            scored.append((fact.id, score))
        except (ValueError, TypeError):
            continue

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:limit]


# ---------------------------------------------------------------------------
# Token budget filtering
# ---------------------------------------------------------------------------

def _apply_token_budget(facts: list[Fact], max_tokens: int) -> list[Fact]:
    """Filter facts to stay within token budget.

    Estimates tokens at ~4 chars per token. Stops adding facts
    when budget would be exceeded.
    """
    result: list[Fact] = []
    token_count = 0

    for fact in facts:
        fact_tokens = len(fact.text) // _CHARS_PER_TOKEN
        if token_count + fact_tokens > max_tokens:
            break
        result.append(fact)
        token_count += fact_tokens

    return result
