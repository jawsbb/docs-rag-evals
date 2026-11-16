"""Page-level ranking metrics with binary relevance."""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence


def recall_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Fraction of relevant pages found in the top k (multi-hop items need both pages)."""
    if not relevant:
        raise ValueError("recall is undefined without relevant items")
    return len(set(ranked[:k]) & set(relevant)) / len(set(relevant))


def reciprocal_rank(ranked: Sequence[str], relevant: Collection[str]) -> float:
    for rank, page in enumerate(ranked, start=1):
        if page in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(rank + 1) for rank, page in enumerate(ranked[:k], start=1) if page in relevant)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(set(relevant)), k) + 1))
    return dcg / ideal if ideal else 0.0


def score_ranking(ranked: Sequence[str], relevant: Collection[str], ks: Sequence[int]) -> dict[str, float]:
    scores = {f"recall@{k}": recall_at_k(ranked, relevant, k) for k in ks}
    scores["mrr"] = reciprocal_rank(ranked, relevant)
    scores["ndcg@10"] = ndcg_at_k(ranked, relevant, 10)
    return scores


def mean_scores(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {key: sum(r[key] for r in rows) / len(rows) for key in rows[0]}
