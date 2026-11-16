import math

import pytest

from docs_rag.evals.metrics import mean_scores, ndcg_at_k, recall_at_k, reciprocal_rank, score_ranking
from docs_rag.retrieval import Hit, rrf, unique_pages


def test_recall_counts_every_expected_page():
    ranked = ["a", "b", "c", "d"]
    assert recall_at_k(ranked, ["c"], 1) == 0.0
    assert recall_at_k(ranked, ["c"], 3) == 1.0
    assert recall_at_k(ranked, ["a", "d"], 3) == 0.5  # multi-hop: half the evidence found
    with pytest.raises(ValueError, match="undefined"):
        recall_at_k(ranked, [], 3)


def test_reciprocal_rank_uses_first_relevant_hit():
    assert reciprocal_rank(["x", "y", "a"], ["a", "y"]) == 0.5
    assert reciprocal_rank(["x"], ["a"]) == 0.0


def test_ndcg_binary_relevance():
    assert ndcg_at_k(["a", "b"], ["a"], 10) == 1.0
    assert ndcg_at_k(["b", "a"], ["a"], 10) == pytest.approx(1 / math.log2(3))
    ideal = 1 + 1 / math.log2(3)
    assert ndcg_at_k(["a", "x", "b"], ["a", "b"], 10) == pytest.approx((1 + 0.5) / ideal)


def test_score_ranking_and_mean():
    rows = [score_ranking(["a"], ["a"], [1, 5]), score_ranking(["b", "a"], ["a"], [1, 5])]
    assert mean_scores(rows) == {"recall@1": 0.5, "recall@5": 1.0, "mrr": 0.75, "ndcg@10": pytest.approx(0.8155, 1e-3)}


def test_rrf_rewards_agreement_between_lists():
    fused = rrf([["a", "b", "c"], ["c", "a", "d"]], k=60)
    ids = [doc for doc, _ in fused]
    assert ids[0] == "a"  # ranks 1 and 2 beat c's ranks 3 and 1
    assert ids.index("c") < ids.index("b")
    assert set(ids) == {"a", "b", "c", "d"}
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)


def test_unique_pages_keeps_first_occurrence():
    def hit(cid: str) -> Hit:
        return Hit(cid, cid.split("#")[0], "", "", "", "", 0.0)

    assert unique_pages([hit("p1#0"), hit("p2#3"), hit("p1#2"), hit("p3#0")]) == ["p1", "p2", "p3"]
