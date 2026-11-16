from docs_rag.evals.golden import GoldenItem
from docs_rag.evals.retrieval_eval import check_gate, evaluate, render_markdown
from docs_rag.retrieval import Hit

GATED = ["recall@5", "mrr"]
BASELINE = {
    "n_answerable": 10,
    "variants": {"hybrid": {"recall@5": 0.90, "mrr": 0.70}, "fts": {"recall@5": 0.60, "mrr": 0.40}},
}
FTS_OK = {"recall@5": 0.60, "mrr": 0.40}


def test_gate_passes_within_tolerance():
    current = {"hybrid": {"recall@5": 0.88, "mrr": 0.69}, "fts": FTS_OK}
    assert check_gate(current, "hybrid", 10, BASELINE, {"recall@5": 0.8}, 0.03, GATED) == []


def test_gate_fails_on_regression_vs_baseline():
    current = {"hybrid": {"recall@5": 0.85, "mrr": 0.70}, "fts": FTS_OK}
    failures = check_gate(current, "hybrid", 10, BASELINE, {}, 0.03, GATED)
    assert len(failures) == 1
    assert "hybrid: recall@5 regressed" in failures[0]


def test_gate_catches_regression_in_a_non_main_variant():
    current = {"hybrid": {"recall@5": 0.90, "mrr": 0.70}, "fts": {"recall@5": 0.10, "mrr": 0.40}}
    assert check_gate(current, "hybrid", 10, BASELINE, {}, 0.03, GATED) == [
        "fts: recall@5 regressed 0.100 < baseline 0.600 - tolerance 0.030"
    ]
    assert (
        "fts: present in the baseline"
        in check_gate({"hybrid": current["hybrid"]}, "hybrid", 10, BASELINE, {}, 0.03, GATED)[0]
    )


def test_gate_fails_below_absolute_floor_even_if_baseline_is_low():
    low = {"n_answerable": 10, "variants": {"hybrid": {"recall@5": 0.50, "mrr": 0.40}}}
    failures = check_gate(
        {"hybrid": {"recall@5": 0.55, "mrr": 0.45}}, "hybrid", 10, low, {"recall@5": 0.6}, 0.03, GATED
    )
    assert failures == ["hybrid: recall@5 = 0.550 is below the absolute floor 0.600"]


def test_gate_refuses_stale_or_missing_baseline():
    perfect = {"hybrid": {"recall@5": 1.0, "mrr": 1.0}, "fts": FTS_OK}
    assert "golden set or corpus changed" in check_gate(perfect, "hybrid", 12, BASELINE, {}, 0.03, GATED)[0]
    assert "no committed baseline" in check_gate(perfect, "hybrid", 10, None, {}, 0.03, GATED)[0]


def _item(qid: str, qtype: str, pages: list[str]) -> GoldenItem:
    return GoldenItem(id=qid, question=f"question {qid}?", type=qtype, expected_pages=pages, reference_answer="r")


def test_evaluate_end_to_end_with_fake_search():
    items = [
        _item("q001", "lookup", ["a"]),
        _item("q002", "multi_hop", ["a", "c"]),
        _item("q003", "unanswerable", []),
    ]

    def search(query: str, n: int, mode: str) -> list[Hit]:
        pages = ["a", "b", "c"] if mode == "good" else ["b", "b", "a"]
        return [Hit(f"{p}#{i}", p, "", "", "", "", 0.5) for i, p in enumerate(pages)][:n]

    report = evaluate(items, search, ["good", "vector"], ks=[1, 3, 5, 10], pool=10)
    good = report["variants"]["good"]
    assert good["overall"]["recall@1"] == 0.75  # q001 full hit, q002 one of two pages
    assert good["overall"]["recall@3"] == 1.0
    assert good["by_type"]["multi_hop"]["n"] == 1
    assert report["variants"]["vector"]["overall"]["mrr"] == 0.5  # "b" duplicates collapse, "a" is rank 2
    assert report["n_answerable"] == 2
    assert report["n_unanswerable"] == 1
    assert "q003" in report["no_answer_signal"]["vector_top1_similarity_unanswerable"]

    report.update(corpus="fixture", corpus_sha="0" * 40, embedding_model="m", pages=3, chunks=3, generated_at="now")
    markdown = render_markdown(report, "good", items)
    assert "| **good** (gated) |" in markdown
    assert "misses (recall@10 < 1): 0" in markdown
