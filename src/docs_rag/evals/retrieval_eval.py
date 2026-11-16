"""Deterministic retrieval evals + regression gate. No API key needed.

Every variant is scored on the same golden items at page level: the chunk
ranking is collapsed to unique page ids (first occurrence wins), then
recall@k / MRR / nDCG@10 are computed against `expected_pages`.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

from docs_rag.config import Settings, load_settings
from docs_rag.db import corpus_stats
from docs_rag.evals.golden import GoldenItem, in_corpus, load_golden
from docs_rag.evals.metrics import mean_scores, score_ranking
from docs_rag.retrieval import Hit, unique_pages

SearchFn = Callable[[str, int, str], list[Hit]]


def evaluate(
    items: Sequence[GoldenItem], search: SearchFn, variants: Sequence[str], ks: Sequence[int], pool: int
) -> dict[str, Any]:
    answerable = [i for i in items if i.answerable]
    unanswerable = [i for i in items if not i.answerable]
    out: dict[str, Any] = {"variants": {}, "per_question": {}}
    for variant in variants:
        scored: list[tuple[GoldenItem, dict[str, float]]] = []
        latencies: list[float] = []
        per_q: list[dict[str, Any]] = []
        for item in answerable:
            started = time.perf_counter()
            hits = search(item.question, pool, variant)
            latencies.append((time.perf_counter() - started) * 1000)
            ranked = unique_pages(hits)
            scores = score_ranking(ranked, item.expected_pages, ks)
            scored.append((item, scores))
            per_q.append({"id": item.id, "type": item.type, "ranked": ranked[:10], **scores})
        by_type: dict[str, dict[str, float]] = {}
        for qtype in sorted({item.type for item, _ in scored}):
            rows = [s for item, s in scored if item.type == qtype]
            by_type[qtype] = {**mean_scores(rows), "n": len(rows)}
        out["variants"][variant] = {
            "overall": mean_scores([s for _, s in scored]),
            "by_type": by_type,
            "latency_ms": {
                "p50": round(statistics.median(latencies), 1),
                "p95": round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 1),
            },
        }
        out["per_question"][variant] = per_q
    if "vector" in variants and unanswerable:
        # Cosine similarity of the best chunk: the only calibrated score we have
        # to tell "no good source" apart. Reported, not gated.
        def top_sim(item: GoldenItem) -> float:
            hits = search(item.question, 1, "vector")
            return round(hits[0].score, 4) if hits else 0.0

        out["no_answer_signal"] = {
            "vector_top1_similarity_answerable_mean": round(statistics.mean(top_sim(i) for i in answerable), 4),
            "vector_top1_similarity_unanswerable": {i.id: top_sim(i) for i in unanswerable},
        }
    out["n_answerable"] = len(answerable)
    out["n_unanswerable"] = len(unanswerable)
    return out


def check_gate(
    current: dict[str, dict[str, float]],
    main_variant: str,
    n_answerable: int,
    baseline: dict[str, Any] | None,
    thresholds: dict[str, float],
    tolerance: float,
    gated_metrics: Sequence[str],
) -> list[str]:
    """Return human-readable failures; an empty list means the gate passes.

    - absolute floors apply to the main (shipped) variant;
    - the regression check applies to every variant in the baseline, so a broken
      tsvector or index is caught even if the main variant masks it.
    """
    main = current[main_variant]
    failures = [
        f"{main_variant}: {m} = {main[m]:.3f} is below the absolute floor {floor:.3f}"
        for m, floor in thresholds.items()
        if main[m] < floor
    ]
    if baseline is None:
        return [*failures, "no committed baseline: run with --update-baseline and commit it"]
    if baseline["n_answerable"] != n_answerable:
        return [
            *failures,
            f"baseline was computed on {baseline['n_answerable']} questions, now {n_answerable}: "
            "the golden set or corpus changed, refresh the baseline deliberately (--update-baseline)",
        ]
    for variant, base_metrics in baseline["variants"].items():
        if variant not in current:
            failures.append(f"{variant}: present in the baseline but not evaluated")
            continue
        for m in gated_metrics:
            now, base = current[variant][m], base_metrics[m]
            if now < base - tolerance:
                failures.append(f"{variant}: {m} regressed {now:.3f} < baseline {base:.3f} - tolerance {tolerance:.3f}")
    return failures


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def render_markdown(report: dict[str, Any], main_variant: str, items: Sequence[GoldenItem]) -> str:
    variants = report["variants"]
    metrics = list(next(iter(variants.values()))["overall"])
    lines = [
        f"# Retrieval eval: `{report['corpus']}` corpus",
        "",
        f"- Corpus: {report['pages']} pages / {report['chunks']} chunks, upstream commit `{report['corpus_sha'][:12]}`",
        f"- Golden set: {report['n_answerable']} answerable questions scored "
        f"(+{report['n_unanswerable']} unanswerable, reported separately)",
        f"- Embeddings: `{report['embedding_model']}`; generated {report['generated_at']}",
        "",
        "## Overall (page level)",
        "",
        "| variant | " + " | ".join(metrics) + " | p50 ms | p95 ms |",
        "|---" * (len(metrics) + 3) + "|",
    ]
    for name, v in variants.items():
        label = f"**{name}** (gated)" if name == main_variant else name
        lat = v["latency_ms"]
        lines.append(
            f"| {label} | " + " | ".join(_fmt(v["overall"][m]) for m in metrics) + f" | {lat['p50']} | {lat['p95']} |"
        )

    types = sorted({t for v in variants.values() for t in v["by_type"]})
    lines += ["", "## recall@5 by question type", "", "| type | n | " + " | ".join(variants) + " |"]
    lines.append("|---" * (len(variants) + 2) + "|")
    for t in types:
        n = next(v["by_type"][t]["n"] for v in variants.values() if t in v["by_type"])
        cells = [_fmt(v["by_type"][t]["recall@5"]) if t in v["by_type"] else "-" for v in variants.values()]
        lines.append(f"| {t} | {n} | " + " | ".join(cells) + " |")

    if signal := report.get("no_answer_signal"):
        lines += [
            "",
            "## Unanswerable questions (not gated)",
            "",
            "Top-1 cosine similarity of the vector retriever. A usable no-answer threshold needs the "
            "unanswerable values to sit clearly below the answerable mean.",
            "",
            f"- answerable mean: {signal['vector_top1_similarity_answerable_mean']}",
        ]
        lines += [f"- {qid}: {sim}" for qid, sim in signal["vector_top1_similarity_unanswerable"].items()]

    by_id = {i.id: i for i in items}
    misses = [q for q in report["per_question"][main_variant] if q["recall@10"] < 1.0]
    lines += ["", f"## `{main_variant}` misses (recall@10 < 1): {len(misses)}", ""]
    for q in misses:
        item = by_id[q["id"]]
        lines.append(f"- **{q['id']}** ({q['type']}) {item.question}")
        lines.append(f"  - expected: {', '.join(item.expected_pages)}; got top 3: {', '.join(q['ranked'][:3])}")
    return "\n".join(lines) + "\n"


def run(corpus: str, settings: Settings, update_baseline: bool = False) -> int:
    from docs_rag.service import open_retriever

    cfg = settings.eval
    golden = load_golden(settings.path(cfg.golden))
    with open_retriever(corpus, settings) as retriever:
        pages, n_chunks = corpus_stats(retriever.conn, corpus)
        if not pages:
            print(f"corpus {corpus!r} is empty: run `make ingest{'-fixture' if corpus == 'fixture' else ''}` first")
            return 2
        items = in_corpus(golden, pages)
        report = evaluate(
            items, lambda q, n, mode: retriever.search(q, n, mode), cfg.variants, cfg.ks, settings.retrieval.candidates
        )
    report.update(
        corpus=corpus,
        corpus_sha=settings.corpus.sha,
        embedding_model=settings.embedding.model,
        pages=len(pages),
        chunks=n_chunks,
        generated_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
    )

    reports_dir = settings.path(cfg.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"retrieval_{corpus}.json").write_text(json.dumps(report, indent=2) + "\n")
    markdown = render_markdown(report, settings.retrieval.default_mode, items)
    (reports_dir / f"retrieval_{corpus}.md").write_text(markdown)
    print(markdown.split("## recall@5")[0])

    baseline_path = settings.path(cfg.baselines_dir) / f"retrieval_{corpus}.json"
    current = {name: v["overall"] for name, v in report["variants"].items()}
    if update_baseline:
        new_baseline = {
            "corpus": corpus,
            "main_variant": settings.retrieval.default_mode,
            "n_answerable": report["n_answerable"],
            "variants": {name: {k: round(x, 4) for k, x in m.items()} for name, m in current.items()},
        }
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(new_baseline, indent=2) + "\n")
        print(f"baseline written to {baseline_path}")

    baseline: dict[str, Any] | None = json.loads(baseline_path.read_text()) if baseline_path.exists() else None
    failures = check_gate(
        current,
        settings.retrieval.default_mode,
        report["n_answerable"],
        baseline,
        cfg.thresholds.get(corpus, {}),
        cfg.tolerance,
        cfg.gated_metrics,
    )
    if failures:
        print(f"EVAL GATE FAILED ({corpus}, variant={settings.retrieval.default_mode}):", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print(f"eval gate passed ({corpus}, variant={settings.retrieval.default_mode})")
    return 0


def main(corpus: str, update_baseline: bool = False) -> int:
    return run(corpus, load_settings(), update_baseline)
