"""Optional answer-quality evals with Claude. Skipped cleanly without ANTHROPIC_API_KEY.

Pipeline per golden item: retrieve k chunks -> Claude answers with [chunk-id]
citations -> deterministic checks (cited ids exist in the retrieved set,
abstains iff the question is unanswerable) -> LLM judge scores the answer
against a written rubric. Token usage and estimated cost are reported.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from docs_rag.config import AnswersConfig, Settings, load_settings
from docs_rag.db import corpus_stats
from docs_rag.evals.golden import GoldenItem, in_corpus, load_golden
from docs_rag.retrieval import Hit

IDK = "I don't know based on the provided documentation."
_CITATION = re.compile(r"\[([\w./-]+#\d+)\]")

ANSWER_SYSTEM = f"""You answer questions about the Supabase documentation using ONLY the sources provided.
Rules:
- Every factual sentence must cite at least one source by its id in square brackets,
  e.g. [storage/uploads/s3-uploads#2].
- Cite only ids that appear in the sources. Never invent ids, URLs or facts.
- If the sources do not contain the answer, reply exactly: "{IDK}" and nothing else.
- Be concise: at most 5 sentences."""

JUDGE_RUBRIC = """You are grading an answer produced by a documentation assistant.
You get the question, the sources the assistant was given, a reference answer written by a human,
and the assistant's answer. Score each criterion from 1 (bad) to 5 (perfect):

- faithfulness: every claim in the answer is supported by the cited sources; nothing is invented.
  An "I don't know" answer is fully faithful.
- citation_support: each cited source actually contains the claim it is attached to.
  Use 5 for an "I don't know" answer that cites nothing.
- correctness: the answer agrees with the reference answer on the facts that matter.
  If the reference says the documentation does not cover the topic, 5 means the assistant declined,
  1 means it produced an answer anyway.

Reply with JSON only: {"faithfulness": n, "citation_support": n, "correctness": n, "rationale": "one sentence"}"""


JUDGE_CRITERIA = ("faithfulness", "citation_support", "correctness")

# `anthropic.Anthropic().messages.create`, or a fake in tests.
CreateMessage = Callable[..., Any]


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, response: Any) -> None:
        self.input_tokens += response.usage.input_tokens
        self.output_tokens += response.usage.output_tokens

    def cost_usd(self, cfg: AnswersConfig) -> float:
        return (self.input_tokens * cfg.input_price_per_mtok + self.output_tokens * cfg.output_price_per_mtok) / 1e6


@dataclass
class AnswerResult:
    id: str
    type: str
    answer: str
    retrieved: list[str]
    cited: list[str]
    checks: dict[str, bool]
    judge: dict[str, Any] = field(default_factory=dict)


def format_sources(hits: Sequence[Hit]) -> str:
    return "\n\n".join(
        f'<source id="{h.chunk_id}" url="{h.url}">\n{h.heading_path}\n\n{h.content}\n</source>' for h in hits
    )


def _text(response: Any) -> str:
    return "".join(block.text for block in response.content if getattr(block, "type", "") == "text").strip()


def deterministic_checks(item: GoldenItem, answer: str, retrieved: Sequence[Hit]) -> tuple[list[str], dict[str, bool]]:
    cited = list(dict.fromkeys(_CITATION.findall(answer)))
    retrieved_ids = {h.chunk_id for h in retrieved}
    abstained = IDK.lower().rstrip(".") in answer.lower()
    page_of = {h.chunk_id: h.page_id for h in retrieved}
    checks = {
        "citations_exist_in_retrieved": all(c in retrieved_ids for c in cited),
        "abstention_correct": abstained == (not item.answerable),
        "has_citation_when_answering": abstained or bool(cited),
        "cites_expected_page": (not item.answerable) or any(page_of.get(c) in item.expected_pages for c in cited),
    }
    return cited, checks


def parse_judge(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"error": "judge returned no JSON"}
    try:
        data: dict[str, Any] = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"error": "judge returned invalid JSON"}
    if not all(isinstance(data.get(c), int | float) and 1 <= data[c] <= 5 for c in JUDGE_CRITERIA):
        return {"error": f"judge JSON lacks 1-5 scores for {JUDGE_CRITERIA}", "raw": data}
    return data


def answer_one(
    item: GoldenItem, hits: Sequence[Hit], create: CreateMessage, cfg: AnswersConfig, usage: Usage
) -> AnswerResult:
    sources = format_sources(hits)
    response = create(
        model=cfg.model,
        max_tokens=cfg.max_tokens,
        system=ANSWER_SYSTEM,
        messages=[{"role": "user", "content": f"<sources>\n{sources}\n</sources>\n\nQuestion: {item.question}"}],
    )
    usage.add(response)
    answer = _text(response)
    cited, checks = deterministic_checks(item, answer, hits)

    judge_input = (
        f"<question>{item.question}</question>\n<sources>\n{sources}\n</sources>\n"
        f"<reference_answer>{item.reference_answer}</reference_answer>\n<answer>{answer}</answer>"
    )
    verdict = create(
        model=cfg.judge_model,
        max_tokens=300,
        system=JUDGE_RUBRIC,
        messages=[{"role": "user", "content": judge_input}],
    )
    usage.add(verdict)
    return AnswerResult(
        item.id, item.type, answer, [h.chunk_id for h in hits], cited, checks, parse_judge(_text(verdict))
    )


def summarize(results: Sequence[AnswerResult], usage: Usage, cfg: AnswersConfig) -> dict[str, Any]:
    n = len(results)
    check_names = list(results[0].checks) if results else []
    judged = [r.judge for r in results if "error" not in r.judge]
    return {
        "n": n,
        "model": cfg.model,
        "judge_model": cfg.judge_model,
        "checks_pass_rate": {c: sum(r.checks[c] for r in results) / n for c in check_names} if n else {},
        "judge_mean": {c: sum(float(j[c]) for j in judged) / len(judged) for c in JUDGE_CRITERIA} if judged else {},
        "judge_errors": n - len(judged),
        "usage": {**asdict(usage), "estimated_cost_usd": round(usage.cost_usd(cfg), 4)},
    }


def run_answer_eval(
    items: Sequence[GoldenItem],
    retrieve: Callable[[str], list[Hit]],
    create: CreateMessage,
    cfg: AnswersConfig,
) -> dict[str, Any]:
    usage = Usage()
    results = [answer_one(item, retrieve(item.question), create, cfg, usage) for item in items]
    return {"summary": summarize(results, usage, cfg), "results": [asdict(r) for r in results]}


def render_markdown(report: dict[str, Any], corpus: str) -> str:
    s = report["summary"]
    lines = [
        f"# Answer eval: `{corpus}` corpus",
        "",
        f"- {s['n']} questions, answer model `{s['model']}`, judge `{s['judge_model']}`",
    ]
    u = s["usage"]
    lines.append(
        f"- tokens: {u['input_tokens']} in / {u['output_tokens']} out, estimated cost ${u['estimated_cost_usd']}"
    )
    lines += ["", "| deterministic check | pass rate |", "|---|---|"]
    lines += [f"| {k} | {v:.3f} |" for k, v in s["checks_pass_rate"].items()]
    lines += ["", "| judge criterion (1-5) | mean |", "|---|---|"]
    lines += [f"| {k} | {v:.2f} |" for k, v in s["judge_mean"].items()]
    return "\n".join(lines) + "\n"


def main(corpus: str, limit: int | None = None, settings: Settings | None = None) -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set: skipping answer evals (retrieval evals do not need it).")
        return 0
    import anthropic

    from docs_rag.service import open_retriever

    settings = settings or load_settings()
    cfg = settings.answers
    items = load_golden(settings.path(settings.eval.golden))
    with open_retriever(corpus, settings) as retriever:
        pages, _ = corpus_stats(retriever.conn, corpus)
        items = in_corpus(items, pages)[:limit]
        report = run_answer_eval(
            items, lambda q: retriever.search(q, cfg.k), anthropic.Anthropic().messages.create, cfg
        )
    out = settings.path(settings.eval.reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"answers_{corpus}.json").write_text(json.dumps(report, indent=2) + "\n")
    markdown = render_markdown(report, corpus)
    (out / f"answers_{corpus}.md").write_text(markdown)
    print(markdown, file=sys.stdout)
    return 0
