import json
from types import SimpleNamespace
from typing import Any

import pytest

from docs_rag.config import load_settings
from docs_rag.evals import answer_eval
from docs_rag.evals.answer_eval import IDK, deterministic_checks, parse_judge, run_answer_eval
from docs_rag.evals.golden import GoldenItem
from docs_rag.retrieval import Hit

HITS = [
    Hit(
        "storage/uploads/file-limits#0", "storage/uploads/file-limits", "File limits", "File limits", "u", "50 MB", 1.0
    ),
    Hit("auth/sessions#2", "auth/sessions", "Sessions", "Sessions", "u", "10 seconds", 0.9),
]
ANSWERABLE = GoldenItem(
    id="q001",
    question="What is the max upload size on Free?",
    type="lookup",
    expected_pages=["storage/uploads/file-limits"],
    reference_answer="50 MB",
)
UNANSWERABLE = GoldenItem(
    id="q002", question="How much does SOC2 cost?", type="unanswerable", expected_pages=[], reference_answer="n/a"
)


def _response(text: str, tokens_in: int = 100, tokens_out: int = 20) -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=tokens_in, output_tokens=tokens_out),
    )


class FakeClaude:
    """Stands in for `anthropic.Anthropic().messages.create`; records calls."""

    def __init__(self, answers: dict[str, str], judge: str):
        self.answers, self.judge, self.calls = answers, judge, []

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        prompt = kwargs["messages"][0]["content"]
        if kwargs["system"] == answer_eval.JUDGE_RUBRIC:
            return _response(self.judge, 300, 40)
        question = prompt.rsplit("Question: ", 1)[1]
        return _response(self.answers[question])


def test_checks_flag_hallucinated_citations():
    cited, checks = deterministic_checks(ANSWERABLE, "Up to 50 MB [storage/uploads/file-limits#0] [made/up#9].", HITS)
    assert cited == ["storage/uploads/file-limits#0", "made/up#9"]
    assert checks["citations_exist_in_retrieved"] is False
    assert checks["cites_expected_page"] is True
    assert checks["abstention_correct"] is True


def test_checks_reward_abstaining_on_unanswerable():
    _, checks = deterministic_checks(UNANSWERABLE, IDK, HITS)
    assert all(checks.values())
    _, checks = deterministic_checks(ANSWERABLE, IDK, HITS)
    assert checks["abstention_correct"] is False


def test_parse_judge_validates_scores():
    assert (
        parse_judge('Sure: {"faithfulness": 5, "citation_support": 4, "correctness": 5, "rationale": "ok"}')[
            "citation_support"
        ]
        == 4
    )
    assert "error" in parse_judge("no json here")
    assert "error" in parse_judge('{"faithfulness": 9, "citation_support": 4, "correctness": 5}')


def test_run_answer_eval_with_mocked_client_reports_usage_and_cost():
    cfg = load_settings().answers
    judge = json.dumps({"faithfulness": 5, "citation_support": 4, "correctness": 3, "rationale": "fine"})
    fake = FakeClaude(
        {ANSWERABLE.question: "Free projects: 50 MB [storage/uploads/file-limits#0].", UNANSWERABLE.question: IDK},
        judge,
    )
    report = run_answer_eval([ANSWERABLE, UNANSWERABLE], lambda q: HITS, fake, cfg)

    summary = report["summary"]
    assert summary["n"] == 2
    assert summary["checks_pass_rate"]["abstention_correct"] == 1.0
    assert summary["checks_pass_rate"]["citations_exist_in_retrieved"] == 1.0
    assert summary["judge_mean"] == {"faithfulness": 5.0, "citation_support": 4.0, "correctness": 3.0}
    assert summary["usage"]["input_tokens"] == 2 * 100 + 2 * 300
    assert summary["usage"]["output_tokens"] == 2 * 20 + 2 * 40
    expected_cost = (800 * cfg.input_price_per_mtok + 120 * cfg.output_price_per_mtok) / 1e6
    assert summary["usage"]["estimated_cost_usd"] == round(expected_cost, 4)
    # The model id comes from config and sources are passed with their chunk ids.
    assert fake.calls[0]["model"] == cfg.model == "claude-sonnet-5-5"
    assert '<source id="auth/sessions#2"' in fake.calls[0]["messages"][0]["content"]


def test_main_skips_cleanly_without_api_key(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert answer_eval.main(corpus="fixture") == 0
    assert "skipping answer evals" in capsys.readouterr().out
