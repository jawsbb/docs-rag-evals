"""Offline checks on the committed data: golden set, schema, fixture, manifest."""

from collections import Counter
from pathlib import Path

from docs_rag.chunking import chunk_page
from docs_rag.config import ROOT, load_settings
from docs_rag.corpus import DirectorySource, select_paths
from docs_rag.evals.golden import in_corpus, json_schema, load_golden
from docs_rag.mdx import clean_mdx

SETTINGS = load_settings()
GOLDEN = load_golden(SETTINGS.path(SETTINGS.eval.golden))
MANIFEST = {
    line for line in (ROOT / "data" / "corpus_pages.txt").read_text().splitlines() if line and not line.startswith("#")
}
FIXTURE = DirectorySource(SETTINGS.path(SETTINGS.corpus.fixture_dir))


def test_golden_size_and_composition():
    counts = Counter(item.type for item in GOLDEN)
    assert 50 <= len(GOLDEN) <= 80
    assert counts["unanswerable"] >= 5
    assert counts["multi_hop"] >= 5
    assert set(counts) == {"lookup", "how_to", "keyword", "paraphrase", "multi_hop", "unanswerable"}


def test_committed_schema_matches_model():
    assert (ROOT / "data" / "golden" / "golden.schema.json").read_text() == json_schema()


def test_expected_pages_exist_in_pinned_corpus():
    missing = {p for item in GOLDEN for p in item.expected_pages} - MANIFEST
    assert not missing


def test_fixture_is_a_subset_of_the_pinned_corpus_and_covers_most_questions():
    fixture_pages = {page.page_id for page in FIXTURE.pages()}
    assert 15 <= len(fixture_pages) <= 25
    assert fixture_pages <= MANIFEST
    assert len([i for i in in_corpus(GOLDEN, fixture_pages) if i.answerable]) >= 40


def test_manifest_matches_config_selection():
    # Every manifest entry must be selected by the include/exclude globs.
    paths = [f"{p}.mdx" for p in MANIFEST]
    assert len(select_paths(paths, SETTINGS.corpus.include, SETTINGS.corpus.exclude)) == len(MANIFEST)
    assert select_paths(["auth/social-sign-in/auth-github.mdx", "auth/sessions.mdx", "cron.mdx"],
                        SETTINGS.corpus.include, SETTINGS.corpus.exclude) == ["auth/sessions.mdx"]  # fmt: skip


def test_fixture_pages_clean_and_chunk_without_mdx_leftovers():
    for page in FIXTURE.pages():
        doc = clean_mdx(page.raw, FIXTURE.partial)
        assert doc.title, page.page_id
        chunks = chunk_page(page.page_id, doc, SETTINGS.corpus.base_url, SETTINGS.chunking)
        assert chunks, page.page_id
        prose = "\n".join(c.content for c in chunks).split("```")[::2]  # outside code fences
        for leftover in ("<Admonition", "<TabPanel", "<$Partial", "{/*"):
            assert not any(leftover in part for part in prose), (page.page_id, leftover)


def test_fixture_partials_are_vendored():
    # auth/rate-limits pulls its endpoint table from a partial.
    page = next(p for p in FIXTURE.pages() if p.page_id == "auth/rate-limits")
    assert "/auth/v1/token" not in page.raw
    assert "/auth/v1/token" in clean_mdx(page.raw, FIXTURE.partial).body
    assert Path(SETTINGS.path(SETTINGS.corpus.fixture_dir) / "_partials" / "auth_rate_limits.mdx").exists()
