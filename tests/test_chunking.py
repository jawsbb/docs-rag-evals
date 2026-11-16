import itertools

from docs_rag.chunking import chunk_page, slugify, split_sections, window
from docs_rag.config import ChunkingConfig
from docs_rag.mdx import CleanDoc

BASE = "https://supabase.com/docs/guides"


def words(n: int, word: str = "lorem") -> str:
    return " ".join([word] * n)


def test_slugify_matches_docs_anchors():
    assert slugify("Access token (JWT) claims") == "access-token-jwt-claims"
    assert slugify("Using `cacheNonce`") == "using-cachenonce"


def test_sections_follow_heading_hierarchy_and_ignore_code_comments():
    body = f"intro\n\n## Setup\n{words(5)}\n\n```bash\n# not a heading\n```\n\n### Install [#custom-anchor]\ntext"
    sections = split_sections(CleanDoc(title="Page", description="", body=body))
    assert [s.path for s in sections] == [["Page"], ["Page", "Setup"], ["Page", "Setup", "Install"]]
    assert sections[2].anchor == "custom-anchor"
    assert "# not a heading" in sections[1].text


def test_chunk_ids_urls_and_small_section_merge():
    body = f"## Overview\nshort\n\n## Details\n{words(60)}\n\n## Tail\ntiny"
    cfg = ChunkingConfig(max_words=300, overlap_words=50, min_words=40)
    chunks = chunk_page("storage/uploads", CleanDoc(title="Uploads", description="", body=body), BASE, cfg)
    assert [c.id for c in chunks] == ["storage/uploads#0"]
    chunk = chunks[0]
    # "Overview" is too short so it is carried into "Details"; "Tail" is appended to the last chunk.
    assert chunk.url == f"{BASE}/storage/uploads#overview"
    assert "short" in chunk.content
    assert "tiny" in chunk.content
    assert chunk.embedding_text.startswith("Uploads > Overview\n\n")


def test_long_sections_are_windowed_with_overlap():
    paras = [words(100, f"p{i}") for i in range(5)]
    pieces = window("\n\n".join(paras), max_words=250, overlap_words=100)
    assert len(pieces) > 1
    assert all(len(p.split()) <= 250 for p in pieces)
    # Each window after the first starts with the last paragraph of the previous one.
    for prev, nxt in itertools.pairwise(pieces):
        assert nxt.split("\n\n")[0] == prev.split("\n\n")[-1]


def test_oversized_block_is_split_on_lines():
    table = "\n".join(f"| row {i} | {words(20)} |" for i in range(30))
    pieces = window(table, max_words=100, overlap_words=0)
    assert all(len(p.split()) <= 100 for p in pieces)
    assert all(line.startswith("| row") for p in pieces for line in p.splitlines())
