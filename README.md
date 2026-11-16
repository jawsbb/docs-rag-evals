# docs-rag-evals

Retrieval-augmented search over the Supabase docs, built on **Postgres + pgvector**, with an
**evaluation harness that gates CI**, and exposed to agents as an **MCP tool**.

The point is the measurement, not the demo. Four retrieval variants (Postgres full-text,
pgvector, hybrid with Reciprocal Rank Fusion, and hybrid plus a cross-encoder reranker) are
scored on a 70-question golden set. A regression gate runs on every push. The results changed
the design: the shipped default is **vector-only**, because equal-weight hybrid search did
*worse* on this corpus (see [Results](#results)).

- Corpus: 239 Supabase guide pages (Database, Auth, Storage, Edge Functions) at a pinned commit.
- Embeddings: `BAAI/bge-small-en-v1.5` (384 dims), run locally with fastembed/ONNX. No API key.
- Retrieval evals: recall@{1,3,5,10}, MRR and nDCG@10, overall and per question type. Deterministic.
- Answer evals (optional): Claude answers with citations, then deterministic checks and an LLM judge.
- MCP server: `search_docs(query, k, mode)` over stdio.

## Architecture

```mermaid
flowchart LR
    subgraph ingest["ingest (make ingest / ingest-fixture)"]
        GH["supabase/supabase<br/>@ pinned SHA"] -->|raw .mdx + partials| CLEAN["MDX cleaner<br/>(JSX, comments, partials)"]
        FIX["data/fixture<br/>24 vendored pages"] --> CLEAN
        CLEAN --> CHUNK["heading-aware chunker<br/>300 words, 50 overlap"]
        CHUNK --> EMB["bge-small-en-v1.5<br/>(fastembed, local)"]
    end
    EMB -->|COPY| PG[("Postgres 17 + pgvector<br/>schema per corpus<br/>tsvector GIN + HNSW")]
    PG --> R{"Retriever"}
    R -->|fts| R1["ts_rank_cd over OR-query"]
    R -->|vector| R2["HNSW cosine"]
    R -->|hybrid| R3["RRF(fts, vector)"]
    R -->|hybrid_rerank| R4["RRF + MiniLM cross-encoder"]
    R --> EV["retrieval evals<br/>+ regression gate"]
    R --> MCP["MCP server<br/>search_docs"]
    R --> ANS["answer evals<br/>(Claude, optional)"]
    GOLD["golden.jsonl<br/>70 questions"] --> EV
    GOLD --> ANS
```

## Quickstart

Requires Python 3.12, [uv](https://docs.astral.sh/uv/) and Docker.

```bash
uv sync
make up                # Postgres 17 + pgvector on localhost:5433
make ingest-fixture    # 24 vendored pages, offline, ~15 s
make eval-retrieval    # runs the 4 variants on the fixture; non-zero exit on regression
make ingest            # full pinned corpus: downloads 239 pages, ~2 min on CPU
make eval-retrieval-full
make test              # unit tests; DB tests run when Postgres is reachable
uv run docs-rag search "how long is a resumable upload URL valid?" --mode hybrid_rerank
```

The first run downloads the ONNX models (about 150 MB) into `.cache/fastembed`. Set
`DATABASE_URL` to use another Postgres.

### Use it from Claude Code / Claude Desktop (MCP)

`make ingest` first. Then for Claude Code:

```bash
claude mcp add supabase-docs -e DOCS_RAG_CORPUS=full -- \
  uv run --directory /absolute/path/to/docs-rag-evals docs-rag mcp
```

For Claude Desktop, in `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "supabase-docs": {
      "command": "uv",
      "args": ["run", "--directory", "/absolute/path/to/docs-rag-evals", "docs-rag", "mcp"],
      "env": { "DOCS_RAG_CORPUS": "full", "DATABASE_URL": "postgresql://rag:rag@localhost:5433/rag" }
    }
  }
}
```

The tool returns chunks with `title`, `section`, `url` (with the heading anchor), `score` and `content`.
`mode` is optional and defaults to `retrieval.default_mode` from `config/settings.toml`.

## Results

These are the real numbers from local runs on 2026-11-16 (Intel i5-14600KF, CPU only,
pgvector 0.8.7; latency varies a few percent between runs). The full reports, with per-question rankings, are in `results/`. Metrics are
**page-level**: the chunk ranking is collapsed to unique pages before scoring. A multi-hop
question only gets full recall when both of its pages are retrieved.

**Full pinned corpus**: 239 pages, 1,672 chunks, 64 answerable questions.

| variant | recall@1 | recall@3 | recall@5 | recall@10 | MRR | nDCG@10 | p50 latency |
|---|---|---|---|---|---|---|---|
| fts | 0.273 | 0.453 | 0.578 | 0.688 | 0.420 | 0.473 | 15 ms |
| **vector** (shipped, gated) | 0.734 | 0.875 | 0.922 | 0.961 | 0.859 | 0.872 | 10 ms |
| hybrid (RRF) | 0.477 | 0.656 | 0.812 | 0.938 | 0.646 | 0.701 | 29 ms |
| hybrid_rerank | 0.742 | 0.906 | 0.938 | 0.945 | 0.875 | 0.878 | 1,079 ms |

**Fixture**: 24 pages, 197 chunks, 44 answerable questions. This is what CI runs.

| variant | recall@1 | recall@3 | recall@5 | recall@10 | MRR | nDCG@10 | p50 latency |
|---|---|---|---|---|---|---|---|
| fts | 0.545 | 0.727 | 0.841 | 0.977 | 0.711 | 0.764 | 3 ms |
| **vector** (shipped, gated) | 0.830 | 0.977 | 0.977 | 1.000 | 0.936 | 0.953 | 6 ms |
| hybrid (RRF) | 0.682 | 0.943 | 1.000 | 1.000 | 0.846 | 0.876 | 9 ms |
| hybrid_rerank | 0.932 | 0.977 | 1.000 | 1.000 | 1.000 | 0.993 | 1,117 ms |

Recall@5 by question type on the full corpus:

| type | n | fts | vector | hybrid | hybrid_rerank |
|---|---|---|---|---|---|
| lookup | 14 | 0.857 | 1.000 | 1.000 | 1.000 |
| how_to | 14 | 0.643 | 1.000 | 0.786 | 1.000 |
| keyword | 12 | 0.417 | 0.833 | 0.833 | 0.833 |
| paraphrase | 15 | 0.467 | 0.867 | 0.733 | 0.933 |
| multi_hop | 9 | 0.444 | 0.889 | 0.667 | 0.889 |

What the numbers say:

- **Postgres FTS is weak here, even on keyword questions (0.42 recall@5).** `ts_rank_cd` has no
  inverse document frequency. A natural-language question turned into an OR-query ranks chunks
  that repeat common words ("Supabase", "function", "storage") above the one chunk that has
  the rare identifier. Dropping very common query terms is an obvious fix, but any cutoff would
  be tuned on the eval set itself, so it is not part of the shipped variants.
- **Equal-weight RRF lets that noise in.** Hybrid sits between the two lists it fuses. On 239
  pages it loses 0.21 MRR against vector-only. On the fixture (fewer distractors) the gap is
  smaller, and hybrid even wins recall@5. Fusion only helps when both lists are reasonably good.
- **A cross-encoder recovers the ranking but costs about 100x the latency on CPU.** It is the
  best variant on recall@1/MRR, but the gain over vector-only on the full corpus (+0.016 MRR,
  about 1 question in 64) is within noise. It is available as `mode="hybrid_rerank"`, not the
  default.
- **No-answer detection by score threshold does not work.** For the 6 unanswerable questions,
  the top-1 cosine similarity is 0.71–0.85. The mean for answerable questions is 0.78. Saying
  "I don't know" has to happen at generation time (the answer evals check for it), not by
  thresholding retrieval.
- Sample sizes are small (64 and 44 questions). Treat differences under about 0.03–0.05 as noise.

Answer evals: implemented and unit-tested with a mocked client, **but not run**. No API key was
used to build this repo, so there are no answer-quality numbers here.

## The eval gate

`make eval-retrieval` (and the CI `eval-gate` job) ingests the fixture, evaluates all four
variants, writes `results/retrieval_fixture.{md,json}`, and exits non-zero if:

1. the shipped variant (`retrieval.default_mode`, currently `vector`) drops below an absolute
   floor: recall@5 ≥ 0.90, recall@10 ≥ 0.95, MRR ≥ 0.85, nDCG@10 ≥ 0.88 on the fixture
   (full corpus: 0.85 / 0.90 / 0.78 / 0.80); **or**
2. **any** variant loses more than 0.03 (absolute) on recall@5, recall@10, MRR or nDCG@10 against
   the committed baseline in `evals/baselines/`; **or**
3. the number of evaluable questions differs from the baseline. A changed golden set or
   corpus needs a deliberate `make baseline`, committed and visible in review.

Gating every variant matters. As a check, I set the chunk size to 40 words: `vector` stayed
above its floors, but the gate failed on fts (recall@10 0.977 → 0.750), hybrid and
hybrid_rerank. A main-variant-only gate would have let that through.

Re-ingesting the fixture from scratch reproduced identical metrics (the HNSW build and ONNX
inference were deterministic in my runs). The 0.03 tolerance absorbs one question flipping on
the 44-question fixture.

## Golden set

`data/golden/golden.jsonl` has 70 items, validated by `data/golden/golden.schema.json`
(generated from the pydantic model; a test checks they stay in sync).

| type | count | what it tests |
|---|---|---|
| lookup | 14 | a specific fact (a limit, a default) |
| how_to | 14 | procedural "how do I …" |
| keyword | 12 | exact identifiers: `cacheNonce`, `42P17`, `SB_EXECUTION_ID`, `prepareThreshold=0` … |
| paraphrase | 15 | the page's vocabulary deliberately avoided |
| multi_hop | 9 | needs two specific pages |
| unanswerable | 6 | realistic near-misses the corpus does not answer (expected: no good source) |

Each item has `expected_pages`, `expected_sections`, a short `reference_answer`, and `notes`
(including how its uniqueness was checked). Of the 64 answerable items, 44 are answerable
from the fixture pages; the rest need the full corpus.

**How it was written:** with AI assistance (Claude), by reading the cleaned pages and grepping
the whole corpus so each fact is answered by the listed page(s) and not elsewhere. Automated
checks: schema, page existence against the pinned manifest (`data/corpus_pages.txt`), and
consistency per type. A sample of facts was cross-checked against the source text. It is a
small, single-author set, so expect a few debatable items (see `notes`).

## Design decisions and trade-offs

- **One Postgres, two retrieval paths.** `tsvector` (title A, heading path B, body D) with GIN,
  and `vector(384)` with HNSW, in the same table. No extra search service, and fusion happens
  in one round trip per retriever.
- **One schema per corpus** (`"full"`, `fixture`) rather than a `corpus` column. A filtered HNSW
  scan is post-filtered and can return fewer than k rows (pgvector's iterative scans help, but
  separate tables avoid the problem).
- **OR-ed `plainto_tsquery`.** The default AND semantics almost never match a full question.
- **Heading-aware chunks with contextual headers.** Pages are split at headings outside code
  fences. Sections under 40 words are merged forward, and long sections are packed by paragraph
  with about 50 words of overlap. The embedded text is prefixed with `Title > Section > Sub`.
  Chunk URLs carry the heading anchor (the docs' explicit `[#anchor]` syntax is respected).
- **MDX cleaning keeps content, drops markup.** It unwraps `<Admonition>`/`<Tabs>`, inlines
  `<$Partial>` includes from the same commit, drops `<$CodeSample>`/`<Image>`, and keeps
  code blocks (identifiers matter for keyword search). Tags are parsed with a small scanner
  (quotes and `{…}` aware), not a regex, because JSX props contain `>`.
- **Page-level metrics.** Chunk ids change whenever chunking changes. Pages are stable, so the
  golden set survives chunking experiments, and those experiments are exactly what the gate
  should judge.
- **Pinned and reproducible.** The commit SHA, include/exclude globs and model ids live in
  `config/settings.toml`. Downloads are cached per SHA. Index builds happen after a bulk `COPY`,
  in one transaction.
- **Answer evals are separate and optional.** Retrieval evals need no key and run on every push.
  Answer evals cost money, so they run only on `main`/manual runs, and only if the
  `ANTHROPIC_API_KEY` secret exists. Deterministic checks (cited ids must be in the retrieved
  set, abstain if and only if the question is unanswerable, cite an expected page) sit
  alongside an LLM judge with a written rubric (`src/docs_rag/evals/answer_eval.py`). Token
  usage and estimated cost are reported. The prices in config are placeholders to set.

## Limitations

- Raw MDX is not the rendered site. Values injected at build time (e.g. some auth rate-limit
  defaults rendered from shared data files) appear as placeholders such as
  `auth.rate_limits.token_refresh.requests_burst`. `<$CodeSample>` files are not fetched.
- The golden set is small and was written by one author with AI assistance. 70 questions
  cannot separate variants that are a couple of points apart.
- The corpus scope is a choice: provider-specific social-login pages, framework quickstarts and
  example apps are excluded (near-duplicates). The search therefore does not know about them.
- No IDF-aware lexical ranking (BM25) in plain Postgres. Extensions like `pg_search` would be
  the next experiment, together with weighted RRF.
- Latencies are single-process CPU timings, not a load test.

## Repository layout

```
config/settings.toml        corpus SHA, chunking, models, gate thresholds
src/docs_rag/               mdx.py, chunking.py, corpus.py, db.py, embeddings.py,
                            retrieval.py, ingest.py, service.py, mcp_server.py, cli.py
src/docs_rag/evals/         golden.py, metrics.py, retrieval_eval.py, answer_eval.py
data/golden/                golden.jsonl + golden.schema.json
data/fixture/               24 upstream pages (Apache-2.0) + NOTICE + license
data/corpus_pages.txt       manifest of the 239 pages at the pinned commit
evals/baselines/            committed baselines the gate compares against
results/                    reports behind the numbers above
tests/                      unit tests + DB integration tests + MCP in-process tests
```

## License

Code: MIT, see `LICENSE`. Documentation content: the Supabase docs are © Supabase, licensed
under Apache-2.0 (the `supabase/supabase` repository license, which covers `apps/docs`).
The vendored copies in `data/fixture/pages/` are unmodified and come with
`data/fixture/NOTICE.md` and the license text. The full corpus is downloaded at ingest time,
not redistributed.
