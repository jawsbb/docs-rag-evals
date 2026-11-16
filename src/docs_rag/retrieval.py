"""Four retrieval variants over one Postgres table.

- fts:    Postgres full-text search, OR-ed query terms ranked with ts_rank_cd.
- vector: pgvector HNSW, cosine distance on bge-small embeddings.
- hybrid: Reciprocal Rank Fusion of the two candidate lists.
- hybrid_rerank: hybrid candidates re-scored by a local cross-encoder.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal, get_args

import psycopg
from psycopg import sql

from docs_rag.config import RetrievalConfig
from docs_rag.db import schema_name
from docs_rag.embeddings import Embedder, Reranker

Mode = Literal["fts", "vector", "hybrid", "hybrid_rerank"]
MODES: tuple[str, ...] = get_args(Mode)

_COLUMNS = "id, page_id, title, heading_path, url, content"

# plainto_tsquery AND-s every term, which almost never matches a full natural
# language question. Turning `&` into `|` gives a recall-oriented OR query;
# ts_rank_cd then rewards chunks that cover more (and denser) query terms.
# Normalization 1 divides by 1 + log(length) so long chunks do not win by size.
_FTS_SQL = """
WITH q AS (SELECT replace(plainto_tsquery('english', %(query)s)::text, '&', '|')::tsquery AS q)
SELECT {columns}, ts_rank_cd(tsv, q.q, 1) AS score
FROM {schema}.chunks, q
WHERE tsv @@ q.q
ORDER BY score DESC, id
LIMIT %(n)s
"""

_VECTOR_SQL = """
SELECT {columns}, 1 - (embedding <=> %(vec)s) AS score
FROM {schema}.chunks
ORDER BY embedding <=> %(vec)s
LIMIT %(n)s
"""


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    page_id: str
    title: str
    heading_path: str
    url: str
    content: str
    score: float


def rrf(rankings: Sequence[Sequence[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank(d)), rank from 1."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def unique_pages(hits: Sequence[Hit]) -> list[str]:
    """Page ids in first-seen order: the ranking used for page-level metrics."""
    return list(dict.fromkeys(h.page_id for h in hits))


class Retriever:
    def __init__(
        self,
        conn: psycopg.Connection,
        corpus: str,
        embedder: Embedder,
        cfg: RetrievalConfig,
        reranker: Reranker | None = None,
    ):
        self.conn = conn
        self.schema = schema_name(corpus)
        self.embedder = embedder
        self.reranker = reranker
        self.cfg = cfg
        conn.execute(sql.SQL("SET hnsw.ef_search = {}").format(sql.Literal(cfg.hnsw_ef_search)))

    def _run(self, template: str, params: dict[str, object]) -> list[Hit]:
        query = sql.SQL(template).format(columns=sql.SQL(_COLUMNS), schema=self.schema)
        rows = self.conn.execute(query, params).fetchall()
        return [Hit(row[0], row[1], row[2], row[3], row[4], row[5], score=float(row[6])) for row in rows]

    def fts(self, query: str, n: int) -> list[Hit]:
        return self._run(_FTS_SQL, {"query": query, "n": n})

    def vector(self, query: str, n: int) -> list[Hit]:
        return self._run(_VECTOR_SQL, {"vec": self.embedder.embed_query(query), "n": n})

    def hybrid(self, query: str, n: int) -> list[Hit]:
        pool = self.cfg.candidates
        lexical, semantic = self.fts(query, pool), self.vector(query, pool)
        by_id = {h.chunk_id: h for h in [*semantic, *lexical]}
        fused = rrf([[h.chunk_id for h in lexical], [h.chunk_id for h in semantic]], k=self.cfg.rrf_k)
        return [replace(by_id[cid], score=score) for cid, score in fused[:n]]

    def hybrid_rerank(self, query: str, n: int) -> list[Hit]:
        if self.reranker is None:
            raise RuntimeError("hybrid_rerank needs a reranker")
        # Only the top `rerank_candidates` go through the cross-encoder (the expensive
        # part); any further hits keep their hybrid order below the reranked block.
        fused = self.hybrid(query, max(n, self.cfg.rerank_candidates))
        head, tail = fused[: self.cfg.rerank_candidates], fused[self.cfg.rerank_candidates :]
        scores = self.reranker.score(query, [f"{h.heading_path}\n\n{h.content}" for h in head])
        ranked = [replace(h, score=s) for h, s in sorted(zip(head, scores, strict=True), key=lambda p: -p[1])]
        floor = min(scores, default=0.0)
        return (ranked + [replace(h, score=floor - 1 - i) for i, h in enumerate(tail)])[:n]

    def search(self, query: str, k: int = 5, mode: str | None = None) -> list[Hit]:
        mode = mode or self.cfg.default_mode
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        method = getattr(self, mode)
        hits: list[Hit] = method(query, k)
        return hits
