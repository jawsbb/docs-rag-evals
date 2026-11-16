"""SQL-level tests against a real pgvector Postgres. Skipped when none is reachable.

A hashed bag-of-words embedder replaces the ONNX model so the test is fast and
exercises the SQL (tsvector, HNSW, COPY, RRF wiring) rather than the model.
"""

import hashlib
import os
import re
from collections.abc import Iterator, Sequence

import numpy as np
import psycopg
import pytest

from docs_rag import db
from docs_rag.chunking import Chunk
from docs_rag.config import DEFAULT_DATABASE_URL, load_settings
from docs_rag.retrieval import Retriever

CORPUS = "pytest_corpus"
DIM = 64


class HashEmbedder:
    dim = DIM

    def _vec(self, text: str) -> np.ndarray:
        vec = np.zeros(DIM, dtype=np.float32)
        for word in re.findall(r"[a-z]+", text.lower()):
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
        return vec / (np.linalg.norm(vec) or 1.0)

    def embed_documents(self, texts: Sequence[str]) -> list[np.ndarray]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


class LengthReranker:
    """Prefers shorter documents: easy to assert that reranking changed the order."""

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        return [-float(len(d)) for d in documents]


def _chunk(page: str, i: int, text: str) -> Chunk:
    return Chunk(f"{page}#{i}", page, i, page.title(), page.title(), f"https://example.test/{page}", text)


CHUNKS = [
    _chunk("uploads", 0, "Resumable uploads use the TUS protocol for large files over unreliable networks."),
    _chunk("uploads", 1, "Standard uploads send the whole file in one multipart request, best for small files."),
    _chunk("sessions", 0, "Refresh tokens can be reused within a reuse interval of ten seconds."),
    _chunk("timeouts", 0, "The anon role has a statement_timeout of three seconds; authenticated has eight."),
]


@pytest.fixture(scope="module")
def conn() -> Iterator[psycopg.Connection]:
    url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    try:
        connection = db.connect(url)
    except psycopg.OperationalError:
        pytest.skip(f"no Postgres at {url}")
    embedder = HashEmbedder()
    db.load_corpus(connection, CORPUS, DIM, CHUNKS, embedder.embed_documents([c.embedding_text for c in CHUNKS]))
    yield connection
    connection.execute(f"DROP SCHEMA {CORPUS} CASCADE")
    connection.close()


@pytest.fixture
def retriever(conn: psycopg.Connection) -> Retriever:
    return Retriever(conn, CORPUS, HashEmbedder(), load_settings().retrieval, reranker=LengthReranker())


@pytest.mark.db
def test_corpus_stats(conn: psycopg.Connection):
    assert db.corpus_stats(conn, CORPUS) == ({"uploads", "sessions", "timeouts"}, 4)
    assert db.corpus_stats(conn, "never_ingested") == (set(), 0)


@pytest.mark.db
def test_fts_uses_or_semantics_and_stemming(retriever: Retriever):
    hits = retriever.fts("what is the statement_timeout for anon users", 3)
    assert hits[0].chunk_id == "timeouts#0"
    # "uploading" stems to "upload"; OR-query still matches with extra words around it.
    assert {h.page_id for h in retriever.fts("uploading huge files", 5)} == {"uploads"}
    assert retriever.fts("the of and", 5) == []  # stop words only


@pytest.mark.db
def test_vector_and_hybrid_rank_the_relevant_chunk_first(retriever: Retriever):
    query = "reuse interval for refresh tokens"
    assert retriever.vector(query, 1)[0].chunk_id == "sessions#0"
    hybrid = retriever.hybrid(query, 4)
    assert hybrid[0].chunk_id == "sessions#0"
    assert hybrid[0].score > hybrid[-1].score


@pytest.mark.db
def test_rerank_reorders_candidates_and_search_validates_mode(retriever: Retriever):
    reranked = retriever.search("uploads files", k=2, mode="hybrid_rerank")
    assert [len(h.content) for h in reranked] == sorted(len(h.content) for h in reranked)
    with pytest.raises(ValueError, match="mode must be one of"):
        retriever.search("x", mode="bm25")
