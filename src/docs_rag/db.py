"""Postgres schema and writes. Each corpus lives in its own schema (`full`, `fixture`).

Separate schemas instead of a `corpus` column: an HNSW scan with a WHERE filter
is post-filtered, so a shared index would silently return fewer than k rows
for the smaller corpus.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql

from docs_rag.chunking import Chunk

_SCHEMA_NAME = re.compile(r"^[a-z][a-z0-9_]{0,40}$")

_TABLE_DDL = """
CREATE SCHEMA IF NOT EXISTS {schema};
DROP TABLE IF EXISTS {schema}.chunks;
CREATE TABLE {schema}.chunks (
    id           text PRIMARY KEY,
    page_id      text NOT NULL,
    ordinal      int  NOT NULL,
    title        text NOT NULL,
    heading_path text NOT NULL,
    url          text NOT NULL,
    content      text NOT NULL,
    embedding    vector({dim}) NOT NULL,
    tsv tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', title), 'A') ||
        setweight(to_tsvector('english', heading_path), 'B') ||
        setweight(to_tsvector('english', content), 'D')
    ) STORED
);
"""

# Indexes are built after the bulk load (faster than maintaining them row by row).
_INDEX_DDL = """
CREATE INDEX ON {schema}.chunks USING gin (tsv);
CREATE INDEX ON {schema}.chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON {schema}.chunks (page_id);
ANALYZE {schema}.chunks;
"""


def schema_name(corpus: str) -> sql.Identifier:
    if not _SCHEMA_NAME.match(corpus):
        raise ValueError(f"invalid corpus name {corpus!r}")
    return sql.Identifier(corpus)


def connect(url: str) -> psycopg.Connection:
    conn = psycopg.connect(url, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def load_corpus(
    conn: psycopg.Connection, corpus: str, dim: int, chunks: Sequence[Chunk], vectors: Sequence[np.ndarray]
) -> None:
    """Atomically replace a corpus: readers see either the old or the new table."""
    schema = schema_name(corpus)
    copy = sql.SQL(
        "COPY {schema}.chunks (id, page_id, ordinal, title, heading_path, url, content, embedding) FROM STDIN"
    ).format(schema=schema)
    with conn.transaction(), conn.cursor() as cur:
        cur.execute(sql.SQL(_TABLE_DDL).format(schema=schema, dim=sql.Literal(dim)))
        with cur.copy(copy) as cp:
            cp.set_types(["text", "text", "int4", "text", "text", "text", "text", "vector"])
            for c, v in zip(chunks, vectors, strict=True):
                cp.write_row((c.id, c.page_id, c.ordinal, c.title, c.heading_path, c.url, c.content, v))
        cur.execute(sql.SQL(_INDEX_DDL).format(schema=schema))


def corpus_stats(conn: psycopg.Connection, corpus: str) -> tuple[set[str], int]:
    """(page ids, chunk count) currently loaded for a corpus; empty if never ingested."""
    exists = conn.execute("SELECT to_regclass(%s)", (f"{corpus}.chunks",)).fetchone()
    if not exists or exists[0] is None:
        return set(), 0
    query = sql.SQL("SELECT page_id, count(*) FROM {schema}.chunks GROUP BY page_id").format(schema=schema_name(corpus))
    rows = conn.execute(query).fetchall()
    return {r[0] for r in rows}, sum(r[1] for r in rows)
