"""Download (or read) pages, clean MDX, chunk, embed, load into Postgres."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from docs_rag import db
from docs_rag.chunking import Chunk, chunk_page
from docs_rag.config import Settings
from docs_rag.corpus import DirectorySource, PinnedSource, source_for
from docs_rag.embeddings import Embedder
from docs_rag.mdx import clean_mdx

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestStats:
    corpus: str
    pages: int
    chunks: int
    seconds: float


def build_chunks(source: PinnedSource | DirectorySource, settings: Settings) -> tuple[int, list[Chunk]]:
    pages = source.pages()
    chunks: list[Chunk] = []
    for page in pages:
        doc = clean_mdx(page.raw, resolve_partial=source.partial)
        chunks.extend(chunk_page(page.page_id, doc, settings.corpus.base_url, settings.chunking))
    return len(pages), chunks


def ingest(corpus: str, settings: Settings, embedder: Embedder) -> IngestStats:
    started = time.perf_counter()
    n_pages, chunks = build_chunks(source_for(corpus, settings), settings)
    log.info("%s: %d pages -> %d chunks, embedding...", corpus, n_pages, len(chunks))
    vectors = embedder.embed_documents([c.embedding_text for c in chunks])
    with db.connect(settings.database_url) as conn:
        db.load_corpus(conn, corpus, embedder.dim, chunks, vectors)
    return IngestStats(corpus, n_pages, len(chunks), round(time.perf_counter() - started, 1))
