"""Wiring: build a Retriever from settings (shared by the CLI, evals and MCP server)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from docs_rag import db
from docs_rag.config import Settings, load_settings
from docs_rag.embeddings import FastEmbedEmbedder, FastEmbedReranker
from docs_rag.retrieval import Retriever


@contextmanager
def open_retriever(corpus: str, settings: Settings | None = None) -> Iterator[Retriever]:
    settings = settings or load_settings()
    with db.connect(settings.database_url) as conn:
        yield Retriever(
            conn,
            corpus,
            FastEmbedEmbedder(settings.embedding),
            settings.retrieval,
            reranker=FastEmbedReranker(settings.retrieval.reranker_model),
        )
