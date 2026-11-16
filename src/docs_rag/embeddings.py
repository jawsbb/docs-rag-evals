"""Local ONNX models via fastembed: no API key, deterministic on CPU."""

from __future__ import annotations

import os
from collections.abc import Sequence
from functools import cached_property
from typing import Protocol

import numpy as np

from docs_rag.config import ROOT, EmbeddingConfig


def model_cache_dir() -> str:
    return os.environ.get("FASTEMBED_CACHE_PATH", str(ROOT / ".cache" / "fastembed"))


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> list[np.ndarray]: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class Reranker(Protocol):
    def score(self, query: str, documents: Sequence[str]) -> list[float]: ...


class FastEmbedEmbedder:
    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg
        self.dim = cfg.dim

    @cached_property
    def _model(self):  # type: ignore[no-untyped-def]
        from fastembed import TextEmbedding

        return TextEmbedding(self.cfg.model, cache_dir=model_cache_dir())

    def embed_documents(self, texts: Sequence[str]) -> list[np.ndarray]:
        return list(self._model.embed(list(texts), batch_size=32))

    def embed_query(self, text: str) -> np.ndarray:
        vec: np.ndarray = next(iter(self._model.embed([self.cfg.query_prefix + text])))
        return vec


class FastEmbedReranker:
    def __init__(self, model: str):
        self.model_name = model

    @cached_property
    def _model(self):  # type: ignore[no-untyped-def]
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        return TextCrossEncoder(self.model_name, cache_dir=model_cache_dir())

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        return [float(s) for s in self._model.rerank(query, list(documents))]
