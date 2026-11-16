"""Typed access to config/settings.toml. DATABASE_URL comes from the environment."""

from __future__ import annotations

import os
import tomllib
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE_URL = "postgresql://rag:rag@localhost:5433/rag"


class CorpusConfig(BaseModel):
    repo: str
    sha: str
    docs_root: str
    base_url: str
    include: list[str]
    exclude: list[str]
    cache_dir: Path
    fixture_dir: Path


class ChunkingConfig(BaseModel):
    max_words: int = 300
    overlap_words: int = 50
    min_words: int = 40


class EmbeddingConfig(BaseModel):
    model: str
    dim: int
    query_prefix: str = ""


class RetrievalConfig(BaseModel):
    default_mode: str = "vector"
    candidates: int = 50
    rrf_k: int = 60
    hnsw_ef_search: int = 100
    reranker_model: str
    rerank_candidates: int = 30


class EvalConfig(BaseModel):
    golden: Path
    ks: list[int]
    variants: list[str]
    reports_dir: Path
    baselines_dir: Path
    tolerance: float
    gated_metrics: list[str]
    thresholds: dict[str, dict[str, float]] = Field(default_factory=dict)


class AnswersConfig(BaseModel):
    model: str
    judge_model: str
    k: int = 6
    max_tokens: int = 700
    input_price_per_mtok: float
    output_price_per_mtok: float


class Settings(BaseModel):
    corpus: CorpusConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    eval: EvalConfig
    answers: AnswersConfig

    @property
    def database_url(self) -> str:
        return os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)

    def path(self, relative: Path) -> Path:
        return relative if relative.is_absolute() else ROOT / relative


@lru_cache(maxsize=1)
def load_settings(path: Path | None = None) -> Settings:
    with open(path or ROOT / "config" / "settings.toml", "rb") as fh:
        return Settings.model_validate(tomllib.load(fh))
