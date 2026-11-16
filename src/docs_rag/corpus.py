"""Load docs pages: from the pinned upstream commit (full corpus) or the vendored fixture."""

from __future__ import annotations

import fnmatch
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import httpx

from docs_rag.config import CorpusConfig, Settings

RAW_URL = "https://raw.githubusercontent.com/{repo}/{sha}/{path}"
TREE_URL = "https://api.github.com/repos/{repo}/git/trees/{sha}?recursive=1"
PARTIALS_ROOT = "apps/docs/content/_partials"
FIXTURE_PARTIALS = "_partials"


@dataclass(frozen=True)
class Page:
    page_id: str  # path relative to docs_root without extension, e.g. "storage/uploads/s3-uploads"
    raw: str


def select_paths(paths: list[str], include: list[str], exclude: list[str]) -> list[str]:
    def matches(path: str, patterns: list[str]) -> bool:
        return any(fnmatch.fnmatchcase(path, p) for p in patterns)

    return sorted(p for p in paths if matches(p, include) and not matches(p, exclude))


def page_id_for(relative_path: str) -> str:
    return relative_path.removesuffix(".mdx")


class PinnedSource:
    """Downloads files of `repo` at the pinned `sha`, cached on disk (immutable per sha)."""

    def __init__(self, cfg: CorpusConfig, cache_root: Path, client: httpx.Client | None = None):
        self.cfg = cfg
        self.cache = cache_root / cfg.sha
        self.client = client or httpx.Client(timeout=30, follow_redirects=True)

    def _fetch(self, repo_path: str) -> str | None:
        target = self.cache / repo_path
        if target.exists():
            return target.read_text(encoding="utf-8")
        resp = self.client.get(RAW_URL.format(repo=self.cfg.repo, sha=self.cfg.sha, path=repo_path))
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(resp.text, encoding="utf-8")
        return resp.text

    def list_doc_paths(self) -> list[str]:
        listing = self.cache / "tree.txt"
        if not listing.exists():
            resp = self.client.get(TREE_URL.format(repo=self.cfg.repo, sha=self.cfg.sha))
            resp.raise_for_status()
            tree = resp.json()
            if tree.get("truncated"):
                raise RuntimeError("GitHub tree listing truncated; cannot list the corpus reliably")
            prefix = self.cfg.docs_root.rstrip("/") + "/"
            paths = [
                item["path"].removeprefix(prefix)
                for item in tree["tree"]
                if item["type"] == "blob" and item["path"].startswith(prefix) and item["path"].endswith(".mdx")
            ]
            listing.parent.mkdir(parents=True, exist_ok=True)
            listing.write_text("\n".join(sorted(paths)) + "\n", encoding="utf-8")
        return listing.read_text(encoding="utf-8").split()

    def pages(self) -> list[Page]:
        selected = select_paths(self.list_doc_paths(), self.cfg.include, self.cfg.exclude)
        root = self.cfg.docs_root.rstrip("/")
        with ThreadPoolExecutor(max_workers=8) as pool:
            texts = list(pool.map(lambda p: self._fetch(f"{root}/{p}"), selected))
        return [Page(page_id_for(p), t) for p, t in zip(selected, texts, strict=True) if t is not None]

    def partial(self, path: str) -> str | None:
        return self._fetch(f"{PARTIALS_ROOT}/{path}")


class DirectorySource:
    """Pages vendored in a directory tree (the offline fixture)."""

    def __init__(self, root: Path):
        self.root = root

    def pages(self) -> list[Page]:
        files = sorted(p for p in self.root.rglob("*.mdx") if FIXTURE_PARTIALS not in p.parts)
        return [Page(page_id_for(f.relative_to(self.root).as_posix()), f.read_text(encoding="utf-8")) for f in files]

    def partial(self, path: str) -> str | None:
        target = self.root / FIXTURE_PARTIALS / path
        return target.read_text(encoding="utf-8") if target.exists() else None


def source_for(corpus: str, settings: Settings) -> PinnedSource | DirectorySource:
    if corpus == "full":
        return PinnedSource(settings.corpus, settings.path(settings.corpus.cache_dir))
    if corpus == "fixture":
        return DirectorySource(settings.path(settings.corpus.fixture_dir))
    raise ValueError(f"unknown corpus {corpus!r} (expected 'full' or 'fixture')")
