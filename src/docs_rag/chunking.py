"""Heading-aware chunking.

A page is split at markdown headings (outside code fences). Sections shorter
than `min_words` are carried into the next section so we do not index
fragments like a lone "## Overview" line. Sections longer than `max_words` are
packed paragraph by paragraph into windows, and each new window starts with the
trailing paragraphs of the previous one (up to `overlap_words`), so a fact that
straddles a window boundary is still retrievable from at least one chunk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from docs_rag.config import ChunkingConfig
from docs_rag.mdx import CleanDoc

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_EXPLICIT_ANCHOR = re.compile(r"\s*\[#([\w-]+)\]\s*$")


@dataclass(frozen=True)
class Chunk:
    id: str
    page_id: str
    ordinal: int
    title: str
    heading_path: str
    url: str
    content: str

    @property
    def embedding_text(self) -> str:
        """Contextual header + content: the heading path disambiguates short sections."""
        return f"{self.heading_path}\n\n{self.content}"


@dataclass
class _Section:
    path: list[str]
    anchor: str
    lines: list[str]

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip()


def slugify(heading: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", heading.lower().replace("`", ""))
    return re.sub(r"[\s_]+", "-", slug).strip("-")


def _word_count(text: str) -> int:
    return len(text.split())


def split_sections(doc: CleanDoc) -> list[_Section]:
    stack: list[tuple[int, str]] = []
    sections = [_Section(path=[doc.title], anchor="", lines=[])]
    in_code = False
    for line in doc.body.splitlines():
        if line.startswith("```"):
            in_code = not in_code
        match = None if in_code else _HEADING.match(line)
        if not match:
            sections[-1].lines.append(line)
            continue
        level, raw = len(match.group(1)), match.group(2)
        explicit = _EXPLICIT_ANCHOR.search(raw)
        heading = _EXPLICIT_ANCHOR.sub("", raw).strip()
        anchor = explicit.group(1) if explicit else slugify(heading)
        stack = [(lvl, h) for lvl, h in stack if lvl < level] + [(level, heading)]
        path = [doc.title] + [h for _, h in stack]
        sections.append(_Section(path=path, anchor=anchor, lines=[f"{match.group(1)} {heading}"]))
    return sections


def _paragraphs(text: str) -> list[str]:
    """Blank-line separated blocks, never splitting inside a code fence."""
    paras: list[str] = []
    buf: list[str] = []
    in_code = False
    for line in text.splitlines():
        if line.startswith("```"):
            in_code = not in_code
        if not line.strip() and not in_code:
            if buf:
                paras.append("\n".join(buf))
                buf = []
            continue
        buf.append(line)
    if buf:
        paras.append("\n".join(buf))
    return paras


def _split_long(para: str, max_words: int) -> list[str]:
    """Split an oversized block (big table or code sample) at line boundaries."""
    pieces: list[str] = []
    current: list[str] = []
    for line in para.splitlines():
        words = line.split()
        if len(words) > max_words:  # a single giant line: fall back to word windows
            pieces.extend(" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words))
            continue
        if current and _word_count("\n".join([*current, line])) > max_words:
            pieces.append("\n".join(current))
            current = []
        current.append(line)
    if current:
        pieces.append("\n".join(current))
    return pieces


def window(text: str, max_words: int, overlap_words: int) -> list[str]:
    if _word_count(text) <= max_words:
        return [text]
    paras = [p for para in _paragraphs(text) for p in _split_long(para, max_words)]
    windows: list[str] = []
    current: list[str] = []
    for para in paras:
        if current and _word_count("\n\n".join([*current, para])) > max_words:
            windows.append("\n\n".join(current))
            overlap: list[str] = []
            for prev in reversed(current):
                if _word_count("\n\n".join([prev, *overlap])) > overlap_words:
                    break
                overlap.insert(0, prev)
            current = overlap
        current.append(para)
    if current:
        windows.append("\n\n".join(current))
    return windows


def chunk_page(page_id: str, doc: CleanDoc, base_url: str, cfg: ChunkingConfig) -> list[Chunk]:
    page_url = f"{base_url.rstrip('/')}/{page_id}"
    title = doc.title or page_id
    merged: list[_Section] = []
    carry: _Section | None = None
    sections = split_sections(CleanDoc(title=title, description=doc.description, body=doc.body))
    if doc.description:
        sections[0].lines.insert(0, doc.description + "\n")
    for section in sections:
        if not section.text:
            continue
        if carry is not None:
            section = _Section(path=carry.path, anchor=carry.anchor, lines=[*carry.lines, "", *section.lines])
            carry = None
        if _word_count(section.text) < cfg.min_words:
            carry = section
        else:
            merged.append(section)
    if carry is not None:
        if merged:
            merged[-1].lines += ["", *carry.lines]
        else:
            merged.append(carry)

    chunks: list[Chunk] = []
    for section in merged:
        url = f"{page_url}#{section.anchor}" if section.anchor else page_url
        for piece in window(section.text, cfg.max_words, cfg.overlap_words):
            chunks.append(
                Chunk(
                    id=f"{page_id}#{len(chunks)}",
                    page_id=page_id,
                    ordinal=len(chunks),
                    title=title,
                    heading_path=" > ".join(section.path),
                    url=url,
                    content=piece,
                )
            )
    return chunks
