"""Turn Supabase-flavoured MDX into plain markdown suitable for chunking.

What is removed: frontmatter (parsed into metadata), `{/* comments */}`, JSX/HTML
tags (their children are kept, e.g. the text inside `<Admonition>`), self-closing
components that pull content from elsewhere (`<$CodeSample/>`, `<Image/>`), and
top-level `import`/`export` statements. `<$Partial path="..."/>` includes are
inlined through a resolver callback. Fenced code blocks are kept verbatim
(dedented) because identifiers in code are valuable for keyword search.
"""

from __future__ import annotations

import re
import textwrap
from collections.abc import Callable
from dataclasses import dataclass

PartialResolver = Callable[[str], str | None]

_FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
_JSX_COMMENT = re.compile(r"\{/\*.*?\*/\}", re.DOTALL)
_MODULE_LINE = re.compile(r"^(import|export)\s.*$", re.MULTILINE)
_PARTIAL_PATH = re.compile(r"""path=["']([^"']+)["']""")
_SPACES = re.compile(r"[ \t]{2,}")
_TABLE_RULE = re.compile(r"^\|[\s|:-]+\|$")
_FENCE = re.compile(r"^(\s*)(`{3,}|~{3,})")
_HTML_TAGS = {
    "a", "br", "details", "div", "em", "figure", "figcaption", "hr", "iframe", "img",
    "kbd", "li", "ol", "p", "pre", "source", "span", "strong", "sub", "summary", "sup",
    "table", "tbody", "td", "th", "thead", "tr", "u", "ul", "video",
}  # fmt: skip
_MAX_PARTIAL_DEPTH = 3


@dataclass(frozen=True)
class CleanDoc:
    title: str
    description: str
    body: str


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Parse the flat `key: value` frontmatter used by the docs (no PyYAML needed)."""
    match = _FRONTMATTER.match(text)
    if not match:
        return {}, text
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        key, sep, value = line.partition(":")
        if sep and key and not key.startswith((" ", "-")):
            meta[key.strip()] = value.strip().strip("'\"")
    return meta, text[match.end() :]


def _tag_end(text: str, start: int) -> int:
    """Index just past the `>` closing the tag that opens at `start`.

    Attribute values may contain `>` inside quotes or `{...}` expressions, so a
    plain regex is not enough.
    """
    depth, quote, i = 0, "", start + 1
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == quote:
                quote = ""
        elif ch in "\"'`":
            quote = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif ch == ">" and depth <= 0:
            return i + 1
        i += 1
    return -1


def _tag_name(text: str, start: int) -> str:
    match = re.match(r"</?([$A-Za-z][\w.$-]*)", text[start:])
    return match.group(1) if match else ""


def _is_markup_tag(name: str) -> bool:
    # PascalCase / $Component = JSX; ALL_CAPS like <SUPABASE_URL> is a placeholder.
    if name.startswith("$"):
        return True
    if name[:1].isupper():
        return any(c.islower() for c in name)
    return name.lower() in _HTML_TAGS


def _strip_tags(prose: str, resolve_partial: PartialResolver | None, depth: int) -> str:
    out: list[str] = []
    i = 0
    while i < len(prose):
        ch = prose[i]
        if ch == "`":  # copy inline code spans untouched
            end = prose.find("`", i + 1)
            if end == -1 or "\n\n" in prose[i:end]:
                out.append(ch)
                i += 1
                continue
            out.append(prose[i : end + 1])
            i = end + 1
            continue
        if ch == "<":
            name = _tag_name(prose, i)
            if name and _is_markup_tag(name):
                end = _tag_end(prose, i)
                if end != -1:
                    tag = prose[i:end]
                    if name == "$Partial" and resolve_partial and depth < _MAX_PARTIAL_DEPTH:
                        path = _PARTIAL_PATH.search(tag)
                        partial = resolve_partial(path.group(1)) if path else None
                        if partial:
                            _, partial_body = parse_frontmatter(partial)
                            out.append("\n" + _clean_body(partial_body, resolve_partial, depth + 1) + "\n")
                    elif name in {"br", "p", "div", "li", "tr"} or tag.endswith("/>"):
                        out.append("\n")
                    i = end
                    continue
        out.append(ch)
        i += 1
    return "".join(out)


def _clean_prose(prose: str, resolve_partial: PartialResolver | None, depth: int) -> str:
    prose = _JSX_COMMENT.sub("", prose)
    prose = _MODULE_LINE.sub("", prose)
    prose = _strip_tags(prose, resolve_partial, depth)
    lines = [_SPACES.sub(" ", line.strip()) for line in prose.splitlines()]
    return "\n".join(line for line in lines if not _TABLE_RULE.match(line))


def _split_fences(body: str) -> list[tuple[bool, str]]:
    """Split into (is_code, text) segments. Fences may be indented (inside JSX)."""
    segments: list[tuple[bool, str]] = []
    buf: list[str] = []
    fence = ""
    for line in body.splitlines():
        match = _FENCE.match(line)
        if not fence and match:
            segments.append((False, "\n".join(buf)))
            buf, fence = [line], match.group(2)
        elif fence and line.strip().startswith(fence) and line.strip().strip(fence[0]) == "":
            buf.append(line)
            segments.append((True, "\n".join(buf)))
            buf, fence = [], ""
        else:
            buf.append(line)
    segments.append((bool(fence), "\n".join(buf)))
    return segments


def _clean_code(block: str) -> str:
    lines = textwrap.dedent(block).splitlines()
    opener = re.match(r"(`{3,}|~{3,})\s*([\w+-]*)", lines[0])
    lines[0] = f"```{opener.group(2)}" if opener else "```"
    if len(lines) > 1 and _FENCE.match(lines[-1]):
        lines[-1] = "```"
    else:
        lines.append("```")
    return "\n".join(lines)


def _clean_body(body: str, resolve_partial: PartialResolver | None, depth: int = 0) -> str:
    parts = [
        _clean_code(text) if is_code else _clean_prose(text, resolve_partial, depth)
        for is_code, text in _split_fences(body)
    ]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(parts)).strip()


def clean_mdx(text: str, resolve_partial: PartialResolver | None = None) -> CleanDoc:
    meta, body = parse_frontmatter(text)
    return CleanDoc(
        title=meta.get("title", ""),
        description=meta.get("subtitle") or meta.get("description", ""),
        body=_clean_body(body, resolve_partial),
    )
