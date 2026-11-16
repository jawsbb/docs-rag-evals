"""stdio MCP server exposing `search_docs` over the ingested Supabase docs.

Run with `uv run docs-rag mcp`. DOCS_RAG_CORPUS selects the corpus (default
"full"), DATABASE_URL the Postgres instance.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from mcp.server.mcpserver import MCPServer

from docs_rag.retrieval import MODES, Hit

SearchFn = Callable[[str, int, str], list[Hit]]
MAX_K = 20


def create_server(search: SearchFn, default_mode: str = "vector") -> MCPServer:
    server = MCPServer(
        "supabase-docs-search",
        instructions="Search the Supabase guides (Database, Auth, Storage, Edge Functions). "
        "Cite the returned `url` when you use a result.",
    )

    @server.tool()
    def search_docs(query: str, k: int = 5, mode: str = "") -> list[dict[str, Any]]:
        """Search the Supabase documentation and return the most relevant chunks.

        Args:
            query: natural-language question or keywords.
            k: number of chunks to return (1-20).
            mode: "vector" (pgvector cosine), "fts" (Postgres full-text), "hybrid" (both fused with RRF)
                or "hybrid_rerank" (hybrid + cross-encoder, ~1 s on CPU). Empty = server default.
        """
        mode = mode or default_mode
        if mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}")
        hits = search(query, max(1, min(k, MAX_K)), mode)
        return [
            {
                "chunk_id": h.chunk_id,
                "title": h.title,
                "section": h.heading_path,
                "url": h.url,
                "score": round(h.score, 4),
                "content": h.content,
            }
            for h in hits
        ]

    return server


def main() -> None:
    from docs_rag.service import open_retriever

    with open_retriever(os.environ.get("DOCS_RAG_CORPUS", "full")) as retriever:
        server = create_server(lambda q, k, mode: retriever.search(q, k, mode), retriever.cfg.default_mode)
        server.run("stdio")
