import pytest
from mcp import Client

from docs_rag.mcp_server import create_server
from docs_rag.retrieval import Hit


def fake_search(query: str, k: int, mode: str) -> list[Hit]:
    hit = Hit(
        chunk_id="storage/uploads/resumable-uploads#0",
        page_id="storage/uploads/resumable-uploads",
        title="Resumable Uploads",
        heading_path="Resumable Uploads",
        url="https://supabase.com/docs/guides/storage/uploads/resumable-uploads",
        content=f"{mode}: {query}",
        score=0.123456,
    )
    return [hit] * k


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_search_docs_tool_over_mcp():
    async with Client(create_server(fake_search)) as client:
        tools = await client.list_tools()
        tool = next(t for t in tools.tools if t.name == "search_docs")
        assert set(tool.input_schema["properties"]) == {"query", "k", "mode"}

        result = await client.call_tool("search_docs", {"query": "resume upload", "k": 2, "mode": "fts"})
        assert not result.is_error
        rows = result.structured_content["result"]
        assert len(rows) == 2
        assert rows[0]["url"].startswith("https://supabase.com/docs/guides/storage/")
        assert rows[0]["content"] == "fts: resume upload"
        assert rows[0]["score"] == 0.1235


@pytest.mark.anyio
async def test_search_docs_rejects_unknown_mode_and_clamps_k():
    async with Client(create_server(fake_search)) as client:
        bad = await client.call_tool("search_docs", {"query": "x", "mode": "magic"})
        assert bad.is_error
        many = await client.call_tool("search_docs", {"query": "x", "k": 500})
        assert len(many.structured_content["result"]) == 20
        assert many.structured_content["result"][0]["content"] == "vector: x"  # server default mode
