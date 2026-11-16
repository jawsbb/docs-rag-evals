.PHONY: up down ingest ingest-fixture eval-retrieval eval-retrieval-full baseline baseline-full eval-answers test lint typecheck mcp

up:  ## start Postgres + pgvector
	docker compose up -d --wait

down:
	docker compose down

ingest:  ## full corpus, downloaded at the pinned commit
	uv run docs-rag ingest --corpus full

ingest-fixture:  ## vendored 24-page subset (offline)
	uv run docs-rag ingest --corpus fixture

eval-retrieval:  ## fixture corpus; non-zero exit on regression (this is the CI gate)
	uv run docs-rag eval-retrieval --corpus fixture

eval-retrieval-full:
	uv run docs-rag eval-retrieval --corpus full

baseline:  ## deliberately refresh the committed fixture baseline
	uv run docs-rag eval-retrieval --corpus fixture --update-baseline

baseline-full:
	uv run docs-rag eval-retrieval --corpus full --update-baseline

eval-answers:  ## needs ANTHROPIC_API_KEY, skips otherwise
	uv run docs-rag eval-answers --corpus fixture

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy

mcp:  ## stdio MCP server (DOCS_RAG_CORPUS=full|fixture)
	uv run docs-rag mcp
