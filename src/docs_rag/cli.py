"""`docs-rag` command line: ingest, search, eval-retrieval, eval-answers, mcp."""

from __future__ import annotations

import argparse
import logging
import sys

from docs_rag.config import load_settings
from docs_rag.retrieval import MODES


def _cmd_ingest(args: argparse.Namespace) -> int:
    from docs_rag.embeddings import FastEmbedEmbedder
    from docs_rag.ingest import ingest

    settings = load_settings()
    stats = ingest(args.corpus, settings, FastEmbedEmbedder(settings.embedding))
    print(f"ingested corpus={stats.corpus} pages={stats.pages} chunks={stats.chunks} in {stats.seconds}s")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    from docs_rag.service import open_retriever

    with open_retriever(args.corpus) as retriever:
        for hit in retriever.search(args.query, k=args.k, mode=args.mode):
            print(f"{hit.score:.4f}  {hit.chunk_id}\n        {hit.heading_path}\n        {hit.url}")
    return 0


def _cmd_eval_retrieval(args: argparse.Namespace) -> int:
    from docs_rag.evals.retrieval_eval import main as run

    return run(corpus=args.corpus, update_baseline=args.update_baseline)


def _cmd_eval_answers(args: argparse.Namespace) -> int:
    from docs_rag.evals.answer_eval import main as run

    return run(corpus=args.corpus, limit=args.limit)


def _cmd_mcp(_: argparse.Namespace) -> int:
    from docs_rag.mcp_server import main as run

    run()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="docs-rag")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="download/clean/chunk/embed a corpus into Postgres")
    p.add_argument("--corpus", choices=["full", "fixture"], default="full")
    p.set_defaults(func=_cmd_ingest)

    p = sub.add_parser("search", help="ad-hoc query")
    p.add_argument("query")
    p.add_argument("--corpus", choices=["full", "fixture"], default="full")
    p.add_argument("--mode", choices=MODES, default=None, help="default: retrieval.default_mode")
    p.add_argument("-k", type=int, default=5)
    p.set_defaults(func=_cmd_search)

    p = sub.add_parser("eval-retrieval", help="run retrieval metrics and the regression gate")
    p.add_argument("--corpus", choices=["full", "fixture"], default="fixture")
    p.add_argument("--update-baseline", action="store_true", help="overwrite the committed baseline")
    p.set_defaults(func=_cmd_eval_retrieval)

    p = sub.add_parser("eval-answers", help="LLM answer evals (needs ANTHROPIC_API_KEY)")
    p.add_argument("--corpus", choices=["full", "fixture"], default="fixture")
    p.add_argument("--limit", type=int, default=None, help="only the first N questions")
    p.set_defaults(func=_cmd_eval_answers)

    p = sub.add_parser("mcp", help="run the stdio MCP server")
    p.set_defaults(func=_cmd_mcp)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    code: int = args.func(args)
    return code


if __name__ == "__main__":
    sys.exit(main())
