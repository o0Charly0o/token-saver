#!/usr/bin/env python3
"""
MCP server exposing search_context and index_project tools for Claude Code.
"""
import os
import subprocess
import sys
from pathlib import Path

# allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server.fastmcp import FastMCP
from common import collection_name
from retriever.search import search

mcp = FastMCP("tokens-saver")

DEFAULT_COLLECTION = os.getenv("DEFAULT_COLLECTION", "")


def _resolve_collection(project_path: str | None, collection: str | None) -> str:
    if collection:
        return collection
    if project_path:
        return collection_name(project_path)
    if DEFAULT_COLLECTION:
        return DEFAULT_COLLECTION
    raise ValueError("Provide collection or project_path")


@mcp.tool()
def search_context(query: str, project_path: str = "", collection: str = "", n_results: int = 5) -> str:
    """
    Search the indexed project for chunks relevant to the query.
    Returns formatted context ready to include in a Claude prompt.
    Use this instead of reading full files to save tokens.

    Args:
        query: What you're looking for (natural language or code keywords)
        project_path: Absolute path to the project root (used to infer collection name)
        collection: Qdrant collection name (overrides project_path)
        n_results: Number of chunks to return (default 5)
    """
    col = _resolve_collection(project_path or None, collection or None)
    results = search(query, col, n_results=n_results)

    if not results:
        return f"No results found in collection '{col}' for: {query}"

    lines = [f"# Context for: {query}\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"## [{i}] {r['path']}:{r['line']}  (score: {r['score']:.3f})")
        lines.append("```")
        lines.append(r["text"].strip())
        lines.append("```\n")
    return "\n".join(lines)


@mcp.tool()
def index_project(project_path: str, collection: str = "") -> str:
    """
    Index or re-index a project directory into Qdrant + BM25.
    Run this after adding or changing files in the project.

    Args:
        project_path: Absolute path to the project root
        collection: Collection name (defaults to directory name)
    """
    indexer = Path(__file__).parent.parent / "indexer" / "index.py"
    python = Path(__file__).parent.parent / ".venv" / "bin" / "python3"
    if not python.exists():
        python = Path(sys.executable)

    cmd = [str(python), str(indexer), project_path]
    if collection:
        cmd += ["--collection", collection]

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)
    output = result.stdout + result.stderr
    if result.returncode != 0:
        return f"ERROR (exit {result.returncode}):\n{output}"
    # per-batch progress lines are noise once finished
    return "\n".join(l for l in output.splitlines() if "%" not in l)


if __name__ == "__main__":
    mcp.run(transport="stdio")
