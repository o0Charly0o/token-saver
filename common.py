"""
Settings and helpers shared by the indexer, the retriever and the MCP server.
"""
import hashlib
import os
import re
import uuid
from pathlib import Path

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
CACHE_DIR = Path(os.getenv("TOKENS_SAVER_CACHE", Path.home() / ".cache" / "tokens-saver"))

# multilingual defaults: queries and specs are often in Spanish, code in English
EMBEDDING_MODEL = os.getenv("TS_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
RERANKER_MODEL = os.getenv("TS_RERANKER_MODEL", "jinaai/jina-reranker-v2-base-multilingual")

_POINT_NAMESPACE = uuid.UUID("6f1c2a4e-3b7d-4c1e-9a55-0d2f8e7b1c90")


def collection_name(project_path: str | Path) -> str:
    """Directory name plus a short hash of the absolute path, so two projects
    with the same folder name on one machine don't share a collection."""
    root = Path(project_path).expanduser().resolve()
    slug = re.sub(r"[^a-z0-9_-]+", "_", root.name.lower()).strip("_") or "project"
    return f"{slug}_{hashlib.sha1(str(root).encode()).hexdigest()[:8]}"


def point_id(path: str, chunk_idx: int) -> str:
    """128-bit id, shared by Qdrant and the BM25 index."""
    return str(uuid.uuid5(_POINT_NAMESPACE, f"{path}:{chunk_idx}"))


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def bm25_path(collection: str) -> Path:
    return CACHE_DIR / f"{collection}_bm25.pkl"


def state_path(collection: str) -> Path:
    return CACHE_DIR / f"{collection}_state.json"
