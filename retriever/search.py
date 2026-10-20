"""
Hybrid retrieval: BM25 + Qdrant semantic search, fused with RRF, re-ranked.
"""
import pickle
import sys
from pathlib import Path

from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from common import EMBEDDING_MODEL, QDRANT_URL, RERANKER_MODEL, bm25_path, tokenize

RRF_K = 60

_embedder = None
_reranker = None
_client = None
_bm25_cache: dict[str, tuple[float, dict]] = {}


def _get_client():
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL)
    return _client


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(model_name=EMBEDDING_MODEL)
    return _embedder


def _get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = TextCrossEncoder(model_name=RERANKER_MODEL)
    return _reranker


def _load_bm25(collection: str) -> dict:
    p = bm25_path(collection)
    if not p.exists():
        raise FileNotFoundError(f"BM25 index not found: {p}. Run indexer first.")
    mtime = p.stat().st_mtime
    cached = _bm25_cache.get(collection)
    if cached is None or cached[0] != mtime:
        with open(p, "rb") as fh:
            _bm25_cache[collection] = (mtime, pickle.load(fh))
    return _bm25_cache[collection][1]


def _rrf(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


def search(query: str, collection: str, n_results: int = 5, top_k: int = 20) -> list[dict]:
    """
    Returns up to n_results chunks most relevant to query.
    Each result: {text, path, line, score}
    """
    client = _get_client()
    embedder = _get_embedder()

    # --- semantic search ---
    query_emb = next(iter(embedder.query_embed(query))).tolist()
    sem_hits = client.query_points(
        collection_name=collection,
        query=query_emb,
        limit=top_k,
        with_payload=True,
    ).points
    sem_ids = [str(h.id) for h in sem_hits]
    sem_payloads = {str(h.id): h.payload for h in sem_hits}

    # --- BM25 search (same ids as Qdrant, so RRF merges hits from both) ---
    data = _load_bm25(collection)
    bm25_ids: list[str] = []
    bm25_payloads: dict[str, dict] = {}
    if data["bm25"] is not None:
        scores = data["bm25"].get_scores(tokenize(query))
        top_bm25_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        top_bm25_idx = [i for i in top_bm25_idx if scores[i] > 0]
        bm25_ids = [data["ids"][i] for i in top_bm25_idx]
        bm25_payloads = {data["ids"][i]: data["payloads"][i] for i in top_bm25_idx}

    # --- RRF fusion ---
    rrf_scores = _rrf([sem_ids, bm25_ids])
    all_payloads: dict[str, dict] = {**bm25_payloads, **sem_payloads}

    # take top candidates for re-ranking
    candidates = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:top_k]
    candidate_payloads = [all_payloads[c] for c in candidates if c in all_payloads]

    if not candidate_payloads:
        return []

    # --- re-rank ---
    reranker = _get_reranker()
    rerank_scores = list(reranker.rerank(query, [p["text"] for p in candidate_payloads]))

    ranked = sorted(zip(rerank_scores, candidate_payloads), key=lambda x: x[0], reverse=True)

    return [
        {"text": p["text"], "path": p["path"], "line": p.get("line", 0), "score": float(s)}
        for s, p in ranked[:n_results]
    ]
