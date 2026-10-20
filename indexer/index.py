#!/usr/bin/env python3
"""
Indexes a project directory into Qdrant (semantic) + BM25 (lexical).
Usage: python indexer/index.py <project_path> [--collection NAME] [--dry-run]
"""
import argparse
import fnmatch
import hashlib
import json
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from common import (CACHE_DIR, EMBEDDING_MODEL, QDRANT_URL, bm25_path,
                    collection_name, point_id, state_path, tokenize)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))   # tokens approx (chars/4)
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
MAX_FILE_BYTES = int(os.getenv("MAX_FILE_BYTES", str(512 * 1024)))
BATCH_SIZE = 64

TEXT_EXTS = {".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".vue", ".svelte",
             ".md", ".txt", ".rst", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
             ".html", ".css", ".scss", ".sass", ".less", ".sh", ".go", ".rs", ".java",
             ".kt", ".rb", ".cs", ".c", ".cpp", ".h", ".graphql", ".feature",
             # PHP / Drupal
             ".php", ".twig", ".module", ".install", ".theme", ".inc", ".profile", ".engine"}

# directories never indexed, wherever they appear
IGNORE_DIRS = {".git", ".venv", "venv", "node_modules", "vendor", "__pycache__", ".mypy_cache",
               ".pytest_cache", ".next", ".nuxt", ".svelte-kit", ".turbo", ".cache",
               "dist", "build", "coverage", "target", ".claude", "contrib"}
# path patterns never indexed (generated or uploaded content)
IGNORE_PATH_RE = re.compile(
    r"(^|/)sites/[^/]+/(files|private)(/|$)"   # Drupal uploads
    r"|(^|/)(web|docroot|html)/(core|libraries)(/|$)"   # Drupal core / libraries
)
# files that may hold credentials or data dumps: never indexed
SECRET_PATTERNS = ["settings*.php", ".env", ".env.*", ".env_*", "*.env", "*.sql", "*.sql.*",
                   "*.dump", "*.key", "*.pem", "*.p12", "*.pfx", "*.crt", "*.jks", "*.kdbx",
                   "id_rsa*", "id_ed25519*", "credentials*", "secrets*", "*.secret",
                   ".npmrc", ".pypirc", ".netrc", "auth.json"]
GENERATED_PATTERNS = ["*.min.js", "*.min.css", "*.map", "*.lock", "package-lock.json",
                      "pnpm-lock.yaml", "yarn.lock"]
# indexed even when the project's .gitignore excludes them
ALWAYS_INCLUDE_DIRS = ["openspec"]
PROJECT_IGNORE_FILE = ".tokens-saver-ignore"

# bump to force a full re-index when chunk content rules change
INDEX_VERSION = 2

# values of credential-like keys are masked before indexing
_SECRET_KEY = r"""['"]?[\w.-]*(?:pass(?:word|wd)?|pwd|secret|token|api[_-]?key|apikey|private[_-]?key|access[_-]?key)[\w.-]*['"]?\]?"""
# quoted literal in any file: 'password' => 'x', "api_key": "x", PASS="x"
QUOTED_SECRET_RE = re.compile(rf"""(?i)({_SECRET_KEY}[ \t]*(?:=>|:|=)[ \t]*)(['"])([^'"\n]+)\2""")
# unquoted scalar in config files only: "password: x", "PASS=x"
CONFIG_SECRET_RE = re.compile(
    rf"""(?im)^([ \t#-]*{_SECRET_KEY}(?::[ \t]+|[ \t]*=[ \t]*))(?!['"{{\[]|null\b|true\b|false\b|~(?!\S))([^\s#]+)"""
)
CONFIG_EXTS = {".yml", ".yaml", ".ini", ".cfg", ".toml", ".conf", ".properties"}


def redact(text: str, path: str = "") -> str:
    text = QUOTED_SECRET_RE.sub(r"\1\2[REDACTED]\2", text)
    if Path(path).suffix.lower() in CONFIG_EXTS:
        text = CONFIG_SECRET_RE.sub(r"\1[REDACTED]", text)
    return text


def log(msg: str):
    print(msg, flush=True)


def chunk_text(text: str, path: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Splits on line boundaries into ~chunk_size-token chunks with ~overlap tokens shared."""
    max_chars = chunk_size * 4
    overlap_chars = overlap * 4
    lines = text.splitlines(keepends=True)
    chunks = []
    start = 0
    while start < len(lines):
        end, size = start, 0
        while end < len(lines) and (size == 0 or size + len(lines[end]) <= max_chars):
            size += len(lines[end])
            end += 1
        snippet = "".join(lines[start:end])
        if snippet.strip():
            chunks.append({"text": snippet, "path": path, "line": start + 1, "chunk_id": len(chunks)})
        if end >= len(lines):
            break
        # step back to share ~overlap_chars with the next chunk, always moving forward
        back, kept = end, 0
        while back - 1 > start and kept + len(lines[back - 1]) <= overlap_chars:
            back -= 1
            kept += len(lines[back])
        start = back
    return chunks


def load_project_ignores(root: Path) -> list[str]:
    p = root / PROJECT_IGNORE_FILE
    if not p.exists():
        return []
    lines = (l.strip() for l in p.read_text(errors="replace").splitlines())
    return [l for l in lines if l and not l.startswith("#")]


def matches_ignore(rel: str, patterns: list[str]) -> bool:
    for pat in patterns:
        pat = pat.lstrip("/")
        if pat.endswith("/"):
            d = pat.rstrip("/")
            if fnmatch.fnmatch(rel, f"{d}/*") or any(fnmatch.fnmatch(part, d) for part in rel.split("/")[:-1]):
                return True
        elif fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(Path(rel).name, pat):
            return True
    return False


def is_indexable(rel: str, extra_ignores: list[str]) -> bool:
    parts = rel.split("/")
    name = parts[-1]
    if any(part in IGNORE_DIRS for part in parts[:-1]):
        return False
    if IGNORE_PATH_RE.search(rel):
        return False
    if any(fnmatch.fnmatch(name, p) for p in SECRET_PATTERNS + GENERATED_PATTERNS):
        return False
    if Path(name).suffix.lower() not in TEXT_EXTS:
        return False
    return not matches_ignore(rel, extra_ignores)


def git_files(root: Path) -> list[str] | None:
    """Tracked + untracked-but-not-ignored files, or None if root is not in a git repo."""
    try:
        out = subprocess.run(["git", "-C", str(root), "ls-files", "-co", "--exclude-standard", "-z"],
                             capture_output=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return [p for p in out.decode(errors="replace").split("\0") if p]


def collect_files(root: Path) -> list[str]:
    """Relative POSIX paths of the files to index."""
    rels = git_files(root)
    if rels is None:
        rels = [p.relative_to(root).as_posix() for p in root.rglob("*")]
    rels = set(rels)
    for d in ALWAYS_INCLUDE_DIRS:
        if (root / d).is_dir():
            rels.update(p.relative_to(root).as_posix() for p in (root / d).rglob("*"))

    extra_ignores = load_project_ignores(root)
    files = []
    for rel in sorted(rels):
        if not is_indexable(rel, extra_ignores):
            continue
        p = root / rel
        if p.is_file() and not p.is_symlink() and p.stat().st_size <= MAX_FILE_BYTES:
            files.append(rel)
    return files


def file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def load_state(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_state(path: Path, state: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(path)


def delete_paths(client, collection: str, paths: list[str]):
    from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchAny
    for i in range(0, len(paths), 256):
        client.delete(
            collection_name=collection,
            points_selector=FilterSelector(filter=Filter(must=[
                FieldCondition(key="path", match=MatchAny(any=paths[i:i + 256])),
            ])),
            wait=True,
        )


def scroll_all(client, collection: str) -> tuple[list[str], list[dict]]:
    ids, payloads = [], []
    offset = None
    while True:
        points, offset = client.scroll(collection_name=collection, limit=2048, offset=offset,
                                       with_payload=True, with_vectors=False)
        for p in points:
            ids.append(str(p.id))
            payloads.append(p.payload)
        if offset is None:
            return ids, payloads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project_path", help="Root of the project to index")
    parser.add_argument("--collection", default=None, help="Qdrant collection name")
    parser.add_argument("--dry-run", action="store_true", help="List the files that would be indexed and exit")
    args = parser.parse_args()

    root = Path(args.project_path).expanduser().resolve()
    if not root.is_dir():
        print(f"ERROR: {root} does not exist", file=sys.stderr)
        sys.exit(1)

    files = collect_files(root)
    if args.dry_run:
        for rel in files:
            print(rel)
        print(f"{len(files)} files", file=sys.stderr)
        return

    from fastembed import TextEmbedding
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams
    from rank_bm25 import BM25Okapi

    collection = args.collection or collection_name(root)
    log(f"Indexing '{root}' → collection '{collection}'")
    log(f"Found {len(files)} files")

    st_path = state_path(collection)
    state = load_state(st_path)
    hashes: dict = state.get("files", {})

    client = QdrantClient(url=QDRANT_URL)
    embedder = TextEmbedding(model_name=EMBEDDING_MODEL)
    vector_size = len(next(iter(embedder.passage_embed(["x"]))))

    # (re)create the collection if missing or built with another model
    exists = client.collection_exists(collection)
    if exists and (state.get("model"), state.get("version")) != (EMBEDDING_MODEL, INDEX_VERSION):
        log(f"Collection '{collection}' was built with another model or index version; rebuilding")
        client.delete_collection(collection)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        client.create_payload_index(collection, field_name="path", field_schema=PayloadSchemaType.KEYWORD)
        hashes = {}
        log(f"Created collection '{collection}'")

    current = {}
    changed = []
    for rel in files:
        h = file_hash(root / rel)
        current[rel] = h
        if hashes.get(rel) != h:
            changed.append(rel)
    removed = [rel for rel in hashes if rel not in current]
    log(f"{len(changed)} files changed / new, {len(removed)} removed")

    # drop every old chunk of changed and removed files before inserting the new ones
    if changed or removed:
        delete_paths(client, collection, changed + removed)

    all_chunks = []
    for rel in changed:
        try:
            text = (root / rel).read_text(errors="replace")
        except OSError:
            current.pop(rel, None)
            continue
        all_chunks.extend(chunk_text(redact(text, rel), rel))

    if all_chunks:
        log(f"Embedding {len(all_chunks)} chunks…")
        done = 0
        for i in range(0, len(all_chunks), BATCH_SIZE):
            batch = all_chunks[i:i + BATCH_SIZE]
            embeddings = embedder.passage_embed([c["text"] for c in batch])
            points = [
                PointStruct(
                    id=point_id(c["path"], c["chunk_id"]),
                    vector=emb.tolist(),
                    payload={"text": c["text"], "path": c["path"], "line": c["line"]},
                )
                for c, emb in zip(batch, embeddings)
            ]
            client.upsert(collection_name=collection, points=points)
            done += len(batch)
            log(f"  {done * 100 // len(all_chunks):3d}%  {done}/{len(all_chunks)}")

    # rebuild BM25 over ALL indexed chunks, keyed by the same ids as Qdrant
    log("Building BM25 index over full corpus…")
    ids, payloads = scroll_all(client, collection)
    bm25 = BM25Okapi([tokenize(p["text"]) for p in payloads]) if payloads else None
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(bm25_path(collection), "wb") as fh:
        pickle.dump({"bm25": bm25, "ids": ids, "payloads": payloads}, fh)
    log(f"BM25 index saved → {bm25_path(collection)} ({len(ids)} chunks)")

    save_state(st_path, {"root": str(root), "model": EMBEDDING_MODEL, "version": INDEX_VERSION, "files": current})
    log("Done.")


if __name__ == "__main__":
    main()
