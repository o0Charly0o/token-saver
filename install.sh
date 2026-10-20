#!/usr/bin/env bash
# Installs tokens-saver on this machine: venv, Qdrant and the MCP server in Claude Code.
# Usage: ./install.sh [--no-mcp] [--no-models]
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
NAME="tokens-saver"
REGISTER_MCP=1
DOWNLOAD_MODELS=1
for arg in "$@"; do
  case "$arg" in
    --no-mcp) REGISTER_MCP=0 ;;
    --no-models) DOWNLOAD_MODELS=0 ;;
    *) echo "Unknown option: $arg" >&2; exit 1 ;;
  esac
done

# --- Python >= 3.10 (required by mcp) ---
PYTHON=""
for cand in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$cand" >/dev/null 2>&1 &&
     "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
    PYTHON="$(command -v "$cand")"; break
  fi
done
if [ -z "$PYTHON" ]; then
  echo "ERROR: Python >= 3.10 not found (macOS: brew install python@3.13)" >&2
  exit 1
fi
echo "→ Python: $PYTHON"

# --- virtualenv ---
if [ ! -x "$DIR/.venv/bin/python" ]; then
  "$PYTHON" -m venv "$DIR/.venv"
fi
"$DIR/.venv/bin/pip" install -q --upgrade pip
"$DIR/.venv/bin/pip" install -q -r "$DIR/requirements.txt"
echo "→ Dependencies installed"

# --- Qdrant ---
if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker not found (install Docker Desktop or OrbStack)" >&2
  exit 1
fi
docker compose -f "$DIR/docker-compose.yml" up -d
for _ in $(seq 1 30); do
  curl -sf http://localhost:6333/healthz >/dev/null && break
  sleep 1
done
curl -sf http://localhost:6333/healthz >/dev/null || { echo "ERROR: Qdrant not responding on :6333" >&2; exit 1; }
echo "→ Qdrant running on localhost:6333"

# --- models (first download is ~1.3 GB, cached in ~/.cache) ---
if [ "$DOWNLOAD_MODELS" = 1 ]; then
  echo "→ Downloading models…"
  "$DIR/.venv/bin/python" -c "
import sys; sys.path.insert(0, '$DIR')
from common import EMBEDDING_MODEL, RERANKER_MODEL
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
TextEmbedding(EMBEDDING_MODEL); TextCrossEncoder(RERANKER_MODEL)
print('   ', EMBEDDING_MODEL, '+', RERANKER_MODEL)
"
fi

# --- MCP server in Claude Code (user scope: available in every project) ---
if [ "$REGISTER_MCP" = 1 ]; then
  if command -v claude >/dev/null 2>&1; then
    claude mcp remove "$NAME" -s user >/dev/null 2>&1 || true
    claude mcp add "$NAME" -s user -- "$DIR/.venv/bin/python" "$DIR/mcp_server/server.py"
    echo "→ MCP '$NAME' registered (check with: claude mcp list)"
  else
    echo "! 'claude' CLI not found; register later with:"
    echo "  claude mcp add $NAME -s user -- \"$DIR/.venv/bin/python\" \"$DIR/mcp_server/server.py\""
  fi
fi

echo
echo "Done. Index a project with:"
echo "  $DIR/.venv/bin/python $DIR/indexer/index.py /path/to/project"
