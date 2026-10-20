# tokens-saver

Pipeline de recuperación híbrida (BM25 + Qdrant + re-ranker) para Claude Code.
En lugar de abrir ficheros completos, Claude busca solo los fragmentos relevantes del proyecto.

## Stack

- **Qdrant** — búsqueda semántica (vectores), en Docker, solo en `localhost`
- **BM25** — búsqueda léxica (rank-bm25)
- **Embeddings** — `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (fastembed, local, CPU)
- **Re-ranker** — `jinaai/jina-reranker-v2-base-multilingual` (fastembed, local, CPU)
- **MCP Server** — integrado en Claude Code

Los dos modelos son multilingües: las consultas y las specs pueden estar en español.

## Instalación

Requisitos: Python ≥ 3.10, Docker y el CLI `claude`.

```bash
git clone <este-repo> tokens-saver
cd tokens-saver
./install.sh
```

El script:

1. crea `.venv` e instala `requirements.txt`;
2. levanta Qdrant (`docker compose up -d`, imagen fijada);
3. descarga los modelos (~1,3 GB la primera vez, en `~/.cache`);
4. registra el MCP con `claude mcp add tokens-saver -s user` (disponible en todos tus proyectos).

Opciones: `--no-mcp` (no registra el MCP), `--no-models` (no descarga modelos ahora).
Se puede repetir sin problema; para mover el repo de carpeta, vuelve a ejecutarlo.

## Indexar un proyecto

```bash
.venv/bin/python indexer/index.py /ruta/al/proyecto
# ver qué ficheros entrarían, sin indexar:
.venv/bin/python indexer/index.py /ruta/al/proyecto --dry-run
```

- El indexado es **incremental**: solo re-procesa ficheros modificados y borra del índice los fragmentos de ficheros cambiados o eliminados.
- No escribe nada en el proyecto: el estado y el índice BM25 se guardan en `~/.cache/tokens-saver/`.
- La colección se llama `<carpeta>_<hash de la ruta>`; dos proyectos con el mismo nombre de carpeta no se pisan.
- Si cambias de modelo de embeddings, la colección se reconstruye sola.

### Qué ficheros se indexan

- En repos git, los que devuelve `git ls-files` (tracked + no ignorados): se respeta el `.gitignore`.
- `openspec/` se indexa siempre, aunque esté en `.gitignore`.
- Extensiones de código y docs habituales, incluidas las de Drupal (`.twig`, `.module`, `.install`, `.theme`, `.inc`).
- Nunca: `settings*.php`, `.env*`, `*.sql`, claves y certificados, `vendor/`, `node_modules/`, `.next/`, `dist/`, `build/`, `sites/*/files`, `web/core`, `contrib/`, minificados, lockfiles y ficheros > 512 KB.
- Los valores de claves con aspecto de credencial (`password`, `secret`, `token`, `api_key`…) se sustituyen por `[REDACTED]` antes de indexar.

Exclusiones extra por proyecto: crea `.tokens-saver-ignore` en la raíz del proyecto, un patrón por línea:

```
config/stage*/
tests/fixtures/
*.feature
```

## Uso desde Claude Code

Tools disponibles en cualquier sesión:

- **`search_context(query, project_path)`** — devuelve los N fragmentos más relevantes con `ruta:línea`
- **`index_project(project_path)`** — re-indexa desde Claude

> "Busca en el proyecto cómo se gestiona la suscripción a la newsletter"
> → Claude llama `search_context("suscripción newsletter", project_path="/ruta")` y recibe los 5 fragmentos más relevantes.

## Medir

```bash
.venv/bin/python medicion/measure_tokens.py /ruta/al/proyecto -q "consulta 1" -q "consulta 2"
```

Compara los tokens recuperados frente a leer **completos los ficheros donde están esos fragmentos**
(lo que haría Grep + Read para ver el mismo código). Es una referencia, no el ahorro real de una
sesión: no mide cuántos ficheros habría abierto Claude ni la calidad de la respuesta. No hay
todavía una medición del ahorro en sesiones reales.

Para sesiones reales, haz la misma pregunta en dos conversaciones nuevas (una pidiendo usar
`search_context` y otra pidiendo no usarlo) y compara sus transcripts:

```bash
.venv/bin/python medicion/session_tokens.py /ruta/al/proyecto --last 2
```

Muestra por sesión el contexto final, la entrada total procesada, la salida y los tokens que
aportó cada tool (Read, Grep, search_context…).

## Variables de entorno

| Variable | Default | Descripción |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | URL de Qdrant |
| `TOKENS_SAVER_CACHE` | `~/.cache/tokens-saver` | Estado e índices BM25 |
| `TS_EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | Modelo de embeddings (fastembed) |
| `TS_RERANKER_MODEL` | `jinaai/jina-reranker-v2-base-multilingual` | Re-ranker (fastembed) |
| `CHUNK_SIZE` | `400` | Tamaño de fragmento en tokens aprox. |
| `CHUNK_OVERLAP` | `80` | Solape entre fragmentos |
| `MAX_FILE_BYTES` | `524288` | Tamaño máximo de fichero indexado |
| `DEFAULT_COLLECTION` | `""` | Colección por defecto en el MCP |

Para usar variables en el MCP: `claude mcp add tokens-saver -s user -e TS_RERANKER_MODEL=... -- ...`.

## Limpieza de configuración antigua

Si probaste versiones anteriores:

- Un bloque `mcpServers` en `~/.claude/settings.json` no tiene efecto: bórralo. El MCP válido es el de `claude mcp list`.
- Revisa `claude mcp list` dentro de cada proyecto y quita los MCP que ya no uses (`claude mcp remove <nombre> -s local`): sus tools ocupan contexto en cada sesión.
- Revisa los marketplaces de plugins locales (`/plugin` en Claude Code) que apunten a carpetas que ya no existen.
- Las colecciones de la versión anterior (IDs de 32 bits, otro modelo) hay que re-indexarlas.
