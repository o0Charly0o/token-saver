#!/usr/bin/env python3
"""
Compara, para consultas reales, los tokens de los fragmentos recuperados frente a
los de leer completos los ficheros donde están esos fragmentos (lo que haría
Claude con Grep + Read para ver el mismo código).

No mide calidad ni si Claude habría leído más o menos ficheros: es una cota de
referencia, no el ahorro real de una sesión.

Uso:
  python medicion/measure_tokens.py <project_path> -q "consulta 1" -q "consulta 2"
  python medicion/measure_tokens.py <project_path> --queries consultas.txt
"""
import argparse
import sys
import time
from pathlib import Path

import tiktoken

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import collection_name
from retriever.search import search

enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(enc.encode(text, disallowed_special=()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project_path", help="Root of the indexed project")
    parser.add_argument("--collection", default=None)
    parser.add_argument("--n-results", type=int, default=5)
    parser.add_argument("-q", "--query", action="append", default=[], help="Query (repeatable)")
    parser.add_argument("--queries", help="File with one query per line")
    args = parser.parse_args()

    queries = list(args.query)
    if args.queries:
        queries += [l.strip() for l in Path(args.queries).read_text().splitlines() if l.strip()]
    if not queries:
        parser.error("give at least one query with -q or --queries")

    root = Path(args.project_path).expanduser().resolve()
    collection = args.collection or collection_name(root)

    print(f"\nMedición de tokens")
    print(f"Proyecto : {root}")
    print(f"Colección: {collection}")
    print(f"Resultados por query: {args.n_results}\n", flush=True)

    rows = []
    for query in queries:
        t0 = time.time()
        results = search(query, collection, n_results=args.n_results)
        elapsed = time.time() - t0
        retrieved = count_tokens("\n".join(r["text"] for r in results))
        files = sorted({r["path"] for r in results})
        full = sum(count_tokens((root / f).read_text(errors="replace")) for f in files if (root / f).is_file())
        ratio = retrieved / full * 100 if full else 0
        rows.append((query, retrieved, full, len(files), ratio, elapsed))
        print(f"  [{retrieved:6,} tok vs {full:7,} tok en {len(files)} ficheros ({ratio:5.1f}%) | {elapsed:.2f}s]  {query}",
              flush=True)

    avg = lambda i: sum(r[i] for r in rows) / len(rows)
    summary = f"""
# Resultados de medición — {root.name}

Fragmentos recuperados frente a leer completos los ficheros que los contienen.
No es el ahorro de una sesión real de Claude Code: no mide cuántos ficheros
habría abierto Claude ni la calidad de la respuesta.

| Métrica | Valor |
|---|---|
| Tokens recuperados (media) | {int(avg(1)):,} |
| Tokens ficheros completos (media) | {int(avg(2)):,} |
| Recuperado / completo (media) | {avg(4):.1f}% |
| Latencia media búsqueda | {avg(5):.2f}s |
| Resultados por query | {args.n_results} |

## Por query

| Query | Recuperado | Ficheros completos | Nº ficheros | % | Latencia |
|---|---|---|---|---|---|
"""
    for query, tok, full, n, pct, t in rows:
        summary += f"| {query} | {tok:,} | {full:,} | {n} | {pct:.1f}% | {t:.2f}s |\n"

    results_path = Path(__file__).parent / "results.md"
    results_path.write_text(summary.strip() + "\n")
    print(f"\nResultados guardados en {results_path}")


if __name__ == "__main__":
    main()
