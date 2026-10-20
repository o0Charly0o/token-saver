#!/usr/bin/env python3
"""
Resume el uso de tokens de sesiones reales de Claude Code a partir de sus
transcripts (~/.claude/projects/<proyecto>/<sesión>.jsonl).

Para comparar, haz la misma pregunta en dos conversaciones nuevas, una pidiendo
usar search_context y otra pidiendo no usarlo, y lanza el script.

Uso:
  python medicion/session_tokens.py <project_path> [--last 2]
  python medicion/session_tokens.py --session <fichero.jsonl>
"""
import argparse
import json
import re
from pathlib import Path

import tiktoken

enc = tiktoken.get_encoding("cl100k_base")
TRANSCRIPTS = Path.home() / ".claude" / "projects"


def transcript_dir(project_path: str) -> Path:
    root = Path(project_path).expanduser().resolve()
    return TRANSCRIPTS / re.sub(r"[^A-Za-z0-9]", "-", str(root))


def result_text(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(c.get("text", "") for c in content or [] if isinstance(c, dict))


def summarize(path: Path) -> dict:
    tool_names: dict[str, str] = {}
    per_tool: dict[str, list[int]] = {}
    usage_by_request: dict[str, dict] = {}
    first_prompt = ""
    for line in path.open():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = d.get("message") or {}
        content = msg.get("content")
        if d.get("type") == "assistant":
            if "usage" in msg:
                usage_by_request[d.get("requestId") or d.get("uuid")] = msg["usage"]
            for c in content or []:
                if c.get("type") == "tool_use":
                    tool_names[c["id"]] = c["name"].replace("mcp__tokens-saver__", "")
        elif d.get("type") == "user":
            if isinstance(content, str) and not first_prompt:
                first_prompt = content
            for c in content if isinstance(content, list) else []:
                if c.get("type") == "tool_result":
                    name = tool_names.get(c.get("tool_use_id"), "?")
                    calls_tokens = per_tool.setdefault(name, [0, 0])
                    calls_tokens[0] += 1
                    calls_tokens[1] += len(enc.encode(result_text(c.get("content")), disallowed_special=()))
                elif c.get("type") == "text" and not first_prompt:
                    first_prompt = c["text"]

    usages = list(usage_by_request.values())
    context = lambda u: u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
    return {
        "session": path.stem,
        "prompt": first_prompt.strip().splitlines()[0][:70] if first_prompt.strip() else "",
        "requests": len(usages),
        "final_context": context(usages[-1]) if usages else 0,
        "input_processed": sum(context(u) for u in usages),
        "output": sum(u.get("output_tokens", 0) for u in usages),
        "tools": per_tool,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project_path", nargs="?", help="Project whose latest sessions to summarize")
    parser.add_argument("--last", type=int, default=2, help="Number of latest sessions (default 2)")
    parser.add_argument("--session", action="append", default=[], help="Transcript file (repeatable)")
    args = parser.parse_args()

    files = [Path(s) for s in args.session]
    if args.project_path:
        d = transcript_dir(args.project_path)
        files += sorted(d.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)[:args.last]
    if not files:
        parser.error("give a project_path or --session")

    for f in files:
        s = summarize(f)
        print(f"\n=== {s['session']}  «{s['prompt']}»")
        print(f"  Peticiones al modelo     : {s['requests']}")
        print(f"  Contexto final           : {s['final_context']:,} tokens")
        print(f"  Entrada procesada (total): {s['input_processed']:,} tokens (suma de todas las peticiones, incluye caché)")
        print(f"  Salida generada          : {s['output']:,} tokens")
        print("  Resultados de tools (tokens aprox. que entraron al contexto):")
        for name, (calls, tok) in sorted(s["tools"].items(), key=lambda x: -x[1][1]):
            print(f"    {name:<22} {calls:3d} llamadas  {tok:8,} tokens")


if __name__ == "__main__":
    main()
