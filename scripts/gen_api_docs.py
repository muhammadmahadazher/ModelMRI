"""Regenerate docs/reference/api.md from the app's own OpenAPI schema.

    uv run python scripts/gen_api_docs.py [--check]

A hand-maintained endpoint table is a list of things that used to be true.
This reads the routes off the application object — no server needed — so the
reference cannot describe an API that no longer exists.

`--check` exits non-zero if the file is out of date, which is what CI wants.
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "reference" / "api.md"

# Order matters: the first matching prefix wins, so more specific groups come
# first. Anything unmatched lands in "Other" and is therefore visible rather
# than silently dropped.
GROUPS: list[tuple[str, tuple[str, ...]]] = [
    # Before "Session", and matching the trailing slash on purpose: these are
    # the `.mri` routes, not the app-status one at `/api/session`.
    ("Shared sessions (.mri)", ("/api/session/",)),
    ("Session", ("/api/session", "/")),
    ("Model", ("/api/model", "/api/models", "/api/accelerator", "/api/hub/models")),
    ("Discovery", ("/api/hub", "/api/ollama")),
    ("Attention", ("/api/attention", "/api/vla/attention")),
    ("Features", ("/api/features", "/api/sae", "/api/steer")),
    ("Custom models", ("/api/custom",)),
    ("Robot policy", ("/api/vla",)),
    ("Agents", ("/api/traces",)),
]

HEADER = """# HTTP API

Everything the UI does goes through this API, so anything you can see you
can script. Generated from the app's own OpenAPI schema by
`scripts/gen_api_docs.py` — run it after adding a route.

Base URL: `http://127.0.0.1:5900`. Interactive docs: `/docs`.
"""

FOOTER = """
## Streaming

`GET /ws/generate` (WebSocket). Send `{"prompt": "..."}` and receive
`{"type":"token","text":"..."}` frames, then one `{"type":"done"}`.
A generation that fails mid-stream sends `{"type":"error","message":...}` —
it does not silently close as a success.

## Status codes

| code | meaning |
|---|---|
| 200 | fine |
| 409 | you asked for something in the wrong order, or a dependency is down. The body has an actionable message. |
| 413 | the upload was larger than the limit — a `.mri` is not that big, so it is probably not one. |
| 422 | the request was malformed, the model is gated and you have not accepted its licence, a custom adapter could not be loaded, or a session file could not be read. |

There is deliberately no 500 path for ordinary failures: an unreachable
Ollama, a stalled download, an out-of-memory load and a user's adapter
raising on import all return a status that says what happened.
"""


def _matches(path: str, prefixes: tuple[str, ...]) -> bool:
    """Prefix match, except `/` which is exact.

    As a prefix, "/" matches every route in the schema — Session silently
    swallowed the entire API on the first run of this script.
    """
    for prefix in prefixes:
        if prefix == "/":
            if path == "/":
                return True
        elif path == prefix or path.startswith(prefix):
            return True
    return False


def render() -> str:
    import modelmri

    # Which modelmri got imported is not a detail. Running this as
    # `python scripts/gen_api_docs.py` puts `scripts/` on sys.path -- not the
    # repo -- so `import modelmri` finds whatever is installed. With an older
    # wheel installed that silently REMOVED three live endpoints from the
    # reference and reported success, which is the exact failure this
    # generator exists to prevent: a table of things that used to be true.
    used = Path(modelmri.__file__).resolve().parent
    if used != ROOT / "modelmri":
        raise SystemExit(
            f"refusing to generate docs from {used}\n"
            f"  (expected {ROOT / 'modelmri'})\n"
            f"That is a different copy of modelmri -- version {modelmri.__version__} "
            f"-- and its routes are not this repo's. Run from the repo root with "
            f"the repo on the path:\n"
            f"  python -c \"import sys; sys.path.insert(0, '.'); "
            f'from scripts.gen_api_docs import main; raise SystemExit(main())"\n'
            f"or install this checkout first: pip install -e ."
        )

    from modelmri.server import create_app

    schema = create_app().openapi()
    rows: list[tuple[str, str, str]] = []
    for path, ops in schema["paths"].items():
        for method, op in ops.items():
            if method.lower() not in ("get", "post", "put", "patch", "delete"):
                continue
            summary = op.get("summary") or op.get("operationId", "")
            rows.append((method.upper(), path, summary))

    used: set[tuple[str, str]] = set()
    out = [HEADER]
    for title, prefixes in GROUPS:
        picked = sorted(
            (m, p, s)
            for m, p, s in rows
            if (m, p) not in used and _matches(p, prefixes)
        )
        if not picked:
            continue
        used.update((m, p) for m, p, _ in picked)
        out.append(f"\n## {title}\n")
        out.append("| method | path | notes |")
        out.append("|---|---|---|")
        for m, p, s in picked:
            out.append(f"| `{m}` | `{p}` | {s} |")
        out.append("")

    leftover = sorted((m, p, s) for m, p, s in rows if (m, p) not in used)
    if leftover:
        out.append("\n## Other\n")
        out.append("| method | path | notes |")
        out.append("|---|---|---|")
        for m, p, s in leftover:
            out.append(f"| `{m}` | `{p}` | {s} |")
        out.append("")

    return "\n".join(out).rstrip() + "\n" + FOOTER


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if out of date")
    args = ap.parse_args()

    fresh = render()
    current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
    if args.check:
        if fresh != current:
            print(f"{OUT.relative_to(ROOT)} is out of date - run:")
            print("  uv run python scripts/gen_api_docs.py")
            return 1
        print(f"{OUT.relative_to(ROOT)} matches the schema")
        return 0

    OUT.write_text(fresh, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({fresh.count(chr(10))} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
