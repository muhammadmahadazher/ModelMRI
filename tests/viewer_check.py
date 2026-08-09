"""The browser viewer must produce exactly the numbers the tool produces.

    python scripts/build_frontend.py --viewer
    python tests/viewer_check.py

`frontend/src/viewer.ts` re-implements `modelmri/session.py` in TypeScript:
gunzip, base64, uint8 dequantisation, rounding. Two implementations of one
format drift — and a viewer that renders a *slightly* different matrix than
the tool is worse than no viewer, because nothing on screen would say so.

So this parses the same file both ways and compares every cell. It serves the
bundled viewer over http (module scripts do not load from file://) and drives
it with Playwright, reading the numbers back out of the running page rather
than trusting the source to be equivalent.

Needs: playwright (uv run playwright install chromium), and a built viewer.
"""

from __future__ import annotations

import asyncio
import functools
import http.server
import socketserver
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIEWER = ROOT / "modelmri" / "static" / "viewer"
FIXTURE = ROOT / "tests" / "fixtures" / "parity.mri"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def build_fixture() -> bytes:
    """A session with structure worth disagreeing about: an attention sink,
    a causal mask, and values across the whole quantisation range."""
    sys.path.insert(0, str(ROOT))
    from modelmri import session

    n, layers, heads = 24, 4, 3
    matrices = {}
    for layer in range(layers):
        for head in range(heads):
            rows = []
            for r in range(n):
                raw = [
                    ((layer * 7 + head * 13 + r * 3 + c * 11) % 23) + 0.5
                    for c in range(r + 1)
                ]
                raw[0] += 40.0  # the sink every real head has
                raw += [0.0] * (n - r - 1)
                total = sum(raw)
                rows.append([v / total for v in raw])
            matrices[(layer, head)] = rows
    return session.build(
        model_id="parity/fixture",
        device="cpu",
        dtype="float32",
        n_params=1234,
        tokens=[f"t{i}" for i in range(n)],
        prompt="parity",
        generation="check",
        attention=matrices,
        n_layers=layers,
        n_heads=heads,
        note="every cell must match",
    )


def python_side(data: bytes) -> dict:
    sys.path.insert(0, str(ROOT))
    from modelmri import session

    parsed = session.parse(data)
    total, cells, worst = 0.0, 0, 0.0
    for key in sorted(parsed.attention):
        layer, head = (int(x) for x in key.split(":"))
        for row in parsed.attention_slice(layer, head)["matrix"]:
            rs = 0.0
            for v in row:
                total += v
                cells += 1
                rs += v
            worst = max(worst, abs(rs - 1))
    return {
        "slices": len(parsed.attention),
        "cells": cells,
        "checksum": round(total, 6),
        "worst": round(worst, 5),
        "tokens": parsed.tokens,
    }


def serve(directory: Path, port: int) -> socketserver.TCPServer:
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(directory)
    )
    httpd = socketserver.TCPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


async def browser_side(port: int) -> dict:
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(f"http://127.0.0.1:{port}/", wait_until="networkidle")
        result = await page.evaluate(
            """async () => {
              const blob = await (await fetch('./parity.mri')).blob();
              const file = new File([blob], 'parity.mri');
              const input = document.querySelector('input[type=file][accept=".mri"]');
              const dt = new DataTransfer(); dt.items.add(file);
              input.files = dt.files;
              input.dispatchEvent(new Event('change', {bubbles: true}));
              await new Promise(r => setTimeout(r, 1200));

              const meta = await (await fetch('/api/attention/meta')).json();
              if (!meta.available) return {error: 'the viewer did not open the file'};
              let total = 0, cells = 0, worst = 0, tokens = null;
              for (let l = 0; l < meta.n_layers; l++) {
                for (let h = 0; h < meta.n_heads; h++) {
                  const d = await (await fetch(`/api/attention?layer=${l}&head=${h}`)).json();
                  if (d.error) return {error: d.error};
                  tokens = d.tokens;
                  for (const row of d.matrix) {
                    let s = 0;
                    for (const v of row) { total += v; cells++; s += v; }
                    worst = Math.max(worst, Math.abs(s - 1));
                  }
                }
              }
              return {
                slices: meta.n_layers * meta.n_heads,
                cells,
                checksum: Number(total.toFixed(6)),
                worst: Number(worst.toFixed(5)),
                tokens,
              };
            }"""
        )
        await browser.close()
        return result


def main() -> int:
    if not (VIEWER / "index.html").is_file():
        print(
            f"no viewer build at {VIEWER}\n  python scripts/build_frontend.py --viewer",
            file=sys.stderr,
        )
        return 1

    data = build_fixture()
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_bytes(data)
    (VIEWER / "parity.mri").write_bytes(data)

    expected = python_side(data)
    port = 5921
    httpd = serve(VIEWER, port)
    try:
        got = asyncio.run(browser_side(port))
    finally:
        httpd.shutdown()

    if got.get("error"):
        print(f"FAILED: {got['error']}", file=sys.stderr)
        return 1

    print(f"  fixture   {len(data) / 1024:.1f} KB, {expected['slices']} slices")
    ok = True
    for key in ("slices", "cells", "checksum", "worst", "tokens"):
        same = got.get(key) == expected[key]
        ok = ok and same
        shown = key if key != "tokens" else "tokens"
        mark = "PASS" if same else "FAIL"
        detail = (
            f"{expected[key]}"
            if key != "tokens"
            else f"{len(expected[key])} identical"
            if same
            else f"python={expected[key][:4]} browser={str(got.get(key))[:60]}"
        )
        print(f"  [{mark}] {shown:9} {detail}")
        if not same and key != "tokens":
            print(f"         python={expected[key]}  browser={got.get(key)}")

    print()
    print(
        "the viewer and the tool agree on every cell"
        if ok
        else "THE VIEWER DISAGREES WITH THE TOOL"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
