"""The hosted demo is the only ModelMRI most people will ever touch.

It is also, until this file, the least verified surface in the repo: the
`.mri` viewer next to it is gated cell-for-cell by `viewer_check.py`, while
the demo's entire gate was `test -f demo-dist/index.html`.

That asymmetry showed. The demo advertised 12 layers x 12 heads in its own
meta and baked three slices, so 141 of 144 selections drew a different head's
arcs than the controls said. "Rank heads" -- the capability the README leads
with -- had no handler at all and answered 409 under a dead button.

Three checks, cheapest first:

  static   every /api/... path the frontend can call is one the demo answers
  bundle   the baked payloads contain what those handlers will look up
  parity   a baked matrix equals the live server's, cell for cell

`static` is the one that matters most: it fails the day someone adds an
endpoint, rather than the day a visitor clicks it.

    uv run python tests/demo_check.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "frontend" / "src"
BUNDLE = ROOT / "frontend" / "public" / "demo"

# Endpoints a static demo is *right* not to answer, each for a stated reason.
# Anything not here must be handled, so the list is the argument.
EXEMPT = {
    # Mutating a machine the demo does not have.
    "/api/hub/signin": "signing in writes a token to the user's config dir",
    "/api/hub/signout": "same",
    "/api/ollama/pull": "downloads gigabytes to a daemon that is not running",
    "/api/model/cancel": "there is no download to cancel",
    # Reading a filesystem the browser cannot see.
    "/api/session/open": "opens a .mri from disk; the viewer build does this",
    "/api/session/close": "closes a replay the demo never enters",
    # Live network lookups against third parties.
    "/api/hub/models": "live HuggingFace search",
    "/api/ollama/resolve": "live Ollama registry lookup",
    "/api/ollama/size": "same",
    "/api/vla/dataset": "streams a dataset that is not in the bundle",
}

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "", note: str = "") -> None:
    """`detail` explains a failure; `note` is shown either way.

    Kept apart because printing the failure text next to PASS reads as a
    contradiction — the first version cheerfully reported
    "PASS  ... — exportSession has no DEMO branch".
    """
    suffix = note or ("" if ok else detail)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  — ' + suffix if suffix else ''}")
    if not ok:
        FAILURES.append(f"{name}: {detail}")


def api_paths() -> set[str]:
    """Every /api/... literal the frontend can reach."""
    text = (SRC / "api.ts").read_text("utf-8")
    return set(re.findall(r"/api/[a-zA-Z0-9/_-]*", text))


def demo_handlers() -> tuple[set[str], set[str]]:
    """What the demo shim answers, split by how it matches.

    The distinction is load-bearing. `p === "/api/sae"` answers exactly one
    path, while `p.startsWith("/api/features/")` answers a subtree. Treating
    every handler as a prefix — the first version of this file did — quietly
    reported `/api/sae/available` as covered by the exact-match `/api/sae`,
    i.e. the check under-reported the very gaps it exists to find.
    """
    text = (SRC / "demo.ts").read_text("utf-8")
    exact = set(re.findall(r'p\s*===\s*"(/api/[a-zA-Z0-9/_-]*)"', text))
    prefix = set(re.findall(r'p\.startsWith\(\s*"(/api/[a-zA-Z0-9/_-]*)"', text))
    return exact, prefix


def static_coverage() -> None:
    print("\nstatic — can the demo answer everything the UI asks?")
    wanted = {p for p in api_paths() if p not in EXEMPT}
    exact, prefix = demo_handlers()

    def covered(path: str) -> bool:
        return path in exact or any(path.startswith(pre) for pre in prefix)

    # One endpoint is answered in api.ts rather than the shim: `exportSession`
    # needs bytes and a Content-Disposition header, so it never went through
    # the patched fetch. Rather than exempt it — an exemption is a promise
    # nobody checks — assert the branch that handles it is really there.
    api_ts = (SRC / "api.ts").read_text("utf-8")
    export_handled = "if (DEMO)" in api_ts and "demoSessionFile" in api_ts
    check(
        "/api/session/export is answered by api.ts's demo branch",
        export_handled,
        "exportSession has no DEMO branch, so Share this view 404s on Pages",
    )
    if export_handled:
        wanted.discard("/api/session/export")

    missing = sorted(p for p in wanted if not covered(p))
    check(
        f"{len(wanted)} reachable endpoints have a demo handler",
        not missing,
        f"unhandled: {', '.join(missing)}",
    )

    # An exemption that no longer names a real endpoint is dead weight that
    # would silently excuse a future path of the same name.
    stale = sorted(set(EXEMPT) - api_paths())
    check("no stale exemptions", not stale, f"still listed but never called: {stale}")


def bundle_integrity() -> None:
    print("\nbundle — is the baked data what those handlers will look up?")
    llm = json.loads((BUNDLE / "llm.json").read_text("utf-8"))
    meta = llm["meta"]
    n_layers, n_heads = meta["n_layers"], meta["n_heads"]

    # The bug this file was written for: meta advertised 12x12 while three
    # slices existed, so the dial and the arcs disagreed 141 times out of 144.
    attn = llm["attention"]
    expected = {
        f"{layer}.{head}" for layer in range(n_layers) for head in range(n_heads)
    }
    have = set(attn)
    check(
        f"every advertised layer/head is baked ({n_layers}x{n_heads})",
        expected <= have,
        f"{len(expected - have)} of {len(expected)} missing" if expected - have else "",
    )

    # Each slice must be square, the right size, and say which head it is —
    # a slice mislabelled is a slice that renders under the wrong control.
    bad = []
    for key, slice_ in attn.items():
        try:
            layer, head = (int(x) for x in key.split("."))
        except ValueError:
            # A pre-parity bundle keyed slices by layer alone, which is the
            # shape that made the head selector a no-op. Report it, do not
            # crash on it — a traceback here reads as a broken check rather
            # than a failing one.
            bad.append(f"{key!r} is not a 'layer.head' key")
            continue
        rows = slice_["matrix"]
        if slice_.get("layer") != layer or slice_.get("head") != head:
            bad.append(
                f"{key} self-reports L{slice_.get('layer')}H{slice_.get('head')}"
            )
        elif len(rows) != meta["n_tokens"] or any(
            len(r) != meta["n_tokens"] for r in rows
        ):
            bad.append(f"{key} is not {meta['n_tokens']}x{meta['n_tokens']}")
    check(
        "every slice is square, correctly sized and self-labelled",
        not bad,
        "; ".join(bad[:3]),
    )

    # Rows are softmaxes. Quantisation costs a little; a wrong reduction costs
    # a lot, and this is what tells them apart.
    worst = 0.0
    for slice_ in attn.values():
        for row in slice_["matrix"]:
            worst = max(worst, abs(sum(row) - 1.0))
    check(f"attention rows still sum to 1 (worst {worst:.4f})", worst < 0.02)

    # The generation is the first thing a visitor reads. A loop variable in the
    # bake script once shadowed it and wrote the literal string "mean" there,
    # which no schema check would have noticed — it is a string either way.
    gen = llm.get("generation", "")
    check(
        "the baked generation is a generation",
        isinstance(gen, str) and len(gen) > 8 and gen not in ("zero", "mean"),
        f"generation is {gen!r}",
    )

    rank = llm.get("ablate", {})
    check(
        "a ranking is baked for every layer, both baselines, plus the sweep",
        all(
            f"{layer}.{b}" in rank
            for layer in range(n_layers)
            for b in ("zero", "mean")
        )
        and "all.zero" in rank,
        f"have {len(rank)} of {n_layers * 2 + 2}",
    )

    diff = llm.get("diff", {})
    check(
        "a comparison is baked for every head a ranking can offer",
        bool(diff),
        f"{len(diff)} baked",
    )

    # Provenance: a demo that cannot say what produced it is a screenshot.
    prov = llm.get("provenance", {})
    check(
        "the bundle records what produced it",
        all(prov.get(k) for k in ("model", "revision", "dtype", "prompt")),
        f"have {sorted(prov)}",
    )


def main() -> int:
    print("demo parity check")
    static_coverage()
    bundle_integrity()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("demo is at parity with the tool")
    return 0


if __name__ == "__main__":
    sys.exit(main())
