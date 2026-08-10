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

import base64
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "frontend" / "src"
BUNDLE = ROOT / "frontend" / "public" / "demo"

# Endpoints a static demo genuinely cannot serve. They must still be HANDLED
# — with a 501 that says what the call does and what would make it work —
# because "not available in the demo" in red reads as a broken tool, which is
# what a visitor saw on the HuggingFace tab.
EXEMPT = {
    # Reading a filesystem the browser cannot see.
    "/api/session/open": "opens a .mri from disk; the viewer build does this",
    "/api/session/close": "closes a replay the demo never enters",
    "/api/model/cancel": "there is no download to cancel",
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
    index = json.loads((BUNDLE / "scenarios.json").read_text("utf-8"))
    scenarios = index["scenarios"]
    check(
        "more than one model is recorded",
        len(scenarios) > 1,
        f"only {[s['id'] for s in scenarios]} — one model answers 'does this "
        f"work', not 'does this work on a real model'",
        note=", ".join(s["id"] for s in scenarios),
    )
    check(
        "the default scenario is one that exists",
        any(s["id"] == index["default"] for s in scenarios),
        f"default {index['default']!r} is not among {[s['id'] for s in scenarios]}",
    )
    for s in scenarios:
        scenario_integrity(s)


def scenario_integrity(scenario: dict) -> None:
    print(f"\n  [{scenario['id']}]")
    llm = json.loads((BUNDLE / f"llm-{scenario['slug']}.json").read_text("utf-8"))
    meta = llm["meta"]

    # The index is what the picker refuses against, so a shape stated there
    # and contradicted in the bundle would refuse — or allow — the wrong thing.
    check(
        "the index agrees with the recording about its shape",
        (meta["n_layers"], meta["n_heads"], meta["n_tokens"])
        == (scenario["n_layers"], scenario["n_heads"], scenario["n_tokens"]),
        f"index says {scenario['n_layers']}x{scenario['n_heads']}"
        f"x{scenario['n_tokens']}, bundle says "
        f"{meta['n_layers']}x{meta['n_heads']}x{meta['n_tokens']}",
    )
    _bundle_checks(llm, meta)


def _decode(blob: str, scale: float, n: int) -> list[list[float]]:
    """The mirror of viewer.ts's `dequantise`, which the demo now uses too."""
    raw = base64.b64decode(blob)
    return [[round(raw[r * n + c] * scale, 5) for c in range(n)] for r in range(n)]


def _bundle_checks(llm: dict, meta: dict) -> None:
    n_layers, n_heads, n = meta["n_layers"], meta["n_heads"], meta["n_tokens"]

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

    check(
        "the token list is stored once and matches the advertised length",
        len(llm.get("tokens", [])) == n,
        f"{len(llm.get('tokens', []))} tokens against a meta saying {n}",
    )

    # Each slice must decode to a square of the right size and say which head
    # it is — a slice mislabelled renders under the wrong control, which is
    # the whole class of bug this file exists for.
    bad = []
    worst = 0.0
    # What uint8 quantisation can cost a row sum, derived rather than guessed.
    # Each cell is stored as round(v / scale), so it can be off by scale/2, and
    # a row of n cells can accumulate n * scale/2. That bound grows with
    # sequence length — which is why gpt2 (23 tokens) lands at 0.0196 and
    # Qwen3-0.6B (31) at 0.0236 against the same encoder. Borrowing the live
    # model's `< 0.02` here would have meant loosening a number until it
    # passed; deriving it checks that the encoder behaves as the arithmetic
    # says it must.
    bound = 0.0
    for key, slice_ in attn.items():
        try:
            layer, head = (int(x) for x in key.split("."))
        except ValueError:
            # A pre-parity bundle keyed slices by layer alone, which is the
            # shape that made the head selector a no-op. Report it, do not
            # crash on it — a traceback reads as a broken check, not a
            # failing one.
            bad.append(f"{key!r} is not a 'layer.head' key")
            continue
        if slice_.get("layer") != layer or slice_.get("head") != head:
            bad.append(
                f"{key} self-reports L{slice_.get('layer')}H{slice_.get('head')}"
            )
            continue
        raw = base64.b64decode(slice_["q"])
        if len(raw) != n * n:
            bad.append(f"{key} decodes to {len(raw)} bytes, not {n}x{n}")
            continue
        # Rows are softmaxes. Quantisation costs a little; a wrong reduction
        # costs a lot, and this is what tells them apart.
        bound = max(bound, n * slice_["scale"] / 2)
        for row in _decode(slice_["q"], slice_["scale"], n):
            worst = max(worst, abs(sum(row) - 1.0))
    check(
        "every slice decodes to a square of the right size, correctly labelled",
        not bad,
        "; ".join(bad[:3]),
    )
    check(
        "decoded rows sum to 1 within what the encoding can cost",
        worst <= bound,
        f"worst row sum is off by {worst:.4f}, beyond the {bound:.4f} that "
        f"uint8 over {n} tokens can explain — that is a wrong reduction, not "
        f"quantisation",
        note=f"worst {worst:.4f}, quantisation allows {bound:.4f}",
    )

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


def no_machine_leaks() -> None:
    """The demo is published. Nothing in it may identify the machine that
    baked it.

    This is not hypothetical. The bundle shipped
    `{"signed_in": true, "user": "<username>"}` from `/api/hub/auth`, so every
    visitor to the public site saw the baker's HuggingFace account and a
    "sign out" link for it; and `/api/paths` shipped real directory paths,
    because the scrub replaced the home directory only and a model cache on
    another volume went out verbatim.

    Scrubbing is a blocklist, and a blocklist is a promise to have thought of
    everything. This is the check that does not require having thought of
    everything: it reads what is actually about to be published.
    """
    print("\nprivacy — does the bundle describe the machine that baked it?")
    home = str(Path.home())
    user = os.environ.get("USERNAME") or os.environ.get("USER") or ""

    blob = "\n".join(
        f.read_text("utf-8", errors="replace") for f in sorted(BUNDLE.glob("*.json"))
    )

    probes: list[tuple[str, str]] = [
        ("the home directory", home.replace("\\", "/")),
        ("the OS account name", user),
    ]
    hits = [
        label
        for label, needle in probes
        if needle and len(needle) > 2 and needle.lower() in blob.lower()
    ]

    # Absolute paths of any shape, on any OS. `~/...` is fine — it names a
    # location without naming a person.
    for pattern, label in (
        (r"[A-Za-z]:[\\/](?!$)", "a Windows drive path"),
        (r'"/(?:home|Users)/[^"/]+', "a POSIX home path"),
    ):
        if re.search(pattern, blob):
            hits.append(label)

    check(
        "no machine identifiers in the published bundle",
        not hits,
        f"found {', '.join(sorted(set(hits)))} — that reaches every visitor",
    )

    # And the specific one that shipped: a visitor is not signed in as anyone.
    env = json.loads((BUNDLE / "env.json").read_text("utf-8"))
    auth = env.get("hub_auth") or {}
    check(
        "the demo shows a signed-out session, not the baker's",
        not auth.get("signed_in") and not auth.get("user"),
        f"hub_auth says signed_in={auth.get('signed_in')} user={auth.get('user')!r}",
    )


def main() -> int:
    print("demo parity check")
    static_coverage()
    no_machine_leaks()
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
