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
# What `npm run build:demo` writes, and what GitHub Pages publishes.
DIST = ROOT / "demo-dist"

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
    # Exempt on the OTHER route: not "handled with a 501", but never reachable,
    # because the control that would call it does not exist in this build. See
    # `token_ranking_is_not_offered` below for why that is the right shape of
    # answer here and a 501 is not.
    # Exempt on the same route as `/api/attention/attribute` below: never
    # reachable, because the panel that would call it is not built here.
    # Building a graph replaces activations and re-runs the model thousands of
    # times -- MEASURED at 1,735 forward passes on Qwen3-1.7B at depth 2 --
    # against a live model this page does not have. A 501 would be the wrong
    # answer for the same reason it is wrong there: a control that can only
    # fail teaches a visitor that the measurement is broken.
    # --- needs YOUR input, or a file only your machine has -------------
    #
    # None of these can be baked, and that is the point rather than a gap: the
    # answer depends on something the visitor supplies. `api.ts` refuses each
    # in DEMO and VIEWER builds through `noModelHere`, naming what the
    # measurement would cost, so the panel says "no model here" instead of
    # showing a 404 that reads as "this measurement is broken".
    "/api/probe": "trains on YOUR labelled examples against a live residual stream",
    "/api/attention/baselines": (
        "runs the resample arm, whose replacements come from a corpus the "
        "reader supplies; a bundled one would be somebody else's text"
    ),
    "/api/ground": "masks passages out of YOUR document and re-runs the model",
    "/api/patchscope": "hands a hidden state back to the model to describe",
    "/api/custom/ablate": "ablates a network the visitor loaded, not a recording",
    "/api/diff/models": "loads two checkpoints and runs a prompt set through both",
    "/api/lens/tune": "a training run over a corpus, minutes of live compute",
    "/api/quantdiff/behaviour": "loads both quantisations of a model side by side",
    "/api/rubric/score": "runs rubric predicates over a live generation",
    "/api/judge": (
        "reads the probability mass a LIVE model puts on the verdict token, "
        "one forward pass per paraphrase; there are no weights here to read"
    ),
    "/api/judge/plan": (
        "prices that run by listing the prompts it would make — real "
        "arithmetic, but pricing a purchase this page cannot make would "
        "describe a wait nobody here is going to have"
    ),
    "/api/vla/audit": (
        "reads the parquet, the video files and the recorded statistics of a "
        "dataset on YOUR disk; a browser cannot see a filesystem, and a baked "
        "audit would be a claim about somebody else's data"
    ),
    "/api/vla/actions/cost": (
        "counts the forward passes an action run would spend against a policy "
        "sidecar this page does not have"
    ),
    "/api/vla/actions/compare": (
        "runs the action expert once per sampled frame; a baked curve would be "
        "a fabricated comparison sitting beside real recordings"
    ),
    "/api/vla/actions/swap": (
        "re-runs one frame under every task string the dataset contains and "
        "again under several seeds, against a policy this page does not carry"
    ),
    "/api/vla/actions/knockout": (
        "replaces each input with its episode mean and re-runs the policy, "
        "which needs the policy"
    ),
    "/api/attention/anchors": (
        "perturbs every token that is not being held and re-runs the model "
        "once per draw, then again per candidate subset — 83 forward passes "
        "MEASURED on the narrowest search this offers, thousands on a wide "
        "one, and every one of them against a live model"
    ),
    "/api/attention/gradients": (
        "a forward AND a backward pass at every step of the path from the "
        "baseline to your prompt; a backward pass needs the graph a live "
        "model builds, which a recording does not carry"
    ),
    "/api/patch/screen": (
        "one gradient pass over a live model, then one re-run per shortlisted "
        "site to check the screen against the exact answer — the checking is "
        "the point, and it is the half a bundle could never bake"
    ),
    "/api/neurons/evidence": (
        "runs YOUR text through a live model and taps one MLP layer, once per "
        "sequence; the corpus is the reader's and so is the answer"
    ),
    "/api/graph": "opens a circuit-tracer `.pt` from disk, which a page cannot see",
    "/api/gguf": "reads a GGUF from disk; the browser cannot see the filesystem",
    "/api/gguf/plan": "reads a GGUF header from disk to project its memory cost",
    "/api/gguf/load": "loads a GGUF from disk as the live model",
    "/api/patch/graph": (
        "walks the patching grid backwards, thousands of forward passes "
        "against a live model; the whole panel is gated off in demo and "
        "viewer builds instead"
    ),
    "/api/attention/attribute": (
        "ranks tokens by masking them one at a time, which is dozens of "
        "forward passes against a live model; the button is gated off in "
        "demo and viewer builds instead"
    ),
    # Exempt for the same reason, and it MUST be listed rather than left to
    # `covered()`. `covered()` would have passed it: demo.ts answers
    # `/api/features/` as a PREFIX, so this path matched a handler whose body
    # returns the single-feature DETAIL payload with a 200. A prefix handler
    # returning a different shape is not coverage, it is a fabricated ranking
    # rendered as a measurement — the one failure this project cannot ship. The
    # exemption records that the answer is "the control does not exist here",
    # and `feature_ranking_is_not_offered` below is what keeps that true.
    "/api/features/ablate": (
        "ranks SAE features by removing one at a time, two forward passes per "
        "feature against a live model; the button is gated off in demo and "
        "viewer builds instead, and demo.ts's /api/features/ prefix would "
        "otherwise answer it 200 "
        "with a single feature's detail payload"
    ),
    # --- the control, and two readers of a database this page does not have --
    "/api/attention/control": (
        "builds a SECOND model — the same architecture with random weights — "
        "and runs the identical head ranking over the same tokens, so it is "
        "two full sweeps and a second model resident against a live runtime "
        "this page has none of; api.ts refuses it in DEMO and VIEWER builds "
        "and names what the run would cost"
    ),
    "/api/sweeps": (
        "lists the sweeps saved in the trace database on your own machine. A "
        "static bundle has no database, so any list here would be somebody "
        "else's runs — the whole panel is gated off at its mount in App.tsx "
        "for demo builds, and api.ts refuses the call as a second lock"
    ),
    "/api/sweeps/": (
        "the resume plan for one saved sweep, priced against its stored rows "
        "and checked against the model loaded now — neither of which exists "
        "behind a static page. Reached only from the same gated panel as "
        "/api/sweeps above"
    ),
    "/api/vla/export": (
        "writes a sweep this machine measured into an MCAP file, reading the "
        "rows AND the run record beside them — the unit, the two strides and "
        "the frame total — out of the trace database. A static page has no "
        "database and never ran a sweep, so there is nothing to export; baking "
        "one would put somebody else's dataset in the visitor's Foxglove under "
        "our provenance, which is the one thing robot_export.py exists to stop. "
        "api.ts refuses it here through `refusedHere` rather than "
        "`noModelHere`, because what is missing is the run, not a checkpoint"
    ),
    "/api/patterns/across": (
        "counts one structural finding over every run recorded on a machine. "
        "This page carries a single recording, so any answer would be a "
        "pattern of one — the same argument demo.ts already makes for the "
        "per-run /patterns route, and api.ts refuses it here in that route's "
        "own words rather than through noModelHere, because what is missing "
        "is the run database and not a checkpoint"
    ),
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


def _own_vocabulary(needle: str) -> bool:
    """Does the project's own source already write this string?

    Only the authored text matters, so this reads the source rather than the
    build output: a match in `node_modules` or a `.pyc` would be somebody
    else's vocabulary standing in for ours.
    """
    roots = (
        (ROOT / "modelmri", "*.py"),
        (SRC, "*.ts"),
        (SRC, "*.tsx"),
    )
    lowered = needle.lower()
    for base, pattern in roots:
        for f in base.rglob(pattern):
            if "node_modules" in f.parts or "__pycache__" in f.parts:
                continue
            if lowered in f.read_text("utf-8", errors="replace").lower():
                return True
    return False


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


def token_ranking_is_not_offered() -> None:
    """The token-ranking control must be absent from the demo build.

    Every other unanswerable endpoint here is handled with a 501 that says
    what the call does and what would make it work, because "not available in
    the demo" in red reads as a broken tool. Token attribution is the one
    place that argument runs the other way. Its whole claim is a measurement —
    "masking THIS token moves the answer at THIS position by THIS much" — and
    a button that can only ever answer with a failure does not teach a visitor
    that the page has no model behind it. It teaches them the measurement does
    not work, which is the one impression this project cannot afford in the
    one surface most people will ever touch.

    So the control is gated off, and gating it is also what stops the call
    from existing at all: with no button there is no unhandled path left to
    fall through to demo.ts's catch-all and become a 409 under advice that
    cannot help. This check is the thing that makes "just switch the button
    on" fail here rather than ship. Someone who later bakes a real recorded
    attribution into the bundle has to come and change this deliberately —
    which is the point, because they will have to decide what a recorded
    ranking at a fixed position is honestly claiming first.
    """
    print("\ndemo — is the token ranking correctly not offered?")
    panel = (SRC / "AttentionPanel.tsx").read_text("utf-8")

    at = panel.find('"Rank tokens"')
    check(
        "the panel still has a Rank tokens control to gate",
        at >= 0,
        "no 'Rank tokens' label found in AttentionPanel.tsx — if the control "
        "was renamed, this check went blind and must be renamed with it",
    )
    # `check` here reports and returns None, unlike ui_check.py's, so it cannot
    # be used as the condition of an early return — written that way, the two
    # assertions below were skipped in silence and the section printed one
    # cheerful PASS while checking nothing. Negative index instead: a missing
    # label leaves the window empty and both checks fail loudly, which is the
    # correct behaviour for a check that has gone blind.
    at = max(at, 0)

    # The guard has to sit between the enclosing JSX and the label, and the
    # window is deliberately tight: measured at 250 characters, which is the
    # button's four props. A wider window would start reaching back over the
    # head-ranking controls above it and would pass on their `!replay`
    # instead of this button's own gate.
    guard = panel[max(0, at - 600) : at]
    check(
        "Rank tokens is gated on !replay && !DEMO && !VIEWER",
        "!replay && !DEMO && !VIEWER" in guard,
        "that exact guard is not in the 600 characters before the label. A "
        "recording, the static demo and the .mri viewer each have no model, "
        "so the control would render somewhere it can only fail",
    )

    # And the call itself: if some other path reaches the endpoint, the gate on
    # the button is decoration.
    calls = re.findall(r"attributeTokens\s*\(", panel)
    check(
        "attribution is requested from exactly one place",
        len(calls) == 1,
        f"{len(calls)} calls to attributeTokens — every one of them needs the "
        f"same gate, and a second call site is how the gate stops being true",
    )

    # The gate above is the enforceable claim; this is the observed one. DEMO
    # is `import.meta.env.VITE_DEMO === "1"`, which the demo build folds to a
    # constant, so rollup drops the branch entirely: measured on this machine,
    # the demo bundle contains neither the label nor the endpoint string while
    # the normal app bundle contains both. Only asserted when a bundle is
    # sitting there — `npm run build:demo` is what CI does before publishing,
    # and a clean checkout has nothing to read.
    built = sorted(DIST.glob("assets/*.js")) if DIST.is_dir() else []
    if not built:
        print(f"    (no demo bundle at {DIST.name}/ — run npm run build:demo)")
        return
    js = "\n".join(f.read_text("utf-8", errors="replace") for f in built)
    leaked = [s for s in ("Rank tokens", "/api/attention/attribute") if s in js]
    check(
        "the built demo bundle carries neither the control nor its endpoint",
        not leaked,
        f"{leaked} survived into the published JavaScript, so the gate is a "
        f"runtime one and the demo is shipping a control it cannot answer",
    )


def feature_ranking_is_not_offered() -> None:
    """The same argument as `token_ranking_is_not_offered`, and one more.

    A feature ranking is a measurement — "removing THIS feature moves the
    answer at THIS token by THIS much" — so a button that can only answer with
    a failure teaches a visitor that the measurement does not work rather than
    that the page has no model behind it.

    The one more: demo.ts answers `/api/features/` as a PREFIX, and its body
    returns the single-feature DETAIL payload. So unlike token attribution,
    an ungated control here would not get a 409 it could report — it would get
    a 200 carrying the wrong shape, and the panel would render a fabricated
    ranking as a measurement. `rankFeatures` also refuses in DEMO and VIEWER
    builds, which is a second lock; this checks the first one, and the bundle
    assertion at the end checks that neither is needed at runtime because the
    code is not there at all.
    """
    print("\ndemo — is the feature ranking correctly not offered?")
    panel = (SRC / "FeaturesPanel.tsx").read_text("utf-8")

    at = panel.find('"Rank features"')
    check(
        "the panel still has a Rank features control to gate",
        at >= 0,
        "no 'Rank features' label found in FeaturesPanel.tsx — if the control "
        "was renamed, this check went blind and must be renamed with it",
    )
    at = max(at, 0)
    guard = panel[max(0, at - 600) : at]
    check(
        "Rank features is gated on !DEMO && !VIEWER",
        "!DEMO && !VIEWER" in guard,
        "that guard is not in the 600 characters before the label. The static "
        "demo and the .mri viewer have no model, and demo.ts would answer the "
        "call 200 with a single feature's detail payload",
    )

    calls = re.findall(r"rankFeatures\s*\(", panel)
    check(
        "the ranking is requested from exactly one place",
        len(calls) == 1,
        f"{len(calls)} calls to rankFeatures — every one of them needs the "
        f"same gate, and a second call site is how the gate stops being true",
    )

    built = sorted(DIST.glob("assets/*.js")) if DIST.is_dir() else []
    if not built:
        print(f"    (no demo bundle at {DIST.name}/ — run npm run build:demo)")
        return
    js = "\n".join(f.read_text("utf-8", errors="replace") for f in built)
    leaked = [s for s in ("Rank features", "/api/features/ablate") if s in js]
    check(
        "the built demo bundle carries neither the control nor its endpoint",
        not leaked,
        f"{leaked} survived into the published JavaScript, so the gate is a "
        f"runtime one and the demo is shipping a control whose endpoint would "
        f"answer 200 with the wrong payload",
    )
    # And the annotation that hangs off a ranking. It cannot render in this
    # build — `measured` is only ever set by the gated button — but rollup
    # cannot prove that from a runtime value, so the JSX and its tooltip prose
    # shipped as ~2.4 kB of text nothing could reach. Hoisting it behind the
    # same folded constants is what makes the exclusion a build-time fact.
    dead = [s for s in ("below_resolution", "same-size random edit") if s in js]
    check(
        "the per-row KL annotation is dropped rather than shipped dead",
        not dead,
        f"{dead} is in the demo bundle, which means the annotation is gated on "
        f"a runtime value instead of on !DEMO && !VIEWER and rollup could not "
        f"drop it",
    )


def _ts_interface(name: str) -> tuple[set[str], set[str]]:
    """The field names an `api.ts` interface declares, split required/optional.

    A regex rather than a TypeScript parse, and the shapes it reads are flat
    records of `name: type;` — anything nested would need a real parser and
    would also be a shape this check has no business asserting about.
    """
    text = (SRC / "api.ts").read_text("utf-8")
    match = re.search(
        r"export interface " + re.escape(name) + r"\s*\{(.*?)\n\}", text, re.S
    )
    if match is None:
        return set(), set()
    body = re.sub(r"/\*.*?\*/", "", match.group(1), flags=re.S)
    body = re.sub(r"//[^\n]*", "", body)
    required: set[str] = set()
    optional: set[str] = set()
    for field, mark in re.findall(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)(\??):", body, re.M):
        (optional if mark else required).add(field)
    return required, optional


def payload_shapes() -> None:
    """Does what the demo SENDS match what the client declares it receives?

    The gap this closes: `static_coverage` asks whether a path is answered and
    never looks inside the answer, and `json<T>(r)` in `api.ts` is a bare type
    cast — nothing checks a key at runtime either. So two routes drifted in
    silence.

    MEASURED before this check existed:

      /api/paths   the baker synthesised 8 of the live route's 15 keys, and
                   the 7 it dropped — models_home, inherited_caches,
                   models_dirs, cwd, legacy, undelivered_traces, platform —
                   are the ones a reader opens the storage panel FOR. On top
                   of that, `demo_note`, the sentence explaining that every
                   remaining row is a placeholder, was baked from the start
                   and typed nowhere, so seven unexplained placeholders
                   rendered with nothing to say why.
      /api/ollama  the shim answered `{up, models, reason}` against a live
                   `{host, installed, models, suggested, up}`. `reason` was
                   declared and read nowhere, so the handler's own sentence
                   never rendered and the picker fell back to "Ollama is not
                   running, install it from ollama.com, start it, then reopen
                   this panel" — two next steps that cannot change anything on
                   a static recording.
    """
    print("\nshapes — does the demo send the keys the client declares?")

    env = json.loads((BUNDLE / "env.json").read_text("utf-8"))
    for bundled, interface in (("paths", "PathInfo"), ("hub_auth", "HubAuth")):
        required, optional = _ts_interface(interface)
        if not required and not optional:
            check(f"{interface} is declared in api.ts", False, "no such interface")
            continue
        sent = set(env.get(bundled) or {})
        check(
            f"env.{bundled} sends every key {interface} requires",
            required <= sent,
            f"missing {sorted(required - sent)}",
            note=f"{len(sent)} keys",
        )
        check(
            f"env.{bundled} sends nothing {interface} does not declare",
            sent <= (required | optional),
            f"undeclared {sorted(sent - required - optional)}",
        )

    # The Ollama answer is written inline in `demo.ts` rather than baked, so
    # this reads the literal. Keys only — the values are the point of the
    # handler and not this check's business.
    text = (SRC / "demo.ts").read_text("utf-8")
    block = re.search(r'p === "/api/ollama"\)\s*\{(.*?)\n  \}', text, re.S)
    required, optional = _ts_interface("OllamaState")
    sent = (
        set(re.findall(r"^\s{6}([a-zA-Z_][a-zA-Z0-9_]*):", block.group(1), re.M))
        if block
        else set()
    )
    check(
        "the /api/ollama shim sends every key OllamaState requires",
        bool(block) and required <= sent,
        f"missing {sorted(required - sent)}" if block else "handler not found",
        note=f"{len(sent)} keys",
    )
    check(
        "the /api/ollama shim sends nothing OllamaState does not declare",
        bool(block) and sent <= (required | optional),
        f"undeclared {sorted(sent - required - optional)}" if block else "",
    )


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
    vla_integrity()


def vla_integrity() -> None:
    """The robot bundle, and the one readout it serves WHOLE.

    Everything else in `vla.json` is a strip — six frames of 159, five layers
    of twelve — because a decoded frame is base64 megabytes. The timeline is a
    few hundred floats per track, so the demo carries every timestep and draws
    the real series rather than a sample of them. Which makes it worth
    checking as data rather than as a blob: the panel's whole claim is that
    every track is indexed by the SAME `t`, and a bundle where one track came
    out short would draw two series on two axes under one playhead.
    """
    path = BUNDLE / "vla.json"
    if not path.is_file():
        return
    print("\n  [robot]")
    vla = json.loads(path.read_text("utf-8"))
    timeline = vla.get("timeline")
    check(
        "the episode's series are baked, not just its frames",
        isinstance(timeline, dict) and bool(timeline.get("tracks")),
        "vla.json has no `timeline`, so /api/vla/timeline answers nothing — "
        "re-run scripts/bake_demo.py against a cached dataset",
    )
    if not isinstance(timeline, dict) or not timeline.get("tracks"):
        return

    # The handler refuses any episode but this one BY NUMBER, so a timeline
    # baked from a different episode would be served under the frame strip's
    # label — the exact substitution the scrubber and the layer dial were
    # fixed for.
    check(
        "the timeline is of the episode the rest of this bundle recorded",
        timeline.get("episode") == vla.get("episode"),
        f"frames are episode {vla.get('episode')}, timeline is episode "
        f"{timeline.get('episode')}",
    )
    steps = timeline.get("timesteps") or []
    ragged = [
        f"{t['column']}[{d}]"
        for t in timeline["tracks"]
        for d, series in enumerate(t["series"])
        if len(series) != len(steps)
    ]
    check(
        f"every track is indexed by the same {len(steps)} timesteps",
        not ragged,
        f"off the shared axis: {', '.join(ragged)}",
        note=", ".join(t["column"] for t in timeline["tracks"]),
    )
    # A finite number can be drawn and a `null` is a hole. Anything else — a
    # NaN that survived JSON as a bare token, a string — is a point the chart
    # would place somewhere arbitrary.
    unplottable = sum(
        1
        for t in timeline["tracks"]
        for series in t["series"]
        for v in series
        if v is not None and not isinstance(v, (int, float))
    )
    check(
        "every baked point is a number or an honest null",
        unplottable == 0,
        f"{unplottable} point(s) are neither",
    )


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

    # Where the prompt ends. The panel rests on the last prompt token, so a
    # missing or wrong boundary puts the demo back to the blank canvas this
    # replaced — or marks the wrong chips as the model's own words.
    n_prompt = llm.get("n_prompt", 0)
    check(
        "the prompt/output boundary is recorded and inside the sequence",
        isinstance(n_prompt, int) and 0 < n_prompt < n,
        f"n_prompt={n_prompt!r} against {n} tokens",
        note=f"prompt ends at {n_prompt}, first generated token "
        f"{llm['tokens'][n_prompt]!r}"
        if isinstance(n_prompt, int) and 0 < n_prompt < n
        else "",
    )

    # Each slice must decode to a square of the right size and say which head
    # it is — a slice mislabelled renders under the wrong control, which is
    # the whole class of bug this file exists for.
    bad = []
    worst = 0.0
    # What uint8 quantisation can cost a row sum, derived rather than guessed.
    # Each cell is stored as round(v / scale), so it can be off by scale/2, and
    # a row of n cells can accumulate n * scale/2. That bound grows with
    # sequence length — which is why Qwen3-0.6B (31 tokens) lands at 0.0236
    # against the same encoder. Borrowing the live model's `< 0.02` here would
    # have meant loosening a number until it
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
    hits = []
    for label, needle in probes:
        if not needle or len(needle) <= 2:
            continue
        if needle.lower() not in blob.lower():
            continue
        # A string this project already writes cannot be evidence that the
        # BUNDLE leaked who baked it.
        #
        # The account-name probe is a bare substring test, which is the right
        # shape for a name nobody would type by accident and the wrong shape
        # for one that is also an English word. On GitHub Actions the account
        # is `runner`, and `telemetry.py`'s own sentence — "the attention
        # scores ModelMRI asks for and a plain runner never allocates" — ships
        # in the bundle as authored copy. So this reported a leak that was the
        # tool describing itself, and it did it only in CI, where the username
        # happens to be a common word. It cost a red main.
        #
        # Reported rather than silently dropped: a probe that quietly stops
        # probing is the failure this whole function exists to avoid.
        if _own_vocabulary(needle):
            print(
                f"  note  '{needle}' is in this project's own source, so "
                f"{label} cannot be told from the tool's own prose here "
                f"— skipped, and the path probes below still run"
            )
            continue
        hits.append(label)

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

    # The second one that shipped, and the reason the scan above did not catch
    # it: "NVIDIA GeForce RTX 4060 Laptop GPU" contains no path, no username
    # and no drive letter. A blocklist of identifier *shapes* cannot find a
    # device name, so this asserts the positive form instead — the demo names
    # no device at all, because there is no device behind a static page. A
    # visitor on a phone was being told it was running CUDA on 8.6 GB of VRAM.
    accel = env.get("accelerator") or {}
    check(
        "the demo names no accelerator, because it has none",
        accel.get("kind") == "recorded"
        and not accel.get("name")
        and not accel.get("vram_gb")
        and not accel.get("torch_device"),
        f"accelerator says kind={accel.get('kind')!r} name={accel.get('name')!r} "
        f"vram_gb={accel.get('vram_gb')!r} — that is the baking machine's GPU, "
        f"published to every visitor",
    )

    # The third: `/api/models/discovered` was published verbatim, so the "On
    # this machine" tab listed one person's entire HuggingFace cache — 17
    # repositories, annotated "cached, loads offline" — to strangers whose
    # machine has none of it. The demo may only offer what it can actually
    # replay, which is exactly the recorded scenarios.
    scenarios = {
        s["id"]
        for s in json.loads((BUNDLE / "scenarios.json").read_text("utf-8"))["scenarios"]
    }
    disco = json.loads((BUNDLE / "discovered.json").read_text("utf-8"))
    listed = {m.get("id") for m in disco.get("models") or []}
    check(
        "the model list is the demo's own recordings, not a machine's cache",
        listed == scenarios and not disco.get("roots"),
        f"discovered lists {sorted(listed - scenarios)} which no scenario can "
        f"replay, roots={disco.get('roots')!r}",
    )

    # WHICH MODEL THE FEATURE RECORDINGS CAME FROM. `features.json` is baked
    # while SAE_MODEL is resident, which is not either scenario — the demo
    # ships one public SAE and no Qwen has one. The shim gates the whole
    # features section on this field, and while it was ABSENT the gate keyed
    # on the scenario index's default instead: true for a Qwen, over numbers
    # measured on google/gemma-2-2b. The panel then reported
    # `gemma-scope-2b-pt-res`, `d_in 2304` and Gemma's token strip under a
    # Qwen session, and the steering A/B paired Qwen's baseline against
    # Gemma's steered sentence as though one caused the other.
    feats = json.loads((BUNDLE / "features.json").read_text("utf-8"))
    feat_model = feats.get("model")
    check(
        "features.json says which model it was baked on",
        feat_model is not None,
        "no `model` key — the shim cannot tell whether these features belong "
        "to the selected scenario, and defaulted to assuming they did",
    )
    # It may legitimately be a model no scenario offers: that is the case the
    # gate exists to close, and it closes correctly as long as the field is
    # there to read.
    if feat_model is not None and feat_model not in scenarios:
        check(
            "a features bundle from outside the scenarios is gated, not served",
            'const saeScenario = async () => (await bundle<any>("features")).model'
            in (SRC / "demo.ts").read_text("utf-8"),
            f"features.json was baked on {feat_model}, which no scenario "
            f"offers, and demo.ts keys the SAE gate on something else",
        )

    # The status pill renders `${hf_id} · ${device}`, so a baked "cuda:0" told
    # a visitor on a phone that their model was running on CUDA. Six of these
    # were published across five payloads. `demo.ts` already falls back to
    # "recorded" when the field is null, so the only requirement is that the
    # data stop asserting a device.
    devices = []

    def walk(node, where):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "device" and isinstance(v, str):
                    devices.append(f"{where}:{v}")
                else:
                    walk(v, where)
        elif isinstance(node, list):
            for v in node:
                walk(v, where)

    for path in sorted(BUNDLE.glob("*.json")):
        walk(json.loads(path.read_text("utf-8")), path.name)
    check(
        "no payload names a torch device",
        not devices,
        f"{sorted(set(devices))} — that is the baking machine's device, and the "
        f"status pill prints it next to the model name",
    )

    # Every model row must be replayable. A picker entry that cannot load is
    # worse here than a shorter list, because the demo's whole claim is that
    # everything on screen is real.
    unloadable = [
        m.get("id") for m in disco.get("models") or [] if not m.get("loadable")
    ]
    check(
        "every model the demo offers can be opened",
        not unloadable,
        f"{unloadable} are listed but not loadable",
    )


def main() -> int:
    print("demo parity check")
    static_coverage()
    token_ranking_is_not_offered()
    feature_ranking_is_not_offered()
    no_machine_leaks()
    payload_shapes()
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
