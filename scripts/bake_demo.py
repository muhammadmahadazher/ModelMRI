"""Bake a static demo bundle from a live server.

    modelmri serve
    uv run python scripts/bake_demo.py

Captures real responses (attention, features, steering A/B, an agent trace,
robot frames + per-layer heatmaps) into frontend/public/demo/*.json. The
frontend, built with VITE_DEMO=1, reads those files instead of calling the
API — so GitHub Pages can serve the whole experience with no backend, no
model download, and no GPU.

Nothing here is synthetic: every number in the demo came out of the real
pipeline on this machine.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:5900"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "frontend" / "public" / "demo"
PROMPT = "The Eiffel Tower is located in the city of"
VLA_EPISODE, VLA_FRAME = 3, 60
VLA_LAYERS = [0, 3, 6, 9, 11]

# EVERY key `/api/paths` sends, synthesised. Never the baker's own values —
# that route once shipped real directories to the public site, and the scrub
# that was supposed to catch it replaced only the home directory, so a model
# cache on another volume went out verbatim. See the block in `bake_env`.
#
# Module level so a one-off repair of an already-baked bundle reads the same
# dict the next bake writes, instead of a second copy that can disagree.
#
# The first version had eight of these fifteen, and the seven it dropped —
# `models_home`, `inherited_caches`, `models_dirs`, `cwd`, `legacy`,
# `undelivered_traces`, `platform` — are the ones somebody opens the storage
# panel FOR. `PathInfo` declares them all as required and `json<T>` is a bare
# cast, so nothing complained; `tests/demo_check.py::payload_shapes` does now.
SYNTHETIC_PATHS = {
    "override": None,
    "data": "<your data dir>",
    "config": "<your config dir>",
    "cache": "<your cache dir>",
    "hf_home": "<your HuggingFace home>",
    "hf_hub_cache": "<your HuggingFace hub cache>",
    "trace_db": "<your data dir>/traces.sqlite",
    "hub_token": "<your config dir>/hub.json",
    "undelivered_traces": "<your data dir>/undelivered",
    # Lists and nullables at their real EMPTY shape rather than a placeholder
    # string: the panel skips a row whose value is empty, and "<your models
    # dir>" against a setting nobody has set would invent a configuration for
    # the reader.
    "models_dirs": [],
    "models_home": None,
    "inherited_caches": [],
    "cwd": "<wherever you started it>",
    "legacy": None,
    "platform": "<your platform>",
    "demo_note": (
        "These are placeholders. Run `modelmri where` and the panel shows "
        "the real locations for your OS and account — they are resolved "
        "per-platform, never hardcoded."
    ),
}

# Two recorded models, because one answers "does this work" and two answer
# "does this work at more than one size". Both are current instruct models
# that think out loud, and both are 28 x 16 — the shape where the whole-model
# sweep stops being instant.
#
# gpt2 used to be the first of these, on the argument that it is small, fast
# and famously wrong. It is gone, and not because of the size: a demo is the
# first thing anybody sees, and putting a 2019 model there says this tool is
# about 2019 models. The numbers it produced were real, so they were DELETED
# rather than relabelled — a Qwen3 name over a gpt2 measurement would have
# been a fabrication, which is worse than an old model.
#
# The picker offers every model it discovers, so a demo with one recording
# lets you select the other and then keeps replaying the first underneath —
# the page attributing one model's sentence to another. Either the scenario
# exists or the load is refused; there is no third honest option.
SCENARIOS = [
    {"id": "Qwen/Qwen3-0.6B", "slug": "qwen3-0.6b", "max_new_tokens": 12},
    {"id": "Qwen/Qwen3-1.7B", "slug": "qwen3-1.7b", "max_new_tokens": 12},
]

# Which model the features/steering bundle is baked on. Separate from
# SCENARIOS because it answers a different question — "has a published sparse
# autoencoder this tool can open" — and the two lists have no reason to agree.
# `modelmri/sae_registry.py` is the source of truth for what that means; if
# this name has no supported entry there, the bake stops rather than writing a
# features.json the demo would render as an empty panel.
SAE_MODEL = "google/gemma-2-2b"

# The LLM bundle bakes EVERY layer/head, not a sample. Three slices used to be
# baked against a meta advertising 12 x 12, so 141 of 144 selections drew a
# different head's arcs than the controls said — and the fallback was silent,
# which is the only kind of wrong nobody reports. 28 x 16 is 448 slices per
# model, which is what the bundle costs; a sampled bundle would be smaller and
# would put a different head's arcs under most of the controls.
#
# How many ranked heads offer a "what changes?" button. AttentionPanel renders
# `.slice(0, 5)`, so five is the reachable set, not a sample of it.
RANKED_ROWS = 5


def get(path: str, timeout: float = 900) -> dict:
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.load(r)


def post(path: str, body: dict, timeout: float = 900) -> dict:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def model_revision(hf_id: str) -> str:
    """The exact snapshot the demo was baked from, so it can be reproduced.

    "gpt2" is a moving target; `gpt2@607a30d` is not. Read from the local hub
    cache rather than the network, because the bake already ran against these
    weights and a second lookup could answer about different ones.
    """
    try:
        from modelmri import paths

        repo = paths.hf_hub_cache() / f"models--{hf_id.replace('/', '--')}"
        snaps = sorted((repo / "snapshots").iterdir())
        return snaps[-1].name if snaps else "unknown"
    except Exception:
        return "unknown"


def _forget_the_device(payload):
    """Null every `device` field on the way out.

    The status pill renders `${hf_id} · ${device}`, so a baked `"cuda:0"` told
    every visitor their model was on CUDA — on a phone, on a Mac, on anything.
    It is the same mistake as publishing the GPU's name, one field further
    down, and it survived the first privacy pass for the same reason: "cuda:0"
    is not a path, a username or a drive letter, so an identifier-shaped scan
    walks past it.

    Done here rather than at each call site because there were four
    (scenarios, both llm bundles, vla) and a fifth endpoint would have been a
    fifth chance to forget. `demo.ts` already reads `s.device ?? "recorded"` —
    the frontend was always ready for this; only the data was not.

    `dtype` deliberately survives. It is a property of the recording — these
    weights really were bfloat16 — not a claim about the reader's machine.
    """
    if isinstance(payload, dict):
        return {
            k: (None if k == "device" and isinstance(v, str) else _forget_the_device(v))
            for k, v in payload.items()
        }
    if isinstance(payload, list):
        return [_forget_the_device(v) for v in payload]
    return payload


def write(name: str, payload: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    payload = _forget_the_device(payload)
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"  {name:<26} {path.stat().st_size / 1024:7.1f} KB")


def bake_llm(scenario: dict) -> dict:
    """One model's whole instrument: every slice, ranking and comparison.

    Attention is quantised exactly as the `.mri` format does — uint8 against
    each matrix's own peak, base64 — and the browser decodes it with the same
    function the viewer uses. As raw JSON, Qwen3-0.6B's 28 x 16 x 31 x 31 is
    3.9 MB; the format that exists for precisely this problem takes it to a
    fraction of that, at a worst measured error of 0.002.
    """
    from modelmri.session import _quantise

    hf_id, slug = scenario["id"], scenario["slug"]
    print(f"\nLLM [{hf_id}]: load + generate")
    post("/api/model/load", {"hf_id": hf_id, "confirm": True}, timeout=1800)
    generation = post(
        "/api/model/prompt",
        {
            "prompt": PROMPT,
            "max_new_tokens": scenario["max_new_tokens"],
            "temperature": 0,
        },
    )["generation"]

    meta = get("/api/attention/meta")
    n_layers, n_heads, n_tokens = meta["n_layers"], meta["n_heads"], meta["n_tokens"]

    print(f"  attention: {n_layers} x {n_heads} = {n_layers * n_heads} slices")
    attn: dict[str, dict] = {}
    tokens: list[str] = []
    n_prompt = 0
    for layer in range(n_layers):
        for head in range(n_heads):
            block = get(f"/api/attention?layer={layer}&head={head}")
            tokens = tokens or block["tokens"]
            n_prompt = n_prompt or int(block.get("n_prompt") or 0)
            blob, scale = _quantise(block["matrix"])
            # Tokens are identical across every slice of one run, so storing
            # them 448 times would cost more than the matrices do.
            attn[f"{layer}.{head}"] = {
                "layer": layer,
                "head": head,
                "q": blob,
                "scale": scale,
            }

    # Rank heads is the capability the README leads with, and the demo had no
    # handler for it at all — the button answered 409 under advice that could
    # not work. Both baselines, because the panel offers both and they
    # genuinely disagree; plus the whole-model sweep the second button runs.
    print(f"  rankings: {n_layers} layers x 2 baselines + 2 sweeps")
    # NOT `baseline` — that name already holds the generated text, and reusing
    # it here silently wrote the string "mean" into the demo's generation and
    # into features.json. Caught by opening the built demo and reading it.
    ablate: dict[str, dict] = {}
    for cut in ("zero", "mean"):
        for layer in range(n_layers):
            ablate[f"{layer}.{cut}"] = get(
                f"/api/attention/ablate?layer={layer}&baseline={cut}&scope=layer"
            )
        ablate[f"all.{cut}"] = get(f"/api/attention/ablate?baseline={cut}&scope=all")

    # "what changes?" on each ranked row. The panel opens the comparison at
    # layer+1 (an ablation cannot change its own layer) against the head the
    # ranking just selected, so that is exactly the reachable set.
    # "what changes?" on each ranked row.
    #
    # The panel opens the comparison at layer+1 (an ablation cannot change its
    # own layer) against the head the ranking just selected — and WHICH head
    # that is depends on the baseline, because the two baselines rank
    # differently. Baking only the zero-baseline heads left every mean-baseline
    # comparison 422ing under a message claiming the demo records what the
    # ranking offers. A whole-model sweep jumps to the global winner, so its
    # top rows need their own entries too.
    #
    # No substituting a nearby head: the diff is of one specific head's
    # attention, so serving another under the same controls is exactly the
    # bug this file exists to prevent. Bake the reachable set instead.
    print(f"  comparisons: {n_layers} layers x 2 baselines x {RANKED_ROWS} rows")
    diff: dict[str, dict] = {}

    def bake_diff(at: int, head: int, cut_layer: int, cut_head: int) -> None:
        key = f"{at}.{head}.{cut_layer}.{cut_head}"
        if key in diff:
            return
        diff[key] = get(
            f"/api/attention/diff?layer={at}&head={head}"
            f"&a=live&b=ablate:{cut_layer}.{cut_head}"
        )

    for cut in ("zero", "mean"):
        for layer in range(n_layers):
            rows = ablate[f"{layer}.{cut}"]["ranked"][:RANKED_ROWS]
            if not rows:
                continue
            at = min(layer + 1, n_layers - 1)
            head = rows[0]["head"]  # rank() selects the top head first
            for row in rows:
                bake_diff(at, head, layer, row["head"])

        # The whole-model sweep selects the global winner, which is usually a
        # different layer AND head from any single-layer ranking.
        sweep = ablate[f"all.{cut}"]["ranked"][:RANKED_ROWS]
        if sweep:
            best = sweep[0]
            at = min(best["layer"] + 1, n_layers - 1)
            for row in sweep:
                bake_diff(at, best["head"], row["layer"], row["head"])

    session = get("/api/session")

    # Six endpoints the panels reach that had no recorded answer, so their
    # controls 404'd on Pages. Each is a GET with no argument the visitor
    # chooses, which is exactly what makes it bakeable: the response is a
    # property of THIS recording rather than of something typed on the day.
    #
    # `/api/patch/path` is deliberately NOT here. Its parent `/api/patch` is
    # refused by the shim -- the grid is not baked, because patching re-runs a
    # live model -- and an edge trace hanging off a grid that does not exist
    # would be a measurement with nothing to click it from.
    # NOT `/api/attention/baselines`: it runs the resample arm, which draws
    # replacements from a corpus of at least 8 long sentences that the reader
    # supplies. There is no HTTP way to install one, and bundling a corpus
    # would publish a baseline measured against somebody else's text. It is
    # refused in demo builds instead, naming the corpus.
    extra: dict = {}
    for name, path in (
        ("types", "/api/attention/types"),
        ("direct", "/api/attention/direct"),
        ("ablate_estimate", "/api/attention/ablate/estimate"),
        ("telemetry", "/api/telemetry"),
        ("lens_tuned", "/api/lens/tuned"),
        # WITH THIS SCENARIO'S MODEL RESIDENT. It used to be baked into
        # `env.json`, once, while SAE_MODEL was loaded — so the demo served
        # google/gemma-2-2b's 27-layer logit lens under every Qwen scenario,
        # receipt and all. A lens is per model in the most literal way: its
        # rows are that model's layers and its tokens are that model's
        # vocabulary.
        ("lens", "/api/lens?top_k=5"),
    ):
        try:
            extra[name] = get(path)
            print(f"  {path}")
        except urllib.error.HTTPError as err:
            # NOT silently skipped. A missing key makes the demo answer 404
            # again, and the whole point of this pass is that it stops doing
            # that -- so the bake says which one and why, out loud.
            print(f"  SKIP {path}: {err.code} {err.read().decode()[:120]}")

    write(
        f"llm-{slug}.json",
        {
            "prompt": PROMPT,
            "generation": generation,
            "meta": meta,
            "tokens": tokens,
            # So the demo's panel rests on the last prompt token too, rather
            # than reverting to the empty canvas this replaced.
            "n_prompt": n_prompt,
            "layers": list(range(n_layers)),
            "attention": attn,
            "ablate": ablate,
            "diff": diff,
            # See the capture block above for why each of these is bakeable.
            "extra": extra,
            # What produced this bundle. A demo that cannot say what it
            # replayed is a screenshot with buttons.
            "provenance": {
                "model": session["model"]["hf_id"],
                "revision": model_revision(session["model"]["hf_id"]),
                "dtype": session["model"]["dtype"],
                "device": session["model"]["device"],
                "prompt": PROMPT,
                "modelmri": session["version"],
            },
        },
    )
    return {
        "id": hf_id,
        "slug": slug,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_tokens": n_tokens,
        "generation": generation,
        "n_params": session["model"]["n_params"],
        # Base vs instruction-tuned, so the demo's caveat can say which.
        "instruct": bool(session["model"].get("instruct")),
        "device": session["model"]["device"],
        "dtype": session["model"]["dtype"],
    }


def main() -> int:
    try:
        get("/api/session", timeout=5)
    except urllib.error.URLError:
        print("No server on :5900 - run `modelmri serve` first.")
        return 1

    print("baking demo bundle")

    baked = [bake_llm(s) for s in SCENARIOS]
    # The index the demo reads first: which recordings exist, so a load of
    # anything else can be refused by name rather than silently ignored.
    write("scenarios.json", {"default": SCENARIOS[0]["id"], "scenarios": baked})

    # NAMED, not `SCENARIOS[0]`. The SAE bundle needs a model with a published
    # sparse autoencoder, which is a different constraint from "a model worth
    # demonstrating attention on" — and writing it as "whichever scenario is
    # first" meant reordering that list silently pointed the SAE bake at a
    # model that has no SAE. It would have loaded, produced nothing, and
    # written a features.json the demo renders as an empty panel.
    # Checked BEFORE the multi-gigabyte load, and against the registry rather
    # than against a hope. An unsupported entry is listed there precisely so
    # that "we know this SAE exists and cannot open it" is sayable — and the
    # bake must not spend twenty minutes loading weights to discover it.
    from modelmri import sae_registry

    usable = [e for e in sae_registry.for_model(SAE_MODEL) if e.get("supported")]
    if not usable:
        listed = sae_registry.for_model(SAE_MODEL)
        why = listed[0].get("note") if listed else "no release is registered for it"
        raise SystemExit(
            f"SAE_MODEL is {SAE_MODEL}, and this build cannot open an SAE for "
            f"it: {why}\n"
            f"Baking anyway would write a features.json the demo renders as an "
            f"empty panel. Point SAE_MODEL at a model with a supported entry "
            f"in modelmri/sae_registry.py, or make that entry supported."
        )

    print(f"\nLoading {SAE_MODEL} for the SAE bundle ({usable[0]['repo']})")
    post("/api/model/load", {"hf_id": SAE_MODEL, "confirm": True}, timeout=1800)
    baseline = post(
        "/api/model/prompt",
        {"prompt": PROMPT, "max_new_tokens": 12, "temperature": 0},
    )["generation"]

    print("\nSAE: features + steering A/B")
    sae = post("/api/sae/load", {})
    summary = get("/api/features/summary?top_k=8")
    idx = next((i for i, t in enumerate(summary["tokens"]) if "Paris" in t), -1)
    feature = summary["top"][idx][0][0] if idx >= 0 else summary["top"][-1][0][0]
    detail = get(f"/api/features/{feature}")
    post("/api/steer", {"feature_id": feature, "scale": -40})
    steered = post(
        "/api/model/prompt",
        {"prompt": PROMPT, "max_new_tokens": 12, "temperature": 0},
    )["generation"]
    post("/api/steer", {"feature_id": None})
    write(
        "features.json",
        {
            # WHICH MODEL THIS WAS BAKED ON. Without it the shim gated the
            # whole features section on the scenario INDEX's default, which is
            # a Qwen — while every number in this file came from
            # google/gemma-2-2b. The gate was true for exactly the wrong model.
            "model": SAE_MODEL,
            "sae": sae,
            "summary": summary,
            "feature": feature,
            "detail": detail,
            "baseline": baseline,
            "steered": steered,
            "scale": -40,
        },
    )

    # The endpoints the panels call on first paint. Each of these used to
    # answer 409 "not available in the demo", which is how the accelerator
    # badge, the storage panel, the logit lens and the HF tab all rendered as
    # broken rather than as recorded.
    print("\nEnvironment: the small endpoints every panel calls")
    env: dict[str, object] = {}
    for name, path in (
        ("accelerator", "/api/accelerator"),
        ("progress", "/api/model/progress"),
        ("sae_available", "/api/sae/available"),
        ("session_state", "/api/session/state"),
        ("vla_datasets", "/api/vla/datasets"),
        # Public registry facts, not machine facts — safe to publish, and it
        # gives the demo's Ollama tab the same "somewhere to start" the
        # HuggingFace tab has. The fit verdict is dropped below, because a
        # static page does not know the visitor's GPU.
        ("ollama_suggested", "/api/ollama/suggested"),
    ):
        try:
            env[name] = get(path, timeout=120)
            print(f"  {name:<16} ok")
        except urllib.error.HTTPError as err:
            print(f"  {name:<16} skipped ({err.code})")

    # NOTHING ABOUT THE BAKER'S MACHINE IS PUBLISHED.
    #
    # This block used to bake the two endpoints that describe *who and where*
    # the baker is, and both leaked onto the public site:
    #
    #   /api/hub/auth  shipped `{"signed_in": true, "user": "<username>"}`, so
    #                  every visitor saw the baker's HuggingFace account and a
    #                  "sign out" link for it.
    #   /api/paths     shipped real directory paths. The scrub replaced the
    #                  home directory only, so a model cache on another volume
    #                  went out verbatim.
    #
    # Scrubbing is the wrong shape for this: it is a blocklist, and a
    # blocklist is a promise to have thought of everything. These are
    # SYNTHESISED instead — the panels get the right shape and a visitor gets
    # the truth about their own session, which is that nobody is signed in.
    # tests/demo_check.py scans the whole bundle for machine identifiers so a
    # future endpoint cannot reintroduce this quietly.
    env["hub_auth"] = {"signed_in": False, "user": None, "source": None}
    env["paths"] = dict(SYNTHETIC_PATHS)
    # The accelerator is the same class of leak and was missed the first time,
    # because it carries no username and no path — so the machine-identifier
    # scan walked straight past "NVIDIA GeForce RTX 4060 Laptop GPU". A visitor
    # on a phone was shown CUDA, that GPU's name, and 8.6 GB of its VRAM.
    #
    # There is no honest device to report: nothing runs behind this page. The
    # dtype stays because it is a property of the recording rather than of any
    # machine reading it — these attention weights really were produced in
    # bfloat16, and that is worth saying.
    env["accelerator"] = {
        "kind": "recorded",
        "torch_device": None,
        "name": None,
        "vram_gb": None,
        "dtype": "bfloat16",
        "reason": (
            "This page is a recording — nothing runs here, so there is no "
            "device to detect. Installed, ModelMRI finds your own accelerator "
            "(NVIDIA, AMD, Intel or Apple silicon), names it, and explains "
            "which it chose and why."
        ),
    }
    print("  hub_auth         synthesised (signed out)")
    print("  paths            synthesised (no machine paths published)")
    print("  accelerator      synthesised (no device named)")

    # A real .mri of the demo's own run, so "Share this view" produces a file
    # that actually opens in the viewer next door — the one hop the demo was
    # describing but could not perform.
    try:
        import base64

        with urllib.request.urlopen(
            f"{BASE}/api/session/export?layer=0&head=0&note="
            + urllib.parse.quote("Baked from the ModelMRI demo run"),
            timeout=300,
        ) as r:
            env["session_mri"] = base64.b64encode(r.read()).decode("ascii")
        print(f"  session_mri      ok ({len(env['session_mri']) / 1024:.1f} KB b64)")
    except (urllib.error.HTTPError, urllib.error.URLError) as err:
        print(f"  session_mri      skipped ({err})")

    write("env.json", env)

    print("\nAgents: trace")
    traces = get("/api/traces")
    # One entry, not three. The sample trace had been imported into the local
    # database more than once, so the demo shipped two runs with the same name,
    # the same ten steps and the same 17,110 ms — which reads as two real runs
    # that happened to be identical, rather than one sample recorded twice.
    # Only the trace the panel actually opens is published, so the list and the
    # detail view cannot disagree.
    trace = get(f"/api/traces/{traces[0]['id']}") if traces else None
    write("traces.json", {"list": traces[:1], "trace": trace})

    print("\nDiscovery: the models this recording can actually replay")
    # This used to publish `/api/models/discovered` verbatim, generalising only
    # the paths. That left the model ids and sizes intact — which is an
    # inventory of what is in one person's HuggingFace cache, listed on a
    # public website under "On this machine" and annotated "cached, loads
    # offline". For a visitor every word of that is false: it is not their
    # machine, nothing is cached, and clicking a model the recording has no
    # scenario for cannot load anything.
    #
    # Synthesised from the scenarios instead, which is the same rule already
    # applied to hub_auth and paths. The list is now exactly what the demo can
    # replay, so the picker is honest in both directions: everything shown
    # works, and nothing shown belongs to anybody.
    disco = {
        "models": [
            {
                "id": s["id"],
                "name": s["id"],
                "path": None,
                "kind": "demo",
                "size_gb": None,
                "loadable": True,
                "note": "recorded for this demo — opens instantly, nothing to download",
            }
            for s in SCENARIOS
        ],
        "roots": [],
        "truncated": False,
        "demo_note": (
            "This list is the demo's own recordings. Installed, this tab shows "
            "the models already on your machine — your HuggingFace cache, plain "
            "folders and GGUF files — found without you typing a path."
        ),
    }
    write("discovered.json", disco)
    print(f"  discovered       synthesised from {len(disco['models'])} scenario(s)")

    print("\nCustom: the adapter template, inspected")
    try:
        template = str(ROOT / "examples" / "adapter_template.py")
        status = post("/api/custom/load", {"path": template})
        run = post("/api/custom/run", {"shape": None})
        write(
            "custom.json",
            {
                # The path is rewritten so the demo does not publish the
                # baker's directory layout to the internet.
                "status": {**status, "path": "examples/adapter_template.py"},
                "run": run,
                "candidates": {
                    "adapters": [
                        {
                            "path": "examples/adapter_template.py",
                            "name": "adapter_template.py",
                            "dir": "examples",
                            "has_example": True,
                            "hint": True,
                        }
                    ],
                    "torchscript": [],
                    "roots": ["your project directory"],
                },
            },
        )
        post("/api/custom/unload", {})
    except urllib.error.HTTPError as err:
        print(f"  skipped custom ({err.code})")

    print("\nVLA: frames + heatmaps")
    try:
        eps = get("/api/vla/episodes")
        post("/api/vla/load", {})
        post("/api/vla/analyse", {"episode": VLA_EPISODE, "t": VLA_FRAME})
        heat = {
            str(layer): get(f"/api/vla/attention?layer={layer}&head=-1")
            for layer in VLA_LAYERS
        }
        # a short strip of frames so the scrubber works offline
        frames = {
            str(t): get(f"/api/vla/frame?episode={VLA_EPISODE}&t={t}")
            for t in range(VLA_FRAME - 6, VLA_FRAME + 7, 2)
        }
        write(
            "vla.json",
            {
                "dataset": {k: v for k, v in eps.items() if k != "episodes"},
                "episode": VLA_EPISODE,
                "frame": VLA_FRAME,
                "status": get("/api/vla"),
                "layers": VLA_LAYERS,
                "attention": heat,
                "frames": frames,
                # Every series of this episode on ONE axis. Cheap to bake —
                # a few hundred floats per track against the frames' base64
                # megabytes — and it is the one VLA readout that needs no
                # video decode, so the demo can serve the real thing rather
                # than refuse. The frames above are a strip of six; this is
                # all 159 timesteps.
                "timeline": get(f"/api/vla/timeline?episode={VLA_EPISODE}"),
            },
        )
    except urllib.error.HTTPError as err:
        print(
            f"  skipped VLA ({err.code}) - install modelmri[vla-lite] and cache a dataset"
        )

    print(f"\nbundle written to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
