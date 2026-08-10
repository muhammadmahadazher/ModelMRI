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
DEMO_MODEL = "gpt2"
VLA_EPISODE, VLA_FRAME = 3, 60
VLA_LAYERS = [0, 3, 6, 9, 11]

# The LLM bundle bakes EVERY layer/head, not a sample. Three slices used to be
# baked against a meta advertising 12 x 12, so 141 of 144 selections drew a
# different head's arcs than the controls said — and the fallback was silent,
# which is the only kind of wrong nobody reports. Completeness is also nearly
# free here: gpt2's 144 slices of 23 x 23 cost about as much as the robot
# bundle already does.
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


def write(name: str, payload: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"  {name:<22} {path.stat().st_size / 1024:7.1f} KB")


def main() -> int:
    try:
        get("/api/session", timeout=5)
    except urllib.error.URLError:
        print("No server on :5900 - run `modelmri serve` first.")
        return 1

    print("baking demo bundle")

    print("\nLLM: load + generate")
    post("/api/model/load", {"hf_id": "gpt2"})
    baseline = post(
        "/api/model/prompt",
        {"prompt": PROMPT, "max_new_tokens": 12, "temperature": 0},
    )["generation"]

    meta = get("/api/attention/meta")
    n_layers, n_heads = meta["n_layers"], meta["n_heads"]

    print(f"  attention: {n_layers} x {n_heads} = {n_layers * n_heads} slices")
    attn = {
        f"{layer}.{head}": get(f"/api/attention?layer={layer}&head={head}")
        for layer in range(n_layers)
        for head in range(n_heads)
    }

    # Rank heads is the capability the README leads with, and the demo had no
    # handler for it at all — the button answered 409 under advice that could
    # not work. Both baselines, because the panel offers both and they
    # genuinely disagree; plus the whole-model sweep the second button runs.
    print(f"  rankings: {n_layers} layers x 2 baselines + 2 sweeps")
    # NOT `baseline` — that name already holds the generated text, and reusing
    # it here silently wrote the string "mean" into the demo's generation and
    # into features.json. Caught by opening the built demo and reading what it
    # said, which is the only place it was visible.
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
    print(f"  comparisons: {n_layers} layers x {RANKED_ROWS} ranked rows")
    diff: dict[str, dict] = {}
    for layer in range(n_layers):
        rows = ablate[f"{layer}.zero"]["ranked"][:RANKED_ROWS]
        if not rows:
            continue
        at = min(layer + 1, n_layers - 1)
        head = rows[0]["head"]  # rank() selects the top head before compare()
        for row in rows:
            key = f"{at}.{head}.{layer}.{row['head']}"
            diff[key] = get(
                f"/api/attention/diff?layer={at}&head={head}"
                f"&a=live&b=ablate:{layer}.{row['head']}"
            )

    session = get("/api/session")
    write(
        "llm.json",
        {
            "prompt": PROMPT,
            "generation": baseline,
            "meta": meta,
            "layers": list(range(n_layers)),
            "attention": attn,
            "ablate": ablate,
            "diff": diff,
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
        ("lens", "/api/lens?top_k=5"),
        ("session_state", "/api/session/state"),
        ("hub_auth", "/api/hub/auth"),
        ("vla_datasets", "/api/vla/datasets"),
    ):
        try:
            env[name] = get(path, timeout=120)
            print(f"  {name:<16} ok")
        except urllib.error.HTTPError as err:
            print(f"  {name:<16} skipped ({err.code})")

    # `/api/paths` publishes this machine's directory layout. The shapes are
    # real; the paths are generalised, exactly as the discovery bundle does.
    try:
        p = get("/api/paths")
        home = str(Path.home()).replace("\\", "/")

        def scrub(value: object) -> object:
            if isinstance(value, str):
                return value.replace("\\", "/").replace(home, "~")
            if isinstance(value, dict):
                return {k: scrub(v) for k, v in value.items()}
            if isinstance(value, list):
                return [scrub(v) for v in value]
            return value

        env["paths"] = scrub(p)
        print("  paths            ok (generalised)")
    except urllib.error.HTTPError as err:
        print(f"  paths            skipped ({err.code})")

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
    trace = get(f"/api/traces/{traces[0]['id']}") if traces else None
    write("traces.json", {"list": traces[:3], "trace": trace})

    print("\nDiscovery: what is actually on this machine")
    disco = get("/api/models/discovered")
    # The model ids and sizes are real; the paths are this machine's directory
    # layout and have no business on the public internet. Replaced with a
    # plausible generic root so the demo's picker still reads as a real one.
    for m in disco.get("models", []):
        leaf = m["path"].replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
        m["path"] = f"~/.cache/huggingface/hub/{leaf}"
    disco["roots"] = ["~/models"]
    write("discovered.json", disco)

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
