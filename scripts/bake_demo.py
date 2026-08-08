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
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:5900"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "frontend" / "public" / "demo"
PROMPT = "The Eiffel Tower is located in the city of"
VLA_EPISODE, VLA_FRAME = 3, 60
# keep the payload small: these are the layers the UI offers in demo mode
LLM_LAYERS = [0, 6, 11]
VLA_LAYERS = [0, 3, 6, 9, 11]


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
    attn = {
        str(layer): get(f"/api/attention?layer={layer}&head=0") for layer in LLM_LAYERS
    }
    write(
        "llm.json",
        {
            "prompt": PROMPT,
            "generation": baseline,
            "meta": meta,
            "layers": LLM_LAYERS,
            "attention": attn,
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

    print("\nAgents: trace")
    traces = get("/api/traces")
    trace = get(f"/api/traces/{traces[0]['id']}") if traces else None
    write("traces.json", {"list": traces[:3], "trace": trace})

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
