# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""Measure how a robot policy's attention sharpens with depth.

    modelmri serve
    uv run python examples/vla_layers_demo.py

Runs one real PushT frame through SmolVLA's own vision tower and reports,
per layer, how much of the attention mass lands in the top 5% of image
patches. Reproduces the table in the README.
"""

import json
import urllib.request

BASE = "http://127.0.0.1:5900"
EPISODE, FRAME = 3, 60


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)


def get(path: str) -> dict:
    with urllib.request.urlopen(BASE + path, timeout=900) as r:
        return json.load(r)


def main() -> None:
    # ASCII only in console output: Windows consoles are cp1252 and a stray
    # em-dash is enough to crash a script mid-run.
    ds = get("/api/vla/episodes")
    print(f"dataset {ds['repo_id']} - {ds['n_episodes']} episodes @ {ds['fps']} fps")

    print("loading SmolVLA's vision tower ...")
    vla = post("/api/vla/load", {})
    print(
        f"  {vla['repo']} - {vla['n_layers']} layers x {vla['n_heads']} heads, "
        f"{vla['grid'][0]}x{vla['grid'][1]} patches ({vla['warmup_ms']} ms)"
    )

    print(f"running the policy on episode {EPISODE}, frame {FRAME} ...")
    run = post("/api/vla/analyse", {"episode": EPISODE, "t": FRAME})
    print(f"  {run['latency_ms']} ms\n")

    print("layer   attention mass in the top 5% of patches")
    for layer in range(vla["n_layers"]):
        heat = get(f"/api/vla/attention?layer={layer}&head=-1")["heat"]
        flat = sorted((v for row in heat for v in row), reverse=True)
        cut = max(1, len(flat) // 20)
        share = sum(flat[:cut]) / max(sum(flat), 1e-9)
        bar = "#" * round(share * 40)
        print(f"{layer:>5}   {share:5.1%}  {bar}")

    print("\nEarly layers look everywhere. Deep layers lock on.")


if __name__ == "__main__":
    main()
