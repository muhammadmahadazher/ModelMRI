# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""Reproduce the steering result from the README, end to end.

    modelmri serve                      # in one terminal
    uv run python examples/steering_demo.py

Loads GPT-2 + the public layer-8 SAE, finds the feature that fires on the
answer token, and shows the same prompt with that concept turned down.
Everything is greedy (temperature 0), so the output is deterministic.
"""

import json
import urllib.request

BASE = "http://127.0.0.1:5900"
PROMPT = "The Eiffel Tower is located in the city of"


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
    print("loading gpt2 ...")
    post("/api/model/load", {"hf_id": "gpt2"})

    print("generating baseline ...")
    base = post(
        "/api/model/prompt",
        {"prompt": PROMPT, "max_new_tokens": 12, "temperature": 0},
    )["generation"]

    print("loading the layer-8 SAE (~150 MB on first run) ...")
    sae = post("/api/sae/load", {})
    # ASCII only: Windows consoles are cp1252 and choke on typographic dashes.
    print(f"  {sae['repo']} @ {sae['hook']} - {sae['d_sae']:,} features")

    summary = get("/api/features/summary?top_k=1")
    answer_idx = next(
        (i for i, t in enumerate(summary["tokens"]) if "Paris" in t),
        len(summary["tokens"]) - 1,
    )
    token = summary["tokens"][answer_idx]
    feature, activation = summary["top"][answer_idx][0]
    print(f"\ntop feature on {token!r}: #{feature} (activation {activation})")

    print(f"steering #{feature} to -40 ...")
    post("/api/steer", {"feature_id": feature, "scale": -40})
    steered = post(
        "/api/model/prompt",
        {"prompt": PROMPT, "max_new_tokens": 12, "temperature": 0},
    )["generation"]
    post("/api/steer", {"feature_id": None})  # always leave the model clean

    print("\n" + "=" * 66)
    print(f"prompt    {PROMPT}")
    print(f"baseline  {base.strip()}")
    print(f"steered   {steered.strip()}")
    print("=" * 66)
    print("\nSame prompt. Same seed. One number changed inside layer 8.")


if __name__ == "__main__":
    main()
