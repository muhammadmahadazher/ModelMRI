# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""Full-stack smoke check — the load-bearing paths, against a live server.

    modelmri serve
    uv run python tests/e2e_check.py

Unlike tests/test_smoke.py (fast, no downloads, runs in CI) this drives the
real thing: real models, real SAEs, real robot frames. Run it before any
release. Exit code is non-zero if anything fails.

It does NOT drive every feature, and the docstring used to say it did.
Measured by intersecting the routes declared in modelmri/server.py against
the paths named in this file: 22 of 51 before the three attention
interventions below were added, 25 after. Among the 26 it still never calls
are /api/lens, /api/accelerator and the whole /api/session export/open/close
round trip. Most of that gap is older than any one feature, and "run it
before a release" is still the right instruction — but a green run here is
not a statement about a route this file does not name, and reading it as one
is how a new endpoint ships unexercised.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:5900"
PASS, FAIL = [], []


def check(name: str, condition: bool, detail: str = "") -> bool:
    (PASS if condition else FAIL).append(name)
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {name}{(' - ' + detail) if detail else ''}")
    return condition


def get(path: str, timeout: float = 900) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def post(path: str, body: dict, timeout: float = 900) -> tuple[int, dict]:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def section(title: str) -> None:
    print(f"\n{title}")


def main() -> int:
    section("1. server + static app")
    code, sess = get("/api/session")
    check("GET /api/session", code == 200 and sess.get("app") == "modelmri")
    with urllib.request.urlopen(BASE + "/") as r:
        html = r.read().decode()
        cache = r.headers.get("cache-control", "")
    check("index served", '<div id="root"></div>' in html or "root" in html)
    check("index is no-cache", "no-cache" in cache, cache)
    check("react bundle referenced", "/app/assets/" in html)
    asset = html.split('src="')[1].split('"')[0] if 'src="' in html else ""
    if asset:
        with urllib.request.urlopen(BASE + asset) as r:
            check("js bundle 200", r.status == 200, f"{len(r.read())} bytes")

    section("2. model discovery")
    code, local = get("/api/models/local")
    check(
        "GET /api/models/local",
        code == 200 and isinstance(local, list),
        f"{len(local)} cached models",
    )
    code, oll = get("/api/ollama")
    check(
        "GET /api/ollama (graceful when off)",
        code == 200 and "up" in oll,
        f"up={oll.get('up')}",
    )

    section("3. model load + generation")
    code, prog = get("/api/model/progress")
    check(
        "GET /api/model/progress idle",
        code == 200 and prog.get("active") is False,
        f"stage={prog.get('stage')!r}",
    )
    t0 = time.time()
    code, st = post("/api/model/load", {"hf_id": "gpt2"})
    check(
        "POST /api/model/load gpt2",
        code == 200 and st.get("loaded"),
        f"{time.time() - t0:.1f}s",
    )
    code, prog = get("/api/model/progress")
    check(
        "progress settles to ready after a load",
        code == 200 and prog.get("active") is False and prog.get("stage") == "ready",
        f"{prog.get('stage')} in {prog.get('elapsed_s')}s",
    )
    code, gen = post(
        "/api/model/prompt",
        {
            "prompt": "The Eiffel Tower is located in the city of",
            "max_new_tokens": 12,
            "temperature": 0,
        },
    )
    baseline = gen.get("generation", "")
    check(
        "POST /api/model/prompt", code == 200 and len(baseline) > 0, repr(baseline[:40])
    )
    check(
        "greedy output is deterministic-looking",
        "Paris" in baseline,
        repr(baseline[:30]),
    )

    section("4. attention")
    code, meta = get("/api/attention/meta")
    check(
        "GET /api/attention/meta available",
        code == 200 and meta.get("available"),
        f"{meta.get('n_layers')}L x {meta.get('n_heads')}H x {meta.get('n_tokens')} tok",
    )
    code, attn = get("/api/attention?layer=6&head=0")
    ok = code == 200 and len(attn.get("matrix", [])) > 0
    check("GET /api/attention", ok)
    if ok:
        row = attn["matrix"][-1]
        check(
            "attention rows are a distribution",
            abs(sum(row) - 1.0) < 0.02,
            f"last row sums to {sum(row):.3f}",
        )
        check("tokens align with matrix", len(attn["tokens"]) == len(attn["matrix"]))
    code, _ = get("/api/attention?layer=999")
    check("bad layer -> 422", code == 422)

    # The three interventions. They were the largest hole in this file: it
    # drove the heat map and none of the things built on top of it, so a
    # ranking endpoint could return a confidently wrong shape and still leave
    # a green release check.
    code, rank = get("/api/attention/ablate?layer=0")
    ok = code == 200 and rank.get("ranked")
    check("GET /api/attention/ablate", ok, f"{rank.get('passes')} passes")
    if ok:
        scores = [r["kl"] for r in rank["ranked"]]
        check("head scores are ordered", scores == sorted(scores, reverse=True))
        check(
            "the ablation baseline is named in the response",
            bool(rank.get("baseline")) and bool(rank.get("means")),
            str(rank.get("baseline")),
        )

    code, diff = get("/api/attention/diff?layer=0&head=0&a=live&b=live")
    check(
        "GET /api/attention/diff",
        code == 200 and ("matrix" in diff or "note" in diff),
        str(diff.get("note", ""))[:60],
    )

    code, attr = get("/api/attention/attribute")
    ok = code == 200 and attr.get("ranked")
    check(
        "GET /api/attention/attribute",
        ok,
        f"{attr.get('passes')} passes, floor {attr.get('noise_floor_kl')}"
        if ok
        else str(attr)[:80],
    )
    if ok:
        check(
            "the token ranking states its window and its floor",
            attr.get("tested_span") is not None
            and attr.get("noise_floor_kl") is not None
            and "not found unimportant" in attr.get("coverage", ""),
        )
        # "typed" is a claim about the user's own words. A row that is neither
        # inside the located span nor before the end of the prompt must not
        # carry it, and both of the other labels used to collapse into it.
        groups = {r["group"] for r in attr["ranked"]}
        check(
            "every row carries a group the server can justify",
            groups <= {"typed", "template", "generated", "unknown"},
            str(sorted(groups)),
        )
    code, _ = get("/api/attention/attribute?position=999999")
    check("attributing outside the sequence -> 422", code == 422)

    section("5. SAE features + steering")
    code, sae = post("/api/sae/load", {})
    check(
        "POST /api/sae/load",
        code == 200 and sae.get("loaded"),
        f"{sae.get('d_sae')} features @ {sae.get('hook')}",
    )
    code, feats = get("/api/features/summary?top_k=3")
    ok = code == 200 and len(feats.get("tokens", [])) > 0
    check("GET /api/features/summary", ok)
    feature = None
    if ok:
        idx = next((i for i, t in enumerate(feats["tokens"]) if "Paris" in t), -1)
        check("found a feature on the answer token", idx >= 0)
        if idx >= 0:
            feature, act = feats["top"][idx][0]
            check("feature has real activation", act > 1.0, f"#{feature} @ {act}")
    if feature is not None:
        code, det = get(f"/api/features/{feature}")
        check("GET /api/features/{id}", code == 200 and "activations" in det)
        code, steer = post("/api/steer", {"feature_id": feature, "scale": -40})
        check("POST /api/steer on", code == 200 and steer.get("active"))
        code, out = post(
            "/api/model/prompt",
            {
                "prompt": "The Eiffel Tower is located in the city of",
                "max_new_tokens": 12,
                "temperature": 0,
            },
        )
        steered = out.get("generation", "")
        check(
            "steering changes the output",
            steered != baseline,
            f"{baseline.strip()[:22]!r} -> {steered.strip()[:22]!r}",
        )
        post("/api/steer", {"feature_id": None})
        code, out2 = post(
            "/api/model/prompt",
            {
                "prompt": "The Eiffel Tower is located in the city of",
                "max_new_tokens": 12,
                "temperature": 0,
            },
        )
        check(
            "clearing steer restores baseline exactly",
            out2.get("generation") == baseline,
        )

    section("6. agent traces")
    doc = {
        "name": "e2e-check-run",
        "started_at": "2026-08-07T00:00:00Z",
        "steps": [
            {"kind": "llm_call", "name": "plan", "started_ms": 0, "duration_ms": 100},
            {
                "kind": "tool_call",
                "name": "boom",
                "started_ms": 120,
                "duration_ms": 50,
                "error": True,
                "output": "exploded",
            },
        ],
    }
    code, imp = post("/api/traces/import", doc)
    check("POST /api/traces/import", code == 200 and "id" in imp)
    tid = imp.get("id")
    code, lst = get("/api/traces")
    check("GET /api/traces", code == 200 and any(t["id"] == tid for t in lst))
    code, tr = get(f"/api/traces/{tid}")
    check("GET /api/traces/{id}", code == 200 and len(tr.get("steps", [])) == 2)
    check("error step preserved", tr["steps"][1]["error"] is True)
    code, _ = get("/api/traces/nope")
    check("unknown trace -> 404", code == 404)
    code, _ = post("/api/traces/import", {"name": "x", "steps": [{"kind": "bogus"}]})
    check("bad step kind -> 422", code == 422)

    section("7. robot policy (VLA)")
    code, eps = get("/api/vla/episodes")
    have_data = code == 200 and eps.get("n_episodes", 0) > 0
    check(
        "GET /api/vla/episodes",
        have_data,
        f"{eps.get('n_episodes')} episodes" if have_data else str(eps)[:60],
    )
    if have_data:
        code, fr = get("/api/vla/frame?episode=3&t=60")
        check(
            "GET /api/vla/frame",
            code == 200 and fr.get("image", "").startswith("data:image/png"),
            f"{fr.get('width')}x{fr.get('height')}",
        )
        check(
            "frame carries state + action",
            len(fr.get("state", [])) > 0 and len(fr.get("action", [])) > 0,
        )
        code, _ = get("/api/vla/frame?episode=3&t=99999")
        check("out-of-range frame -> 422", code == 422)

        t0 = time.time()
        code, vst = post("/api/vla/load", {})
        check(
            "POST /api/vla/load",
            code == 200 and vst.get("loaded"),
            f"{vst.get('n_layers')}L x {vst.get('n_heads')}H, {time.time() - t0:.1f}s",
        )
        code, run = post("/api/vla/analyse", {"episode": 3, "t": 60})
        check(
            "POST /api/vla/analyse",
            code == 200 and run.get("layers", 0) > 0,
            f"{run.get('latency_ms')} ms",
        )

        shares = []
        for layer in (0, 6, 11):
            code, heat = get(f"/api/vla/attention?layer={layer}&head=-1")
            if code != 200:
                check(f"attention layer {layer}", False, str(heat)[:50])
                continue
            flat = sorted((v for row in heat["heat"] for v in row), reverse=True)
            share = sum(flat[: max(1, len(flat) // 20)]) / max(sum(flat), 1e-9)
            shares.append(share)
            check(f"attention layer {layer}", True, f"top-5% mass {share:.1%}")
        if len(shares) == 3:
            check(
                "attention sharpens with depth",
                shares[2] > shares[0],
                f"{shares[0]:.1%} -> {shares[2]:.1%}",
            )
        code, _ = get("/api/vla/attention?layer=999")
        check("bad VLA layer -> 422", code == 422)

    print("\n" + "=" * 60)
    print(f"  {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for name in FAIL:
            print(f"    FAILED: {name}")
    print("=" * 60)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
