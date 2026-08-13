"""Regenerate every measured number the documentation quotes.

The README is full of figures — head rankings, patching peaks, pass counts,
bits per weight. Each one was true when it was written and each one was then
copied by hand, which is how a repo ends up quoting a KL that no longer
reproduces. This runs them all against a live model and prints the block to
paste, so a stale number is a diff rather than a discovery.

    python scripts/measure_docs.py --model Qwen/Qwen3-1.7B

Nothing here is hardcoded. Every figure comes from the same functions the
server calls, on whatever model you point it at, and every one prints the
setup that produced it — the model, the dtype, the device, the prompt, the
baseline, the corpus. A number without those cannot be checked by anybody,
which is the whole argument this project makes.

What it does NOT do is decide the numbers are good. It measures and prints;
you read them and paste them.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    # So this runs against the checkout rather than whatever `pip install
    # modelmri` left in site-packages. That shadowing has bitten this project
    # repeatedly, including while writing the very features measured below.
    sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument(
        "--prompt",
        default="The capital of France is",
        help="the prompt every attribution is taken at",
    )
    ap.add_argument("--clean", default="The Eiffel Tower is located in the city of")
    ap.add_argument("--corrupt", default="The Colosseum is located in the city of")
    ap.add_argument("--layer", type=int, default=0, help="layer for the head sweep")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument(
        "--gguf",
        help="a .gguf to measure instead of --model. Every figure below is then "
        "about the QUANTISED weights, dequantised -- which is the point, but it "
        "is a different model from the safetensors of the same name.",
    )
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from modelmri import ablate, budget, corpus, devices, lens, nullmodel, telemetry

    accel = devices.detect()
    dtype_bytes = 2 if accel.dtype in ("float16", "bfloat16") else 4
    out: dict = {
        "model": args.model,
        "device": accel.torch_device,
        "accelerator": accel.name,
        "dtype": accel.dtype,
        "prompt": args.prompt,
    }

    say = (lambda *a: None) if args.json else print
    say(f"# measured on {accel.name} · {accel.dtype} · {args.gguf or args.model}\n")
    if args.gguf:
        say("# these are the QUANTISED weights, dequantised. Not the original.\n")
        # The preflight prints BEFORE the load, because the load is the
        # expensive part and the preflight is what says whether it is worth
        # starting. A GGUF header is a few hundred kilobytes of a multi-gigabyte
        # file, so this costs nothing and can save half an hour.
        from modelmri import gguf_load

        plan = gguf_load.plan(args.gguf, dtype=accel.dtype, device_kind=accel.kind)
        say(
            f"## gguf preflight\n  {plan.file_bytes / 1e9:.3f} GB on disk -> "
            f"{plan.resident_bytes / 1e9:.3f} GB resident at {plan.dtype} "
            f"({plan.expansion:.2f}x), {plan.peak_host_bytes / 1e9:.3f} GB peak"
        )
        say(f"  verdict {plan.verdict}: {plan.why}")
        loaded = gguf_load.load(
            args.gguf,
            dtype=accel.dtype,
            device=accel.torch_device,
            device_kind=accel.kind,
            # Run deliberately, by someone reading the preflight it just
            # printed. A tight fit is theirs to accept; "will not fit" is
            # arithmetic and refuses regardless of this flag.
            confirm=True,
        )
        model, tok = loaded.model, loaded.tokenizer
        args.model = Path(plan.path).name
        out["model"] = args.model
        out["gguf"] = loaded.to_dict()
        out["load_seconds"] = round(loaded.load_seconds, 2)
        say(
            f"  prediction error {loaded.prediction_error:+.6f} "
            f"({loaded.measured_resident_bytes:,} weighed against "
            f"{plan.resident_bytes:,} predicted)"
        )
    else:
        t0 = time.perf_counter()
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=devices.torch_dtype(accel),
            attn_implementation="eager",
        ).to(accel.torch_device)
        model.eval()
        out["load_seconds"] = round(time.perf_counter() - t0, 2)

    cfg = model.config
    n_layers = int(cfg.num_hidden_layers)
    n_heads = int(cfg.num_attention_heads)
    n_params = sum(p.numel() for p in model.parameters())
    out.update(
        n_layers=n_layers,
        n_heads=n_heads,
        n_params=n_params,
        hidden_size=int(getattr(cfg, "hidden_size", 0) or 0),
        n_kv_heads=int(getattr(cfg, "num_key_value_heads", n_heads) or n_heads),
    )
    say(f"{n_params:,} parameters · {n_layers} layers × {n_heads} heads")

    def blocks(i):
        for path in ("model.layers", "transformer.h", "gpt_neox.layers"):
            node = model
            for part in path.split("."):
                node = getattr(node, part, None)
                if node is None:
                    break
            if node is not None:
                return node[i]
        raise SystemExit("cannot find this model's transformer blocks")

    ids = tok(args.prompt, return_tensors="pt").input_ids.to(accel.torch_device)
    size = int(ids.shape[1])
    position = size - 1
    out["prompt_tokens"] = size

    # ---------------------------------------------------------- pass counts
    out["passes_one_layer"] = n_heads + 2
    out["passes_whole_model"] = n_layers * n_heads + 2
    say(
        f"\n## cost\none layer: {out['passes_one_layer']} passes · "
        f"whole model: {out['passes_whole_model']} passes"
    )

    # ------------------------------------------------------- the preflight
    est = ablate.estimate_cost(
        model,
        blocks,
        ids,
        position=position,
        layers=list(range(n_layers)),
        n_heads=n_heads,
        device_kind=accel.kind,
    )["estimate"]
    out["preflight"] = est
    say(
        f"preflight: {est['passes']} passes, {est['seconds']}s projected, "
        f"verdict {est['verdict']} ({est['basis']})"
    )

    # ------------------------------------------------- the three baselines
    say("\n## head ranking, three baselines")
    rankings, summaries = {}, {}
    sentences, corpus_label = corpus.load()
    for baseline in ablate.BASELINES:
        extra = {}
        if baseline == "resample":
            donors = [
                ablate.capture_projection_inputs(model, blocks, d, [args.layer])
                for d in corpus.donor_ids(
                    tok,
                    sentences,
                    at_least=size,
                    want=ablate.RESAMPLE_DRAWS,
                    device=accel.torch_device,
                )
            ]
            extra = {"donors": donors, "corpus": corpus_label}
        r = ablate.rank_heads(
            model,
            blocks,
            ids,
            position=position,
            layers=[args.layer],
            n_heads=n_heads,
            baseline=baseline,
            decode=lambda t: tok.decode([t]),
            **extra,
        )
        rankings[baseline] = r["ranked"]
        top = ", ".join(f"H{x['head']}" for x in r["ranked"][:5])
        summaries[baseline] = {
            "passes": r["passes"],
            "elapsed_s": r["elapsed_s"],
            "top5": [x["head"] for x in r["ranked"][:5]],
            "top_kl": r["ranked"][0]["kl"],
            "target_token": r["target_token"],
            "noise_floor_kl": r["noise_floor_kl"],
        }
        spread = ""
        if baseline == "resample":
            w = max(
                r["ranked"],
                key=lambda x: (x.get("kl_max") or 0) - (x.get("kl_min") or 0),
            )
            summaries[baseline]["widest"] = {
                "head": w["head"],
                "min": w["kl_min"],
                "max": w["kl_max"],
                "median": w["kl"],
            }
            spread = (
                f"   widest H{w['head']} {w['kl_min']}–{w['kl_max']} (median {w['kl']})"
            )
        say(
            f"  {baseline:<9} {r['passes']:>4} passes {r['elapsed_s']:>6.2f}s  top: {top}{spread}"
        )
    out["baselines"] = summaries
    out["corpus"] = corpus_label

    agree = ablate.compare_baselines(rankings, top=5)
    out["disagreement"] = agree["pairs"]
    say("\n  disagreement:")
    for p in agree["pairs"]:
        say(
            f"    {p['baselines'][0]:<9} vs {p['baselines'][1]:<9} "
            f"spearman {p['spearman']}   top-{p['top_k']} disagree on {p['top_k_disagree']}"
        )

    # ------------------------------------------------ the untrained control
    try:
        twin = nullmodel.build_twin(
            cfg, seed=0, dtype=model.dtype, device=accel.torch_device
        )
        try:
            ctrl = ablate.rank_heads(
                twin,
                lambda i: _blocks_of(twin, i),
                ids,
                position=position,
                layers=[args.layer],
                n_heads=n_heads,
                baseline="zero",
                decode=lambda t: tok.decode([t]),
            )
            cmp = ablate.compare_baselines(
                {"model": rankings["zero"], "untrained": ctrl["ranked"]}, top=5
            )["pairs"][0]
            out["control"] = {
                "spearman": cmp["spearman"],
                "top_k": cmp["top_k"],
                "top_k_shared": cmp["top_k_shared"],
                "untrained_top5": [x["head"] for x in ctrl["ranked"][:5]],
                "untrained_top_kl": ctrl["ranked"][0]["kl"],
                "verdict": nullmodel.verdict(
                    cmp["spearman"],
                    top_k_shared=cmp["top_k_shared"],
                    top_k=cmp["top_k"],
                ),
            }
            say(
                f"\n## untrained control\n  spearman {cmp['spearman']}, "
                f"sharing {cmp['top_k_shared']} of the top {cmp['top_k']}"
            )
            say(
                f"  untrained top: {[x['head'] for x in ctrl['ranked'][:5]]} "
                f"(top KL {ctrl['ranked'][0]['kl']} vs {rankings['zero'][0]['kl']} trained)"
            )
        finally:
            nullmodel.teardown(twin)
    except Exception as err:  # a control that cannot be built is not a failure
        out["control"] = {"unavailable": f"{type(err).__name__}"}
        say(f"\n## untrained control\n  unavailable ({type(err).__name__})")

    # ------------------------------------------------------------ the lens
    lr = lens.logit_lens(model, tok, ids, top_k=1)
    rel = lr["reliability"]
    out["lens"] = {
        "final": lr["final"],
        "settled_at": lr["settled_at"],
        "best_kl": rel.get("best_kl"),
        "median_kl": rel.get("median_kl"),
        "floor_kl": rel.get("floor_kl"),
        "usable": rel.get("usable"),
        "trajectory": [
            {"layer": r["layer"], "token": r["tokens"][0], "kl": r["kl_to_final"]}
            for r in lr["layers"]
        ],
    }
    say(
        f"\n## logit lens\n  answer {lr['final']!r}, settles at layer {lr['settled_at']}"
    )
    say(
        f"  best row {rel.get('best_kl')} nats, median {rel.get('median_kl')}, "
        f"floor {rel.get('floor_kl')}, usable={rel.get('usable')}"
    )

    # --------------------------------------------------------- the patching
    try:
        from modelmri import patch as patch_mod

        pr = patch_mod.trace(
            model,
            tok,
            [blocks(i) for i in range(n_layers)],
            args.clean,
            args.corrupt,
            device=accel.torch_device,
        )
        clean_tokens = (pr.get("clean") or {}).get("tokens") or []
        peaks = {}
        for comp, grid in (pr.get("grids") or {}).items():
            best = None
            for li, row in enumerate(grid):
                for pi, v in enumerate(row):
                    # Layer 0's input IS the embedding, so patching every
                    # position of that row restores the clean prompt outright
                    # and scores 1.0 by construction. patch.py says so in its
                    # own notes; reporting it as the peak would be quoting a
                    # definition as a discovery.
                    if li == 0:
                        continue
                    if best is None or v > best["score"]:
                        best = {
                            "layer": li,
                            "pos": pi,
                            "score": round(v, 4),
                            "token": clean_tokens[pi]
                            if pi < len(clean_tokens)
                            else "?",
                        }
            if best:
                peaks[comp] = best
        out["patching"] = {
            "clean": args.clean,
            "corrupt": args.corrupt,
            "passes": pr.get("passes"),
            "seconds": pr.get("seconds"),
            "peaks": peaks,
        }
        say(f"\n## patching\n  {pr.get('passes')} passes in {pr.get('seconds')}s")
        for comp, best in peaks.items():
            say(
                f"    {comp:<6} +{best['score']} · L{best['layer']} · "
                f"pos {best['pos']} {best['token']!r}"
            )
    except Exception as err:
        out["patching"] = {"unavailable": f"{type(err).__name__}: {str(err)[:90]}"}
        say(f"\n## patching\n  unavailable ({type(err).__name__})")

    # -------------------------------------------------------- the telemetry
    out["introspection_bytes_at_512"] = telemetry.eager_attention_bytes(
        n_layers, n_heads, 512, dtype_bytes
    )
    out["introspection_bytes_at_4096"] = telemetry.eager_attention_bytes(
        n_layers, n_heads, 4096, dtype_bytes
    )
    mem = budget.free_memory(accel.kind)
    out["free_bytes"] = mem.free_bytes
    say(
        f"\n## what introspection costs\n"
        f"  {n_layers}L × {n_heads}H × S² × {dtype_bytes}B — "
        f"{out['introspection_bytes_at_512'] / 1e6:.0f} MB at 512 tokens, "
        f"{out['introspection_bytes_at_4096'] / 1e9:.2f} GB at 4096"
    )

    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"\n(json: rerun with --json to get {len(out)} fields for scripting)")
    return 0


def _blocks_of(m, index: int):
    for path in ("model.layers", "transformer.h", "gpt_neox.layers"):
        node = m
        for part in path.split("."):
            node = getattr(node, part, None)
            if node is None:
                break
        if node is not None:
            return node[index]
    raise SystemExit("cannot find the twin's transformer blocks")


if __name__ == "__main__":
    raise SystemExit(main())
