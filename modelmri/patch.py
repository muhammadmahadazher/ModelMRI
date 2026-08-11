"""Where in the model does the answer actually get decided?

Every other ranking in this tool takes one prompt and removes something from
it: a head, a token, an SAE feature. This one takes *two* prompts that differ
in one fact, and moves an activation from the run that knows the answer into
the run that does not. If the answer comes back, that (layer, position) is
carrying the fact. This is causal tracing, and it localises in a way ablation
cannot: ablation says "this mattered", patching says "this is where it is".

    clean    "The Eiffel Tower is located in the city of"  -> " Paris"
    corrupt  "The Colosseum is located in the city of"     -> " P"

Six things were measured before any of this was written, on gpt2 float32 cuda,
and four of them decided the shape of what is here.

THE METRIC IS SIGNED, AND SO IT IS NOT KL. Every other panel reports KL nats,
which is the right answer when the question is "how far did the distribution
move". It is the wrong answer here, because patching has a direction: the
question is whether the answer moved TOWARD the clean run, and a patch can
move it away. Measured: 5 of 132 sites moved it away, the worst by -0.157, and
KL reports those as simply "far" with no sign to tell them from a site that
recovered nothing. KL also disagrees about the ranking -- top-8 by recovery
against bottom-8 by KL-to-clean overlap on only 5 of 8. So the number here is
the fraction of the clean-vs-corrupt logit gap that the patch restores.

THE HARNESS IS EXACT, AT EVERY PRECISION. Patching the corrupted run with its
own activations must be a no-op, and is: the worst |recovery| over all 132
identity patches is exactly 0.0, in float32, bfloat16 AND float16. Replacing a
tensor with itself does no arithmetic, which is why this needs no float32-only
refusal of the kind `feature_ablate` carries. What dtype does change is the
*reference*: in bfloat16 the corrupted prompt's own answer is " T" rather than
" P" and the gap is 4.000 rather than 4.467, so scores are self-consistent
within a dtype and are not comparable across one. Both reference tokens and
the dtype are therefore reported with every result.

A SINGLE CONTROL DRAW IS A COIN FLIP. Patching a same-norm random vector at
the same site is not a formality: over 8 draws at one site the control ran from
-2.038 to +0.616, a spread of 2.654, against a real recovery of +0.427. The
gate moves with the number of draws -- 76 of 132 sites beat one draw, 27 beat
the 95th percentile of eight, 20 beat all eight. One draw would have passed
almost four times as many sites as survive. And it matters at the top: of the
8 highest-recovery sites, 3 clear both controls and 5 do not, including
+0.435 at layer 3 and +0.427 at layer 9.

MOST PAIRS OF PROMPTS CANNOT BE USED AT ALL, and the failures are quiet ones.
Position P of one run only corresponds to position P of another if the two
tokenize to the same length -- 6 of 8 natural minimal pairs did, and the two
that did not differ by 2 tokens with nothing on screen to say so. Worse, the
denominator is the gap between the two answers, and 2 of 3 casually-written
pairs produced the *same* answer, making it exactly 0.000000. Both are
refusals here, and both name what to change.
"""

from __future__ import annotations

import time
from typing import Any

import torch

# Same-norm Gaussian draws per site. Eight, because one is a sample: see the
# module docstring for the 2.654 spread that decided it. The cost is why
# controls run on the top sites rather than the whole grid.
CONTROL_DRAWS = 8

# The seed. Fixed so the same question twice gives the same answer, and stated
# so nobody reads the control as a fresh random number each time.
CONTROL_SEED = 0

# How many sites get controls. The grid is n_layers x n_positions and every
# cell costs a forward pass; controls multiply that by CONTROL_DRAWS + 1. On
# gpt2 the full grid is 132 passes in 3.00 s, and controlling all of it would
# be 1320. Controlling the top slice instead is 132 + 24 * 9 = 348.
MAX_CONTROLLED = 24

# Below this the two prompts do not disagree enough for the fraction to mean
# anything: the denominator is the gap between the two answers, and a gap of
# 0.3158 nats (measured on "The doctor said he" / "The nurse said she") already
# makes a 0.1 movement read as 32% recovered.
MIN_GAP = 0.5


# What gets replaced. The residual stream answers WHERE; the two sublayers
# answer THROUGH WHAT, and they do not agree, which is the reason to run all
# three. Measured on the reference pair: the MLP grid peaks at +0.365 on the
# SUBJECT token in layer 0, the attention grid at +0.232 on the LAST token in
# layer 9. Early MLP writes the fact, late attention moves it to where the
# prediction is made -- and the residual grid, which is their sum, shows only
# the destination.
COMPONENTS = ("resid", "attn", "mlp")


class PatchError(RuntimeError):
    """The two prompts cannot be compared. Always says which one to change."""


def _sublayer(block: torch.nn.Module, kind: str) -> torch.nn.Module:
    """The attention or MLP submodule of a decoder block.

    Two spellings, because the two families this tool supports disagree:
    GPT-2 calls it `attn`, Llama/Qwen/Gemma call it `self_attn`. Both call the
    MLP `mlp`. Refuses rather than guessing, for the same reason
    `ModelRuntime._block` does.
    """
    if kind == "mlp" and hasattr(block, "mlp"):
        return block.mlp
    if kind == "attn":
        for name in ("attn", "self_attn"):
            if hasattr(block, name):
                return getattr(block, name)
    raise PatchError(
        f"This model's blocks expose no '{kind}' submodule, so there is "
        f"nothing to patch there. Known spellings: attn (GPT-2) and self_attn "
        f"(Llama, Qwen, Gemma), plus mlp for both. Ask for component='resid', "
        f"which reads the residual stream and works on any layout."
    )


def _capture_out(module: torch.nn.Module, layer: int, sink: dict):
    """Read a sublayer's OUTPUT. Attention returns a tuple; the MLP does not."""

    def post(module, args, output):
        y = output[0] if isinstance(output, tuple) else output
        sink[layer] = y.detach().clone()

    return module.register_forward_hook(post)


def _splice_out(module: torch.nn.Module, pos: int, vec: torch.Tensor):
    """Replace one position of a sublayer's output.

    Rebuilds the tuple rather than mutating it: attention returns the present
    key/value cache alongside the hidden states on several versions, and
    dropping the tail would silently change what the rest of the block sees.
    """

    def post(module, args, output):
        is_tuple = isinstance(output, tuple)
        y = (output[0] if is_tuple else output).clone()
        y[:, pos, :] = vec
        return (y,) + output[1:] if is_tuple else y

    return module.register_forward_hook(post)


def _tokens(tokenizer: Any, ids: torch.Tensor) -> list[str]:
    return [tokenizer.decode([int(t)]) for t in ids[0]]


def _capture(block: torch.nn.Module, layer: int, sink: dict) -> Any:
    """Read the residual stream entering `block`."""

    def pre(module, args):
        sink[layer] = args[0].detach().clone()

    return block.register_forward_pre_hook(pre)


def _splice(block: torch.nn.Module, pos: int, vec: torch.Tensor) -> Any:
    """Replace one position of the stream entering `block` with `vec`.

    Clones rather than writing in place: the source tensor belongs to the
    cached clean run, and an in-place write would corrupt the cache for every
    later patch. That failure would not raise -- it would quietly make each
    site's number depend on the order the sites were visited.
    """

    def pre(module, args):
        x = args[0].clone()
        x[:, pos, :] = vec
        return (x,) + args[1:]

    return block.register_forward_pre_hook(pre)


def _answer(logits: torch.Tensor, tokenizer: Any) -> dict:
    probs = logits.softmax(-1)
    tid = int(logits.argmax())
    return {
        "id": tid,
        "text": tokenizer.decode([tid]),
        "p": round(float(probs[tid]), 6),
    }


def trace(
    model: Any,
    tokenizer: Any,
    blocks: list,
    clean: str,
    corrupt: str,
    *,
    device: Any,
    draws: int = CONTROL_DRAWS,
    max_controlled: int = MAX_CONTROLLED,
) -> dict:
    """Patch every (layer, position) from the clean run into the corrupted one.

    `blocks` is the decoder block list, passed in rather than found here so
    that the one place which knows the architecture layouts stays the one
    place -- see ModelRuntime._block.
    """
    if not clean.strip() or not corrupt.strip():
        raise PatchError("Both prompts have to have something in them.")
    if clean == corrupt:
        raise PatchError(
            "The two prompts are identical, so there is nothing to trace. "
            "Change one fact in the second one — a name, a number, a place — "
            "and keep everything else the same."
        )

    clean_ids = tokenizer(clean, return_tensors="pt").input_ids.to(device)
    corrupt_ids = tokenizer(corrupt, return_tensors="pt").input_ids.to(device)
    n_pos = clean_ids.shape[1]

    # Refusal 1: position P of one run only means the same thing as position P
    # of the other when the two tokenize the same. This is not rare -- 2 of 8
    # natural minimal pairs failed it -- and it is invisible, because both
    # prompts run fine on their own.
    if corrupt_ids.shape[1] != n_pos:
        raise PatchError(
            f"The two prompts tokenize to different lengths ({n_pos} and "
            f"{corrupt_ids.shape[1]}), so position 3 of one is not position 3 "
            f"of the other and patching them together would compare unrelated "
            f"places. Clean:  {_tokens(tokenizer, clean_ids)}. Corrupt: "
            f"{_tokens(tokenizer, corrupt_ids)}. Change the second prompt so "
            f"it splits into the same number of pieces — a shorter or longer "
            f"name is usually all it takes."
        )

    with torch.no_grad():
        clean_logits = model(clean_ids).logits[0, -1].float()
        corrupt_logits = model(corrupt_ids).logits[0, -1].float()

    a = int(clean_logits.argmax())
    b = int(corrupt_logits.argmax())

    # Refusal 2: the denominator. Measured on three casually-written pairs, two
    # of them produced the SAME next token, which makes the gap exactly
    # 0.000000 and every recovery fraction a division by zero.
    if a == b:
        raise PatchError(
            f"Both prompts predict the same next token "
            f"({tokenizer.decode([a])!r}), so there is no difference for a "
            f"patch to restore. Pick a pair whose answers actually differ."
        )

    def gap_of(logits: torch.Tensor) -> float:
        return float(logits[a] - logits[b])

    ld_clean, ld_corrupt = gap_of(clean_logits), gap_of(corrupt_logits)
    gap = ld_clean - ld_corrupt
    if gap < MIN_GAP:
        raise PatchError(
            f"The two prompts disagree by only {gap:.4f} logits, which is too "
            f"little to divide by: a patch that moves the answer a fraction of "
            f"that would read as a large share of it. Pick a pair whose answers "
            f"differ more clearly."
        )

    n_layers = len(blocks)
    passes = 2  # the two baselines above
    t0 = time.perf_counter()

    def cache_for(component: str) -> dict[int, torch.Tensor]:
        """The clean run's activations at one component, cached once."""
        got: dict[int, torch.Tensor] = {}
        if component == "resid":
            handles = [_capture(b_, i, got) for i, b_ in enumerate(blocks)]
        else:
            handles = [
                _capture_out(_sublayer(b_, component), i, got)
                for i, b_ in enumerate(blocks)
            ]
        try:
            with torch.no_grad():
                model(clean_ids)
        finally:
            for h in handles:
                h.remove()
        return got

    def run_patched(component: str, layer: int, pos: int, vec: torch.Tensor) -> float:
        nonlocal passes
        target = (
            blocks[layer]
            if component == "resid"
            else _sublayer(blocks[layer], component)
        )
        h = (
            _splice(target, pos, vec)
            if component == "resid"
            else _splice_out(target, pos, vec)
        )
        try:
            with torch.no_grad():
                out = model(corrupt_ids).logits[0, -1].float()
        finally:
            h.remove()
        passes += 1
        return (gap_of(out) - ld_corrupt) / gap

    caches: dict[str, dict[int, torch.Tensor]] = {}
    grids: dict[str, list[list[float]]] = {}
    for component in COMPONENTS:
        caches[component] = cache_for(component)
        grids[component] = [
            [
                run_patched(component, li, pi, caches[component][li][:, pi, :])
                for pi in range(n_pos)
            ]
            for li in range(n_layers)
        ]

    # Controls on the strongest sites, PER COMPONENT rather than pooled.
    #
    # Pooling them was the first cut and it was wrong: the residual stream
    # carries both sublayers plus what was already there, so its scores are
    # systematically larger and a shared ranking is not a comparison between
    # equals. Measured, a pooled top-24 came back 20 resid, 2 mlp, 2 attn —
    # so the two grids that answer "through what" got almost no verdicts, and
    # the MLP's own peak was one of only two rows it was allowed.
    per_component = max(1, max_controlled // len(COMPONENTS))
    ranked = [
        (score, c, li, pi)
        for c in COMPONENTS
        for score, li, pi in sorted(
            (
                (grids[c][li][pi], li, pi)
                for li in range(n_layers)
                for pi in range(n_pos)
            ),
            key=lambda z: -z[0],
        )[:per_component]
    ]
    gen = torch.Generator().manual_seed(CONTROL_SEED)
    sites: list[dict] = []
    for score, component, li, pi in ranked:
        real = caches[component][li][:, pi, :]
        norm = real.norm()
        control = []
        for _ in range(draws):
            r = torch.randn(real.shape, generator=gen).to(real.device, real.dtype)
            control.append(run_patched(component, li, pi, r / r.norm() * norm))
        # A second, different question: is it THIS position, or would any
        # activation from this layer do? Cheap (one pass) and it separates a
        # site that carries the fact from a layer that is simply influential.
        alt = (pi + 1) % n_pos
        shifted = run_patched(component, li, pi, caches[component][li][:, alt, :])
        worst = max(control)
        sites.append(
            {
                "component": component,
                "layer": li,
                "position": pi,
                "recovery": round(score, 6),
                "control_max": round(worst, 6),
                "control_min": round(min(control), 6),
                "control_draws": draws,
                "shifted_position": round(shifted, 6),
                # Both, deliberately. Beating noise says the number is not the
                # size of the edit; beating the shifted patch says it is not
                # just the layer. 19 of 132 sites cleared both on the reference
                # pair, against 76 that beat a single draw.
                "clears_control": bool(score > worst),
                "clears_position": bool(score > shifted),
            }
        )

    elapsed = time.perf_counter() - t0
    dtype = str(next(model.parameters()).dtype).removeprefix("torch.")
    return {
        "clean": {
            "prompt": clean,
            "tokens": _tokens(tokenizer, clean_ids),
            "answer": _answer(clean_logits, tokenizer),
        },
        "corrupt": {
            "prompt": corrupt,
            "tokens": _tokens(tokenizer, corrupt_ids),
            "answer": _answer(corrupt_logits, tokenizer),
        },
        "gap": round(gap, 6),
        "n_layers": n_layers,
        "n_positions": n_pos,
        "components": list(COMPONENTS),
        "grids": {
            c: [[round(v, 6) for v in row] for row in grids[c]] for c in COMPONENTS
        },
        "sites": sites,
        "controlled": len(sites),
        "dtype": dtype,
        "passes": passes,
        "seconds": round(elapsed, 2),
        # Not decoration. Layer 0's input IS the embedding, so patching every
        # position of that row restores the clean prompt outright and scores
        # 1.00000 by construction -- measured, not assumed. A reader who sees a
        # bright bottom row should know it is a definition, not a discovery.
        "notes": [
            "Scores are the share of the clean-vs-corrupt logit gap that the "
            "patch restores. 1.0 is the clean answer, 0.0 is the corrupted "
            "one, and negative means the patch pushed the answer further away.",
            "Layer 0's input is the embedding, so patching all of its "
            "positions at once restores the prompt itself and scores 1.0 by "
            "construction.",
            "The residual grid says where; the two sublayer grids say through "
            "what, and they disagree. On the reference pair the MLP grid peaks "
            "at +0.365 on a SUBJECT token in layer 0 and the attention grid at "
            "+0.232 on the LAST token in layer 9 — early MLP writing the fact, "
            "late attention moving it to where the prediction is made.",
            "A sublayer's output at an earlier position cannot reach the "
            "prediction from the final layer, so the last rows of the attn and "
            "mlp grids are exactly 0 everywhere but the last column. That is "
            "the geometry, not a measurement.",
            f"Scores depend on the dtype the model is loaded in: the reference "
            f"tokens themselves change. These were measured in {dtype} against "
            f"{tokenizer.decode([a])!r} and {tokenizer.decode([b])!r}. On the "
            f"same pair, float32 gave ' Paris' against ' P' with a gap of "
            f"4.467, and bfloat16 gave ' Paris' against ' T' with a gap of "
            f"exactly 4.000 — which also quantises the scores themselves into "
            f"steps of an eighth. Compare within a dtype, not across one.",
        ],
    }
