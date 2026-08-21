"""Where in the model does the answer actually get decided?

Every other ranking in this tool takes one prompt and removes something from
it: a head, a token, an SAE feature. This one takes *two* prompts that differ
in one fact, and moves an activation from the run that knows the answer into
the run that does not. If the answer comes back, that (layer, position) is
carrying the fact. This is causal tracing, and it localises in a way ablation
cannot: ablation says "this mattered", patching says "this is where it is".

    clean    "The Eiffel Tower is located in the city of"  -> " Paris"
    corrupt  "The Colosseum is located in the city of"     -> " P"

Six things were measured before any of this was written, in float32 on cuda,
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

from . import fmt

# Same-norm Gaussian draws per site. Eight, because one is a sample: see the
# module docstring for the 2.654 spread that decided it. The cost is why
# controls run on the top sites rather than the whole grid.
CONTROL_DRAWS = 8

# The seed. Fixed so the same question twice gives the same answer, and stated
# so nobody reads the control as a fresh random number each time.
CONTROL_SEED = 0

# How many sites get controls. The grid is n_layers x n_positions and every
# cell costs a forward pass; controls multiply that by CONTROL_DRAWS + 1. On
# a 12-layer model over an 11-token prompt the full grid is 132 passes, and
# controlling all of it would
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
    skipped: list[str] = []
    for component in COMPONENTS:
        # A model with no submodule of this name still has a residual stream,
        # and the residual grid is the one that answers "where". Refusing the
        # whole trace because a sublayer could not be found threw away the two
        # thirds that would have worked — so the missing component is recorded
        # and the rest is measured.
        try:
            caches[component] = cache_for(component)
        except PatchError as err:
            # A PatchError is this project's own class and carries an
            # authored sentence, so the text is fine here -- but these notes
            # travel: they go into the response AND, since sessions carry a
            # patch trace, into a `.mri` somebody forwards. Stated so the next
            # edit does not swap in a library exception without noticing.
            skipped.append(f"{component}: {err}")  # leak-ok: PatchError is authored
            continue
        grids[component] = [
            [
                run_patched(component, li, pi, caches[component][li][:, pi, :])
                for pi in range(n_pos)
            ]
            for li in range(n_layers)
        ]

    if not grids:
        raise PatchError(
            "None of the components could be read on this architecture. "
            + " ".join(skipped)
        )

    # Controls on the strongest sites, PER COMPONENT rather than pooled.
    #
    # Pooling them was the first cut and it was wrong: the residual stream
    # carries both sublayers plus what was already there, so its scores are
    # systematically larger and a shared ranking is not a comparison between
    # equals. Measured, a pooled top-24 came back 20 resid, 2 mlp, 2 attn —
    # so the two grids that answer "through what" got almost no verdicts, and
    # the MLP's own peak was one of only two rows it was allowed.
    # Over the components that were MEASURED, not the three this module knows
    # about. `ranked` below iterates `grids`, so dividing the budget by 3 when
    # only two were readable spent two thirds of it and silently controlled
    # fewer sites than asked for.
    per_component = max(1, max_controlled // len(grids))
    ranked = [
        (score, c, li, pi)
        for c in grids
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
    # The same blind spot the edge trace has, in the same formula: two sites
    # closer than this are tied rather than ranked.
    resolution = recovery_resolution(model, clean_logits, gap)
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
        "components": list(grids),
        # OVER `grids`, not over COMPONENTS. The loop above deliberately
        # catches a PatchError from a component this architecture does not
        # expose, records it and continues so the rest is still measured --
        # and then this rebuilt the response from the fixed tuple and raised
        # `KeyError: 'mlp'`, after every one of the 2 x n_layers x n_positions
        # forward passes had already been spent. The entire recovery path was
        # unreachable: a Mixtral or OLMoE, whose blocks name the sublayer
        # `block_sparse_moe`, paid for the whole trace and got a 500.
        "grids": {c: [[round(v, 6) for v in row] for row in grids[c]] for c in grids},
        # And the reader is told which one was dropped and why. `skipped` was
        # collected, commented, and then never put in the payload -- so even
        # with the KeyError fixed, two grids would have arrived looking like
        # the whole answer.
        "skipped": skipped,
        "sites": sites,
        "controlled": len(sites),
        "dtype": dtype,
        # Two sites closer than this are tied, not ranked. In bfloat16 the
        # logits can reach a magnitude where one representable step is large,
        # so a recovery fraction lands on a coarse grid of `step / gap`.
        # Every score in `grids` and `sites` shares it.
        # NOT rounded. This is one step of the model's number format on the
        # recovery scale, and in float32 it lives at 1e-6 and below: measured
        # at |logit|max 22 over a gap of 4 it is 6.557e-07, which `round(_, 6)`
        # turned into 1e-06 — most of the figure gone — and which becomes
        # exactly 0.0 whenever the gap is wider. A tie threshold of 0.0 says
        # nothing is tied, which is the opposite of what this number is for,
        # and no formatter downstream can recover a value already flattened
        # here. bfloat16 was the only dtype this survived, because there the
        # resolution is ~1e-2.
        "recovery_resolution": resolution,
        "passes": passes,
        "seconds": round(elapsed, 2),
        # Not decoration. Layer 0's input IS the embedding, so patching every
        # position of that row restores the clean prompt outright and scores
        # 1.00000 by construction -- measured, not assumed. A reader who sees a
        # bright bottom row should know it is a definition, not a discovery.
        "notes": [
            (
                "Scores are the share of the clean-vs-corrupt logit gap that the "
                "patch restores. 1.0 is the clean answer, 0.0 is the corrupted "
                "one, and negative means the patch pushed the answer further away. "
                "It is a share, not a percentage of a whole: one site can overshoot "
                "the clean run, and one does — gemma-3-270m-it reads 1.010 at its "
                "last layer on the reference pair."
            ),
            (
                "Layer 0's input is the embedding, so patching all of its "
                "positions at once restores the prompt itself and scores 1.0 by "
                "construction."
            ),
            (
                "The residual grid says where; the two sublayer grids say through "
                "what, and they disagree. On the reference pair the MLP grid peaks "
                "at +0.365 on a SUBJECT token in layer 0 and the attention grid at "
                "+0.232 on the LAST token in layer 9 — early MLP writing the fact, "
                "late attention moving it to where the prediction is made."
            ),
            (
                "A sublayer's output at an earlier position cannot reach the "
                "prediction from the final layer, so the last rows of the attn and "
                "mlp grids are exactly 0 everywhere but the last column. That is "
                "the geometry, not a measurement."
            ),
            (
                f"Scores depend on the dtype the model is loaded in: the reference "
                f"tokens themselves change. These were measured in {dtype} against "
                f"{tokenizer.decode([a])!r} and {tokenizer.decode([b])!r}. On the "
                f"same pair, float32 gave ' Paris' against ' P' with a gap of "
                f"4.467, and bfloat16 gave ' Paris' against ' T' with a gap of "
                f"exactly 4.000 — which also quantises the scores themselves into "
                f"steps of an eighth. Compare within a dtype, not across one."
            ),
        ],
    }


# ---------------------------------------------------------------- edges
#
# The node grid answers WHERE: "position 7, layer 12 carries the answer". It
# cannot answer what put it there, because patching a residual stream restores
# everything that ever wrote into it at once.
#
# Path patching splits that. Take one bright cell as the RECEIVER, then for
# each earlier component -- one attention head, or one MLP -- add just that
# component's clean contribution into the receiver's residual input, with
# everything else still corrupt. A sender that recovers the answer on its own
# is the thing that wrote it. "Position 7 layer 12 matters" becomes "head 9.6
# wrote it."


# How many senders get the full control treatment. Same reasoning as
# MAX_CONTROLLED for nodes: every sender is scored, but the eight same-norm
# draws plus the shifted-position pass cost nine more forward passes each, and
# running them on a sender that scored near zero buys nothing. What is left
# out is NAMED in the response rather than implied by its absence.
MAX_CONTROLLED_EDGES = 12


def _write_of_head(projection, packed, position: int, head: int, head_dim: int):
    """What one head wrote into the residual stream at `position`.

    The out-projection's INPUT is the heads side by side; after it they are
    summed and no slice belongs to any one of them. `ablate.head_geometry`
    supplies the width because `hidden_size // n_heads` is wrong by 2x on
    Qwen3-0.6B and wrong on gemma-3-270m-it, and a wrong head_dim silently
    reads half of one head and half of the next.
    """
    weight = projection.weight.detach()
    # Conv1D (GPT-2) stores [in, out]; nn.Linear stores [out, in]. Getting
    # this backwards attributes every head to a different head's slot.
    if getattr(projection, "in_features", None) is None:
        weight = weight.T
    span = slice(head * head_dim, (head + 1) * head_dim)
    return weight[:, span].float() @ packed[0, position, span].float()


def _add_at(block: torch.nn.Module, pos: int, delta: torch.Tensor):
    """Add a vector into one position of a block's residual INPUT.

    A pre-hook and an ADDITION, not a replacement. The receiver's residual
    input on the corrupt run already holds everything the corrupt prompt
    wrote; replacing it would restore every sender at once, which is the node
    patch and the thing this is trying to take apart. Adding
    (clean_sender - corrupt_sender) swaps exactly one contribution.
    """

    def pre(module, args):
        stream = args[0]
        if pos >= stream.shape[1]:
            return None
        edited = stream.clone()
        edited[0, pos, :] = edited[0, pos, :] + delta.to(edited.dtype).to(edited.device)
        return (edited,) + args[1:]

    return block.register_forward_pre_hook(pre)


def recovery_resolution(model: Any, logits: torch.Tensor, gap: float) -> float:
    """The smallest change in a recovery fraction this dtype can express.

    A recovery is `(gap_of(out) - ld_corrupt) / gap`, and every term is a
    difference of logits. In bfloat16 a logit near 128 has a representable
    step of 1.0 -- so the numerator moves in whole units and the fraction
    lands on a grid of `step / gap`.

    MEASURED, not predicted: with the reference pair, every sender in
    a path trace scored a multiple of the same step and a dozen of them tied
    exactly.
    Ranking those against each other is reading noise, and without this number
    on screen there is nothing to say so. It is reported beside the scores so
    a reader can see which part of the ordering is real.
    """
    step = float(logits.abs().max()) * float(
        torch.finfo(next(model.parameters()).dtype).eps
    )
    return step / gap if gap else 0.0


def path_trace(
    model: Any,
    tokenizer: Any,
    blocks: list,
    clean: str,
    corrupt: str,
    *,
    device: Any,
    receiver_layer: int,
    receiver_position: int,
    draws: int = CONTROL_DRAWS,
    max_controlled: int = MAX_CONTROLLED_EDGES,
) -> dict:
    """Which earlier component wrote what makes this receiver matter.

    Scored with the SAME fraction `trace` uses -- (gap_of(out) - ld_corrupt) /
    gap -- so an edge number and a node number are on one scale and can be read
    together. Both of `trace`'s controls run here too: eight same-norm random
    draws, and the same edit taken from a different position.

    V1 DOES NOT SPLIT Q/K/V. A sender is patched into the receiver's residual
    input as a whole, so this says "head 9.6 wrote what layer 12 reads", not
    "head 9.6 reached layer 12 through its query". Freezing q/k/v across GQA,
    fused QKV and rotary embeddings in arbitrary HuggingFace architectures is
    the fiddliest thing this package could attempt, and getting it subtly wrong
    produces confident, ordered, plausible and wrong numbers. The scope that
    ran is named in the response rather than left to be assumed.
    """
    from . import ablate

    t0 = time.perf_counter()
    n_layers = len(blocks)
    if not 0 <= receiver_layer < n_layers:
        raise PatchError(f"layer {receiver_layer} is outside this model's {n_layers}.")
    if receiver_layer == 0:
        raise PatchError(
            "layer 0 has no earlier component to have written into it. Pick a "
            "receiver deeper than the first block."
        )

    clean_ids = tokenizer(clean, return_tensors="pt")["input_ids"].to(device)
    corrupt_ids = tokenizer(corrupt, return_tensors="pt")["input_ids"].to(device)
    if clean_ids.shape != corrupt_ids.shape:
        raise PatchError(
            f"these prompts tokenise to {clean_ids.shape[-1]} and "
            f"{corrupt_ids.shape[-1]} tokens. Patching needs position N to "
            f"mean the same thing in both runs."
        )
    n_pos = int(clean_ids.shape[-1])
    if not 0 <= receiver_position < n_pos:
        raise PatchError(
            f"position {receiver_position} is outside these {n_pos} tokens."
        )

    n_heads = int(model.config.num_attention_heads)
    senders = range(receiver_layer)

    def capture(ids):
        """Per-layer out-projection inputs and MLP outputs for one run."""
        packed: dict[int, torch.Tensor] = {}
        mlp: dict[int, torch.Tensor] = {}
        handles = []

        def catch(layer: int):
            def pre(module, args):
                packed[layer] = args[0].detach().clone()

            return ablate.out_projection(blocks[layer]).register_forward_pre_hook(pre)

        try:
            for layer in senders:
                handles.append(catch(layer))
                handles.append(
                    _capture_out(_sublayer(blocks[layer], "mlp"), layer, mlp)
                )
            with torch.no_grad():
                logits = model(ids).logits[0, -1].float()
        finally:
            for handle in handles:
                handle.remove()
        return packed, mlp, logits

    clean_packed, clean_mlp, clean_logits = capture(clean_ids)
    corrupt_packed, corrupt_mlp, corrupt_logits = capture(corrupt_ids)

    a = int(clean_logits.argmax())
    b = int(corrupt_logits.argmax())
    if a == b:
        raise PatchError(
            "both prompts predict the same token, so there is no answer for a "
            "patch to restore. Pick a pair whose answers actually differ."
        )

    def gap_of(logits: torch.Tensor) -> float:
        return float(logits[a] - logits[b])

    ld_clean, ld_corrupt = gap_of(clean_logits), gap_of(corrupt_logits)
    gap = ld_clean - ld_corrupt
    if gap < MIN_GAP:
        raise PatchError(
            f"The two prompts disagree by only {gap:.4f} logits, which is too "
            f"little to divide by: a patch that moves the answer a fraction of "
            f"that would read as a large share of it. Pick a pair whose "
            f"answers differ more clearly."
        )

    passes = 2

    def run_with(delta: torch.Tensor) -> float:
        nonlocal passes
        handle = _add_at(blocks[receiver_layer], receiver_position, delta)
        try:
            with torch.no_grad():
                out = model(corrupt_ids).logits[0, -1].float()
        finally:
            handle.remove()
        passes += 1
        return (gap_of(out) - ld_corrupt) / gap

    def delta_of(layer: int, head: int | None, position: int):
        """(clean - corrupt) for one sender at one position."""
        if head is None:
            return (
                clean_mlp[layer][0, position, :].float()
                - corrupt_mlp[layer][0, position, :].float()
            )
        projection = ablate.out_projection(blocks[layer])
        head_dim = ablate.head_geometry(blocks[layer], n_heads)
        return _write_of_head(
            projection, clean_packed[layer], position, head, head_dim
        ) - _write_of_head(projection, corrupt_packed[layer], position, head, head_dim)

    scored: list[dict] = []
    for layer in senders:
        for head in list(range(n_heads)) + [None]:
            delta = delta_of(layer, head, receiver_position)
            scored.append(
                {
                    "layer": layer,
                    "head": head,
                    "name": f"L{layer}H{head}" if head is not None else f"L{layer} MLP",
                    "recovery": round(run_with(delta), 6),
                    "delta_norm": round(float(delta.norm()), 6),
                }
            )

    scored.sort(key=lambda row: -row["recovery"])
    gen = torch.Generator().manual_seed(CONTROL_SEED)
    controlled = 0
    for row in scored[:max_controlled]:
        delta = delta_of(row["layer"], row["head"], receiver_position)
        norm = delta.norm()
        control = []
        for _ in range(draws):
            noise = torch.randn(delta.shape, generator=gen).to(delta.device)
            control.append(run_with(noise / noise.norm() * norm))
        # The same second question the node grid asks: is it THIS position, or
        # would this sender's writing anywhere do? One pass, and it separates a
        # sender that carries the fact from one that is simply loud.
        alt = (receiver_position + 1) % n_pos
        shifted = run_with(delta_of(row["layer"], row["head"], alt))
        worst = max(control)
        row.update(
            {
                "control_max": round(worst, 6),
                "control_min": round(min(control), 6),
                # EVERY draw, not just the two extremes. The spread is the
                # finding: at one site on the reference pair the eight ran from
                # -0.02 to 0.28, and a verdict quoted as "beat 0.28" hides that
                # seven of them were nowhere near it. Eight floats per
                # controlled sender, at most `max_controlled` per receiver.
                "controls": [round(c, 6) for c in control],
                "control_draws": draws,
                "shifted_position": round(shifted, 6),
                "clears_control": bool(row["recovery"] > worst),
                "clears_position": bool(row["recovery"] > shifted),
            }
        )
        controlled += 1

    return {
        "receiver": {"layer": receiver_layer, "position": receiver_position},
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
        # Differences below this are not a ranking. See `recovery_resolution`.
        # Unrounded, for the reason written at the other call site.
        "recovery_resolution": recovery_resolution(model, clean_logits, gap),
        "senders": scored,
        "n_senders": len(scored),
        "n_controlled": controlled,
        "passes": passes,
        "seconds": round(time.perf_counter() - t0, 2),
        # THE SEEDING RULE, STATED. Edge count is quadratic in the general
        # case; this is linear only because the receiver is fixed to the one
        # site you asked about. Saying which edges were even considered is the
        # difference between "head 9.6 is the strongest sender" and "head 9.6
        # is the strongest sender we looked at".
        "seeding": (
            f"every attention head and MLP in layers 0-{receiver_layer - 1} "
            f"was scored as a sender into layer {receiver_layer} at position "
            f"{receiver_position} — {len(scored)} edges, all of them. Controls "
            f"ran on the top {controlled} by recovery; the rest carry a score "
            f"and no verdict."
        ),
        # THE SCOPE, NAMED. See the docstring: q/k/v is not split in v1.
        "scope": (
            "residual receivers only. A sender is patched into the receiver's "
            "residual input as a whole, so this says which component WROTE "
            "what the receiver reads — not which of its query, key or value "
            "paths carried it. Splitting those across GQA, fused QKV and "
            "rotary embeddings would produce confident and subtly wrong "
            "numbers, so it is not attempted here."
        ),
        "means": (
            f"Share of the clean-to-corrupt gap ({gap:.4f} logits) recovered by "
            f"restoring ONE component's contribution into layer "
            f"{receiver_layer} at position {receiver_position}, with everything "
            f"else still corrupt. Same fraction the node grid reports, so the "
            f"two are comparable. A sender clears its controls when it beats "
            f"all {draws} same-norm random draws AND the same edit taken from a "
            f"neighbouring position — the first says the number is not the size "
            f"of the edit, the second that it is not just this layer being "
            f"loud. RESOLUTION "
            f"{fmt.measured(recovery_resolution(model, clean_logits, gap), 3)}: two senders "
            f"closer than that are tied, not ranked — a recovery is a "
            f"difference of logits divided by the gap, and "
            f"{str(next(model.parameters()).dtype).removeprefix('torch.')} "
            f"cannot express a finer step at this logit magnitude."
        ),
    }


# ----------------------------------------------------------- patchscopes
#
# Every other reading in this file asks the model a question in numbers. This
# one asks it in words: take a hidden state from somewhere in one run, splice
# it into a second prompt built to make the model describe whatever it is
# holding, and read what comes out.
#
# The method's known failure is that a good target prompt describes ANYTHING
# fluently. Hand it a random vector and it will still produce a confident
# sentence. So a decode on its own is not evidence, and this never returns one
# alone.


# The identity target from Ghandeharioun et al. -- a few-shot pattern with
# nothing but "x -> x", so the model's only job at the final position is to
# say what is in front of it. VISIBLE AND EDITABLE, never a hidden constant:
# the target prompt is part of the result, and two decodes taken under
# different targets are not comparable. It is returned with every response for
# that reason.
DEFAULT_TARGET = "cat -> cat\n1135 -> 1135\nhello -> hello\n?"


def _overlap(a: str, b: str) -> float:
    """How much two decodes share, as a fraction of the smaller vocabulary.

        An EXACT-MATCH check is not enough on its own and measuring it showed why.
        Splicing a layer-8 state into the identity target gave ", hello, hello,
        hello" while the UNTOUCHED target gave " -> hello
    ? -> hello": different
        strings, so an equality test called the decode informative -- and both are
        plainly the few-shot pattern talking rather than anything about the state.
        A reader looking at the two would see it instantly; a boolean would not.

        Reported rather than thresholded. There is no principled cut-off for "the
        same", so the number goes on screen beside both decodes and the reader
        judges. Word-level and case-folded, because the failure being caught is
        the target prompt's vocabulary reappearing, not its punctuation.
    """

    # STRIPPED, not merely filtered. The first version used `.strip()` as the
    # test and kept the original word, so "hello," never matched "hello" --
    # which defeats the whole check, since the failure it exists to catch is
    # the target prompt's own words coming back in slightly different
    # punctuation. Caught by a test, not by reading it.
    def words(text: str) -> set[str]:
        stripped = (w.strip(".,:;!?\"'()->") for w in text.lower().split())
        return {w for w in stripped if w}

    left, right = words(a), words(b)
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def _splice_prefill(block: torch.nn.Module, pos: int, vec: torch.Tensor):
    """Replace one position of a block's residual input, ON THE PREFILL ONLY.

    A separate helper from `_splice`, and the difference is generation.
    `trace` runs one forward pass per patch, so the sequence dimension is
    always the whole prompt and a plain splice is right. A patchscope
    GENERATES: after the prefill the model runs with a KV cache and each step
    passes a single new token, so the stream arrives with sequence length 1
    and writing at position 14 raises IndexError -- inside a generation
    worker thread, where it surfaces as a streamer timeout rather than as the
    bug it is. Measured exactly that way.

    Skipping the decode steps is also the correct semantics: the patch goes
    into the PROMPT's residual stream, and the continuation should flow from
    that rather than have the same vector stamped over every new token.

    `_splice` is left alone rather than given this guard, because a silent
    no-op on an out-of-range position would hide a real mistake for its
    single-pass callers.
    """

    def pre(module, args):
        x = args[0]
        if pos >= x.shape[1]:
            return None
        edited = x.clone()
        edited[:, pos, :] = vec.to(edited.dtype).to(edited.device)
        return (edited,) + args[1:]

    return block.register_forward_pre_hook(pre)


def patchscope(
    model: Any,
    tokenizer: Any,
    blocks: list,
    decode: Any,
    source_prompt: str,
    *,
    device: Any,
    source_layer: int,
    source_position: int,
    target_prompt: str = DEFAULT_TARGET,
    target_layer: int | None = None,
    target_position: int = -1,
    draws: int = 1,
) -> dict:
    """Ask the model to describe a hidden state, with two controls beside it.

    `decode` is a callable taking a prompt and returning generated text --
    supplied by the caller so this module keeps no generation logic of its own.
    It must be GREEDY: three decodes are being compared, and sampling would put
    a second source of difference between them.

    Returns the patched decode and both controls:

      identity   the target prompt with its OWN activation, untouched
      random     the target prompt with a same-norm random vector

    A decode that reads the same across all three is the TARGET PROMPT TALKING,
    and the response says so rather than leaving the reader to notice.
    """
    t0 = time.perf_counter()
    n_layers = len(blocks)
    if not 0 <= source_layer < n_layers:
        raise PatchError(
            f"source layer {source_layer} is outside this model's {n_layers}."
        )
    if target_layer is None:
        target_layer = source_layer
    if not 0 <= target_layer < n_layers:
        raise PatchError(
            f"target layer {target_layer} is outside this model's {n_layers}."
        )

    source_ids = tokenizer(source_prompt, return_tensors="pt")["input_ids"].to(device)
    n_source = int(source_ids.shape[-1])
    if not -n_source <= source_position < n_source:
        raise PatchError(
            f"source position {source_position} is outside the "
            f"{n_source} tokens of that prompt."
        )
    at = source_position if source_position >= 0 else n_source + source_position

    target_ids = tokenizer(target_prompt, return_tensors="pt")["input_ids"].to(device)
    n_target = int(target_ids.shape[-1])
    if not -n_target <= target_position < n_target:
        raise PatchError(
            f"target position {target_position} is outside the "
            f"{n_target} tokens of the target prompt."
        )
    into = target_position if target_position >= 0 else n_target + target_position

    # The source state, read at the pre-hook point every other measurement in
    # this file uses, so a patchscope reads the same stream the grid patches.
    sink: dict = {}
    handle = _capture(blocks[source_layer], source_layer, sink)
    try:
        with torch.no_grad():
            model(source_ids)
    finally:
        handle.remove()
    if source_layer not in sink:
        raise PatchError(
            f"layer {source_layer} produced no residual stream on this model."
        )
    state = sink[source_layer][0, at, :].clone()
    norm = float(state.norm())

    def decode_with(vec: torch.Tensor | None) -> str:
        if vec is None:
            return decode(target_prompt)
        handle = _splice_prefill(blocks[target_layer], into, vec)
        try:
            return decode(target_prompt)
        finally:
            handle.remove()

    patched = decode_with(state)
    identity = decode_with(None)

    gen = torch.Generator().manual_seed(CONTROL_SEED)
    randoms = []
    for _ in range(max(1, draws)):
        r = torch.randn(state.shape, generator=gen).to(state.device, state.dtype)
        randoms.append(decode_with(r / r.norm() * norm))

    # The verdict the reader would otherwise have to reach by eye, and often
    # would not: a decode identical to the untouched target prompt means the
    # patch changed nothing, and one identical to the random control means the
    # target prompt says this whatever it is handed.
    same_as_identity = patched.strip() == identity.strip()
    same_as_random = any(patched.strip() == r.strip() for r in randoms)
    overlap_identity = _overlap(patched, identity)
    overlap_random = max(_overlap(patched, r) for r in randoms)

    return {
        "source": {
            "prompt": source_prompt,
            "layer": source_layer,
            "position": at,
            "tokens": _tokens(tokenizer, source_ids),
            "norm": round(norm, 4),
        },
        "target": {
            # RETURNED, ALWAYS. The target prompt is part of the result: two
            # decodes taken under different targets are not comparable, and a
            # hidden default would make that invisible.
            "prompt": target_prompt,
            "layer": target_layer,
            "position": into,
            "tokens": _tokens(tokenizer, target_ids),
        },
        "decode": patched,
        "controls": {
            "identity": identity,
            "random": randoms,
            "draws": len(randoms),
        },
        "same_as_identity": same_as_identity,
        "same_as_random": same_as_random,
        # How much of the decode's vocabulary the controls already had. The
        # exact-match booleans above catch only the clearest case; these are
        # what a reader actually needs when the decode merely ECHOES the
        # target prompt in different words.
        "overlap_identity": round(overlap_identity, 3),
        "overlap_random": round(overlap_random, 3),
        # Differs from both controls as a string AND says at least one word
        # neither of them already said.
        #
        # The string test alone was not enough, and it was measured: a layer-8
        # state decoded as ", hello, hello, hello" against an untouched target
        # that was also nothing but "hello" repeated. Different strings, so it
        # was flagged informative -- with 100% of its vocabulary already in
        # the control.
        # Complete containment is a test, not a tuned threshold: the decode
        # used no word the target prompt was not already using.
        "informative": bool(
            not same_as_identity
            and not same_as_random
            and overlap_identity < 1.0
            and overlap_random < 1.0
        ),
        "cross_layer": source_layer != target_layer,
        "seconds": round(time.perf_counter() - t0, 2),
        "means": (
            f"The model was shown a prompt built to make it describe whatever "
            f"is in front of it, with the layer-{source_layer} state from "
            f"position {at} of your prompt spliced in at layer {target_layer}. "
            + (
                f"SOURCE LAYER {source_layer} INTO TARGET LAYER {target_layer}: "
                f"the two streams are only comparable where the model treats "
                f"them alike, and nothing here checks that they do. "
                if source_layer != target_layer
                else ""
            )
            + (
                "THE DECODE MATCHES A CONTROL, so it is not about the state: "
                + (
                    "it is identical to the untouched target prompt, meaning "
                    "the patch changed nothing. "
                    if same_as_identity
                    else "it is identical to what a same-norm RANDOM vector "
                    "produced, meaning the target prompt says this whatever it "
                    "is handed. "
                )
                if (same_as_identity or same_as_random)
                else "It differs from both controls — the untouched target "
                "prompt and a same-norm random vector — so it is at least "
                "responding to what was patched. "
            )
            + (
                f"It shares {overlap_identity:.0%} of its words with the "
                f"untouched target and {overlap_random:.0%} with the random "
                f"control — read those beside the decodes themselves, which "
                f"are all three on screen for exactly this reason. "
            )
            + "A DECODE IS A GENERATION AND THEREFORE A SAMPLE. It is what the "
            "model said when handed this state through this target prompt, not "
            "what the state means."
        ),
    }
