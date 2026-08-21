"""Which heads actually changed the answer?

The attention panel offers 144 heat maps and no reason to look at any
particular one. You pick a layer and a head, you see arcs, and you have no
idea whether that head mattered. This ranks them: zero one head's
contribution, run the model again, and measure how far the next-token
distribution moved.

That turns browsing into asking. It is also a feature that could very easily
produce a confident, ordered, wrong list — so most of this file is the four
things that make the number mean what it says.

**Where the cut goes.** A head is only separable *before* the output
projection. `attn_output` arrives at `c_proj`/`o_proj` as
`[B, S, n_heads * head_dim]` with the heads contiguous, so head *h* owns
columns `[h*head_dim : (h+1)*head_dim]`. After the projection the heads are
summed and cannot be pulled apart at all.

**head_dim is not `hidden_size // n_heads`.** Measured: correct for
Qwen2.5-0.5B (896/14 = 64), wrong by 2x on Qwen3-0.6B
(quotient 64, real 128) and wrong on gemma-3-270m-it (quotient 160, real
256). Using the quotient there ablates half of one head plus half of the
next and produces a ranking that is confidently about nothing. It is read
off the projection's input width instead, and asserted.

Every number below names the model and the prompt it came from. A KL without
them is not reproducible — which is how this docstring previously came to
carry four figures that were wrong.

**KL, not a logit difference.** Softmax is invariant to a constant shift, and
ablation shifts whole logit vectors. Zeroing one head moves the top token's
logit, but most of that move is a shift of the whole vocabulary; the honest
residual is what is left after subtracting it, and a raw logit difference can
overstate a head by a factor of two or more.

**The baseline is part of the answer.** Zeroing a head is one choice, and it
can be most of the result. Zero-ablation and mean-ablation over the same
layer put different heads at the top, and a head near the top under one can
fall to the bottom under the other. Same
model, same prompt, different question. So the baseline is named in the
response and on screen, and both are offered — a single unlabelled number
here would be the lie.

**And a third baseline, because the first two are both off-distribution.** No
input makes a head output exactly zero, and none makes it output its own
average at every position, so a model fed either may be damaged by the
impossibility rather than by the missing information. Resampling replaces the
head with what it really does compute on a different sentence, eight times.

Ranked over one layer, on one prompt, against a corpus of 8 plain sentences
about weather, trains and coffee, the three do not agree: their top fives
differ by two or three heads in every pair, and the rank correlation between
any two of them across the whole layer is weak. Three baselines, one model,
one prompt, three different answers — and
before this the panel showed whichever one you happened to have selected, with
nothing on screen to say the others existed. `compare_baselines` is that
missing line.

**One donor is a coin flip, which is why there are eight.** A single head's
score across the draws of that same run spanned better than a tenfold range
around its own median. A single draw could have reported any number
in that range as the head's score. This is the same lesson the SAE controls
taught in 0.9 and the patching controls in 0.10, arrived at from a third
direction: a number measured once is a sample, not a property.

One thing this does NOT measure: a head's share of the prediction. Per-head
scores are not additive and are not close to it, and which way they miss is
not fixed — on some models the singles sum to several times the whole layer's
ablation. On gemma-3-270m-it layer 0 it goes the other way and much
harder: four per-head KLs sum to 0.0007 against 6.57 for the whole layer, so
every head looks irrelevant alone while the layer is load-bearing. It answers
"removing this one head alone moves the answer most", which is a different
and smaller claim.
"""

from __future__ import annotations

import math
import time
from typing import Any

import torch

from .errors import BadRequest

# Baselines we know how to justify. "zero" removes the head's contribution;
# "mean" replaces it with its own average over positions, which asks the
# softer question "does this head's *variation* matter"; "resample" replaces it
# with the same head's activation from a DIFFERENT real sequence, which is the
# only one of the three that keeps the model on its own distribution.
#
# Zero and mean are both off-distribution: no input ever makes a head output
# exactly zero, and no input makes it output its own average at every position.
# A model asked to continue from an activation it could never produce may be
# damaged by the impossibility rather than by the missing information, and the
# score cannot tell those apart. Resampling asks the narrower, answerable
# question — "does it matter that this head computed THIS rather than something
# else it really does compute?"
BASELINES = ("zero", "mean", "resample")

# Draws per head for the resample baseline. One donor is a coin flip: which
# sequence you happened to draw decides the score. Eight is the same number
# patch.py uses for its controls, for the same reason and with the same cost
# shape — see `patch.trace`.
RESAMPLE_DRAWS = 8


class AblationError(RuntimeError):
    """We cannot take this measurement, and we say why rather than guess."""


def out_projection(block: torch.nn.Module) -> torch.nn.Module:
    """The module that mixes the heads back together.

    GPT-2 calls it `c_proj` (a Conv1D), Llama/Qwen/Gemma call it `o_proj`
    (an nn.Linear). Both take `[B, S, n_heads * head_dim]`.
    """
    attn = getattr(block, "attn", None) or getattr(block, "self_attn", None)
    if attn is None:
        raise AblationError(
            "this model's block has neither `attn` nor `self_attn`, so there "
            "is no attention output projection to cut a head out of."
        )
    for name in ("o_proj", "c_proj", "out_proj", "dense"):
        proj = getattr(attn, name, None)
        if proj is not None:
            return proj
    raise AblationError(
        f"cannot find the attention output projection on {type(attn).__name__}. "
        "Head ablation needs the tensor before the heads are summed."
    )


def head_geometry(block: torch.nn.Module, n_heads: int) -> int:
    """head_dim, from the projection's own input width.

    NOT `hidden_size // n_heads`. On Qwen3-0.6B that quotient is 64 and the
    real head_dim is 128; on gemma-3-270m-it it is 160 against a real 256.
    The wrong value silently ablates half of one head and half of the next,
    and the ranking that comes out is about nothing. The assertion below is
    the difference between a wrong number and no number.
    """
    proj = out_projection(block)
    attn = getattr(block, "attn", None) or getattr(block, "self_attn", None)
    width = getattr(proj, "in_features", None)
    if width is None:  # Conv1D keeps weight as [in, out] and has no in_features
        width = int(proj.weight.shape[0])
    head_dim = getattr(attn, "head_dim", None) or (width // n_heads)

    if n_heads * head_dim != width:
        raise AblationError(
            f"this model's attention output is {width} wide but "
            f"{n_heads} heads x {head_dim} would be {n_heads * head_dim}. "
            "ModelMRI cannot tell where one head ends and the next begins "
            "here, and a guess would produce a ranking about nothing."
        )
    return int(head_dim)


def _cut(head: int, head_dim: int, baseline: str, donor: torch.Tensor | None = None):
    """A pre-hook that removes one head's contribution from the projection.

    `donor` is required for the resample baseline and ignored by the others:
    a `[S_donor, width]` capture of this same projection's input from another
    sequence, sliced to this head's columns and this sequence's length.
    """

    def hook(module, args):  # torch's signature
        x = args[0].clone()
        lo, hi = head * head_dim, (head + 1) * head_dim
        if baseline == "mean":
            # The head's own average over positions: its constant part stays,
            # its variation goes. A different question from zeroing, and on
            # some layers a different answer — which is the point of offering
            # both rather than picking one and calling it "importance".
            x[..., lo:hi] = x[..., lo:hi].mean(dim=-2, keepdim=True)
        elif baseline == "resample":
            # Position-matched, not broadcast: token i is replaced by the
            # donor's token i. Broadcasting one donor position across the
            # sequence would replace a varying signal with a constant, which
            # is what "mean" already does and would make the two baselines
            # quietly measure the same thing.
            x[..., lo:hi] = donor[: x.shape[-2], lo:hi].to(x.dtype)
        else:
            x[..., lo:hi] = 0
        return (x,) + args[1:]

    return hook


def capture_projection_inputs(model: Any, blocks, ids: torch.Tensor, layers: list[int]):
    """One pass, returning each layer's attention-projection input: [S, width].

    This is the tensor `_cut` slices, so a donor captured here is exactly the
    quantity being replaced — the same hook point, the same layout. Capturing
    somewhere else and reshaping would be a second implementation of
    `head_geometry`'s assumptions, free to drift from it.
    """
    sink: dict[int, torch.Tensor] = {}
    handles = []

    def make(layer: int):
        def hook(module, args):
            sink[layer] = args[0].detach()[0]

        return hook

    for layer in layers:
        handles.append(
            out_projection(blocks(layer)).register_forward_pre_hook(make(layer))
        )
    try:
        with torch.no_grad():
            model(ids)
    finally:
        for handle in handles:
            handle.remove()
    return sink


def _donors_for(
    donors: list[dict[int, torch.Tensor]] | None, layer: int, size: int, corpus: str
) -> list[torch.Tensor]:
    """The donor slices for one layer, or a refusal naming what is short.

    Every failure here is a refusal rather than a fallback, and that is the
    whole point of the function existing separately. A resample score computed
    from three donors when eight were promised, or from a donor padded out to
    length, is a different measurement wearing the name of the one that was
    asked for.
    """
    if not donors:
        raise BadRequest(
            "the resample baseline replaces a head with its own activation "
            "from another real sequence, so it needs a corpus to draw from. "
            "None was supplied."
        )

    out: list[torch.Tensor] = []
    for index, capture in enumerate(donors):
        tensor = capture.get(layer)
        if tensor is None:
            raise AblationError(
                f"donor {index} from {corpus or 'the corpus'} has no capture "
                f"for layer {layer}, so this layer cannot be resampled. "
                "Refusing rather than scoring it against a different baseline."
            )
        if int(tensor.shape[0]) < size:
            raise AblationError(
                f"donor {index} from {corpus or 'the corpus'} is "
                f"{int(tensor.shape[0])} tokens and this prompt is {size}, so "
                "there is no activation to put at the later positions. Use a "
                "corpus with longer sequences, or a shorter prompt — padding "
                "one out would score the padding."
            )
        out.append(tensor)
    return out


def _ranks(values: list[float]) -> list[float]:
    """Average ranks, so ties do not manufacture disagreement.

    A layer where six heads all score exactly 0.0 has no opinion about their
    order. Assigning them arbitrary distinct ranks would let two baselines that
    agree completely read as correlated by luck, in either direction.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def spearman(a: list[float], b: list[float]) -> float | None:
    """Rank correlation, or None when it is undefined.

    None rather than 0.0 or nan when either side is constant: "these two
    rankings are uncorrelated" and "one of them is not a ranking" are different
    statements, and a 0.0 printed for the second is a made-up number.
    """
    if len(a) != len(b) or len(a) < 2:
        return None
    ra, rb = _ranks(a), _ranks(b)
    n = len(ra)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb, strict=True))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    if not da or not db:
        return None
    return round(num / (da * db), 4)


def compare_baselines(rankings: dict[str, list[dict]], top: int = 10) -> dict:
    """How much do these baselines actually disagree, on this model and prompt?

    `rankings` maps baseline name -> the `ranked` list `rank_heads` returns.
    Reports, for each pair, the rank correlation over every head and how many
    of one's top-`top` are missing from the other's. The second number is the
    one a reader can act on: "zero and resample disagree on 6 of the top 10"
    says the choice of baseline is deciding the answer.
    """
    names = sorted(rankings)
    pairs = []
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            a = {(r["layer"], r["head"]): r["kl"] for r in rankings[left]}
            b = {(r["layer"], r["head"]): r["kl"] for r in rankings[right]}
            shared = sorted(set(a) & set(b))
            if not shared:
                continue
            top_a = [k for k in sorted(a, key=lambda k: -a[k])[:top]]
            top_b = [k for k in sorted(b, key=lambda k: -b[k])[:top]]
            overlap = len(set(top_a) & set(top_b))
            pairs.append(
                {
                    "baselines": [left, right],
                    "spearman": spearman(
                        [a[k] for k in shared], [b[k] for k in shared]
                    ),
                    "heads_compared": len(shared),
                    "top_k": min(top, len(top_a), len(top_b)),
                    "top_k_shared": overlap,
                    "top_k_disagree": min(top, len(top_a), len(top_b)) - overlap,
                }
            )
    return {
        "pairs": pairs,
        "means": (
            "Rank correlation over every head, and how many of the top heads "
            "one baseline names that the other does not. A low overlap means "
            "the baseline you picked, not the model, is deciding the ranking."
        ),
    }


def distribution(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits.float(), dim=-1)


def kl_nats(p: torch.Tensor, q: torch.Tensor) -> float:
    """KL(p || q) in nats, with q floored so a zeroed tail cannot give inf.

    Public, and imported by attribute.py rather than copied. Two KLs in one
    package would drift into meaning two different things — the same argument
    this file already makes for one `_cut` — and "how far the answer moved"
    has to be one quantity if a head score and a token score are ever going to
    be read on the same screen.

    **The floor is part of the definition, and it is not free.** Every number
    this package reports is this quantity, reproducibly — a score comes back
    the same in float32 and in float64 to the precision of the arithmetic
    itself, so it is not an accumulation artifact. But it is
    BELOW the unfloored KL whenever the intervention collapses q's tail past
    1e-12, and that is exactly the intervention with the largest score.
    Masking each index in turn, the gap is confined to the one index with the
    largest score: thousands of vocabulary entries fall under the floor there
    and the p-weighted cost of clamping them accounts for the whole difference
    against a `log_softmax`-based KL, while at the neighbouring indices the
    gap is 0.000000 to six places. So a shipped score is a close estimate of
    the unfloored KL and an exact statement of this
    package's own quantity, and the two claims are not the same claim.
    """
    q = q.clamp_min(1e-12)
    return float((p * (p.clamp_min(1e-12).log() - q.log())).sum())


def estimate_cost(
    model: Any,
    blocks,
    ids: torch.Tensor,
    *,
    position: int,
    layers: list[int],
    n_heads: int,
    baseline: str = "zero",
    device_kind: str = "cpu",
) -> dict:
    """What would `rank_heads` cost here? Measured, not quoted.

    The pass count is exact and portable — `len(layers) * n_heads + 2`. What a
    pass costs is neither, so this runs ONE real iteration on this machine and
    projects from it. The probe body is built here rather than by the caller
    because it has to match the loop below exactly: the hook, the clone it
    makes, and the fp32 softmax. `budget.probe_pass` records what happens when
    it does not.

    Retained bytes are the base distribution the loop holds for the whole
    sweep — one fp32 vector over the vocabulary — stated from the config rather
    than measured, because it is arithmetic and not an observation.
    """
    from . import budget

    if baseline not in BASELINES:
        raise BadRequest(
            f"unknown baseline {baseline!r} — use one of {', '.join(BASELINES)}"
        )

    block = blocks(layers[0])
    head_dim = head_geometry(block, n_heads)
    proj = out_projection(block)

    # The resample hook indexes its donor, so probing without one raises
    # TypeError inside a forward pass and the route answers 500 — which is
    # exactly what /api/attention/ablate/estimate?baseline=resample did until
    # a browser asked it. The sequence's own projection input is the right
    # shape and costs the same to splice, and this function measures COST, not
    # a score: nothing computed here is reported as a head's importance.
    probe_donor = None
    if baseline == "resample":
        probe_donor = capture_projection_inputs(model, blocks, ids, [layers[0]])[
            layers[0]
        ]

    def one_iteration() -> None:
        handle = proj.register_forward_pre_hook(
            _cut(0, head_dim, baseline, probe_donor)
        )
        try:
            with torch.no_grad():
                distribution(model(ids).logits[0, position])
        finally:
            handle.remove()

    # Warm the kernels first. The first pass after a load pays CUDA init and
    # measured 3-4x the steady rate on Qwen3-0.6B, so probing it would predict
    # a sweep several times slower than the one that actually runs.
    with torch.no_grad():
        model(ids)

    vocab = int(getattr(model.config, "vocab_size", 0) or 0)
    probe = budget.probe_pass(one_iteration, device_kind)
    estimate = budget.project(
        probe, len(layers) * n_heads + 2, retained_bytes=vocab * 4
    )
    return {
        "estimate": estimate.to_dict(),
        "probe": probe.to_dict(),
        "baseline": baseline,
        "layers": len(layers),
        "n_heads": n_heads,
    }


def rank_heads(
    model: Any,
    blocks,
    ids: torch.Tensor,
    *,
    position: int,
    layers: list[int],
    n_heads: int,
    baseline: str = "zero",
    decode=None,
    donors: list[dict[int, torch.Tensor]] | None = None,
    corpus: str = "",
) -> dict:
    """Rank (layer, head) by how far removing that head moves the answer.

    `blocks` is a callable layer -> module. `position` is the index whose
    next-token distribution we attribute; `decode` turns a token id into a
    string for the readout.

    `donors` is required when `baseline="resample"`: a list of per-layer
    captures from `capture_projection_inputs`, one per draw, each taken on a
    different real sequence. `corpus` names where they came from and travels
    with the numbers — a resample score without its corpus is unreproducible,
    because a different corpus is a different measurement.
    """
    # A BadRequest and not an AblationError, alone among the raises in this
    # file. The other three describe an architecture this code cannot read —
    # nothing is wrong with the call, and there is no parameter to change. This
    # one is `?baseline=banana` in a URL, which errors.py names as the type
    # example of a BadRequest ("a bad layer index, an unknown baseline name").
    #
    # It mattered because runtime.py converts AblationError to Refusal without
    # judging it: measured, `/api/attention/ablate?baseline=banana` answered
    # 409 while `?layer=99` three lines away in the same handler answered 422 —
    # two malformed query parameters on one endpoint, two different statuses.
    # BadRequest is a ValueError, so it passes the AblationError wrap untouched
    # and lands on the handler's 422.
    if baseline not in BASELINES:
        raise BadRequest(
            f"unknown baseline {baseline!r} — use one of {', '.join(BASELINES)}"
        )

    started = time.perf_counter()
    with torch.no_grad():
        base_logits = model(ids).logits[0, position]
        base = distribution(base_logits)

        # The noise floor, measured rather than assumed: the same forward
        # pass twice, with nothing ablated. Anything at or below this is the
        # arithmetic moving, not the model.
        #
        # On this code path it measures exactly 0.0 — checked on CPU and on
        # CUDA, in fp32, bf16 and fp16, because a single unbatched sequence
        # replayed through the same kernels is bit-identical. That is the
        # argument for keeping the pass, not for dropping it: the floor is
        # zero *here*, and one pass is what proves it rather than assumes it.
        # Batching, TF32, or a different accelerator can all lift it above
        # the smallest real signals, and this is the only thing that would
        # notice.
        floor = kl_nats(base, distribution(model(ids).logits[0, position]))

        top_id = int(base.argmax())
        ranked: list[dict] = []
        passes = 2
        size = int(ids.shape[-1])
        for layer in layers:
            block = blocks(layer)
            head_dim = head_geometry(block, n_heads)
            proj = out_projection(block)

            # Checked per layer, before any pass is spent, because a donor set
            # that cannot cover this layer cannot cover any head in it. Falling
            # back to mean here is precisely the `.get(name, 0.0)` shape that
            # made 206 robot episodes show one video: a different measurement,
            # returned under the name of the one that was asked for.
            layer_donors = (
                _donors_for(donors, layer, size, corpus)
                if baseline == "resample"
                else []
            )

            for head in range(n_heads):
                if baseline == "resample":
                    draws = []
                    for donor in layer_donors:
                        handle = proj.register_forward_pre_hook(
                            _cut(head, head_dim, baseline, donor)
                        )
                        try:
                            after = distribution(model(ids).logits[0, position])
                        finally:
                            handle.remove()
                        passes += 1
                        draws.append((kl_nats(base, after), int(after.argmax())))
                    scores = sorted(d[0] for d in draws)
                    middle = len(scores) // 2
                    median = (
                        scores[middle]
                        if len(scores) % 2
                        else (scores[middle - 1] + scores[middle]) / 2
                    )
                    ranked.append(
                        {
                            "layer": layer,
                            "head": head,
                            # The median across draws, not one draw's number.
                            "kl": round(median, 5),
                            "kl_min": round(scores[0], 5),
                            "kl_max": round(scores[-1], 5),
                            "draws": len(scores),
                            "p_top_before": round(float(base[top_id]), 5),
                            "p_top_after": None,
                            # Only if it flipped under EVERY draw. A flip that
                            # depends on which donor arrived is not a property
                            # of the head.
                            "flips_top": all(t != top_id for _, t in draws),
                        }
                    )
                    continue

                handle = proj.register_forward_pre_hook(_cut(head, head_dim, baseline))
                try:
                    after = distribution(model(ids).logits[0, position])
                finally:
                    handle.remove()
                passes += 1
                ranked.append(
                    {
                        "layer": layer,
                        "head": head,
                        "kl": round(kl_nats(base, after), 5),
                        "p_top_before": round(float(base[top_id]), 5),
                        "p_top_after": round(float(after[top_id]), 5),
                        # Did removing this head change the model's mind?
                        "flips_top": int(after.argmax()) != top_id,
                    }
                )

    ranked.sort(key=lambda r: -r["kl"])
    return {
        "baseline": baseline,
        "position": position,
        "target_token": decode(top_id) if decode else str(top_id),
        # Named so the caller can grey out what is indistinguishable from
        # arithmetic rather than presenting it as a weak result.
        "noise_floor_kl": round(floor, 6),
        "passes": passes,
        "elapsed_s": round(time.perf_counter() - started, 2),
        # Part of the measurement, not provenance trivia: the same head scores
        # differently against a different corpus, so a resample number quoted
        # without it cannot be checked by anyone.
        **(
            {"corpus": corpus, "draws": len(donors or [])}
            if baseline == "resample"
            else {}
        ),
        "ranked": ranked,
        # Said here so it travels with the numbers, not only in the UI.
        "means": (
            "KL divergence of the next-token distribution when this head "
            "alone is removed. Larger = removing it alone moves the answer "
            "more. These are NOT each head's share of the prediction: they "
            "do not add up, and are not meant to."
        ),
    }
