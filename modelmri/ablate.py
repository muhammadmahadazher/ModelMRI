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

**head_dim is not `hidden_size // n_heads`.** Measured: correct for gpt2
(768/12 = 64) and Qwen2.5-0.5B (896/14 = 64), wrong by 2x on Qwen3-0.6B
(quotient 64, real 128) and wrong on gemma-3-270m-it (quotient 160, real
256). Using the quotient there ablates half of one head plus half of the
next and produces a ranking that is confidently about nothing. It is read
off the projection's input width instead, and asserted.

Every number below was measured on gpt2 with the prompt "The capital of
France is", attributing at the last prompt token. The prompt is named
because a KL without one is not reproducible — which is how this docstring
previously came to carry four figures that were wrong.

**KL, not a logit difference.** Softmax is invariant to a constant shift, and
ablation shifts whole logit vectors. Zeroing L0H0 moves the top token's logit
by -0.258, but the mean move across the whole vocabulary is -0.145 — so the
honest residual is -0.113, and a raw logit difference would call that head
2.3x more important than it is.

**The baseline is part of the answer.** Zeroing a head is one choice, and on
gpt2 layer 0 it is most of the result. Zero-ablation ranks heads 7 (0.784),
10 (0.543) and 9 (0.415) far above the rest. Replacing each head with its own
mean over positions ranks 3 (0.0105), 1 (0.0080) and 10 (0.0062) — head 7
falls to sixth, head 9 to eighth, and head 0 goes from fifth to last. Same
model, same prompt, different question. So the baseline is named in the
response and on screen, and both are offered — a single unlabelled number
here would be the lie.

One thing this does NOT measure: a head's share of the prediction. Per-head
scores are not additive and are not close to it. On gpt2 layer 0 the twelve
per-head KLs sum to 1.995 while zeroing the whole layer gives 0.208 — eight
times too much. On gemma-3-270m-it layer 0 it goes the other way and much
harder: four per-head KLs sum to 0.0007 against 6.57 for the whole layer, so
every head looks irrelevant alone while the layer is load-bearing. It answers
"removing this one head alone moves the answer most", which is a different
and smaller claim.
"""

from __future__ import annotations

import time
from typing import Any

import torch

# Baselines we know how to justify. "zero" removes the head's contribution;
# "mean" replaces it with its own average over positions, which asks the
# softer question "does this head's *variation* matter".
BASELINES = ("zero", "mean")


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


def _cut(head: int, head_dim: int, baseline: str):
    """A pre-hook that removes one head's contribution from the projection."""

    def hook(module, args):  # noqa: ANN001 - torch's signature
        x = args[0].clone()
        lo, hi = head * head_dim, (head + 1) * head_dim
        if baseline == "mean":
            # The head's own average over positions: its constant part stays,
            # its variation goes. A different question from zeroing, and on
            # some layers a different answer — which is the point of offering
            # both rather than picking one and calling it "importance".
            x[..., lo:hi] = x[..., lo:hi].mean(dim=-2, keepdim=True)
        else:
            x[..., lo:hi] = 0
        return (x,) + args[1:]

    return hook


def _distribution(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits.float(), dim=-1)


def _kl(p: torch.Tensor, q: torch.Tensor) -> float:
    """KL(p || q) in nats, with q floored so a zeroed tail cannot give inf."""
    q = q.clamp_min(1e-12)
    return float((p * (p.clamp_min(1e-12).log() - q.log())).sum())


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
) -> dict:
    """Rank (layer, head) by how far removing that head moves the answer.

    `blocks` is a callable layer -> module. `position` is the index whose
    next-token distribution we attribute; `decode` turns a token id into a
    string for the readout.
    """
    if baseline not in BASELINES:
        raise AblationError(
            f"unknown baseline {baseline!r} — use one of {', '.join(BASELINES)}"
        )

    started = time.perf_counter()
    with torch.no_grad():
        base_logits = model(ids).logits[0, position]
        base = _distribution(base_logits)

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
        floor = _kl(base, _distribution(model(ids).logits[0, position]))

        top_id = int(base.argmax())
        ranked: list[dict] = []
        for layer in layers:
            block = blocks(layer)
            head_dim = head_geometry(block, n_heads)
            proj = out_projection(block)
            for head in range(n_heads):
                handle = proj.register_forward_pre_hook(_cut(head, head_dim, baseline))
                try:
                    after = _distribution(model(ids).logits[0, position])
                finally:
                    handle.remove()
                ranked.append(
                    {
                        "layer": layer,
                        "head": head,
                        "kl": round(_kl(base, after), 5),
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
        "passes": len(ranked) + 2,
        "elapsed_s": round(time.perf_counter() - started, 2),
        "ranked": ranked,
        # Said here so it travels with the numbers, not only in the UI.
        "means": (
            "KL divergence of the next-token distribution when this head "
            "alone is removed. Larger = removing it alone moves the answer "
            "more. These are NOT each head's share of the prediction: they "
            "do not add up, and are not meant to."
        ),
    }
