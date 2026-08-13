"""How many logits did each head put behind the token the model actually said?

The ablation ranking answers "what breaks if I remove this head". Direct logit
attribution answers a narrower and more literal question: of the logit the
model assigned to the token it predicted, how much came straight from each
component down the residual stream.

The two disagree, often. A head can contribute almost nothing directly and
still be load-bearing, because its output is what a later head reads. That is
not a flaw in either measurement, it is the difference between them, and this
module refuses to let a near-zero bar be read as "this head does not matter".

WHY THE MODEL IS NOT MODIFIED TO MAKE THIS EXACT

DLA is only linear if the final normalisation is. TransformerLens makes it
exact by folding LayerNorm into the weights, which changes the model you are
studying -- and once it is folded there is nothing in the output that says what
the folding cost.

Here the model stays exactly as loaded, the normalisation is frozen at the
scale a hook recorded from the real forward pass, and the cost of that
approximation is measured and printed: the RECONSTRUCTION RESIDUAL, the gap
between every component's contribution summed and the logit the model really
produced. On GPT-2 that gap is not zero. Showing it is mandatory -- without it
the chart is a fabricated 100%, which is the failure mode this whole feature
would otherwise have.

THE RESIDUAL IS ALSO THE FLOOR

A component whose contribution is smaller than the reconstruction error cannot
be distinguished from the reconstruction error. So the residual is not only
reported, it is the threshold below which a bar is labelled unreadable rather
than small -- a measured floor, in the same spirit as `ablate.rank_heads`
running the same forward twice to find out what zero looks like.

EVERY NUMBER IS SHIFT-CORRECTED

Softmax is invariant to a constant added to every logit, so a contribution that
raises the whole vocabulary equally has changed nothing. Each component's
contribution is reported relative to the mean it added across the vocabulary,
for the same reason `ablate.py` uses KL instead of a logit difference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .errors import Refusal

# How many of the model dtype's own representable steps the frozen-norm
# reconstruction may differ by before it is treated as a different function.
#
# RELATIVE TO THE DTYPE, not an absolute epsilon. The first version used
# 1e-3 absolute and refused gpt2 on a bf16 load: the reconstruction differed by
# 0.347 at a norm output of magnitude 199, which is 0.17% -- and bfloat16's
# precision at 199 is 199 * 2^-8 = 0.78, so the two agreed to BETTER than one
# representable step and the check was measuring the dtype rather than the
# model. `lens.py` records finding exactly this bug in its own agreement check
# ("the logits are ~128 and bf16's precision there is 0.5, so that IS
# bit-identical to the last representable digit"), and this is the same
# mistake in the same package.
#
# 4 steps rather than 1: the reconstruction sums d_model terms, and rounding
# accumulates across them.
NORM_AGREEMENT_STEPS = 4.0


@dataclass
class Contribution:
    """One component's direct push on the predicted token."""

    name: str
    kind: str  # "embed" | "head" | "mlp"
    layer: int | None
    head: int | None
    # Logits, already shift-corrected against this component's own vocabulary
    # mean. Signed: a component can and does push AGAINST the token the model
    # chose.
    logits: float
    # True when |logits| is under the reconstruction residual, i.e. this
    # component's direct effect is smaller than the error the approximation
    # already makes. NOT the same as "this head does not matter".
    unreadable: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Attribution:
    token: str
    token_id: int
    position: int
    real_logit: float
    # The constant part: the norm's own bias through the unembedding, plus the
    # unembedding's bias. It belongs to no component and is reported on its own
    # rather than spread across them.
    bias: float
    residual: float
    residual_share: float
    norm_kind: str
    components: list[Contribution] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            **{k: v for k, v in asdict(self).items() if k not in ("components",)},
            "components": [c.to_dict() for c in self.components],
            "n_unreadable": sum(1 for c in self.components if c.unreadable),
            "means": self.means(),
        }

    def means(self) -> str:
        unreadable = sum(1 for c in self.components if c.unreadable)
        return (
            f"Direct contribution to the logit for {self.token!r}, in logits, "
            f"each shift-corrected against its own vocabulary mean. They sum "
            f"to {self.real_logit - self.residual:.4f} against the model's "
            f"real {self.real_logit:.4f} — a reconstruction residual of "
            f"{self.residual:.4f} ({self.residual_share:.2%}), which is what "
            f"freezing the {self.norm_kind} scale cost on this run. "
            f"{unreadable} components fall below that residual: their direct "
            f"effect cannot be told from the approximation's own error, which "
            f"is not the same as their being unimportant. DIRECT-PATH ONLY — "
            f"a head that feeds a later head shows near zero here and can "
            f"still decide the answer."
        )


def _norm_parts(norm_module, stream):
    """(kind, gamma, beta, scale) for this model's final normalisation.

    The scale is FROZEN at what the real pass produced for this position, which
    is the whole approximation: with it fixed, the normalisation becomes affine
    and the unembedding distributes over a sum of components.
    """
    import torch

    weight = getattr(norm_module, "weight", None)
    bias = getattr(norm_module, "bias", None)
    eps = float(
        getattr(norm_module, "eps", None)
        or getattr(norm_module, "variance_epsilon", None)
        or 1e-5
    )
    if weight is None:
        raise Refusal(
            "this model's final normalisation has no weight to read, so the "
            "affine form direct attribution needs cannot be written for it."
        )

    gamma = weight.detach().float()
    beta = bias.detach().float() if bias is not None else torch.zeros_like(gamma)

    # LayerNorm centres; RMSNorm does not. Told apart by whether the module
    # has a bias AND by checking the reconstruction below -- a family that
    # centres without a bias would otherwise be attributed through the wrong
    # transform and produce a plausible, wrong chart.
    centred = bias is not None or "layernorm" in type(norm_module).__name__.lower()
    if centred:
        variance = stream.var(dim=-1, unbiased=False)
        scale = torch.rsqrt(variance + eps)
        kind = "LayerNorm"
    else:
        scale = torch.rsqrt(stream.pow(2).mean(dim=-1) + eps)
        kind = "RMSNorm"
    return kind, gamma, beta, scale, centred


def _frozen(stream, gamma, scale, centred):
    """The normalisation as a linear map, at the frozen scale.

    Deliberately without beta: the bias is constant across components and
    adding it to each would count it once per component. It is reported once,
    on its own row.
    """
    x = stream - stream.mean(dim=-1, keepdim=True) if centred else stream
    return x * scale * gamma


def attribute(model, tokenizer, ids, *, position: int = -1, top_k: int = 0):
    """Decompose the predicted token's logit across heads and MLPs.

    Blocking; call from a worker thread. `top_k` trims the returned component
    list to the strongest by magnitude — 0 returns all of them, and the count
    that was dropped is never implied by silence.
    """
    import torch

    from . import ablate
    from .lens import _final_norm
    from .patch import _capture_out, _sublayer

    head_module = model.get_output_embeddings()
    if head_module is None:
        raise Refusal(
            "this model has no output embedding, so there are no logits to attribute."
        )
    norm_module = _final_norm(model)

    base = getattr(model, "model", None) or getattr(model, "transformer", None)
    blocks = getattr(base, "layers", None) or getattr(base, "h", None)
    if blocks is None:
        raise Refusal(
            "could not find this model's decoder blocks, so its components "
            "cannot be separated. Supported layouts: model.layers, "
            "transformer.h."
        )
    n_layers = len(blocks)
    n_heads = int(model.config.num_attention_heads)

    device = next(model.parameters()).device
    ids = ids.to(device)
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)

    mlp_out: dict = {}
    attn_in: dict = {}
    pre_norm: dict = {}
    handles = []

    def catch_norm_input(module, args):
        """The residual stream BEFORE the final normalisation.

        `hidden_states[-1]` is not it. HuggingFace decoders apply the final
        norm and then record the hidden state -- `lens.py` documents this at
        length, because reading it the other way produced a confident wrong
        answer there too. Decomposing the POST-norm stream means every
        component is compared against a vector that has already been through
        the transform being frozen, and the reconstruction misses by 0.716 on
        gpt2. Measured: that is what this module's own norm check reported
        before this hook existed.
        """
        pre_norm["h"] = args[0].detach().clone()

    def catch_attn_input(layer: int):
        """The input to the out-projection, which is the heads side by side.

        A PRE-hook, because after the projection the heads are already mixed
        and there is no slice that belongs to any one of them.
        """

        def pre(module, args):
            attn_in[layer] = args[0].detach().clone()

        return ablate.out_projection(blocks[layer]).register_forward_pre_hook(pre)

    try:
        handles.append(norm_module.register_forward_pre_hook(catch_norm_input))
        for layer in range(n_layers):
            handles.append(catch_attn_input(layer))
            handles.append(
                _capture_out(_sublayer(blocks[layer], "mlp"), layer, mlp_out)
            )
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
    finally:
        for handle in handles:
            handle.remove()

    if "h" not in pre_norm:
        raise Refusal(
            "this model's final normalisation was never called during the "
            "forward pass, so the residual stream going into it could not be "
            "read. Direct attribution needs the stream before the norm, not "
            "the recorded hidden state, which is already normalised."
        )
    stream = pre_norm["h"][0, position].float()
    embed = out.hidden_states[0][0, position].float()
    logits = out.logits[0, position].float()

    kind, gamma, beta, scale, centred = _norm_parts(norm_module, stream)

    # THE APPROXIMATION IS CHECKED BEFORE IT IS USED. If the affine form
    # written above is not the function this model applies -- a gated norm, a
    # different centring, Gemma's (1 + weight) convention -- then every number
    # below would be an attribution through the wrong transform, which is
    # exactly the kind of confident wrongness this package refuses to ship.
    with torch.no_grad():
        theirs = norm_module(stream.to(next(model.parameters()).dtype)).float()
    mine = _frozen(stream, gamma, scale, centred) + beta
    gap = float((theirs - mine).abs().max())
    magnitude = max(float(theirs.abs().max()), 1e-9)
    # The size of one representable step at this magnitude, in the dtype the
    # model actually runs in. Derived, never chosen.
    step = magnitude * float(torch.finfo(next(model.parameters()).dtype).eps)
    allowed = step * NORM_AGREEMENT_STEPS
    if gap > allowed:
        raise Refusal(
            f"this model's final normalisation does not match the affine form "
            f"direct attribution needs — reconstructing it differs by {gap:.3g} "
            f"at the worst dimension, against {allowed:.3g} explainable by "
            f"{next(model.parameters()).dtype} rounding at this magnitude. "
            f"Attributing through the wrong transform would produce a "
            f"plausible chart about nothing, so it is refused. The ablation "
            f"ranking works on this model and answers a related question."
        )

    unembed = head_module.weight.detach().float()  # [vocab, d_model]
    unembed_bias = getattr(head_module, "bias", None)
    token_id = int(logits.argmax())
    row = unembed[token_id]

    def contribution(vector) -> float:
        """This component's push on the predicted token, shift-corrected.

        Softmax ignores a constant added to every logit, so what counts is how
        far this component moved the target ABOVE what it moved the whole
        vocabulary. Subtracting the component's own vocabulary mean is that
        correction, and it is why a component that lifts everything equally
        reports zero rather than a large number.
        """
        projected = _frozen(vector, gamma, scale, centred)
        target = float(row @ projected)
        mean = float((unembed @ projected).mean())
        return target - mean

    components: list[Contribution] = [
        Contribution(
            name="embed",
            kind="embed",
            layer=None,
            head=None,
            logits=round(contribution(embed), 5),
        )
    ]

    for layer in range(n_layers):
        block = blocks[layer]
        projection = ablate.out_projection(block)
        # head_geometry, not hidden_size // n_heads. On Qwen3-0.6B the
        # quotient is 64 against a real 128 and on gemma-3-270m-it 160 against
        # 256 -- a wrong head_dim silently attributes half of one head and
        # half of the next.
        head_dim = ablate.head_geometry(block, n_heads)
        packed = attn_in.get(layer)
        if packed is not None:
            heads = packed[0, position].float()
            weight = projection.weight.detach().float()
            # Conv1D (GPT-2) stores [in, out]; nn.Linear stores [out, in].
            # Getting this backwards transposes every head's contribution into
            # a different head's slot.
            if getattr(projection, "in_features", None) is None:
                weight = weight.T
            for head in range(n_heads):
                span = slice(head * head_dim, (head + 1) * head_dim)
                written = weight[:, span] @ heads[span]
                components.append(
                    Contribution(
                        name=f"L{layer}H{head}",
                        kind="head",
                        layer=layer,
                        head=head,
                        logits=round(contribution(written), 5),
                    )
                )

        mlp = mlp_out.get(layer)
        if mlp is not None:
            components.append(
                Contribution(
                    name=f"L{layer} MLP",
                    kind="mlp",
                    layer=layer,
                    head=None,
                    logits=round(contribution(mlp[0, position].float()), 5),
                )
            )

    # The constant term, once. Beta through the unembedding plus the
    # unembedding's own bias, both shift-corrected the same way.
    beta_target = float(row @ beta)
    beta_mean = float((unembed @ beta).mean())
    bias_total = beta_target - beta_mean
    if unembed_bias is not None:
        b = unembed_bias.detach().float()
        bias_total += float(b[token_id] - b.mean())

    real = float(logits[token_id] - logits.mean())
    explained = sum(c.logits for c in components) + bias_total
    residual = real - explained
    floor = abs(residual)

    for component in components:
        component.unreadable = abs(component.logits) < floor

    components.sort(key=lambda c: -abs(c.logits))
    if top_k > 0:
        components = components[:top_k]

    return Attribution(
        token=tokenizer.decode([token_id]),
        token_id=token_id,
        position=int(position if position >= 0 else ids.shape[-1] + position),
        real_logit=round(real, 5),
        bias=round(bias_total, 5),
        residual=round(residual, 5),
        residual_share=round(abs(residual) / abs(real), 5) if real else 0.0,
        norm_kind=kind,
        components=components,
    )
