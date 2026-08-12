"""Logit lens — what the model would have said if it had stopped at layer N.

Most models have no sparse autoencoder and never will, so the features panel
is empty for them through no fault of the model. This is the other thing you
can ask of a residual stream, and it needs nothing but the model itself:

    take the hidden state at layer N, apply the final norm, multiply by the
    unembedding, and read off the top tokens.

The answer is a genuine forward-pass quantity, not an approximation. What it
is *not* is features: it tells you which token the residual stream is pointing
at as it climbs, which is a different and much coarser question than "which
interpretable concepts are active". Watching the answer stabilise — noise for
the first third, then the right token arriving and holding — is the useful
part, and it works on every causal LM.

Caveat worth stating: models trained with a normalisation the unembedding
never saw applied mid-stack (and models that tie or scale embeddings oddly)
give a lens that is directionally right and numerically off. It is a probe,
not a measurement of what layer N "believes".

**Every row now carries its own error.** `kl_to_final` is the KL from the
model's real next-token distribution to that layer's lens distribution, so a
row is never a confident ranked list with nothing to say how much it can be
trusted — which is the plain lens's documented silent failure and the reason
tuned-lens exists. Measured on gpt2, bf16 on an RTX 4060, "The Eiffel Tower is
located in the city of": layer 0 is 21.58 nats away and reads ' destro', the
middle layers hold ' the' around 4-6, layer 9 turns to ' Rome', layer 10 to
' London', and layer 11 reaches ' Paris' at 0.96 — the closest any layer gets.
`reliability` reports that best figure and refuses to call the lens usable
past a stated threshold.

That number immediately earned itself. The last row IS the model, so its KL is
an arithmetic floor and must read ~0 — it read **2.12**, which is how the
double-norm check below was found to have been failing on every bfloat16 load
since it was written. See the comment at `last_is_normed`.
"""

from __future__ import annotations

from .errors import Refusal

# Below this many nats between the model's own distribution and the one you get
# by projecting the last hidden state straight through the unembedding, the
# last hidden state is already normalised. See the long comment in
# `logit_lens` — the previous logit-space version of this check silently failed
# on every bfloat16 model.
NORMED_KL_TOLERANCE = 0.01


def _final_norm(model):
    """The norm applied before the unembedding, whatever this family calls it.

    Refuses rather than guessing: applying the wrong norm — or none — produces
    a plausible ranked list that describes nothing.
    """
    base = getattr(model, "model", None) or getattr(model, "transformer", None)
    for holder in (base, model):
        if holder is None:
            continue
        for name in ("norm", "ln_f", "final_layer_norm", "final_layernorm"):
            found = getattr(holder, name, None)
            if found is not None and callable(found):
                return found
    # A Refusal, not a crash: this architecture is one we cannot read, and the
    # message says which layouts we can. Nothing is broken here — the lens is
    # simply not a measurement this model supports.
    raise Refusal(
        "could not find this model's final norm, so a logit lens would be "
        "reading the residual stream through the wrong transform. Supported "
        "layouts: model.norm, transformer.ln_f, final_layer_norm."
    )


def logit_lens(model, tokenizer, ids, top_k: int = 5) -> dict:
    """Top predictions at every layer for the final position."""
    import torch

    head = model.get_output_embeddings()
    if head is None:
        # Same family as _final_norm's refusal: a limitation of this
        # architecture that ModelMRI knows about, not something breaking.
        raise Refusal("this model has no output embedding to project through")
    norm = _final_norm(model)

    device = next(model.parameters()).device
    ids = ids.to(device)
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)

    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
        states = out.hidden_states  # (n_layers + 1) x [B, S, d]

        # Is the LAST hidden state already normalised? In HuggingFace decoders
        # it is — the forward pass applies the final norm and then records it,
        # so `lm_head(hidden_states[-1])` reproduces `logits` exactly. Applying
        # the norm again computes head(norm(norm(h))), and a norm with learned
        # gamma/beta is not idempotent: it removes the very per-dimension
        # scaling the unembedding was trained to read.
        #
        # That was not theoretical. On gpt2 completing "…located in the city
        # of", the double-normed top row read ' the' while the model actually
        # said ' Paris' — a confident, plausible, wrong answer on the one row a
        # reader can check. And `final` is taken from that row, which anchors
        # settled_at and the whole agreement column.
        #
        # Detected rather than assumed, so this holds across transformers
        # versions and model families instead of encoding today's internals.
        # Compared as DISTRIBUTIONS, not as logits. The logit version used
        # `allclose(..., atol=1e-3, rtol=1e-3)` and was wrong on every bf16
        # load, which is the default on most current GPUs.
        #
        # Measured on gpt2, cuda, "The Eiffel Tower is located in the city of":
        # in float32 the two logit vectors differ by at most 0.00007 and the
        # check passed; in bfloat16 they differ by 0.5 and it failed. 0.5 is
        # not disagreement — the logits are ~128 and bf16's precision there is
        # 128 * 2^-8 = 0.5, so that IS bit-identical to the last representable
        # digit. The check was measuring the dtype, not the model.
        #
        # The cost of getting it wrong is the exact failure this block exists
        # to prevent: the final row double-normed, reading ' the' where the
        # model says ' Paris', and `final`/`settled_at` derived from that row.
        # So every bf16 session shipped a wrong top row and a wrong settled
        # layer, silently, on the one row a reader can check by eye.
        #
        # Softmax is scale-free, so a KL between the two distributions does not
        # care about dtype precision: bf16 rounding lands around 1e-4 nats,
        # while a genuine double-norm measured 2.12 nats on the prompt above.
        # 0.01 sits between them by two orders of magnitude either way.
        truth = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
        unnormed = torch.softmax(head(states[-1][:, -1, :]).float()[0], dim=-1)
        last_is_normed = (
            float(
                (
                    truth
                    * (truth.clamp_min(1e-12).log() - unnormed.clamp_min(1e-12).log())
                ).sum()
            )
            < NORMED_KL_TOLERANCE
        )

        rows = []
        for layer, hidden in enumerate(states):
            x = hidden[:, -1, :]
            final_row = layer == len(states) - 1
            logits = head(x if (final_row and last_is_normed) else norm(x)).float()[0]
            probs = torch.softmax(logits, dim=-1)
            top = torch.topk(probs, k=min(top_k, probs.numel()))
            rows.append(
                {
                    "layer": layer,
                    "tokens": [
                        tokenizer.decode([int(i)]) for i in top.indices.tolist()
                    ],
                    "probs": [round(float(p), 5) for p in top.values.tolist()],
                    # How concentrated the distribution is. A high-entropy row
                    # early and a low-entropy row late is the shape you expect.
                    "entropy": round(
                        float(-(probs * torch.log(probs.clamp_min(1e-12))).sum()), 3
                    ),
                    # The lens's own error at this layer: how far this row's
                    # distribution is from what the model actually predicts.
                    # Without it a lens row is a confident ranked list with no
                    # indication of whether it describes anything — which is
                    # the documented silent failure of the plain lens and the
                    # reason tuned-lens exists.
                    "kl_to_final": None,  # filled below, once `truth` is known
                    "_probs": probs,
                }
            )

        # `truth` came from the same pass, above. Not a second forward — the
        # two must be one run or the KL measures sampling noise as lens error.
        for row in rows:
            p = row.pop("_probs")
            # KL(truth || lens): how much information is lost by reading this
            # layer instead of the model. Same direction and same floor as
            # `ablate.kl_nats`, so a lens error and a head score on one screen
            # are the same quantity.
            row["kl_to_final"] = round(
                float(
                    (truth * (truth.clamp_min(1e-12).log() - p.clamp_min(1e-12).log())).sum()
                ),
                5,
            )

    final = rows[-1]["tokens"][0] if rows else ""
    # The layer from which the eventual answer stays on top — where the model
    # "decides", in the only sense this probe can support.
    settled = None
    for row in rows:
        if row["tokens"] and row["tokens"][0] == final:
            settled = settled if settled is not None else row["layer"]
        else:
            settled = None
    # The last row IS the model, so its KL is the arithmetic floor rather than
    # a reading about the lens — quoting it as lens error would advertise an
    # accuracy the lens does not have at any other layer.
    floor = rows[-1]["kl_to_final"] if rows else 0.0
    middle = [r["kl_to_final"] for r in rows[:-1]] if len(rows) > 1 else []
    return {
        "layers": rows,
        "n_layers": len(rows) - 1,
        "final": final,
        "settled_at": settled,
        "reliability": _reliability(middle, floor),
    }


# Above this, the plain lens's distribution has essentially nothing to do with
# the model's. Stated rather than tuned: 2.3 nats is roughly a factor of ten in
# the probability assigned to the true next token, which is the point past
# which a ranked list of tokens is not describing the model's answer.
LENS_UNUSABLE_NATS = 2.3


def _reliability(middle: list[float], floor: float) -> dict:
    """How far the lens's own rows are from the model, and whether to trust them.

    The plain logit lens fails silently on some model families — it returns a
    ranked list of plausible tokens computed through a transform the
    unembedding never saw, and nothing on screen says so. This is the number
    that says so. It is measured on the prompt in front of you, so it is a
    reading about THIS text, not a property of the model.
    """
    if not middle:
        return {
            "measured": False,
            "why": "a single-layer model has no intermediate rows to check",
        }
    best = min(middle)
    return {
        "measured": True,
        "floor_kl": floor,
        "best_kl": best,
        "median_kl": round(sorted(middle)[len(middle) // 2], 5),
        "usable": best < LENS_UNUSABLE_NATS,
        "threshold": LENS_UNUSABLE_NATS,
        "means": (
            "KL from the model's real next-token distribution to each layer's "
            "lens distribution, in nats, on this prompt. The last row is the "
            "model itself and is the arithmetic floor."
            if best < LENS_UNUSABLE_NATS
            else (
                f"The closest any layer's lens gets to the model is {best} "
                f"nats, past the {LENS_UNUSABLE_NATS} this package treats as "
                "unusable. The rows below are a confident ranked list that "
                "does not describe what this model predicts — the plain lens "
                "does not transfer to this family."
            )
        ),
    }
