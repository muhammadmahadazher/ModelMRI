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
"""

from __future__ import annotations


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
    raise RuntimeError(
        "could not find this model's final norm, so a logit lens would be "
        "reading the residual stream through the wrong transform. Supported "
        "layouts: model.norm, transformer.ln_f, final_layer_norm."
    )


def logit_lens(model, tokenizer, ids, top_k: int = 5) -> dict:
    """Top predictions at every layer for the final position."""
    import torch

    head = model.get_output_embeddings()
    if head is None:
        raise RuntimeError("this model has no output embedding to project through")
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
        last_is_normed = torch.allclose(
            head(states[-1][:, -1, :]).float(),
            out.logits[:, -1, :].float(),
            atol=1e-3,
            rtol=1e-3,
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
                }
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
    return {
        "layers": rows,
        "n_layers": len(rows) - 1,
        "final": final,
        "settled_at": settled,
    }
