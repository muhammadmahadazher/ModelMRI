"""What a head does to the vocabulary, read off its weights. No corpus, no sampling.

An attention head has two circuits and they answer different questions. The QK
circuit decides WHERE it looks; the OV circuit decides WHAT it writes when it
looks there. Both are products of two weight matrices and neither needs a single
token of text to read — which is the whole reason this module exists beside
`ablate.py` rather than inside it. `ablate.rank_heads` answers "does this head
matter for THIS prompt", and the answer moves with the prompt. This answers "what
is this head wired to do", and the answer is the same every time.

    W_OV = W_O[h] @ W_V[kv(h)]      residual -> residual, rank <= head_dim
    W_QK = W_Q[h].T @ W_K[kv(h)]    residual -> residual, the bilinear form
                                    whose (i,j) entry scores token i attending
                                    to token j

## Nothing V x V is ever built

The literature's OV circuit is `W_U @ W_OV @ W_E`: vocabulary in, vocabulary
out. On Qwen3-1.7B that is 151,936 x 151,936 float32 — **92 TB**. Every
readout here is factored so the big matrix is never formed:

  one source token   `W_O[h] @ (W_V[kv] @ e)` is two matrix-vector products
                     against a `[d_model]` embedding. Peak is one `[d_model]`
                     vector, not a matrix.
  the spectrum       `(U_s @ W_O[h]) @ (W_V[kv] @ E_s.T)` for a SAMPLE of N
                     tokens is `[N, head_dim] @ [head_dim, N]`. Peak is
                     `N x head_dim`, and `d_model x d_model` is never formed
                     either.

## Grouped-query attention, which is why this is not four lines

`o_proj` is `n_heads * head_dim` wide and `v_proj` is `n_kv_heads * head_dim`
wide, and on every recent model those two numbers differ: Qwen3-1.7B has 16
query heads over 8 key/value heads. Head `h` reads the value head
`h // (n_heads // n_kv_heads)`, so slicing `W_V` by `h` instead of by `kv(h)`
reads a neighbouring head's values and produces a confident wrong answer for
half the heads. The geometry is read off the projections' own widths — never
`hidden_size // n_heads`, for the reason `ablate.head_geometry` states — and
`n_kv_heads` is derived from `v_proj`'s width and the measured `head_dim`
rather than trusted from a config field that several architectures do not set.

MEASURED on the real checkpoint, through `/api/attention/ov` on a loaded
Qwen3-1.7B: geometry `{n_heads: 16, n_kv_heads: 8, head_dim: 128,
d_model: 2048, group_size: 2}`, read entirely off the projections, with no
config field consulted. Heads 0 and 1 both map to value head 0 and head 2 to
value head 1 — `h // 2`, as grouped-query attention requires — while heads 0
and 1 still promote different tokens, because they share `W_V` and not `W_O`.
That difference is the evidence the slice is right: a version that indexed
`W_V` by the query head would give head 1 head 1's values, which do not exist,
and a version that ignored grouping entirely would make the two identical.

The spectrum on that same model and head reports 243 of 512 sampled
eigenvalues positive (47.5%) with `imaginary_mass` 0.5627 — over half the
spectrum's mass off the real line. That is what an ordinary head looks like,
and it is why there is no label: 47.5% is indistinguishable from chance, and
a "copying score" attached to it would be a verdict about nothing.

## What this refuses to say

**No "copying score" verdict, and no induction label.** The spectrum readout
reports the fraction of eigenvalues with a positive real part over a NAMED
sample of the vocabulary, with the sample size beside it. That fraction is a
measurement; "this is a copying head" is a claim about every token including
the ones nobody sampled, and `head_types.py` already shows what it costs to
attach a label without a null to gate it on. The eigenvalues of a
non-symmetric real matrix are complex, so "positive" is stated as "positive
real part" rather than quietly taking `.real` and calling them eigenvalues.

**Relative, not absolute.** The final norm's scale depends on the stream the
direction would be added to, and there is no stream here. So the vocabulary
readout ranks tokens; it does not predict logit amounts. `feature_corpus.
logit_weights` carries the same caveat for the same reason, in the same words.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .errors import BadRequest, Refusal

#: How many vocabulary rows the spectrum readout samples. The full matrix is
#: V x V and cannot be formed for any real vocabulary (92 TB on Qwen3-1.7B), so
#: the eigenvalues are of a SAMPLE and the result says so. 512 is large enough
#: that the positive fraction is stable across seeds to about a percentage
#: point on the models this was tried against, and small enough that the
#: eigendecomposition is milliseconds.
SPECTRUM_SAMPLE = 512

#: Names an attention module might give its projections, in the order they are
#: tried. `q_proj`/`k_proj`/`v_proj` is the Llama/Qwen/Gemma spelling; GPT-2
#: fuses all three into `c_attn` and is handled separately below.
_Q_NAMES = ("q_proj", "query", "q_lin")
_K_NAMES = ("k_proj", "key", "k_lin")
_V_NAMES = ("v_proj", "value", "v_lin")


@dataclass
class HeadGeometry:
    """Where one head's columns are, in every projection it touches.

    Every field is READ from a projection's own width. Nothing here is derived
    from `hidden_size // n_heads`, which is wrong by 2x on Qwen3-0.6B and by
    a different factor on gemma-3-270m — `ablate.head_geometry` states the
    measurement that showed it.
    """

    #: Query heads.
    n_heads: int
    #: Key/value heads. Equal to `n_heads` for ordinary multi-head attention.
    n_kv_heads: int
    #: Width of one head, from the output projection's input width.
    head_dim: int
    #: Residual stream width.
    d_model: int
    #: How many query heads share one key/value head.
    group_size: int

    def kv_head(self, head: int) -> int:
        """Which key/value head query head `head` reads.

        The one line grouped-query attention exists for, and the one that
        silently produces a neighbouring head's answer if it is skipped.
        """
        return head // self.group_size

    def to_dict(self) -> dict:
        return asdict(self)


def _attn(block):
    """The attention submodule, whatever this architecture calls it."""
    found = getattr(block, "attn", None) or getattr(block, "self_attn", None)
    if found is None:
        raise Refusal(
            "this model's block has neither `attn` nor `self_attn`, so there "
            "are no query, key and value projections to read a circuit from. "
            "Open `/api/weights` to see what this block does hold."
        )
    return found


def _named(attn, names: tuple[str, ...]):
    """The first projection present under any of `names`, or a refusal."""
    for name in names:
        proj = getattr(attn, name, None)
        if proj is not None:
            return proj
    return None


def _weight_of(proj, *, d_model: int):
    """A projection's weight as `[out, in]`, whatever the module stores.

    `nn.Linear` keeps `[out, in]` and applies `x @ W.T`. GPT-2's `Conv1D`
    keeps `[in, out]` and applies `x @ W`, which is the transpose — and a
    module that reads one as the other gets a matrix of the right SHAPE on a
    square projection and the wrong numbers, which is the failure mode with no
    symptom. `d_model` decides the orientation because it is the one dimension
    both projections share and this module already knows it.
    """
    import torch

    weight = proj.weight
    if not isinstance(weight, torch.Tensor):
        raise Refusal(
            f"{type(proj).__name__} has no weight tensor to read a circuit "
            f"from. This reads the projections directly, so a wrapped or "
            f"quantised layer that hides its weight cannot be read this way."
        )
    weight = weight.detach()
    if weight.ndim != 2:
        raise Refusal(
            f"this projection's weight is {weight.ndim}-dimensional, and a "
            f"circuit is a product of two matrices. A quantised or packed "
            f"layer needs to be loaded at full precision to be read."
        )
    # `nn.Linear` -> [out, in]; `Conv1D` -> [in, out]. Both have exactly one
    # side equal to d_model on the projections this reads, so the ambiguity is
    # only real for a square projection — where the two agree in shape and the
    # transpose is invisible. `in_features` settles it when the module says.
    n_in = getattr(proj, "in_features", None)
    if n_in is not None:
        return weight if weight.shape[1] == n_in else weight.T
    return weight.T if weight.shape[0] == d_model else weight


def geometry(block, *, n_heads: int, d_model: int) -> HeadGeometry:
    """Head widths and the query-to-kv mapping, measured off the projections.

    `n_kv_heads` is DERIVED from `v_proj`'s output width divided by the
    measured `head_dim`, not read from `config.num_key_value_heads`. The config
    field is absent on several architectures and, on the ones that set it, this
    division has to agree with it anyway — so deriving it makes the
    disagreement impossible instead of merely unlikely.
    """
    from . import ablate

    if isinstance(n_heads, bool) or not isinstance(n_heads, int) or n_heads < 1:
        raise BadRequest(
            f"a model has at least one attention head, and this was told "
            f"{n_heads!r}. The count comes from the loaded model's config."
        )
    # `_attn` FIRST, so a block with no attention at all gets this module's
    # refusal rather than `ablate`'s. Both say the same thing, but `ablate`
    # raises `AblationError`, which is a RuntimeError — it would reach a route
    # as a 500 instead of the 409 a refusal earns. The arm below converts the
    # other `AblationError` this can hit for the same reason.
    attn = _attn(block)
    try:
        head_dim = ablate.head_geometry(block, n_heads)
    except ablate.AblationError as err:
        # The sentence is already authored and already right — only its type is
        # wrong for a route boundary. `AblationError` is this project's own,
        # raised nowhere with anything but a written sentence, which is why
        # the seven sites in `runtime.py` relay it exactly this way and carry
        # the same mark. `test_no_machine_leaks` caught this one unmarked,
        # which is the guard doing its job on the person who was enforcing it.
        raise Refusal(str(err)) from err  # leak-ok: authored, see test_no_machine_leaks

    v_proj = _named(attn, _V_NAMES)
    if v_proj is None:
        # GPT-2 and friends fuse q, k and v into one projection. Handled, but
        # named separately so the refusal below can tell "fused" from "absent".
        if getattr(attn, "c_attn", None) is not None:
            v_width = int(_weight_of(attn.c_attn, d_model=d_model).shape[0]) // 3
        else:
            raise Refusal(
                "this attention module publishes no value projection under "
                "`v_proj`, `value`, `v_lin` or a fused `c_attn`, so what a "
                "head writes cannot be read from its weights. The causal "
                "answer is still available: ablate the head and measure what "
                "changes."
            )
    else:
        v_width = int(_weight_of(v_proj, d_model=d_model).shape[0])

    if v_width % head_dim:
        raise Refusal(
            f"this model's value projection is {v_width} wide, which is not a "
            f"whole number of {head_dim}-wide heads. ModelMRI cannot tell "
            f"where one value head ends and the next begins here, and a guess "
            f"would read a neighbouring head's values for half the heads."
        )
    n_kv_heads = v_width // head_dim
    if n_kv_heads < 1 or n_heads % n_kv_heads:
        raise Refusal(
            f"this model has {n_heads} query heads over {n_kv_heads} "
            f"key/value heads, and grouped-query attention needs the first to "
            f"be a whole multiple of the second. Nothing here can say which "
            f"value head a query head reads."
        )
    return HeadGeometry(
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=int(head_dim),
        d_model=int(d_model),
        group_size=n_heads // n_kv_heads,
    )


def _fused_slice(attn, geo: HeadGeometry, which: str):
    """One of q/k/v out of a fused `c_attn`, as `[out, in]`.

    GPT-2 concatenates the three along the output dimension in q, k, v order.
    It has no grouped-query variant, so all three thirds are the same width —
    which is why this may divide by three rather than reading each part's own
    geometry.
    """
    weight = _weight_of(attn.c_attn, d_model=geo.d_model)
    third = weight.shape[0] // 3
    at = {"q": 0, "k": 1, "v": 2}[which]
    return weight[at * third : (at + 1) * third]


def _projection_weight(block, geo: HeadGeometry, which: str):
    """`[out, in]` for q, k, v or o on this block, fused or not."""
    attn = _attn(block)
    if which == "o":
        from . import ablate

        return _weight_of(ablate.out_projection(block), d_model=geo.d_model)
    names = {"q": _Q_NAMES, "k": _K_NAMES, "v": _V_NAMES}[which]
    proj = _named(attn, names)
    if proj is not None:
        return _weight_of(proj, d_model=geo.d_model)
    if getattr(attn, "c_attn", None) is not None:
        return _fused_slice(attn, geo, which)
    raise Refusal(
        f"this attention module publishes no {which} projection, so the "
        f"circuit that needs it cannot be read from weights here."
    )


def _check_head(head: int, geo: HeadGeometry) -> None:
    if isinstance(head, bool) or not isinstance(head, int):
        raise BadRequest(
            f"a head is an index, and this was given {head!r}. Heads on this "
            f"model are numbered 0 to {geo.n_heads - 1}."
        )
    if not 0 <= head < geo.n_heads:
        raise BadRequest(
            f"head {head} is outside this layer's {geo.n_heads} heads "
            f"(0 to {geo.n_heads - 1})."
        )


def ov_factors(block, head: int, *, n_heads: int, d_model: int):
    """`(W_O_head, W_V_head)` for one head — the OV circuit, unmultiplied.

    Returned FACTORED, `[d_model, head_dim]` and `[head_dim, d_model]`, and
    that is the point: their product is `d_model x d_model` and every readout
    below can be written so it is never formed. A caller that genuinely wants
    the dense matrix can multiply them and pay for it deliberately.
    """
    geo = geometry(block, n_heads=n_heads, d_model=d_model)
    _check_head(head, geo)
    kv = geo.kv_head(head)
    hd = geo.head_dim

    # `[d_model, n_heads*head_dim]` -> this head's columns.
    w_o = _projection_weight(block, geo, "o")[:, head * hd : (head + 1) * hd]
    # `[n_kv_heads*head_dim, d_model]` -> the VALUE head this query head reads.
    w_v = _projection_weight(block, geo, "v")[kv * hd : (kv + 1) * hd, :]
    return w_o.float(), w_v.float(), geo


def qk_factors(block, head: int, *, n_heads: int, d_model: int):
    """`(W_Q_head, W_K_head)` for one head — the QK circuit, unmultiplied.

    Same shape of promise as `ov_factors`: `W_Q.T @ W_K` is `d_model x
    d_model` and is never needed whole to answer a question about a pair of
    tokens.
    """
    geo = geometry(block, n_heads=n_heads, d_model=d_model)
    _check_head(head, geo)
    kv = geo.kv_head(head)
    hd = geo.head_dim

    w_q = _projection_weight(block, geo, "q")[head * hd : (head + 1) * hd, :]
    w_k = _projection_weight(block, geo, "k")[kv * hd : (kv + 1) * hd, :]
    return w_q.float(), w_k.float(), geo


def _embeddings(model):
    """The input embedding matrix, or a refusal naming what is missing."""
    table = model.get_input_embeddings()
    if table is None or getattr(table, "weight", None) is None:
        raise Refusal(
            "this model publishes no input embedding table, so what a head "
            "reads FROM a token cannot be answered from weights. The causal "
            "answer is still available: ablate the head and measure what "
            "changes."
        )
    return table.weight.detach()


def ov_vocabulary(
    model,
    tokenizer,
    block,
    head: int,
    *,
    n_heads: int,
    source_token_id: int,
    top_k: int = 10,
) -> dict:
    """What this head writes into the stream when it attends to one token.

    EXACT and needs no corpus. Take the source token's embedding, push it
    through this head's value projection and back out through its slice of the
    output projection, then read the resulting residual direction through the
    final norm and the unembedding. That is `W_U @ W_O[h] @ W_V[kv] @ e`, and
    it says which tokens this head promotes when it looks at `e`.

    NEVER FORMS `W_OV`. `W_V @ e` is a matrix-vector product to `[head_dim]`,
    and `W_O @ that` is another to `[d_model]`. Peak memory is one `[d_model]`
    vector — against 92 TB for the literature's `W_U W_OV W_E` on a
    151,936-token vocabulary.

    Relative, not absolute, for the reason the module docstring gives: there is
    no stream here for the final norm to scale against.
    """
    import torch

    from .lens import _final_norm

    d_model = int(model.config.hidden_size)
    w_o, w_v, geo = ov_factors(block, head, n_heads=n_heads, d_model=d_model)

    table = _embeddings(model)
    n_vocab = int(table.shape[0])
    if isinstance(source_token_id, bool) or not isinstance(source_token_id, int):
        raise BadRequest(
            f"a source token is an id, and this was given {source_token_id!r}."
        )
    if not 0 <= source_token_id < n_vocab:
        raise BadRequest(
            f"token id {source_token_id} is outside this model's "
            f"{n_vocab:,}-token vocabulary."
        )
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise BadRequest(
            f"`top_k` is how many tokens to list and must be at least 1; this "
            f"was given {top_k!r}."
        )

    out_head = model.get_output_embeddings()
    if out_head is None:
        raise Refusal(
            "this model has no output embedding, so what a head does to the "
            "vocabulary cannot be read."
        )
    norm = _final_norm(model)
    where = next(model.parameters()).device

    with torch.no_grad():
        e = table[source_token_id].float().to(where)
        # Two matvecs, in this order, so nothing d_model x d_model exists.
        written = w_o.to(where) @ (w_v.to(where) @ e)
        projected = norm(written.to(next(norm.parameters()).dtype))
        logits = out_head(projected).float()
        # Against the vocabulary mean, because softmax ignores a constant: a
        # direction that lifts every logit equally has changed nothing. The
        # same centring `feature_corpus.logit_weights` and `dla.py` apply.
        centred = logits - logits.mean()
        k = min(top_k, int(centred.shape[-1]))
        top = torch.topk(centred, k)
        bottom = torch.topk(-centred, k)

    return {
        "head": head,
        "kv_head": geo.kv_head(head),
        "source_token_id": source_token_id,
        "source_token": tokenizer.decode([source_token_id]),
        "geometry": geo.to_dict(),
        "promotes": [
            {"token": tokenizer.decode([int(i)]), "score": round(float(v), 5)}
            for i, v in zip(top.indices, top.values, strict=True)
        ],
        "suppresses": [
            {"token": tokenizer.decode([int(i)]), "score": round(float(-v), 5)}
            for i, v in zip(bottom.indices, bottom.values, strict=True)
        ],
        "exact": True,
        "means": (
            f"What head {head} writes into the stream when it attends to "
            f"{tokenizer.decode([source_token_id])!r}, read straight through "
            f"its value and output projections, the final norm and the "
            f"unembedding. NO CORPUS AND NO SAMPLING — this is weight "
            f"arithmetic and it is the same every time. Scores are relative to "
            f"the vocabulary mean and at unit scale, so they RANK tokens "
            f"rather than predict logit amounts: the norm's real scale depends "
            f"on the stream this direction would be added to, and there is no "
            f"stream here. It says what the head is wired to do with this "
            f"token, not that it ever did."
        ),
    }


def ov_spectrum(
    model,
    block,
    head: int,
    *,
    n_heads: int,
    n_samples: int = SPECTRUM_SAMPLE,
    seed: int = 0,
) -> dict:
    """The eigenvalue readout of this head's OV circuit, over a NAMED sample.

    The literature's copying score is the fraction of positive eigenvalues of
    `W_U @ W_OV @ W_E`, which is `V x V` — 92 TB on Qwen3-1.7B, and there is
    no version of this that forms it. So the matrix is built over a uniform
    sample of `n_samples` token ids and the result carries the sample size,
    the seed and the vocabulary it was drawn from. It is a MEASUREMENT OF A
    SAMPLE, said in those words, rather than a score presented as a property
    of the head.

    Factored twice over: `(U_s @ W_O) @ (W_V @ E_s.T)` is
    `[N, head_dim] @ [head_dim, N]`, so neither `V x V` nor `d_model x
    d_model` is ever formed. Peak is `N x head_dim`.

    NO LABEL. "Positive fraction 0.83" is a number; "this is a copying head"
    is a claim about the tokens nobody sampled. `head_types.py` gates every
    label it attaches on a measured null, and this has none — so it attaches
    none.
    """
    import torch

    d_model = int(model.config.hidden_size)
    w_o, w_v, geo = ov_factors(block, head, n_heads=n_heads, d_model=d_model)

    table = _embeddings(model)
    out_head = model.get_output_embeddings()
    if out_head is None or getattr(out_head, "weight", None) is None:
        raise Refusal(
            "this model has no output embedding, so the OV circuit has no "
            "vocabulary side to have a spectrum over."
        )
    unembed = out_head.weight.detach()
    n_vocab = int(table.shape[0])

    if isinstance(n_samples, bool) or not isinstance(n_samples, int):
        raise BadRequest(f"`n_samples` is a count and this was {n_samples!r}.")
    if n_samples < 2:
        raise BadRequest(
            f"a spectrum needs at least 2 sampled tokens to have more than one "
            f"eigenvalue; this was asked for {n_samples}."
        )
    # The cap is the vocabulary itself, and it is REPORTED rather than applied
    # quietly: asking for more tokens than exist is not an error, but a result
    # that said 100,000 when it measured 32,000 would be.
    used = min(n_samples, n_vocab)

    where = next(model.parameters()).device
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    ids = torch.randperm(n_vocab, generator=generator)[:used]

    with torch.no_grad():
        e_s = table[ids].float().to(where)  # [N, d_model]
        u_s = unembed[ids].float().to(where)  # [N, d_model]
        left = u_s @ w_o.to(where)  # [N, head_dim]
        right = w_v.to(where) @ e_s.T  # [head_dim, N]
        matrix = (left @ right).cpu()  # [N, N]
        # A real non-symmetric matrix has COMPLEX eigenvalues. Taking `.real`
        # and calling the result "the eigenvalues" would hide that; the
        # imaginary part is reported so a reader can see how far from
        # symmetric this circuit is.
        values = torch.linalg.eigvals(matrix)
        real = values.real
        positive = int((real > 0).sum())
        trace = float(matrix.diagonal().sum())
        imaginary_mass = float(
            values.imag.abs().sum() / values.abs().sum().clamp(min=1e-12)
        )

    return {
        "head": head,
        "kv_head": geo.kv_head(head),
        "geometry": geo.to_dict(),
        "n_sampled": used,
        "n_vocab": n_vocab,
        "seed": int(seed),
        # The cap, stated. `n_samples` above the vocabulary is not an error and
        # not silently honoured either.
        "sample_capped": used < n_samples,
        "positive_fraction": round(positive / used, 5),
        "positive": positive,
        "trace": round(trace, 5),
        # How much of the spectrum is off the real line. 0 means a symmetric
        # circuit; a large value means "positive fraction" describes rotation
        # as much as sign, and a reader deserves to know which they are looking
        # at rather than being handed one number.
        "imaginary_mass": round(imaginary_mass, 5),
        "means": (
            f"{positive} of {used} sampled eigenvalues of head {head}'s OV "
            f"circuit have a positive real part ({positive / used:.1%}). "
            f"MEASURED OVER A SAMPLE of {used:,} of this model's "
            f"{n_vocab:,} tokens at seed {seed}, because the full circuit is "
            f"{n_vocab:,} x {n_vocab:,} and cannot be formed. A high fraction "
            f"is what a head that copies its source token looks like — this "
            f"does not say it IS one, because that would be a claim about the "
            f"tokens nobody sampled and nothing here measured a null to gate "
            f"it on. {imaginary_mass:.1%} of the spectrum's mass is off the "
            f"real line."
        ),
    }
