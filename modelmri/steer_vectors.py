# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""Steer a model that has no sparse autoencoder — which is almost all of them.

The features panel can steer, and it needs an SAE. SAEs are published for a
handful of models and nobody is going to train one for the checkpoint you
finetuned last week, so for most models this tool could show you attention and
ablation and then had nothing to offer when you asked "can I push it".

A contrastive direction needs no SAE and no training run: take pairs of prompts
that differ in the property you care about, read the residual stream at the
last token of each, and the difference of the means is a direction. Add it back
during generation and the model moves.

That is also the problem. **Difference-of-means always returns a direction.**
Feed it two arbitrary sets of sentences and it produces a vector with a norm
and a layer and a confident-looking sweep, and adding any large vector to a
residual stream changes the output. The number that comes back looks identical
whether you found something or nothing.

So every direction here is scored against **its own shuffled null**. Refit the
same pairs with the positive/negative labels randomly reassigned, K times, and
measure the same effect. A real direction beats its shuffles; a direction that
does not is reported as *not measured* — not as a small discovery.

Measured with bf16 on an RTX 4060, twelve sentiment pairs ("I loved it" /
"I hated it" and similar), fitting on six and scoring on six: under both `caa`
and `repe`, all but one layer beat their own null, and the one that did not is
layer 0.

And then the control that matters — the same 24 sentences split at random
instead of by sentiment: **not one layer beat its null**, under the
identical estimator. Signal passes, noise does not, and layer 0 failing is
right: that is the embedding, before anything has been computed.

**RepE is not centred, and it is not randomised either.** The first
version of this file subtracted the mean of the paired differences before the
PCA, reasoning that otherwise PC1 is dominated by the centroid shift. That is
backwards for contrastive pairs — the centroid shift IS the signal — and on
two clouds separated by 6.0 along one axis it scored 1.04 against a null whose
worst shuffle reached 1.30. A real direction failing its own control, because
the estimator had thrown the direction away before scoring it.

**The gate is not free of false positives, and here is its rate.** With
`NULL_REFITS` draws the smallest attainable permutation p-value is
`1/(K+1)` — 0.111 at eight — so the gate cannot assert more than that however
clean the data is. Measured directly, 200 trials per method on structureless
Gaussian clouds with arbitrary labels and no real direction at all:

    caa    32/200 beat their own null   16.0%
    repe   26/200                       13.0%

against 50/50 detection of a real separation of 4.0 sigma, both methods.
(`repe` read 24/200 while it used `torch.pca_lowrank`; see `_fit` for why
that call is gone. `caa` never touched it and is unchanged, which is the
control on the re-measurement.) So it is a useful
screen and it is not a significance test: roughly one direction in six that
this reports as real, on data with nothing in it, is noise. `p_value` is
published alongside `beats_null` for that reason — a boolean hides whether the
call was 1/9 or 9/9. Raising `NULL_REFITS` is the lever if you need a tighter
gate, at linear cost in refits.

**Fit on half, score on the other half.** A direction scored on the pairs it
was fitted from will separate them, because it was built to. The split is the
difference between "this direction encodes the property" and "this direction
memorised these sentences".

**A layer with nothing in it is a row, not a failed sweep.** The residual
stream ENTERING block 0 is the last token's own embedding, so two sets of real
sentences — which all end with a full stop — are the same vector there, and
`_fit` is right to say there is no direction between them. `sweep` used to let
that one ordinary answer abort the other twenty-nine, which is how the panel's
own default contrast pairs came back as a 409 with nothing fitted anywhere. It
is now one row: `effect` 0, `beats_null` false, and NO `p_value`, `null_mean`
or `null_max` at all — a layer that was never scored has no permutation
quantile and no shuffles to take a worst of, and zero is the most confident
number in that range — plus a note naming the layer and the cause. The sweep
refuses only when EVERY requested layer is like that, because that is the case
where there is no table to read.

**Both estimators have to be able to say it.** `caa` finds nothing to fit
because the mean difference is the zero vector; `repe` had no way to say it at
all, because the SVD of all-zero differences returns zero singular values and
an arbitrary orthonormal basis, so PC1 came back a unit vector and the whole
row-not-a-refusal path was dead behind the method dropdown. Each branch of
`_fit` now tests the thing that is actually degenerate for it, and both raise
the same sentence.

**Coefficients are not portable.** A scale of 5 means nothing across models or
even across layers: residual norms differ by an order of magnitude between
early and late layers of the same network. Every applied strength here is
reported relative to the measured residual norm at that layer, so "0.5x the
stream's own norm" travels and "5.0" does not.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .errors import BadRequest, Refusal

# Below this, a difference of means is mostly the noise of whichever sentences
# happened to be written. Stated rather than tuned: with four pairs the fit
# half is two, and two sentences do not define a direction in 768 dimensions.
MIN_PAIRS = 8

# Shuffled refits per direction. Same count and same reasoning as
# `patch.CONTROL_DRAWS` and `ablate.RESAMPLE_DRAWS` — one shuffle is a coin
# flip, and this package has now been bitten by that three times.
NULL_REFITS = 8

# Reused rather than redefined so every control in the package draws from one
# documented stream. See patch.py.
from .patch import CONTROL_SEED  # noqa: E402

METHODS = ("caa", "repe")


@dataclass
class Direction:
    """One fitted direction, and everything needed to judge it."""

    layer: int
    method: str
    # Cosine separation on the HELD-OUT half, and the same statistic on
    # `NULL_REFITS` label-shuffled refits.
    effect: float
    null_mean: float | None
    null_max: float | None
    beats_null: bool
    # The standard permutation p-value, (1 + #{null >= |effect|}) / (1 + draws).
    # `beats_null` is the same gate expressed as a boolean, and a boolean hides
    # how close the call was — 1/9 and 9/9 both read as "no" today.
    #
    # None when the layer was never scored at all. A layer whose two sets have
    # identical mean activations has no direction to project onto and no null
    # to take a quantile of, and 0.0 there would be the most confident number
    # in the range — published, ranked and believed. `to_dict` drops the key
    # entirely rather than shipping a null, so a reader's `"p_value" in row`
    # is the honest question and `api.ts` types the field optional.
    #
    # `null_mean` and `null_max` above are None for the same layer and for the
    # same reason, which is a correction: they shipped as 0.0 beside a note
    # that said "no null was run at this layer", so the row contradicted
    # itself. Nothing was shuffled, nothing was refitted, and "the worst
    # label-shuffled refit reached 0.000" is a sentence about eight draws that
    # were never drawn. `vla_ood.OodReference` already spells the same concept
    # the same way — `null_max: float | None`, None when no null was run.
    p_value: float | None
    n_pairs: int
    n_fit: int
    n_score: int
    # The residual norm measured at this layer, so an applied coefficient can
    # be reported relative to something instead of as a bare number.
    residual_norm: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        row = asdict(self)
        # ABSENT, not null and not zero — see `p_value` above. These three are
        # everything the null produced, and a layer that never had a null
        # produced none of it; every other field on this row is a measurement
        # that was actually taken. `p_value` is the one a consumer keys off to
        # tell "no result" from "a result of zero", because it is the field
        # the chart and the store already read.
        for absent in ("p_value", "null_mean", "null_max"):
            if row[absent] is None:
                del row[absent]
        return row


def _last_token_states(model, blocks, ids_list, layers: list[int]):
    """Residual stream entering each layer, at the final token, per prompt.

    Uses the same pre-hook point as `patch._capture`, so a direction is fitted
    in the space the patching grid already measures rather than in a second,
    subtly different one.
    """
    import torch

    out: dict[int, list] = {layer: [] for layer in layers}
    for ids in ids_list:
        sink: dict[int, Any] = {}
        handles = [_capture_into(blocks(layer), layer, sink) for layer in layers]
        try:
            with torch.no_grad():
                model(ids)
        finally:
            for handle in handles:
                handle.remove()
        for layer in layers:
            if layer not in sink:
                raise Refusal(
                    f"layer {layer} produced no residual stream on this model, "
                    "so no direction can be fitted there."
                )
            out[layer].append(sink[layer][0, -1, :].float())
    return {layer: torch.stack(vecs) for layer, vecs in out.items()}


def _capture_into(block, layer: int, sink: dict):
    def pre(module, args):
        sink[layer] = args[0].detach()

    return block.register_forward_pre_hook(pre)


class NoDirection(Refusal):
    """The two sets have identical mean activations — there is nothing to fit.

    A SUBCLASS RATHER THAN A FLAG, because `sweep` has to tell this refusal
    apart from the others `fit_direction` raises. "not enough pairs" and "this
    model has no layer 40" are statements about the whole request and have to
    abort it. This one is a statement about ONE LAYER, and at layer 0 it is the
    ordinary answer for any two sets of real sentences: they end with a full
    stop, and entering block 0 the residual stream is that token's own
    embedding. A sweep that threw away twenty-nine measured layers because the
    thirtieth had nothing in it is the defect this class exists to make hard to
    write again.

    Still a `Refusal`, and still carrying the estimator's own sentence: called
    directly, `fit_direction` refuses exactly as it did before, and the
    `except Refusal` in its own null loop keeps catching this unchanged.
    """


# ONE SENTENCE, TWO ESTIMATORS. `caa` reaches it through a zero mean
# difference and `repe` through paired differences that are all zero, and
# everything downstream — the note on the degenerate row, `_nothing_to_fit`'s
# whole-request refusal, the panel's verdict column — tells one story about
# what happened. Two spellings of the same fact become two stories the first
# time one of them is edited. Both conditions imply the other's wording is
# true: differences that are all zero have a zero mean.
_NO_DIRECTION_SENTENCE = (
    "the positive and negative sets have identical mean activations at this "
    "layer, so there is no direction between them to fit."
)


def _fit(pos, neg, method: str):
    """A direction from paired activations. Unit norm, so scale is separate."""
    import torch

    if method == "caa":
        # Contrastive activation addition: the mean difference. Simple, and
        # the thing repeng and steering-vectors both ship.
        direction = pos.mean(0) - neg.mean(0)
    elif method == "repe":
        # First principal component of the paired differences.
        #
        # NOT centred. `pca_lowrank` centres by default, which is why this was
        # spelled out when that was the call. The first version subtracted the mean of the
        # differences first, reasoning that otherwise PC1 is dominated by the
        # centroid shift and the method is CAA with extra steps. That reasoning
        # is backwards: with contrastive pairs the centroid shift IS the
        # signal, so centring deletes precisely what is being looked for and
        # leaves PC1 fitting noise. Measured on two clouds separated by 6.0
        # along one axis, the centred version scored 1.04 against a null whose
        # worst shuffle reached 1.30 — a real direction that failed its own
        # control, because the estimator had thrown the direction away.
        diffs = pos - neg
        # EXACT SVD, not `torch.pca_lowrank`. That function is a RANDOMISED
        # algorithm -- it draws a projection matrix internally -- and it takes
        # no `generator`, so it draws from the GLOBAL torch RNG. Every `repe`
        # direction was therefore a function of whatever had last touched that
        # RNG, while `fit_direction` published a `seed` and this module's own
        # comment two functions down promises the split and the shuffles are
        # reproducible.
        #
        # MEASURED on 24x32 structureless activations, identical data and
        # identical seed, varying only the global RNG over 400 states: the
        # effect ranged 0.289 to 0.519, `null_max` 0.777 to 0.982, and the
        # margin that decides `beats_null` came within 0.0014 of flipping the
        # verdict. `caa` over the same sweep was identical to the last digit.
        # A seed that does not reproduce its own result is not a seed, and a
        # receipt naming one is claiming something it cannot deliver.
        #
        # The exact call is also cheap here, and would be on a real model: the
        # pair axis is the small one -- half of at least 8 pairs against a
        # d_model in the thousands -- and `full_matrices=False` makes the cost
        # scale with that axis rather than with d_model.
        _, singular, vh = torch.linalg.svd(diffs, full_matrices=False)
        # THE `norm == 0.0` GUARD AT THE BOTTOM CANNOT FIRE FOR THIS BRANCH,
        # which is why the degenerate case is caught here instead. Handed
        # all-zero differences — layer 0 with a shared final token, the exact
        # case the sweep's no-direction row exists for — LAPACK returns zero
        # singular values and an ARBITRARY orthonormal V, so `vh[0]` comes back
        # a perfectly good unit vector. MEASURED, twelve identical pairs: it
        # returned `e_0` with a norm of exactly 1.0, and the sweep published
        # `effect 0.0, p_value 1.0` carrying the note "this direction does not
        # beat its own label-shuffled refits" — a verdict about eight shuffles
        # that had each estimated a basis vector, on a direction nobody fitted.
        # The panel then drew it as a bar. `caa` refused the same input, so the
        # whole no-direction path was live on one estimator and dead on the
        # other, with the defect intact behind the dropdown.
        #
        # The condition is this estimator's own rather than CAA's transplanted:
        # `repe` estimates PC1 of the paired DIFFERENCES, so what makes it
        # degenerate is every difference being zero, not the two means
        # coinciding. Sets whose differences vary but average to nothing still
        # have a real first component, and this leaves that case alone.
        if float(singular[0]) == 0.0:
            raise NoDirection(_NO_DIRECTION_SENTENCE)
        direction = vh[0]
        # PCA has no sign convention; align it with the mean difference so
        # "positive" means the same thing it means for CAA.
        if float(direction @ (pos.mean(0) - neg.mean(0))) < 0:
            direction = -direction
    else:
        raise BadRequest(f"unknown method {method!r} — use one of {', '.join(METHODS)}")

    norm = float(direction.norm())
    if norm == 0.0:
        raise NoDirection(_NO_DIRECTION_SENTENCE)
    return direction / norm


def _separation(direction, pos, neg) -> float:
    """How far the two sets sit apart along this direction, in units of spread.

    A projection difference in raw units is unreadable and not comparable
    across layers; dividing by the pooled standard deviation of the projections
    makes it a standardised effect size, which is what the null is compared
    against.
    """
    import torch

    p = pos @ direction
    n = neg @ direction
    spread = torch.cat([p - p.mean(), n - n.mean()]).std()
    if float(spread) == 0.0:
        return 0.0
    return float((p.mean() - n.mean()) / spread)


def fit_direction(
    states,
    layer: int,
    *,
    method: str = "caa",
    refits: int = NULL_REFITS,
    seed: int = CONTROL_SEED,
) -> tuple[Direction, Any]:
    """(judgement, unit direction). Fit on half the pairs, score on the other.

    `states` is `(positive, negative)` stacked activations for one layer, both
    `[n_pairs, d_model]` and row-aligned: row i of each is one pair.
    """
    import torch

    pos, neg = states
    n = int(pos.shape[0])
    if n != int(neg.shape[0]):
        raise BadRequest(
            f"{n} positive prompts against {int(neg.shape[0])} negative ones — "
            "contrastive pairs must be matched, because the direction is "
            "fitted from their differences."
        )
    if n < MIN_PAIRS:
        raise Refusal(
            f"{n} pairs is not enough to fit a direction that can be checked. "
            f"This needs at least {MIN_PAIRS}, because half are held out for "
            "scoring and a direction scored on its own fitting set separates "
            "it by construction."
        )

    half = n // 2
    # The generator stays on CPU so the split and the shuffles are identical
    # whatever device the activations live on — a control that depends on the
    # accelerator is not reproducible. The masks move to the data.
    gen = torch.Generator().manual_seed(seed)
    device = pos.device
    order = torch.randperm(n, generator=gen).to(device)
    fit_idx, score_idx = order[:half], order[half:]

    direction = _fit(pos[fit_idx], neg[fit_idx], method)
    effect = _separation(direction, pos[score_idx], neg[score_idx])

    # The null: same pairs, same split, same estimator — labels shuffled.
    # Anything the pipeline produces from structureless data shows up here.
    nulls = []
    for _ in range(refits):
        swap = torch.randint(0, 2, (n,), generator=gen).bool().to(device)
        shuffled_pos = torch.where(swap.unsqueeze(1), neg, pos)
        shuffled_neg = torch.where(swap.unsqueeze(1), pos, neg)
        try:
            null_dir = _fit(shuffled_pos[fit_idx], shuffled_neg[fit_idx], method)
        except Refusal:
            continue
        nulls.append(
            abs(_separation(null_dir, shuffled_pos[score_idx], shuffled_neg[score_idx]))
        )

    null_mean = sum(nulls) / len(nulls) if nulls else 0.0
    null_max = max(nulls) if nulls else 0.0
    beats = bool(nulls) and abs(effect) > null_max
    # Textbook permutation p-value. With K draws the smallest attainable value
    # is 1/(K+1), so 8 draws can never assert better than p = 0.111 — which is
    # exactly why the measured false-positive rate below is what it is, and why
    # this number is published rather than only the boolean.
    p_value = (
        (1 + sum(1 for x in nulls if x >= abs(effect))) / (1 + len(nulls))
        if nulls
        else 1.0
    )

    notes = []
    if not nulls:
        notes.append(
            "every shuffled refit collapsed, so there is no null to compare "
            "against and this direction is unverified"
        )
    if not beats and nulls:
        notes.append(
            "this direction does not beat its own label-shuffled refits, so "
            "the separation is not evidence of anything — it is what this "
            "estimator produces from these activations regardless of labels"
        )

    return Direction(
        layer=layer,
        method=method,
        effect=round(effect, 4),
        null_mean=round(null_mean, 4),
        null_max=round(null_max, 4),
        beats_null=beats,
        n_pairs=n,
        n_fit=int(half),
        n_score=int(n - half),
        p_value=round(p_value, 4),
        residual_norm=round(float(torch.cat([pos, neg]).norm(dim=-1).mean()), 3),
        notes=notes,
    ), direction


def _no_direction_note(layer: int) -> str:
    """What a layer with no direction says about itself, naming the layer.

    The layer number is IN the sentence and not only in the row's `layer`
    field, because notes are read away from their row — the panel prints them
    under the chart, `to_dict` hands them to anything that stores the fit — and
    "the two sets have identical mean activations" with no layer attached is
    exactly the sentence that sent this defect to review.
    """
    embedding = (
        " Entering layer 0 the residual stream is the last token's own "
        "embedding, before the model has computed anything, so any two sets "
        "whose prompts end with the same token — a full stop, usually — are "
        "the same vector here."
        if layer == 0
        else ""
    )
    return (
        f"no direction at layer {layer}: the two sets have identical mean "
        "activations there, so there was nothing to fit and nothing to score."
        + embedding
        + " The p-value and the two null statistics are absent rather than "
        "zero — no null was run at this layer, and zero would be the most "
        "confident number in each of those ranges."
    )


def _no_direction_row(states, layer: int, *, method: str) -> Direction:
    """The row a layer with no direction gets. Same shape as every other row.

    Every field here is either a measurement or an absence, and none of them is
    a verdict nobody reached.

    `effect` is 0.0. In the case this actually happens in — layer 0, where
    every prompt's state is the same embedding — that is the reading and not a
    placeholder: the two sets are one cloud, and the separation between a cloud
    and itself is zero along every direction.

    `null_mean` and `null_max` are ABSENT, alongside `p_value`. They were 0.0,
    and that was the one dishonest thing left on this row: no labels were
    shuffled and no refit was scored here, so "the worst label-shuffled refit
    reached 0.000" — which is what the chart's own tooltip says out of that
    field — described eight draws that did not happen. An unknown is not a
    zero. Together the three absences are what carries "no result"; the note
    says it in words for a reader who has only the JSON.

    `residual_norm` is measured, by the same definition `fit_direction` uses:
    the stream has a norm at this layer whether or not anything separates in
    it, and a receipt is not less true for being about a layer with no answer.

    `beats_null` is False, and it is a CLASSIFICATION rather than a verdict
    about a null: it is this row saying it is not a survivor, which is what
    keeps it out of `best_layer` by construction — `sweep` picks the strongest
    among rows that beat their null — so nothing downstream ever reaches for
    the `p_value` that is not here. The words a reader sees for it are the
    note's and the panel's ("no direction here"), never "did not beat its
    null", which would be a claim about a comparison nobody made.
    """
    import torch

    pos, neg = states
    n = int(pos.shape[0])
    # The split `fit_direction` would have used, reported so the row reads like
    # the others rather than like a hole in the table.
    half = n // 2
    return Direction(
        layer=layer,
        method=method,
        effect=0.0,
        null_mean=None,
        null_max=None,
        beats_null=False,
        p_value=None,
        n_pairs=n,
        n_fit=int(half),
        n_score=int(n - half),
        residual_norm=round(float(torch.cat([pos, neg]).norm(dim=-1).mean()), 3),
        notes=[_no_direction_note(layer)],
    )


def _nothing_to_fit(layers: list[int]) -> str:
    """The sentence for a sweep where EVERY requested layer had no direction.

    One such layer among thirty is a row. All of them is a table with no
    reading in it, and that is a refusal — but the sentence has to do the work
    `_fit`'s could not, because `_fit` knows about one layer and not about the
    request: name the layers, name the cause, say what to do instead. The
    single-layer form is the one a reader reaches by asking for `layers=[0]`
    after reading that layer 0 is the embedding.
    """
    named = ", ".join(str(layer) for layer in layers)
    if len(layers) == 1:
        head = (
            f"layer {layers[0]} is the only layer this fit asked for, and the "
            f"two sets have identical mean activations at layer {layers[0]}, so "
            "there is no direction between them to fit."
        )
    else:
        head = (
            f"every layer this fit asked for — {named} — has identical mean "
            "activations for the two sets, so there is no direction between "
            "them to fit anywhere in this sweep."
        )
    # Two causes, and a request can hit both at once — so this is additive
    # rather than a choice. Blaming a shared final token for a degenerate layer
    # 12 would be an explanation nobody checked: the embedding argument is
    # about layer 0 and says nothing about what happens above it.
    why = ""
    if 0 in layers:
        why += (
            " Entering layer 0 the residual stream is the last token's own "
            "embedding, before the model has computed anything, so two sets "
            "whose prompts all end with the same token — a full stop, which is "
            "how sentences end — are the same vector there."
        )
    if any(layer > 0 for layer in layers):
        # "MORE THAN THAT" NEEDS A THAT, and the that is the clause above,
        # which is only written when layer 0 was among the layers asked for.
        # `layers=[7]` therefore opened with "Above layer 0 identical means say
        # more than that", pointing at a sentence the reader was never shown —
        # and that request reaches here straight off the route, since `layers`
        # comes through from the body with nothing done to it but a bounds
        # check. The above-zero clause states its own premise when it has to
        # stand alone, and keeps the shorter back-reference when it does not.
        why += (
            " Above layer 0 identical means say more than that: "
            if 0 in layers
            else " Above layer 0 the residual stream is no longer the raw "
            "embedding, so identical means there say something stronger: "
        )
        why += (
            "the two sets left the same trace in a network that had already "
            "computed something, which is what happens when the two sets are "
            "the same text, or differ only in something this model does not "
            "represent."
        )
    if layers == [0]:
        what = (
            " Ask for layer 1 or above, where the model has computed something "
            "for the two sets to differ in, or end the two sets on different "
            "words."
        )
    elif len(layers) == 1:
        what = (
            " Sweep the whole stack rather than this one layer — a layer with "
            "nothing between the two sets comes back as one row with no result, "
            "not as a failed fit — or give the two sets prompts that differ in "
            "more than their labels."
        )
    else:
        # NOT "sweep the whole stack" HERE, which is what this used to say. A
        # fit with no `layers` in the body sweeps every layer the model has
        # (`runtime.fit_steering_direction`), so for the panel's own button
        # this branch IS the whole stack, and the first thing the refusal told
        # that reader to do was the thing they had just done — a remedy that
        # reproduces the refusal word for word. Nor is widening the sweep
        # advice this function can stand behind: it has looked at every layer
        # in the request and has nothing to say about one it did not look at.
        # What is always true when every layer looked at had nothing in it is
        # that the two sets did not differ.
        what = (
            " Give the two sets prompts that differ in more than their labels "
            "— as written they left the same trace at every layer this fit "
            "looked at."
        )
    return head + why + what


def sweep(model, blocks, positive_ids, negative_ids, layers, *, method: str = "caa"):
    """Fit a direction at every layer and report which ones survive their null.

    Returns `(rows, vectors)` — the judgements and the directions themselves,
    so a caller can steer with one without refitting.
    """
    if method not in METHODS:
        raise BadRequest(f"unknown method {method!r} — use one of {', '.join(METHODS)}")

    pos_states = _last_token_states(model, blocks, positive_ids, layers)
    neg_states = _last_token_states(model, blocks, negative_ids, layers)

    rows, vectors = [], {}
    no_direction: list[int] = []
    for layer in layers:
        try:
            judgement, direction = fit_direction(
                (pos_states[layer], neg_states[layer]), layer, method=method
            )
        except NoDirection:
            # ONE LAYER WITH NOTHING IN IT IS A ROW. This loop used to let the
            # refusal out, and since layer 0 is the last token's embedding —
            # identical for any two sets of sentences that end the same way —
            # the panel's own defaults refused at the first layer and reported
            # nothing about the other twenty-nine, which were fine. The
            # precedent for treating a degenerate estimate as an outcome rather
            # than a failure is already inside `fit_direction`: its null loop
            # skips a shuffle that collapses and scores the rest.
            no_direction.append(layer)
            rows.append(
                _no_direction_row(
                    (pos_states[layer], neg_states[layer]), layer, method=method
                ).to_dict()
            )
            continue
        rows.append(judgement.to_dict())
        vectors[layer] = direction

    # And the one case where it is the whole answer: a table whose every row
    # says "no result" is not a table, and the reader is owed the sentence
    # rather than thirty rows of nothing. `no_direction` is checked for
    # emptiness first so that a caller passing no layers at all gets the empty
    # sweep it asked for instead of this.
    if no_direction and len(no_direction) == len(layers):
        raise Refusal(_nothing_to_fit(no_direction))

    survivors = [r for r in rows if r["beats_null"]]
    return {
        "method": method,
        "layers": rows,
        "best_layer": (
            max(survivors, key=lambda r: abs(r["effect"]))["layer"]
            if survivors
            else None
        ),
        "survived": len(survivors),
        "means": (
            "Standardised separation between the two sets along the fitted "
            "direction, on held-out pairs, beside the same statistic from "
            "label-shuffled refits. A layer that does not beat its own null "
            "is not a weak result — it is no result."
        ),
    }, vectors


# ------------------------------------------------------------- the store (#9)
#
# Persistence for directions, not a panel of its own. A direction is only
# meaningful against the model it came from, so what is stored is the vector
# AND the facts needed to refuse it later: model id, revision, layer, dtype,
# hidden size, how it was derived, and whether it beat its null.


def store_dir():
    """Where saved directions live. Same platform discipline as the trace db.

    ASKING DOES NOT CREATE IT. `paths.py` states the rule in its own module
    docstring — "nothing here creates a directory as a side effect of being
    asked a question" — and this function used to break it, which mattered
    the moment `catalogue()` got a route: opening the steering panel on a
    machine that has never fitted a vector would have written a directory
    into the user's data folder merely by looking. `save()` calls
    `paths.ensure` at the point of writing, which is where the rule says the
    creation belongs.
    """
    from . import paths

    return paths.data_dir() / "vectors"


def _slug(name: str) -> str:
    """A filename that cannot escape the store.

    Names arrive from a text field. `..` and separators are removed rather than
    escaped — this writes files, and a name is not worth a path traversal.
    """
    cleaned = "".join(c if (c.isalnum() or c in "-_") else "-" for c in name).strip("-")
    if not cleaned:
        raise BadRequest(
            "a vector name needs at least one letter or digit — it becomes a filename."
        )
    return cleaned[:80]


def _store_path(name: str):
    """`store_dir()/<slug>.json`, proven to be inside the store before it is used.

    THE SLUG ALREADY MAKES THIS UNREACHABLE, and the check is here anyway.
    `_slug` is a WHITELIST — every character that is not alphanumeric, `-` or
    `_` becomes `-` — so a separator, a colon and a dot cannot survive it, and
    `..` arrives as `--` and is then stripped to nothing and refused. Measured
    against twelve traversal payloads (`../../etc/passwd`, the Windows
    separator form, an absolute path, a drive letter, a doubled-up
    `....//....//`, a NUL byte, a percent-encoded `%2e%2e%2f`): none produced a
    path outside the store, and `..` and `.` were refused outright.

    So why write it. Two reasons, and neither is the vulnerability.

    First, the guarantee was implicit in a comprehension. `remove()` calls
    `path.unlink()` on a name that arrives from `DELETE
    /api/steer/directions/{name}`, which is an unauthenticated local route —
    the strongest thing in the store's blast radius — and the only thing
    standing between that and the filesystem was a reader's willingness to
    reason about a generator expression. A containment check states the
    property where the delete happens, in one line a reader can check without
    reasoning at all.

    Second, CodeQL cannot see it. `py/path-injection` flagged both the join and
    the unlink at high severity, and it is right to: it has no way to know that
    comprehension is a sanitiser. Suppressing that with a `# nosec` would be
    asserting the answer; `resolve()` plus `is_relative_to` is the answer, and
    it is the form the analysis recognises. An alert dismissed by assertion
    stays dismissed when the sanitiser later changes.

    Raises rather than returns on a miss: if this ever fires, the slug has
    stopped doing its job, and the honest response to a filename that escaped
    its directory is not to carry on with a different one.
    """
    root = store_dir().resolve()
    path = (root / f"{_slug(name)}.json").resolve()
    if not path.is_relative_to(root):
        raise BadRequest(
            f"the name {name!r} does not resolve to a file inside the "
            "direction store, so nothing will be read or written for it."
        )
    return path


def _existing(name: str):
    """The stored file this name refers to, found in the store's own listing.

    A LOOKUP, NOT A JOIN, and that is the whole difference. `_store_path` has
    to build a path because `save` writes a file that does not exist yet;
    `remove` and `load` are asking about a file that either exists or does not,
    and the answer to that is already on disk. Matching `_slug(name)` against
    the stems `glob` returns means the path those two hand to `unlink` and
    `read_text` comes from the DIRECTORY, not from the request — there is no
    string from the URL anywhere in it.

    That is a real property and not a formality. `_store_path`'s containment
    check is a guard placed in front of a join; this has nothing to guard,
    because nothing was joined. It is also what CodeQL's `py/path-injection`
    was pointing at: it kept reporting `remove` after the containment check
    landed, and it was right that a URL segment still reached `unlink`, even
    though the value could not escape. Now it does not reach it at all.

    Returns None for "not here", which both callers already have a sentence
    for. `glob` rather than `iterdir` so a directory that has been deleted
    underneath us is an empty answer rather than an exception.
    """
    directory = store_dir()
    if not directory.is_dir():
        return None
    wanted = _slug(name)
    for path in directory.glob("*.json"):
        if path.stem == wanted:
            return path
    return None


def _occupant(name: str):
    """The stored file this name's slug would be written over, and whose it is.

    `_slug` IS MANY-TO-ONE, and `save` used to write straight through it. Every
    character outside `[A-Za-z0-9_-]` becomes `-` and the result is cut at 80,
    so there are three separate ways for two directions a user thinks of as
    distinct to arrive at one filename:

        "sycophancy v2" and "sycophancy-v2"        -> sycophancy-v2
        "Sycophancy" and "sycophancy"              -> one file on Windows and
                                                      macOS, two on Linux
        two names differing only past character 80 -> the same 80 characters

    In all three the second `save` overwrote the first and returned success.
    Nothing said so, and the loser was somebody's measurement — the exact
    silent-wrongness this module exists to refuse.

    Returns `(path, stored_name)` when a file is already standing there, or
    None when the slug is free. `stored_name` is None for a file this version
    cannot read, which is a real state: `catalogue()` already lists damaged
    files rather than dropping them, so one can be occupying the slug.

    A LOOKUP, NOT A JOIN — the same reason `_existing` is written this way.
    The path handed back comes off `glob`, so nothing derived from the request
    is used to read a file.

    The comparison folds case, and that is a decision rather than an accident.
    Two names differing only in case are one file on Windows and macOS and two
    on Linux, and these files get copied between machines and shared: a store
    that means something different depending on which filesystem it is sitting
    on is worse than one that refuses the ambiguity everywhere. `_existing`
    stays case-sensitive, because after this nothing can create the pair it
    would have to disambiguate.
    """
    import json

    directory = store_dir()
    if not directory.is_dir():
        return None
    wanted = _slug(name).casefold()
    for path in directory.glob("*.json"):
        if path.stem.casefold() != wanted:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError):
            payload = None
        # `json.loads` succeeding does not mean an object came back — a file
        # holding `[]` parses fine and has no `.get`. Every shape that cannot
        # name itself lands on None and is refused rather than overwritten:
        # unreadable, valid JSON that is not an object, an object with no
        # `name`, and a `name` that is not a string.
        stored = payload.get("name") if isinstance(payload, dict) else None
        return path, stored if isinstance(stored, str) else None
    return None


def save(name: str, direction, meta: dict) -> dict:
    """Write a direction with the provenance needed to judge it later.

    JSON, not safetensors: a `d_model` vector is a few thousand floats, and a
    format `modelmri open` can read without torch is worth more here than the
    bytes saved.
    """
    import json
    from datetime import datetime, timezone

    from . import paths

    required = ("model", "layer", "hidden_size", "method", "dtype")
    missing = [k for k in required if meta.get(k) in (None, "")]
    if missing:
        raise BadRequest(
            f"a saved direction needs {', '.join(missing)} — a vector without "
            "its provenance cannot be checked against a model later, and "
            "steering with the wrong one produces plausible nonsense."
        )

    # Whose file is standing on this slug, before anything is written over it.
    # Re-saving under your own name stays allowed — that is how a direction
    # gets corrected — but it stops being silent, so a caller can tell the
    # difference between writing a new one and replacing one.
    occupied = _occupant(name)
    replaced = False
    if occupied is not None:
        standing, stored_name = occupied
        if stored_name is None:
            raise Refusal(
                f"{standing.name} is already in the direction store and cannot "
                "be read, so there is no way to tell whether it is this "
                f"direction or a different one. Nothing was written. Delete "
                f"{standing.name} if it is yours, then save again."
            )
        if stored_name != name:
            raise Refusal(
                f"the name {name!r} becomes the same filename as the saved "
                f"direction {stored_name!r} ({standing.name}), so saving it "
                "here would overwrite that one. Nothing was written. Pick a "
                f"name that differs by more than punctuation, case or the "
                "characters past the eightieth, or delete "
                f"{stored_name!r} first."
            )
        replaced = True

    paths.ensure(store_dir())
    path = _store_path(name)
    payload = {
        "name": name,
        # STAMPED HERE, because `catalogue()` sorts on it and promises
        # "newest first". Nothing wrote this key for the first two versions of
        # the store, so every row sorted equal on an empty string and the
        # order a reader saw was whatever `glob` returned — a false claim in a
        # tool whose whole discipline is not making them. Before `**meta` so a
        # caller re-saving an imported vector can carry its original date
        # through rather than restamping it as new.
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "values": [float(x) for x in direction.detach().float().cpu().tolist()],
        **meta,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "name": name,
        "path": str(path),
        "dims": len(payload["values"]),
        "replaced": replaced,
    }


def remove(name: str) -> dict:
    """Delete one saved direction, or refuse by name.

    The store is the only place these live, so this is the only way to take
    one out — and a delete that silently succeeds on a name that was never
    there teaches a reader their typo worked.
    """
    path = _existing(name)
    if path is None:
        raise Refusal(
            f"there is no saved direction called {name!r}, so there is "
            "nothing here to delete."
        )
    path.unlink()
    return {"removed": name, "path": str(path)}


def load(name: str, *, hidden_size: int, model: str = ""):
    """Read a direction back, refusing a shape it cannot belong to.

    Dimension mismatch is refused by name — the same rule `saes.py` applies to
    `d_in`. A different model with the SAME hidden size is warned about rather
    than blocked: lifting a direction from a base model onto its finetune is a
    legitimate experiment when the person running it knows that is what they
    are doing, and a wrong-but-plausible steer is exactly what the warning is
    for.
    """
    import json

    import torch

    path = _existing(name)
    if path is None:
        raise Refusal(f"no saved direction called {name!r} in {store_dir()}.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = payload["values"]
    except (OSError, UnicodeDecodeError, ValueError, KeyError) as err:
        raise Refusal(f"{path.name} is not a direction this version can read.") from err

    if len(values) != hidden_size:
        # BOTH MODELS BY NAME. "shapes disagree" is a sentence about tensors
        # and this is a question about provenance: the reader has to be able
        # to see which end is the wrong one, and the only way to see that is
        # to be told what the direction came from and what it is being pushed
        # into. `model` falls back to "this model" for a caller that did not
        # name one, which is still better than a bare number.
        raise Refusal(
            f"{name!r} is a {len(values)}-dimensional direction and "
            f"{model or 'this model'}'s residual stream is {hidden_size}. It "
            f"was fitted on {payload.get('model') or 'another model'}. "
            "Refusing rather than reshaping it into something that would "
            "steer, plausibly, at random."
        )

    warnings = []
    origin = payload.get("model", "")
    if model and origin and origin != model:
        warnings.append(
            f"this direction was fitted on {origin} and you are steering "
            f"{model}. The hidden sizes match, but equal size is not equal "
            "basis — the result may be confident and meaningless."
        )
    if payload.get("beats_null") is False:
        warnings.append(
            "this direction did not beat its own label-shuffled null when it "
            "was fitted, so it was never evidence of anything."
        )
    return torch.tensor(values), payload, warnings


def relative_strength(alpha: float, residual_norm: float | None) -> float | None:
    """`alpha` as a multiple of the stream's own norm, or None when unknown.

    One function, because this arithmetic is published in three places — the
    status, the receipt and the slider's own label — and three copies of a
    division is three chances for the panel to say one thing while the receipt
    says another.

    Every direction in this package is unit norm (`_fit` divides by it,
    `probe.direction_at` divides by it, `saes.SAEHandle.steering_vector`
    divides by it), so the applied coefficient IS in raw residual-stream
    units and the honest relative figure is exactly this quotient.

    `None` in, `None` out, and `None` for a norm of zero — an unmeasured
    strength is not a small one, and 4.0 / 0 is not infinite push, it is a
    measurement that did not happen.
    """
    if residual_norm is None or residual_norm == 0.0:
        return None
    return alpha / residual_norm


def catalogue() -> list[dict]:
    """Every saved direction, without its values. Newest first."""
    import json

    directory = store_dir()
    # A store that was never written to is an empty catalogue, not an error
    # and not a directory this call creates. See `store_dir`.
    if not directory.is_dir():
        return []

    rows = []
    for path in directory.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError):
            # A file this version cannot read is listed as unreadable rather
            # than dropped: a vector silently missing from its own catalogue
            # is worse than one that says it is damaged.
            rows.append({"name": path.stem, "unreadable": True})
            continue
        payload.pop("values", None)
        payload["dims"] = payload.get("hidden_size")
        rows.append(payload)
    return sorted(rows, key=lambda r: r.get("saved_at", ""), reverse=True)
