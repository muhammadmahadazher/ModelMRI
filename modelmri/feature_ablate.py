"""Which features actually changed the answer?

The features panel ranks by raw activation. That answers "what fired", and it
is read as "what mattered" — the same gap `ablate.py` closed for heads and
`attribute.py` closed for prompt tokens. This closes it for features: remove
one feature's contribution from the residual stream, run the model again, and
measure how far the next-token distribution at `position` moved. The KL is
imported from ablate.py rather than defined again, so a head score, a token
score and a feature score on one screen mean the same thing.

Every number below was measured on this machine before this file existed:
gpt2, float32, cuda, eager attention; SAE
jbloom/GPT2-Small-SAEs-Reformatted @ blocks.8.hook_resid_pre (d_sae 24576),
calibrated by saes.py to `centered+b_dec`; prompt "The Eiffel Tower is located
in the city of" (11 tokens); attributing at position 10, top token " Paris" at
p=0.06378; 596 forward passes at ~33 ms. A KL without those conditions
attached cannot be checked by anyone.

**This is only meaningful because the SAE is calibrated.** Before saes.py
learned to choose its input convention, the same SAE on the same hook scored
FVU 13579.24 with L0 7491.5. Ranking 7,491 simultaneously-firing features
would have been ranking noise with a confident number attached, which is why
an unusable calibration is refused here rather than plotted.

**One intervention, and the obvious three are not three.** Subtracting
`act[f] * W_dec[f]` from the stream (a), and zeroing feature f, decoding, and
adding the reconstruction error back (c), are not merely close: decode is
affine, so `decode(feats with f zeroed) + (x - recon) == x - act[f]*W_dec[f]`
identically. Measured max |x_a - x_c| = 3.81e-06 over the edited stream,
2.18e-07 relative to the edit's own norm — float32 error. Probe feature 5856
(activation 35.546): (a) 0.4174529, (c) 0.4174514, and the cheap one is (a),
which is one rank-1 subtraction with no decode at all.

Replacing the stream with the SAE's reconstruction (b) is a different edit and
it is disqualified by its own no-op: substituting the reconstruction while
removing NOTHING already costs 0.0775 nats at the attribution position
(0.2212 applied at every position), which exceeds the single-feature effect of
41 of the 43 features firing there. On feature 16649 (activation 1.871) it
reports 0.0855506 against (a)'s 0.00054927 — 156x, almost all of it the
discarded reconstruction error. It also reorders (5 of 8 shared with (a)'s
top-8) and its weakest candidate still reads 0.0731 nats, so shipped, every
feature would have looked important and in the wrong order.

**Centering, and the reason (a) is right is not the obvious one.** The winning
convention on this SAE centers, so the SAE reconstructs `x - mean(x)` along
d_model and the feature's contribution lives in that centered space. Removing
it there gives `c' = (x - mu) - act*W_dec[f]`, and the stream handed back to
the model is `c' + mu`, i.e. `x - act*W_dec[f]` — the ORIGINAL per-token mean
re-added, not the edited stream's own. That is why (a) and (c) coincide.

It is NOT because subtracting in the centered and the raw space are the same
subtraction; that sentence stood here and is measurably false. `act*W_dec[f]`
does not have zero d_model mean, so centering strips part of it: measured at
position 10, mean(act*W_dec[5856]) = -0.0903948, whose |mean|*sqrt(768) =
2.5051 is 7.05% of the edit's own norm 35.5482 (feature 11149: -0.0467742,
1.2962, 4.26%). The visible consequence is that the edited stream's own
d_model mean moves — 0.0786990 to 0.1690938 at position 10 for the 5856 edit,
2.1x — which is correct and intended, because `mu` is held at the value the
decomposition was taken with rather than recomputed.

**What the edit does, and the two things it was claimed to do and does not.**
The stream moves by exactly one rank-1 term: measured through the hook in
float32, max |stream the model received - (x - act*W_dec[f])| is 0.0 and no
row but the edited one changes. That is the mechanism, it is a property of the
edit and the dtype, and `removal_verified` is now that check and nothing else.

The first thing it was claimed to do is take the feature out of the SAE's
reading of the stream. It usually does not. Re-encoding the edited stream at
the edited position leaves 38 of the 43 features firing there reading ABOVE
the 1% tolerance, from 10.1% of the original activation up to 60.3%; the 5
that read 0.0 do so because relu clamped an OVERSHOOT, not because the removal
was clean — feature 5856's pre-activation goes 35.546 to -2.331, a drop of
37.877 against an activation of 35.546. The cause is that the encoder and
decoder directions of a feature are not dual: `W_enc[:,f] . W_dec[f]` has mean
0.8387 over d_sae, min -0.3819, max 1.3072, and for the top-8 firing features
[0.9081, 0.9492, 0.7369, 0.6105, 0.4356, 0.8710, 0.8452, 0.5117]. This varies
per feature from 0% to 60%, so it is reported PER ROW (`encoder_residual`),
which costs no forward pass at all — the re-encode of the cast stream agrees
with one taken through the model to 3.6e-06.

The second is that nothing else in the stream moves. Nothing else in the
STREAM moves, but the SAE's decomposition of the result is not the original
decomposition minus one row. Removing 5856 (activation 35.546) at position 10
moves 44 other features by more than 1e-6, drives 33 previously-firing
features to exactly zero, starts 2 silent ones firing, and moves a total of
42.4943 of activation outside the target — 119.55% of what it removed inside
it. `||err||` at that position goes 21.3036 to 31.8553, so the unmodelled
remainder grows by half while the edit is being measured. Removing 11149:
39.7637 moved (130.66%), 28 firing features killed, err 21.3036 to 31.7511.

**A score is not only the feature; part of it is the size of the edit.**
Subtracting a random Gaussian direction of the SAME norm at the SAME tokens is
not free: at feature 5856's norm of 35.5, five draws cost
[0.105611, 0.109279, 0.091864, 0.066590, 0.089114] nats against the feature's
own 0.417461, and five decoder directions of features that do not fire there
cost [0.086332, 0.051339, 0.063355, 0.101893, 0.140937]. So the top row clears
its own control by about 4x, not by everything. One control per scored row is
therefore measured and returned (`control_kl`, `clears_control`), which doubles
the passes and is the reason the cost is `2 x tested + 6`. Measured on the
reference prompt, 34 of 43 rows clear their own-norm control and 9 do not —
including 22852 and 1288, which sit 5th and 6th in the activation chart. It is
ONE draw per row with a fixed seed: at the top row's norm the five draws above
span 0.0666 to 0.1093, a factor of 1.6, so a row within that factor of its
control is not separated by this test.

**The floor is zero and the resolution is not.** The edit hook installed with
the captured stream written back unchanged scores KL 0.0 against the plain
pass, exactly, on 4 repeats, matching a no-hook replay. But two scores in the
position-local ranking came back NEGATIVE, -1e-08 and -3e-08, which is
impossible for a KL and is float32 summation over 50257 vocabulary entries. So
the floor is 0.0 with a numerical resolution of about 1e-7 nats, `RESOLUTION_KL`
is set an order above that, and a caller greying out "at or below the floor"
would be greying out nothing. 2 of 43 position-local scores and 11 of 494
global scores land at or below 0.0.

**These do not add up, and they miss in the OPPOSITE direction from heads.**
Over the 43 features firing at the attribution position the singles sum to
0.66446 while one joint ablation removing all 43 gives 2.135221 — the singles
UNDER-count by 3.2x. On the top-8 alone, 0.660624 against 1.34494 (2.0x
under); globally over all 494 features, 0.811683 against 5.862094 (7.2x
under). Head ablation on gpt2 layer 0 OVER-counts 8x, and ablate.py's wording
for that must not be copied onto these numbers. The algebra says why: removing
every firing feature leaves the stream equal to `err + b_dec + mean`, which
guts it, and the model's response is superlinear.

**The calibration's FVU is not the number to print beside a feature, and it is
not even in the same units.** FVU 0.000984 and rel_err 0.029397 are aggregates
over all 11 tokens and both are dominated by token 0, whose residual-stream
norm is 3077.3 against 94.6-116.8 for every other token — the GPT-2 attention
sink — and whose relative error is 0.012. Per-token ||x-recon||/||x|| in the
calibrated space:
[0.012, 0.186, 0.367, 0.425, 0.258, 0.177, 0.205, 0.208, 0.199, 0.209, 0.204].
At the attribution position the SAE fails to model 20.36% of the stream's norm
(21.30 against 104.65); at token 3, 42.5%.

`residual_share` is a NORM fraction and `fvu` is a SQUARED-error fraction, so
putting 0.204 next to 0.000984 states a 200x gap where the like-for-like one is
7x. `rel_err` (0.029397) is the aggregate in the same units as
`residual_share` and is returned here for exactly that comparison; the
squared-units version of the same contrast is per-token FVU 0.00541 at
position 10 against the aggregate 0.000984, 5.5x. Either pairing is honest.
Mixing them is not.

**And the error has to be measured over the window the edits use.** At
`scope="prompt"` the edits land at every token where a tested feature fires,
so a baseline substituted at one token understates what the decomposition gets
wrong during the measurement. Measured: substituting the reconstruction at
position 10 alone costs 0.077530 nats; over positions 0-10 it costs 0.221217,
2.85x more. So `residual_kl` follows the scope, `residual_share` is the WORST
per-position share over that same window (0.425 at prompt scope on this
prompt, 0.204 at position scope), and `residual_share_at_position` is kept
beside it for the token being attributed. At position scope only 2 of 43
features clear 0.0775; at prompt scope only 1 clears 0.2212.

**It does differ from the activation ranking, and more than a little.**
Position-local, top-8 by activation [5856, 11149, 2194, 21062, 22852, 1288,
18994, 16649] against top-8 by ablation KL [5856, 11149, 2194, 21062, 6807,
8628, 22852, 1288]: 6 of 8 overlap, Spearman over all 43 candidates 0.965,
top 4 identical and in order. 6807 (activation 1.328) and 8628 (1.561) enter
the causal top-6 from outside the plotted set while 18994 (2.114, plotted 7th)
falls to 12th. The magnitudes are the bigger story: activations span 35.5 to
1.87 (19x) while the KLs span 0.417 to -3e-08, dropping 25x from rank 1 to
rank 2 and another 13x to rank 3. The bar chart's smooth decay is not what the
causal picture looks like. Ranking across the whole prompt
(`scope="prompt"`) differs far more — top-8 [5856, 11149, 19941, 1066, 2194,
7703, 20110, 2319], only 3 of 8 shared with the activation top-8 the panel
plots at this token — because features firing at EARLIER tokens reach the
prediction through attention: 19941 fires only at token 1, 1066 only at token
4, 7703 at 2 and 3, 20110 at 3, and the panel cannot show any of them today.
104 of those 494 score above 1e-4, and 139 fall below this file's resolution.

**Activations are re-encoded in float32 here, not read from runtime's cache.**
`runtime._compute_features` stores features as float16 for display. Ablating
with fp16-rounded activations moved the top feature's KL from 0.4174529 to
0.4170800 — 0.09% low, max activation error 0.0916. Small, and the error is
proportional to the activation, so it is largest exactly where the ranking is
decided.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import torch

from modelmri.ablate import distribution, kl_nats
from modelmri.errors import BadRequest

# The one edit this file will make, named so the caller can put it on screen.
# ablate.py's lesson is that an unlabelled importance number is the lie, and
# here the label carries more than usual: "removing this feature" is three
# different experiments in the literature and two of them are measurably
# indefensible on this SAE (see the module docstring).
SUBTRACT = "subtract_decoder_direction"
INTERVENTIONS = (SUBTRACT,)

# Which features are on trial. "position" ablates a feature only at the token
# being attributed — 43 candidates on the measured prompt, 92 passes. "prompt"
# removes it wherever it fires at or before that token — 494 candidates, 256
# tested, 518 passes. They give different answers and the second is the one
# that finds features the current panel cannot show at all: 4 of its top-8 fire
# only at earlier tokens and reach the prediction through attention.
SCOPES = ("position", "prompt")

# Every scored row costs TWO passes, not one: the feature's own edit and a
# random direction of the same norm at the same tokens, because a score is
# partly the size of the edit and the module docstring has the numbers. So the
# cost is `2 * tested + 6`, and this cap is what keeps prompt scope inside a
# minute on a CPU.
#
# Measured cost is ~33 ms per pass on this box (gpt2, fp32, cuda, 11 tokens)
# and roughly 20x that on CPU, so this cap is ~17 s of GPU and ~6 min of CPU.
# It does not bite at scope="position" (43 candidates); it bites at
# scope="prompt" (494), and the ordering it truncates by is peak activation —
# which is exactly the ordering this file exists to say is not the causal one.
# Measured, at prompt scope on the reference prompt, that ordering shares 0 of
# 8 with the causal top-8.
#
# So the cap was checked against an uncapped run rather than assumed safe:
# 494 candidates tested in full against the 256 this admits, the two top-20s
# are identical, and the best feature the cap drops sits at causal rank 42 of
# 494 with KL 0.00053383 — below the 0.077529 the SAE's own reconstruction
# error costs at that position. That is one prompt on one model, which is why
# truncation is still reported in three places: a feature the cap dropped was
# NOT TESTED, not found unimportant, and on a longer prompt it could be one
# that mattered.
MAX_CANDIDATES = 256

# Below this, a score is arithmetic. NOT the noise floor: the floor is exactly
# 0.0 here and two measured scores came back negative (-1e-08, -3e-08), which
# puts the real resolution of a float32 KL over 50257 vocabulary entries near
# 1e-7. A panel greying out "at or below the floor" would grey out nothing.
RESOLUTION_KL = 1e-6

# How far the write-back may move the answer before we stop. Editing the
# stream is only a measurement if the hook writes to the same tensor the
# capture came from; if it does not — a resid_post SAE hooked on the block's
# input, say — every score below would be the difference between two sides of
# a transformer block. Measured 0.0 exactly, four repeats, so this is slack for
# accelerators that are not bit-reproducible, and it is compared against the
# no-hook replay so a nondeterministic path raises its own bar.
WRITEBACK_TOLERANCE = 1e-6

# How far the stream the model RECEIVED may differ from the edit that was
# intended. This is the mechanism check: the edit is `x - act*W_dec[f]` and
# nothing else, so a dtype or a device that rounded it away, or a hook that did
# not land, shows up as a deviation here. Measured exactly 0.0 in float32
# through the hook on gpt2 blocks.8.hook_resid_pre, with no row but the edited
# one changed; the tolerance is slack for accelerators that are not
# bit-reproducible, not room for a swallowed edit.
EDIT_TOLERANCE = 1e-4

# The share of a feature's original activation the SAE's ENCODER may still
# report after that feature's contribution has been subtracted. This is NOT the
# mechanism check and it does not gate anything — it is a per-row property of
# the SAE, and on the reference SAE it fails far more often than it passes: 38
# of 43 rows read above this, from 10.1% to 60.3%, because W_enc[:,f] and
# W_dec[f] are not dual (dot product mean 0.8387 over d_sae, min -0.3819, max
# 1.3072). The five that read 0.0 do so because relu clamped an overshoot.
# Reported per row so a reader can see which rows those are; a single verdict
# taken on one row would have been a green tick the data contradicts 38 times.
ENCODER_RESIDUAL_TOLERANCE = 0.01

# One control per scored row, drawn from a fixed seed so two runs of the same
# request agree. A Gaussian direction rather than another feature's decoder
# direction: the question is what ANY edit of that size costs, and the decoder
# alternative measured slightly cheaper on the reference prompt
# ([0.086, 0.051, 0.063, 0.102, 0.141] against [0.106, 0.109, 0.092, 0.067,
# 0.089] at norm 35.5), so the Gaussian is the stricter of the two.
CONTROL_SEED = 0


class FeatureAblationError(RuntimeError):
    """We cannot take this measurement, and we say why rather than guess."""


def _register_capture(block: torch.nn.Module, point: str, sink: list) -> Any:
    """Collect the residual stream at the SAE's own hook point.

    resid_pre is the block's INPUT, resid_post its OUTPUT. saes.py already
    refuses anything else, and reading the wrong side of a block does not
    error — it yields features for activations the SAE never saw.
    """
    if point == "resid_post":

        def post(module, args, output):  # torch's signature
            sink.append((output[0] if isinstance(output, tuple) else output).detach())

        return block.register_forward_hook(post)

    def pre(module, args):  # torch's signature
        sink.append(args[0].detach())

    return block.register_forward_pre_hook(pre)


def _register_edit(block: torch.nn.Module, point: str, x: torch.Tensor) -> Any:
    """Write `x` ([1, S, d_in]) into the stream at the SAE's hook point."""
    if point == "resid_post":

        def post(module, args, output):  # torch's signature
            return (x,) + output[1:] if isinstance(output, tuple) else x

        return block.register_forward_hook(post)

    def pre(module, args):  # torch's signature
        return (x,) + args[1:]

    return block.register_forward_pre_hook(pre)


def rank_features(
    model: Any,
    block: torch.nn.Module,
    ids: torch.Tensor,
    sae: Any,
    *,
    position: int,
    scope: str = "position",
    intervention: str = SUBTRACT,
    max_candidates: int = MAX_CANDIDATES,
    decode=None,
) -> dict:
    """Rank SAE features by how far removing one moves the answer.

    `block` is the module the SAE is attached to (`runtime._block(sae.layer)`),
    `ids` one unbatched sequence `[1, S]`, `sae` a calibrated-or-calibratable
    `SAEHandle`. `position` is the index whose next-token distribution is
    attributed; `decode` turns a token id into a string for the readout.

    The residual stream is captured here rather than passed in, from the same
    forward pass that produces the base distribution and through the same hook
    point the edit will use. That is deliberate: the stream that is edited has
    to be the stream that was measured, and a caller handing in activations
    from an earlier generation would produce a ranking about a different
    prompt with no way to notice.

    Features are re-encoded in float32. `runtime._compute_features` caches them
    as float16 for display, and fp16-rounded activations moved the top
    feature's KL by 0.09% on the measured prompt — small, proportional to the
    activation, and therefore largest exactly where the ranking is decided.

    Cost is `2 * tested + 6` forward passes. The second per row is the control:
    a random direction of the same norm at the same tokens, without which a
    score cannot be told apart from the size of the edit that produced it.
    """
    # BadRequest, not FeatureAblationError, for the three that are parameters
    # in a URL — runtime.py turns a FeatureAblationError into a 409 "ModelMRI
    # decided not to answer", which is the wrong sentence for `?scope=banana`.
    # errors.py names an unknown enum value as the type example of a 422, and
    # ablate.py and attribute.py both already draw the line here.
    if intervention not in INTERVENTIONS:
        raise BadRequest(
            f"unknown intervention {intervention!r} — this measurement offers "
            f"only {', '.join(INTERVENTIONS)}. Replacing the stream with the "
            "SAE's reconstruction is the obvious alternative and it is "
            "disqualified by its own no-op: on gpt2 blocks.8.hook_resid_pre it "
            "costs 0.0775 nats before removing anything, which is more than 41 "
            "of the 43 features firing at the attribution position score in "
            "total."
        )
    if scope not in SCOPES:
        raise BadRequest(
            f"unknown scope {scope!r} — use one of {', '.join(SCOPES)}. "
            "'position' ranks the features firing at the attributed token; "
            "'prompt' removes each feature wherever it fires at or before it."
        )
    if ids.dim() != 2 or int(ids.shape[0]) != 1:
        # A plain RuntimeError: runtime.py builds `ids` itself out of
        # `last_ids`, so a violation is this package contradicting itself and
        # belongs on the 500 path with a traceback, not in front of a reader
        # who cannot act on it.
        raise RuntimeError(
            f"feature ablation needs one unbatched sequence shaped [1, S], got "
            f"{tuple(ids.shape)}. Batching changes the kernel and the noise "
            "floor measured for this path was measured unbatched."
        )
    seq = int(ids.shape[1])
    if not 0 <= position < seq:
        raise BadRequest(f"position {position} is outside a sequence of {seq} tokens.")
    if max_candidates < 1:
        raise RuntimeError("max_candidates must be at least 1.")

    started = time.perf_counter()
    passes = 0

    # ---- base pass, and the stream it ran on -------------------------------
    sink: list[torch.Tensor] = []
    handle = _register_capture(block, sae.point, sink)
    try:
        with torch.no_grad():
            base = distribution(model(ids).logits[0, position])
    finally:
        handle.remove()
    passes += 1
    if not sink:
        raise RuntimeError(
            "the capture hook on the SAE's block never fired, so there is no "
            "residual stream to ablate a feature out of."
        )
    captured = sink[0]
    x = captured[0].detach().to("cpu").float()  # [S, d_in]
    if x.shape[-1] != sae.d_in:
        raise RuntimeError(
            f"captured a stream of width {x.shape[-1]} for an SAE with "
            f"d_in={sae.d_in}; runtime.py checks this at load time, so "
            "reaching it means the wrong block was handed in."
        )

    # ---- the SAE's claim, and whether it is checkable ----------------------
    feats = sae.encode(x)  # calibrates on first use; [S, d_sae] float32
    cal = sae.calibration
    assert cal is not None  # encode() always leaves one behind
    if not cal.usable:
        # THE FIRST REFUSAL. FVU >= 1 means the reconstruction carries less of
        # the activation than a constant vector would, so the features are not
        # a decomposition of anything and their causal effects would be the
        # causal effects of arbitrary directions. Measured, this is not
        # hypothetical: the same SAE on the same hook scored FVU 13579.24
        # before saes.py calibrated its input convention, with 7491.5 features
        # firing per token. The number goes in the message because "unusable"
        # without it is an opinion.
        raise FeatureAblationError(
            f"this SAE does not reconstruct the stream it is attached to — "
            f"fraction of variance unexplained {cal.fvu:g} at "
            f"{sae.hook}, against the {cal.unusable_at:g} above which the "
            f"reconstruction carries less than a constant vector would (best "
            f"convention tried: {cal.convention}). Its features are not a "
            "decomposition of these activations, so ranking them by causal "
            "effect would rank arbitrary directions. It would work with an SAE "
            "trained on this model at this hook point."
        )

    # `mu` is the per-token d_model mean the calibrated convention removed, and
    # it is HELD rather than recomputed. The feature's contribution lives in the
    # centered space, so removing it gives (x - mu) - act*W_dec[f], and the
    # stream handed back is that plus the ORIGINAL mu — which is why the edit
    # below writes into raw x directly and why it equals intervention (c). It is
    # not because centering leaves the subtraction alone: act*W_dec[f] has a
    # non-zero d_model mean (measured -0.0903948 for feature 5856 at position
    # 10, 7.05% of the edit's norm), so the edited stream's OWN mean moves, by
    # design.
    mu = x.mean(-1, keepdim=True) if cal.center else torch.zeros_like(x[:, :1])
    recon = sae.decode(feats)
    err = (x - mu) - recon

    # ---- passes ------------------------------------------------------------
    def as_received(edited: torch.Tensor) -> torch.Tensor:
        """The edited stream after the round trip into the model's own dtype.

        Every check below reads THIS rather than the float32 tensor this
        function built, so a dtype that rounded the edit away is visible
        instead of assumed absent.
        """
        return edited.to(device=captured.device, dtype=captured.dtype).to("cpu").float()

    def run(edited: torch.Tensor | None = None) -> torch.Tensor:
        nonlocal passes
        handles = []
        if edited is not None:
            xd = edited.to(device=captured.device, dtype=captured.dtype).unsqueeze(0)
            handles.append(_register_edit(block, sae.point, xd))
        try:
            with torch.no_grad():
                out = model(ids).logits[0, position]
        finally:
            for h in handles:
                h.remove()
        passes += 1
        return distribution(out)

    def edit(rows: Sequence[tuple[int, list[int]]]) -> torch.Tensor:
        """The intervention: subtract each feature's own contribution.

        One rank-1 subtraction per (feature, position), and the STREAM moves by
        exactly that and nothing else — verified below as `removal_verified`.

        What does NOT follow, and was claimed here: that the SAE's reading of
        the result is the old reading minus one row. Measured on gpt2
        blocks.8.hook_resid_pre at position 10, removing feature 5856 moves 44
        other features by more than 1e-6, drives 33 previously-firing features
        to exactly zero, starts 2 silent ones, and moves 42.4943 of activation
        outside the target against the 35.546 it removed inside it (119.55%);
        the unmodelled remainder at that position grows from 21.3036 to
        31.8553. So this is an edit to the stream, not a surgical deletion from
        the decomposition, and `encoder_residual` on each row is what that
        costs there.
        """
        xn = x.clone()
        for f, positions in rows:
            for p in positions:
                xn[p] = xn[p] - float(feats[p, f]) * sae.W_dec[f]
        return xn

    # The null this ranking is read against. Same norm as the real edit, same
    # tokens, a direction with nothing to do with the feature — so a row that
    # does not clear it is a row whose score is the size of the edit rather
    # than the identity of the feature. Seeded, so the same request twice gives
    # the same controls; drawn in row order, so it is the row order that fixes
    # which draw a row gets.
    gen = torch.Generator().manual_seed(CONTROL_SEED)

    def control(rows: Sequence[tuple[int, list[int]]]) -> torch.Tensor:
        xn = x.clone()
        for f, positions in rows:
            for p in positions:
                norm = float(feats[p, f]) * float(sae.W_dec[f].norm())
                r = torch.randn(sae.d_in, generator=gen)
                xn[p] = xn[p] - r / (float(r.norm()) + 1e-12) * norm
        return xn

    # The floor, twice, because two different things can be zero. `replay` is
    # the plain pass run again with no hook at all — ablate.py's floor.
    # `floor` adds the edit hook and writes the captured stream back unchanged,
    # which is the floor these scores are actually measured against, and it is
    # also the only check that the edit lands where the capture came from.
    replay = kl_nats(base, run())
    floor = kl_nats(base, run(x.clone()))
    if floor > max(replay, WRITEBACK_TOLERANCE):
        raise FeatureAblationError(
            f"writing this model's own residual stream back into {sae.hook} "
            f"unchanged moves its answer by {floor:.3e} nats (a plain replay "
            f"moves it {replay:.3e}). The edit is not landing where the "
            "capture came from, so every feature score would be measuring the "
            "write-back rather than the feature. It would work on a block "
            "whose input can be replaced by the tensor read out of it."
        )

    top_id = int(base.argmax())

    # ---- candidates --------------------------------------------------------
    # ONLY FEATURES THAT FIRE. A feature with activation 0 contributes exactly
    # nothing to the stream, so `x - 0*W_dec[f]` is the identity and its score
    # is the floor by construction. Including them would pad the list with
    # zeros that are indistinguishable from measurements — 24576 rows of which
    # 43 are real, on the measured prompt.
    if scope == "position":
        fires = (feats[position] > 0).nonzero().flatten().tolist()
        rows = [(int(f), [position]) for f in fires]
        peak = {f: float(feats[position, f]) for f, _ in rows}
    else:
        window = feats[: position + 1] > 0
        rows = [
            (int(f), window[:, f].nonzero().flatten().tolist())
            for f in window.any(0).nonzero().flatten().tolist()
        ]
        peak = {f: float(feats[: position + 1, f].max()) for f, _ in rows}

    if not rows:
        raise FeatureAblationError(
            f"no feature of this SAE fires at position {position}"
            + ("" if scope == "position" else " or before it")
            + f" ({sae.hook}, d_sae {sae.d_sae}). There is nothing to remove, "
            "and a list of features whose activation is zero would be 24576 "
            "rows of the noise floor. Attribute at a position where the SAE "
            "sees something."
        )

    n_candidates = len(rows)
    # Truncation orders by peak activation because before the measurement is
    # taken there is no other ordering available — the same argument
    # attribute.py makes for recency. It is a weak one HERE and the response
    # says so: activation order and causal order share 6 of 8 at one position
    # and 3 of 8 across the prompt, so the cap can drop a feature that would
    # have ranked. Reported, never silent.
    rows.sort(key=lambda r: -peak[r[0]])
    tested = rows[:max_candidates]

    def encoder_residual(edited: torch.Tensor, f: int, positions: list[int]) -> float:
        """How much of feature f the SAE still reads once its own term is gone.

        Costs no forward pass: the stream the model receives at a resid_pre or
        resid_post hook IS the tensor written in, so re-encoding the cast copy
        answers the same question as capturing during an edited pass — measured
        agreement 3.6e-06 over the 43 rows of the reference prompt. One column
        of W_enc, not all of them, or this would be 19M multiply-adds per row.

        The worst position, not the mean: at prompt scope a feature removed at
        five tokens can be cleanly gone from four of them.
        """
        seen = sae.encode_feature(edited[positions], f)
        return max(
            (float(seen[i]) / float(feats[p, f]) if float(feats[p, f]) else 0.0)
            for i, p in enumerate(positions)
        )

    def row_of(f: int, positions: list[int]) -> dict:
        edited = edit([(f, positions)])
        after = run(edited)
        kl = kl_nats(base, after)
        ctrl = kl_nats(base, run(control([(f, positions)])))
        return {
            "feature_id": f,
            "activation": round(peak[f], 4),
            "positions": positions,
            # Eight places, not five. Rounding to five would print the two
            # measured negative scores (-1e-08, -3e-08) as -0.0 and hide the
            # evidence that the resolution is not the floor.
            "kl": round(kl, 8),
            # What a random direction of the same norm at the same tokens cost.
            # A row scoring under this is not distinguished from any edit of its
            # size — measured, 9 of 43 rows on the reference prompt, two of them
            # (22852, 1288) in the bar chart's plotted top-8.
            "control_kl": round(ctrl, 8),
            "clears_control": kl > ctrl,
            # Per row, because it varies from 0% to 60.3% and a single verdict
            # taken on the top row would be a green tick 38 rows contradict.
            "encoder_residual": round(
                encoder_residual(as_received(edited), f, positions), 4
            ),
            "p_top_before": round(float(base[top_id]), 5),
            "p_top_after": round(float(after[top_id]), 5),
            "flips_top": int(after.argmax()) != top_id,
            "below_resolution": abs(kl) < RESOLUTION_KL,
        }

    ranked = [row_of(f, ps) for f, ps in tested]

    # Sum against joint over exactly the rows returned, so the comparison is
    # between numbers the panel shows. The direction is the finding: features
    # UNDER-count.
    joint = kl_nats(base, run(edit(tested)))

    # What the SAE does not model, in nats: substitute the reconstruction for
    # the true stream with NO feature removed. OVER THE WINDOW THE EDITS USE,
    # which at prompt scope is not one token — measured on the reference
    # prompt, at position 10 alone this costs 0.077530 and over positions 0-10
    # it costs 0.221217, 2.85x more, and the second is what a prompt-scope
    # ranking has to be read against. Below this line a feature score is
    # smaller than what the decomposition gets wrong while it is being taken:
    # 2 of 43 clear it at position scope, 1 of 256 at prompt scope.
    window = sorted({p for _, ps in tested for p in ps})
    x_recon = x.clone()
    for p in window:
        x_recon[p] = recon[p] + mu[p]
    residual_kl = kl_nats(base, run(x_recon))

    ranked.sort(key=lambda r: -r["kl"])

    # ---- THE MECHANISM CHECK: did the edit land, exactly ------------------
    # One pass, and it asks the one thing that IS a property of the edit rather
    # than of each feature: does the stream the model received differ from
    # `x - act*W_dec[f]` by anything at all? It reads the tensor captured
    # through the same hook DURING an edited pass, after the cast to the
    # model's dtype and device, so a dtype that rounded the edit away shows up
    # here. Measured exactly 0.0 in float32 on gpt2 blocks.8.hook_resid_pre.
    #
    # What it deliberately no longer claims is that the feature has left the
    # SAE's reading of the stream. That is `encoder_residual`, it is per row,
    # and on this SAE it fails on 38 of 43 rows — a single probe row was
    # reporting a property 38 rows contradict.
    probe_f = int(ranked[0]["feature_id"])
    probe_ps = list(ranked[0]["positions"])

    seen: list[torch.Tensor] = []
    intended = edit([(probe_f, probe_ps)])
    xd = intended.to(device=captured.device, dtype=captured.dtype).unsqueeze(0)
    h_edit = _register_edit(block, sae.point, xd)
    # Registered second, so it sees the edited args: torch passes each hook's
    # return value on to the next one.
    h_see = _register_capture(block, sae.point, seen)
    try:
        with torch.no_grad():
            model(ids)
    finally:
        h_see.remove()
        h_edit.remove()
    passes += 1
    received = seen[0][0].detach().to("cpu").float()
    edit_deviation = float((received - intended).abs().max())
    removal_verified = edit_deviation <= EDIT_TOLERANCE

    n_tested = len(tested)
    n_encoder_residual = sum(
        1 for r in ranked if r["encoder_residual"] > ENCODER_RESIDUAL_TOLERANCE
    )
    worst_encoder = max((r["encoder_residual"] for r in ranked), default=0.0)
    n_clearing_control = sum(1 for r in ranked if r["clears_control"])
    at_pos_err = float(err[position].norm())
    at_pos_norm = float((x - mu)[position].norm())
    # The WORST position in the window, not the attributed one: at prompt scope
    # the edits land at token 3 too, where the SAE misses 42.5% of the norm
    # against 20.4% here.
    shares = [
        float(err[p].norm()) / float((x - mu)[p].norm())
        for p in window
        if float((x - mu)[p].norm()) > 0
    ]
    worst_share = max(shares) if shares else None
    at_pos_share = at_pos_err / at_pos_norm if at_pos_norm else None
    # `position` is in `window` at position scope by construction and at prompt
    # scope only if some tested feature fires there, which is the usual case
    # and not a guarantee. Said rather than assumed.
    here = (
        f"; at the attributed token itself {at_pos_share:.1%} "
        f"({at_pos_err:.2f} of {at_pos_norm:.2f})"
        if at_pos_share is not None
        else ""
    )
    return {
        "intervention": intervention,
        "scope": scope,
        "position": position,
        "target_token": decode(top_id) if decode else str(top_id),
        "hook": sae.hook,
        "layer": sae.layer,
        # Named so the caller can grey out what is indistinguishable from
        # arithmetic. The floor and the resolution are different numbers and
        # both are here: greying out at the floor greys out nothing.
        "noise_floor_kl": round(floor, 8),
        "replay_kl": round(replay, 8),
        "resolution_kl": RESOLUTION_KL,
        "passes": passes,
        "elapsed_s": round(time.perf_counter() - started, 2),
        "ranked": ranked,
        "n_tested": n_tested,
        "n_candidates": n_candidates,
        "truncated": n_tested < n_candidates,
        "coverage": (
            f"{n_tested} of {n_candidates} firing features were tested, chosen "
            "by peak activation; one not listed was NOT TESTED, not found "
            "unimportant. Features whose activation is zero are not candidates "
            "at all — removing nothing is the identity."
        ),
        "sum_of_singles": round(sum(r["kl"] for r in ranked), 6),
        "joint_kl": round(joint, 6),
        # The SAE's own calibration, and the reason it must not be read as a
        # per-token accuracy.
        "convention": cal.convention,
        "fvu": cal.fvu,
        # The aggregate in the SAME units as residual_share, so a caller can
        # make the comparison without squaring one side of it. Measured
        # 0.029397 against a per-position 0.203571 — 7x, where fvu against
        # residual_share reads 200x for no reason but the units.
        "rel_err": cal.rel_err,
        # The WORST share over the window the edits landed in, which is the
        # window this ranking has to be read against. `residual_share` at
        # prompt scope is therefore not the attributed token's — that one is
        # kept beside it rather than replaced.
        "residual_share": round(worst_share, 6) if worst_share is not None else None,
        "residual_share_at_position": (
            round(at_pos_share, 6) if at_pos_share is not None else None
        ),
        "residual_window": [window[0], window[-1]],
        "residual_kl": round(residual_kl, 6),
        "residual_means": (
            f"Substituting the SAE's reconstruction for the true stream over "
            f"the {len(window)} token(s) these edits land in, with NO feature "
            f"removed, already moves the answer {residual_kl:.4f} nats. Across "
            f"that window the SAE fails to model up to {worst_share:.1%} of a "
            f"token's norm{here}. "
            f"The calibration's fvu {cal.fvu:g} is an aggregate over every "
            f"token, is dominated by token 0 whose stream norm is ~30x the "
            f"rest, and is a SQUARED-error fraction — the like-for-like "
            f"aggregate against these shares is rel_err {cal.rel_err:g}. A "
            "feature scoring below residual_kl is smaller than what the "
            "decomposition gets wrong while the score is being taken."
        )
        if worst_share is not None
        else (
            "The stream has zero norm over the window these edits land in, so "
            "there is no share of it the SAE can be said to miss."
        ),
        # Now exactly one claim: the stream the model received IS the intended
        # edit. Whether the SAE still reads the feature afterwards is per row.
        "removal_verified": removal_verified,
        "edit_deviation": edit_deviation,
        "removal_check": (
            f"The edit landed: the stream the model received differs from "
            f"`x - activation x W_dec[{probe_f}]` by at most {edit_deviation:g} "
            f"(tolerance {EDIT_TOLERANCE:g}), measured on the top row through "
            f"the same hook during an edited pass, after the cast to the "
            f"model's dtype. Measured exactly 0.0 on gpt2 "
            f"blocks.8.hook_resid_pre in float32. This does NOT mean the "
            f"feature left the SAE's reading of the stream: encoder and decoder "
            f"directions are not dual, so re-encoding afterwards still reports "
            f"some of it — {n_encoder_residual} of {n_tested} rows here read "
            f"above {ENCODER_RESIDUAL_TOLERANCE:.0%} of their original "
            f"activation, the worst at {worst_encoder:.1%}. That is per row, in "
            f"`encoder_residual`, because it varies (0% to 60.3% on gpt2 "
            f"blocks.8.hook_resid_pre) and one row cannot speak for the rest."
        ),
        "n_encoder_residual": n_encoder_residual,
        "encoder_residual_max": round(worst_encoder, 4),
        "n_clearing_control": n_clearing_control,
        "control_means": (
            f"Every score above is paired with `control_kl`: the same tokens "
            f"edited by a random Gaussian direction of the same norm, seed "
            f"{CONTROL_SEED}. It is not zero — on gpt2 blocks.8.hook_resid_pre "
            f"a random direction at the top feature's norm of 35.5 costs about "
            f"0.09 nats, against that feature's own 0.417 — so part of a score "
            f"is the SIZE of the edit rather than the identity of the feature. "
            f"{n_clearing_control} of {n_tested} rows here score above their own "
            f"control. One draw per row: five draws at 35.5 spanned 0.0666 to "
            f"0.1093, a factor of 1.6, so a row within that factor of its "
            f"control is not separated by this test."
        ),
        "means": (
            "KL divergence in nats of the next-token distribution at this "
            "position when this feature's own contribution "
            f"(activation x W_dec) is subtracted from the residual stream at "
            f"{sae.hook}. Larger = removing it alone moves the answer more. "
            "These are NOT each feature's share of the prediction and they do "
            "not add up: measured on gpt2 with the prompt 'The Eiffel Tower is "
            "located in the city of' at the last prompt token, the 43 "
            "single-feature scores sum to 0.66446 while removing all 43 at "
            "once gives 2.135221, so the singles UNDER-count by 3.2x. Head "
            "ablation misses in the other direction, and that panel's wording "
            "does not transfer. sum_of_singles and joint_kl are both here so "
            "the gap is visible; neither is a correction factor. This is also "
            "an intervention on the SAE's MODEL of the stream, not on the "
            "model's own units — see residual_means for what that model misses "
            "here."
        ),
    }
