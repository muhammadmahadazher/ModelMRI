# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""The smallest edit to your words that makes the model say something ELSE.

This package could already ask two questions about a prompt, and they are not
this one.

  `attribute.rank_tokens`  NECESSITY. Mask a token out; how far did the answer
                           move? "Was this word needed?"
  `anchors.find_anchor`    SUFFICIENCY. Keep only these tokens and perturb the
                           rest; does the prediction hold? "Do these words
                           carry it on their own?"

Both are about the answer the model already gives. Neither can tell you what to
write instead. A counterfactual is the third question — REACHABILITY — and it
is directional: *what is the smallest substitution into this prompt that makes
the next token become the one I NAME?*

The difference is not academic. Masking "Eiffel" tells you the answer leaned on
it. Replacing "Eiffel" with a word that makes the model say " Rome" tells you
the Paris/Rome axis runs through that one position, and hands you the sentence
that does it.

WHY THIS FILE IS THE FRONT END TO `patch_graph.py`

`patch.py` and `patch_graph.py` both take a PAIR — a clean prompt and a corrupt
one — and every number they publish is a difference between the two. Until now
the corrupt prompt was the caller's problem, written by hand, and a badly
chosen one does not fail loudly. It shifts the numbers.

This repository has a measured example of exactly that, and it is the reason
this module exists rather than a hypothetical. Two prompt pairs in this project
were both called "the reference pair" and differ only in "is in" versus "is
located in". They resolve differently — 0.007571 against 0.006231 — and because
`recovery_resolution` sets the prune threshold, which sets how much of the
patching graph survives, the same depth-2 run costs a different number of
forward passes on each. A figure measured on one was published against the
other for weeks. Nothing crashed.

So a counterfactual here is not only an explanation. `edited_ids` in the
payload is a corrupt prompt that was SEARCHED FOR against a named target and
CONTROLLED, rather than typed, and it can be handed straight to `patch.py`.

A FLIPPED ANSWER IS NOT YET A FINDING

Edit enough of a prompt and any model says something else. The claim "this edit
made it say Rome" is worth nothing until the alternatives have been measured,
so every counterfactual this module returns is scored against two controls, and
they answer different questions:

  `same_positions`  the same indices, donors drawn at random. If random words
                    at those positions also reach the target, the POSITIONS
                    carry the flip and the particular words do not. The finding
                    is about where, not what.
  `any_positions`   as many indices as the edit used, drawn at random, donors
                    drawn at random. If this reaches the target too, the model
                    is simply fragile to edits of that size and the search
                    learned nothing about this prompt.

Both come back as a count out of a sample with a Wilson interval, never as a
bare rate, for the reason `anchors.py` sets out at length: 0 of 24 and 0 of 240
are both "0%" and they are not the same evidence. A counterfactual that does
not beat both controls is still returned — with `beats_controls` False and the
counts beside it — because hiding it would leave the reader with a flipped
answer and no way to know it was a coin toss.

THE EDIT VOCABULARY IS PART OF THE CLAIM

Replacement ids come from `anchors.donor_pool` — this project's bundled donor
corpus, frequency-weighted, the same distribution `anchors.py` perturbs with
and `ablate.py` resamples heads from. There is no list of "interesting" words
in this file and there must never be one: a search that can only substitute
words somebody chose in advance is measuring that choice. Sharing the pool with
`anchors.py` is also what lets a sufficiency result and a reachability result
be read on one screen without a footnote about which words each was allowed.

The search is greedy over (position, donor) and `minimality.smaller_may_exist`
is True in every payload this function produces. It means it: greedy forward
selection finds *a* small edit, never a proof that none is smaller.

WHAT IT REFUSES

Naming a target the model already predicts is refused rather than answered with
an empty edit. So is a target outside the vocabulary, a prompt with nothing
perturbable ahead of the queried position, and a model whose own argmax moves
between two identical forward passes — that last one on `anchors.py`'s
reasoning, because every comparison below is an argmax against the base and a
model that cannot reproduce its own top token has nothing to be moved away
from.

When the budget is spent without the target ever becoming the argmax, the
payload says so — `found` is False, `stopped_because` names which bound was
hit, and `best_effort` carries the closest the search came WITHOUT calling it a
counterfactual. A near miss is a real thing to report and it is not the thing
that was asked for.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from typing import Any

import torch

from .ablate import distribution, kl_nats
from .anchors import donor_pool, wilson_interval
from .errors import BadRequest, Refusal

__all__ = [
    "CounterfactualError",
    "MAX_EDITS",
    "N_CONTROLS",
    "N_DONORS",
    "N_PROPOSALS",
    "CANDIDATE_SOURCES",
    "cost_passes",
    "donor_pool",
    "estimate_cost",
    "find_counterfactual",
    "is_finding",
    "propose_by_gradient",
]


# How many positions the edit may touch. Greedy adds one per step, so this is
# also the step ceiling. Three is not a tuned number and is not claimed to be:
# it is the point past which "smallest edit" stops being a useful description
# of the result, since an edit touching four of a ten-token prompt is a rewrite.
MAX_EDITS = 3

# Donors screened per position per step. Drawn from the pool, seeded, and NOT
# deduped — the pool is a frequency distribution and taking unique types from it
# would silently change the measurement into uniform-over-types.
N_DONORS = 8

# Control draws per arm. 24 gives a Wilson upper bound of 0.1370 at 0 successes
# and 95% confidence, which is the strongest "this does not happen by chance"
# a sample of that size can support; the payload publishes the bound so nobody
# has to take 0/24 for proof of zero.
N_CONTROLS = 24

# (position, token) pairs the gradient screen proposes per step. Every one of
# them is then MEASURED with a real forward pass, so this is the width of the
# shortlist, not a number of results.
N_PROPOSALS = 24

# Where the substitutions a step tries come from. `gradient` is the default
# because a corpus pool cannot reach a target its vocabulary does not contain —
# see `propose_by_gradient` for the measurement that settled it.
CANDIDATE_SOURCES = ("gradient", "pool")

# How many times a control draw may resample before the sample is abandoned.
# A control must not substitute a token for itself — see `control_edit` — and
# a frequency-weighted pool can offer the same common token repeatedly. Bounded
# so a degenerate pool ends the draw instead of the process.
MAX_CONTROL_REDRAWS = 16

# Rows of the embedding table cast to fp32 at once inside the gradient
# screen. Bounds the screen's peak allocation independently of vocabulary
# size — see `propose_by_gradient`.
VOCAB_CHUNK = 16384

CONFIDENCE = 0.95

# base, a repeat of it for the noise floor, and one plain model(ids) for the
# agreement check. Identical to `anchors.py`'s triple and for its reasons.
FIXED_PASSES = 3


class CounterfactualError(Refusal):
    """A counterfactual search that cannot be run, or cannot be believed."""


def _reject_bools(pairs: Iterable[tuple[str, Any]]) -> None:
    """`isinstance(True, int)` is True, so counts must reject bools by name.

    `position=True` would otherwise quietly query index 1 and return a payload
    about a token the caller never asked about.
    """
    for name, value in pairs:
        if isinstance(value, bool):
            raise BadRequest(f"{name} is a count, not a boolean.")


def _interval(successes: int, samples: int, *, confidence: float) -> dict:
    """A count, its rate, and the Wilson interval — never a bare float.

    `samples` of 0 is an ABSENCE, not a rate of zero: `measured` is False and
    `point` is None. See `anchors.py` on why unknown must not render as 0.00.
    """
    if samples <= 0:
        return {
            "measured": False,
            "successes": 0,
            "samples": 0,
            "point": None,
            "interval": None,
            "confidence": confidence,
        }
    low, high = wilson_interval(successes, samples, confidence=confidence)
    return {
        "measured": True,
        "successes": successes,
        "samples": samples,
        "point": round(successes / samples, 5),
        "interval": [round(low, 5), round(high, 5)],
        "confidence": confidence,
        "method": "Wilson score interval",
    }


def is_finding(same_positions: dict, any_positions: dict) -> bool:
    """Whether a found edit survived both controls — the one call that decides.

    Both arms must have been MEASURED and both must have come back empty. One
    success in either is enough to say a random edit of the same size reaches
    the target too, and then the search isolated nothing.

    The guard that earns this its own function is `measured`. An arm whose every
    draw was abandoned — a donor pool that could offer some position nothing but
    the token already there — carries `successes: 0` and `samples: 0`, and a
    rule written as "no control reached the target" reads that as the STRONGEST
    possible evidence for the finding. It is the absence of evidence. Zero out
    of zero is not zero, and this is the exact shape of the `?? 0` bug this
    project has now fixed in nine call sites: unknown must never render as a
    measurement, least of all as a confirming one.

    Kept out of `find_counterfactual` so it can be checked directly against
    intervals a real run is unlikely to produce on demand.
    """
    for arm in (same_positions, any_positions):
        if not arm.get("measured"):
            return False
        if arm.get("successes") != 0:
            return False
    return True


def propose_by_gradient(
    model: Any,
    ids: torch.Tensor,
    *,
    position: int,
    target_token_id: int,
    indices: Sequence[int],
    k: int,
    fixed: dict[int, int] | None = None,
) -> list[dict]:
    """First-order candidates: which single substitution most raises the target.

    The gradient of the target's logit with respect to the input embedding at
    index i says which DIRECTION in embedding space raises it. A substitution
    moves that position from `E[current]` to `E[v]`, so the first-order estimate
    of the change is

        (E[v] - E[current]) . d logit_target / d e_i

    which is one matrix-vector product per position and needs no forward pass
    per candidate. This is HotFlip (Ebrahimi et al., 2018), and naming it
    matters: it is a linear approximation to a discrete, highly non-linear
    change, and the literature it comes from is where its failure modes are
    written down.

    THE ESTIMATE PROPOSES AND THE FORWARD PASS DECIDES. Nothing here is
    reported as a result. `find_counterfactual` runs a real pass on every pair
    this returns and keeps only what actually moved the argmax — the same
    posture `patch_screen.py` takes toward attribution patching, and for the
    same reason: a first-order screen is a cheap way to choose what to measure
    and a terrible way to decide what is true. The payload publishes how often
    the ranking agreed with the measurement, so the screen's quality is visible
    rather than assumed.

    WHY IT EXISTS AT ALL, MEASURED

    The donor pool is a PERTURBATION distribution — the bundled corpus, 16
    sentences, 137 distinct ids. Measured on Qwen3-1.7B, "The Eiffel Tower is in
    the city of" against a target of " Rome": the pool search spent its whole
    budget of 3 edits, reached a target probability of 0.01366, and correctly
    returned `found: False`. It could not have succeeded — no word in a generic
    corpus steers a model to Rome, so the search was drawing from a vocabulary
    that does not contain an answer. A pool is the right null for a control and
    the wrong instrument for a search, and this function is the difference.

    `fixed` is the edit already committed, applied before the gradient is taken,
    so step two proposes against the prompt step one produced rather than
    against the original.

    Returns at most `k` dicts, best estimate first. Fewer than `k` only when the
    candidate space is smaller than `k`.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise BadRequest(f"k must be a positive integer, got {k!r}.")
    getter = getattr(model, "get_input_embeddings", None)
    embed = getter() if callable(getter) else None
    if embed is None or getattr(embed, "weight", None) is None:
        raise CounterfactualError(
            "this model does not expose an input embedding table, so there is "
            "no continuous space to take a substitution gradient in. Run with "
            "candidates='pool' instead — a corpus search needs no embeddings — "
            "and expect it to reach only targets the pool's vocabulary can "
            "spell."
        )

    row = ids.clone()
    for index, token_id in (fixed or {}).items():
        row[0, index] = int(token_id)

    weight = embed.weight
    vocab = int(weight.shape[0])
    if not 0 <= int(target_token_id) < vocab:
        raise BadRequest(
            f"target_token_id {target_token_id} is outside the embedding "
            f"table's {vocab} rows."
        )

    point = embed(row).detach().clone().requires_grad_(True)
    out = model(inputs_embeds=point)
    scalar = out.logits[0, position, int(target_token_id)]
    (grad,) = torch.autograd.grad(scalar, point)

    # One position at a time, keeping only that position's top-k. The whole
    # [positions, vocab] score matrix is never formed: on Qwen3-1.7B that is
    # 151,936 floats per position, and a long prompt would turn a screen into
    # the largest allocation in the run.
    table = weight.detach()
    found: list[dict] = []
    for index in indices:
        g = grad[0, index].detach().to(torch.float32)
        current = int(row[0, index])
        # Chunked over the vocabulary, and cast to fp32 a chunk at a time.
        # `table.to(torch.float32)` would be a second copy of the embedding
        # matrix: 151,936 x 2,048 x 4 bytes = 1.24 GB on Qwen3-1.7B, allocated
        # beside a model that already fills half of an 8 GB card. The chunk
        # peaks at VOCAB_CHUNK x hidden instead, and the arithmetic is
        # identical because a dot product does not span rows.
        scores = torch.empty(vocab, dtype=torch.float32, device=g.device)
        for start in range(0, vocab, VOCAB_CHUNK):
            stop = min(start + VOCAB_CHUNK, vocab)
            scores[start:stop] = table[start:stop].to(torch.float32) @ g
        scores = scores - scores[current]
        # A token cannot be replaced by itself, and the estimate for doing so is
        # exactly zero, which would otherwise sit in the middle of the ranking.
        scores[current] = float("-inf")
        top = torch.topk(scores, min(k, vocab))
        for value, token_id in zip(
            top.values.tolist(), top.indices.tolist(), strict=True
        ):
            found.append(
                {
                    "index": index,
                    "token_id": int(token_id),
                    "from_token_id": current,
                    "estimate": float(value),
                }
            )
        del scores, top
    del table, grad, point

    found.sort(key=lambda row_: -row_["estimate"])
    return found[:k]


def cost_passes(
    *,
    n_positions: int,
    steps: int,
    n_donors: int = N_DONORS,
    control_passes: int,
    trials_not_run: int = 0,
    proposals_per_step: int | None = None,
    proposer_passes: int = 0,
) -> int:
    """Exactly how many forward passes a run of this shape takes.

    Exact, not an estimate: `find_counterfactual` counts its own passes and the
    tests assert the two agree to the pass. The parts, in the order spent:

        3                              base, its repeat for the floor, and one
                                       plain model(ids) for the agreement check
        n_donors * sum(P - i)          screening, step i having P - i positions
                                       left and every one of them tried against
                                       every donor
        - trials_not_run               the trials that scan reached but never
                                       paid for (below)
        control_passes                 the two control arms, one pass per draw
                                       actually taken

    `steps` is not known before the run — greedy stops the moment the target
    becomes the argmax — so `estimate_cost` prices the shortest and the longest
    run and says which is which, rather than averaging them into a number that
    describes neither.

    `trials_not_run` is what makes the word "exact" above true, and leaving it
    out is what made it false. A step's scan does not always pay for every
    (position, donor) it could: a donor that is ALREADY the token at that
    position is a no-op that is skipped without a pass, and the whole scan
    short-circuits the moment one trial makes the target the argmax, so every
    pair after it in that step is never reached. Both counts are published in
    the payload as `trials_skipped_self` and `trials_short_circuited`, so a
    reader can redo this arithmetic rather than take the total on faith.

    `control_passes` is 0 for a run that found nothing: there is no edit to draw
    same-size alternatives against, so those passes are never spent. It is also
    below `2 * n_controls` when a draw had to be abandoned — see
    `control_draws_abandoned` in the payload — because a sample that was never
    taken must not be priced or counted.
    """
    _reject_bools(
        (
            ("n_positions", n_positions),
            ("steps", steps),
            ("n_donors", n_donors),
            ("control_passes", control_passes),
        )
    )
    for name, value in (
        ("n_positions", n_positions),
        ("steps", steps),
        ("n_donors", n_donors),
        ("control_passes", control_passes),
    ):
        if not isinstance(value, int) or value < 0:
            raise BadRequest(f"{name} must be a non-negative integer, got {value!r}.")
    if steps > n_positions:
        raise BadRequest(
            f"a greedy search over {n_positions} positions cannot take {steps} "
            "steps — each step consumes one position."
        )
    if proposals_per_step is None:
        # Pool mode: every remaining position is tried against every donor.
        scanned = sum(n_positions - i for i in range(steps)) * n_donors
    else:
        # Gradient mode: the proposer returns a fixed number of (position,
        # token) PAIRS per step, already spread across positions, so the scan
        # does not grow with the window.
        if not isinstance(proposals_per_step, int) or isinstance(
            proposals_per_step, bool
        ):
            raise BadRequest("proposals_per_step is a count, not a boolean.")
        if proposals_per_step < 1:
            raise BadRequest(
                f"proposals_per_step must be at least 1, got "
                f"{proposals_per_step}. A step offered no candidates cannot "
                "change anything and would spend a step doing it."
            )
        scanned = steps * proposals_per_step
    if not isinstance(trials_not_run, int) or isinstance(trials_not_run, bool):
        raise BadRequest("trials_not_run is a count, not a boolean.")
    if not 0 <= trials_not_run <= scanned:
        raise BadRequest(
            f"trials_not_run must be in [0, {scanned}] for a run of this "
            f"shape, got {trials_not_run}. More trials skipped than the scan "
            "contains is a caller describing a run that cannot happen."
        )
    if not isinstance(proposer_passes, int) or isinstance(proposer_passes, bool):
        raise BadRequest("proposer_passes is a count, not a boolean.")
    return FIXED_PASSES + (scanned - trials_not_run) + control_passes + proposer_passes


def estimate_cost(
    *,
    n_positions: int,
    max_edits: int = MAX_EDITS,
    n_donors: int = N_DONORS,
    n_controls: int = N_CONTROLS,
) -> dict:
    """What the run will cost, priced at both ends rather than averaged.

    The shortest run is one step that reaches the target and is then controlled.
    The longest is every permitted step spent without reaching it, which is also
    the run that spends NO control passes — so the longest search is not
    automatically the most expensive, and both numbers are published rather than
    collapsed into one.

    Every figure here is a CEILING, priced at `trials_not_run=0`. A real run
    comes in at or under it: a step stops scanning the moment it reaches the
    target, and a donor that is already the token at a position is skipped
    without a pass. Neither is knowable in advance — that is why they are not
    estimated here — so this prices the run that skips nothing and the payload
    reports what the run actually spent beside it.
    """
    _reject_bools((("max_edits", max_edits),))
    if max_edits < 1:
        raise BadRequest(f"max_edits must be at least 1, got {max_edits}.")
    steps_max = min(max_edits, n_positions)
    if steps_max < 1:
        raise BadRequest(
            "there are no perturbable positions ahead of the queried one, so "
            "there is no edit to price."
        )
    shortest = cost_passes(
        n_positions=n_positions,
        steps=1,
        n_donors=n_donors,
        control_passes=2 * n_controls,
    )
    longest = cost_passes(
        n_positions=n_positions,
        steps=steps_max,
        n_donors=n_donors,
        control_passes=0,
    )
    found_at_max = cost_passes(
        n_positions=n_positions,
        steps=steps_max,
        n_donors=n_donors,
        control_passes=2 * n_controls,
    )
    return {
        "shortest": shortest,
        "shortest_is": "one step reaches the target, then both controls run",
        "longest_search": longest,
        "longest_search_is": (
            f"all {steps_max} permitted steps spent without reaching the "
            "target, so no control passes are spent"
        ),
        "most_expensive": max(shortest, longest, found_at_max),
        "most_expensive_is": (
            "the target reached only on the final permitted step, which pays "
            "for the whole search AND both controls"
        ),
        "n_positions": n_positions,
        "max_edits": steps_max,
        "n_donors": n_donors,
        "n_controls": n_controls,
    }


def _token(tokenizer_decode, token_id: int) -> str:
    """Decode one id, on `anchors.py`'s contract: `decode(id)`, not
    `decode([id])`.

    Every caller in `runtime.py` passes `lambda t: self.tokenizer.decode([t])`,
    which already does the wrapping. Passing a list to it produced
    `decode([[id]])` and every token in the payload came back as a one-element
    LIST — `["The"]` where a string belonged, which a renderer prints as
    `The` with brackets or, in a template literal, as the whole array. Measured
    through the live route before this was fixed. The fallback is `str(id)`
    rather than None for the same reason `anchors._token_id` uses it: a bare id
    is still information, where a null is a second case every renderer has to
    handle.
    """
    if tokenizer_decode is None:
        return str(int(token_id))
    try:
        return tokenizer_decode(int(token_id))
    except Exception:
        return str(int(token_id))


def _check_inputs(
    ids: torch.Tensor,
    *,
    position: int,
    target_token_id: int,
    pool: Sequence[int],
    max_edits: int,
    n_donors: int,
    n_controls: int,
) -> int:
    """Shape, range and type checks, all of them before any pass is spent."""
    _reject_bools(
        (
            ("position", position),
            ("target_token_id", target_token_id),
            ("max_edits", max_edits),
            ("n_donors", n_donors),
            ("n_controls", n_controls),
        )
    )
    if not isinstance(ids, torch.Tensor):
        raise BadRequest(f"ids must be a torch.Tensor, got {type(ids).__name__}.")
    if ids.dim() != 2 or ids.shape[0] != 1:
        raise BadRequest(
            f"ids must be one unbatched sequence shaped [1, S], got "
            f"{tuple(ids.shape)}. A batch would make every count below a sum "
            "over sequences that were never named."
        )
    seq = int(ids.shape[1])
    if not 0 <= position < seq:
        raise BadRequest(f"position {position} is outside a sequence of {seq} tokens.")
    if max_edits < 1:
        raise BadRequest(f"max_edits must be at least 1, got {max_edits}.")
    if n_donors < 1:
        raise BadRequest(
            f"n_donors must be at least 1, got {n_donors}. A step that tries no "
            "donors cannot change anything and would spend a step doing it."
        )
    if n_controls < 0:
        raise BadRequest(f"n_controls must be non-negative, got {n_controls}.")
    if len(pool) == 0:
        raise BadRequest(
            "the donor pool is empty, so there is nothing to substitute IN. "
            "Build one with `donor_pool(tokenizer)`."
        )
    return seq


def find_counterfactual(
    model: Any,
    ids: torch.Tensor,
    *,
    position: int,
    target_token_id: int,
    pool: Sequence[int],
    perturbation: dict | None = None,
    control_ids: Iterable[int] = (),
    typed_span: tuple[int, int] | None = None,
    n_prompt: int | None = None,
    max_edits: int = MAX_EDITS,
    candidates: str = "gradient",
    n_donors: int = N_DONORS,
    n_proposals: int = N_PROPOSALS,
    n_controls: int = N_CONTROLS,
    confidence: float = CONFIDENCE,
    seed: int = 0,
    decode=None,
) -> dict:
    """The smallest edit found that makes `target_token_id` the next token.

    `ids` is one unbatched sequence `[1, S]`; `position` is the index whose
    next-token argmax is being moved. `pool` is the replacement ids and
    `perturbation` is the description `donor_pool` returned beside them — pass
    it, or the payload cannot say what the words were replaced WITH, and an
    edit whose vocabulary is unnamed can neither be interpreted nor reproduced.

    `typed_span` and `n_prompt` carry the same meaning and the same warning as
    in `attribute.rank_tokens`: `None` for the span is "the caller could not
    locate your words", not "all of it is yours".

    Returns a dict. `edited_ids` is the answer — a corrupt prompt suitable for
    `patch.py` — and everything else is the evidence for it.
    """
    seq = _check_inputs(
        ids,
        position=position,
        target_token_id=target_token_id,
        pool=pool,
        max_edits=max_edits,
        n_donors=n_donors,
        n_controls=n_controls,
    )
    if not 0.0 < confidence < 1.0:
        raise BadRequest(
            f"confidence must be in (0, 1), got {confidence}. It is the level "
            "the published intervals are computed at."
        )

    if candidates not in CANDIDATE_SOURCES:
        raise BadRequest(
            f"candidates must be one of {CANDIDATE_SOURCES}, got "
            f"{candidates!r}. 'gradient' screens substitutions with a "
            "first-order estimate and measures every one it shortlists; "
            "'pool' searches the donor corpus, which is the right null for a "
            "control and reaches only targets that corpus can spell."
        )
    use_gradient = candidates == "gradient"

    started = time.perf_counter()
    device = ids.device
    control = {int(c) for c in control_ids}

    # THE CANDIDATE SET. Deliberately identical to `attribute.rank_tokens`'s and
    # `anchors.find_anchor`'s: index 0 is an attention sink whose score follows
    # the position rather than the token, `position` is its own query, and
    # anything after `position` cannot reach it through a causal mask. Control
    # tokens are held because editing the chat template measures the template.
    # The sameness is the point — three questions over one candidate set can be
    # read on one screen; over three sets they cannot.
    perturbable = [i for i in range(1, position) if int(ids[0, i]) not in control]
    if not perturbable:
        raise CounterfactualError(
            f"there is nothing editable before position {position}: every "
            "earlier index is either the attention sink at 0 or a control "
            "token being held fixed. An edit has to change a token the query "
            "can actually see, so this prompt has no counterfactual to find. "
            "Query a later position, or pass fewer control_ids."
        )

    ones = torch.ones((1, seq), dtype=torch.long, device=device)
    position_ids = torch.arange(seq, dtype=torch.long, device=device).unsqueeze(0)
    passes = 0

    def forward_probs(row: torch.Tensor) -> torch.Tensor:
        nonlocal passes
        passes += 1
        if tuple(row.shape) != (1, seq):
            raise RuntimeError(
                f"an edited sequence came out shaped {tuple(row.shape)} rather "
                f"than {(1, seq)}; substitution must preserve every index, or "
                "the result is about the renumbering."
            )
        out = model(input_ids=row, attention_mask=ones, position_ids=position_ids)
        return distribution(out.logits[0, position])

    def edited(assignment: dict[int, int]) -> torch.Tensor:
        """`ids` with `assignment` applied. One clone, freed by the caller."""
        row = ids.clone()
        for index, token_id in assignment.items():
            row[0, index] = int(token_id)
        return row

    with torch.no_grad():
        base = forward_probs(ids)
        vocab = int(base.shape[-1])
        if not 0 <= target_token_id < vocab:
            raise BadRequest(
                f"target_token_id {target_token_id} is outside this model's "
                f"vocabulary of {vocab}. Nothing can be steered toward a token "
                "the model cannot emit."
            )
        base_top = int(base.argmax())
        if base_top == int(target_token_id):
            raise BadRequest(
                f"the model already predicts token {target_token_id} at "
                f"position {position}, so the smallest edit that reaches it is "
                "the empty edit. Name a token it does NOT currently predict — "
                "a counterfactual is a change of answer, not a confirmation "
                "of one."
            )

        # The floor, on ablate.py's pattern: the same pass twice. Not decoration
        # — every comparison below is an argmax against `base_top`, so an argmax
        # that is not stable against a repeat of its own pass has nothing to be
        # moved away from.
        repeat = forward_probs(ids)
        floor = kl_nats(base, repeat)
        if int(repeat.argmax()) != base_top:
            raise CounterfactualError(
                "this model's own top token at that position changed between "
                "two identical forward passes, so 'the edit changed the answer' "
                "has nothing stable to change. Every comparison below would be "
                "measuring non-determinism. Try a shorter sequence or a "
                "deterministic kernel."
            )

        # The plain call every other reader of this model makes. Substitution
        # cannot re-phase RoPE the way masking can — the mask stays all-ones and
        # the length never changes — but "our explicit arguments select the same
        # computation" is a claim, and it is one pass to check.
        passes += 1
        plain = distribution(model(ids).logits[0, position])
        agreement = kl_nats(plain, base)
        if int(plain.argmax()) != base_top:
            raise CounterfactualError(
                "supplying an all-ones attention mask and explicit position_ids "
                f"changes this model's top token at position {position} (it "
                f"moved the distribution by {agreement:.3e} nats). Every edit "
                "below is measured under those arguments, so the search would "
                "be explaining a prediction the plain call does not make."
            )

        base_target_p = float(base[int(target_token_id)])
        base_target_rank = int((base > base[int(target_token_id)]).sum()) + 1

        # Donors are drawn ONCE per step from the pool, seeded, so a rerun with
        # the same seed screens the same words. Drawing per position instead
        # would make "this position won" partly a statement about which words
        # that position happened to be offered.
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        pool_tensor = torch.tensor([int(p) for p in pool], dtype=torch.long)

        def draw(n: int) -> list[int]:
            picks = torch.randint(
                low=0, high=int(pool_tensor.numel()), size=(n,), generator=generator
            )
            return [int(pool_tensor[i]) for i in picks.tolist()]

        assignment: dict[int, int] = {}
        remaining = list(perturbable)
        steps_taken = 0
        found = False
        stopped = ""
        trail: list[dict] = []
        best_effort: dict | None = None
        # The two ways the scan reaches a (position, donor) pair and never pays
        # for it. Both are published, because they are what makes `passes` and
        # `passes_expected` agree, and an unexplained gap between those two is
        # indistinguishable from a miscount.
        trials_skipped_self = 0
        trials_short_circuited = 0
        # Pairs the shape says a step contains that the proposer could not
        # offer. Named separately from the two above because it is a statement
        # about the candidate space, not about the scan.
        trials_unavailable = 0
        proposer_passes = 0
        steps_scanned = 0
        # p(target) under the edit committed so far; the bar each step has to
        # clear. Starts at the unedited prompt's own value.
        current_p = base_target_p
        screen_agreed = 0
        screen_steps = 0

        limit = min(max_edits, len(perturbable))
        for _ in range(limit):
            steps_scanned += 1
            if use_gradient:
                # The one place a gradient is taken. Outside `no_grad`, and
                # only for the proposal — every number below still comes from
                # a forward pass on a real substitution.
                with torch.enable_grad():
                    proposals = propose_by_gradient(
                        model,
                        ids,
                        position=position,
                        target_token_id=int(target_token_id),
                        indices=remaining,
                        k=n_proposals,
                        fixed=assignment,
                    )
                # The proposer's own forward IS a forward pass through the
                # model and is counted as one. Its backward is not a forward
                # pass and is published separately as `backward_passes`.
                passes += 1
                proposer_passes += 1
                pairs = [(p["index"], p["token_id"]) for p in proposals]
                trials_unavailable += max(0, n_proposals - len(pairs))
            else:
                donors = draw(n_donors)
                pairs = [(i, d) for i in remaining for d in donors]

            best_here: tuple[float, int, int] | None = None
            hit: tuple[int, int] | None = None
            for scanned, (index, donor) in enumerate(pairs):
                if int(ids[0, index]) == donor:
                    # Substituting a token for itself is a no-op that would
                    # still cost a pass and could "win" a step by tying.
                    trials_skipped_self += 1
                    continue
                trial = dict(assignment)
                trial[index] = donor
                row = edited(trial)
                probs = forward_probs(row)
                del row
                score = float(probs[int(target_token_id)])
                if int(probs.argmax()) == int(target_token_id):
                    hit = (index, donor)
                    best_here = (score, index, donor)
                    # Every pair after this one in the scan order is never
                    # reached, and unreached pairs are not spent passes.
                    trials_short_circuited += len(pairs) - (scanned + 1)
                    break
                if best_here is None or score > best_here[0]:
                    best_here = (score, index, donor)

            if best_here is None:
                stopped = (
                    "every candidate offered for this step was already the "
                    "token at its position, so no substitution was possible"
                )
                break

            # A STEP THAT MOVES AWAY FROM THE TARGET IS NOT PROGRESS.
            # Greedy picks the best candidate on offer, and "best on offer" can
            # still be worse than committing nothing. Measured on Qwen3-1.7B
            # before this guard existed, searching "The Eiffel Tower is in the
            # city of" toward its own runner-up: the three committed steps took
            # p(target) 0.004418 -> 0.002118 -> 0.000441, each one further from
            # the target than the last, and the run reported an edit of size 3
            # as its best effort. The edit was real and the direction was
            # backwards. Stop instead, and say which it was.
            if hit is None and best_here[0] <= current_p:
                stopped = (
                    "no remaining substitution raised the target's "
                    f"probability above {current_p:.6g} — the best one "
                    f"on offer reached {best_here[0]:.6g} — so the "
                    "search stopped rather than commit a step that moves away "
                    "from the target"
                )
                break

            if use_gradient and pairs:
                # Did the estimate's FIRST choice win the step it was ranked
                # for? Published as a count, so the screen's quality is visible
                # rather than assumed. A first-order estimate that never agrees
                # is a screen the reader should stop trusting.
                screen_steps += 1
                screen_agreed += int((best_here[1], best_here[2]) == pairs[0])

            score, index, donor = best_here
            current_p = score
            assignment[index] = donor
            remaining.remove(index)
            steps_taken += 1
            trail.append(
                {
                    "step": steps_taken,
                    "index": index,
                    "from_token_id": int(ids[0, index]),
                    "from_token": _token(decode, int(ids[0, index])),
                    "to_token_id": donor,
                    "to_token": _token(decode, donor),
                    "target_p_after": float(score),
                }
            )
            best_effort = {
                "size": steps_taken,
                "target_p": float(score),
                "reached_target": hit is not None,
            }
            if hit is not None:
                found = True
                stopped = "the target became the top-1 prediction"
                break
        else:
            stopped = (
                f"the edit budget of {limit} position(s) was spent without the "
                "target becoming the top-1 prediction"
            )

        # THE CONTROLS. Only once there is an edit to draw alternatives against:
        # controlling nothing would spend 2 * n_controls passes to measure the
        # base rate a second time.
        same_hits = 0
        any_hits = 0
        same_taken = 0
        any_taken = 0
        abandoned = 0
        controlled = found and n_controls > 0

        def control_edit(where: Sequence[int]) -> dict[int, int] | None:
            """A random edit at `where`, with no self-substitutions in it.

            The SEARCH skips a donor that is already the token at a position, so
            a control that allowed one would be drawing systematically weaker
            edits than the edit it is controlling — a fraction of its draws
            would be no-ops — and would clear the bar by construction. That is
            the difference between a control and a formality.

            Bounded rather than looped: a pool that cannot offer this position
            anything but its own token returns None, the sample is abandoned,
            and the count it was going to join shrinks by one. A draw nobody
            took must not be priced as a pass or counted as a failure.
            """
            nonlocal abandoned
            trial: dict[int, int] = {}
            for index in where:
                current = int(ids[0, index])
                for _attempt in range(MAX_CONTROL_REDRAWS):
                    donor = draw(1)[0]
                    if donor != current:
                        trial[index] = donor
                        break
                else:
                    abandoned += 1
                    return None
            return trial

        if controlled:
            size = len(assignment)
            indices = sorted(assignment)
            for _ in range(n_controls):
                trial = control_edit(indices)
                if trial is None:
                    continue
                row = edited(trial)
                probs = forward_probs(row)
                del row
                same_taken += 1
                same_hits += int(int(probs.argmax()) == int(target_token_id))
            for _ in range(n_controls):
                picks = torch.randperm(len(perturbable), generator=generator)[:size]
                where = [perturbable[int(i)] for i in picks.tolist()]
                trial = control_edit(where)
                if trial is None:
                    continue
                row = edited(trial)
                probs = forward_probs(row)
                del row
                any_taken += 1
                any_hits += int(int(probs.argmax()) == int(target_token_id))

    same_positions = _interval(same_hits, same_taken, confidence=confidence)
    any_positions = _interval(any_hits, any_taken, confidence=confidence)
    beats = is_finding(same_positions, any_positions)

    expected = cost_passes(
        n_positions=len(perturbable),
        # SCANNED, not committed. A step that screened its candidates and then
        # declined to commit any of them — because none raised the target —
        # paid for that screen in full. Pricing committed steps only understates
        # the run by one whole step's scan, which is how this number and
        # `passes` first came apart.
        steps=steps_scanned,
        n_donors=n_donors,
        control_passes=same_taken + any_taken,
        trials_not_run=(
            trials_skipped_self + trials_short_circuited + trials_unavailable
        ),
        proposals_per_step=n_proposals if use_gradient else None,
        proposer_passes=proposer_passes,
    )

    edited_row = edited(assignment)
    return {
        "position": position,
        "base_token_id": base_top,
        "base_token": _token(decode, base_top),
        "target_token_id": int(target_token_id),
        "target_token": _token(decode, int(target_token_id)),
        "base_target_p": float(base_target_p),
        "base_target_rank": base_target_rank,
        "noise_floor_kl": round(floor, 6),
        "agreement_kl": round(agreement, 6),
        "found": found,
        "stopped_because": stopped,
        "edit": trail,
        "size": len(assignment),
        "edited_ids": edited_row[0].tolist(),
        "edited_ids_are": (
            "the prompt with the edit applied, every index preserved. Suitable "
            "as the corrupt prompt for patch.py: it was searched for against a "
            "named target and controlled, rather than typed."
        ),
        # Present whether or not the search succeeded. A near miss is a real
        # thing to report and it is not the thing that was asked for, so it
        # never appears under `found`.
        "best_effort": best_effort,
        "controls": {
            "same_positions": same_positions,
            "same_positions_asks": (
                "do random words at the SAME indices also reach the target? If "
                "they do, the positions carry the flip and the chosen words do "
                "not."
            ),
            "any_positions": any_positions,
            "any_positions_asks": (
                "does an edit of this SIZE anywhere in the perturbable window "
                "reach the target? If it does, the model is fragile to edits "
                "of that size and this search isolated nothing."
            ),
            "measured": bool(controlled and same_taken and any_taken),
            "not_measured_because": (
                None
                if controlled and same_taken and any_taken
                else (
                    "no edit was found, so there is no size to draw alternatives at"
                    if not found
                    else "n_controls was 0, so no control was requested"
                    if n_controls == 0
                    else (
                        "every control draw was abandoned: the donor pool could "
                        "not offer these positions any token other than the one "
                        f"already there within {MAX_CONTROL_REDRAWS} redraws"
                    )
                )
            ),
            "draws_requested_per_arm": n_controls,
            "draws_abandoned": abandoned,
            "no_self_substitution": (
                "control edits exclude a donor that is already the token at "
                "that position, exactly as the search does. Allowing them would "
                "make a fraction of every control draw a no-op and clear the "
                "bar by construction."
            ),
        },
        "beats_controls": beats,
        "beats_controls_means": (
            "both control arms returned zero successes over "
            f"{same_taken} and {any_taken} draws. That is not proof of zero — "
            "read each interval's upper bound beside it."
        ),
        "trials_skipped_self": trials_skipped_self,
        "trials_short_circuited": trials_short_circuited,
        "trials_unavailable": trials_unavailable,
        "candidates": candidates,
        "screen": {
            "source": (
                "first-order substitution estimate (HotFlip), measured by a "
                "real forward pass on every pair it shortlists"
                if use_gradient
                else "the donor corpus pool, drawn frequency-weighted"
            ),
            "proposals_per_step": n_proposals if use_gradient else None,
            "backward_passes": proposer_passes,
            "top_choice_won": _interval(
                screen_agreed, screen_steps, confidence=confidence
            ),
            "top_choice_won_asks": (
                "how often the estimate's FIRST choice was the pair that "
                "actually won its step. A screen that never agrees is a screen "
                "to stop trusting; one that always agrees on one prompt has "
                "still only been checked on one prompt."
            ),
        },
        "minimality": {
            "search": "greedy forward selection over (position, donor)",
            "smaller_may_exist": True,
            "positions_considered": len(perturbable),
            "donors_per_step": n_donors,
        },
        "perturbation": perturbation,
        "candidate_window": {
            "perturbable_indices": perturbable,
            "excluded": (
                "index 0 as an attention sink, the queried position as its own "
                "query, everything after it as unreachable through a causal "
                "mask, and any control_ids being held fixed"
            ),
        },
        "typed_span": list(typed_span) if typed_span is not None else None,
        "n_prompt": n_prompt,
        "passes": passes,
        "passes_expected": expected,
        "seconds": round(time.perf_counter() - started, 3),
        "seed": int(seed),
    }
