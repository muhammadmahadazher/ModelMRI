# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""The three robot-action measurements, and what each of them refuses.

An action curve is the most persuasive-looking thing this project draws: two
lines on one axis, one labelled "policy" and one "recorded", read as a score.
Somebody will conclude something about a robot from the gap between them.

So the tests here are mostly about refusals, and each one names the wrong
conclusion it exists to prevent.

Nothing here needs lerobot, a policy or a GPU — `vla_actions` does no I/O and
holds no model, which is exactly what makes the arithmetic checkable.
"""

from __future__ import annotations

import math

import pytest

from modelmri import vla_actions as va
from modelmri.errors import BadRequest

# --------------------------------------------------------------- units first


def test_a_policy_that_never_published_its_units_cannot_be_overlaid():
    """The substantive refusal. Two unlabelled lists of floats of the same
    length is exactly the case that looks comparable and is not."""
    agree, why = va.units_agree({}, {"action": {"mean": [0.0, 1.0]}})
    assert not agree
    assert "does not publish" in why
    assert "one axis" in why


def test_a_dataset_that_never_published_its_units_cannot_be_overlaid_either():
    """Symmetric, because the failure is symmetric."""
    agree, why = va.units_agree({"action": {"mean": [0.0]}}, {})
    assert not agree
    assert "no stated units" in why


def test_different_action_widths_are_different_robots():
    agree, why = va.units_agree(
        {"action": {"mean": [0.0] * 7}}, {"action": {"mean": [0.0] * 6}}
    )
    assert not agree
    assert "7 action dimensions" in why and "recorded 6" in why
    assert "unrelated joints" in why


def test_matching_units_still_say_what_the_comparison_is():
    """A chart that CAN be drawn still needs to say on what basis. Agreement
    is not permission to drop the caveat."""
    agree, why = va.units_agree(
        {"action": {"mean": [0.0] * 6}}, {"action": {"mean": [0.0] * 6}}
    )
    assert agree
    assert "ONE human demonstration" in why
    assert "not against ground truth" in why


# ------------------------------------------------- predicted versus recorded


def _rows(n=4, width=3, offset=0.1):
    return [(t, [offset + t * 0.01] * width, [0.0] * width) for t in range(n)]


def test_a_comparison_reports_signed_bias_not_just_error():
    """A policy that reaches 2 cm further EVERY frame and one that is randomly
    wrong by 2 cm have the same mean absolute error and are different
    policies. Signed delta is what tells them apart."""
    out = va.compare(frames=_rows(), total_frames=4)
    assert out["dimensions"] == 3
    assert all(b > 0 for b in out["bias"]), (
        "a consistent overshoot must not average to zero"
    )


def test_the_caveats_travel_in_the_body_not_the_docs():
    """The person who needs "this is one demonstration" is looking at the
    chart, not reading a manual."""
    said = va.compare(frames=_rows(), total_frames=4)["means"]
    assert "NOT GROUND TRUTH" in said
    assert "open-loop teacher forcing" in said
    assert "error never compounds" in said


def test_a_strided_run_says_how_many_frames_are_missing():
    """Silent truncation reads as "covered the episode". A divergence between
    sampled frames is not in the chart and the sentence has to say so."""
    out = va.compare(frames=_rows(4), total_frames=200, stride=50)
    assert out["frames_skipped"] == 196
    assert "196 frames were skipped" in out["means"]


def test_an_unseeded_run_says_it_is_unseeded():
    """ "No seed" and "seed 0" are different runs, and only one of them
    reproduces."""
    assert (
        "re-running gives a different curve"
        in (va.compare(frames=_rows(), total_frames=4)["means"])
    )
    assert "seed 7" in va.compare(frames=_rows(), total_frames=4, seed=7)["means"]


def test_mismatched_widths_at_one_frame_refuse_the_whole_comparison():
    frames = [(0, [0.1, 0.2, 0.3], [0.0, 0.0, 0.0]), (1, [0.1, 0.2], [0.0, 0.0, 0.0])]
    with pytest.raises(va.NotComparable, match="unrelated joints"):
        va.compare(frames=frames, total_frames=2)


def test_an_episode_that_changes_action_width_partway_refuses_rather_than_crashing():
    """The width check next door is PER FRAME -- predicted against recorded at
    that one frame -- so an episode whose frames are internally consistent and
    disagree with EACH OTHER walks past it and reaches the bias comprehension,
    where `r.delta[d]` runs off the end of the narrow row and raises a bare
    IndexError. That is a 500 carrying no sentence, from the one module in
    this package whose whole argument is that a refusal is an authored
    sentence naming what is missing."""
    frames = [(0, [0.1, 0.2], [0.0, 0.0]), (1, [0.5], [0.0])]
    with pytest.raises(va.NotComparable, match="cannot change action space"):
        va.compare(frames=frames, total_frames=2)


def test_a_nan_action_is_refused_rather_than_drawn_as_a_gap():
    """A NaN draws as a break in the line, which reads as the policy deciding
    to hold still — indistinguishable from a real hold."""
    frames = [(0, [0.1, float("nan")], [0.0, 0.0])]
    with pytest.raises(va.NotComparable, match="hold still"):
        va.compare(frames=frames, total_frames=1)


def test_no_frames_is_a_refusal_not_an_empty_chart():
    """An empty result reads as "the policy matched perfectly"."""
    with pytest.raises(BadRequest, match="did not happen"):
        va.compare(frames=[])


def test_joint_names_that_do_not_match_the_width_are_dropped_entirely():
    """A chart with six curves and five labels mislabels at least one, and a
    mislabelled joint is worse than an unlabelled one."""
    out = va.compare(frames=_rows(width=3), joint_names=["a", "b"], total_frames=4)
    assert out["joint_names"] == []


# ---------------------------------------------------------- chunk consistency
#
# The overlap between two SUCCESSIVE predictions, and never between a
# prediction and a recorded action. `chunk_p[dt + k]` and `chunk_q[k]` are two
# claims about the SAME absolute timestep `t_p + dt + k`, made from two
# different observations, and `sample.action` is on neither side of that
# subtraction.
#
# Every test here is about that one index, because getting it wrong produces a
# perfectly reasonable-looking number that measures something else: the
# tempting `chunk_p[k]` against `chunk_q[k]` compares two frames `dt` apart and
# reads as a policy that revises constantly, on a policy that revised nothing.


def _consistency(chunks, **kw):
    """The consistency block, with `frames` built from the SAME chunks.

    One source for both arguments on purpose: a test that let the first-step
    rows drift away from the chunks they came out of would be checking a
    wiring the server cannot produce.
    """
    frames = [(t, list(chunk[0]), [0.0] * len(chunk[0])) for t, chunk in chunks]
    out = va.compare(frames=frames, chunks=chunks, total_frames=len(frames), **kw)
    return out["chunk_consistency"]


def test_identical_overlapping_predictions_are_a_consistency_of_exactly_zero():
    """0.0 IS the measurement here, and it is the one number the not-measured
    branch must never produce. A policy that revises nothing scores zero; a
    policy nobody could measure scores nothing at all."""
    step = [0.1, 0.2, 0.3]
    out = _consistency([(0, [step, step, step]), (1, [step, step, step])])
    assert out["measurable"] is True
    assert out["median"] == 0.0
    assert out["p25"] == 0.0 and out["p75"] == 0.0
    assert out["horizon"] == 3
    assert out["worst_pair"]["distance"] == 0.0


def test_the_smallest_overlap_is_exactly_one_shared_timestep():
    """H=2 one frame apart shares ONE absolute timestep. `chunk_p[k]` vs
    `chunk_q[k]` -- the tempting wrong pairing -- finds two, and
    `chunk_p[dt + k]` vs `chunk_q[k + 1]` finds none. The COUNT is the
    off-by-one detector, and on this ramp so is the value: the wrong pairing
    also reports a divergence of 1.0 for a policy that agrees with itself
    perfectly."""
    out = _consistency([(0, [[0.0], [1.0]]), (1, [[1.0], [2.0]])])
    assert out["overlapping_steps"] == 1
    assert out["pairs"] == 1
    assert out["median"] == 0.0
    assert out["by_steps_ahead"] == [
        {
            "steps_ahead": 1,
            "pairs": 1,
            "overlapping_steps": 1,
            "median": 0.0,
            "p25": 0.0,
            "p75": 0.0,
        }
    ]


def test_a_four_step_horizon_two_frames_apart_shares_two_timesteps():
    """`L = min(H_q, H_p - dt) = 2`, and the two are `(chunk_p[2], chunk_q[0])`
    and `(chunk_p[3], chunk_q[1])` -- absolute frames 2 and 3."""
    out = _consistency(
        [
            (0, [[float(k)] for k in range(4)]),
            (2, [[float(2 + k)] for k in range(4)]),
        ],
        stride=2,
    )
    assert out["pairs"] == 1
    assert out["overlapping_steps"] == 2
    assert out["median"] == 0.0


def test_a_hand_computed_divergence_lands_in_the_dimension_it_was_put_in():
    """+0.25 in dimension 1 alone, over a two-step overlap. Exact in binary, so
    this asserts the arithmetic rather than a float's last bit -- and the two
    dimensions that did not move must read exactly 0.0, because a metric that
    smears one joint's revision across all of them cannot say which joint."""
    early = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]]
    late = [[0.0, 1.25, 0.0], [0.0, 2.25, 0.0], [0.0, 3.25, 0.0]]
    out = _consistency([(0, early), (1, late)])
    assert out["overlapping_steps"] == 2
    assert out["median"] == 0.25
    assert [d["revision_bias"] for d in out["by_dimension"]] == [0.0, 0.25, 0.0]
    assert [d["disagreement"] for d in out["by_dimension"]] == [0.0, 0.25, 0.0]


def test_the_revision_bias_is_signed_so_a_consistent_revision_does_not_average_away():
    """The same argument `Divergence.delta` already makes: a policy that
    revises this joint DOWNWARD every time and one that jitters around it have
    the same absolute disagreement and are different policies."""
    early = [[0.0], [1.0], [2.0]]
    late = [[0.5], [1.5], [2.5]]
    out = _consistency([(0, early), (1, late)])
    # chunk_0[1] = 1.0 against chunk_1[0] = 0.5, and chunk_0[2] = 2.0 against
    # chunk_1[1] = 1.5: the later chunk revises DOWN by 0.5 both times.
    assert out["by_dimension"][0]["revision_bias"] == -0.5
    assert out["by_dimension"][0]["disagreement"] == 0.5


def test_a_stride_past_the_horizon_refuses_and_says_it_is_not_a_zero():
    """The common case rather than an exotic one: a 5,000-frame episode plans a
    stride of 79 and plenty of policies return fewer steps than that. An empty
    result or a 0.0 would read as a policy that agreed with itself perfectly,
    which is the opposite of "nobody could tell"."""
    out = _consistency([(0, [[0.0], [1.0]]), (5, [[5.0], [6.0]])], stride=5)
    assert out["measurable"] is False
    assert "NOT A CONSISTENCY OF ZERO" in out["means"]
    assert "5 frames apart" in out["means"]
    assert "is 2 steps" in out["means"]
    assert "stride of 5" in out["means"]
    assert out["median"] is None
    assert out["overlapping_steps"] is None
    assert out["pairs"] is None
    assert out["by_steps_ahead"] == []
    assert out["by_dimension"] == []


def test_a_one_step_chunk_has_no_future_for_a_later_chunk_to_disagree_about():
    """A different sentence from the stride refusal, because a different thing
    is wrong and a different thing would fix it: no stride makes a one-step
    chunk overlap."""
    out = _consistency([(0, [[0.0]]), (1, [[1.0]]), (2, [[2.0]])], stride=1)
    assert out["measurable"] is False
    assert "ONE step long" in out["means"]
    assert "NOT A CONSISTENCY OF ZERO" in out["means"]
    assert out["worst_pair"] is None


def test_one_sampled_frame_leaves_no_second_prediction_to_disagree_with():
    """ONE chunk, and the sentence has to say one PREDICTION rather than one
    frame. "One frame was sampled" is the shared opening of both single-frame
    leads, so asserting only that would pass against the OTHER one -- the
    several-chunks-on-one-frame lead, which would read "1 chunks were
    predicted on it" here and describe a run that did not happen. The mirror
    of `test_two_chunks_at_one_frame_and_nowhere_else_are_still_one_frame`,
    which pins the same seam from the other side."""
    out = _consistency([(0, [[0.0], [1.0], [2.0]])])
    assert out["measurable"] is False
    assert "this policy made one prediction" in out["means"]
    assert "chunks were predicted on it" not in out["means"]
    assert "NOT A CONSISTENCY OF ZERO" in out["means"]
    assert out["pairs"] is None


def test_unequal_horizons_report_a_range_rather_than_one_number():
    """A single `horizon` field would be a lie when the chunks were not all the
    same length, and the overlap has to be computed from BOTH of them."""
    out = _consistency([(0, [[0.0]] * 5), (1, [[0.0]] * 2)])
    assert out["horizon"] is None
    assert out["horizon_min"] == 2 and out["horizon_max"] == 5
    # min(H_q, H_p - dt) = min(2, 4) = 2 -- not 4, and not 5.
    assert out["overlapping_steps"] == 2
    assert "VARIED" in out["means"]


def test_two_chunks_at_the_identical_frame_are_skipped_and_counted():
    """They differ by SAMPLING NOISE, which is exactly what the instruction
    swap's seed reference measures. Folding it in here would silently turn
    this into a different metric under the same name."""
    out = _consistency([(0, [[0.0], [1.0]]), (0, [[0.0], [5.0]]), (1, [[1.0], [2.0]])])
    assert out["pairs_skipped_same_frame"] == 1
    assert out["pairs"] == 2
    assert out["overlapping_steps"] == 2
    assert "sampling noise" in out["means"]
    # The COUNT in the sentence too, not only the field. A skipped pair the
    # sentence miscounts is the same failure as one it does not mention.
    assert "1 pair(s) of chunks predicted at the identical frame" in out["means"]


def test_the_denominator_travels_with_the_median():
    """Five frames at a stride of 2 with a 4-step horizon is EIGHT shared
    timesteps from four chunk pairs. A median over eight and a median over one
    are different claims and the reader has to be able to tell."""
    chunks = [(t, [[float(t + k)] for k in range(4)]) for t in (0, 2, 4, 6, 8)]
    out = _consistency(chunks, stride=2)
    assert out["overlapping_steps"] == 8
    assert out["pairs"] == 4
    assert [r["steps_ahead"] for r in out["by_steps_ahead"]] == [2]
    assert out["by_steps_ahead"][0]["overlapping_steps"] == 8


def test_a_non_finite_later_step_refuses_the_block_and_spares_the_error_section():
    """`_finite` reads step 0 only. Extending it over the whole chunk would let
    a NaN at step 7 take down a comparison that succeeds today -- and dropping
    the frame instead would hide it inside an aggregate, where a missing
    disagreement reads as agreement."""
    chunks = [(0, [[0.0], [float("nan")]]), (1, [[1.0], [2.0]])]
    frames = [(0, [0.0], [0.0]), (1, [1.0], [0.0])]
    out = va.compare(frames=frames, chunks=chunks, total_frames=2)
    assert out["worst_frame"] == 1
    assert out["rows"][0]["distance"] == 0.0
    block = out["chunk_consistency"]
    assert block["measurable"] is False
    assert "non-finite" in block["means"]
    # The FRAME as well as the step. "step 1" alone would pass against a
    # sentence naming the wrong chunk, and the reader's next move is to go
    # look at that chunk.
    assert "non-finite value at step 1 of the chunk at frame 0" in block["means"]
    assert block["median"] is None


def test_an_infinity_in_a_later_chunk_step_refuses_like_a_nan():
    """The guard is "not finite", not "is a NaN", and the difference is not
    cosmetic: an infinity that reached `fmt.measured_value` comes back
    unchanged and serialises as a bare `Infinity`, which `JSON.parse` rejects
    outright -- the panel would show no block at all rather than a sentence
    saying why."""
    chunks = [(0, [[0.0], [float("inf")]]), (1, [[1.0], [2.0]])]
    frames = [(0, [0.0], [0.0]), (1, [1.0], [0.0])]
    block = va.compare(frames=frames, chunks=chunks, total_frames=2)[
        "chunk_consistency"
    ]
    assert block["measurable"] is False
    assert "non-finite value at step 1 of the chunk at frame 0" in block["means"]
    assert block["median"] is None


def test_a_chunk_that_changes_width_partway_refuses_the_block():
    """Pairing dimension 3 with dimension 3 across two widths compares
    unrelated joints -- `units_agree`'s argument, one field over."""
    chunks = [(0, [[0.0, 1.0], [0.0]]), (1, [[1.0, 2.0], [1.0, 2.0]])]
    frames = [(0, [0.0, 1.0], [0.0, 0.0]), (1, [1.0, 2.0], [0.0, 0.0])]
    block = va.compare(frames=frames, chunks=chunks, total_frames=2)[
        "chunk_consistency"
    ]
    assert block["measurable"] is False
    assert "unrelated joints" in block["means"]
    assert "2 dimensions wide at step 0 and 1 at step 1" in block["means"]


def test_a_comparison_given_no_chunks_says_so_rather_than_reporting_zero():
    """Every caller that predates the chunk plumbing still gets a block, and it
    says nothing was compared rather than reporting a consistency."""
    out = va.compare(frames=_rows(), total_frames=4)["chunk_consistency"]
    assert out["measurable"] is False
    assert out["median"] is None
    assert "only the first step" in out["means"]


def test_the_block_says_it_compares_the_policy_against_itself():
    """A reader who has just read "A RECORDED ACTION IS NOT GROUND TRUTH" two
    paragraphs up will fold this number into that frame unless it says
    otherwise. Nothing recorded is on either side of this subtraction, and a
    policy that predicts the same wrong action every time scores perfectly."""
    out = _consistency(
        [(0, [[0.0], [1.0]]), (1, [[1.0], [2.0]])],
        policy_repo="a/b",
        revision="rev",
    )
    said = out["means"]
    assert "AGAINST ITSELF" in said
    assert "NO THRESHOLD IS APPLIED" in said
    assert "one dataset frame" in said
    assert "a/b" in said and "rev" in said
    # The declined ROADMAP feature next door refuses on deterministic heads.
    # This one does not, and a reader has to be able to tell them apart.
    assert "deterministic" in said


def test_the_first_step_comparison_is_untouched_by_the_chunks_beside_it():
    """The whole point of a separate block. If plumbing the chunk moved one
    digit of `rows`, `bias`, `worst_frame` or `means`, the new measurement
    would have been bought with the old one."""
    chunks = [
        (0, [[0.10, 0.10], [0.9, 0.9]]),
        (1, [[0.11, 0.11], [0.8, 0.8]]),
        (2, [[0.12, 0.12], [0.7, 0.7]]),
    ]
    frames = [(t, list(c[0]), [0.0, 0.0]) for t, c in chunks]
    before = va.compare(frames=frames, total_frames=3, stride=1)
    after = va.compare(frames=frames, chunks=chunks, total_frames=3, stride=1)
    assert {k: v for k, v in after.items() if k != "chunk_consistency"} == {
        k: v for k, v in before.items() if k != "chunk_consistency"
    }
    assert after["chunk_consistency"]["measurable"] is True
    assert before["chunk_consistency"]["measurable"] is False


def test_the_error_block_still_reports_the_numbers_it_reported_before():
    """The same fixture written out as LITERALS, captured from `compare` before
    the chunk plumbing existed. A regression that recomputes both sides with
    the same arithmetic agrees with itself whatever that arithmetic became."""
    chunks = [(0, [[3.0, 4.0], [1.0, 1.0]]), (1, [[6.0, 8.0], [2.0, 2.0]])]
    frames = [(0, [3.0, 4.0], [0.0, 0.0]), (1, [6.0, 8.0], [0.0, 0.0])]
    out = va.compare(
        frames=frames,
        chunks=chunks,
        total_frames=2,
        stride=1,
        policy_repo="a/b",
        revision="rev",
        seed=3,
    )
    assert out["rows"] == [
        {
            "t": 0,
            "predicted": [3.0, 4.0],
            "recorded": [0.0, 0.0],
            "delta": [3.0, 4.0],
            "distance": 5.0,
        },
        {
            "t": 1,
            "predicted": [6.0, 8.0],
            "recorded": [0.0, 0.0],
            "delta": [6.0, 8.0],
            "distance": 10.0,
        },
    ]
    assert out["worst_frame"] == 1
    assert out["worst_distance"] == 10.0
    assert out["bias"] == [4.5, 6.0]
    assert out["dimensions"] == 2
    assert out["joint_names"] == []
    assert out["frames_measured"] == 2
    assert out["frames_in_episode"] == 2
    assert out["frames_skipped"] == 0
    assert out["stride"] == 1
    assert out["policy_repo"] == "a/b"
    assert out["revision"] == "rev"
    assert out["seed"] == 3
    assert out["means"] == (
        "2 of 2 frames, comparing a/b at revision rev against what the "
        "demonstrator did. The largest gap is at frame 1, mostly in "
        "dimension 1. Sampled at seed 3; another seed gives another curve. "
        "A RECORDED ACTION IS ONE HUMAN DEMONSTRATION, NOT GROUND TRUTH: a "
        "policy can differ and be right, because a different grasp that also "
        "works looks exactly like an error here. And every prediction was "
        "made on the human's own observations, so this is open-loop teacher "
        "forcing — error never compounds the way it would on a robot."
    )


def test_chunks_that_describe_different_frames_from_the_rows_are_refused():
    """The two arguments come out of ONE loop in the server. If they ever stop
    matching, this block would be measuring a different set of frames from the
    section above it and nothing on screen would say so."""
    frames = [(0, [0.0], [0.0]), (1, [1.0], [0.0])]
    chunks = [(0, [[0.0], [1.0]]), (7, [[7.0], [8.0]])]
    with pytest.raises(BadRequest, match="different frames"):
        va.compare(frames=frames, chunks=chunks, total_frames=2)


def test_an_empty_chunk_refuses_the_block_and_names_the_frame():
    """The route already refuses an empty chunk before it gets here, so this is
    the belt to that brace -- and it names the frame, because "some chunk was
    empty" is not something anybody can go and look at."""
    chunks = [(0, [[0.0], [1.0]]), (1, [])]
    frames = [(0, [0.0], [0.0]), (1, [1.0], [0.0])]
    block = va.compare(frames=frames, chunks=chunks, total_frames=2)[
        "chunk_consistency"
    ]
    assert block["measurable"] is False
    assert "chunk at frame 1 has no steps in it" in block["means"]


def test_a_zero_width_chunk_is_not_a_disagreement_of_zero():
    """An L2 over no dimensions is 0.0 for every pair, which would render as a
    policy in perfect agreement with itself. It is a sum over nothing."""
    chunks = [(0, [[], []]), (1, [[], []])]
    frames = [(0, [0.0], [0.0]), (1, [1.0], [0.0])]
    block = va.compare(frames=frames, chunks=chunks, total_frames=2)[
        "chunk_consistency"
    ]
    assert block["measurable"] is False
    assert "zero action dimensions wide" in block["means"]
    assert block["median"] is None


def test_a_chunk_step_that_is_not_a_number_refuses_rather_than_crashing():
    """A 500 carrying "Internal Server Error" throws away a first-step
    comparison that is perfectly good, over a value in a step it never read."""
    chunks = [(0, [[0.0], ["later"]]), (1, [[1.0], [2.0]])]
    frames = [(0, [0.0], [0.0]), (1, [1.0], [0.0])]
    block = va.compare(frames=frames, chunks=chunks, total_frames=2)[
        "chunk_consistency"
    ]
    assert block["measurable"] is False
    assert "not a number" in block["means"]
    assert "Step 1 of the chunk at frame 0" in block["means"]


def test_a_stride_refusal_says_when_the_chunks_were_not_all_one_length():
    """The refusal names ONE horizon, and naming the longest without saying it
    was the longest would describe a policy that never ran."""
    out = _consistency([(0, [[0.0], [1.0]]), (4, [[0.0]] * 3)], stride=4)
    assert out["measurable"] is False
    assert "not all the same length" in out["means"]
    assert "from 2 to 3 steps" in out["means"]
    assert "planned at a stride of 4" in out["means"]


def test_the_stride_a_ragged_refusal_recommends_is_set_by_the_shortest_chunk():
    """`overlap = min(H_q, H_p - dt)` is zero the moment `H_p <= dt`, however
    long `H_q` is, so the stride that overlaps every consecutive pair is set by
    the SHORTEST chunk and the longest is only a ceiling. They are the same
    number when the horizons agree, which is exactly why reading both off the
    longest looked right -- and on ragged horizons it names a stride that does
    not do what the sentence says it does.

    The assertion that gives this teeth is the last one: the number the
    refusal hands the reader has to be one that actually overlaps."""
    out = _consistency(
        [(0, [[0.0], [1.0]]), (4, [[0.0]] * 3), (8, [[0.0]] * 3)], stride=4
    )
    assert out["measurable"] is False
    assert "A stride of 1 or less overlaps every consecutive pair" in out["means"]
    # The longest chunk is 3 steps, and `hi - 1` is the number this sentence
    # used to recommend. At a stride of 2 the pair whose earlier chunk is two
    # steps long shares nothing.
    assert "A stride of 2 or less overlaps" not in out["means"]
    assert "3 steps" in out["means"]
    # The same three horizons, sampled at the stride the refusal names.
    closer = [(0, [[0.0], [1.0]]), (1, [[0.0]] * 3), (2, [[0.0]] * 3)]
    assert _consistency(closer, stride=1)["measurable"] is True


def test_a_ragged_refusal_never_recommends_the_stride_the_run_already_used():
    """The self-refuting case. The frames here are ONE apart and the longest
    chunk is THREE steps, and those two numbers are the case that overlaps --
    yet nothing overlapped, because the chunk at the earlier frame is a single
    step. A sentence built from `gap` and `hi` says "so no two chunks ever
    describe the same absolute timestep" about numbers that contradict it, and
    then recommends a stride of 2 to a run that already sampled at 1."""
    out = _consistency([(0, [[0.0]]), (1, [[0.0], [1.0], [2.0]])], stride=1)
    assert out["measurable"] is False
    assert "A stride of 2 or less overlaps" not in out["means"]
    assert "no stride makes THAT one overlap" in out["means"]
    # And the one number in this sentence the plural rule had not reached.
    assert "1 frame apart" in out["means"]
    assert "1 frames apart" not in out["means"]


def test_two_chunks_at_one_frame_and_nowhere_else_are_still_one_frame():
    """Two predictions, and no pair of them a frame apart. The count of what
    was skipped travels, because a skipped pair that nothing mentions reads as
    a pair that agreed."""
    out = _consistency([(0, [[0.0], [1.0]]), (0, [[0.0], [5.0]])])
    assert out["measurable"] is False
    # TWO predictions, and the sentence says two. "This policy made one
    # prediction" is the lead for the single-chunk case and it would be false
    # here, inside a sentence whose whole job is saying what was measured.
    assert "One frame was sampled and 2 chunks were predicted on it" in out["means"]
    assert "made one prediction" not in out["means"]
    assert "sampling noise" in out["means"]


def test_the_curve_past_the_nearest_gap_is_labelled_as_an_extension():
    """Rows beyond the stride compare chunks that were never consecutive. All
    three published detectors compare consecutive inference steps only, and an
    unlabelled row would read as though they had done this too."""
    chunks = [(t, [[float(t + k)] for k in range(4)]) for t in (0, 1, 2)]
    out = _consistency(chunks, stride=1)
    assert [r["steps_ahead"] for r in out["by_steps_ahead"]] == [1, 2]
    assert "NON-ADJACENT" in out["means"]


def test_a_run_with_only_adjacent_overlaps_claims_no_extension():
    """The other half, because a caveat that is always printed is a caveat
    nobody reads."""
    out = _consistency([(0, [[0.0], [1.0]]), (1, [[1.0], [2.0]])])
    assert "NON-ADJACENT" not in out["means"]


def test_the_middle_half_is_an_exact_quartile_over_the_readings_in_hand():
    """Eight readings: six of 0.25 one frame ahead and two of 0.5 two frames
    ahead. Inclusive quartiles over that list put p75 a quarter of the way
    from 0.25 to 0.5. Exact, off the sorted values, rather than interpolated
    off a histogram the way `vla_data`'s percentiles have to be."""
    chunks = [
        (t, [[float(t + k), 10.0 + 0.25 * t] for k in range(3)]) for t in range(4)
    ]
    out = _consistency(chunks, stride=1)
    assert out["overlapping_steps"] == 8
    assert out["pairs"] == 5
    assert out["median"] == 0.25
    assert out["p25"] == 0.25
    assert out["p75"] == 0.3125
    # The largest single disagreement, and which two chunks it came from.
    assert out["worst_pair"] == {
        "t_earlier": 0,
        "t_later": 2,
        "steps_ahead": 2,
        "step": 0,
        "distance": 0.5,
    }


def test_chunks_that_arrive_out_of_order_are_measured_in_frame_order():
    """`dt` has to be positive for the index algebra to mean what it says, and
    ascending order is not a promise the caller made -- only `plan_frames`
    guarantees it, and `compare(frames=...)` is open to anyone. Handed the same
    run backwards, an unsorted implementation computes every `dt` as negative,
    every overlap as empty, and reports "no two chunks ever describe the same
    absolute timestep" about a run where three pairs do."""
    ascending = [(t, [[float(t + k)] for k in range(3)]) for t in (0, 1, 2)]
    forwards = _consistency(ascending, stride=1)
    backwards = _consistency(list(reversed(ascending)), stride=1)
    assert forwards["measurable"] is True
    assert forwards["pairs"] == 3
    assert backwards == forwards


def test_a_divergence_too_small_for_six_places_is_not_stored_as_zero():
    """`round(x, 6)` writes 3e-07 as 0.0 -- a policy that just revised its own
    prediction, published with the reading of one that never revises anything.
    That exact contradiction is what `fmt.measured_value` was written to
    record, and this is the quantity it was written about."""
    early = [[0.0], [1.0], [2.0]]
    late = [[1.0 + 3e-07], [2.0 + 3e-07], [3.0]]
    out = _consistency([(0, early), (1, late)])
    assert out["median"] != 0.0
    assert out["median"] == 3e-07
    assert out["by_dimension"][0]["disagreement"] == 3e-07


def test_the_sentences_count_in_english():
    """ "1 frames later" and "rows past 1 steps ahead" are the "every 2th
    eligible row" typo `fmt.ordinal` exists to stop, and both fall on the
    ORDINARY run: a stride of 1 is what `plan_frames` picks for any episode
    under 64 frames, and a horizon of 2 is the smallest one this can measure.
    A reader is being asked to trust a measurement through these sentences."""
    refused = _consistency([(0, [[0.0], [1.0]]), (5, [[5.0], [6.0]])], stride=5)
    assert "ends 1 frame later" in refused["means"]
    assert "1 frames later" not in refused["means"]

    measured = _consistency(
        [(t, [[float(t + k)] for k in range(4)]) for t in (0, 1, 2)], stride=1
    )
    assert "past 1 step ahead" in measured["means"]
    assert "1 steps ahead" not in measured["means"]

    # And the plural branch of each, so the fix is a rule rather than a
    # special case for the number one.
    wider = _consistency([(0, [[0.0]] * 4), (9, [[0.0]] * 4)], stride=9)
    assert "ends 3 frames later" in wider["means"]
    spread = _consistency(
        [(t, [[float(t + k)] for k in range(6)]) for t in (0, 2, 4)], stride=2
    )
    assert "past 2 steps ahead" in spread["means"]

    # The gap between the two closest sampled frames, which is one clause over
    # from `ends 1 frame later` and was the last plural in this sentence still
    # hardcoded. It reads singular only when the chunks were ragged: with one
    # horizon, nothing can fail to overlap at a gap of one.
    assert "5 frames apart" in refused["means"]
    ragged = _consistency([(0, [[0.0]]), (1, [[0.0], [1.0], [2.0]])], stride=1)
    assert "1 frame apart" in ragged["means"]
    assert "1 frames apart" not in ragged["means"]


def test_two_chunks_of_different_widths_name_the_two_frames_they_came_from():
    """`width` is fixed by the FIRST chunk and checked against every step of
    every chunk after it, so a chunk that is a different width at its own step
    0 lands in the within-chunk sentence and prints "wide at step 0 and 1 at
    step 0" -- one frame, two widths, step 0 twice, and no way for the reader
    to see that the two numbers came from two different chunks.

    `frames` is deliberately uniform here while `chunks` is not. The server
    cannot produce that -- it builds both lists from one loop -- but
    `compare()` is a public function whose frame-alignment guard checks the
    frame numbers and not the widths, so this input reaches the block."""
    block = va.compare(
        frames=[(0, [0.0, 1.0], [0.0, 0.0]), (1, [1.0, 2.0], [0.0, 0.0])],
        chunks=[(0, [[0.0, 1.0], [0.0, 1.0]]), (1, [[1.0], [2.0]])],
        total_frames=2,
    )["chunk_consistency"]
    assert block["measurable"] is False
    assert "unrelated joints" in block["means"]
    said = block["means"]
    assert "at frame 0 is 2 dimensions wide and the chunk at frame 1 is 1" in said
    assert "at step 0 and 1 at step 0" not in said


def test_a_single_shared_timestep_reports_its_own_value_rather_than_a_zero():
    """ONE reading is the ordinary answer at a stride of H-1, the most common
    degenerate run this will meet, and `statistics.quantiles` raises below two
    points -- so that branch is hand-written and every other test of it feeds
    it a reading of 0.0, where "the median is the reading" and "the median is
    zero" cannot be told apart. Here the policy revised by 4.0."""
    out = _consistency([(0, [[0.0], [1.0]]), (1, [[5.0], [6.0]])])
    assert out["measurable"] is True
    assert out["overlapping_steps"] == 1
    assert out["median"] == 4.0
    assert out["p25"] == 4.0 and out["p75"] == 4.0
    # And the same branch again one level down, where each row of the strip
    # runs it over its own bucket.
    assert out["by_steps_ahead"] == [
        {
            "steps_ahead": 1,
            "pairs": 1,
            "overlapping_steps": 1,
            "median": 4.0,
            "p25": 4.0,
            "p75": 4.0,
        }
    ]
    assert out["by_dimension"][0]["revision_bias"] == 4.0
    assert out["by_dimension"][0]["disagreement"] == 4.0


def test_steps_ahead_come_from_the_frame_numbers_not_from_the_stride():
    """`plan_frames` returns a `range`, so every OTHER test here has a frame
    gap equal to the stride and cannot tell `t_q - t_p` from `stride` times a
    row distance. `compare(frames=...)` is caller-supplied and promises no
    such thing. The stride passed here is a number no pair is.

    This is also the only fixture in the file whose buckets arrive out of
    order -- the pairs are found as dt 1, 3, 2 -- so it is what holds the sort
    on the strip."""
    chunks = [(t, [[float(t + k)] for k in range(4)]) for t in (0, 1, 3)]
    out = _consistency(chunks, stride=99)
    assert [
        (r["steps_ahead"], r["pairs"], r["overlapping_steps"])
        for r in out["by_steps_ahead"]
    ] == [(1, 1, 3), (2, 1, 2), (3, 1, 1)]
    assert out["overlapping_steps"] == 6
    # The strip has to read standalone, so it carries the stride it was told
    # even when no pair of frames is that far apart.
    assert out["stride"] == 99


def test_the_measured_sentence_prints_the_median_the_field_carries():
    """A field and the sentence naming it must be the same quantity to the
    same precision -- `fmt.py` exists because they once were not. Nothing else
    in this file asserts a NUMBER inside the measured sentence, so `{:.4f}` in
    place of `fmt.measured` publishes `0.0000` beside a JSON field of 3e-07
    and passes: a policy that revised its own prediction, described in words
    as one that never revises anything."""
    early = [[0.0], [1.0], [2.0]]
    late = [[1.0 + 3e-07], [2.0 + 3e-07], [3.0]]
    out = _consistency([(0, early), (1, late)])
    assert out["median"] == 3e-07
    assert "median of 3.0e-07" in out["means"]
    assert "0.0000" not in out["means"]


def test_dimensions_the_dataset_could_not_name_are_left_unlabelled():
    """The block reuses the `names` list `compare()` already validated and
    dropped rather than re-deriving one, and a dimension it cannot name says
    so. A label that does not fit the width is worse than no label."""
    out = _consistency(
        [(0, [[0.0], [1.0]]), (1, [[1.0], [2.0]])], joint_names=["a", "b", "c"]
    )
    assert [d["name"] for d in out["by_dimension"]] == [None]

    # And the case that the block's OWN guard is for, which the one above
    # cannot reach: `compare()` sizes its name check against the width of the
    # first-step ROWS, and this block's width comes from the CHUNKS. Three
    # names that fit the rows exactly still do not fit a two-wide chunk, and
    # the block has to drop them a second time rather than take the first two
    # and mislabel the second joint.
    block = va.compare(
        frames=[
            (0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            (1, [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        ],
        chunks=[(0, [[0.0, 0.0], [1.0, 0.0]]), (1, [[1.0, 0.0], [2.0, 0.0]])],
        joint_names=["a", "b", "c"],
        total_frames=2,
    )
    assert block["joint_names"] == ["a", "b", "c"]
    assert [d["name"] for d in block["chunk_consistency"]["by_dimension"]] == [
        None,
        None,
    ]


# ----------------------------------------------------------- instruction swap


def test_a_single_task_dataset_is_refused_by_name():
    """Fabricating a distractor instruction would measure a sentence somebody
    invented rather than the policy."""
    with pytest.raises(va.NotComparable) as caught:
        va.instruction_swap(
            own_instruction="push the block",
            swapped=[("push the block", [0.1, 0.2])],
            seed_samples=[[0.1, 0.2], [0.11, 0.21]],
        )
    said = str(caught.value)
    assert "one distinct task string" in said
    assert "would not be a measurement of this policy" in said


def test_a_deterministic_policy_is_refused_rather_than_divided_by_zero():
    """The reference IS the sampling spread. A ratio against zero is not a
    number, and substituting a threshold from another paper would be
    borrowing a calibration this project does not have."""
    with pytest.raises(va.NotComparable) as caught:
        va.instruction_swap(
            own_instruction="a",
            swapped=[("a", [1.0, 0.0]), ("b", [0.0, 1.0])],
            seed_samples=[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
        )
    said = str(caught.value)
    assert "deterministic" in said
    assert "ratio against zero" in said


def test_an_instruction_that_moves_less_than_noise_is_said_in_those_words():
    """ROADMAP #50 asks for this sentence specifically."""
    out = va.instruction_swap(
        own_instruction="a",
        # Two instructions almost on top of each other...
        swapped=[("a", [1.000, 0.0]), ("b", [1.001, 0.0])],
        # ...against a sampler that moves the action a great deal more.
        seed_samples=[[1.0, 0.0], [1.4, 0.0], [0.6, 0.0]],
    )
    assert out["listens"] is False
    assert "LESS THAN RE-ROLLING THE SAMPLER" in out["means"]


def test_an_instruction_that_moves_more_than_noise_says_that_too():
    out = va.instruction_swap(
        own_instruction="a",
        swapped=[("a", [0.0, 0.0]), ("b", [5.0, 0.0])],
        seed_samples=[[0.0, 0.0], [0.01, 0.0], [-0.01, 0.0]],
    )
    assert out["listens"] is True
    assert out["ratio"] > 1.0


def test_an_empty_instruction_arm_is_labelled_no_instruction():
    """Never "the instruction did not matter" — that is a conclusion, and this
    is a condition the experiment ran on purpose."""
    out = va.instruction_swap(
        own_instruction="a",
        swapped=[("a", [1.0]), ("", [2.0])],
        seed_samples=[[1.0], [1.2], [0.8]],
    )
    labels = [arm["instruction"] for arm in out["arms"]]
    assert "(no instruction)" in labels


def test_the_sample_count_travels_because_five_and_five_hundred_differ():
    out = va.instruction_swap(
        own_instruction="a",
        swapped=[("a", [1.0]), ("b", [2.0])],
        seed_samples=[[1.0], [1.2], [0.8]],
    )
    assert out["seeds"] == 3
    assert "3 seeds" in out["means"]
    assert "ONE frame" in out["means"]


def test_dropped_instructions_are_reported_as_a_lower_bound():
    out = va.instruction_swap(
        own_instruction="a",
        swapped=[("a", [1.0]), ("b", [2.0])],
        seed_samples=[[1.0], [1.2], [0.8]],
        dropped_instructions=31,
    )
    assert "31 further distinct instructions" in out["means"]
    assert "lower bound" in out["means"]


# ------------------------------------------------------------------ knockout


def test_the_bars_carry_the_non_additivity_caveat():
    """Single-stream knockouts do not add up, and a reader who sums them
    reaches a wrong number that looks like a total."""
    out = va.knockout(
        baseline=[0.0, 0.0],
        arms=[
            ("observation.images.top", "top camera", [1.0, 0.0]),
            ("instruction", "no instruction", [0.2, 0.0]),
        ],
    )
    assert "DO NOT ADD UP" in out["means"]
    assert "interact" in out["means"]


def test_mean_substitution_is_never_described_as_removal():
    out = va.knockout(
        baseline=[0.0],
        arms=[("observation.state", "proprioceptive state", [0.5])],
    )
    assert "NOT REMOVAL" in out["means"]
    assert "it is not absence" in out["means"]


def test_without_a_sampling_reference_nothing_claims_a_bar_beats_noise():
    """Inventing a denominator would be worse than leaving the comparison
    out."""
    out = va.knockout(baseline=[0.0], arms=[("a", "camera a", [0.5])])
    assert out["rows"][0]["ratio_to_sampling"] is None
    assert out["rows"][0]["above_noise"] is None
    assert "nothing here says whether a bar is larger" in out["means"]


def test_with_a_sampling_reference_each_bar_is_compared_to_it():
    out = va.knockout(
        baseline=[0.0],
        arms=[("a", "camera a", [1.0]), ("b", "camera b", [0.01])],
        sampling_spread=0.1,
    )
    big, small = out["rows"]
    assert big["above_noise"] is True
    assert small["above_noise"] is False


def test_bars_are_sorted_by_effect_so_the_chart_reads_top_down():
    out = va.knockout(
        baseline=[0.0],
        arms=[("a", "a", [0.1]), ("b", "b", [0.9]), ("c", "c", [0.5])],
    )
    assert [r["stream"] for r in out["rows"]] == ["b", "c", "a"]


def test_no_streams_is_a_refusal():
    with pytest.raises(BadRequest, match="not a policy"):
        va.knockout(baseline=[0.0], arms=[])


# ------------------------------------------------------------- frame planning


def test_a_long_episode_is_strided_rather_than_truncated():
    """Measuring the first 64 frames of a 200-frame episode answers a question
    about the BEGINNING of the episode and labels it as the episode."""
    chosen, stride = va.plan_frames(200)
    assert stride > 1
    assert len(chosen) <= va.MAX_FRAMES_PER_RUN
    assert chosen[0] == 0
    assert chosen[-1] > 150, "an even stride must sample the whole episode"


def test_a_short_episode_is_measured_whole():
    chosen, stride = va.plan_frames(10)
    assert stride == 1
    assert chosen == list(range(10))


def test_an_empty_episode_refuses():
    with pytest.raises(BadRequest, match="no frames"):
        va.plan_frames(0)


def test_a_negative_stride_is_refused_rather_than_quietly_made_a_one():
    """MEASURED on a 161-frame episode: `stride=0` priced 54 forward passes
    and `stride=-1` ran 161 — 2.98x the cost the preflight exists to quote —
    while the response reported `stride: 1`, a number nobody had asked for.

    `-1` is truthy, so it took the "the caller set one" branch, and
    `max(1, -1)` made it a 1 without saying so. The sibling POST route bounds
    the same field at `ge=0`; this is the GET query param catching up.
    """
    with pytest.raises(BadRequest, match="step backwards"):
        va.plan_frames(161, stride=-1)


def test_a_bool_is_not_a_stride():
    """`isinstance(True, int)` is True, so `stride=True` came out of the same
    branch as a deliberate 1 and measured all 161 frames of the episode
    above."""
    for value in (True, False, 2.5, "3"):
        with pytest.raises(BadRequest, match="not a number of frames"):
            va.plan_frames(161, stride=value)


def test_the_planner_still_reports_the_stride_it_actually_used():
    """The half that must not move: a positive stride is honoured exactly, and
    0 still means "choose one that fits the budget"."""
    frames, stride = va.plan_frames(161, stride=3)
    assert stride == 3 and len(frames) == 54
    frames, stride = va.plan_frames(161, stride=0)
    assert stride == 3 and len(frames) == 54


# ------------------------------------------------------------------- spread


def test_spread_is_mean_distance_so_one_outlier_widens_it_without_defining_it():
    tight = va._spread([[0.0], [0.1], [-0.1]])
    with_outlier = va._spread([[0.0], [0.1], [-0.1], [10.0]])
    assert with_outlier > tight
    # A max would put the reference AT the outlier; a mean moves toward it.
    assert with_outlier < 10.0


def test_a_single_sample_has_no_spread():
    """Zero, and every caller treats zero as a refusal rather than as a very
    small number."""
    assert va._spread([[1.0, 2.0]]) == 0.0


def test_spread_of_identical_vectors_is_exactly_zero():
    assert va._spread([[1.0], [1.0], [1.0]]) == 0.0


def test_distance_is_euclidean_over_the_dimensions():
    out = va.compare(frames=[(0, [3.0, 4.0], [0.0, 0.0])], total_frames=1)
    assert math.isclose(out["rows"][0]["distance"], 5.0)
