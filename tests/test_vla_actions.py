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
