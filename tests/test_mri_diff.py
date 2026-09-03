# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""A regression check that cannot fail is a green tick that means nothing.

Most of this file builds a second `.mri` that differs from the first in one
specific way and checks that `diff` names that way and exits non-zero. The rest
is about what it REFUSES to compare — two files that are not about the same run
produce numbers that look like a regression and are a category error.
"""

from __future__ import annotations

import gzip
import json

import pytest

from modelmri import mri_diff, session
from modelmri.errors import BadRequest


def _doc(raw: bytes) -> dict:
    return json.loads(gzip.decompress(raw))


def _pack(doc: dict) -> bytes:
    return gzip.compress(json.dumps(doc).encode())


def _mri(**over) -> bytes:
    kw = dict(
        model_id="gpt2",
        device="cpu",
        dtype="float32",
        n_params=124_439_808,
        tokens=["The", " capital", " of", " France", " is", " Paris"],
        prompt="The capital of France is",
        generation=" Paris",
        attention={(0, 0): [[1.0, 0.0], [0.5, 0.5]]},
        n_layers=1,
        n_heads=1,
        n_prompt=5,
        ranking={
            "baseline": "zero",
            "noise_floor_kl": 0.0,
            "ranked": [
                {"layer": 0, "head": 0, "kl": 0.9},
                {"layer": 0, "head": 1, "kl": 0.2},
            ],
        },
        patch={
            "grids": {"resid": [[0.5, -0.5], [0.1, 0.2]]},
            "sites": [],
            "clean": "The capital of France is",
            "corrupt": "The capital of Italy is",
        },
        receipts=[
            {
                "op": "generate",
                "dtype": "float32",
                "device": "cpu",
                "revision": "a" * 40,
                "request": {"greedy": True, "temperature": 0.0},
            }
        ],
    )
    kw.update(over)
    return session.build(**kw)


@pytest.fixture
def pair(tmp_path):
    """Two identical files, for a test to move one of them."""
    a, b = tmp_path / "a.mri", tmp_path / "b.mri"
    raw = _mri()
    a.write_bytes(raw)
    b.write_bytes(raw)
    return a, b


def _by_name(report) -> dict:
    return {d.name: d for d in report.deltas}


# ------------------------------------------------------------- nothing moved


def test_two_identical_files_report_no_change(pair):
    a, b = pair
    report = mri_diff.diff(a, b)
    assert report.changed == []
    assert report.exit_code() == 0
    assert _by_name(report)["head ranking"].status == mri_diff.SAME


def test_diff_needs_no_torch():
    """It has to be usable as a CI step, and a job that installs torch to
    check a regression is a job nobody adds. Both sides are already measured;
    comparing them is arithmetic."""
    import subprocess
    import sys

    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from modelmri import mri_diff; "
            "print('torch' in sys.modules, 'transformers' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "False False"


# ---------------------------------------------------------------- it catches


def test_a_changed_generation_is_caught_and_shows_both(pair):
    a, b = pair
    doc = _doc(b.read_bytes())
    doc["generation"] = " Berlin"
    b.write_bytes(_pack(doc))

    delta = _by_name(mri_diff.diff(a, b))["generation"]
    assert delta.status == mri_diff.CHANGED
    assert "Paris" in delta.detail and "Berlin" in delta.detail


def test_a_changed_generation_fails_at_any_threshold(pair):
    """There is no magnitude at which "the model now says something else" is
    within tolerance."""
    a, b = pair
    doc = _doc(b.read_bytes())
    doc["generation"] = " Berlin"
    b.write_bytes(_pack(doc))

    report = mri_diff.diff(a, b)
    assert report.exit_code(fail_over=999.0) == 1


def test_a_moved_head_score_is_caught_with_both_numbers(pair):
    a, b = pair
    doc = _doc(b.read_bytes())
    doc["ranking"]["ranked"][0]["kl"] = 0.5  # was 0.9
    b.write_bytes(_pack(doc))

    delta = _by_name(mri_diff.diff(a, b))["head ranking"]
    assert delta.status == mri_diff.CHANGED
    assert delta.measured["moved"][0]["head"] == "L0H0"
    assert delta.measured["moved"][0]["from"] == 0.9


def test_a_reordered_ranking_names_what_entered_and_left(pair):
    """ "Which head carries this answer" is what a reader acts on, so the
    headline is the order when it moved, not the magnitude."""
    # Eight heads, so the top five is a real subset and membership can
    # actually change. With the two-head default every head is in the top k
    # whatever the order, and the reorder would show up as a score move.
    a, b = pair
    doc_a = _doc(a.read_bytes())
    doc_a["ranking"]["ranked"] = [
        {"layer": 0, "head": h, "kl": 1.0 - h * 0.1} for h in range(8)
    ]
    a.write_bytes(_pack(doc_a))

    doc_b = _doc(b.read_bytes())
    # Exactly reversed: heads 0 and 1 fall out of the top five, 6 and 7 enter.
    doc_b["ranking"]["ranked"] = [
        {"layer": 0, "head": h, "kl": 0.3 + h * 0.1} for h in range(8)
    ]
    b.write_bytes(_pack(doc_b))

    delta = _by_name(mri_diff.diff(a, b))["head ranking"]
    assert delta.status == mri_diff.CHANGED
    assert "entered" in delta.detail
    assert "L0H7" in delta.measured["entered_top_k"]
    assert "L0H0" in delta.measured["left_top_k"]


def test_a_patching_site_changing_sign_is_the_finding(pair):
    """A cell that recovered the clean answer and now pushes away from it is a
    different causal story, however small the numbers are."""
    a, b = pair
    doc = _doc(b.read_bytes())
    doc["patch"]["grids"]["resid"][0][0] = -0.5  # was +0.5
    b.write_bytes(_pack(doc))

    delta = _by_name(mri_diff.diff(a, b))["patching"]
    assert delta.status == mri_diff.CHANGED
    assert delta.measured["sites_changed_sign"] >= 1
    assert "changed sign" in delta.detail


def test_an_attention_change_is_judged_against_the_files_own_precision(pair):
    a, b = pair
    doc = _doc(b.read_bytes())
    key = next(iter(doc["attention"]))
    doc["attention"][key]["scale"] = doc["attention"][key]["scale"] * 3
    b.write_bytes(_pack(doc))

    delta = _by_name(mri_diff.diff(a, b))["attention"]
    assert delta.floor is not None and delta.floor > 0
    assert "quantisation" in delta.measured["floor_from"]


# ------------------------------------------------------------- the threshold


def test_fail_over_lets_a_small_move_pass_and_stops_a_large_one(pair):
    a, b = pair
    doc = _doc(b.read_bytes())
    doc["ranking"]["ranked"][0]["kl"] = 0.4  # a 0.5 nat move
    b.write_bytes(_pack(doc))

    report = mri_diff.diff(a, b)
    assert report.exit_code(fail_over=1.0) == 0, "0.5 nats is under 1.0"
    assert report.exit_code(fail_over=0.05) == 1, "and over 0.05"
    assert report.exit_code() == 1, "with no threshold, past the floor fails"


# ------------------------------------------------- what it refuses to compare


def test_two_different_prompts_are_refused_not_diffed(pair):
    a, b = pair
    doc = _doc(b.read_bytes())
    doc["tokens"] = doc["tokens"][:-1]
    b.write_bytes(_pack(doc))

    with pytest.raises(BadRequest, match="not about the same run"):
        mri_diff.diff(a, b)


def test_a_different_shape_is_refused(pair):
    a, b = pair
    doc = _doc(b.read_bytes())
    doc["n_heads"] = 2
    b.write_bytes(_pack(doc))

    with pytest.raises(BadRequest, match="head count"):
        mri_diff.diff(a, b)


def test_a_sampled_run_is_refused(pair):
    """Two `.mri` at temperature > 0 differ for reasons that are not the
    model. The `generate` receipt records it, so this is checked."""
    a, b = pair
    doc = _doc(b.read_bytes())
    doc["receipts"][0]["request"]["greedy"] = False
    doc["receipts"][0]["request"]["temperature"] = 0.7
    b.write_bytes(_pack(doc))

    with pytest.raises(BadRequest, match="sampled"):
        mri_diff.diff(a, b)


def test_a_file_with_no_sampling_record_is_noted_not_assumed_greedy(pair):
    a, b = pair
    doc = _doc(b.read_bytes())
    doc["receipts"] = []
    b.write_bytes(_pack(doc))

    report = mri_diff.diff(a, b)
    assert any("does not record whether" in n for n in report.notes)


def test_a_dtype_difference_is_labelled_rather_than_silently_diffed(pair):
    """`patch.py` records bf16 moving the reference gap from 4.000 to 4.467
    and changing the reference token itself."""
    a, b = pair
    doc = _doc(b.read_bytes())
    doc["receipts"][0]["dtype"] = "bfloat16"
    doc["meta"]["dtype"] = "bfloat16"
    b.write_bytes(_pack(doc))

    report = mri_diff.diff(a, b)
    assert any("float32" in n and "bfloat16" in n for n in report.notes)


def test_a_different_commit_is_noted(pair):
    a, b = pair
    doc = _doc(b.read_bytes())
    doc["receipts"][0]["revision"] = "b" * 40
    b.write_bytes(_pack(doc))

    report = mri_diff.diff(a, b)
    assert any("different commits" in n for n in report.notes)


# ------------------------------------------- a missing block is not a zero


def test_a_missing_ranking_is_not_comparable_rather_than_unchanged(pair):
    """The 0.10 bug class exactly: an absent section read as a default. A
    missing ranking must not report as "same"."""
    a, b = pair
    doc = _doc(b.read_bytes())
    doc.pop("ranking", None)
    b.write_bytes(_pack(doc))

    delta = _by_name(mri_diff.diff(a, b))["head ranking"]
    assert delta.status == mri_diff.NOT_COMPARABLE
    assert "not a zero" in delta.detail


def test_a_missing_patch_is_not_comparable(pair):
    a, b = pair
    doc = _doc(b.read_bytes())
    doc.pop("patch", None)
    b.write_bytes(_pack(doc))

    assert _by_name(mri_diff.diff(a, b))["patching"].status == (mri_diff.NOT_COMPARABLE)


def test_nothing_comparable_is_not_a_failure(pair):
    """A pair this tool cannot compare is not a regression, and exiting
    non-zero for it would make `diff` unusable in CI."""
    a, b = pair
    doc = _doc(b.read_bytes())
    doc.pop("ranking", None)
    doc.pop("patch", None)
    b.write_bytes(_pack(doc))

    report = mri_diff.diff(a, b)
    assert report.exit_code() == 0


def test_rankings_from_different_baselines_are_not_compared(pair):
    """`ablate.py` measures the baselines agreeing only weakly, so a
    cross-baseline diff measures the baseline."""
    a, b = pair
    doc = _doc(b.read_bytes())
    doc["ranking"]["baseline"] = "resample"
    b.write_bytes(_pack(doc))

    delta = _by_name(mri_diff.diff(a, b))["head ranking"]
    assert delta.status == mri_diff.NOT_COMPARABLE
    assert "different baselines" in delta.detail


# ------------------------------------------------------------------- output


def test_a_missing_file_says_so(tmp_path, pair):
    a, _ = pair
    with pytest.raises(BadRequest, match="could not be read"):
        mri_diff.diff(a, tmp_path / "nope.mri")


def test_the_report_survives_json(pair):
    a, b = pair
    report = mri_diff.diff(a, b)
    round_tripped = json.loads(json.dumps(report.to_dict(), allow_nan=False))
    assert round_tripped["totals"]["same"] >= 1
    assert {d["name"] for d in round_tripped["deltas"]} >= {"generation", "attention"}


def test_the_rendered_report_names_the_units_of_the_threshold(pair):
    a, b = pair
    text = mri_diff.render(mri_diff.diff(a, b), fail_over=0.05)
    assert "own units" in text and "nats" in text


# --------------------------------------------------------------- logit lens


def _with_lens(doc: dict, leaders: list[str], settled: int | None = 2) -> dict:
    doc["lens"] = [
        {"layer": i, "tokens": [t, " other"], "probs": [0.7, 0.3], "entropy": 0.5}
        for i, t in enumerate(leaders)
    ]
    doc["lens_info"] = {
        "final": leaders[-1],
        "settled_at": settled,
        "n_layers": len(leaders),
    }
    return doc


def test_two_identical_trajectories_are_the_same(pair):
    a, b = pair
    for path in (a, b):
        path.write_bytes(
            _pack(_with_lens(_doc(path.read_bytes()), [" a", " b", " Paris"]))
        )
    delta = _by_name(mri_diff.diff(a, b))["logit lens"]
    assert delta.status == mri_diff.SAME
    assert delta.measured["layers_compared"] == 3


def test_the_layer_where_the_trajectory_diverges_is_the_finding(pair):
    """ "The answer used to be decided by layer 8 and now is not" is what a
    reader acts on; which token it was there is the supporting detail."""
    a, b = pair
    a.write_bytes(_pack(_with_lens(_doc(a.read_bytes()), [" a", " b", " Paris"])))
    b.write_bytes(_pack(_with_lens(_doc(b.read_bytes()), [" a", " Berlin", " Paris"])))

    delta = _by_name(mri_diff.diff(a, b))["logit lens"]
    assert delta.status == mri_diff.CHANGED
    assert delta.measured["first_divergence_layer"] == 1
    assert delta.measured["now"] == " Berlin"


def test_a_settling_layer_that_moved_is_caught_even_when_every_leader_matches(pair):
    """The subtle one: the same token can lead at every layer while the layer
    the model commits at moves, and that is a real change in the trajectory."""
    a, b = pair
    a.write_bytes(
        _pack(_with_lens(_doc(a.read_bytes()), [" a", " b", " Paris"], settled=2))
    )
    b.write_bytes(
        _pack(_with_lens(_doc(b.read_bytes()), [" a", " b", " Paris"], settled=1))
    )

    delta = _by_name(mri_diff.diff(a, b))["logit lens"]
    assert delta.status == mri_diff.CHANGED
    assert delta.measured == {"settled_at_a": 2, "settled_at_b": 1}


def test_a_file_with_no_lens_is_not_comparable_rather_than_unchanged(pair):
    a, b = pair
    a.write_bytes(_pack(_with_lens(_doc(a.read_bytes()), [" a", " b"])))
    assert _by_name(mri_diff.diff(a, b))["logit lens"].status == (
        mri_diff.NOT_COMPARABLE
    )


# ------------------------------------- an agreement is still a measurement


class _Maps:
    """Just the attribute `_diff_attention` reads."""

    def __init__(self, attention):
        self.attention = attention


def _cell(byte: int, scale: float) -> dict:
    """A 1x1 stored head map holding one value: `byte * scale`."""
    import base64

    return {"q": base64.b64encode(bytes([byte])).decode("ascii"), "scale": scale}


def test_two_agreeing_files_report_the_gap_and_floor_they_measured():
    """The SAME branch used to publish a fabricated (0.0, 0.0).

    The seed was `shared[0], 0.0, 0.0` and the replacement test is
    `gap - floor > worst_gap - worst_floor`, which against that seed reduces
    to `gap > floor`. In the SAME case nothing exceeds its floor by
    definition, so nothing ever replaced the seed: the answer came back as a
    difference of exactly zero at whichever block sorted first, and a floor of
    exactly zero -- a quantisation step that cannot exist, sitting next to a
    `floor_from` line describing where it had supposedly been read.

    "Worst" means the largest MARGIN -- gap minus floor, the block that came
    closest to exceeding what its files can represent -- so both blocks here
    are given the same floor and 1:1 the larger gap.
    """
    a = _Maps({"0:0": _cell(100, 0.004), "1:1": _cell(60, 0.004)})
    b = _Maps({"0:0": _cell(133, 0.003), "1:1": _cell(79, 0.003)})
    # floor = max(0.004, 0.003) = 0.004 on both blocks
    # 0:0 -> |0.400 - 0.399| = 0.001, margin -0.003
    # 1:1 -> |0.240 - 0.237| = 0.003, margin -0.001  <- worst
    delta = mri_diff._diff_attention(a, b)

    assert delta.status == mri_diff.SAME
    assert delta.measured["floor"] > 0, "a uint8 step is never zero"
    assert delta.measured["max_abs_diff"] > 0, "these files do differ"
    assert delta.measured["worst_block"] == "1:1", (
        "the worst block is the one with the largest margin, not the one that "
        "sorted first"
    )
    assert f"{delta.measured['max_abs_diff']:.2e}" in delta.detail
    assert "0.00e+00" not in delta.detail


def test_identical_files_still_report_a_real_floor():
    """A genuine zero difference is fine; a zero FLOOR never is."""
    same = {"0:0": _cell(100, 0.004)}
    delta = mri_diff._diff_attention(_Maps(dict(same)), _Maps(dict(same)))
    assert delta.status == mri_diff.SAME
    assert delta.measured["max_abs_diff"] == 0.0
    assert delta.measured["floor"] == pytest.approx(0.004)


# ------------------------------------------- an absent floor is not a zero


def _ranked(rows, floor):
    """A `.mri` whose ranking may or may not record a noise floor."""
    rank = {"baseline": "zero", "ranked": rows}
    if floor is not None:
        rank["noise_floor_kl"] = floor
    return session.build(
        model_id="Qwen/Qwen3-1.7B",
        device="cpu",
        dtype="float32",
        n_params=1,
        tokens=["a", "b"],
        prompt="p",
        generation="g",
        attention={(0, 0): [[1.0, 0.0], [0.5, 0.5]]},
        n_layers=1,
        n_heads=2,
        n_prompt=1,
        ranking=rank,
    )


_STEADY = [{"layer": 0, "head": 0, "kl": 0.10}, {"layer": 0, "head": 1, "kl": 0.05}]
_DRIFT = [{"layer": 0, "head": 0, "kl": 0.100001}, {"layer": 0, "head": 1, "kl": 0.05}]
_MOVED = [{"layer": 0, "head": 0, "kl": 0.30}, {"layer": 0, "head": 1, "kl": 0.05}]


def _ranking_delta(tmp_path, rows_a, floor_a, rows_b, floor_b):
    a, b = tmp_path / "a.mri", tmp_path / "b.mri"
    a.write_bytes(_ranked(rows_a, floor_a))
    b.write_bytes(_ranked(rows_b, floor_b))
    report = mri_diff.diff(str(a), str(b))
    delta = next(d for d in report.deltas if d.name == "head ranking")
    return delta, report.exit_code()


def test_a_ranking_with_no_recorded_floor_is_not_a_floor_of_zero(tmp_path):
    """`or 0.0` fabricated a CI failure against a floor no file claimed.

    Measured before the fix, on two files whose rankings carry rows and a
    baseline and no `noise_floor_kl`:

        head ranking  changed
        the top 2 are unchanged, but L0H0 moved 0.10000 -> 0.10000.
        1 of 2 heads moved past the 0.00e+00 noise floor.
        exit code: 1

    A red build, over last-digit drift, against a number that was invented
    here and then labelled "the coarser of the two files' recorded noise
    floors". This module states the rule three times for other sections — "A
    missing section is not a zero" — and broke it for the floor.
    """
    delta, code = _ranking_delta(tmp_path, _STEADY, None, _MOVED, 0.001)
    assert delta.status == mri_diff.NOT_COMPARABLE
    assert "the first file's ranking records no noise floor" in delta.detail
    assert "not a floor of zero" in delta.detail
    # And it names the next step rather than stopping at the refusal.
    assert "modelmri ablate" in delta.detail
    assert code == 0, "a comparison that could not be made is not a failure"

    # The other side, so neither is special-cased.
    delta, _ = _ranking_delta(tmp_path, _STEADY, 0.001, _MOVED, None)
    assert "the second file's ranking records no noise floor" in delta.detail


def test_a_floor_recorded_as_zero_is_still_a_measurement(tmp_path):
    """THE DISTINCTION THE FIX EXISTS FOR.

    0.0 is a legal recorded value — `ground.py` documents it as the measured
    CPU/float32 case — and a file that genuinely measured it must still get
    its comparison. A guard that swallowed this too would be quieter and just
    as wrong in the other direction.
    """
    delta, code = _ranking_delta(tmp_path, _STEADY, 0.0, _DRIFT, 0.0)
    assert delta.status == mri_diff.CHANGED
    assert delta.floor == 0.0
    assert code == 1


def test_the_ordinary_comparison_is_untouched(tmp_path):
    """The guard must not fire when both files carry a floor."""
    moved, code = _ranking_delta(tmp_path, _STEADY, 0.001, _MOVED, 0.001)
    assert moved.status == mri_diff.CHANGED and code == 1

    quiet, code = _ranking_delta(tmp_path, _STEADY, 0.001, _DRIFT, 0.001)
    assert quiet.status == mri_diff.SAME and code == 0


def test_a_head_that_moved_is_not_printed_as_two_identical_numbers(tmp_path):
    """`:.5f` printed a 1e-06 move as "moved 0.10000 -> 0.10000" — the same
    number twice, in a sentence whose whole content is that it changed."""
    delta, _ = _ranking_delta(tmp_path, _STEADY, 0.0, _DRIFT, 0.0)
    assert "0.10000 → 0.10000" not in delta.detail, delta.detail


def _eight(kls, floor):
    """A ranking over eight heads, so the top FIVE can gain and lose members."""
    rank = {
        "baseline": "zero",
        "ranked": [{"layer": 0, "head": i, "kl": v} for i, v in enumerate(kls)],
    }
    if floor is not None:
        rank["noise_floor_kl"] = floor
    return session.build(
        model_id="Qwen/Qwen3-1.7B",
        device="cpu",
        dtype="float32",
        n_params=1,
        tokens=["a", "b"],
        prompt="p",
        generation="g",
        attention={(0, 0): [[1.0, 0.0], [0.5, 0.5]]},
        n_layers=1,
        n_heads=8,
        n_prompt=1,
        ranking=rank,
    )


_BEFORE = [0.90, 0.80, 0.70, 0.60, 0.50, 0.10, 0.05, 0.01]
#: H7 rockets into the top five and H4 drops out of it.
_AFTER = [0.90, 0.80, 0.70, 0.60, 0.02, 0.10, 0.05, 0.95]


def _ranking_of(tmp_path, kls_a, floor_a, kls_b, floor_b):
    a, b = tmp_path / "a.mri", tmp_path / "b.mri"
    a.write_bytes(_eight(kls_a, floor_a))
    b.write_bytes(_eight(kls_b, floor_b))
    report = mri_diff.diff(str(a), str(b))
    return next(
        d for d in report.deltas if d.name == "head ranking"
    ), report.exit_code()


def test_a_missing_floor_does_not_hide_a_top_five_change(tmp_path):
    """WHICH HEADS ARE IN THE TOP FIVE needs no noise floor.

    It is a comparison of two orderings — "L0H7 entered and L0H4 left" says the
    answer moved to a different head — and it is the finding a reader acts on.

    The first version of the missing-floor guard returned NOT_COMPARABLE for
    the whole section and took that finding down with it: a real top-five
    change, reported as nothing, exit 0. Refusing to invent a floor is right;
    refusing to report what does not need one is a different mistake in the
    same place.
    """
    delta, code = _ranking_of(tmp_path, _BEFORE, None, _AFTER, 0.001)
    assert delta.status == mri_diff.CHANGED
    assert "L0H7 entered" in delta.detail and "L0H4 left" in delta.detail
    assert code == 1, "a changed ranking must still fail the build"
    # And the magnitude question is named as unanswered rather than guessed.
    assert "records no noise floor" in delta.detail
    assert delta.floor is None
    assert delta.magnitude is None


def test_a_missing_floor_still_refuses_the_magnitude_question(tmp_path):
    """With the ordering unchanged there is nothing floor-independent left to
    report, so the section is genuinely not comparable — and says which half it
    could answer."""
    delta, code = _ranking_of(tmp_path, _BEFORE, None, _BEFORE, 0.001)
    assert delta.status == mri_diff.NOT_COMPARABLE
    assert "the top 5 are the same heads in both files" in delta.detail
    assert code == 0


def test_the_ordinary_path_reports_both(tmp_path):
    """With floors on both sides the ordering and the magnitudes travel
    together, and the guard must not have changed that."""
    delta, code = _ranking_of(tmp_path, _BEFORE, 0.001, _AFTER, 0.001)
    assert delta.status == mri_diff.CHANGED
    assert "L0H7 entered" in delta.detail
    assert "noise floor" in delta.detail
    assert delta.floor == 0.001
    assert code == 1


# ------------------------------------------------------- the patching graph
#
# `.mri` carries two graph sections and `diff` compared neither, so a walk
# that moved to a different circuit read as six unchanged sections and exit 0.
# These two are not one section twice: `patch_graph` is a graph THIS tool
# measured, edge by edge against eight same-norm draws, and `graph` is a
# transcoder attribution graph another tool computed and this one forwards.


def _pg_edge(source, target, recovery=0.5, **over):
    """One measured edge, with the verdict every stored edge must carry."""
    edge = {
        "source": source,
        "target": target,
        "recovery": recovery,
        # `abs`, so a negative recovery does not produce a negative control:
        # "random noise recovered minus an eighth" is not a thing any draw
        # could report, and a fixture that says it teaches the wrong shape.
        "control_max": abs(recovery) / 2,
        "control_draws": 8,
        "clears_control": True,
        "clears_position": True,
    }
    edge.update(over)
    return edge


def _walk(**over) -> dict:
    """A patching graph over the pair its prune threshold was measured on.

    The threshold and the prompts travel together on purpose, copying
    `tests/test_session_patch_graph.py`: MEASURED on Qwen3-1.7B/bfloat16, "is
    located in" resolves to 0.006231 and "is in" to 0.007571, so a fixture
    carrying one prompt pair and the other pair's resolution would put an
    invented number where a measured one belongs.
    """
    graph = {
        "nodes": [
            {
                "id": "L11 MLP@9",
                "layer": 11,
                "head": None,
                "position": 9,
                "role": "seed",
            },
            {"id": "L10 MLP@9", "layer": 10, "head": None, "position": 9, "depth": 1},
            {"id": "L9H6@9", "layer": 9, "head": 6, "position": 9, "depth": 2},
        ],
        "edges": [
            _pg_edge("L10 MLP@9", "L11 MLP@9", 0.25),
            _pg_edge("L9H6@9", "L10 MLP@9", 0.25),
        ],
        "seeding": "Seeded from the 4 strongest sites the node grid flagged "
        "and walked back 2 level(s). 153 senders in total, 105 pruned.",
        "clean": "The Eiffel Tower is located in the city of",
        "corrupt": "The Colosseum is located in the city of",
        "depth": 2,
        "n_scored": 153,
        "n_pruned": 105,
        "prune_threshold": 0.006231,
        "prune_from": "the dtype's own recovery resolution",
        "frontier": ["L9H6@9"],
    }
    graph.update(over)
    return graph


def _attribution(**over) -> dict:
    """A borrowed attribution graph, in the shape the reader keeps.

    Edges name their endpoints by INDEX into a node list the section does not
    carry — `session._graph` builds its output from scratch and never copies
    `nodes` — which is why every guard below is about whether two files'
    indices name the same thing.
    """
    graph = {
        "n_nodes": 4,
        "edges": [
            {"source": 0, "target": 1, "weight": 0.5},
            {"source": 1, "target": 2, "weight": -0.25},
            {"source": 2, "target": 3, "weight": 0.125},
        ],
        "provenance": {
            "measured_by": "This graph was computed by another tool and read "
            "here. ModelMRI did not run the model.",
            "producer": "circuit-tracer",
            "model": "Qwen/Qwen3-1.7B",
        },
        "prompt": "The Eiffel Tower is located in the city of",
        "summary": {"nodes": 4, "returned_edges": 3, "edge_limit": 2000},
    }
    graph.update(over)
    return graph


def _graphed(patch_graph=None, graph=None) -> bytes:
    """A `.mri` over the pair both fixtures above are written against."""
    return session.build(
        model_id="Qwen/Qwen3-1.7B",
        device="cuda",
        dtype="bfloat16",
        n_params=1_720_574_976,
        tokens=[
            "The",
            " Eiffel",
            " Tower",
            " is",
            " located",
            " in",
            " the",
            " city",
            " of",
            " Paris",
        ],
        prompt="The Eiffel Tower is located in the city of",
        generation=" Paris",
        attention={(0, 0): [[1.0, 0.0], [0.5, 0.5]]},
        n_layers=12,
        n_heads=8,
        n_prompt=9,
        patch_graph=patch_graph,
        graph=graph,
    )


def _section_delta(tmp_path, name, raw_a, raw_b):
    a, b = tmp_path / "a.mri", tmp_path / "b.mri"
    a.write_bytes(raw_a)
    b.write_bytes(raw_b)
    report = mri_diff.diff(str(a), str(b))
    return next(d for d in report.deltas if d.name == name), report


def _walked(tmp_path, section_a, section_b):
    return _section_delta(
        tmp_path, "patching graph", _graphed(section_a), _graphed(section_b)
    )


def _attributed(tmp_path, section_a, section_b):
    return _section_delta(
        tmp_path,
        "attribution graph",
        _graphed(graph=section_a),
        _graphed(graph=section_b),
    )


def test_two_identical_patching_graphs_are_the_same(tmp_path):
    delta, report = _walked(tmp_path, _walk(), _walk())
    assert delta.status == mri_diff.SAME
    assert delta.magnitude == 0.0
    assert delta.floor == 0.006231
    assert report.exit_code() == 0


def test_a_missing_patching_graph_is_not_comparable_rather_than_unchanged(tmp_path):
    delta, report = _walked(tmp_path, _walk(), None)
    assert delta.status == mri_diff.NOT_COMPARABLE
    assert "the second file" in delta.detail
    assert "not a zero" in delta.detail
    assert report.exit_code() == 0, (
        "a comparison that could not be made is not a failure"
    )

    delta, _ = _walked(tmp_path, None, _walk())
    assert "the first file" in delta.detail


def test_an_edge_whose_control_verdict_flipped_is_the_headline(tmp_path):
    """The section's whole guarantee is that every drawn edge beat its eight
    same-norm draws. An edge that used to and no longer does is the strongest
    finding in it — and a flipped boolean has no magnitude in recovery
    fractions, so there is no threshold at which it is within tolerance."""
    moved = _walk(
        edges=[
            _pg_edge("L10 MLP@9", "L11 MLP@9", 0.25, clears_control=False),
            _pg_edge("L9H6@9", "L10 MLP@9", 0.25),
        ]
    )
    delta, report = _walked(tmp_path, _walk(), moved)
    assert delta.status == mri_diff.CHANGED
    assert "L10 MLP@9 → L11 MLP@9" in delta.detail
    assert delta.magnitude is None
    assert report.exit_code(fail_over=999.0) == 1


def test_a_second_finding_travels_with_the_headline_rather_than_under_it(tmp_path):
    """The branches are ordered by strength of claim and the first draft
    returned on the first one that matched, so a run where an edge lost its
    verdict AND another edge vanished printed only the verdict — and the
    vanished edge lived in `measured`, where a terminal reader never looks."""
    both = _walk(edges=[_pg_edge("L10 MLP@9", "L11 MLP@9", 0.25, clears_control=False)])
    delta, _ = _walked(tmp_path, _walk(), both)
    assert delta.status == mri_diff.CHANGED
    assert "no longer clears" in delta.detail
    assert "L9H6@9 → L10 MLP@9 left" in delta.detail


def test_an_edge_that_regained_its_control_verdict_is_not_reported_as_losing_it(
    tmp_path,
):
    """`patch_graph` KEEPS an edge that failed its controls and marks it
    `clears_control: false` rather than dropping it — "we tested this and it
    did not survive" and "we never saw this" are different findings. So a file
    carrying `false` is an ordinary file, False → True is an ordinary outcome,
    and a flip collected with a bare `!=` and printed with the losing verb
    published the exact opposite of what was measured."""
    failing = _walk(
        edges=[
            _pg_edge("L10 MLP@9", "L11 MLP@9", 0.25, clears_control=False),
            _pg_edge("L9H6@9", "L10 MLP@9", 0.25),
        ]
    )
    delta, report = _walked(tmp_path, failing, _walk())
    assert delta.status == mri_diff.CHANGED
    assert "no longer clears" not in delta.detail
    assert "L10 MLP@9 → L11 MLP@9 now clears" in delta.detail
    assert delta.measured["control_verdicts_gained"] == ["L10 MLP@9 → L11 MLP@9"]
    assert delta.measured["control_verdicts_lost"] == []
    assert delta.measured["verdicts_flipped"] == 1
    assert delta.magnitude is None, "a flipped boolean has no size either way"
    assert report.exit_code(fail_over=999.0) == 1


def test_both_directions_of_a_control_flip_are_named_in_one_sentence(tmp_path):
    """A mixed run is the case a single verb cannot describe: one edge lost its
    controls and another gained them, and both are the same finding with
    opposite words."""
    mixed = _walk(
        edges=[
            _pg_edge("L10 MLP@9", "L11 MLP@9", 0.25, clears_control=False),
            _pg_edge("L9H6@9", "L10 MLP@9", 0.25),
        ]
    )
    swapped = _walk(
        edges=[
            _pg_edge("L10 MLP@9", "L11 MLP@9", 0.25),
            _pg_edge("L9H6@9", "L10 MLP@9", 0.25, clears_control=False),
        ]
    )
    delta, _ = _walked(tmp_path, mixed, swapped)
    assert delta.status == mri_diff.CHANGED
    assert "L9H6@9 → L10 MLP@9 no longer clears" in delta.detail
    assert "L10 MLP@9 → L11 MLP@9 now clears" in delta.detail
    assert delta.measured["control_verdicts_lost"] == ["L9H6@9 → L10 MLP@9"]
    assert delta.measured["control_verdicts_gained"] == ["L10 MLP@9 → L11 MLP@9"]


def test_a_position_verdict_that_flipped_is_a_change(tmp_path):
    """The positive path of the three-valued check, which the None → True test
    below does not reach: two real booleans that disagree."""
    moved = _walk(
        edges=[
            _pg_edge("L10 MLP@9", "L11 MLP@9", 0.25, clears_position=False),
            _pg_edge("L9H6@9", "L10 MLP@9", 0.25),
        ]
    )
    delta, report = _walked(tmp_path, _walk(), moved)
    assert delta.status == mri_diff.CHANGED
    assert "L10 MLP@9 → L11 MLP@9 no longer clears the shifted-position" in delta.detail
    assert delta.magnitude is None
    assert delta.measured["position_verdicts_flipped"] == 1
    assert delta.measured["position_verdicts_lost"] == ["L10 MLP@9 → L11 MLP@9"]
    assert report.exit_code() == 1

    # And the other direction, which has its own verb for the same reason the
    # control verdict does: an edge that now clears a control it used to fail
    # is not one that stopped clearing it.
    delta, _ = _walked(tmp_path, moved, _walk())
    assert delta.status == mri_diff.CHANGED
    assert "no longer clears" not in delta.detail
    assert "L10 MLP@9 → L11 MLP@9 now clears the shifted-position" in delta.detail
    assert delta.measured["position_verdicts_gained"] == ["L10 MLP@9 → L11 MLP@9"]
    assert delta.measured["position_verdicts_lost"] == []


def test_a_position_flip_is_named_even_when_a_control_flip_outranks_it(tmp_path):
    """`flipped_control or flipped_position` returns the first truthy list, so
    the position flip was dropped from the sentence — and `measured` published
    a bare count for it, a number with nothing behind it. Both classes of
    verdict get a clause, and both get their names in the receipt."""
    both = _walk(
        edges=[
            _pg_edge("L10 MLP@9", "L11 MLP@9", 0.25, clears_control=False),
            _pg_edge("L9H6@9", "L10 MLP@9", 0.25, clears_position=False),
        ]
    )
    delta, _ = _walked(tmp_path, _walk(), both)
    assert delta.status == mri_diff.CHANGED
    assert "L10 MLP@9 → L11 MLP@9 no longer clears the eight same-norm" in delta.detail
    assert "L9H6@9 → L10 MLP@9 no longer clears the shifted-position" in delta.detail
    assert delta.measured["position_verdicts_lost"] == ["L9H6@9 → L10 MLP@9"]


def test_a_position_verdict_that_was_never_run_is_not_a_change(tmp_path):
    """`clears_position` is three-valued by design. None becoming True is a
    pass that was not run becoming one that was — a change in what the walk
    did, not in what the model does — and coercing None to False here would
    turn "not run" into "run, and failed"."""
    later = _walk(
        edges=[
            _pg_edge("L10 MLP@9", "L11 MLP@9", 0.25, clears_position=None),
            _pg_edge("L9H6@9", "L10 MLP@9", 0.25),
        ]
    )
    delta, report = _walked(tmp_path, later, _walk())
    assert delta.status == mri_diff.SAME
    assert report.exit_code() == 0


def test_a_recovery_that_changed_sign_is_a_different_causal_story(tmp_path):
    flipped = _walk(
        edges=[
            _pg_edge("L10 MLP@9", "L11 MLP@9", -0.25),
            _pg_edge("L9H6@9", "L10 MLP@9", 0.25),
        ]
    )
    delta, _ = _walked(tmp_path, _walk(), flipped)
    assert delta.status == mri_diff.CHANGED
    assert "changed sign" in delta.detail
    assert delta.measured["edges_changed_sign"] == 1


def test_an_edge_that_vanished_or_appeared_is_named(tmp_path):
    thinner = _walk(edges=[_pg_edge("L10 MLP@9", "L11 MLP@9", 0.25)])
    delta, report = _walked(tmp_path, _walk(), thinner)
    assert delta.status == mri_diff.CHANGED
    assert delta.measured["edges_left"] == ["L9H6@9 → L10 MLP@9"]
    assert delta.measured["edges_entered"] == []
    assert "left" in delta.detail
    assert report.exit_code() == 1

    # And the other direction, so neither side is special-cased.
    delta, _ = _walked(tmp_path, thinner, _walk())
    assert delta.measured["edges_entered"] == ["L9H6@9 → L10 MLP@9"]


def test_an_edge_that_vanished_names_the_threshold_it_may_have_crossed(tmp_path):
    """An absent edge is not a zero: the file that dropped it does not record
    what it scored. A recovery that fell below the prune threshold and one
    that stopped existing look identical from here, and the sentence has to
    say so rather than assert the stronger of the two."""
    thinner = _walk(edges=[_pg_edge("L10 MLP@9", "L11 MLP@9", 0.25)])
    delta, _ = _walked(tmp_path, _walk(), thinner)
    assert "prune threshold" in delta.detail
    assert "0.006231" in delta.detail


def test_a_recovery_that_moved_is_judged_against_the_files_own_threshold(tmp_path):
    """Not an epsilon chosen here. `patch_graph` records the threshold it
    pruned at and where it came from, and the coarser of the two files' is the
    finest difference this comparison can claim."""
    louder = _walk(
        edges=[
            _pg_edge("L10 MLP@9", "L11 MLP@9", 0.75),
            _pg_edge("L9H6@9", "L10 MLP@9", 0.25),
        ]
    )
    delta, report = _walked(tmp_path, _walk(), louder)
    assert delta.status == mri_diff.CHANGED
    assert delta.floor == 0.006231
    assert delta.magnitude == pytest.approx(0.5)
    assert delta.unit == "recovery fraction"
    assert "recovery resolution" in delta.measured["floor_from"]
    assert report.exit_code(fail_over=1.0) == 0, "0.5 is under 1.0"
    assert report.exit_code(fail_over=0.05) == 1, "and over 0.05"


def test_a_recovery_that_moved_under_the_threshold_is_not_a_change(tmp_path):
    quiet = _walk(
        edges=[
            _pg_edge("L10 MLP@9", "L11 MLP@9", 0.2500005),
            _pg_edge("L9H6@9", "L10 MLP@9", 0.25),
        ]
    )
    delta, report = _walked(tmp_path, _walk(), quiet)
    assert delta.status == mri_diff.SAME
    assert report.exit_code() == 0


def test_two_walks_of_different_reach_are_not_compared(tmp_path):
    """Edge count is quadratic in sites, so every such graph is a subset by
    construction. Two walks that went back different distances hold different
    edges for a reason that is not the model."""
    delta, report = _walked(tmp_path, _walk(), _walk(depth=3))
    assert delta.status == mri_diff.NOT_COMPARABLE
    assert "went back 2 level(s) and 3" in delta.detail
    assert report.exit_code() == 0


def test_walks_over_different_prompt_pairs_are_not_compared(tmp_path):
    other = _walk(corrupt="The Statue of Liberty is located in the city of")
    delta, _ = _walked(tmp_path, _walk(), other)
    assert delta.status == mri_diff.NOT_COMPARABLE
    assert "Statue of Liberty" in delta.detail

    # The clean prompt is the one the file is ABOUT, and two walks over two of
    # them is the likelier mistake of the pair. Guarded by the same loop, which
    # is why both members of it are pinned.
    delta, _ = _walked(
        tmp_path, _walk(), _walk(clean="The Colosseum is located in the city of")
    )
    assert delta.status == mri_diff.NOT_COMPARABLE
    assert "different clean prompts" in delta.detail


def test_a_walk_that_recorded_no_prompt_pair_is_not_a_walk_over_a_different_one(
    tmp_path,
):
    """`session._patch_graph` defaults an absent `clean`/`corrupt` to "", so a
    file that never recorded the pair arrived here indistinguishable from one
    that recorded the empty string — and the refusal then stated as fact that
    the two walks ran over different prompts, quoting '' as the second. An
    unknown is not a value, least of all inside an authored refusal."""
    silent = _walk(clean="", corrupt="")
    delta, report = _walked(tmp_path, _walk(), silent)
    assert delta.status == mri_diff.SAME, "one side saying nothing is not a difference"
    assert "different clean prompts" not in delta.detail
    assert report.exit_code() == 0

    # And both sides reach the receipt, so the reader can see which file was
    # the silent one rather than being told it ran over ''.
    assert delta.measured["clean_a"] == "The Eiffel Tower is located in the city of"
    assert delta.measured["clean_b"] is None
    assert delta.measured["corrupt_b"] is None


def test_a_zero_threshold_says_bit_for_bit_rather_than_a_tolerance(tmp_path):
    """The reader defaults a missing `prune_threshold` to 0.0, so a file that
    recorded zero and one that recorded nothing arrive identical here. Claiming
    a tolerance was applied would name a measurement neither file made."""
    bare = _walk(prune_threshold=0.0, prune_from="")
    delta, _ = _walked(tmp_path, bare, bare)
    assert delta.status == mri_diff.SAME
    assert "bit-for-bit" in delta.detail
    assert delta.floor == 0.0


def test_the_order_of_the_edge_list_is_not_a_change(tmp_path):
    """Two lists holding the same edges in a different order are the same
    graph, and a differ that walked them pairwise would report the ordering."""
    reversed_edges = _walk(
        edges=[
            _pg_edge("L9H6@9", "L10 MLP@9", 0.25),
            _pg_edge("L10 MLP@9", "L11 MLP@9", 0.25),
        ]
    )
    delta, _ = _walked(tmp_path, _walk(), reversed_edges)
    assert delta.status == mri_diff.SAME


def test_a_count_the_walk_never_recorded_stays_unknown(tmp_path):
    """`n_weak: 0` is a result — nothing was too weak — and a file that never
    recorded the number is a different fact. The reader keeps them apart with
    None, and the diff must not fold one into the other."""
    delta, _ = _walked(tmp_path, _walk(), _walk(n_weak=0))
    assert delta.measured["n_weak_a"] is None
    assert delta.measured["n_weak_b"] == 0


def test_an_edge_carried_twice_is_refused_rather_than_collapsed(tmp_path):
    """Two edges between one pair make the list a multiset, and joining two
    multisets on the pair compares an arbitrary member of one against an
    arbitrary member of the other."""
    doubled = _walk(
        edges=[
            _pg_edge("L10 MLP@9", "L11 MLP@9", 0.25),
            _pg_edge("L10 MLP@9", "L11 MLP@9", 0.75),
        ]
    )
    delta, report = _walked(tmp_path, _walk(), doubled)
    assert delta.status == mri_diff.NOT_COMPARABLE
    assert "more than once" in delta.detail
    assert report.exit_code() == 0


def test_two_walks_that_drew_no_edge_agree(tmp_path):
    """`build` refuses to write an edgeless patching graph, so this one is
    repacked in — and two walks that both found nothing above their threshold
    agree about that, rather than being incomparable."""
    a, b = tmp_path / "a.mri", tmp_path / "b.mri"
    for path in (a, b):
        doc = _doc(_graphed(_walk()))
        doc["patch_graph"] = _walk(edges=[])
        path.write_bytes(_pack(doc))

    delta = _by_name(mri_diff.diff(a, b))["patching graph"]
    assert delta.status == mri_diff.SAME
    assert "neither walk drew an edge" in delta.detail


def test_an_empty_edge_list_is_reported_without_a_reason_for_it(tmp_path):
    """The sentence used to say no sender cleared both its prune threshold and
    its controls — a cause nothing in the section records, and one this
    fixture's own numbers refute twice over: `patch_graph` KEEPS an edge that
    failed its controls, and 48 of these 153 senders survived the pruning."""
    a, b = tmp_path / "a.mri", tmp_path / "b.mri"
    for path in (a, b):
        doc = _doc(_graphed(_walk()))
        doc["patch_graph"] = _walk(edges=[])
        path.write_bytes(_pack(doc))

    delta = _by_name(mri_diff.diff(a, b))["patching graph"]
    assert "cleared both" not in delta.detail
    assert "its controls" not in delta.detail
    assert delta.measured["n_scored_a"] == 153, "and the counts carry the rest"
    assert delta.measured["n_pruned_a"] == 105


def test_the_coarser_of_the_two_thresholds_is_the_one_applied(tmp_path):
    """This section's only floor, and the direct analogue of the ranking's
    "coarser of the two recorded noise floors" — every other fixture here sets
    the same threshold on both sides, where `max` and `min` cannot differ.

    Both figures are MEASURED on Qwen3-1.7B/bfloat16 and both belong to this
    pair: the resolution is one representable step of the GAP between the two
    runs' answers, so an edit that moves the gap moves the threshold with it —
    0.006231 at a gap of 30.25 and 0.007571 at 24.25. Two files over one pair
    recording two thresholds is what a diff is FOR."""
    coarser = _walk(
        prune_threshold=0.007571,
        prune_from="the dtype's own recovery resolution at a gap of 24.25",
        edges=[
            _pg_edge("L10 MLP@9", "L11 MLP@9", 0.257),
            _pg_edge("L9H6@9", "L10 MLP@9", 0.25),
        ],
    )
    delta, report = _walked(tmp_path, _walk(), coarser)
    assert delta.floor == 0.007571, "the coarser of the two, not the finer"
    assert delta.status == mri_diff.SAME, "0.007 does not clear 0.007571"
    assert "24.25" in delta.measured["floor_from"], "the coarser side's sentence"
    assert report.exit_code() == 0


def test_the_highest_recovering_edges_reordering_is_a_change(tmp_path):
    """Which edges lead needs no floor, exactly as the ranking's top five does
    not — and the SAME sentence one branch below asserts precisely this ("the
    same edges recover the most"), so without it a real reorder reads as a
    clean build with a false sentence. Every fixture above carries two edges,
    where `top` is 2 and this branch cannot fire."""
    senders = ["L10 MLP@9", "L9H6@9", "L9H5@9", "L8H4@9", "L8H3@9", "L7H2@9"]
    nodes = [
        {"id": "L11 MLP@9", "layer": 11, "head": None, "position": 9, "role": "seed"}
    ]
    for i, name in enumerate(senders):
        nodes.append(
            {
                "id": name,
                "layer": 10 - i,
                "head": None if name.endswith("MLP@9") else i,
                "position": 9,
                "depth": 1,
            }
        )

    def _walk_of(recoveries):
        return _walk(
            nodes=nodes,
            edges=[
                _pg_edge(name, "L11 MLP@9", r)
                for name, r in zip(senders, recoveries, strict=True)
            ],
        )

    # The 5th and 6th strongest swap, by 0.0005 — well under the 0.006231
    # threshold, so nothing "moved" and only the ordering says anything.
    before = _walk_of([0.9, 0.8, 0.7, 0.6, 0.5, 0.4995])
    after = _walk_of([0.9, 0.8, 0.7, 0.6, 0.4995, 0.5])
    delta, report = _walked(tmp_path, before, after)
    assert delta.status == mri_diff.CHANGED
    assert delta.measured["entered_top_k"] == ["L7H2@9 → L11 MLP@9"]
    assert delta.measured["left_top_k"] == ["L8H3@9 → L11 MLP@9"]
    assert delta.measured["edges_moved"] == 0, "under the threshold, so not a move"
    assert "highest-recovering 5 changed" in delta.detail
    assert report.exit_code() == 1
    assert report.exit_code(fail_over=0.05) == 0, "and the floor still gates it"


def test_a_node_the_walk_reached_only_once_is_named(tmp_path):
    """A node that entered or left with the edge set unchanged — an isolated or
    frontier node — is a change in what the walk reached. And it gets its own
    sentence: the edge caveat was appended to the whole block, so a node-only
    finding was explained by a paragraph about an edge that neither entered nor
    left."""
    wider = _walk(
        nodes=_walk()["nodes"]
        + [{"id": "L8H1@9", "layer": 8, "head": 1, "position": 9, "depth": 2}]
    )
    delta, report = _walked(tmp_path, _walk(), wider)
    assert delta.status == mri_diff.CHANGED
    assert delta.measured["nodes_entered"] == ["L8H1@9"]
    assert delta.measured["edges_entered"] == []
    assert "the nodes walked changed" in delta.detail
    assert "An edge carried by one file and not the other" not in delta.detail
    assert "that threshold" not in delta.detail, "nothing here named one"
    assert report.exit_code() == 1


def test_a_recovery_of_zero_has_no_sign_to_have_changed(tmp_path):
    """Zero has no direction, so it cannot have changed direction. Excluding it
    is what keeps "this sender now pushes the answer away" from being said
    about a sender that was not pushing at all."""
    from_zero = _walk(
        edges=[
            _pg_edge("L10 MLP@9", "L11 MLP@9", 0.0),
            _pg_edge("L9H6@9", "L10 MLP@9", 0.25),
        ]
    )
    delta, _ = _walked(
        tmp_path,
        from_zero,
        _walk(
            edges=[
                _pg_edge("L10 MLP@9", "L11 MLP@9", 0.5),
                _pg_edge("L9H6@9", "L10 MLP@9", 0.25),
            ]
        ),
    )
    assert delta.status == mri_diff.CHANGED, "0.5 is well past the threshold"
    assert delta.measured["edges_changed_sign"] == 0
    assert "changed sign" not in delta.detail


# ---------------------------------------------------- the attribution graph


def test_two_identical_attribution_graphs_are_the_same(tmp_path):
    delta, report = _attributed(tmp_path, _attribution(), _attribution())
    assert delta.status == mri_diff.SAME
    assert report.exit_code() == 0


def test_a_missing_attribution_graph_is_not_comparable(tmp_path):
    delta, report = _attributed(tmp_path, _attribution(), None)
    assert delta.status == mri_diff.NOT_COMPARABLE
    assert "not a zero" in delta.detail
    assert report.exit_code() == 0


def test_attribution_graphs_from_different_tools_are_not_compared(tmp_path):
    """ModelMRI computed neither of these. Two tools' outputs differ for
    reasons that belong to the tools."""
    other = _attribution(
        provenance={
            "measured_by": "Another tool computed this.",
            "producer": "attribution-graphs",
            "model": "Qwen/Qwen3-1.7B",
        }
    )
    delta, _ = _attributed(tmp_path, _attribution(), other)
    assert delta.status == mri_diff.NOT_COMPARABLE
    assert "circuit-tracer" in delta.detail
    assert "attribution-graphs" in delta.detail


def test_attribution_graphs_of_different_sizes_are_not_compared(tmp_path):
    """An edge names its endpoints by index into a node list the section does
    not carry. With two different node counts the indices do not name the same
    nodes, and a per-edge comparison is arithmetic over two different graphs."""
    delta, _ = _attributed(tmp_path, _attribution(), _attribution(n_nodes=5))
    assert delta.status == mri_diff.NOT_COMPARABLE
    assert "declare 4 nodes and 5" in delta.detail
    assert "by index into a node list the section does not carry" in delta.detail


def test_attribution_graphs_over_different_prompts_are_not_compared(tmp_path):
    other = _attribution(prompt="The Colosseum is located in the city of")
    delta, _ = _attributed(tmp_path, _attribution(), other)
    assert delta.status == mri_diff.NOT_COMPARABLE
    assert "Colosseum" in delta.detail


def test_an_attribution_graph_that_records_no_prompt_is_not_a_different_prompt(
    tmp_path,
):
    """`prompt` is optional the whole way down — `circuit.to_session` forwards a
    `graph.prompt` that may be None and `session._graph` writes the key only for
    a string. Read as `.get("prompt") or ""`, a file that never recorded one
    became a file that recorded the empty prompt, and the refusal said in so
    many words that these were computed over different prompts, quoting '' as
    the second."""
    delta, report = _attributed(tmp_path, _attribution(), _attribution(prompt=None))
    assert delta.status == mri_diff.SAME
    assert "different prompts" not in delta.detail
    assert delta.measured["prompt_a"] == "The Eiffel Tower is located in the city of"
    assert delta.measured["prompt_b"] is None, "not measured, and not the empty string"
    assert report.exit_code() == 0


def test_attribution_graphs_exported_at_different_limits_are_not_compared(tmp_path):
    """The sibling of the `truncated` guard, and the one that fires on the
    common case: two exports at two limits, where `truncated` is often absent
    on both and that guard therefore declines."""
    smaller = _attribution(summary={"nodes": 4, "returned_edges": 3, "edge_limit": 50})
    delta, report = _attributed(tmp_path, _attribution(), smaller)
    assert delta.status == mri_diff.NOT_COMPARABLE
    assert "2000" in delta.detail and "50" in delta.detail
    assert "decided by the export and not by the graph" in delta.detail
    assert report.exit_code() == 0


def test_an_export_setting_only_one_file_recorded_reaches_the_receipt(tmp_path):
    """Both guards above fire only when BOTH files carry the field, so what
    survives them is exactly the one-sided case — and a receipt publishing the
    first file's value under an unqualified key then reports "no limit" for a
    pair where the second file's list was cut at two edges. Which is the
    confusion those guards exist to prevent."""
    whole = _attribution(summary={"nodes": 4, "returned_edges": 3})
    cut = _attribution(
        edges=[
            {"source": 0, "target": 1, "weight": 0.5},
            {"source": 1, "target": 2, "weight": -0.25},
        ],
        summary={"nodes": 4, "returned_edges": 2, "edge_limit": 2, "truncated": True},
    )
    delta, _ = _attributed(tmp_path, whole, cut)
    assert delta.status == mri_diff.CHANGED
    assert delta.measured["edges_left"] == ["#2 → #3"]
    assert delta.measured["edge_limit_a"] is None
    assert delta.measured["edge_limit_b"] == 2
    assert delta.measured["truncated_a"] is None
    assert delta.measured["truncated_b"] is True


def test_both_disclaimers_reach_the_receipt(tmp_path):
    """`measured_by` is the sentence saying ModelMRI did not compute this
    graph, and it is mandatory but not compared — the guards above check only
    `producer` and `model`. So two files may disclaim in two different
    sentences, and publishing one of them under an unqualified key reports the
    first file's disclaimer as the pair's."""
    reworded = _attribution(
        provenance={
            "measured_by": "Computed by circuit-tracer 0.4 and read here.",
            "producer": "circuit-tracer",
            "model": "Qwen/Qwen3-1.7B",
        }
    )
    delta, _ = _attributed(tmp_path, _attribution(), reworded)
    assert delta.status == mri_diff.SAME, "a reworded disclaimer is not a change"
    assert delta.measured["measured_by_a"].startswith("This graph was computed by")
    assert delta.measured["measured_by_b"] == (
        "Computed by circuit-tracer 0.4 and read here."
    )


def test_attribution_graphs_cut_at_different_points_are_not_compared(tmp_path):
    """`circuit.Graph.edges` returns only the strongest `edge_limit`, so one
    list being truncated and the other not means membership was decided by the
    limit rather than by the graph."""
    cut = _attribution(
        summary={
            "nodes": 4,
            "returned_edges": 3,
            "edge_limit": 2000,
            "truncated": True,
        }
    )
    whole = _attribution(
        summary={
            "nodes": 4,
            "returned_edges": 3,
            "edge_limit": 2000,
            "truncated": False,
        }
    )
    delta, _ = _attributed(tmp_path, whole, cut)
    assert delta.status == mri_diff.NOT_COMPARABLE
    assert "truncated" in delta.detail


def test_an_attribution_weight_that_changed_sign_is_reported_with_no_magnitude(
    tmp_path,
):
    """Nothing in the section says what the producing tool could resolve, so
    there is no floor to judge a weight against and none is invented. The sign
    needs none: an edge that pushed toward the answer and now pushes away is a
    different claim at any size."""
    flipped = _attribution(
        edges=[
            {"source": 0, "target": 1, "weight": 0.5},
            {"source": 1, "target": 2, "weight": -0.25},
            {"source": 2, "target": 3, "weight": -0.125},
        ]
    )
    delta, report = _attributed(tmp_path, _attribution(), flipped)
    assert delta.status == mri_diff.CHANGED
    assert "changed sign" in delta.detail
    assert delta.magnitude is None
    assert delta.floor is None
    assert delta.unit == "attribution weight"
    assert report.exit_code() == 1


def test_an_attribution_edge_that_entered_or_left_is_named(tmp_path):
    thinner = _attribution(
        edges=[
            {"source": 0, "target": 1, "weight": 0.5},
            {"source": 1, "target": 2, "weight": -0.25},
        ]
    )
    delta, report = _attributed(tmp_path, _attribution(), thinner)
    assert delta.status == mri_diff.CHANGED
    assert delta.measured["edges_left"] == ["#2 → #3"]
    assert report.exit_code() == 1


def _attribution_of(weights: list[float]) -> dict:
    """One attribution graph whose edges carry `weights`, in order."""
    return _attribution(
        n_nodes=len(weights) + 1,
        edges=[
            {"source": 0, "target": i + 1, "weight": w} for i, w in enumerate(weights)
        ],
    )


def test_the_strongest_attribution_edges_are_named_but_not_called_a_change(tmp_path):
    """Which edges are strongest is an ORDERING of the weights this section
    says it cannot judge, so it travels with the refusal rather than as a
    change. Reported as a change it decided a CI outcome by arithmetic: a
    CHANGED delta with no magnitude fails at every `--fail-over`."""
    before = _attribution_of([0.9, 0.8, 0.7, 0.6, 0.5, 0.1, 0.05, 0.01])
    after = _attribution_of([0.9, 0.8, 0.7, 0.6, 0.02, 0.1, 0.05, 0.95])
    delta, report = _attributed(tmp_path, before, after)
    assert delta.status == mri_diff.NOT_COMPARABLE
    assert "#0 → #8" in delta.measured["entered_top_k"]
    assert "#0 → #5" in delta.measured["left_top_k"]
    assert "#0 → #8 entered" in delta.detail, "and named where a reader looks"
    assert report.exit_code() == 0


def test_a_reorder_inside_the_last_digits_does_not_fail_ci(tmp_path):
    """The defect this fixes, at the size it appears: two adjacent weights a
    hair apart swap ranks 5 and 6, and the section that refuses to say a 0.5
    move is a change failed `--fail-over 1e9` on 1e-10."""
    before = _attribution_of([0.9, 0.8, 0.7, 0.6, 0.5, 0.5 - 1e-10, 0.2])
    after = _attribution_of([0.9, 0.8, 0.7, 0.6, 0.5 - 1e-10, 0.5, 0.2])
    delta, report = _attributed(tmp_path, before, after)
    assert delta.status == mri_diff.NOT_COMPARABLE
    assert delta.measured["entered_top_k"] == ["#0 → #6"]
    assert report.exit_code(fail_over=1e9) == 0
    assert report.exit_code() == 0

    # And the comparison it was inconsistent with: a weight that moved by 0.5
    # without reordering anything has always exited 0, and still does.
    delta, report = _attributed(
        tmp_path, _attribution_of([0.9, 0.2]), _attribution_of([1.4, 0.2])
    )
    assert delta.status == mri_diff.NOT_COMPARABLE
    assert report.exit_code() == 0


def test_the_receipt_survives_the_branch_that_prints_a_number(tmp_path):
    """The one branch of this differ where a number was actually computed, and
    the only one that published nothing to check it against: `worst_edge` and
    both files' summary scalars were all in scope and dropped, so `--json` got
    a sentence with a figure in it and no receipt."""
    delta, _ = _attributed(
        tmp_path, _attribution_of([0.9, 0.2]), _attribution_of([1.4, 0.2])
    )
    assert delta.status == mri_diff.NOT_COMPARABLE
    assert delta.measured["worst_edge"] == "#0 → #1"
    assert delta.measured["max_abs_weight_diff"] == pytest.approx(0.5)
    assert delta.measured["floor"] is None


def test_a_weight_that_only_moved_has_no_floor_to_be_judged_against(tmp_path):
    """The honest half-answer: membership and order held, and the remaining
    question — how far a weight moved — needs a resolution no attribution graph
    carries. Reported as unavailable rather than as a magnitude of unknown
    significance, and the number is still printed."""
    louder = _attribution(
        edges=[
            {"source": 0, "target": 1, "weight": 0.9},
            {"source": 1, "target": 2, "weight": -0.25},
            {"source": 2, "target": 3, "weight": 0.125},
        ]
    )
    delta, report = _attributed(tmp_path, _attribution(), louder)
    assert delta.status == mri_diff.NOT_COMPARABLE
    assert "#0 → #1" in delta.detail
    assert "moved by 0.40000" in delta.detail, "the number as `fmt.measured` prints it"
    assert report.exit_code() == 0


def test_a_summary_field_this_version_does_not_know_is_not_a_change(tmp_path):
    """The summary is the producing tool's own block and its keys are open
    ended. Deriving a verdict from one would report a producer's version bump
    as a change in the model."""
    newer = _attribution(
        summary={
            "nodes": 4,
            "returned_edges": 3,
            "edge_limit": 2000,
            "some_future_statistic": 17,
        }
    )
    delta, _ = _attributed(tmp_path, _attribution(), newer)
    assert delta.status == mri_diff.SAME


def test_a_summary_the_reader_lets_through_does_not_crash_the_json(tmp_path):
    """`report.to_dict()` is dumped with `allow_nan=False`, and
    `session._graph` checks a summary value for finiteness only at the TOP
    level — a nested block still travels with whatever it holds. Copying the
    summary wholesale into `measured` would make `modelmri diff --json` end in
    a serialiser crash rather than in an answer, so only the fields the reader
    already checked go in."""
    nested = _attribution(
        summary={
            "nodes": 4,
            "edge_limit": 2000,
            "means": {"density": float("nan")},
        }
    )
    a, b = tmp_path / "a.mri", tmp_path / "b.mri"
    for path in (a, b):
        doc = _doc(_graphed(graph=_attribution()))
        doc["graph"] = nested
        path.write_bytes(_pack(doc))

    report = mri_diff.diff(a, b)
    assert _by_name(report)["attribution graph"].status == mri_diff.SAME
    json.dumps(report.to_dict(), allow_nan=False)


def test_a_non_finite_attribution_weight_never_reaches_the_diff(tmp_path):
    """The reader refuses it, so the differ has no NaN branch to get wrong —
    and `report.to_dict()` is serialised with `allow_nan=False`, where a NaN
    that got this far would be a crash rather than a wrong number."""
    a, b = tmp_path / "a.mri", tmp_path / "b.mri"
    a.write_bytes(_graphed(graph=_attribution()))
    doc = _doc(_graphed(graph=_attribution()))
    doc["graph"]["edges"][0]["weight"] = float("nan")
    b.write_bytes(_pack(doc))

    with pytest.raises(BadRequest, match="non-finite"):
        mri_diff.diff(a, b)


def test_both_graph_sections_reach_the_report_and_the_render(tmp_path):
    """The gap this closes: `diff` compared six sections and a `.mri` carries
    eight, so two runs with different circuits read as all-clear."""
    delta, report = _walked(tmp_path, _walk(), _walk())
    text = mri_diff.render(report, fail_over=0.05)
    assert "patching graph" in text
    assert "attribution graph" in text
    assert "recovery fraction" in text, "the threshold's units, named per metric"


def test_the_units_sentence_names_what_the_threshold_can_gate_and_nothing_else(
    tmp_path,
):
    """Read off this run's magnitude, the sentence selected almost exactly the
    wrong set. A SAME generation carries `magnitude=0.0`, so `--fail-over` was
    announced in "text units"; a patching graph whose verdict flipped carries
    None — the delta that produced the exit 1 — and its units were left out of
    the sentence explaining the threshold that failed it. Eligibility is a
    property of the section, not of how this run came out."""
    flipped = _walk(
        edges=[
            _pg_edge("L10 MLP@9", "L11 MLP@9", 0.25, clears_control=False),
            _pg_edge("L9H6@9", "L10 MLP@9", 0.25),
        ]
    )
    _, report = _walked(tmp_path, _walk(), flipped)
    text = mri_diff.render(report, fail_over=0.05)
    assert report.exit_code(fail_over=0.05) == 1
    assert "recovery fraction for patching graph" in text, "the unit that failed"
    assert "attention weight for attention" in text
    assert "text" not in text.split("own units")[1], "no threshold is set in text"

    # And the section whose every CHANGED is categorical never names its unit,
    # even when it is present and agreeing.
    a, b = tmp_path / "c.mri", tmp_path / "d.mri"
    raw = _graphed(_walk(), _attribution())
    a.write_bytes(raw)
    b.write_bytes(raw)
    text = mri_diff.render(mri_diff.diff(a, b), fail_over=0.05)
    assert "attribution weight" not in text
    assert "recovery fraction" in text


def test_the_new_sections_survive_json(tmp_path):
    a, b = tmp_path / "a.mri", tmp_path / "b.mri"
    raw = _graphed(_walk(), _attribution())
    a.write_bytes(raw)
    b.write_bytes(raw)
    report = mri_diff.diff(a, b)
    round_tripped = json.loads(json.dumps(report.to_dict(), allow_nan=False))
    assert {d["name"] for d in round_tripped["deltas"]} >= {
        "patching graph",
        "attribution graph",
    }
