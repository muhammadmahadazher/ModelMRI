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
