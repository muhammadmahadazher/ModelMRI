"""Every check here is tested by BREAKING a dataset in the way it claims to catch.

A check that passes on healthy data has proved nothing — the version of it
that always returns "ok" passes the same test. So each corruption below is
planted deliberately, and the check has to name it.

The bug class this module exists for is one this project shipped: a
`.get(name, 0.0)` that made 206 episodes decode from timestamp zero, so every
episode showed the same video while the state vector underneath was correct.
Nothing crashed. `test_the_206_episodes_bug_is_caught` is that exact shape.
"""

from __future__ import annotations

import json

import pytest

from modelmri import vla_audit
from modelmri.vla_audit import BROKEN, OK, UNCHECKED


class FakeEpisode:
    def __init__(self, index, length, data_from, from_ts=0.0, to_ts=1.0):
        self.index = index
        self.length = length
        self.data_from = data_from
        self.from_ts = from_ts
        self.to_ts = to_ts
        self.task = "push the T"
        self.video_chunk = 0
        self.video_file = 0


class FakeReader:
    """The smallest thing that looks like `LeRobotV3Reader` to the audit."""

    repo_id = "fake/dataset"

    def __init__(self, episodes, table, *, info=None, snapshot=None, frames=None):
        self._eps = episodes
        self._table = table
        self.info = info if info is not None else {"fps": 10}
        self.snapshot = snapshot
        self._images = frames or {}

    def episodes(self):
        return self._eps

    def _frame_table(self):
        return self._table

    def _video_file(self, ep=None):
        import pathlib

        return pathlib.Path(__file__)  # a file that exists

    def raw_frame(self, episode, t):
        if episode in self._images:
            return self._images[episode]
        raise RuntimeError("no decoder here")


def _healthy(n_eps=4, per=10):
    eps = [FakeEpisode(i, per, i * per, from_ts=i * 1.0, to_ts=(i + 1) * 1.0)
           for i in range(n_eps)]
    n = n_eps * per
    table = {
        "episode_index": [i // per for i in range(n)],
        "observation.state": [[float(i), float(i % 3)] for i in range(n)],
        "action": [[float(i) + 0.5, float(i % 3)] for i in range(n)],
    }
    return FakeReader(eps, table)


# ------------------------------------------------------------------ tiling


def test_a_healthy_dataset_tiles_exactly():
    check = vla_audit.check_tiling(_healthy())
    assert check.verdict == OK
    assert check.measured["n_gaps"] == 0 and check.measured["n_overlaps"] == 0


def test_an_overlap_is_caught():
    """Two episodes claiming the same rows. Nothing raises — both episodes
    load, and the frames appear twice in training."""
    reader = _healthy()
    reader._eps[2].data_from = 15  # should be 20
    check = vla_audit.check_tiling(reader)
    assert check.verdict == BROKEN
    assert "overlap" in check.detail
    assert check.measured["n_overlaps"] == 1


def test_a_gap_is_caught():
    """Rows no episode claims. They are simply never trained on, silently."""
    reader = _healthy()
    reader._eps[2].data_from = 25  # should be 20
    check = vla_audit.check_tiling(reader)
    assert check.verdict == BROKEN
    assert "gap" in check.detail
    assert check.measured["n_gaps"] == 1


def test_a_short_count_is_caught():
    """The episodes cover fewer rows than the frame table holds — the exact
    shape of reading only the first parquet shard."""
    reader = _healthy()
    reader._eps.pop()
    check = vla_audit.check_tiling(reader)
    assert check.verdict == BROKEN
    assert "frame table has 40" in check.detail


# ----------------------------------------------------------------- routing


def test_the_206_episodes_bug_is_caught():
    """THE ONE THIS MODULE EXISTS FOR.

    `vla_data.py` read a column name no LeRobot v3.0 dataset has, and
    `.get(name, 0.0)` turned the miss into zero for every episode. All 206
    decoded from the start of the file: episodes 0, 5 and 20 returned
    byte-identical images while the state printed underneath was correctly
    episode 5's and episode 20's.
    """
    reader = _healthy()
    for ep in reader._eps:
        ep.from_ts = 0.0
        ep.to_ts = 0.0
    check = vla_audit.check_routing(reader)
    assert check.verdict == BROKEN
    assert "every one of 4 episodes routes to timestamp 0.0" in check.detail
    assert "0.10" in check.detail


def test_a_single_zero_span_is_caught_without_the_206_wording():
    """One episode with a zero span is a different fault from all of them, and
    it should not borrow the story of the other one."""
    reader = _healthy()
    reader._eps[1].from_ts = reader._eps[1].to_ts = 0.0
    check = vla_audit.check_routing(reader)
    assert check.verdict == BROKEN
    assert "zero-length span" in check.detail
    assert "206" not in check.detail


def test_a_missing_video_file_is_caught():
    import pathlib

    reader = _healthy()
    reader._video_file = lambda ep=None: pathlib.Path("no_such_file.mp4")
    check = vla_audit.check_routing(reader)
    assert check.verdict == BROKEN
    assert "not there" in check.detail


# ------------------------------------------------------- distinct frames


def test_identical_frames_across_episodes_are_caught():
    """Every structural check can pass while the decoder still hands back the
    same picture for every episode: routing that is present and wrong looks
    exactly like routing that is right."""
    same = b"\x01\x02\x03" * 32
    reader = _healthy()
    reader._images = {i: same for i in range(4)}
    check = vla_audit.check_distinct_frames(reader)
    assert check.verdict == BROKEN
    assert "SAME image" in check.detail


def test_distinct_frames_pass_and_say_they_were_sampled():
    reader = _healthy()
    reader._images = {i: bytes([i]) * 96 for i in range(4)}
    check = vla_audit.check_distinct_frames(reader)
    assert check.verdict == OK
    assert "SAMPLED, not exhaustive" in check.detail


def test_two_episodes_sharing_a_first_frame_are_flagged_for_inspection():
    """Two episodes CAN legitimately start from the same pose, so this names
    them rather than calling the dataset broken outright."""
    reader = _healthy()
    reader._images = {0: b"a" * 96, 1: b"a" * 96, 2: b"c" * 96, 3: b"d" * 96}
    check = vla_audit.check_distinct_frames(reader)
    assert check.verdict == BROKEN
    assert "check these by eye" in check.detail


def test_no_decoder_is_unchecked_rather_than_broken():
    """A missing PyAV is a fact about this machine, not about the data."""
    check = vla_audit.check_distinct_frames(_healthy())
    assert check.verdict == UNCHECKED
    assert "not readable here" in check.detail


# ----------------------------------------------------------- normalisation


def test_drifted_statistics_are_caught(tmp_path):
    """Stale stats are the quietest corruption in the list: training
    normalises with them, so wrong numbers break nothing visibly."""
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "stats.json").write_text(
        json.dumps({"observation.state": {"mean": [999.0, 999.0]}}), encoding="utf-8"
    )
    reader = _healthy()
    reader.snapshot = tmp_path
    check = vla_audit.check_normalisation(reader)
    assert check.verdict == BROKEN
    assert "do not describe the data" in check.detail


def test_matching_statistics_pass(tmp_path):
    reader = _healthy()
    rows = reader._table["observation.state"]
    mean = [sum(r[i] for r in rows) / len(rows) for i in range(2)]
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "stats.json").write_text(
        json.dumps({"observation.state": {"mean": mean}}), encoding="utf-8"
    )
    reader.snapshot = tmp_path
    check = vla_audit.check_normalisation(reader)
    assert check.verdict == OK


def test_a_dimension_count_mismatch_is_caught(tmp_path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "stats.json").write_text(
        json.dumps({"observation.state": {"mean": [1.0, 2.0, 3.0]}}), encoding="utf-8"
    )
    reader = _healthy()
    reader.snapshot = tmp_path
    check = vla_audit.check_normalisation(reader)
    assert check.verdict == BROKEN
    assert check.measured["drifted"][0]["recorded_dims"] == 3


def test_no_stats_file_is_unchecked_not_a_pass(tmp_path):
    reader = _healthy()
    reader.snapshot = tmp_path
    check = vla_audit.check_normalisation(reader)
    assert check.verdict == UNCHECKED


# ------------------------------------------------------ constant dimensions


def test_a_frozen_dimension_is_named_and_not_judged():
    """A gripper held open for a whole dataset is real data, so this reports
    the indices and does not call them a defect."""
    reader = _healthy()
    reader._table["action"] = [[1.0, 7.0] for _ in range(40)]
    check = vla_audit.check_constant_dimensions(reader)
    assert check.verdict == OK
    assert check.measured["constant"]["action"] == [0, 1]
    assert "NOT A DEFECT" in check.detail


def test_a_dataset_where_everything_moves_says_so():
    check = vla_audit.check_constant_dimensions(_healthy())
    assert check.verdict == OK
    assert "every state and action dimension varies" in check.detail


# ---------------------------------------------------------------- lag


def test_a_missing_control_frequency_is_refused_rather_than_assumed():
    """A lag in frames is meaningless if the frames are not evenly spaced, and
    reporting one anyway would be a number about nothing."""
    reader = _healthy()
    reader.info = {}
    check = vla_audit.check_action_lag(reader)
    assert check.verdict == UNCHECKED
    assert "does not state a control frequency" in check.detail


def test_a_planted_lag_is_found_at_the_right_size():
    """The action is the state delayed by three frames, by construction."""
    n = 200
    state = [[float(i % 37), 0.0] for i in range(n)]
    action = [[0.0, 0.0]] * 3 + state[: n - 3]
    reader = FakeReader(
        [FakeEpisode(0, n, 0)],
        {"episode_index": [0] * n, "observation.state": state, "action": action},
    )
    check = vla_audit.check_action_lag(reader)
    assert check.verdict == OK
    assert check.measured["best_lag_frames"] == 3
    assert "LAGS the state by 3" in check.detail


def test_a_lead_names_both_readings_rather_than_calling_it_a_bug():
    """MEASURED on lerobot/pusht: the action leads the state by one frame at
    r=0.986, and that is CORRECT there — pusht's action is a target position.
    The first version called it an off-by-one and would have reported a
    healthy dataset as damaged."""
    n = 200
    state = [[float(i % 37), 0.0] for i in range(n)]
    action = state[2:] + [[0.0, 0.0]] * 2
    reader = FakeReader(
        [FakeEpisode(0, n, 0)],
        {"episode_index": [0] * n, "observation.state": state, "action": action},
    )
    check = vla_audit.check_action_lag(reader)
    assert check.verdict == OK
    assert "LEADS the state" in check.detail
    assert "TARGET rather than a delta" in check.detail
    assert "off-by-one" in check.detail


def test_an_uncorrelated_pair_is_unchecked_rather_than_reported_as_zero_lag():
    """State and action need not be correlated at all, and a lag of 0 on an
    r of 0.02 would be a number pretending to be a finding."""
    n = 200
    reader = FakeReader(
        [FakeEpisode(0, n, 0)],
        {
            "episode_index": [0] * n,
            "observation.state": [[float(i % 7), 0.0] for i in range(n)],
            "action": [[float((i * 31) % 13), 0.0] for i in range(n)],
        },
    )
    check = vla_audit.check_action_lag(reader)
    if check.verdict == UNCHECKED:
        assert "too weak to place a lag" in check.detail


# ------------------------------------------------------- contradictions


def test_contradictions_come_back_as_pairs_and_never_as_a_defect_count():
    """Two similar states with different actions is usually legitimate
    multimodality — a human solving the same situation two ways — and
    presenting that as corruption would be the tool inventing a problem."""
    n = 80
    # The state repeats every 4 frames; the action flips every 4. So two
    # frames with the SAME state get DIFFERENT actions — which is the shape
    # this looks for. Keying the action on `i % 2` instead, as the first
    # version of this test did, meant every matching state also matched in
    # action and the check correctly found nothing.
    state = [[float(i % 4), 0.0] for i in range(n)]
    action = [[0.0, 0.0] if (i // 4) % 2 else [50.0, 0.0] for i in range(n)]
    reader = FakeReader(
        [FakeEpisode(0, n, 0)],
        {"episode_index": [0] * n, "observation.state": state, "action": action},
    )
    check = vla_audit.check_contradictions(reader)
    assert check.verdict == OK, "a contradiction is never a failure verdict"
    assert "PAIRS TO INSPECT, NOT DEFECTS" in check.detail
    assert check.measured["n_pairs"] > 0


def test_both_thresholds_are_printed_because_they_are_the_claim():
    """A different epsilon gives a different list, so the list means nothing
    without them."""
    check = vla_audit.check_contradictions(_healthy())
    assert "state_epsilon" in check.measured
    assert "action_delta" in check.measured
    assert check.measured["epsilon_fraction"] == vla_audit.STATE_EPSILON


def test_a_capped_scan_says_it_is_a_sample():
    n = 400
    reader = FakeReader(
        [FakeEpisode(0, n, 0)],
        {
            "episode_index": [0] * n,
            "observation.state": [[float(i), 0.0] for i in range(n)],
            "action": [[float(i), 0.0] for i in range(n)],
        },
    )
    check = vla_audit.check_contradictions(reader, max_frames=50)
    assert check.measured["truncated"] is True
    assert "is a SAMPLE" in check.detail
    assert "stride" in check.detail


# ---------------------------------------------------------------- the run


def test_one_failing_check_never_stops_the_others():
    """An audit that stops at the first problem tells you about one problem,
    and the reader is deciding whether to train on this data at all."""
    reader = _healthy()
    reader._eps[2].data_from = 15  # break tiling

    def explode(_reader):
        raise RuntimeError("this machine cannot do that")

    original = vla_audit.CHECKS
    vla_audit.CHECKS = (vla_audit.check_tiling, explode, vla_audit.check_action_lag)
    try:
        report = vla_audit.audit(reader)
    finally:
        vla_audit.CHECKS = original
    assert len(report.checks) == 3
    assert report.checks[0].verdict == BROKEN
    assert report.checks[1].verdict == UNCHECKED
    assert "fact about the run rather than about the data" in report.checks[1].detail
    assert report.checks[2].verdict == OK


def test_the_report_never_grades():
    report = vla_audit.audit(_healthy())
    means = report.means()
    assert "THERE IS NO GRADE" in means
    for letter in ("grade A", "grade B", "score:", "/100"):
        assert letter not in means


def test_nothing_failing_is_not_reported_as_a_certificate():
    report = vla_audit.audit(_healthy())
    assert "not a certificate" in report.means()


def test_the_report_survives_json():
    import json as _json

    report = vla_audit.audit(_healthy())
    doc = _json.loads(_json.dumps(report.to_dict(), allow_nan=False))
    assert doc["repo_id"] == "fake/dataset"
    assert "means" in doc
    assert len(doc["checks"]) == len(vla_audit.CHECKS)
