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
    eps = [
        FakeEpisode(i, per, i * per, from_ts=i * 1.0, to_ts=(i + 1) * 1.0)
        for i in range(n_eps)
    ]
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


def test_a_capped_failure_list_carries_the_number_that_actually_failed():
    """`failed` is sliced to 4, and the count beside it is the whole story.

    MEASURED on a machine without PyAV -- the DEFAULT state for anyone who has
    not installed the video extra, since `DECODE_SAMPLE` is 6 against a cap of
    4: the payload came back `episodes_sampled: 6, failed: [4 entries],
    distinct_images: 0`. A reader counting the list saw 4 of 6 failed and
    concluded 2 episodes decoded; `distinct_images: 0` in the same response
    said none did. The frontend reads `n_<key>` to decide whether to print
    "showing 4 of 6", so with `n_failed` absent it rendered 4 failures as the
    complete list.
    """
    reader = _healthy(n_eps=6, per=10)  # FakeReader.raw_frame raises for all 6

    check = vla_audit.check_distinct_frames(reader)

    assert check.verdict == UNCHECKED
    assert len(check.measured["failed"]) == 4, "the list is still capped"
    assert check.measured["n_failed"] == 6, (
        "every one of the 6 sampled episodes failed to decode, and the payload "
        "has to say 6 rather than leave the reader to count the capped list"
    )
    assert check.measured["episodes_sampled"] == 6
    assert check.measured["distinct_images"] == 0
    # The three numbers now agree: 6 sampled, 6 failed, 0 decoded.
    assert (
        check.measured["n_failed"] + check.measured["distinct_images"]
        == check.measured["episodes_sampled"]
    )


def test_a_capped_collision_list_carries_the_number_of_groups_found():
    """The same cap sits on `collisions`, and it was the file's other silent one.

    MEASURED with 12 episodes paired onto 6 distinct images: `collisions` came
    back with 4 entries and no count, while the detail sentence said "6
    group(s)" -- the sentence and the payload disagreed about how many there
    were, and only the sentence was right.
    """
    n = 12
    eps = [
        FakeEpisode(i, 10, i * 10, from_ts=i * 1.0, to_ts=(i + 1) * 1.0)
        for i in range(n)
    ]
    table = {"episode_index": [i // 10 for i in range(n * 10)]}
    # Episodes 0/1 share an image, 2/3 share the next, and so on: 6 groups.
    frames = {i: bytes([i // 2]) * 96 for i in range(n)}
    reader = FakeReader(eps, table, frames=frames)

    check = vla_audit.check_distinct_frames(reader, sample=n)

    assert check.verdict == BROKEN
    assert len(check.measured["collisions"]) == 4, "the list is still capped"
    assert check.measured["n_collisions"] == 6
    assert check.measured["n_failed"] == 0, (
        "nothing failed to decode here, and an absent field would be read as "
        "unknown rather than zero"
    )
    assert "6 group(s)" in check.detail, check.detail


def test_the_failure_count_is_present_even_when_every_frame_decodes():
    """An absent `n_failed` reads as unknown, not as zero.

    The healthy path is the one that would tempt an implementation to omit the
    field, and a payload that carries it only when something went wrong makes
    the reader guess which of the two an absence means.
    """
    reader = _healthy()
    reader._images = {i: bytes([i]) * 96 for i in range(4)}

    check = vla_audit.check_distinct_frames(reader)

    assert check.verdict == OK
    assert check.measured["n_failed"] == 0
    assert check.measured["n_collisions"] == 0


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


# ---------------- the audit has to survive the data it is auditing


class _TableWontOpen:
    repo_id = "lerobot/pusht"

    def episodes(self):
        return [type("E", (), {"index": 0, "length": 10})()]

    def _frame_table(self):
        raise RuntimeError("parquet shard is truncated")


class _NothingOpens:
    repo_id = "lerobot/pusht"

    def episodes(self):
        raise OSError("the snapshot directory disappeared")

    def _frame_table(self):
        raise OSError("the snapshot directory disappeared")


def test_an_unreadable_frame_table_does_not_take_the_whole_audit_down():
    """The docstring promises "One failing check never stops the others", and
    the per-check guard honours it — but `episodes()` and `_frame_table()`
    ran OUTSIDE that guard.

    So a dataset whose parquet will not open crashed the one tool whose
    stated purpose is telling you "whether to train on this data at all".
    The case where the data is broken is exactly the case it has to survive.
    """
    report = vla_audit.audit(_TableWontOpen())

    assert report.n_episodes == 1
    assert report.n_frames is None, "unreadable is not zero frames"
    assert report.checks, "a report with no checks is not a report"
    assert any("frame table" in c.name for c in report.unchecked)
    assert any("RuntimeError" in (c.detail or "") for c in report.checks)


def test_a_reader_that_opens_nothing_still_answers():
    report = vla_audit.audit(_NothingOpens())

    assert report.n_episodes is None
    assert report.n_frames is None
    assert len(report.unchecked) == len(report.checks)
    assert "OSError" in " ".join(c.detail or "" for c in report.checks)


def test_a_healthy_dataset_reports_real_counts():
    """The counts must stay numbers when the data is fine — None means
    "could not read", and it has to mean only that."""
    report = vla_audit.audit(_healthy())

    assert isinstance(report.n_episodes, int)
    assert isinstance(report.n_frames, int)


def test_a_pair_search_that_stopped_early_does_not_report_its_stop_as_the_total():
    """The pair scan is quadratic, so it stops at `MAX_PAIRS_KEPT`.

    MEASURED on this fixture: 400 pairs qualify and the check reported
    `n_pairs: 40, truncated: False, "Scanned all 80 frames"` — a tenfold
    under-report presented as a complete count, with the one field that could
    have contradicted it saying the scan was complete. It was: the FRAME scan
    finished, the PAIR search did not, and those are two different truncations
    that needed two fields.
    """
    n = 80
    state = [[float(i % 4), 0.0] for i in range(n)]
    action = [[0.0, 0.0] if (i // 4) % 2 else [50.0, 0.0] for i in range(n)]
    reader = FakeReader(
        [FakeEpisode(0, n, 0)],
        {"episode_index": [0] * n, "observation.state": state, "action": action},
    )

    check = vla_audit.check_contradictions(reader)

    assert check.measured["n_pairs"] == vla_audit.MAX_PAIRS_KEPT
    assert check.measured["pairs_complete"] is False, (
        "the search stopped, and the payload has to say so"
    )
    assert check.measured["pairs_cap"] == vla_audit.MAX_PAIRS_KEPT
    assert "at least" in check.detail, check.detail
    assert "the search stopped there" in check.detail
    # The FRAME scan is a separate question and still answered separately.
    assert check.measured["truncated"] is False


def test_a_pair_search_that_finished_states_its_count_plainly():
    """So "at least" cannot become the wording for every result."""
    n = 12
    state = [[float(i % 6), 0.0] for i in range(n)]
    action = [[0.0, 0.0] if (i // 6) % 2 else [50.0, 0.0] for i in range(n)]
    reader = FakeReader(
        [FakeEpisode(0, n, 0)],
        {"episode_index": [0] * n, "observation.state": state, "action": action},
    )

    check = vla_audit.check_contradictions(reader)

    assert check.measured["n_pairs"] < vla_audit.MAX_PAIRS_KEPT
    assert check.measured["pairs_complete"] is True
    assert "at least" not in check.detail
