"""A cross-episode sweep, and the two things it refuses to become.

Ranking over measured internals is the version of this nobody else can do —
RoboLab ranks over simulator-emitted event labels, Event-SAE clusters
kinematic keyframes, and neither holds the policy and the dataset in one
process. Which is exactly why the output has to be careful about what it
claims: a ranked table invites being read as a diagnosis.

So: no failure-mode names, and no silent stride.
"""

from __future__ import annotations

import json
import math

import pytest

from modelmri import vla_sweep as sw
from modelmri.errors import BadRequest


class Ep:
    def __init__(self, index, length):
        self.index, self.length = index, length


class FakeReader:
    repo_id = "lerobot/pusht"
    camera = "observation.images.top"

    def __init__(self, episodes=6, length=100, broken=()):
        self._eps = [Ep(i, length) for i in range(episodes)]
        self._broken = set(broken)

    def episodes(self):
        return self._eps

    def raw_frame(self, episode, t):
        if (episode, t) in self._broken:
            raise RuntimeError("this frame will not decode")
        return ("frame", episode, t)


class FakeHandle:
    """Returns a value driven by the frame, so a ranking has ground truth."""

    class _S:
        repo = "lerobot/smolvla_base"
        n_layers = 2

    def __init__(self, values=None):
        self.values = values or {}
        self._attn = [None, None]

    def status(self):
        return self._S()

    def analyse(self, rgb, key):
        self._frame = rgb
        return {}

    def attention(self, layer, head):
        _, episode, t = self._frame
        # A one-hot map has entropy 0; a flat one has entropy log(4). The
        # planted value decides which, so the ranking has a known answer.
        peak = self.values.get((episode, t), 0.0)
        flat = [1.0, 1.0, 1.0, 1.0]
        if peak:
            flat = [1.0, 0.0, 0.0, 0.0]
        grid = [flat[:2], flat[2:]]
        # BOTH, like the real method: `heat` stretched to [0,1] for drawing
        # and `values` raw. This double used to return the raw grid under the
        # name `heat`, which is why nothing here noticed that the real
        # `attention()` normalises it -- the flat case above min-max
        # normalises to all zeros, and the entropy metric read it as an empty
        # map rather than as the most spread-out frame possible.
        lo = min(v for row in grid for v in row)
        hi = max(v for row in grid for v in row)
        span = hi - lo
        heat = [
            [((v - lo) / span if span > 1e-12 else 0.0) for v in row] for row in grid
        ]
        return {"heat": heat, "values": grid, "min": lo, "max": hi}


# ------------------------------------------------------------- the planning


def test_the_plan_says_what_share_of_the_dataset_it_covers():
    pairs, total = sw.plan(FakeReader(), frame_stride=25)
    assert len(pairs) == 24
    assert total == 600
    out = sw.estimate(FakeReader(), "attention_entropy", frame_stride=25)
    assert out["coverage"] == pytest.approx(0.04)
    assert out["frames_total"] == 600


def test_the_expensive_metric_costs_what_it_costs():
    """The occlusion metric is dozens of tower passes per frame, and the
    estimate is the reason nobody discovers that by waiting."""
    cheap = sw.estimate(FakeReader(), "attention_entropy", frame_stride=25)
    dear = sw.estimate(FakeReader(), "occlusion_peak", frame_stride=25, grid=[32, 32])
    assert cheap["passes_per_frame"] == 1
    assert dear["passes_per_frame"] > 100
    assert dear["passes"] > cheap["passes"] * 100


def test_seconds_are_not_guessed_from_somebody_elses_hardware():
    """A duration is a number people plan around. Without a measurement from
    THIS machine there is no honest one to give."""
    out = sw.estimate(FakeReader(), "attention_entropy")
    assert out["seconds"] is None
    assert "somebody else's hardware" in out["seconds_from"]

    timed = sw.estimate(FakeReader(), "attention_entropy", seconds_per_pass=0.05)
    assert timed["seconds"] > 0
    assert timed["seconds_from"] == "measured on this machine"


def test_too_many_frames_is_refused_rather_than_truncated():
    """A ranking missing its tail looks exactly like a ranking, and you would
    have no way to tell."""
    with pytest.raises(BadRequest, match="Raise the stride"):
        sw.plan(FakeReader(episodes=200, length=1000), frame_stride=1, max_frames=100)


def test_the_refusal_names_the_frame_count_and_the_cap():
    with pytest.raises(BadRequest) as caught:
        sw.plan(FakeReader(episodes=50, length=100), frame_stride=1, max_frames=100)
    assert "5,000 frames" in str(caught.value)
    assert "cap is 100" in str(caught.value)


def test_a_zero_stride_is_refused():
    with pytest.raises(BadRequest, match="at least 1"):
        sw.plan(FakeReader(), frame_stride=0)


def test_an_unknown_metric_names_the_ones_it_has():
    with pytest.raises(BadRequest, match="attention_entropy"):
        sw.estimate(FakeReader(), "vibes")


def test_an_empty_dataset_is_refused():
    class Empty(FakeReader):
        def episodes(self):
            return []

    with pytest.raises(BadRequest, match="no episodes"):
        sw.plan(Empty())


# ---------------------------------------------------------------- the run


def test_the_ranking_finds_the_frames_that_were_planted():
    """Ground truth: three frames get a one-hot attention map (entropy 0) and
    the rest a flat one (entropy log 4). The ranking must put the flat ones on
    top, because this ranks by entropy and says so."""
    planted = {(1, 25): 1.0, (3, 50): 1.0, (5, 75): 1.0}
    out = sw.run(
        FakeHandle(planted), FakeReader(), "attention_entropy", frame_stride=25
    )
    assert out.n_frames == 24
    # Flat maps score log(4); the planted one-hot maps score 0.
    assert out.rows[0].value == pytest.approx(math.log(4))
    bottom = {(r.episode, r.timestep) for r in out.rows if r.value == 0.0}
    assert bottom == set(planted)


def test_a_frame_that_will_not_decode_is_absent_not_scored_zero():
    """A frame that failed to decode is not a frame with a low score, and a
    zero would sit at the bottom of the table looking like a measurement."""
    out = sw.run(
        FakeHandle(),
        FakeReader(broken=[(2, 0), (2, 25)]),
        "attention_entropy",
        frame_stride=25,
    )
    assert out.n_frames == 22
    assert len(out.failed) == 2
    assert all((r.episode, r.timestep) != (2, 0) for r in out.rows)
    assert "ABSENT from the ranking rather than scored zero" in out.means()


def test_a_machine_with_no_video_decoder_is_refused_not_measured():
    """A missing decoder is not a property of one frame. `av` is imported the
    first time a frame is actually decoded, so a machine without it fails
    EVERY frame identically — and the per-frame handler turned that into a
    completed run.

    MEASURED on a machine with pyarrow and no av: `POST /api/vla/sweep
    {"frame_stride": 1e12}` came back 200 with `rows: []`, a `failed` table of
    `why: "ModuleNotFoundError"`, and the summary "0 of 25650 frames (0.0%)
    across 0 episodes, measured by ATTENTION_ENTROPY" — a measurement of
    nothing, reported as a measurement. The route has carried the 409 naming
    the missing package all along; it was unreachable from inside the loop.
    """

    class NoDecoder(FakeReader):
        def raw_frame(self, episode, t):
            raise ModuleNotFoundError("No module named 'av'", name="av")

    with pytest.raises(ImportError) as caught:
        sw.run(FakeHandle(), NoDecoder(), "attention_entropy", frame_stride=25)
    assert caught.value.name == "av"


def test_one_broken_frame_still_does_not_take_down_the_sweep():
    """The half that must not move with it: a decode failure on SOME frames is
    still per-frame, and an ImportError raised by the METRIC rather than the
    decoder must not be mistaken for a missing decoder either — so this checks
    the ordinary failure path still ranks what it could read."""
    out = sw.run(
        FakeHandle(),
        FakeReader(broken=[(2, 0)]),
        "attention_entropy",
        frame_stride=25,
    )
    assert out.n_frames == 23
    assert out.n_failed == 1


def test_a_sweep_can_be_cancelled_and_still_reports_what_it_covered():
    """A partial sweep is still a ranking over what it covered, and it says
    how much that was."""
    seen = {"n": 0}

    def stop():
        seen["n"] += 1
        return seen["n"] > 5

    out = sw.run(
        FakeHandle(),
        FakeReader(),
        "attention_entropy",
        frame_stride=25,
        should_stop=stop,
    )
    assert out.n_frames == 5
    assert out.frames_total == 600
    assert "5 of 600 frames" in out.means()


def test_progress_is_reported_per_frame():
    calls = []
    sw.run(
        FakeHandle(),
        FakeReader(episodes=2),
        "attention_entropy",
        frame_stride=50,
        on_progress=lambda *a: calls.append(a),
    )
    assert len(calls) == 4
    assert calls[0][1] == 4  # total is passed so a bar can be drawn


# ------------------------------------------------- what it will not become


def test_the_stride_is_in_every_summary():
    """A strided ranking can miss the worst frame entirely. That is the first
    thing a reader needs, not a footnote."""
    out = sw.run(FakeHandle(), FakeReader(), "attention_entropy", frame_stride=25)
    means = out.means()
    assert "THE STRIDE IS 25 FRAMES" in means
    assert "worst frame THAT WAS SAMPLED" in means


def test_the_ranking_says_what_it_is_by_and_names_no_failure_mode():
    """A cluster labelled "dropped the object" that ModelMRI never verified is
    exactly the fabrication this project refuses."""
    out = sw.run(FakeHandle(), FakeReader(), "attention_entropy", frame_stride=25)
    means = out.means()
    assert "ranked by that and nothing else" in means
    assert "not a diagnosis" in means
    assert "no failure mode has been named" in means
    for invented in ("dropped", "failed", "collision", "success"):
        assert f"{invented} the" not in means.lower()


def test_the_unit_travels_with_the_ranking():
    out = sw.run(FakeHandle(), FakeReader(), "attention_entropy", frame_stride=25)
    assert out.unit == "nats over the patch grid"
    assert out.unit in out.means()


def test_the_occlusion_metric_says_it_is_perception_only():
    assert "perception only" in sw.METRICS["occlusion_peak"][1]


# ------------------------------------------------------------ the heat strip


def test_the_strip_is_ragged_rather_than_padded_with_zeros():
    """Episodes have different lengths, so padding to a rectangle would put
    zeros on screen that read as measured lows."""
    reader = FakeReader(episodes=3)
    reader._eps[1].length = 40  # a short episode
    out = sw.run(FakeHandle(), reader, "attention_entropy", frame_stride=25)
    strip = sw.heat_strip(out)
    assert strip["ragged"] is True
    widths = {len(row["values"]) for row in strip["rows"]}
    assert len(widths) > 1, "the short episode was padded to match the others"


def test_the_strip_carries_its_own_stride_and_range():
    out = sw.run(FakeHandle(), FakeReader(), "attention_entropy", frame_stride=25)
    strip = sw.heat_strip(out)
    assert strip["frame_stride"] == 25
    assert strip["low"] <= strip["high"]
    assert strip["unit"]


# --------------------------------------------------------------- storage


def test_a_sweep_is_findable_after_the_process_ends(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELMRI_HOME", str(tmp_path))
    out = sw.run(
        FakeHandle(), FakeReader(episodes=2), "attention_entropy", frame_stride=50
    )
    assert sw.save(out) == out.n_frames
    rows = sw.stored("lerobot/pusht", "lerobot/smolvla_base", "attention_entropy")
    assert len(rows) == out.n_frames
    assert rows[0]["value"] >= rows[-1]["value"]


def test_every_stored_row_carries_its_own_stride(tmp_path, monkeypatch):
    """Two runs at different strides land in the same table, and a row that
    did not carry its own stride would be indistinguishable from one taken at
    a finer step."""
    monkeypatch.setenv("MODELMRI_HOME", str(tmp_path))
    coarse = sw.run(
        FakeHandle(), FakeReader(episodes=2), "attention_entropy", frame_stride=50
    )
    sw.save(coarse)
    rows = sw.stored("lerobot/pusht", "lerobot/smolvla_base", "attention_entropy")
    assert all(r["stride"] == 50 for r in rows)


def test_re_running_replaces_rather_than_duplicates(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELMRI_HOME", str(tmp_path))
    out = sw.run(
        FakeHandle(), FakeReader(episodes=2), "attention_entropy", frame_stride=50
    )
    sw.save(out)
    sw.save(out)
    rows = sw.stored("lerobot/pusht", "lerobot/smolvla_base", "attention_entropy")
    assert len(rows) == out.n_frames


def test_the_report_survives_json():
    out = sw.run(
        FakeHandle(), FakeReader(episodes=2), "attention_entropy", frame_stride=50
    )
    doc = json.loads(json.dumps(out.to_dict(), allow_nan=False))
    assert doc["metric"] == "attention_entropy"
    assert "means" in doc
    assert len(doc["rows"]) == doc["n_frames"]


# ------------------- entropy of the attention, not of the picture of it


class UniformHandle(FakeHandle):
    """Every patch attended equally — the most spread-out frame there is."""

    def attention(self, layer, head):
        grid = [[0.25, 0.25], [0.25, 0.25]]
        # min == max, so the display heatmap is all zeros. That is correct for
        # drawing and catastrophic for a statistic.
        return {
            "heat": [[0.0, 0.0], [0.0, 0.0]],
            "values": grid,
            "min": 0.25,
            "max": 0.25,
        }


def test_a_uniform_attention_map_is_the_maximum_entropy_not_an_empty_one():
    """`heat` is min-max normalised, so a uniform map normalises to all zeros
    and the sweep raised "this frame produced an empty attention map" for the
    single most spread-out frame it could ever see."""
    out = sw.run(
        UniformHandle(), FakeReader(episodes=1), "attention_entropy", frame_stride=50
    )
    assert out.n_frames == 2
    assert out.failed == []
    assert out.rows[0].value == pytest.approx(math.log(4))


def test_the_entropy_reads_the_raw_grid_rather_than_the_display_heatmap():
    """Subtracting the frame's own minimum drives the least-attended patch to
    exactly zero probability, making every frame look more concentrated than
    it was — by an amount that varies per frame. This metric exists to RANK
    frames against each other, so a per-frame distortion is the one error it
    cannot absorb."""
    import inspect

    source = inspect.getsource(sw.attention_entropy)
    code = "\n".join(
        line.split("#", 1)[0] for line in source.splitlines() if line.strip()
    )
    assert '["values"]' in code
    assert '["heat"]' not in code


# ------------------------------------------- the cap is not the measurement


def test_the_number_of_failed_frames_is_not_the_length_of_the_listed_sample():
    """`failed` is truncated to `MAX_FAILED_LISTED` and `means()` counted it.

    MEASURED with PyAV absent over six episodes of a hundred frames: all 600
    frames failed to decode and the report read "20 frame(s) could not be
    measured". The true figure was not derivable from the payload at all —
    there was no field carrying it.
    """
    swept = sw.Sweep(
        metric="attention_entropy",
        unit="nats",
        dataset="someone/robot",
        policy="someone/policy",
        camera="observation.image",
        episode_stride=1,
        frame_stride=25,
        rows=[],
        n_frames=0,
        n_episodes=6,
        frames_total=600,
        failed=[{"episode": 0, "timestep": i, "why": "no decoder"} for i in range(20)],
        n_failed=600,
    )

    said = swept.means()

    assert "600 frame(s) could not be measured" in said, said
    assert "20 of them listed below" in said, "and say the list is a sample"


def test_a_failure_list_that_was_not_truncated_says_nothing_about_listing():
    """So the disclosure only appears when something was actually cut."""
    swept = sw.Sweep(
        metric="attention_entropy",
        unit="nats",
        dataset="d",
        policy="p",
        camera="c",
        episode_stride=1,
        frame_stride=1,
        rows=[],
        n_frames=0,
        n_episodes=1,
        frames_total=3,
        failed=[{"episode": 0, "timestep": 0, "why": "no decoder"}],
        n_failed=1,
    )

    said = swept.means()

    assert "1 frame(s) could not be measured" in said
    assert "listed below" not in said


def test_the_true_count_travels_on_the_wire():
    """`to_dict` is what the panel reads; a field only means() knows is lost."""
    swept = sw.Sweep(
        metric="m",
        unit="u",
        dataset="d",
        policy="p",
        camera="c",
        episode_stride=1,
        frame_stride=1,
        failed=[{"episode": 0, "timestep": 0, "why": "x"}],
        n_failed=97,
    )

    assert swept.to_dict()["n_failed"] == 97
