"""A measurement leaving ModelMRI for somebody else's viewer.

Every test here guards the same worry from a different side. Once a number is
in Foxglove it is beside numbers the robot recorded, in an application that
knows nothing about this tool, in front of a reader who cannot ask us anything.
So the file has to answer three questions on its own:

  * who measured this, and on what — `modelmri/circuit.py:8-11` states the rule
    for a graph arriving; it applies going out.
  * in what unit, at what resolution — `modelmri/vla_actions.py:124` refuses to
    put two unlabelled axes on one chart, and a foreign viewer is that case
    with the labels one application further away.
  * what is MISSING — a strided sweep and a decode failure draw the identical
    gap in a plot, and neither of them is a zero.

The writers are the reason the record sequence is built as data. Neither
`mcap` nor `rerun` is installed on this machine, so `mcap_records` is checked
exhaustively and the replay loop is checked against a recording stand-in that
is a CALL RECORDER, not an MCAP validator: these tests prove what this module
hands a writer, and prove nothing about the bytes a real `mcap` produces.
"""

from __future__ import annotations

import json
import sys
import types

import pytest

from modelmri import robot_export as rx
from modelmri.errors import BadRequest, Refusal

# ------------------------------------------------------------------ fixtures


class FakeRow:
    """A `vla_sweep.Row` by shape. `test_the_real_sweep_still_fits` checks the
    shape against the real class, so these stay honest as that file changes."""

    def __init__(self, episode, timestep, value):
        self.episode, self.timestep, self.value = episode, timestep, value


class FakeSweep:
    metric = "attention_entropy"
    unit = "nats over the patch grid"
    dataset = "lerobot/pusht"
    policy = "lerobot/smolvla_base"
    camera = "observation.images.top"
    episode_stride = 1
    frame_stride = 25
    n_frames = 8
    frames_total = 200
    n_failed = 0

    def __init__(self, rows=None, **overrides):
        self.rows = rows if rows is not None else _default_rows()
        for key, value in overrides.items():
            setattr(self, key, value)


# Two episodes, four sampled frames each at stride 25, in nats under a
# ln(16) = 2.7726 ceiling. Every value is distinct and NONE of them is
# monotonic in time: that is the whole point of the table. An earlier fixture
# used `4.0 - index - episode * 0.5`, which falls as the timestep rises, so
# ranking by value descending already left each episode's frames in ascending
# time order and `test_rows_are_ordered_by_time_and_not_by_rank` could not
# fail — deleting the sort from the module changed nothing. These values
# scramble it, and that test now asserts the scramble itself so the fixture
# cannot quietly go flat again.
_VALUES = {
    0: {0: 2.7014, 25: 2.6132, 50: 2.4415, 75: 2.5188},
    1: {0: 2.6640, 25: 2.7331, 50: 2.5907, 75: 2.4772},
}

#: The stated resolution behind the unit-overhead figure in the module
#: docstring. It lives here rather than in the prose because the number the
#: docstring publishes moves with the length of this string, and a measured
#: number whose inputs are not written down cannot be re-measured by anyone.
_CEILING = "entropy over a 16-patch grid, so a value is bounded by ln(16) = 2.7726 nats"


def _default_rows():
    # In DESCENDING value order, which is the order `vla_sweep.run` ranks in.
    rows = [
        FakeRow(episode, timestep, value)
        for episode, series in _VALUES.items()
        for timestep, value in series.items()
    ]
    rows.sort(key=lambda r: -r.value)
    return rows


def _timeline(**kwargs):
    kwargs.setdefault("fps", 10.0)
    kwargs.setdefault("tool_version", "0.11.0")
    sweep = kwargs.pop("sweep", None) or FakeSweep()
    return rx.timeline_from_sweep(sweep, **kwargs)


def _one_track_timeline(unit="u", resolution="r", samples=1):
    return rx.Timeline(
        provenance=rx.Provenance(
            tool="ModelMRI",
            tool_version="0.11.0",
            dataset="lerobot/pusht",
            camera="top",
            policy=None,
            policy_revision=None,
            measured_by="a test",
            taken_at="2026-08-24T00:00:00+00:00",
        ),
        clock=rx.clock_for(None),
        tracks=(
            rx.Track(
                metric="m",
                unit=unit,
                resolution=resolution,
                episode=0,
                samples=tuple(rx.Sample(t, 1.0) for t in range(samples)),
                frame_stride=1,
            ),
        ),
    )


# --------------------------------------------------- the numbers themselves


def test_every_exported_number_is_the_one_that_was_measured():
    """The claim the whole module rests on, checked against a literal table.

    Everything else here checks a sentence, a guard or a shape, and a suite of
    those cannot tell an export that carries the measurements from one that
    writes `"value": 0.0, "timestep": 0` in every body — a file of zeros
    stamped at frame 0 satisfies every other test in this file. So the eight
    bodies are compared against the fixture's values written out by hand, at
    full precision: `round(value, 1)` on a module that publishes a sentence
    about how many of these digits are real has to fail here too."""
    timeline = _timeline()
    bodies = _messages(timeline)
    assert len(bodies) == 8

    measured = {(b["episode"], b["timestep"]): b["value"] for b in bodies}
    assert measured == {
        (0, 0): 2.7014,
        (0, 25): 2.6132,
        (0, 50): 2.4415,
        (0, 75): 2.5188,
        (1, 0): 2.6640,
        (1, 25): 2.7331,
        (1, 50): 2.5907,
        (1, 75): 2.4772,
    }

    # Byte-for-byte for one body, so a change to the key set, the separators or
    # the float formatting has to be deliberate. This is what lands in the file.
    first = next(
        r
        for r in rx.mcap_records(timeline)
        if r.kind == "message" and json.loads(r.data)["timestep"] == 25
    )
    assert first.data == (
        b'{"episode":0,"frame_stride":25,"metric":"attention_entropy",'
        b'"resolution":"' + rx.RESOLUTION_UNSTATED.encode() + b'",'
        b'"timestep":25,"unit":"nats over the patch grid","value":2.6132}'
    )


def test_every_timestamp_is_the_frame_index_this_clock_maps_it_to():
    """A stamp of 0 on every message is a plot of eight points on top of each
    other, and it passes any test that only counts messages. At 10 fps the
    four sampled frames are 0.0, 2.5, 5.0 and 7.5 seconds — written out rather
    than recomputed from the clock, because `stamp(t)` re-derived here would
    agree with itself whatever it did."""
    pairs = _stamped(_timeline())
    assert {(b["episode"], b["timestep"]): ns for ns, b in pairs} == {
        (0, 0): 0,
        (0, 25): 2_500_000_000,
        (0, 50): 5_000_000_000,
        (0, 75): 7_500_000_000,
        (1, 0): 0,
        (1, 25): 2_500_000_000,
        (1, 50): 5_000_000_000,
        (1, 75): 7_500_000_000,
    }


def test_a_row_this_module_cannot_read_is_refused_not_exported_as_a_zero():
    """`vla_sweep.stored()` returns DICTS and `Sweep.to_dict()` uses `asdict`,
    so a sweep round-tripped through sqlite or JSON arrives with rows that
    answer no attribute at all. Read with `getattr(row, "value", 0.0)` that
    exports a file of zeros stamped at timestep 0 — the "ABSENT rather than
    scored zero" claim inverted, in the reader's application, with no refusal
    anywhere. Mapping rows are read by key; a row that can supply neither
    stops the export and names the row."""
    rows = [
        {"episode": 1, "timestep": 25, "value": 2.7},
        {"episode": 1, "timestep": 50, "value": 0.4},
    ]
    timeline = _timeline(sweep=FakeSweep(rows=rows))
    assert [t.episode for t in timeline.tracks] == [1]
    assert timeline.tracks[0].samples == (rx.Sample(25, 2.7), rx.Sample(50, 0.4))

    for missing in ("value", "timestep", "episode"):
        broken = {k: v for k, v in rows[0].items() if k != missing}
        with pytest.raises(BadRequest) as err:
            _timeline(sweep=FakeSweep(rows=[broken]))
        assert f"row 0 of this sweep has no {missing}" in err.value.sentence

    # An object that answers `episode` and `timestep` but not `value` is the
    # same failure with a different shape, and it used to produce 0.0.
    class Half:
        episode, timestep = 0, 25

    with pytest.raises(BadRequest) as err:
        _timeline(sweep=FakeSweep(rows=[Half()]))
    assert "row 0 of this sweep has no value" in err.value.sentence
    assert "no such attribute" in err.value.sentence

    # And a value that is present but unreadable is not a zero either.
    with pytest.raises(BadRequest) as err:
        _timeline(sweep=FakeSweep(rows=[{**rows[0], "value": "n/a"}]))
    assert "not a number" in err.value.sentence


# ------------------------------------------------------- unit and resolution


def test_a_track_with_no_unit_is_refused():
    """The rule `vla_actions.units_agree` enforces for a chart, enforced for a
    file that outlives the chart. Without this a bare float lands in Foxglove
    beside the robot's own numbers with nothing to say it is in nats — and the
    plot renders identically either way, which is what makes it dangerous."""
    with pytest.raises(Refusal) as err:
        _one_track_timeline(unit="")
    assert "no unit" in err.value.sentence
    assert "vla_actions" in err.value.sentence


def test_a_missing_resolution_must_be_stated_rather_than_left_out():
    """Unknown and absent are different facts, and a blank field is neither.
    Without the sentinel a reader who finds no resolution cannot tell that the
    measurement never published one from the exporter having dropped it, so an
    empty string is refused and `RESOLUTION_UNSTATED` is accepted and written."""
    with pytest.raises(Refusal) as err:
        _one_track_timeline(resolution="   ")
    assert "RESOLUTION_UNSTATED" in err.value.sentence

    timeline = _one_track_timeline(resolution=rx.RESOLUTION_UNSTATED)
    body = _messages(timeline)[0]
    assert body["resolution"] == rx.RESOLUTION_UNSTATED
    assert rx.plan(timeline).tracks_without_resolution == 1


def test_unit_and_resolution_ride_on_every_message_not_only_the_channel():
    """The person about to misread the number is hovering a point, and a
    viewer's hover shows the message body. Channel metadata alone put the unit
    two clicks away from the only moment it matters."""
    timeline = _timeline()
    bodies = _messages(timeline)
    assert len(bodies) == 8
    assert all(b["unit"] == "nats over the patch grid" for b in bodies)
    assert all(b["resolution"] == rx.RESOLUTION_UNSTATED for b in bodies)

    # And the schema declares them required, so the redundancy is machine
    # readable rather than a habit of this writer.
    schema = json.loads(_record(timeline, "schema", "modelmri.Measurement").data)
    assert "unit" in schema["required"]
    assert "resolution" in schema["required"]


def test_the_cost_of_that_redundancy_is_measured_exactly():
    """A deliberate trade with an unmeasured cost is a preference. The two
    strings add exactly `,"resolution":"r"` (17 bytes) and `,"unit":"u"` (11)
    to each body — 28 for these values — and it scales with the sample count,
    which is what makes it worth reporting on a long sweep."""
    assert rx.unit_overhead_bytes(_one_track_timeline(samples=1)) == 28
    assert rx.unit_overhead_bytes(_one_track_timeline(samples=3)) == 84


# ------------------------------------------------------------------ the gaps


def test_the_stride_travels_so_a_gap_is_never_read_as_a_zero():
    """Four points over a hundred frames is the same picture whether the other
    ninety-six were skipped or failed. Nothing in a plot distinguishes them, so
    the stride is on the channel, in every message, and in the omissions."""
    timeline = _timeline()
    channel = _record(timeline, "channel", topic="modelmri/attention_entropy/episode_0")
    assert channel.metadata["frame_stride"] == "25"
    assert "ABSENT" in channel.metadata["sampling"]
    assert all(b["frame_stride"] == 25 for b in _messages(timeline))
    assert any("one frame in 25" in s for s in timeline.omitted)


def test_the_arithmetic_in_the_omission_sentence_is_the_arithmetic_of_the_stride():
    """This sentence is one of the few things the file says for itself, and
    the count in it was unpinned: `{stride - 1} of every {stride}` could be
    written `{stride + 7}` with every other test green, so an exported file
    would tell a stranger "The other 32 of every 25 are ABSENT". Held here
    against numbers written out by hand at two strides — a second stride,
    because one can be matched by an expression that is wrong everywhere
    else."""
    said = " ".join(rx.omissions_for(sweep=FakeSweep()))
    assert "Only one frame in 25 was measured." in said
    assert "The other 24 of every 25 are ABSENT from this file" in said

    said = " ".join(rx.omissions_for(sweep=FakeSweep(frame_stride=4)))
    assert "Only one frame in 4 was measured." in said
    assert "The other 3 of every 4 are ABSENT from this file" in said

    # At stride 1 nothing was skipped and the sentence must not appear at all,
    # because "the other 0 of every 1" is a caveat about nothing.
    said = " ".join(rx.omissions_for(sweep=FakeSweep(frame_stride=1)))
    assert "ABSENT from this file" not in said


def test_a_skipped_episode_is_named_in_the_file_the_way_a_skipped_frame_is():
    """`vla_sweep.Sweep.means()` carries both halves of the caveat — "THE
    STRIDE IS 25 FRAMES (and every 5 episode)" — and this export used to drop
    the second one. A file that names the frame gap and not the episode gap
    tells a reader who counts its topics that the dataset has that many
    episodes, which is the same mistake one level up."""
    said = " ".join(rx.omissions_for(sweep=FakeSweep(episode_stride=5)))
    assert "Only one episode in 5 was opened." in said
    assert "The other 4 of every 5 episodes were never measured" in said

    # And at stride 1 no episode was skipped, so there is nothing to say.
    assert not any(
        "episode in" in s for s in rx.omissions_for(sweep=FakeSweep(episode_stride=1))
    )


def test_an_unknown_coverage_is_stated_rather_than_left_out():
    """A sweep that published no frame total has no coverage fraction, and the
    sentence used to just vanish. Absence of the caveat reads as absence of
    the caveat's subject: the reader is left to assume the file covers what
    they were looking at."""
    said = " ".join(rx.omissions_for(sweep=FakeSweep(frames_total=None)))
    assert "What share of the dataset this covers is NOT stated" in said
    assert "published no frame total" in said
    assert "%" not in said


def test_frames_that_failed_to_decode_are_named_as_absent_not_as_low_scores():
    """`vla_sweep.Sweep.n_failed` counts frames that never produced a number.
    They are already missing from `rows`; without this sentence in the file a
    reader in Foxglove sees the same gap the stride makes and concludes the
    dataset was sampled more thinly than it was."""
    timeline = _timeline(sweep=FakeSweep(n_failed=137))
    said = " ".join(timeline.omitted)
    assert "137 frame(s) could not be measured and are ABSENT" in said
    assert "not a frame with a low score" in said


def test_what_is_not_exported_is_written_into_the_file_not_only_returned():
    """The whole reason this module can exist beside the `.mri` argument: the
    grids and the causal map stay behind, and the file has to say so itself. A
    caveat that lives in our UI is not present in the reader's application."""
    timeline = _timeline()
    record = _record(timeline, "metadata", name="modelmri:not_exported")
    said = " ".join(record.metadata.values())
    assert "per-layer attention grids are NOT in this file" in said
    assert "causal occlusion map" in said
    assert "model weights and the dataset are NOT in this file" in said
    # Numbered keys, because a Foxglove metadata panel is a key/value table and
    # one 900-character value renders as a single unreadable cell.
    assert sorted(record.metadata) == [f"{i:02d}" for i in range(1, 8)]


def test_coverage_is_stated_as_a_share_of_the_dataset():
    """The extreme value in the file is the extreme of what was SAMPLED. That
    is a weaker claim than "the extreme of the dataset" and reads as the
    stronger one unless the file says otherwise."""
    said = " ".join(_timeline().omitted)
    assert "covers 8 of 200 frames (4.0%)" in said


# ------------------------------------------------------------------ the clock


def test_a_published_frame_rate_becomes_seconds_exactly():
    """Arithmetic, not a library: frame 75 at 30 fps is 2.5 s, and a rounding
    slip here shifts every point on the axis against the robot's own topics
    when the two files are merged."""
    clock = rx.clock_for(30.0)
    assert clock.kind == "seconds"
    assert clock.stamp(75) == 2_500_000_000
    assert clock.stamp(0) == 0


def test_no_frame_rate_gets_a_frame_index_axis_that_says_it_is_not_a_duration():
    """A default of 30 fps would invent a duration the reader then measures off
    the axis. `None` is the only way to say the dataset published none, and the
    sentence travels into the file so the axis cannot be read as seconds."""
    clock = rx.clock_for(None)
    assert clock.kind == "frame-index"
    assert clock.fps is None
    assert "do not read a duration off this axis" in clock.sentence
    assert clock.stamp(3) == 3_000_000_000

    timeline = _timeline(fps=None)
    assert (
        _record(timeline, "metadata", name="modelmri:clock").metadata["fps"]
        == "not published by this dataset"
    )


def test_a_broken_frame_rate_is_a_bad_request_and_not_a_substituted_default():
    """A zero or a NaN is a value somebody supplied and got wrong, which is a
    422 — quietly swapping in a default would put a fabricated clock in a file
    that claims a real one."""
    for bad in (0.0, -5.0, float("nan"), float("inf")):
        with pytest.raises(BadRequest):
            rx.clock_for(bad)


# ------------------------------------------------------------------ provenance


def test_provenance_is_required_and_has_no_default_dataset():
    """A file written with an empty dataset is indistinguishable from one
    written before anybody thought about provenance, and by the time it is open
    in Foxglove there is nobody left to ask."""
    with pytest.raises(Refusal) as err:
        rx.Provenance(
            tool="ModelMRI",
            tool_version="0.11.0",
            dataset="",
            camera="top",
            policy=None,
            policy_revision=None,
            measured_by="a test",
            taken_at="2026-08-24T00:00:00+00:00",
        )
    assert "no dataset" in err.value.sentence


def test_an_absent_policy_or_revision_becomes_a_sentence_not_an_empty_value():
    """`vla_sweep.Sweep.policy` is `None` when no policy was resident, and an
    empty metadata value in a viewer reads as an oversight rather than a fact.
    Both absences get words."""
    meta = _one_track_timeline().provenance.to_metadata()
    assert meta["policy"] == "no policy was resident for this measurement"
    assert "could not resolve a commit" in meta["policy_revision"]


def test_the_file_says_the_viewer_did_not_measure_any_of_this():
    """`circuit.py` refuses to let a graph it did not compute look like one it
    did. The same claim outbound: nothing here was computed by Foxglove, Rerun
    or the robot, and the file states it where a reader will meet it."""
    meta = _record(_timeline(), "metadata", name="modelmri:provenance").metadata
    assert "measured by ModelMRI" in meta["not_measured_by_this_viewer"]
    assert "Foxglove, Rerun, or the robot" in meta["not_measured_by_this_viewer"]
    assert "ModelMRI keeps the" in meta["internals_are_not_in_this_file"]


def test_a_named_mri_travels_so_the_internals_stay_findable():
    """The argument that lets this module coexist with the `.mri` decision: the
    timeline is a pointer, not a replacement. Without the pointer in the file
    the reader has a plot and no way back to the grids behind it."""
    timeline = _timeline(mri_pointer="pusht-ep3-t75.mri")
    meta = timeline.provenance.to_metadata()
    assert meta["internals_are_not_in_this_file"] == "pusht-ep3-t75.mri"


# ------------------------------------------------------------------- refusals


def test_a_non_finite_value_is_refused_rather_than_drawn_as_a_gap():
    """A NaN renders as a break in the line, which is exactly what an unsampled
    frame renders as. Two different facts, one picture — so the export stops
    and names the frame instead of publishing the ambiguity."""
    timeline = _one_track_timeline()
    with pytest.raises(Refusal) as err:
        rx.Timeline(
            provenance=timeline.provenance,
            clock=timeline.clock,
            tracks=(
                rx.Track(
                    metric="m",
                    unit="u",
                    resolution="r",
                    episode=4,
                    samples=(rx.Sample(9, float("nan")),),
                    frame_stride=1,
                ),
            ),
        )
    assert "episode 4, timestep 9" in err.value.sentence


def test_two_tracks_cannot_share_one_topic():
    """MCAP permits two channels on one topic and every viewer draws them on a
    single plot with no way to tell which point came from which measurement."""
    track = _one_track_timeline().tracks[0]
    with pytest.raises(Refusal) as err:
        rx.Timeline(
            provenance=_one_track_timeline().provenance,
            clock=rx.clock_for(None),
            tracks=(track, track),
        )
    assert "modelmri/m/episode_0" in err.value.sentence


def test_an_empty_timeline_is_refused():
    """An empty file in Foxglove reads as a policy that produced no
    measurements, which is a different claim from one nobody took."""
    with pytest.raises(Refusal) as err:
        rx.Timeline(
            provenance=_one_track_timeline().provenance,
            clock=rx.clock_for(None),
            tracks=(),
        )
    assert "nothing to export" in err.value.sentence


def test_a_metric_named_camera_cannot_take_the_image_channels_topic():
    """The duplicate-topic guard was built from tracks only, and the image
    channel shares the namespace with them: a track whose metric is "camera"
    lands on exactly `Frame.topic`. Both channels then register on one topic,
    `write_mcap` keys `channel_ids` by topic, and the image channel overwrites
    the measurement one — every measurement written under the id declared
    `foxglove.CompressedImage`, and the measurement channel left empty. That
    is the guard's own refusal text happening past the guard."""
    track = rx.Track(
        metric="camera",
        unit="u",
        resolution="r",
        episode=0,
        samples=(rx.Sample(0, 1.0),),
        frame_stride=1,
    )
    frame = rx.frame_from_mri(_mri_section(provenance={"episode": 0, "timestep": 0}))
    assert track.topic == frame.topic

    with pytest.raises(Refusal) as err:
        rx.Timeline(
            provenance=_one_track_timeline().provenance,
            clock=rx.clock_for(None),
            tracks=(track,),
            frame=frame,
        )
    assert "modelmri/camera/episode_0" in err.value.sentence
    assert "every measurement would be declared an image" in err.value.sentence


def test_a_track_with_no_samples_is_refused_rather_than_written_as_an_empty_topic():
    """`Timeline`'s refusal says "no track carries a sample", and it used to
    check only whether the tuple of tracks was empty. One sample-less track
    passed and produced a 0-message export: channels registered, nothing on
    them — the exact outcome the sentence claims to prevent, and the state in
    which `mean_message_bytes` would have had to answer for a mean of
    nothing."""
    with pytest.raises(Refusal) as err:
        rx.Timeline(
            provenance=_one_track_timeline().provenance,
            clock=rx.clock_for(None),
            tracks=(
                rx.Track(
                    metric="m",
                    unit="u",
                    resolution="r",
                    episode=0,
                    samples=(),
                    frame_stride=1,
                ),
            ),
        )
    assert "carries no samples" in err.value.sentence

    # Which is what makes a message-less plan unreachable: every Timeline that
    # can be built has at least one message in it.
    assert _one_track_timeline().n_messages == 1


def test_a_negative_timestep_is_refused_rather_than_stamped_as_a_negative_time():
    """MCAP's log_time is an unsigned 64-bit field. A negative frame index
    walks straight through a clock that only multiplies and arrives at the
    writer as a negative integer; `session.py` refuses a negative episode or
    timestep for this same data rather than letting it wrap."""
    with pytest.raises(BadRequest) as err:
        rx.Timeline(
            provenance=_one_track_timeline().provenance,
            clock=rx.clock_for(None),
            tracks=(
                rx.Track(
                    metric="m",
                    unit="u",
                    resolution="r",
                    episode=0,
                    samples=(rx.Sample(-5, 1.0),),
                    frame_stride=1,
                ),
            ),
        )
    assert "timestep -5" in err.value.sentence
    assert "unsigned" in err.value.sentence


def test_a_bool_is_refused_everywhere_a_number_is_expected():
    """`isinstance(True, int)` is True and `float(True)` is 1.0, so a flag that
    reached a numeric field passes every bound check below it and is published
    as a measured 1. Three places it used to: an fps of `True` became a
    seconds axis claiming one frame per second; a `True` sample value
    serialised as `"value":true` against a schema declaring `value` a number;
    a `True` timestep became frame 1."""
    with pytest.raises(BadRequest) as err:
        rx.clock_for(True)
    assert "boolean" in err.value.sentence
    # The bool guard has to run first: this is what it would otherwise publish.
    assert rx.clock_for(1.0).fps == float(True)

    for sample in (rx.Sample(0, True), rx.Sample(True, 1.0)):
        with pytest.raises(BadRequest) as err:
            rx.Timeline(
                provenance=_one_track_timeline().provenance,
                clock=rx.clock_for(None),
                tracks=(
                    rx.Track(
                        metric="m",
                        unit="u",
                        resolution="r",
                        episode=0,
                        samples=(sample,),
                        frame_stride=1,
                    ),
                ),
            )
        assert "boolean and not a measurement" in err.value.sentence

    # And the schema this would have violated says so in the file.
    assert rx.MEASUREMENT_SCHEMA["properties"]["value"] == {"type": "number"}


def test_a_stride_below_one_is_a_bad_request():
    """`frame_stride` is not decoration here — it is the file's whole
    explanation of its own gaps, so a value that explains nothing is refused."""
    with pytest.raises(BadRequest):
        rx.Timeline(
            provenance=_one_track_timeline().provenance,
            clock=rx.clock_for(None),
            tracks=(
                rx.Track(
                    metric="m",
                    unit="u",
                    resolution="r",
                    episode=0,
                    samples=(rx.Sample(0, 1.0),),
                    frame_stride=0,
                ),
            ),
        )


def test_an_unknown_container_is_a_422_not_a_crash():
    """The route switches on a string from a request body. `writer_available`
    is the first thing it reaches, so the arm has to be there and not two
    functions later."""
    with pytest.raises(BadRequest):
        rx.writer_available("parquet")
    with pytest.raises(BadRequest):
        rx.write(_timeline(), "x.parquet", container="parquet")


# ------------------------------------------------------- reshaping a sweep


def test_rows_are_ordered_by_time_and_not_by_rank():
    """`vla_sweep.run` ranks its rows by VALUE, which is right for a table and
    draws a line that crosses itself in a plot.

    The first assertion is on the FIXTURE, not the module: unless the rows
    arrive out of time order within an episode, re-sorting them is a no-op and
    the rest of this test passes against a module that never sorts at all.
    That is what the earlier fixture did, so the arrival order is pinned here
    and the test fails loudly if anyone flattens it again."""
    arrival = {}
    for row in _default_rows():
        arrival.setdefault(row.episode, []).append(row.timestep)
    assert arrival == {0: [0, 25, 75, 50], 1: [25, 0, 50, 75]}
    for episode, timesteps in arrival.items():
        assert timesteps != sorted(timesteps), (
            f"episode {episode} arrives already sorted by time, so this test "
            f"cannot tell a module that sorts from one that does not"
        )

    timeline = _timeline()
    assert [t.episode for t in timeline.tracks] == [0, 1]
    for track in timeline.tracks:
        assert [s.timestep for s in track.samples] == [0, 25, 50, 75]
    # And the value stayed with its own frame through the re-sort — an order
    # fixed by moving values onto the wrong timesteps is a straight line
    # through eight points nobody measured.
    for track in timeline.tracks:
        assert [s.value for s in track.samples] == [
            _VALUES[track.episode][t] for t in (0, 25, 50, 75)
        ]


def test_an_episode_length_is_none_when_the_source_did_not_publish_one():
    """Zero would read as an empty episode. The channel says which it is."""
    bare = _record(_timeline(), "channel", topic="modelmri/attention_entropy/episode_0")
    assert bare.metadata["episode_length_frames"] == (
        "not published by the source of this measurement"
    )
    known = rx.mcap_records(_timeline(episode_lengths={0: 100}))
    channel = next(r for r in known if r.kind == "channel")
    assert channel.metadata["episode_length_frames"] == "100"


def test_something_that_is_not_a_sweep_is_named_field_by_field():
    """A wrong object reaching attribute access raises AttributeError, which
    the server answers as a 500 with a generic sentence. Naming the missing
    fields turns it into an answer somebody can act on."""

    class NotASweep:
        metric = "x"

    with pytest.raises(BadRequest) as err:
        rx.timeline_from_sweep(NotASweep(), fps=1.0, tool_version="0.11.0")
    assert "unit" in err.value.sentence
    assert "frames_total" in err.value.sentence


def test_the_real_sweep_still_fits():
    """The contract this module reads is `vla_sweep.Sweep`'s field list. Held
    here against the real class so that renaming a field there fails this test
    rather than producing a `BadRequest` in front of a user who did nothing
    wrong."""
    from modelmri import vla_sweep

    sweep = vla_sweep.Sweep(
        metric="attention_entropy",
        unit=vla_sweep.METRICS["attention_entropy"][1],
        dataset="lerobot/pusht",
        policy=None,
        camera="observation.images.top",
        episode_stride=1,
        frame_stride=25,
        rows=[vla_sweep.Row(episode=0, timestep=0, value=1.5)],
        n_frames=1,
        frames_total=100,
    )
    timeline = rx.timeline_from_sweep(sweep, fps=10.0, tool_version="0.11.0")
    assert timeline.tracks[0].unit == "nats over the patch grid"
    assert timeline.provenance.policy is None


# ----------------------------------------------------------- the camera frame

_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _mri_section(**overrides):
    import base64

    section = {
        "provenance": {"episode": 3, "timestep": 75},
        "frame": "data:image/png;base64," + base64.b64encode(_PNG).decode("ascii"),
        "frame_size": [96, 96],
        "frame_downsampled": False,
    }
    section.update(overrides)
    return section


def test_a_downsampled_frame_says_so_where_the_pixel_readout_is():
    """`vla.share_payload` shrinks a large frame and warns that "a silently
    shrunk frame under a causal map is a wrong picture". Foxglove's image panel
    offers a pixel readout, so the warning has to survive the trip."""
    frame = rx.frame_from_mri(
        _mri_section(
            frame_downsampled=True,
            frame_note="the camera frame was 480x480 and is stored at 256x256",
        )
    )
    assert frame.downsampled is True
    assert frame.episode == 3 and frame.timestep == 75

    timeline = rx.Timeline(
        provenance=_one_track_timeline().provenance,
        clock=rx.clock_for(None),
        tracks=(),
        frame=frame,
        omitted=tuple(rx.omissions_for(frame=frame)),
    )
    channel = _record(timeline, "channel", topic="modelmri/camera/episode_3")
    assert channel.metadata["downsampled"] == "true"
    assert "480x480" in channel.metadata["note"]
    assert any("DOWNSAMPLED" in s for s in timeline.omitted)


def test_a_frame_that_is_not_a_png_data_url_is_refused_not_relabelled():
    """A viewer handed JPEG bytes labelled `png` renders nothing and reports no
    error, so guessing the encoding produces a silent blank panel."""
    with pytest.raises(Refusal) as err:
        rx.frame_from_mri(_mri_section(frame="data:image/jpeg;base64,QQ=="))
    assert "no PNG camera frame" in err.value.sentence

    with pytest.raises(Refusal) as err:
        rx.frame_from_mri(_mri_section(frame="data:image/png;base64,!!!!"))
    assert "not valid base64" in err.value.sentence


def test_an_unpublished_frame_size_is_a_sentence_and_not_a_measured_zero():
    """A `.mri` section with no `frame_size` used to become `0 x 0` in the
    channel metadata. Zero is not a size anything was stored at, and a reader
    comparing it against the resolution the policy saw concludes the frame was
    empty rather than that nobody wrote the dimensions down."""
    section = _mri_section()
    section.pop("frame_size")
    frame = rx.frame_from_mri(section)
    assert frame.width is None and frame.height is None

    timeline = rx.Timeline(
        provenance=_one_track_timeline().provenance,
        clock=rx.clock_for(None),
        tracks=(),
        frame=frame,
    )
    channel = _record(timeline, "channel", topic="modelmri/camera/episode_3")
    assert channel.metadata["width"] == (
        "not published by the .mri section this frame came from"
    )
    assert channel.metadata["height"] == channel.metadata["width"]

    # A published one still travels as the number it is.
    sized = _record(
        rx.Timeline(
            provenance=timeline.provenance,
            clock=rx.clock_for(None),
            tracks=(),
            frame=rx.frame_from_mri(_mri_section()),
        ),
        "channel",
        topic="modelmri/camera/episode_3",
    )
    assert (sized.metadata["width"], sized.metadata["height"]) == ("96", "96")


def test_a_damaged_frame_size_is_a_refusal_and_not_a_traceback():
    """`int("abc")` reached the route as a raw ValueError, which the server
    answers as a 500 with a generic sentence. A damaged section is a fact
    about the file and gets a sentence naming it — the same shape the invalid
    base64 branch already had."""
    for bad in (["abc", "def"], [96], "96x96", [96, 96, 3], [0, 96], [-1, 96]):
        with pytest.raises(Refusal):
            rx.frame_from_mri(_mri_section(frame_size=bad))

    with pytest.raises(Refusal) as err:
        rx.frame_from_mri(_mri_section(frame_size=[96, 0]))
    assert "frame height of 0" in err.value.sentence

    # And a frame whose provenance cannot say which frame it is does not get
    # exported as episode 0, timestep 0, beside measurements from elsewhere.
    with pytest.raises(Refusal) as err:
        rx.frame_from_mri(_mri_section(provenance={"episode": 3}))
    assert "has no timestep in its provenance" in err.value.sentence


def test_with_no_frame_the_file_says_so_rather_than_staying_quiet():
    """Absence of an image panel is not evidence there was no camera."""
    assert any("No camera frame is in this file" in s for s in _timeline().omitted)


# --------------------------------------------------------------- the plan


def test_the_payload_count_is_exact_and_the_estimate_states_its_basis():
    """`budget.py`'s rule: a measured figure and an estimate are different
    kinds of number and must not be mixed. `payload_bytes` is the sum of the
    bytes this module hands the writer, so it is exact; the file size is
    payload + MCAP's 31-byte message header + a 1 kB envelope, uncompressed,
    and the basis says so rather than being presented as a bound."""
    timeline = _timeline()
    shape = rx.plan(timeline)
    messages = [r for r in rx.mcap_records(timeline) if r.kind == "message"]

    assert shape.payload_bytes == sum(len(r.data) for r in messages)
    assert shape.n_messages == 8
    assert shape.n_channels == 2
    assert shape.estimated_file_bytes == shape.payload_bytes + 31 * 8 + 1024
    assert "UNCOMPRESSED" in shape.estimated_basis
    assert "not a size to promise anybody" in shape.estimated_basis
    assert shape.mean_message_bytes == round(shape.payload_bytes / 8, 1)


def test_a_mean_of_no_messages_is_none_and_never_a_zero():
    """A mean of nothing is not 0.0 bytes — 0.0 is a message size somebody
    could quote, computed from no messages.

    This used to be asserted through a one-message timeline, which never
    reached the branch: `... if messages else 0.0` passed the whole suite. The
    rule lives in `mean_message_bytes`, so it is checked there, at zero, where
    it applies."""
    assert rx.mean_message_bytes(0, 0) is None
    assert rx.mean_message_bytes(1630, 0) is None
    assert rx.mean_message_bytes(1630, 8) == 203.8

    # And no valid Timeline can reach it: a message-less export is refused at
    # construction, so `plan()` always divides by at least one.
    timeline = rx.Timeline(
        provenance=_one_track_timeline().provenance,
        clock=rx.clock_for(None),
        tracks=(),
        frame=rx.frame_from_mri(_mri_section()),
    )
    shape = rx.plan(timeline)
    assert shape.n_messages == 1
    assert shape.unit_overhead_bytes == 0
    assert shape.mean_message_bytes == float(shape.payload_bytes)


def test_the_plan_reports_the_cap_that_will_refuse_the_write(monkeypatch):
    """The plan is documented as the whole shape of the file, answering "how
    large will this be, and what will be in it". A plan that describes 600,000
    messages without mentioning that the write refuses above 500,000 has
    described a file that will never exist — and the caller reading it is
    deciding whether to spend the disk on the strength of it.

    The cap is lowered rather than the timeline grown, so this costs eight
    messages instead of half a million."""
    under = rx.plan(_timeline()).to_dict()
    assert under["max_messages"] == 500_000
    assert under["over_cap"] is False
    assert "-message cap" not in under["means"]

    monkeypatch.setattr(rx, "MAX_MESSAGES", 4)
    over = rx.plan(_timeline()).to_dict()
    assert over["n_messages"] == 8
    assert over["max_messages"] == 4
    assert over["over_cap"] is True
    assert "8 messages is over the 4-message cap" in over["means"]
    assert "NOTHING WILL BE WRITTEN" in over["means"]


def test_the_measured_overhead_in_the_module_docstring_is_reproducible():
    """The docstring publishes a measured cost — so much of so much payload,
    at such a share — and a measured number a reader cannot reproduce is a
    number nobody measured. The exact inputs live here: a real
    `vla_sweep.Sweep`, two episodes at stride 25, the metric's own unit and a
    stated ln(16) ceiling. If these move, the docstring is wrong and this
    fails."""
    from modelmri import vla_sweep

    sweep = vla_sweep.Sweep(
        metric="attention_entropy",
        unit=vla_sweep.METRICS["attention_entropy"][1],
        dataset="lerobot/pusht",
        policy="lerobot/smolvla_base",
        camera="observation.images.top",
        episode_stride=1,
        frame_stride=25,
        rows=[
            vla_sweep.Row(episode=episode, timestep=timestep, value=value)
            for episode, series in _VALUES.items()
            for timestep, value in series.items()
        ],
        n_frames=8,
        frames_total=200,
    )
    assert sweep.unit == "nats over the patch grid"
    shape = rx.plan(
        rx.timeline_from_sweep(
            sweep, fps=10.0, tool_version="0.11.0", resolution=_CEILING
        )
    )
    assert shape.n_messages == 8
    assert shape.payload_bytes == 1709
    assert shape.unit_overhead_bytes == 1000
    assert round(100.0 * shape.unit_overhead_bytes / shape.payload_bytes, 1) == 58.5
    assert "58.5%" in shape.sentence()


def test_the_plan_is_complete_even_when_nothing_can_write_it():
    """The useful half on a machine with neither writer: the reader finds out
    what they would get before they install anything. A plan that refused
    alongside the writer would make the install decision unmakeable."""
    for container in rx.FORMATS:
        shape = rx.plan(_timeline(), container=container)
        assert shape.n_messages == 8
        assert shape.payload_bytes > 0
        assert shape.omitted


# ------------------------------------------------------------- the writers


def test_mcap_refuses_with_the_install_command_when_the_package_is_absent(
    monkeypatch, tmp_path
):
    """Measured on this machine: `import mcap` raises ModuleNotFoundError. The
    refusal is the deliverable in that state, and `vla_data.encode_png`'s shape
    is the one to copy — name the package, name the command, and say the
    surrounding work is real."""
    monkeypatch.setitem(sys.modules, "mcap", None)
    (ok, why), package = rx.writer_available("mcap")
    assert ok is False
    assert package == "mcap"
    assert "pip install mcap" in why
    assert "not a ModelMRI dependency" in why

    with pytest.raises(Refusal) as err:
        rx.write_mcap(_timeline(), tmp_path / "out.mcap")
    assert "pip install mcap" in err.value.sentence
    assert not list(tmp_path.iterdir())


def test_rrd_is_refused_for_a_reason_installing_rerun_would_not_fix(tmp_path):
    """Two facts, and the second is the one that matters: rerun-sdk is absent
    here AND this module has never been run against one. Refusing only on the
    import would promise that `pip install rerun-sdk` produces a working
    export, which nobody has verified."""
    (ok, why), package = rx.writer_available("rrd")
    assert ok is False
    assert package == "rerun-sdk"
    assert "rerun-sdk" in why
    assert "correctness is a guess" in why
    assert "Write the MCAP instead" in why

    with pytest.raises(Refusal):
        rx.write_rrd(_timeline(), tmp_path / "out.rrd")
    assert not list(tmp_path.iterdir())


class RecordingWriter:
    """A call recorder shaped like `mcap.writer.Writer`. NOT a validator.

    It proves what this module hands a writer — the order, the ids, the
    keyword names, the timestamps — and proves nothing about the bytes a real
    `mcap` would produce, because there is no `mcap` on this machine to
    produce them. It writes each payload to the stream so the receipt's size
    arithmetic runs against a real file rather than an empty one.
    """

    def __init__(self, stream):
        self.stream = stream
        self.calls: list[tuple] = []
        self._next_id = 0

    def start(self, profile, library):
        self.calls.append(("start", profile, library))

    def register_schema(self, name, encoding, data):
        self._next_id += 1
        self.calls.append(("schema", name, encoding, len(data)))
        return self._next_id

    def register_channel(self, topic, message_encoding, schema_id, metadata):
        self._next_id += 1
        self.calls.append(("channel", topic, message_encoding, schema_id, metadata))
        return self._next_id

    def add_metadata(self, name, data):
        self.calls.append(("metadata", name, data))

    def add_message(self, channel_id, log_time, publish_time, sequence, data):
        # `publish_time` is recorded, not dropped: it is a field of the file
        # and dropping it here is how it stopped being checked at all.
        self.calls.append(
            ("message", channel_id, log_time, sequence, len(data), publish_time)
        )
        self.stream.write(data)

    def finish(self):
        self.calls.append(("finish",))


def _install_fake_mcap(monkeypatch, writer_class=RecordingWriter):
    made: list[RecordingWriter] = []

    class Spy(writer_class):
        def __init__(self, stream):
            super().__init__(stream)
            made.append(self)

    package = types.ModuleType("mcap")
    module = types.ModuleType("mcap.writer")
    module.Writer = Spy
    package.writer = module
    monkeypatch.setitem(sys.modules, "mcap", package)
    monkeypatch.setitem(sys.modules, "mcap.writer", module)
    return made


def test_the_replay_loop_drives_a_writer_in_the_order_the_records_state(
    monkeypatch, tmp_path
):
    """Metadata before schemas before channels before messages, so a reader
    scanning the head of the file meets the provenance and the omissions before
    the first number. Every decision was taken in `mcap_records`; this checks
    the loop that replays them resolves schema and channel ids correctly and
    sequences each topic from zero — an id crossed between channels puts a
    measurement on the wrong plot."""
    made = _install_fake_mcap(monkeypatch)
    receipt = rx.write_mcap(_timeline(), tmp_path / "out.mcap")
    writer = made[0]

    kinds = [c[0] for c in writer.calls]
    assert kinds[0] == "start"
    assert kinds[-1] == "finish"
    assert kinds.count("metadata") == 3
    assert kinds.count("schema") == 1
    assert kinds.count("channel") == 2
    assert kinds.count("message") == 8
    # No message before every channel exists.
    assert max(i for i, k in enumerate(kinds) if k == "channel") < min(
        i for i, k in enumerate(kinds) if k == "message"
    )

    channel_ids = [c[1] for c in writer.calls if c[0] == "message"]
    assert channel_ids == [channel_ids[0]] * 4 + [channel_ids[4]] * 4
    assert channel_ids[0] != channel_ids[4]
    sequences = [c[3] for c in writer.calls if c[0] == "message"]
    assert sequences == [0, 1, 2, 3, 0, 1, 2, 3]
    # 10 fps: frames 0/25/50/75 are 0.0/2.5/5.0/7.5 seconds.
    stamps = [c[2] for c in writer.calls if c[0] == "message"][:4]
    assert stamps == [0, 2_500_000_000, 5_000_000_000, 7_500_000_000]

    assert receipt["container"] == "mcap"
    assert receipt["bytes_written"] == (tmp_path / "out.mcap").stat().st_size
    assert receipt["n_messages"] == 8
    assert receipt["plan"]["payload_bytes"] == receipt["bytes_written"]


def test_the_declarations_handed_to_the_container_are_the_ones_the_file_claims(
    monkeypatch, tmp_path
):
    """Four values cross into the container's own header and channel records,
    and the recorder was capturing all four while nothing asserted any of
    them. The one that matters most is `library`: it is the field inside an
    MCAP file that names the writing tool, and set to "foxglove-studio" this
    module would produce a file that says Foxglove wrote it — precisely the
    "mistakable for something that container's own tools measured" failure the
    module exists to prevent, published in the one field a reader checks for
    it."""
    from modelmri import __version__

    made = _install_fake_mcap(monkeypatch)
    rx.write_mcap(
        rx.Timeline(
            provenance=_one_track_timeline().provenance,
            clock=rx.clock_for(10.0),
            tracks=_timeline().tracks,
            frame=rx.frame_from_mri(_mri_section()),
        ),
        tmp_path / "out.mcap",
    )
    calls = made[0].calls

    _, profile, library = next(c for c in calls if c[0] == "start")
    # ModelMRI's messages are its own JSON. A profile names a schema
    # convention ("ros1", "ros2") and claiming one tells the reader's tooling
    # to interpret these bodies under rules they do not follow.
    assert profile == ""
    assert library.startswith("modelmri ")
    assert __version__ in library
    assert "foxglove" not in library.lower()
    assert "rerun" not in library.lower()

    # Every channel declares JSON, because that is what `_json_bytes` wrote.
    assert {c[2] for c in calls if c[0] == "channel"} == {"json"}
    # ...and the bodies really are JSON, so the declaration is not a label
    # stuck on bytes of some other encoding.
    for record in rx.mcap_records(_timeline()):
        if record.kind == "message":
            json.loads(record.data)

    # The image rides under Foxglove's own schema name, which is what makes it
    # render as a picture instead of a base64 blob in a raw-message panel.
    schemas = [c[1] for c in calls if c[0] == "schema"]
    assert schemas == ["modelmri.Measurement", "foxglove.CompressedImage"]

    # publish_time is a field of the file. These numbers were measured
    # off-line, so there is no separate publication instant to claim.
    for call in (c for c in calls if c[0] == "message"):
        assert call[5] == call[2]
    assert sorted({c[2] for c in calls if c[0] == "message"}) == [
        0,
        2_500_000_000,
        5_000_000_000,
        7_500_000_000,
    ]


def test_the_estimate_is_reported_against_what_actually_happened(monkeypatch, tmp_path):
    """An estimate nobody checks is a number that never improves. The receipt
    carries both figures and their ratio so the first person to run this with a
    real `mcap` installed learns immediately whether the basis was any good."""
    _install_fake_mcap(monkeypatch)
    receipt = rx.write_mcap(_timeline(), tmp_path / "out.mcap")
    assert receipt["bytes_estimated"] > receipt["bytes_written"]
    assert receipt["estimate_over_actual"] == round(
        receipt["bytes_estimated"] / receipt["bytes_written"], 3
    )


def test_a_zero_byte_file_is_refused_rather_than_receipted_as_a_success(
    monkeypatch, tmp_path
):
    """The failure this project exists to prevent, in its purest form: the
    export "succeeds", the panel says so, and Foxglove opens nothing. A writer
    that produces no bytes is named rather than handed back as a receipt with a
    0 in it."""

    class SilentWriter(RecordingWriter):
        def add_message(self, channel_id, log_time, publish_time, sequence, data):
            self.calls.append(
                ("message", channel_id, log_time, sequence, len(data), publish_time)
            )

    _install_fake_mcap(monkeypatch, SilentWriter)
    with pytest.raises(Refusal) as err:
        rx.write_mcap(_timeline(), tmp_path / "out.mcap")
    assert "zero-byte file" in err.value.sentence
    assert "check its version" in err.value.sentence
    # And the file goes with the refusal. Saying "Nothing was written" while
    # leaving an .mcap on disk hands the reader the artifact it just disowned,
    # and the next person to open the directory finds an export.
    assert "The empty file has been removed." in err.value.sentence
    assert list(tmp_path.iterdir()) == []


def test_too_many_messages_is_refused_with_the_count_and_never_truncated(
    monkeypatch, tmp_path
):
    """`vla_sweep.MAX_FRAMES` makes this argument about a ranking; it is worse
    here, because a truncated timeline looks exactly like a timeline and the
    reader is in another application with nothing of ours left to read."""
    _install_fake_mcap(monkeypatch)
    monkeypatch.setattr(rx, "MAX_MESSAGES", 4)
    with pytest.raises(Refusal) as err:
        rx.write_mcap(_timeline(), tmp_path / "out.mcap")
    assert "8 messages and the cap is 4" in err.value.sentence
    assert not list(tmp_path.iterdir())


def test_write_routes_on_the_container_name(monkeypatch, tmp_path):
    """One entry point for a route that switches on a request body string."""
    _install_fake_mcap(monkeypatch)
    assert (
        rx.write(_timeline(), tmp_path / "a.mcap", container="mcap")["n_messages"] == 8
    )
    with pytest.raises(Refusal):
        rx.write(_timeline(), tmp_path / "a.rrd", container="rrd")


# ------------------------------------------------------------------- helpers


def _messages(timeline):
    return [
        json.loads(r.data) for r in rx.mcap_records(timeline) if r.kind == "message"
    ]


def _stamped(timeline):
    """(log_time_ns, body) for every message, in record order."""
    return [
        (r.log_time_ns, json.loads(r.data))
        for r in rx.mcap_records(timeline)
        if r.kind == "message"
    ]


def _record(timeline, kind, name=None, topic=None):
    for record in rx.mcap_records(timeline):
        if record.kind != kind:
            continue
        if name is not None and record.name != name:
            continue
        if topic is not None and record.topic != topic:
            continue
        return record
    raise AssertionError(f"no {kind} record matching name={name} topic={topic}")
