# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""Reading a sweep back out WHOLE, or refusing to pretend it can.

`stored()` answers "which rows are strongest". `retrieve()` answers "what was
the sweep", and the difference is every fact a row does not carry: the unit
the numbers are in, the episode stride, the frame total a coverage claim is
measured against. `robot_export.timeline_from_sweep` requires all of them, and
the module's whole argument is that a number arriving in Foxglove under a unit
this code supplied from memory is indistinguishable from one that was measured
in it.

So the failure mode these tests guard is not an exception. It is a plausible
file: right rows, right dataset, and a unit nobody measured anything in. Every
test below names the wrong reading it stops.
"""

from __future__ import annotations

import sqlite3

import pytest

# The same doubles the sweep's own tests drive, so these stay honest as that
# file changes rather than drifting into a second idea of what a reader is.
from test_vla_sweep import FakeHandle, FakeReader

from modelmri import paths
from modelmri import vla_sweep as sw

DATASET = "lerobot/pusht"
POLICY = "lerobot/smolvla_base"
METRIC = "attention_entropy"


def _saved(tmp_path, monkeypatch, *, episodes=2, **kwargs):
    """One real sweep, run and saved, so these read what `save()` writes."""
    monkeypatch.setenv("MODELMRI_HOME", str(tmp_path))
    out = sw.run(FakeHandle(), FakeReader(episodes=episodes), METRIC, **kwargs)
    sw.save(out)
    return out


def _open_db() -> sqlite3.Connection:
    return sqlite3.connect(str(paths.trace_db_path()))


# ------------------------------------------------------------ the round trip


def test_a_retrieved_sweep_carries_the_facts_a_row_cannot(tmp_path, monkeypatch):
    """The round trip, field by field.

    A retrieval that rebuilt the unit from `METRICS[metric]` and the strides
    from the module defaults would pass any test that only compared rows — and
    it would describe a different sweep the day either of those changed. Every
    field here has to come back out of the database it went into.
    """
    ran = _saved(tmp_path, monkeypatch, frame_stride=50, episode_stride=1)

    back = sw.retrieve(DATASET, POLICY, METRIC)

    assert back.metric == ran.metric
    assert back.unit == ran.unit
    assert back.dataset == ran.dataset
    assert back.policy == ran.policy
    assert back.camera == ran.camera
    assert back.frame_stride == ran.frame_stride == 50
    assert back.episode_stride == ran.episode_stride == 1
    assert back.n_frames == ran.n_frames
    assert back.n_episodes == ran.n_episodes
    assert back.frames_total == ran.frames_total == 200
    assert back.n_failed == ran.n_failed
    assert [(r.episode, r.timestep, r.value) for r in back.rows] == [
        (r.episode, r.timestep, r.value) for r in ran.rows
    ]
    # `means()` is rebuilt from the retrieved fields, so a defaulted frame
    # total would show up here as a coverage claim computed against zero.
    assert "4 of 200 frames (2.0%)" in back.means()
    assert "THE STRIDE IS 50 FRAMES" in back.means()


def test_the_retrieved_sweep_is_the_shape_the_exporter_reads(tmp_path, monkeypatch):
    """The gap this closed, checked against the consumer that named it.

    `robot_export.timeline_from_sweep` refuses anything missing one of eleven
    named fields. This is the assertion that says the retrieval returns a
    sweep an exporter can use, rather than something Sweep-shaped.
    """
    from modelmri import robot_export as rx

    _saved(tmp_path, monkeypatch, frame_stride=50)
    back = sw.retrieve(DATASET, POLICY, METRIC)

    assert [name for name in rx._SWEEP_FIELDS if not hasattr(back, name)] == []
    timeline = rx.timeline_from_sweep(back, fps=None, tool_version="test")
    assert timeline.tracks
    assert all(track.unit == back.unit for track in timeline.tracks)
    assert all(track.frame_stride == 50 for track in timeline.tracks)


def test_the_unit_comes_out_of_the_database_and_not_out_of_this_process(
    tmp_path, monkeypatch
):
    """The one assertion a round trip cannot make on its own.

    `save()` writes `METRICS[metric][1]`, so today the stored unit and the
    recomputed one are the same string and a retrieval that read
    `METRICS[metric]` would pass every comparison in this file. They are the
    same string only until the metric's unit is reworded or its computation
    changes — at which point every sweep already on disk would be relabelled
    with the new unit, silently, and the oldest rows would carry the newest
    claim. So the record is edited here to hold a unit this process would
    never produce, and the retrieval must return THAT.
    """
    _saved(tmp_path, monkeypatch, frame_stride=50)
    db = _open_db()
    db.execute("UPDATE vla_sweep_run SET unit='nats over a 4-patch grid, v0.9'")
    db.commit()
    db.close()

    back = sw.retrieve(DATASET, POLICY, METRIC)

    assert back.unit == "nats over a 4-patch grid, v0.9"
    assert back.unit != sw.METRICS[METRIC][1]
    assert "nats over a 4-patch grid, v0.9" in back.means()


def test_the_failure_sample_and_its_true_count_stay_apart(tmp_path, monkeypatch):
    """`failed` is a SAMPLE capped at 20 and `n_failed` is the count, stored in
    separate columns for the same reason the dataclass keeps them separate. A
    retrieval that rebuilt `n_failed` as `len(failed)` would republish the
    truncated-list bug through the database: 600 failures reported as 20."""
    monkeypatch.setenv("MODELMRI_HOME", str(tmp_path))
    ran = sw.run(FakeHandle(), FakeReader(episodes=2), METRIC, frame_stride=50)
    ran.failed = [{"episode": 0, "timestep": 0, "why": "RuntimeError"}]
    ran.n_failed = 97
    sw.save(ran)

    back = sw.retrieve(DATASET, POLICY, METRIC)

    assert back.n_failed == 97
    assert back.failed == [{"episode": 0, "timestep": 0, "why": "RuntimeError"}]
    assert "97 frame(s) could not be measured" in back.means()


def test_a_policy_that_was_not_resident_comes_back_as_none(tmp_path, monkeypatch):
    """`""` is the lookup key on disk; `None` is the fact. A `""` returned here
    reaches `robot_export.Provenance`, which prints an empty policy as a blank
    — and a reader in Foxglove takes a blank for an oversight rather than for
    "no policy was resident when this was measured"."""
    monkeypatch.setenv("MODELMRI_HOME", str(tmp_path))

    class NoPolicy(FakeHandle):
        class _S:
            repo = ""
            n_layers = 2

        def status(self):
            return self._S()

    sw.save(sw.run(NoPolicy(), FakeReader(episodes=1), METRIC, frame_stride=50))

    back = sw.retrieve(DATASET, "", METRIC)

    assert back.policy is None
    assert back.policy != ""


# ------------------------------------------------------------- the refusals


def test_rows_from_before_the_run_table_are_refused_rather_than_labelled(
    tmp_path, monkeypatch
):
    """The migration case, and the reason this function exists at all.

    An install that has been saving rows since before `vla_sweep_run` existed
    holds measurements and no unit. `METRICS[metric][1]` is right there and
    reading it would produce a file that looks measured — but those rows could
    have been written by a version whose metric of that name computed
    something else, and nothing in the table can say. So: refuse, name the
    migration, and say what fixes it.
    """
    _saved(tmp_path, monkeypatch, frame_stride=50)
    db = _open_db()
    db.execute("DELETE FROM vla_sweep_run")
    db.commit()
    db.close()

    with pytest.raises(sw.SweepError) as err:
        sw.retrieve(DATASET, POLICY, METRIC)

    said = err.value.sentence
    assert "4 measured row(s)" in said, "the rows are still there and it says so"
    assert "run record" in said
    assert "Re-run the sweep" in said, "a refusal with no next step is a dead end"
    # The one thing it must not do is supply the unit out of this process.
    assert sw.METRICS[METRIC][1] not in said
    # And it must not delete the measurements to make its invariant hold.
    assert len(sw.stored(DATASET, POLICY, METRIC)) == 4


def test_nothing_stored_is_a_different_sentence_from_stored_without_a_record(
    tmp_path, monkeypatch
):
    """Collapsing the two would tell somebody holding four thousand measured
    rows that they had never run a sweep."""
    monkeypatch.setenv("MODELMRI_HOME", str(tmp_path))

    with pytest.raises(sw.SweepError) as err:
        sw.retrieve(DATASET, POLICY, METRIC)

    said = err.value.sentence
    assert "no sweep of attention_entropy is stored" in said
    assert "Run one" in said
    assert "run record" not in said, "nothing was migrated, and nothing was lost"


def test_an_unknown_metric_names_the_ones_that_exist(tmp_path, monkeypatch):
    """A typo — `"attention entropy"` with a space — used to be an empty result
    set, which reads as "you have not run that sweep yet"."""
    monkeypatch.setenv("MODELMRI_HOME", str(tmp_path))

    with pytest.raises(sw.SweepError) as err:
        sw.retrieve(DATASET, POLICY, "attention entropy")

    assert "occlusion_peak" in err.value.sentence


def test_two_sweeps_at_different_strides_are_not_returned_as_one(tmp_path, monkeypatch):
    """THE SUPERIMPOSITION. Rows are keyed by (episode, timestep), so a run at
    stride 50 following one at stride 25 REPLACES the frames it re-measured and
    leaves the finer run's 25s and 75s sitting under the same keys. Both sets
    answer the same query, and a `Sweep` states one stride for all of its rows
    — so the coarse run would publish the fine run's points as frames it had
    opened, and the exported file would say every gap is 50 frames wide.
    """
    _saved(tmp_path, monkeypatch, frame_stride=25)
    assert sw.retrieve(DATASET, POLICY, METRIC).n_frames == 8

    _saved(tmp_path, monkeypatch, frame_stride=50)

    with pytest.raises(sw.SweepError) as err:
        sw.retrieve(DATASET, POLICY, METRIC)

    said = err.value.sentence
    assert "25" in said and "50" in said, "both strides are named"
    assert "superimposed" in said
    assert "forget(" in said, "and the refusal says what to do about it"
    # Nothing was thrown away to produce that refusal: `stored()` still shows
    # every row with the stride it was actually taken at.
    assert sorted({r["stride"] for r in sw.stored(DATASET, POLICY, METRIC)}) == [25, 50]


def test_an_earlier_run_at_a_finer_episode_stride_is_caught_by_the_count(
    tmp_path, monkeypatch
):
    """The residual case the stride check cannot see.

    Two runs at frame stride 50 and episode strides 1 and 2 leave rows that all
    agree on the stride and still come from two different samplings. The second
    run's `n_episodes` and `n_frames` describe half of what the query returns,
    so its coverage sentence would count episodes it never opened.
    """
    _saved(tmp_path, monkeypatch, episodes=4, frame_stride=50, episode_stride=1)
    _saved(tmp_path, monkeypatch, episodes=4, frame_stride=50, episode_stride=2)

    with pytest.raises(sw.SweepError) as err:
        sw.retrieve(DATASET, POLICY, METRIC)

    said = err.value.sentence
    assert "4 measured frame(s)" in said and "8 row(s)" in said
    assert "EPISODE stride" in said


def test_two_cameras_are_refused_rather_than_the_first_one_taken(tmp_path, monkeypatch):
    """A `Sweep` names ONE camera. A single-arm SO-100 recording has a wrist
    and an overhead view, and the same metric over both is two rankings —
    picking whichever sorts first would export frames ranked through a lens
    nobody asked for, with nothing downstream saying which."""
    monkeypatch.setenv("MODELMRI_HOME", str(tmp_path))
    for camera in ("observation.images.top", "observation.images.wrist"):
        reader = FakeReader(episodes=2)
        reader.camera = camera
        sw.save(sw.run(FakeHandle(), reader, METRIC, frame_stride=50))

    with pytest.raises(sw.SweepError) as err:
        sw.retrieve(DATASET, POLICY, METRIC)
    assert "observation.images.wrist" in err.value.sentence
    assert "ONE camera" in err.value.sentence

    named = sw.retrieve(DATASET, POLICY, METRIC, camera="observation.images.wrist")
    assert named.camera == "observation.images.wrist"
    assert named.n_frames == 4
    # And the rows are the wrist camera's alone. `stored()` does not filter by
    # camera — it answers a ranking query and carries each row's stride out —
    # so a retrieval that reused its query would have handed back eight rows
    # from two lenses under one camera name.
    assert {r.episode for r in named.rows} == {0, 1}


def test_a_sweep_that_measured_nothing_is_refused_not_returned_empty(
    tmp_path, monkeypatch
):
    """With `av` absent every frame fails and the run record says so: zero
    rows, N failures. Handed back as a Sweep it reaches `robot_export` as an
    empty timeline, which in Foxglove reads as a policy that produced no
    measurements — a different claim from one nobody could take."""
    monkeypatch.setenv("MODELMRI_HOME", str(tmp_path))
    reader = FakeReader(episodes=1, broken=[(0, 0), (0, 50)])
    ran = sw.run(FakeHandle(), reader, METRIC, frame_stride=50)
    assert (ran.n_frames, ran.n_failed) == (0, 2)
    sw.save(ran)

    with pytest.raises(sw.SweepError) as err:
        sw.retrieve(DATASET, POLICY, METRIC)

    said = err.value.sentence
    assert "2 sampled frame(s) failed" in said
    assert "modelmri[vla]" in said, "and it names what usually fixes it"


def test_a_damaged_failure_sample_is_refused_rather_than_read_as_none(
    tmp_path, monkeypatch
):
    """An empty list in place of an unreadable one claims every frame measured
    cleanly, which is the strongest possible reading of a corrupt record."""
    _saved(tmp_path, monkeypatch, frame_stride=50)
    db = _open_db()
    db.execute("UPDATE vla_sweep_run SET failed='{not json'")
    db.commit()
    db.close()

    with pytest.raises(sw.SweepError) as err:
        sw.retrieve(DATASET, POLICY, METRIC)

    assert "failure sample" in err.value.sentence
    assert "Re-run the sweep" in err.value.sentence


# ------------------------------------------------------------- the remedies


def test_forget_is_a_real_next_step_and_says_what_it_removed(tmp_path, monkeypatch):
    """A refusal naming a remedy that does not work is worse than one naming
    none, so the remedy is exercised here rather than described."""
    _saved(tmp_path, monkeypatch, frame_stride=25)
    _saved(tmp_path, monkeypatch, frame_stride=50)

    gone = sw.forget(DATASET, POLICY, METRIC)

    assert gone == {"rows": 8, "runs": 1}
    assert sw.stored(DATASET, POLICY, METRIC) == []

    _saved(tmp_path, monkeypatch, frame_stride=50)
    back = sw.retrieve(DATASET, POLICY, METRIC)
    assert back.frame_stride == 50
    assert back.n_frames == 4


def test_forget_leaves_another_dataset_alone(tmp_path, monkeypatch):
    """It is keyed like everything else here. A `DELETE` that dropped more than
    the named key would take a measurement nobody asked about, and the caller
    would be handed a count that looked right."""
    _saved(tmp_path, monkeypatch, frame_stride=50)
    monkeypatch.setenv("MODELMRI_HOME", str(tmp_path))
    other = FakeReader(episodes=1)
    other.repo_id = "lerobot/aloha"
    sw.save(sw.run(FakeHandle(), other, METRIC, frame_stride=50))

    assert sw.forget(DATASET, POLICY, METRIC) == {"rows": 4, "runs": 1}
    assert sw.retrieve("lerobot/aloha", POLICY, METRIC).n_frames == 2


def test_forget_leaves_the_datasets_other_metric_alone(tmp_path, monkeypatch):
    """The half a differently-keyed dataset cannot check.

    An occlusion sweep and an entropy sweep of the same dataset are separate
    measurements under separate keys, and the expensive one is dozens of tower
    passes per frame. A delete keyed on the dataset alone would drop it while
    reporting the entropy sweep's row count — the caller reads a number that
    matches what they asked to remove and does not notice what went with it.
    """
    ran = _saved(tmp_path, monkeypatch, frame_stride=50)
    occlusion = sw.Sweep(
        metric="occlusion_peak",
        unit=sw.METRICS["occlusion_peak"][1],
        dataset=DATASET,
        policy=POLICY,
        camera=ran.camera,
        episode_stride=1,
        frame_stride=50,
        rows=[sw.Row(episode=0, timestep=0, value=0.5)],
        n_frames=1,
        n_episodes=1,
        frames_total=200,
    )
    sw.save(occlusion)

    assert sw.forget(DATASET, POLICY, METRIC) == {"rows": 4, "runs": 1}

    kept = sw.retrieve(DATASET, POLICY, "occlusion_peak")
    assert kept.n_frames == 1
    assert kept.unit == sw.METRICS["occlusion_peak"][1]


# ------------------------------------------------------------- the migration


def test_a_database_written_before_the_run_table_still_opens(tmp_path, monkeypatch):
    """The migration itself, against a file holding only the old schema.

    `_db()` runs the whole script on every connection, which is what happens to
    a real `traces.sqlite` on upgrade: the new `CREATE TABLE IF NOT EXISTS`
    adds an empty table, the existing rows are untouched, and `stored()` —
    which never learned about the run table — answers exactly as before.
    """
    monkeypatch.setenv("MODELMRI_HOME", str(tmp_path))
    path = paths.trace_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    old = sqlite3.connect(str(path))
    old.executescript(
        """
        CREATE TABLE IF NOT EXISTS vla_sweep (
            dataset TEXT NOT NULL, policy TEXT NOT NULL, metric TEXT NOT NULL,
            camera TEXT NOT NULL, episode INTEGER NOT NULL,
            timestep INTEGER NOT NULL, value REAL NOT NULL,
            stride INTEGER NOT NULL, taken_at TEXT NOT NULL,
            PRIMARY KEY (dataset, policy, metric, camera, episode, timestep)
        );
        """
    )
    old.execute(
        "INSERT INTO vla_sweep VALUES (?,?,?,?,?,?,?,?,?)",
        (DATASET, "p", METRIC, "cam", 0, 0, 1.25, 25, "2026-01-01T00:00:00+00:00"),
    )
    old.commit()
    old.close()

    rows = sw.stored(DATASET, "p", METRIC)
    assert [r["value"] for r in rows] == [1.25]
    assert rows[0]["stride"] == 25

    # And a new sweep saved into that same file works, rather than failing on a
    # table the old install never created.
    sw.save(sw.run(FakeHandle(), FakeReader(episodes=1), METRIC, frame_stride=50))
    assert sw.retrieve(DATASET, POLICY, METRIC).n_frames == 2
