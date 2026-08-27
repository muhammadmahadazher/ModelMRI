"""`POST /api/vla/export` — the route this file used to carry a comment about.

The comment said the export was a library feature "until `vla_sweep` grows a
retrieval that returns what it stored", and it was right: `robot_export.write`
needs the unit, the two strides and the frame total, and `vla_sweep.stored`
returns bare rows. So the tests here are not about HTTP plumbing. They are
about the two ways a route like this goes wrong:

  * it invents the missing halves — a unit from the metric name, a frame rate
    from the reader's decoding default of 10 — and answers 200 with a file
    that looks measured;
  * it answers 500 for a state somebody could have been told about, which is
    every one of the refusals below.

There is no `mcap` on this machine (`import mcap` raises ModuleNotFoundError),
so the written-file tests drive the same call-recorder stand-in
`tests/test_robot_export.py` uses. They prove what this route hands a writer
and prove nothing about the bytes a real `mcap` would produce.
"""

from __future__ import annotations

import sys
import types

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient
from test_vla_sweep import FakeHandle, FakeReader

from modelmri import robot_export
from modelmri import vla_sweep as sw
from modelmri.errors import Refusal
from modelmri.server import create_app

DATASET = "lerobot/pusht"
POLICY = "lerobot/smolvla_base"
METRIC = "attention_entropy"


# ------------------------------------------------------------------ doubles


class Ep:
    def __init__(self, index, length):
        self.index, self.length = index, length


class DatasetDouble:
    """A LeRobot reader by shape, for the two fields this route reads.

    `fps` and `info` DISAGREE on purpose. `vla_data.LeRobotV3Reader` sets
    `self.fps = int(self.info.get("fps", 10))` so its own timestamp arithmetic
    always has a number; `info` is what the dataset actually published. The
    route has to read the second, and this double is the only place that
    difference is visible.
    """

    def __init__(self, repo_id=DATASET, published_fps=None, episodes=2, length=100):
        self.repo_id = repo_id
        self.info = {} if published_fps is None else {"fps": published_fps}
        self.fps = int(self.info.get("fps", 10))
        self._eps = [Ep(i, length) for i in range(episodes)]

    def episodes(self):
        return self._eps


class Recorder:
    """A call recorder shaped like `mcap.writer.Writer`. NOT a validator.

    It writes each payload to the stream so `write_mcap`'s size arithmetic —
    and its refusal on a zero-byte file — run against a real file rather than
    an empty one.
    """

    made: list[Recorder] = []

    def __init__(self, stream):
        self.stream = stream
        self.calls: list[tuple] = []
        self._next_id = 0
        Recorder.made.append(self)

    def start(self, profile, library):
        self.calls.append(("start", profile, library))

    def register_schema(self, name, encoding, data):
        self._next_id += 1
        self.calls.append(("schema", name))
        return self._next_id

    def register_channel(self, topic, message_encoding, schema_id, metadata):
        self._next_id += 1
        self.calls.append(("channel", topic, metadata))
        return self._next_id

    def add_metadata(self, name, data):
        self.calls.append(("metadata", name, dict(data)))

    def add_message(self, channel_id, log_time, publish_time, sequence, data):
        self.calls.append(("message", channel_id, log_time, sequence, data))
        self.stream.write(data)

    def finish(self):
        self.calls.append(("finish",))

    def metadata(self, name: str) -> dict:
        for call in self.calls:
            if call[0] == "metadata" and call[1] == name:
                return call[2]
        raise AssertionError(f"no {name!r} metadata record was written")


def _install_fake_mcap(monkeypatch):
    Recorder.made = []
    package = types.ModuleType("mcap")
    module = types.ModuleType("mcap.writer")
    module.Writer = Recorder
    package.writer = module
    monkeypatch.setitem(sys.modules, "mcap", package)
    monkeypatch.setitem(sys.modules, "mcap.writer", module)
    return Recorder.made


def _install_dataset(monkeypatch, reader=None, raises=None):
    """Whatever `LeRobotV3Reader.discover` should do for this test."""
    from modelmri import vla_data

    def discover(cls, hf_home=None, repo_id=DATASET):
        if raises is not None:
            raise raises
        return reader if reader is not None else DatasetDouble(repo_id=repo_id)

    monkeypatch.setattr(
        vla_data.LeRobotV3Reader, "discover", classmethod(discover), raising=True
    )


def _store_a_sweep(tmp_path, monkeypatch, *, episodes=2, **kwargs):
    monkeypatch.setenv("MODELMRI_HOME", str(tmp_path))
    kwargs.setdefault("frame_stride", 50)
    ran = sw.run(FakeHandle(), FakeReader(episodes=episodes), METRIC, **kwargs)
    sw.save(ran)
    return ran


def _client() -> TestClient:
    return TestClient(create_app(), raise_server_exceptions=False)


def _body(**overrides) -> dict:
    return {"dataset": DATASET, "policy": POLICY, "metric": METRIC, **overrides}


# ---------------------------------------------------------------- it writes


def test_the_route_that_was_a_comment_now_returns_the_file(tmp_path, monkeypatch):
    """The whole gap, closed end to end: stored rows in, MCAP bytes out, with
    the unit read from the run record rather than from the metric's name."""
    ran = _store_a_sweep(tmp_path, monkeypatch)
    made = _install_fake_mcap(monkeypatch)
    _install_dataset(monkeypatch)

    r = _client().post("/api/vla/export", json=_body())

    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "application/octet-stream"
    assert r.headers["content-disposition"] == (
        'attachment; filename="pusht-attention_entropy.mcap"'
    )
    assert r.content, "a zero-byte export is the failure write_mcap refuses"
    assert int(r.headers["content-length"]) == len(r.content)

    writer = made[0]
    messages = [c for c in writer.calls if c[0] == "message"]
    assert len(messages) == ran.n_frames
    # The unit rides on every message body, and it is the stored one.
    assert all(ran.unit.encode() in c[4] for c in messages)
    assert writer.metadata("modelmri:provenance")["dataset"] == DATASET
    assert writer.metadata("modelmri:provenance")["policy"] == POLICY


def test_the_frame_rate_is_the_datasets_own_not_the_readers_default_of_10(
    tmp_path, monkeypatch
):
    """`vla_data.LeRobotV3Reader` defaults `self.fps` to 10 so its own
    timestamp arithmetic always has a number. Exported as a measured rate that
    default draws a seconds axis nobody timed, over a dataset that published
    none — which is the same fabrication as a defaulted unit, one field down.
    """
    _store_a_sweep(tmp_path, monkeypatch)
    made = _install_fake_mcap(monkeypatch)
    silent = DatasetDouble(published_fps=None)
    assert silent.fps == 10, "the double reproduces the reader's default"
    _install_dataset(monkeypatch, reader=silent)

    assert _client().post("/api/vla/export", json=_body()).status_code == 200

    clock = made[0].metadata("modelmri:clock")
    assert clock["kind"] == "frame-index"
    assert clock["fps"] == "not published by this dataset"
    assert "do not read a duration off this axis" in clock["sentence"]
    # One frame per second under that clock: frame 50 is 50 seconds, NOT the
    # 5 seconds a fabricated 10 fps would have produced.
    stamps = sorted(c[2] for c in made[0].calls if c[0] == "message")
    assert 50 * 1_000_000_000 in stamps


def test_a_published_frame_rate_becomes_a_seconds_axis(tmp_path, monkeypatch):
    """The other half: when the dataset DID publish a rate, the file gets a
    real seconds axis rather than the honest substitute."""
    _store_a_sweep(tmp_path, monkeypatch)
    made = _install_fake_mcap(monkeypatch)
    _install_dataset(monkeypatch, reader=DatasetDouble(published_fps=30))

    assert _client().post("/api/vla/export", json=_body()).status_code == 200

    clock = made[0].metadata("modelmri:clock")
    assert clock["kind"] == "seconds"
    assert clock["fps"].startswith("30")
    # Two episodes carry the same two frame indices, so the DISTINCT stamps
    # are the two instants: frame 0 and frame 50, which at 30 fps is 1.667 s.
    stamps = sorted({c[2] for c in made[0].calls if c[0] == "message"})
    assert stamps == [0, 1_666_666_667]


def test_the_file_states_the_gaps_the_stride_left(tmp_path, monkeypatch):
    """A plot of two points over a hundred frames is the same picture whether
    ninety-eight frames were skipped or failed, and the reader is in another
    application with nobody to ask. The stride sentence travels INTO the file.
    """
    _store_a_sweep(tmp_path, monkeypatch, frame_stride=50)
    made = _install_fake_mcap(monkeypatch)
    _install_dataset(monkeypatch)

    assert _client().post("/api/vla/export", json=_body()).status_code == 200

    omitted = " ".join(made[0].metadata("modelmri:not_exported").values())
    assert "Only one frame in 50 was measured" in omitted
    assert "attention grids are NOT in this file" in omitted
    # The coverage share is the stored one — a run record read as zeros would
    # have produced the "share is NOT stated" sentence instead.
    assert "4 of 200 frames (2.0%)" in omitted


def test_episode_lengths_come_from_the_dataset_rather_than_being_left_blank(
    tmp_path, monkeypatch
):
    """`Track.span` is `None` for "the source did not publish one", and that is
    written into the channel as a sentence. The route opens the dataset, so the
    real number is available and the sentence should not appear."""
    _store_a_sweep(tmp_path, monkeypatch)
    made = _install_fake_mcap(monkeypatch)
    _install_dataset(monkeypatch, reader=DatasetDouble(length=100))

    assert _client().post("/api/vla/export", json=_body()).status_code == 200

    channels = [c[2] for c in made[0].calls if c[0] == "channel"]
    assert channels, "no channel was registered"
    assert all(c["episode_length_frames"] == "100" for c in channels)


def test_the_download_name_is_rebuilt_rather_than_escaped(tmp_path, monkeypatch):
    """It reaches a Content-Disposition header, where a backslash is a
    quoted-string escape and a non-Latin character raises UnicodeEncodeError
    out of Starlette's latin-1 header encoding — which `session_export` learned
    as a bare 500 on every export for those users."""
    _store_a_sweep(tmp_path, monkeypatch)
    _install_fake_mcap(monkeypatch)
    _install_dataset(monkeypatch)

    r = _client().post("/api/vla/export", json=_body(name="../../etc/passаwd\\x"))

    assert r.status_code == 200, r.text
    disposition = r.headers["content-disposition"]
    assert "/" not in disposition and "\\" not in disposition
    assert ".." not in disposition
    assert disposition.isascii()


# -------------------------------------------------------------- it refuses


def test_no_stored_sweep_is_a_sentence_rather_than_an_empty_file(tmp_path, monkeypatch):
    """The route must not reach the writer at all here. An empty timeline in
    Foxglove reads as a policy that produced no measurements."""
    monkeypatch.setenv("MODELMRI_HOME", str(tmp_path))
    _install_fake_mcap(monkeypatch)
    _install_dataset(monkeypatch)

    r = _client().post("/api/vla/export", json=_body())

    assert r.status_code == 422, r.text
    assert "no sweep of attention_entropy is stored" in r.json()["error"]
    assert "Run one" in r.json()["error"]


def test_rows_without_a_run_record_reach_the_reader_as_the_migration(
    tmp_path, monkeypatch
):
    """The state every existing install is in on upgrade. It must arrive as a
    sentence naming the migration, not as a file in a unit this code chose."""
    import sqlite3

    from modelmri import paths

    _store_a_sweep(tmp_path, monkeypatch)
    db = sqlite3.connect(str(paths.trace_db_path()))
    db.execute("DELETE FROM vla_sweep_run")
    db.commit()
    db.close()
    _install_fake_mcap(monkeypatch)
    _install_dataset(monkeypatch)

    r = _client().post("/api/vla/export", json=_body())

    assert r.status_code == 422, r.text
    said = r.json()["error"]
    assert "run record" in said and "Re-run the sweep" in said
    assert sw.METRICS[METRIC][1] not in said, "no unit was supplied from memory"


def test_the_missing_writer_is_409_naming_what_installs_it(tmp_path, monkeypatch):
    """MEASURED on this machine: `import mcap` raises ModuleNotFoundError. The
    refusal is the deliverable in that state, and the retrieval and the plan
    behind it are real — which is why the sentence says so."""
    _store_a_sweep(tmp_path, monkeypatch)
    monkeypatch.setitem(sys.modules, "mcap", None)
    _install_dataset(monkeypatch)

    r = _client().post("/api/vla/export", json=_body())

    assert r.status_code == 409, r.text
    assert "pip install mcap" in r.json()["error"]


def test_rrd_is_declined_while_rerun_reports_usage(tmp_path, monkeypatch):
    """The route hands back the sentence, not a 500.

    `.rrd` writes real files as of 0.13.0; what it still refuses is writing one
    through a library that reports usage while this tool's front page says it
    has no telemetry. The refusal is a 409 carrying a command the reader can
    run, which is the whole product when an export cannot happen.
    """
    _store_a_sweep(tmp_path, monkeypatch)
    _install_fake_mcap(monkeypatch)
    _install_dataset(monkeypatch)
    # Forced, so this asserts the code rather than whether the machine running
    # the suite happens to have rerun installed.
    monkeypatch.setattr(
        robot_export, "rerun_analytics", lambda: (True, "reported by rerun")
    )
    monkeypatch.setitem(sys.modules, "rerun", types.ModuleType("rerun"))

    r = _client().post("/api/vla/export", json=_body(container="rrd"))

    assert r.status_code == 409, r.text
    said = r.json()["error"]
    assert "rerun analytics disable" in said
    assert "no telemetry" in said


def test_an_unknown_container_names_the_ones_that_exist(tmp_path, monkeypatch):
    _store_a_sweep(tmp_path, monkeypatch)
    _install_fake_mcap(monkeypatch)
    _install_dataset(monkeypatch)

    r = _client().post("/api/vla/export", json=_body(container="parquet"))

    assert r.status_code == 422, r.text
    assert "unknown export container" in r.json()["error"]
    assert "mcap" in r.json()["error"]


def test_a_dataset_that_cannot_be_opened_refuses_rather_than_guessing_a_clock(
    tmp_path, monkeypatch
):
    """The route needs `info.json` to know whether this dataset published a
    frame rate. Writing "this dataset published no frame rate" because nobody
    looked is the fabrication one field down from a defaulted unit, so the
    export stops and hands over the reader's own sentence."""
    _store_a_sweep(tmp_path, monkeypatch)
    _install_fake_mcap(monkeypatch)
    _install_dataset(
        monkeypatch, raises=Refusal("lerobot/pusht is not cached. Download it first")
    )

    r = _client().post("/api/vla/export", json=_body())

    assert r.status_code == 409, r.text
    assert "not cached" in r.json()["error"]


def test_a_missing_reader_dependency_names_the_package(tmp_path, monkeypatch):
    """`pyarrow` and `av` are optional, and the neighbouring VLA routes all
    answer their absence with the install command rather than a 500."""
    _store_a_sweep(tmp_path, monkeypatch)
    _install_fake_mcap(monkeypatch)
    _install_dataset(monkeypatch, raises=ImportError("No module named 'pyarrow'"))

    r = _client().post("/api/vla/export", json=_body())

    assert r.status_code == 409, r.text
    assert "modelmri[vla]" in r.json()["error"]


def test_a_key_the_body_does_not_know_is_refused_by_name(tmp_path, monkeypatch):
    """`Body` forbids extras for the reason the class docstring gives: a
    misspelled parameter that is silently dropped produces an answer about
    something else, labelled as though the parameter had applied."""
    _store_a_sweep(tmp_path, monkeypatch)

    r = _client().post("/api/vla/export", json=_body(frame_stride=25))

    assert r.status_code == 422, r.text
    assert "frame_stride" in r.text


def test_the_dataset_is_required_rather_than_taken_from_whatever_is_loaded(
    tmp_path, monkeypatch
):
    """A sweep outlives the process that ran it. Defaulting the dataset to the
    open one would export a ranking of whichever dataset happened to be
    selected when the button was pressed, under the requested metric's name."""
    _store_a_sweep(tmp_path, monkeypatch)

    r = _client().post("/api/vla/export", json={"metric": METRIC})

    assert r.status_code == 422, r.text
    assert "dataset" in r.text
