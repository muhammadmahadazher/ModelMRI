# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""The `.rrd` writer, verified by Rerun's own reader rather than by eye.

`write_rrd` refused unconditionally until 2026-08-27, and the reason it gave
was honest: "nothing here has ever been run against an installed rerun-sdk, so
emitting one would be publishing a file whose correctness is a guess." This
file is what makes that no longer true, and it does it the only way that
counts — it writes a real file and hands it to `rerun rrd verify`, whose whole
job is to say whether a recording "can be loaded and correctly interpreted".

Asserting on the bytes instead would be the proxy-for-a-property mistake this
project has now made twice: a file that starts with the right magic number and
is a plausible size is exactly what a broken writer produces.

Two things are tested WITHOUT rerun installed, because CI has no rerun and
these are the parts that decide whether a user ever gets a file:

  * the analytics gate. rerun ships usage analytics ENABLED and ModelMRI's
    front page says it has no telemetry. `rerun_analytics` returning None —
    "could not tell" — must refuse, because an unknown answer to a privacy
    question is not a yes.
  * the refusal sentences, which are the entire product when the writer is
    absent.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from modelmri import robot_export as rx
from modelmri.errors import Refusal

rerun = pytest.importorskip


def _has_rerun() -> bool:
    try:
        import rerun  # noqa: F401
    except ImportError:
        return False
    return rx.rerun_cli() is not None


needs_rerun = pytest.mark.skipif(
    not _has_rerun(), reason="rerun-sdk is not installed on this machine"
)


def _timeline():
    """Two tracks and a frame — enough shape that a writer can drop one."""
    return rx.Timeline(
        provenance=rx.Provenance(
            tool="modelmri",
            tool_version="0.12.0",
            dataset="lerobot/pusht",
            camera="observation.images.top",
            policy="lerobot/smolvla_base",
            policy_revision="abc123",
            measured_by="attention_entropy",
            taken_at="2026-08-27T00:00:00Z",
            mri_pointer="pusht-ep0.mri",
        ),
        clock=rx.Clock(
            kind="timestep",
            fps=10.0,
            sentence="one tick per recorded frame at 10 fps",
        ),
        tracks=(
            rx.Track(
                metric="attention_entropy",
                unit="nats",
                resolution="per frame",
                episode=0,
                samples=(rx.Sample(0, 1.5), rx.Sample(10, 1.75), rx.Sample(20, 1.25)),
                frame_stride=10,
            ),
            rx.Track(
                metric="attention_entropy",
                unit="nats",
                resolution="per frame",
                episode=1,
                samples=(rx.Sample(0, 0.5), rx.Sample(10, 0.75)),
                frame_stride=10,
            ),
        ),
        omitted=("the attention grids themselves stay in the .mri",),
    )


# ------------------------------------------------ the gate, with no rerun


def test_an_unknown_analytics_answer_refuses_rather_than_assuming_no(monkeypatch):
    """ "Could not tell" is not "it is off".

    This is the `?? 0` bug pointed at a privacy promise: a caller that treats a
    missing answer as a safe one publishes through a library that may be
    reporting usage, having told the user it does not.
    """
    monkeypatch.setattr(rx, "rerun_analytics", lambda: (None, "the CLI vanished"))
    monkeypatch.setitem(__import__("sys").modules, "rerun", __import__("types"))
    (available, reason), package = rx.writer_available("rrd")
    assert available is False
    assert package == "rerun-sdk"
    assert "not a no" in reason
    assert "rerun analytics disable" in reason


def test_analytics_enabled_refuses_and_names_the_exact_command(monkeypatch):
    monkeypatch.setattr(rx, "rerun_analytics", lambda: (True, "reported by rerun"))
    monkeypatch.setitem(__import__("sys").modules, "rerun", __import__("types"))
    (available, reason), _ = rx.writer_available("rrd")
    assert available is False
    assert "ENABLED" in reason
    # The refusal has to be actionable in one command, or it is a complaint.
    assert "rerun analytics disable" in reason
    assert "no telemetry" in reason


def test_writing_without_the_package_refuses_with_the_install_command(monkeypatch):
    monkeypatch.setattr(
        rx, "writer_available", lambda c: ((False, "install rerun-sdk"), "rerun-sdk")
    )
    with pytest.raises(Refusal, match="install rerun-sdk"):
        rx.write_rrd(_timeline(), "unused.rrd")


def test_the_entity_path_is_namespaced_so_a_merge_stays_readable():
    track = _timeline().tracks[0]
    assert rx.rrd_entity(track) == "modelmri/episode_0/attention_entropy"


def test_the_provenance_block_names_the_writer_version_and_what_is_missing():
    body = rx._rrd_provenance(_timeline(), "9.9.9")
    assert "rerun-sdk 9.9.9" in body
    assert "read by the version that wrote it" in body
    # `omitted` is the honesty half: what ModelMRI measured that is NOT here.
    assert "Not in this file" in body
    assert "attention grids themselves stay in the .mri" in body


# --------------------------------------- the round trip, with rerun present


@needs_rerun
def test_the_written_file_loads_in_rerun(tmp_path):
    """The assertion this whole module was blocked on.

    `rerun rrd verify` exists to answer exactly one question — can this
    recording be loaded and correctly interpreted — and a zero exit is that
    answer. Nothing here inspects bytes.
    """
    out = tmp_path / "run.rrd"
    receipt = rx.write_rrd(_timeline(), out)

    assert out.is_file()
    assert receipt["bytes_written"] == out.stat().st_size > 0
    assert receipt["container"] == "rrd"
    assert receipt["writer_version"]

    cli = rx.rerun_cli()
    done = subprocess.run(
        [str(cli), "rrd", "verify", str(out)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, (
        f"rerun could not load the file this module wrote:\n{done.stdout}\n{done.stderr}"
    )


@needs_rerun
def test_every_track_and_the_provenance_reach_the_file(tmp_path):
    """The entities are counted from the file, not from what we meant to log.

    A writer that silently dropped the second episode would still produce a
    file `verify` is happy with — "it loads" and "it holds what you measured"
    are different claims and this is the second one.
    """
    out = tmp_path / "run.rrd"
    rx.write_rrd(_timeline(), out)
    cli = rx.rerun_cli()
    done = subprocess.run(
        [str(cli), "rrd", "stats", str(out)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    text = done.stdout
    # Two tracks plus the provenance document. `num_entity_paths` is Rerun's
    # own count of what it found after loading.
    match = [ln for ln in text.splitlines() if "num_entity_paths" in ln]
    assert match, f"stats printed no entity count:\n{text}"
    found = int(match[0].split("=")[1])
    assert found >= 3, f"expected at least 2 tracks + provenance, got {found}"


@needs_rerun
def test_the_same_timeline_writes_the_same_recording_id(tmp_path):
    """Two exports of one measurement must merge, not sit side by side.

    `rr.RecordingStream` defaults to a fresh uuid per call, which would make
    every re-export a different recording to anyone merging them.
    """
    first, second = tmp_path / "a.rrd", tmp_path / "b.rrd"
    rx.write_rrd(_timeline(), first)
    rx.write_rrd(_timeline(), second)
    cli = rx.rerun_cli()
    ids = []
    for path in (first, second):
        done = subprocess.run(
            [str(cli), "rrd", "stats", str(path)],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        line = [ln for ln in done.stdout.splitlines() if "StoreId" in ln]
        assert line, done.stdout
        ids.append(line[0].strip())
    assert ids[0] == ids[1], f"recording ids differ:\n{ids[0]}\n{ids[1]}"


@needs_rerun
def test_a_frame_travels_as_an_encoded_image(tmp_path):
    """A PNG must not be re-encoded on the way in — it is already a PNG."""
    png = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
        b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    timeline = _timeline()
    with_frame = rx.Timeline(
        provenance=timeline.provenance,
        clock=timeline.clock,
        tracks=timeline.tracks,
        frame=rx.Frame(png=png, width=1, height=1, episode=0, timestep=10),
        omitted=timeline.omitted,
    )
    out = tmp_path / "frame.rrd"
    rx.write_rrd(with_frame, out)
    cli = rx.rerun_cli()
    done = subprocess.run(
        [str(cli), "rrd", "verify", str(out)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, f"{done.stdout}\n{done.stderr}"


@needs_rerun
def test_the_analytics_probe_answers_from_rerun_rather_than_a_guess():
    """The probe reads rerun's own config, so the path is never hardcoded."""
    enabled, detail = rx.rerun_analytics()
    assert enabled in (True, False), f"could not read the config: {detail}"
    assert detail
    cli = rx.rerun_cli()
    done = subprocess.run(
        [str(cli), "analytics", "config"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    body = done.stdout[done.stdout.find("{") : done.stdout.rfind("}") + 1]
    assert json.loads(body)["analytics_enabled"] is enabled


@needs_rerun
def test_shutil_which_is_not_how_the_cli_is_found():
    """The bundled CLI, not one on PATH.

    A `rerun` on PATH can be a different build with a different analytics
    config, which would make the privacy check answer a question about the
    wrong program. The one inside the wheel is the one whose version matches
    the SDK doing the writing.
    """
    found = rx.rerun_cli()
    assert found is not None
    assert found.parent.name == "rerun_cli"
    on_path = shutil.which("rerun")
    if on_path:
        assert str(found) != on_path or found.parent.name == "rerun_cli"
