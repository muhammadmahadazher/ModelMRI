"""A `.mri` carries the causal result, and can be read without a browser.

The format held attention, the logit lens, the prompt and the generation. It
did not hold the activation-patching trace — so the one finding in this tool
that is CAUSAL rather than correlational, "the answer is decided at layer 15,
position 4", was the one finding you could not send to anybody. Opening a
shared recording and pressing the patch button got a refusal, correctly:
patching means running the model again with an activation replaced, and a
`.mri` holds activations rather than weights.

Carrying the grids fixes that, and the section is held to the same standard as
`attention`, because a `.mri` is meant to be forwarded and `parse` therefore
runs on bytes a stranger sent. A grid reaches the viewer as nested loop bounds
and its values reach a colour scale.

`modelmri inspect` is the other half: `open` starts a viewer, and someone
triaging an issue with six attached recordings wants to know which is which
without six browser tabs.
"""

from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path

import pytest

from modelmri import session

ROOT = Path(__file__).resolve().parents[1]

TRACE = {
    "grids": {
        "resid": [[0.02, 0.91], [0.15, 0.44]],
        "attn": [[0.00, 0.31], [0.09, 0.02]],
    },
    "sites": [{"layer": 0, "position": 1}],
    "notes": ["mlp skipped: this architecture has no separate mlp submodule"],
    "clean": "The Eiffel Tower is located in the city of",
    "corrupt": "The Colosseum is located in the city of",
    # WHAT READING THE GRID NEEDS. Its columns are token positions, and the
    # recovery numbers mean nothing without the pair of answers the gap was
    # measured between. The section carried neither, so it parsed and served
    # and mounted and could not be read.
    "components": ["resid", "attn"],
    "tokens": {"clean": ["The", " Eiffel"], "corrupt": ["The", " Colosseum"]},
    "answers": {
        "clean": {"text": " Paris", "p": 0.71},
        "corrupt": {"text": " Rome", "p": 0.66},
    },
}


def make(patch=None, **over) -> bytes:
    kw = dict(
        model_id="gpt2",
        device="cpu",
        dtype="float32",
        n_params=124_439_808,
        tokens=["a", "b"],
        prompt="a",
        generation="b",
        attention={(0, 0): [[1.0, 0.0], [0.5, 0.5]]},
        n_layers=2,
        n_heads=2,
    )
    kw.update(over)
    return session.build(patch=patch, **kw)


def repack(raw: bytes, patch) -> bytes:
    """A file whose patch section says exactly what we want it to."""
    doc = json.loads(gzip.decompress(raw))
    doc["patch"] = patch
    return gzip.compress(json.dumps(doc).encode("utf-8"))


# ------------------------------------------------------------- carrying it


def test_a_trace_survives_the_round_trip():
    got = session.parse(make(TRACE))
    assert got.has_patch()
    assert got.patch["grids"]["resid"] == [[0.02, 0.91], [0.15, 0.44]]
    assert got.patch["clean"].startswith("The Eiffel")
    assert got.patch["corrupt"].startswith("The Colosseum")
    assert got.patch["notes"] == TRACE["notes"]


def test_a_session_without_a_trace_does_not_claim_one():
    """An empty `patch` key would make every reader render an empty section."""
    raw = make()
    assert "patch" not in json.loads(gzip.decompress(raw))
    assert session.parse(raw).has_patch() is False


def test_an_empty_grid_set_is_not_written():
    raw = make({"grids": {}, "notes": ["nothing to measure"]})
    assert "patch" not in json.loads(gzip.decompress(raw))


def test_older_files_still_open():
    """Additive, which is why the format version does not move: a file written
    before this has no `patch` key and must not become unreadable."""
    doc = json.loads(gzip.decompress(make(TRACE)))
    doc.pop("patch")
    got = session.parse(gzip.compress(json.dumps(doc).encode("utf-8")))
    assert got.has_patch() is False
    assert got.tokens == ["a", "b"]


def test_the_grids_cost_little():
    """A 12x8 trace over three components is the common case, and a `.mri` is
    something people attach to an issue."""
    grids = {
        c: [[i * 0.01 + j * 0.001 for j in range(8)] for i in range(12)]
        for c in ("resid", "attn", "mlp")
    }
    with_trace = len(make({**TRACE, "grids": grids}))
    without = len(make())
    assert with_trace - without < 4096, (
        f"the trace added {with_trace - without} bytes to the file"
    )


# ------------------------------------------------------------ refusing junk


@pytest.mark.parametrize(
    ("bad", "expect"),
    [
        ({"grids": {"r": [[1.0], [1.0, 2.0]]}}, "different lengths"),
        ({"grids": {"r": [[float("nan")]]}}, "not"),
        ({"grids": {"r": [["x"]]}}, "not a number"),
        ({"grids": {"r": "rows"}}, "malformed"),
        ({"grids": []}, "malformed"),
        ({"grids": {"r": [[True]]}}, "not a number"),
        ("a trace, honest", "not a set of fields"),
    ],
)
def test_a_malformed_trace_is_refused_not_dropped(bad, expect):
    """Dropping the section would present a damaged file as an intact one
    that simply has no patching — which is the failure this module exists to
    avoid. Every one of these reaches a browser if it gets through."""
    with pytest.raises(session.SessionError) as err:
        session.parse(repack(make(), bad))
    assert expect in str(err.value)


def test_infinity_is_refused_as_well_as_nan():
    """Both survive a JSON round trip through most writers, and both
    colour-scale to nothing visible."""
    with pytest.raises(session.SessionError):
        session.parse(repack(make(), {"grids": {"r": [[float("inf")]]}}))


def test_a_grid_larger_than_we_will_render_is_refused():
    huge = {"grids": {"r": [[0.0] * 4000 for _ in range(600)]}}
    with pytest.raises(session.SessionError) as err:
        session.parse(repack(make(), huge))
    assert "cells" in str(err.value)


def test_junk_alongside_good_grids_does_not_survive():
    """`sites` and `notes` reach the panel as lists. A string where a list
    belongs used to be spread."""
    got = session.parse(
        repack(
            make(),
            {"grids": {"r": [[0.5]]}, "sites": "nope", "notes": "also nope"},
        )
    )
    assert got.patch["sites"] == []
    assert got.patch["notes"] == []


# ------------------------------------------------------------ reading it


def run_cli(*args) -> subprocess.CompletedProcess:
    import os

    return subprocess.run(
        [sys.executable, "-c", "from modelmri.cli import main; main()", *args],
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )


def test_inspect_describes_a_file(tmp_path):
    mri = tmp_path / "shared.mri"
    mri.write_bytes(make(TRACE, note="the head that carries the city"))
    done = run_cli("inspect", str(mri))
    assert done.returncode == 0, done.stderr
    for expected in ("gpt2", "124M", "the head that carries the city", "attn, resid"):
        assert expected in done.stdout, f"{expected!r} missing from the summary"


def test_inspect_emits_json_on_request(tmp_path):
    mri = tmp_path / "shared.mri"
    mri.write_bytes(make(TRACE))
    done = run_cli("inspect", str(mri), "--json")
    assert done.returncode == 0, done.stderr
    doc = json.loads(done.stdout)
    assert doc["model"] == "gpt2"
    assert doc["patch"]["present"] is True
    assert doc["patch"]["components"] == ["attn", "resid"]
    # The text form clips the prompt for triage; --json is the whole thing.
    assert doc["prompt"] == "a"


def test_inspect_refuses_something_that_is_not_a_session(tmp_path):
    junk = tmp_path / "notes.mri"
    junk.write_text("just some notes", encoding="utf-8")
    done = run_cli("inspect", str(junk))
    assert done.returncode == 2
    assert "not a ModelMRI session" in done.stderr
    assert "Traceback" not in done.stderr


def test_inspect_says_so_when_the_file_is_missing(tmp_path):
    done = run_cli("inspect", str(tmp_path / "nope.mri"))
    assert done.returncode == 2
    assert "no such file" in done.stderr


# Named, and a single literal. Four adjacent strings inside a LIST is the one
# place implicit concatenation is genuinely dangerous: drop a comma anywhere in
# such a list and the two neighbours silently fuse into one argument, which is
# a different command with no syntax error to show for it.
PROBE = """
import sys
from modelmri.cli import inspect_session
HEAVY = ("torch", "transformers", "fastapi", "uvicorn", "numpy")
print(",".join(sorted(m for m in HEAVY if m in sys.modules)))
"""


def test_inspect_does_not_import_torch():
    """Same rule as `open`, and the same reason: 26 seconds of imports to read
    a 54 KB file reads as a hang, and somebody pressed ctrl-c."""
    done = subprocess.run(
        [sys.executable, "-c", PROBE],
        capture_output=True,
        text=True,
        timeout=180,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT)},
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "", (
        f"`modelmri inspect` now imports {done.stdout.strip()}"
    )


def test_inspect_does_not_print_the_exporters_path(tmp_path):
    """A `.mri` is the artefact designed to leave the machine, and `inspect`
    prints its metadata on the recipient's terminal."""
    mri = tmp_path / "shared.mri"
    mri.write_bytes(make(TRACE, model_id="my-checkpoint"))
    done = run_cli("inspect", str(mri))
    assert "C:\\Users" not in done.stdout
    assert "/home/" not in done.stdout


# ----------------------------------------------- and serving it back again


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from modelmri.server import create_app

    with TestClient(create_app()) as c:
        yield c


def test_a_recording_that_carries_a_trace_serves_it(client):
    """The payoff. Opening a shared `.mri` and asking for the patch used to be
    a refusal even when the sender had measured one."""
    assert client.post("/api/session/open", content=make(TRACE)).status_code == 200

    state = client.get("/api/session/state").json()
    assert state["patch"]["available"] is True
    assert state["patch"]["clean"].startswith("The Eiffel")

    got = client.post("/api/patch", json={"clean": "x", "corrupt": "y"})
    assert got.status_code == 200
    body = got.json()
    # Marked, so the panel can say this came out of a file rather than off a
    # model — the numbers are someone else's measurement, not this machine's.
    assert body["recorded"] is True
    assert body["grids"]["resid"] == [[0.02, 0.91], [0.15, 0.44]]
    # The prompts it was measured on travel with it: a grid without its pair
    # is unreadable. NESTED, because a recording now answers in the same shape
    # a live trace does -- `PatchPanel` has one renderer, and it reads
    # `data.clean.prompt`, `data.corrupt.tokens` and `data.clean.answer`.
    # Serving a second shape here is what made the panel show its "shape this
    # page does not know" banner over a file that held the measurement.
    assert body["clean"]["prompt"].startswith("The Eiffel")
    assert body["corrupt"]["prompt"].startswith("The Colosseum")
    # And the parts that make the grid readable rather than merely present.
    assert body["corrupt"]["tokens"] == ["The", " Colosseum"]
    assert body["clean"]["answer"] == {"text": " Paris", "p": 0.71}
    assert body["components"] == ["resid", "attn"]


def test_a_file_written_before_the_labels_still_opens(client):
    """ADDITIVE, not required. A `.mri` exported before the strip was carried
    has no `tokens` and no `answers`, and must still serve its grid rather
    than refusing over a section it was written without.

    Absent stays absent: an empty strip draws a grid with unlabelled columns,
    which is honest. A blank one would label every column with the empty
    string and look like a measurement of nothing.
    """
    old_shape = {k: v for k, v in TRACE.items() if k not in ("tokens", "answers")}
    assert client.post("/api/session/open", content=make(old_shape)).status_code == 200
    body = client.post("/api/patch", json={"clean": "x", "corrupt": "y"}).json()
    assert body["recorded"] is True
    assert body["grids"]["resid"] == [[0.02, 0.91], [0.15, 0.44]]
    assert body["clean"]["prompt"].startswith("The Eiffel")
    assert body["clean"]["tokens"] == []
    assert body["clean"]["answer"] == {}


def test_the_submitted_prompts_cannot_relabel_a_recording(client):
    """A recording answers with the pair it was MEASURED on. Echoing back
    whatever the caller sent would let a grid be presented under a prompt it
    has nothing to do with."""
    client.post("/api/session/open", content=make(TRACE))
    body = client.post(
        "/api/patch", json={"clean": "something else", "corrupt": "also else"}
    ).json()
    assert body["clean"] != "something else"


def test_a_recording_without_a_trace_still_refuses_and_says_why(client):
    """The refusal is still right — a `.mri` holds activations, not weights —
    and it now points at the person who can fix it."""
    assert client.post("/api/session/open", content=make()).status_code == 200
    got = client.post("/api/patch", json={"clean": "x", "corrupt": "y"})
    assert got.status_code == 409
    assert "does not carry a patching trace" in got.json()["error"]


def test_a_live_session_reports_no_recorded_trace(client):
    """`patch` on the state is about the FILE. With nothing open there is no
    file, and the panel must not offer to draw one."""
    assert client.get("/api/session/state").json() == {"open": False}
