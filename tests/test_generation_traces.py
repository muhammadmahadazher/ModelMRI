"""The agents panel used to be unreachable from the app it lives in.

Loading a model, typing a prompt and generating — the thing the whole page
exists for — put nothing in AGENTS — RECORDED RUNS, on any model, on either
backend, ever. The only way to fill it was to add `modelmri-record` to a
program of your own and run that somewhere else. Nothing on screen said so,
so the panel read as broken, and was reported as broken.

A generation IS an llm_call: a prompt in, text out, a duration, a token
count, and sometimes a failure. The store has held exactly that shape since
it was written. These tests hold the wiring between them:

  * a committed generation lands in the store, over REST and over the socket
  * it carries the prompt, the output, the model, the timing and the counts
  * a failed generation is recorded AS failed, with whatever it managed
  * the tool's own throwaway probes (commit=False) stay out of the list
  * app runs are labelled, and not by borrowing the `demo` flag
  * a store that cannot write does not cost you the generation

Every one of them fails against the version where the panel only ever read
traces posted from outside.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from modelmri.server import create_app


def app_with(tmp_path, pieces=("Hello", " world"), fail: BaseException | None = None):
    """An app whose model is a fake generator, so nothing is downloaded.

    Every patch here is on the runtime INSTANCE this app just built. `loaded`
    is a read-only property on the class, so the tempting version — assigning
    a property to `type(runtime)` and deleting it after — reaches every other
    test in the session, and deleting it does not put the real one back.
    Satisfying the property instead costs one attribute and leaks nothing.
    """
    app = create_app(trace_db=str(tmp_path / "t.sqlite"))

    def fake(prompt, max_new_tokens=256, temperature=0.7, commit=True, **_):
        yield from pieces
        if fail is not None:
            raise fail

    app.state.runtime.generate_stream = fake
    app.state.runtime.hf_id = "acme/tiny-1b"
    app.state.runtime.backend = "hf"
    # What `loaded` actually reads. Nothing under test dereferences it: the
    # generator above is the whole model.
    app.state.runtime.model = object()
    assert app.state.runtime.loaded
    return app


def traces_of(client: TestClient) -> list[dict]:
    return client.get("/api/traces").json()


# --------------------------------------------------------------- the REST path


def test_a_generation_through_the_app_lands_in_the_panel(tmp_path):
    """The headline. Generate here, and the panel that says RECORDED RUNS has
    a run in it — without instrumenting anything."""
    app = app_with(tmp_path)
    c = TestClient(app)
    assert traces_of(c) == []

    r = c.post("/api/model/prompt", json={"prompt": "why is the sky blue?"})
    assert r.status_code == 200
    assert r.json()["generation"] == "Hello world"

    listing = traces_of(c)
    assert len(listing) == 1, "the generation was not recorded"
    assert listing[0]["n_steps"] == 1
    assert listing[0]["n_errors"] == 0

    doc = c.get(f"/api/traces/{listing[0]['id']}").json()
    (step,) = doc["steps"]
    assert step["kind"] == "llm_call"
    assert step["input"] == "why is the sky blue?"
    assert step["output"] == "Hello world"
    assert step["error"] is False


def test_the_trace_names_the_model_that_ran(tmp_path):
    """Any model, any backend — the id is read off the runtime, never a
    constant. The panel groups runs by name, so this is also what makes
    repeated generations on one model collapse into one row."""
    app = app_with(tmp_path)
    c = TestClient(app)
    c.post("/api/model/prompt", json={"prompt": "hi"})
    listing = traces_of(c)
    assert listing[0]["name"] == "acme/tiny-1b"

    doc = c.get(f"/api/traces/{listing[0]['id']}").json()
    assert doc["meta"]["model"] == "acme/tiny-1b"
    assert doc["meta"]["backend"] == "hf"


def test_the_trace_carries_timing_and_token_counts(tmp_path):
    """The playground already reports "257 tok · 14.12s · 18.2 tok/s" for the
    same run. A recording of that run that cannot say how long it took or how
    much came out is not a recording of it."""
    app = app_with(tmp_path, pieces=("a", "b", "c", "d"))
    app.state.runtime.count_tokens = lambda text: len(text.split())
    c = TestClient(app)
    c.post("/api/model/prompt", json={"prompt": "two words"})
    doc = c.get(f"/api/traces/{traces_of(c)[0]['id']}").json()
    (step,) = doc["steps"]
    assert step["tokens_in"] == 2
    assert step["tokens_out"] == 4
    assert step["duration_ms"] >= 0


def test_a_backend_that_cannot_count_tokens_says_so_rather_than_guessing(tmp_path):
    """Ollama tokenises in its own process and streams back text with no
    counts in it. The honest answer is no number, not a character count
    rendered as a token count."""
    app = app_with(tmp_path, pieces=("x", "y"))
    app.state.runtime.backend = "ollama"
    app.state.runtime.tokenizer = None
    c = TestClient(app)
    c.post("/api/model/prompt", json={"prompt": "hi"})
    (step,) = c.get(f"/api/traces/{traces_of(c)[0]['id']}").json()["steps"]
    assert step["tokens_in"] is None
    # The streamed pieces are still a count the app itself displays, and
    # they are available on every backend.
    assert step["tokens_out"] == 2


# ------------------------------------------------------------------ the socket


def test_the_playground_socket_records_its_run(tmp_path):
    """The path the page actually generates on. If only the REST route were
    wired, the panel would still be empty for everybody using the UI."""
    app = app_with(tmp_path, pieces=("to", "ken"))
    c = TestClient(app)
    with c.websocket_connect("/ws/generate") as ws:
        ws.send_text(json.dumps({"prompt": "count to three"}))
        assert ws.receive_json() == {"type": "token", "text": "to"}
        assert ws.receive_json() == {"type": "token", "text": "ken"}
        assert ws.receive_json() == {"type": "done"}

    listing = traces_of(c)
    assert len(listing) == 1
    (step,) = c.get(f"/api/traces/{listing[0]['id']}").json()["steps"]
    assert step["input"] == "count to three"
    assert step["output"] == "token"


def test_the_run_is_stored_before_the_socket_says_done(tmp_path):
    """The panel refreshes when the window regains focus and when the
    playground finishes. Filing the trace after the final frame would mean the
    refresh triggered by "done" races the write it is refreshing for."""
    app = app_with(tmp_path)
    c = TestClient(app)
    with c.websocket_connect("/ws/generate") as ws:
        ws.send_text(json.dumps({"prompt": "hi"}))
        while ws.receive_json()["type"] == "token":
            pass
        # The terminal frame has arrived; the run must already be there.
        assert len(traces_of(c)) == 1


# ------------------------------------------------------------------- failures


def test_a_failed_generation_is_recorded_as_failed(tmp_path):
    """A run that died is the run you most want a record of. It keeps what it
    managed to produce — a stream that stopped after two tokens produced those
    two tokens, and throwing them away loses how far it got."""
    app = app_with(tmp_path, pieces=("half an ans",), fail=RuntimeError("CUDA OOM"))
    c = TestClient(app)
    with c.websocket_connect("/ws/generate") as ws:
        ws.send_text(json.dumps({"prompt": "hi"}))
        while (frame := ws.receive_json())["type"] == "token":
            pass
        assert frame["type"] == "error"

    listing = traces_of(c)
    assert listing[0]["n_errors"] == 1, "the failure was recorded as a success"
    (step,) = c.get(f"/api/traces/{listing[0]['id']}").json()["steps"]
    assert step["error"] is True
    assert "half an ans" in step["output"]
    # The reader's sentence, not torch's — a trace is a document people
    # export and attach to issues, and torch's text names paths on this
    # machine.
    assert "CUDA OOM" not in step["output"]
    assert "failed mid-generation" in step["output"]


def test_a_refusal_is_recorded_in_the_words_the_reader_saw(tmp_path):
    """Ollama going away is a deliberate no with a sentence somebody wrote.
    That sentence is safe to keep, and it is the one that explains the row."""
    from modelmri.errors import Refusal

    words = "ollama unreachable at http://127.0.0.1:11434: Connection refused."
    app = app_with(tmp_path, pieces=(), fail=Refusal(words))
    c = TestClient(app)
    r = c.post("/api/model/prompt", json={"prompt": "hi"})
    assert r.status_code == 409

    listing = traces_of(c)
    assert len(listing) == 1
    assert listing[0]["n_errors"] == 1
    (step,) = c.get(f"/api/traces/{listing[0]['id']}").json()["steps"]
    assert words in step["output"]


# ------------------------------------------------- what does NOT get recorded


def test_the_steering_probes_stay_out_of_the_list(tmp_path):
    """`commit=False` is the A/B firing two throwaway completions to compare.
    Recording them would put four rows in the panel per comparison and bury
    the run they were comparing."""
    app = app_with(tmp_path)
    c = TestClient(app)
    r = c.post("/api/model/prompt", json={"prompt": "hi", "commit": False})
    assert r.status_code == 200
    assert traces_of(c) == []


# ------------------------------------------------------------ telling them apart


def test_app_runs_are_labelled_without_borrowing_the_demo_flag(tmp_path):
    """`demo` means "sample data shipped with ModelMRI, not something you
    produced". A generation you just made is the opposite claim, and reusing
    the flag would file your own runs behind the "Remove sample" button."""
    app = app_with(tmp_path)
    c = TestClient(app)
    c.post("/api/model/prompt", json={"prompt": "hi"})
    c.post(
        "/api/traces/import",
        json={
            "name": "my-agent",
            "started_at": "2026-08-12T00:00:00Z",
            "steps": [{"kind": "llm_call", "started_ms": 0, "duration_ms": 1}],
        },
    )

    by_name = {t["name"]: t for t in traces_of(c)}
    assert by_name["acme/tiny-1b"]["source"] == "app"
    assert by_name["acme/tiny-1b"]["demo"] is False
    # A trace somebody instrumented themselves is unlabelled, exactly as
    # before — this is additive, and every trace written before it exists
    # reads back the same way.
    assert by_name["my-agent"]["source"] == ""
    assert by_name["my-agent"]["demo"] is False


def test_clear_my_runs_removes_them(tmp_path):
    """They are your runs. The button that says so has to mean them too."""
    app = app_with(tmp_path)
    c = TestClient(app)
    c.post("/api/model/prompt", json={"prompt": "hi"})
    # The DELETE is hoisted out of the assert on purpose. Inside one, `python
    # -O` strips the statement and the deletion never happens -- so the check
    # below would be testing a store nobody cleared, which is this project's
    # own "a green result from a check that did not run".
    deleted = c.delete("/api/traces?keep_demo=true").json()["deleted"]
    assert deleted == 1
    assert traces_of(c) == []


# --------------------------------------------------------------- the contract


def test_a_store_that_cannot_write_does_not_cost_you_the_generation(tmp_path):
    """The recorder's contract, and it outranks the recording: recording must
    never crash the host app. Here the host is somebody's generation, and a
    reader watching tokens arrive must not lose them to a failed INSERT."""
    app = app_with(tmp_path)

    def broken(_doc):
        raise RuntimeError("disk full")

    app.state.traces.import_trace = broken
    c = TestClient(app)
    r = c.post("/api/model/prompt", json={"prompt": "hi"})
    assert r.status_code == 200, "a failed recording took the generation with it"
    assert r.json()["generation"] == "Hello world"

    with c.websocket_connect("/ws/generate") as ws:
        ws.send_text(json.dumps({"prompt": "hi"}))
        frames = [ws.receive_json() for _ in range(3)]
    assert frames[-1] == {"type": "done"}


def test_recording_adds_no_network_path(tmp_path):
    """Prompts and generations are the reader's own text. They go to the same
    SQLite file the panel already reads, on this machine, and nowhere else."""
    import pathlib

    import modelmri.traces as traces_mod

    source = pathlib.Path(traces_mod.__file__).read_text(encoding="utf-8")
    for outbound in ("urllib", "requests", "httpx", "socket"):
        assert outbound not in source, f"traces.py grew an outbound path: {outbound}"
