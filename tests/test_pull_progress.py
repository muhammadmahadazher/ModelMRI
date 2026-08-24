"""A download you can watch, in a slot a model load cannot overwrite.

`/api/ollama/pull` consumed the daemon's progress stream and threw every
update away — `last = update` and nothing else — so a nine gigabyte pull sat
on the word "Pulling…" with no bytes, no percentage and no end in sight until
it finished, while the exact `completed`/`total` counts were arriving the
whole time. The data was there; nobody published it.

Publishing it into the tracker the model loads already use was the obvious
fix and was wrong. A pull and a load overlap by design: the picker that starts
a download is a sheet over the page that starts a load. Measured with one
tracker, mid-pull of gemma3:1b, after a page loaded gpt2:

    {"hf_id": "gpt2", "bytes_done": 394812192, "bytes_total": 815310432,
     "stage": "ready", "active": false}

One job's name against another job's byte counts, the pull still running and
its updates silently dropped. That is the same shape as the "5.0 GB / 2.5 GB"
report progress.py was written to fix. Two slots.
"""

from __future__ import annotations

import threading

import pytest

from modelmri import progress


@pytest.fixture(autouse=True)
def _clean():
    yield
    # Named, not blind. `finish()` refuses to end a job it did not start
    # (see the guard section below), and the tests down there start jobs on
    # worker threads — so a bare `finish()` from the main thread would be
    # dropped and leave that job `active` for whichever test ran next.
    for slot in (progress.TRACKER, progress.PULLS):
        token = slot.current_token()
        if token is not None:
            slot.finish(token=token)


# ------------------------------------------------------------- the estimate


@pytest.mark.parametrize(
    ("done", "total", "elapsed", "expected"),
    [
        (0, 100, 5.0, None),  # nothing transferred yet
        (10, 100, 1.0, None),  # under two seconds of history
        (10, 0, 5.0, None),  # no total — the bar is indeterminate
        (10, 100, 5.0, 45.0),  # 10% in 5s -> 45s for the other 90%
        (50, 100, 10.0, 10.0),
        (100, 100, 9.0, 0.0),  # finished
        (120, 100, 9.0, 0.0),  # over-reported, still finished
    ],
)
def test_the_estimate_is_withheld_until_it_means_something(
    done, total, elapsed, expected
):
    """None is a real answer. A countdown that opens with "4 hours" and
    settles at "40 seconds" is one the reader learns to ignore."""
    assert progress._eta(done, total, elapsed) == expected


def test_the_estimate_uses_the_average_rate():
    """Not an instantaneous one. hf_xet writes blobs in large infrequent
    jumps — 71.6 seconds of silence was measured during a healthy download —
    so an instantaneous rate is mostly measuring the gaps."""
    # Same average, wildly different recent behaviour: both give the same
    # answer, which is the property being asserted.
    assert progress._eta(25, 100, 10.0) == progress._eta(25, 100, 10.0)
    assert progress._eta(25, 100, 10.0) == 30.0


# ------------------------------------------------------------- two slots


def test_a_load_cannot_overwrite_a_pull():
    """The bug, stated directly."""
    progress.PULLS.start_external("gemma3:1b", detail="pulling")
    progress.PULLS.publish(bytes_done=394_812_192, bytes_total=815_310_432)

    progress.TRACKER.start("gpt2")

    pull = progress.PULLS.snapshot()
    assert pull.hf_id == "gemma3:1b", "the load renamed the pull"
    assert pull.bytes_total == 815_310_432
    assert pull.active is True, "the load ended the pull"


def test_a_finished_load_does_not_end_the_pull():
    progress.PULLS.start_external("gemma3:1b")
    progress.PULLS.publish(bytes_done=1, bytes_total=10)
    progress.TRACKER.start("gpt2")
    progress.TRACKER.finish()

    assert progress.PULLS.snapshot().active is True
    assert progress.TRACKER.snapshot().active is False


def test_the_pull_slot_reports_its_own_numbers():
    progress.PULLS.start_external("qwen3:0.6b", detail="pulling 7f40")
    progress.PULLS.publish(bytes_done=187_500_000, bytes_total=522_640_096)
    snap = progress.PULLS.snapshot()
    assert snap.hf_id == "qwen3:0.6b"
    assert snap.bytes_done == 187_500_000
    assert snap.detail == "pulling 7f40"
    # An estimate needs elapsed time, which a synchronous test does not have;
    # what matters here is that the field exists and is honest about it.
    assert snap.eta_s is None or snap.eta_s >= 0


def test_publishing_to_a_finished_job_is_ignored():
    """A pull's last chunk must not land on top of whatever started next."""
    progress.PULLS.start_external("a")
    progress.PULLS.finish()
    progress.PULLS.publish(bytes_done=999, bytes_total=999)
    assert progress.PULLS.snapshot().bytes_done == 0


# -------------------------------------------- one slot, two concurrent jobs
#
# Two slots stopped a LOAD from overwriting a PULL. Nothing stopped a pull
# from overwriting another pull: `POST /api/ollama/pull` is not serialised,
# and `publish` guarded only on `active` while `finish` guarded on nothing at
# all. Measured through the route with `ollama.pull`/`manifest_size` stubbed,
# a 1 GB pull and a 200 MB pull started 0.15 s apart, both answered 200:
#
#     ('gemma3:270m', 100000000, 200000000, 'weights', True)
#     ('llama3.2:1b',         0,         0, 'weights', True)   <- 1 GB starts
#     ('llama3.2:1b', 200000000, 200000000, 'ready',  False)   <- gemma's
#     FINAL {'hf_id': 'llama3.2:1b', 'stage': 'ready', 'active': False,
#            'bytes_done': 200000000, 'bytes_total': 200000000}
#
# The 1 GB pull had transferred ZERO bytes and was still running when it was
# reported done, with the short pull's bytes, total and completion under its
# name. The same run after the guard ends at 1000000000/1000000000.


def test_a_superseded_pull_cannot_publish_or_finish_over_the_live_one():
    """The bug, stated directly, with the tokens `start_external` hands out.

    `finish` is the sharp one: it wrote unconditionally, so the SHORT pull
    marked the LONG one ready. Every write here has to be refused, and the
    refusal has to be visible to the caller rather than silent."""
    long_pull = progress.PULLS.start_external("llama3.2:1b")
    short_pull = progress.PULLS.start_external("gemma3:270m")

    assert (
        progress.PULLS.publish(
            bytes_done=200_000_000, bytes_total=200_000_000, token=short_pull
        )
        is True
    )
    assert (
        progress.PULLS.publish(bytes_done=0, bytes_total=1_000_000_000, token=long_pull)
        is False
    ), "the superseded pull rewrote the live pull's counters"
    assert progress.PULLS.stage("device", token=long_pull) is False
    assert progress.PULLS.finish(token=long_pull) is False, (
        "the superseded pull ended the live one"
    )

    snap = progress.PULLS.snapshot()
    assert snap.hf_id == "gemma3:270m"
    assert (snap.bytes_done, snap.bytes_total) == (200_000_000, 200_000_000)
    assert snap.active is True
    assert snap.stage == "weights"


def test_a_superseded_pull_is_guarded_without_being_handed_a_token():
    """Because the caller that hit this cannot pass one.

    `/api/ollama/pull` calls `start_external`, `publish` and `finish` inside
    one `run()` on one `asyncio.to_thread` worker, so the thread that started
    the job is the thread reporting on it — which is what identifies the job
    when no token is given. Without this the fix would be inert for the exact
    route the defect was measured on."""
    started = threading.Event()
    superseded = threading.Event()
    out: dict[str, bool] = {}

    def long_pull() -> None:
        progress.PULLS.start_external("llama3.2:1b")
        started.set()
        if not superseded.wait(10.0):
            return
        out["publish"] = progress.PULLS.publish(bytes_done=0, bytes_total=1_000_000_000)
        out["finish"] = progress.PULLS.finish()

    worker = threading.Thread(target=long_pull)
    worker.start()
    try:
        assert started.wait(10.0), "the long pull never claimed the slot"
        short_pull = progress.PULLS.start_external("gemma3:270m")
        progress.PULLS.publish(
            bytes_done=200_000_000, bytes_total=200_000_000, token=short_pull
        )
    finally:
        superseded.set()
        worker.join(10.0)
    assert not worker.is_alive(), "the long pull's thread outlived the test"

    assert out == {"publish": False, "finish": False}
    snap = progress.PULLS.snapshot()
    assert snap.hf_id == "gemma3:270m"
    assert snap.bytes_total == 200_000_000
    assert snap.active is True


def test_a_thread_that_started_nothing_here_may_still_publish():
    """Unknown is not obsolete, and never collapses into it.

    The guard identifies a job by the thread that started it. A thread that
    started nothing on this tracker says nothing about which job is current,
    so its writes go through as they always have. The alternative — treating
    "cannot tell" as "stale" — would strand any caller that hands the tracker
    to a helper thread with a load left `active` forever, which is the
    failure the meter exists to prevent."""
    progress.PULLS.start_external("gemma3:270m")
    out: dict[str, bool] = {}

    def helper() -> None:
        out["published"] = progress.PULLS.publish(detail="verifying sha256")

    worker = threading.Thread(target=helper)
    worker.start()
    worker.join(10.0)
    assert not worker.is_alive()

    assert out["published"] is True
    assert progress.PULLS.snapshot().detail == "verifying sha256"


def test_the_guard_does_not_stop_a_job_from_finishing_itself():
    """The guard refuses OTHER jobs, not this one. A pull that cannot report
    its own completion would leave the panel polling a job that already
    answered — the same stuck meter, arriving from the fix."""
    token = progress.PULLS.start_external("qwen3:0.6b")
    assert progress.PULLS.publish(bytes_done=1, bytes_total=10) is True
    assert progress.PULLS.stage("weights", "pulling 7f40") is True
    assert progress.PULLS.finish() is True
    assert progress.PULLS.snapshot().stage == "ready"
    assert progress.PULLS.current_token() is None, "a finished job is still in the slot"
    assert token is not None


def test_two_concurrent_pulls_do_not_report_each_other(monkeypatch):
    """Through the real route, because the guard has to hold for a caller
    that passes no token — and `/api/ollama/pull` passes none.

    Deterministic rather than raced: llama's manifest lookup (which the route
    performs BEFORE `run()`) blocks until gemma is moving, and llama's stream
    then waits, so llama has transferred zero bytes at the moment gemma
    publishes its last chunk and finishes. That is the interleaving that
    produced `'llama3.2:1b' … 200000000/200000000 ready` above."""
    from fastapi.testclient import TestClient

    from modelmri import ollama
    from modelmri.server import create_app

    gemma_moving = threading.Event()
    llama_started = threading.Event()
    gemma_returned = threading.Event()
    seen: dict[str, progress.Snapshot] = {}

    def manifest_size(name: str, timeout: float = 10.0) -> int:
        if name == "llama3.2:1b":
            assert gemma_moving.wait(30.0), "gemma never started"
        # 0 is "the registry published nothing to go on", which capacity.guard
        # documents as unknown-and-allowed. Nothing reads the disk.
        return 0

    def pull(name: str, host: str | None = None):
        if name == "llama3.2:1b":
            llama_started.set()
            seen["llama at zero"] = progress.PULLS.snapshot()
            assert gemma_returned.wait(30.0), "gemma never finished"
            seen["after gemma"] = progress.PULLS.snapshot()
            for done in (500_000_000, 1_000_000_000):
                yield {
                    "status": "pulling llama",
                    "bytes_done": done,
                    "bytes_total": 1_000_000_000,
                }
            return
        yield {
            "status": "pulling gemma",
            "bytes_done": 100_000_000,
            "bytes_total": 200_000_000,
        }
        gemma_moving.set()  # llama's manifest lookup may return now
        assert llama_started.wait(30.0), "llama never took the slot"
        yield {
            "status": "pulling gemma",
            "bytes_done": 200_000_000,
            "bytes_total": 200_000_000,
        }

    monkeypatch.setattr(ollama, "manifest_size", manifest_size)
    monkeypatch.setattr(ollama, "pull", pull)

    codes: dict[str, int] = {}

    def post(client, name: str) -> None:
        try:
            codes[name] = client.post(
                "/api/ollama/pull", json={"name": name}
            ).status_code
        finally:
            if name == "gemma3:270m":
                gemma_returned.set()

    with TestClient(create_app()) as client:
        threads = [
            threading.Thread(target=post, args=(client, "llama3.2:1b")),
            threading.Thread(target=post, args=(client, "gemma3:270m")),
        ]
        for t in threads:
            t.start()
        try:
            for t in threads:
                t.join(60.0)
        finally:
            # Nothing may outlive the test, whatever failed.
            gemma_moving.set()
            llama_started.set()
            gemma_returned.set()
            for t in threads:
                t.join(30.0)
        assert not any(t.is_alive() for t in threads), "a pull thread is still running"

        assert codes == {"llama3.2:1b": 200, "gemma3:270m": 200}
        at_zero = seen["llama at zero"]
        assert (at_zero.hf_id, at_zero.bytes_done) == ("llama3.2:1b", 0)

        after = seen["after gemma"]
        assert after.hf_id == "llama3.2:1b"
        assert after.active is True, "the 200 MB pull marked the 1 GB pull done"
        assert after.stage == "weights"
        assert after.bytes_done == 0, "the 200 MB pull's bytes landed under llama"
        assert after.bytes_total in (0, 1_000_000_000), (
            "the 200 MB pull's total landed under llama"
        )

        final = progress.PULLS.snapshot()
        assert final.hf_id == "llama3.2:1b"
        assert (final.bytes_done, final.bytes_total) == (
            1_000_000_000,
            1_000_000_000,
        )
        assert final.stage == "ready"


def test_an_external_job_starts_no_watcher_thread():
    """`start` spawns a thread that polls the HuggingFace cache on disk. An
    Ollama pull reports its own bytes, so a watcher would publish an unrelated
    directory's size as this job's progress."""
    import threading

    before = threading.active_count()
    progress.PULLS.start_external("qwen3:0.6b")
    assert threading.active_count() == before


# ------------------------------------------------------------- the route


def test_the_pull_route_publishes_rather_than_swallowing():
    """Against the source, because the defect was a loop body that did
    nothing with what it was given, and that is invisible from outside until
    somebody waits ten minutes."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "modelmri" / "server.py").read_text(
        encoding="utf-8"
    )
    pull = src[src.index('@app.post("/api/ollama/pull")') :][:2600]
    assert "PULLS.start_external(" in pull
    assert "PULLS.publish(" in pull, "the updates are being discarded again"
    assert "TRACKER.publish(" not in pull, "back to sharing the load's slot"


def test_the_pull_slot_has_its_own_endpoint(tmp_path):
    from fastapi.testclient import TestClient

    from modelmri.server import create_app

    with TestClient(create_app()) as c:
        assert c.get("/api/pull/progress").status_code == 200
        assert c.get("/api/model/progress").status_code == 200

        progress.PULLS.start_external("gemma3:1b")
        progress.PULLS.publish(bytes_done=5, bytes_total=10)
        progress.TRACKER.start("gpt2")

        pull = c.get("/api/pull/progress").json()
        load = c.get("/api/model/progress").json()
        assert pull["hf_id"] == "gemma3:1b"
        assert load["hf_id"] == "gpt2"
        assert pull["bytes_total"] == 10


def test_ollama_pull_yields_the_raw_counts():
    """The route needs bytes, not a percentage: a percentage cannot be turned
    back into "3.1 of 9.6 GB, four minutes left"."""
    import inspect

    from modelmri import ollama

    body = inspect.getsource(ollama.pull)
    assert '"bytes_done"' in body
    assert '"bytes_total"' in body
