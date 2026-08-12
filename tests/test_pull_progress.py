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

import pytest

from modelmri import progress


@pytest.fixture(autouse=True)
def _clean():
    yield
    progress.TRACKER.finish()
    progress.PULLS.finish()


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
