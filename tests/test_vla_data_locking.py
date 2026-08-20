"""The reader is shared between requests, and one of its methods freed memory.

`LeRobotV3Reader` is held on `app.state.vla_reader` and every `/api/vla/*`
request uses the same instance. `frame()` and `raw_frame()` decode inside
`self._lock`; `use_camera()` mutated the routing and called `close()` — freeing
the open PyAV container — outside it.

MEASURED on a two-camera LeRobot v3.0 snapshot: a switcher thread against a
decode loop segfaults the process, exit 139, repeatable. The same loop without
the switcher survives 200 iterations. It is not catchable — `except Exception`
never runs, the process dies, and every other request in flight dies with it.
The camera dropdown reaches it, and its effect cleanup does not abort the
in-flight fetch.

These tests assert the INVARIANT rather than reproducing the crash, because a
test that segfaults takes the suite with it and reports nothing. What is
checked is mutual exclusion: while the lock is held, nothing that frees or
re-routes the container may proceed.
"""

from __future__ import annotations

import threading

import pytest

from modelmri.errors import BadRequest
from modelmri.vla_data import LeRobotV3Reader


class _Container:
    """Stands in for the PyAV container, and remembers being freed."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _reader() -> LeRobotV3Reader:
    """A reader with only the state the locking touches.

    Built without `__init__` on purpose: opening a real snapshot needs a
    dataset on disk, and none of that is what these tests are about.
    """
    r = object.__new__(LeRobotV3Reader)
    r._lock = threading.Lock()
    r._cameras = ["observation.image", "observation.wrist"]
    r._video_key = "observation.image"
    r._episodes = ["routing for camera one"]
    r._container = _Container()
    r._container_key = ("some.mp4", "r")
    r.repo_id = "someone/two-cameras"
    return r


def _runs_without_the_lock(call, *, wait: float = 0.4) -> bool:
    """True if `call` completes while the caller holds the reader's lock."""
    done = threading.Event()

    def go():
        try:
            call()
        finally:
            done.set()

    threading.Thread(target=go, daemon=True).start()
    return done.wait(wait)


def test_switching_camera_cannot_free_the_container_mid_decode():
    """THE crash. `use_camera` freed the container a decode was reading from."""
    r = _reader()

    with r._lock:  # stand in for a decode in progress
        escaped = _runs_without_the_lock(lambda: r.use_camera("observation.wrist"))
        held = r._container
        assert escaped is False, (
            "use_camera ran while a decode held the lock — this is the segfault"
        )
        assert held is not None and held.closed is False, (
            "the container a decode is reading from was freed under it"
        )

    # And once the decode is done, the switch goes through.
    r.use_camera("observation.wrist")
    assert r.camera == "observation.wrist"
    assert r._container is None, "the stale container is dropped, just later"


def test_closing_from_another_thread_waits_for_the_decode():
    r = _reader()
    with r._lock:
        assert _runs_without_the_lock(r.close) is False
        assert r._container.closed is False
    r.close()
    assert r._container is None


def test_a_camera_that_is_not_there_is_refused_before_any_locking():
    """The validation is cheap and must not need the lock — a bad name while a
    long decode runs should answer immediately rather than block on it."""
    r = _reader()
    with r._lock:
        with pytest.raises(BadRequest, match="is not a camera"):
            r.use_camera("observation.nonexistent")


def test_switching_to_the_camera_already_selected_does_nothing():
    """The early return must stay outside the lock: it is the common case on
    every panel mount, and taking the lock for it would queue behind a decode
    for no reason."""
    r = _reader()
    before = r._container
    with r._lock:
        r.use_camera("observation.image")
    assert r._container is before
    assert before.closed is False
    assert r._episodes == ["routing for camera one"], "routing was not discarded"


def test_close_is_reentrant_safe_for_internal_callers():
    """`_decode` closes a stale container while holding the lock. Calling the
    public `close` from there would deadlock on a non-reentrant Lock — which is
    a quieter failure than the segfault and no better."""
    r = _reader()
    with r._lock:
        r._close_locked()
        assert r._container is None

    # The public one still works from outside.
    r._container = _Container()
    r.close()
    assert r._container is None


def test_episodes_is_not_read_while_a_camera_change_is_half_applied():
    """`use_camera` swaps `_video_key` and clears `_episodes`. A reader that
    resolved routing before the lock could pair one camera's rows with the
    other's key — a frame from the wrong view with nothing saying so."""
    r = _reader()
    with r._lock:
        assert _runs_without_the_lock(r.episodes) is False, (
            "episodes() read routing while a camera change could be in flight"
        )
