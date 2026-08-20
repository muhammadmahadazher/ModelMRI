"""The refusal you get when a load is already running.

Reported from the field, with screenshots: a 10.5 GB `google/gemma-2-2b` was
loading on an 8.6 GB card. The reader picked a different model, pressed Load,
and got nothing; pressed Unload, and got

    Cannot load 'unload' yet: another load is running. Stop it first, or wait
    for it to finish.

Two defects in one sentence. `_load_slot`'s parameter was named `hf_id` and
interpolated as a model id, so the three callers that pass an ACTION rather
than a model read as gibberish. And the advice was wrong for the phase they
were in: the meter said "0 bytes left", which means the download had finished
and the load was inside `from_pretrained` — where `TRACKER.cancelled` is never
read again, so Stop does nothing.
"""

from __future__ import annotations

import threading

import pytest

from modelmri import progress
from modelmri.errors import Refusal
from modelmri.runtime import ModelRuntime


@pytest.fixture
def held(monkeypatch):
    """A runtime whose load slot is already taken."""
    rt = ModelRuntime.__new__(ModelRuntime)
    rt._lock = threading.Lock()
    rt._lock.acquire()
    monkeypatch.setattr(rt, "_in_flight", lambda: "'google/gemma-2-2b' is loading")
    return rt


def _refuse(rt, action: str) -> str:
    with pytest.raises(Refusal) as caught:
        with rt._load_slot(action):
            pass
    return caught.value.sentence


def _snap(monkeypatch, **fields):
    snap = progress.Snapshot(active=True, hf_id="google/gemma-2-2b", **fields)
    monkeypatch.setattr(progress.TRACKER, "snapshot", lambda: snap)


def test_each_caller_names_its_own_action(held, monkeypatch):
    """Four callers, one of which passes a model id and three of which do not."""
    _snap(monkeypatch, stage="resolving")
    assert _refuse(held, "unload").startswith("Cannot unload yet")
    assert _refuse(held, "load 'Qwen/Qwen3-1.7B'").startswith(
        "Cannot load 'Qwen/Qwen3-1.7B' yet"
    )
    assert _refuse(held, "run the quantisation comparison").startswith(
        "Cannot run the quantisation comparison yet"
    )
    # The shape that was reported: never again.
    assert "Cannot load 'unload'" not in _refuse(held, "unload")


def test_stop_is_offered_only_while_it_would_do_something(held, monkeypatch):
    """`load` checks `TRACKER.cancelled` once, immediately before
    `from_pretrained`. A stop lands while the weights are arriving and does
    nothing after."""
    _snap(
        monkeypatch,
        stage="weights",
        bytes_done=3_000_000_000,
        bytes_total=10_500_000_000,
    )
    mid = _refuse(held, "unload")
    assert "Press Stop" in mid
    assert "still arriving" in mid


def test_past_the_download_it_says_stop_will_not_help(held, monkeypatch):
    """THE REPORTED CASE. The meter read "0 bytes left" and 311 seconds.

    `stage` is still "weights" here — it does not change between the last byte
    and the end of `from_pretrained` — so branching on the stage told the
    reader to press a button that is no longer read. The byte counts are the
    only signal that separates the two halves.
    """
    _snap(
        monkeypatch,
        stage="weights",
        bytes_done=10_500_000_000,
        bytes_total=10_500_000_000,
        elapsed_s=311,
    )
    past = _refuse(held, "unload")
    assert "Stop will not end this phase" in past
    assert "Press Stop" not in past
    # And it names the likely cause and the actual way out.
    assert "larger than this machine can hold" in past
    assert "restarting the server" in past


def test_an_unknown_size_does_not_promise_stop_either(held, monkeypatch):
    """`bytes_total == 0` means the size could not be read. Under-promising is
    the safe direction: telling somebody to press a button that does nothing
    is worse than telling them it might not help."""
    _snap(monkeypatch, stage="weights", bytes_done=0, bytes_total=0)
    assert "Press Stop" not in _refuse(held, "unload")


def test_the_slot_is_released_so_the_next_caller_gets_it():
    """The guard must not leak the lock it takes."""
    rt = ModelRuntime.__new__(ModelRuntime)
    rt._lock = threading.Lock()
    with rt._load_slot("load 'a'"):
        pass
    with rt._load_slot("load 'b'"):
        pass
    assert rt._lock.acquire(timeout=0.1)
    rt._lock.release()
