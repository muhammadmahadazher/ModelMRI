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
def held():
    """A runtime whose load slot is already taken.

    `_in_flight` used to be stubbed here. It is not any more: both halves of
    the refusal are now built from the snapshot `_snap` installs, so stubbing
    one of them would hide the half these tests exist to pin.
    """
    rt = ModelRuntime.__new__(ModelRuntime)
    rt._lock = threading.Lock()
    rt._lock.acquire()
    return rt


def _refuse(rt, action: str) -> str:
    with pytest.raises(Refusal) as caught:
        with rt._load_slot(action):
            pass
    return caught.value.sentence


def _snap(monkeypatch, *, active: bool = True, **fields):
    """The tracker's answer for one test.

    `active` IS A PARAMETER. It was hardcoded True, so every snapshot these
    tests built described a live load and the `active is False` branch had
    zero coverage — which is the branch that was wrong, and the one a second
    click actually lands in.
    """
    snap = progress.Snapshot(active=active, hf_id="google/gemma-2-2b", **fields)
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


def test_a_held_slot_with_no_live_load_invents_no_diagnosis(held, monkeypatch):
    """`active is False` — the window a second click most often lands in.

    `unload()` takes this slot at the top of its body and never starts the
    tracker, so `active` stays False for the whole teardown: epoch bump,
    dereference, `gc.collect()`, cache clear. That scales with the model being
    freed, which is exactly when somebody clicks again.

    The snapshot is then all defaults, and every branch below `active` reads
    those defaults as evidence: `stage` is "" and `bytes_total` is 0, which
    progress.py documents as "unknown", not "the download finished". So the
    terminal branch fired and said, over plain alternating load/unload HTTP
    with nothing ever loaded and no Hub request ever made:

        Cannot load 'Qwen/Qwen3-1.7B' yet: another load is already running.
        The weights have finished arriving and this load is now inside
        transformers ... restarting the server is the way out.

    Every clause of that was false. Measured at 10 of 38 rounds over 90s.
    """
    _snap(monkeypatch, active=False)
    said = _refuse(held, "load 'Qwen/Qwen3-1.7B'")

    assert "weights have finished arriving" not in said
    assert "restarting the server" not in said
    assert "larger than this machine can hold" not in said
    # And it does not claim a running load either, which is the other half of
    # the same invention: the tracker had nothing to name.
    assert "another load is already running" not in said

    # What is left is what was actually known, plus a step that fits whichever
    # way it resolves.
    assert "the load slot is already held" in said
    assert "nothing to Stop" in said
    assert "try again" in said


def test_both_halves_of_the_refusal_describe_one_instant(held, monkeypatch):
    """One snapshot, read once, passed to both halves.

    `_in_flight` and `_way_out` each took their own. A load that ended between
    the two calls put a live sentence and a dead one in the same refusal — a
    model named as "still loading" followed by advice computed from an empty
    tracker. The counter here fails on any second read.
    """
    taken = []

    def _snapshot():
        # Deliberately a DIFFERENT answer the second time, so a second read
        # cannot pass unnoticed.
        snap = progress.Snapshot(
            active=not taken, hf_id="google/gemma-2-2b", stage="resolving"
        )
        taken.append(snap)
        return snap

    monkeypatch.setattr(progress.TRACKER, "snapshot", _snapshot)
    said = _refuse(held, "unload")

    assert len(taken) == 1
    # Both halves from the live snapshot, not one of each.
    assert "has been loading" in said
    assert "Press Stop" in said


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
