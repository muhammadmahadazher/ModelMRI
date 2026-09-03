# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""One connection, many threads, and whose job it is to serialise them.

`TraceStore.__init__` opens a single sqlite3 connection with
`check_same_thread=False` and shares it across every request. Python permits
that and does not make it safe — serialising access is the caller's job, and
the module is explicit that only `sqlite3.threadsafety == 3` builds would do
it for you.

Every WRITER in the class did the job. Both READERS did not. So a read
arriving while anything else touched the database ran a second statement on
the same connection, the cursors interleaved, and `fetchall()` came back with
rows of the wrong width — surfacing as `IndexError: tuple index out of range`
in a row mapping whose SELECT has a fixed column count and cannot vary.

It reached the reporter as an intermittent 500 from `GET /api/traces` on a
cold start, when the browser's first page load races the store's own setup.
Before the agents panel had any retry, a single one of those left the panel
empty for the rest of the session — indistinguishable from "you have not
recorded anything", which is exactly what it was reported as.

These tests hammer the store from several threads at once. They are not a
proof of absence: a race that survives is a race that got lucky. What they do
is fail loudly against the unlocked version, which the fix has to beat.
"""

from __future__ import annotations

import itertools
import threading

import pytest

from modelmri.traces import TraceStore


def doc(n: int) -> dict:
    return {
        "id": f"trace-{n}",
        "name": f"agent-{n % 3}",
        "started_at": f"2026-08-12T00:00:{n % 60:02d}Z",
        "meta": {"demo": False},
        "steps": [
            {
                "id": f"s-{n}-{i}",
                "kind": "llm_call",
                "name": f"call {i}",
                "started_ms": i * 10,
                "duration_ms": 5,
                "input": "in",
                "output": "out",
            }
            for i in range(4)
        ],
    }


@pytest.fixture
def store(tmp_path):
    return TraceStore(tmp_path / "traces.sqlite")


def run_threads(fns, seconds: float = 2.0) -> list[BaseException]:
    """Run every callable in its own thread until the deadline. Collect what
    they raise rather than letting a thread die silently — an exception on a
    worker thread is exactly how this bug stayed invisible."""
    import time

    errors: list[BaseException] = []
    stop = threading.Event()
    lock = threading.Lock()

    def wrap(fn):
        def inner():
            while not stop.is_set():
                try:
                    fn()
                except BaseException as err:
                    with lock:
                        errors.append(err)
                    return

        return inner

    threads = [threading.Thread(target=wrap(f), daemon=True) for f in fns]
    for t in threads:
        t.start()
    time.sleep(seconds)
    stop.set()
    for t in threads:
        t.join(timeout=5)
    return errors


def test_listing_while_writing_does_not_return_short_rows(store):
    """The reported failure, directly. `list_traces` maps r[0]..r[6] over a
    SELECT with seven fixed columns, so an IndexError there can only mean the
    row did not come from that statement."""
    # itertools.count, not a finite range. A fast runner gets through tens of
    # thousands of writes inside the window and the thread dies on its own
    # StopIteration, which reads exactly like a real defect -- it failed that
    # way on macOS py3.12 after I had already fixed the identical mistake in
    # the delete test below and not looked for it here.
    counter = itertools.count()

    def write():
        store.import_trace(doc(next(counter)))

    def read():
        for row in store.list_traces():
            # Touch every field the route touches. A short row raises here.
            assert set(row) >= {"id", "name", "started_at", "n_steps", "demo"}

    errors = run_threads([write, read, read, read])
    assert not errors, f"concurrent read/write raised: {errors[:3]}"


def test_reading_one_trace_while_writing_stays_consistent(store):
    """`get_trace` runs TWO statements and reads their results together, so an
    interleaving can pair one trace's header with another trace's steps — a
    wrong answer rather than a crash, which is the worse outcome."""
    for n in range(6):
        store.import_trace(doc(n))
    counter = itertools.count(100)

    def write():
        store.import_trace(doc(next(counter)))

    def read():
        got = store.get_trace("trace-3")
        if got is None:
            raise AssertionError("a trace that exists came back as missing")
        # Every step of trace-3 is named `s-3-*`. A step from another trace
        # here is the silent version of this bug.
        for step in got["steps"]:
            assert step["id"].startswith("s-3-"), f"foreign step {step['id']}"

    errors = run_threads([write, read, read])
    assert not errors, f"concurrent get_trace raised or mixed rows: {errors[:3]}"


def test_deleting_while_listing_does_not_raise(store):
    for n in range(20):
        store.import_trace(doc(n))
    ids = itertools.cycle(f"trace-{n}" for n in range(20))

    def delete():
        store.delete(next(ids))

    def read():
        store.list_traces()

    errors = run_threads([delete, read, read], seconds=1.5)
    assert not errors, f"delete raced with list: {errors[:3]}"


def test_every_method_that_touches_the_connection_serialises_it():
    """Against the source, because the defect was an ABSENCE — the two readers
    simply did not do what the three writers did, and nothing in the class
    said the rule existed. A new method added tomorrow is the same bug."""
    import inspect
    import re

    src = inspect.getsource(TraceStore)
    # Split into methods, skipping __init__: nothing else can hold a reference
    # to the store while it is still being constructed.
    parts = re.split(r"\n    def ", src)
    offenders = []
    for part in parts[1:]:
        name = part.split("(")[0]
        if name == "__init__":
            continue
        if re.search(r"self\._db\.(execute|executescript|commit)", part):
            if "with self._lock" not in part:
                offenders.append(name)
    assert not offenders, (
        "these touch the shared sqlite3 connection without holding the lock, "
        "which is what produced short rows under concurrency: " + ", ".join(offenders)
    )


def test_the_store_opens_on_a_machine_that_has_never_run_this_before(
    tmp_path, monkeypatch
):
    """`modelmri serve` died on first run for every genuinely new user.

    `paths.data_dir()` answers where the database belongs and creates nothing
    -- `paths.ensure` is the creator, by design. None of the three TraceStore
    call sites called it, so opening the database on a machine with no
    `~/.modelmri` raised `unable to open database file` inside `create_app`,
    before the server printed its URL. It went unnoticed because every machine
    it was tried on had already been an older version's machine and had the
    legacy directory sitting there.

    `tmp_path` here is a directory that exists but is EMPTY, which is exactly
    the state of a new install.
    """
    from modelmri import paths
    from modelmri.traces import TraceStore

    monkeypatch.setenv("MODELMRI_HOME", str(tmp_path))
    db = paths.trace_db_path()
    assert not db.parent.exists(), "this test is only meaningful on a fresh home"

    store = TraceStore(db)
    assert db.exists()
    # And it is a working database, not merely a file that opened.
    assert store.list_traces() == []
