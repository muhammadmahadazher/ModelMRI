# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""Two captures on one pipeline would mix one run's steps into the other's.

`image_steps.watch` installs a forward hook on the DENOISER MODULE and fills a
per-call collector from it. The hook belongs to the module, not to the call, so
a second capture running at the same time drives its own denoising steps
through the same hook and both collectors end up holding a mixture.

NEITHER RAISES. That is what makes this worse than the text-side version of the
same defect, where `patch.trace` caught a one-token decode step and died with an
IndexError: here the shapes agree, so two filmstrips come back looking exactly
like measurements of their own prompts.

`ImageHandle._lock` did not cover this. It is the LOAD slot -- held across a
multi-gigabyte download -- and the capture routes never took it or anything
else. `measuring` is a second, separate slot for the same reason the load slot
has a timeout: a capture waiting behind a download would be waiting for
something that has nothing to do with it.
"""

from __future__ import annotations

import threading

import pytest

from modelmri.errors import Refusal
from modelmri.image_runtime import ImageHandle


def _handle() -> ImageHandle:
    """A handle with only the two locks built.

    `__init__` brings up tracking, status and a device probe; none of that is
    what is under test, and a real one would need a checkpoint.
    """
    h = ImageHandle.__new__(ImageHandle)
    h._lock = threading.Lock()
    h._measure_lock = threading.Lock()
    return h


def test_one_capture_at_a_time():
    """THE DEFECT. The second caller is refused rather than allowed to run its
    denoiser through the first one's hook."""
    h = _handle()
    with h.measuring("film this run"):
        with pytest.raises(Refusal) as caught:
            with h.measuring("capture cross-attention"):
                pass
    said = caught.value.sentence
    assert "another measurement is already running" in said
    # A refusal names the cause AND what to do about it.
    assert "Wait for the one in flight" in said
    assert "mix one run's steps into the other's" in said


def _a_capture_that_fails() -> None:
    """What a refusing capture looks like from `measuring`'s point of view.

    A CALL, not a bare `raise`. The real failures come from inside
    `image_steps.filmstrip` -- a prompt the pipeline will not take, a model
    with no cross-attention to read -- so raising through a call is the
    faithful shape. It also keeps the statement after the `with` reachable in
    a way an analyser can see: a `with` body whose last statement is `raise`
    reads as "this block always raises", and `pytest.raises` swallowing it
    again is not something reachability analysis models.
    """
    raise ValueError("the capture itself failed")


def test_the_slot_is_released_when_the_capture_finishes():
    """A slot that leaked would turn one bad run into a dead pipeline."""
    h = _handle()
    with h.measuring("film this run"):
        pass
    with h.measuring("capture cross-attention"):
        pass  # no refusal: the first one gave it back


def test_the_slot_is_released_when_the_capture_raises():
    """The failure path matters more than the happy one here: a capture that
    refuses -- a prompt too long, a pipeline with no cross-attention -- must
    not leave the pipeline unmeasurable until restart."""
    h = _handle()
    with pytest.raises(ValueError):
        with h.measuring("film this run"):
            _a_capture_that_fails()
    with h.measuring("capture cross-attention"):
        pass


def test_it_is_not_the_load_slot():
    """Separate locks, deliberately. Sharing one would make every capture wait
    behind a download it has nothing to do with -- and `_load_slot` waits
    `LOAD_QUEUE_WAIT_S` for it, which is not a wait a measurement should
    inherit."""
    h = _handle()
    with h.measuring("film this run"):
        # The LOAD slot is still free while a capture holds the measure slot.
        assert h._lock.acquire(timeout=0) is True
        h._lock.release()


def test_a_second_capture_refuses_immediately_rather_than_queueing():
    """`timeout=0`. A capture can take minutes, and a queued caller holds a
    thread from the default executor the whole time -- the starvation
    `_load_slot`'s own docstring records, where twenty-eight blocked callers
    took the process's request path down."""
    h = _handle()
    started = threading.Event()
    done = threading.Event()

    def hold():
        with h.measuring("film this run"):
            started.set()
            done.wait(timeout=5)

    worker = threading.Thread(target=hold, daemon=True)
    worker.start()
    assert started.wait(timeout=5), "the first capture never started"
    try:
        import time

        t0 = time.perf_counter()
        with pytest.raises(Refusal):
            with h.measuring("read this model out"):
                pass
        # Refused, not waited on. Anything above a small fraction of a second
        # here means it queued.
        assert time.perf_counter() - t0 < 0.5
    finally:
        done.set()
        worker.join(timeout=5)
