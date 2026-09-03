# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""A measurement must not read somebody else's forward pass.

Two more sites of the defect `tests/test_patch_concurrency.py` records. A
forward hook belongs to the MODULE it is installed on, never to the call that
installed it, so any pass through that module while the hook is live fills the
collector too.

`ModelRuntime._compute_features` is the worse of the two, because its answer is
CACHED. One overlap with a streaming generation puts a one-token decode step in
`self._feats`, and every later `/api/features/summary` serves that same
poisoned matrix -- a `tokens` list of length S beside a `top` list of length 1,
pairing every token in the panel with somebody else's row -- until the next
generation replaces it.

`CustomHandle.run` is the quieter one. It releases `_lock` before the forward
pass deliberately, and says why; `inspect` then hooks EVERY LEAF MODULE, so two
overlapping runs publish each other's tensor statistics under their own input
shapes. Nothing raises there either.
"""

from __future__ import annotations

import threading

import pytest

torch = pytest.importorskip("torch")

from modelmri.custom import AdapterError, CustomHandle  # noqa: E402
from modelmri.errors import Refusal  # noqa: E402
from modelmri.runtime import ModelRuntime  # noqa: E402

HIDDEN = 8
N_TOKENS = 5


# ------------------------------------------------------- the features cache


class Block(torch.nn.Module):
    def forward(self, x, *a, **k):
        return (x,)


class Tiny(torch.nn.Module):
    """A model whose forward runs the block once -- or twice, on demand.

    The second call is what a decode step looks like from the hook's point of
    view: the same module, one token wide, from a thread this measurement
    knows nothing about.
    """

    def __init__(self, foreign: bool = False) -> None:
        super().__init__()
        self.block = Block()
        self.foreign = foreign

    def forward(self, ids, **kwargs):
        if self.foreign:
            self.block(torch.zeros(1, 1, HIDDEN))
        return self.block(torch.zeros(1, ids.shape[-1], HIDDEN))


class StubSAE:
    layer = 0
    point = "resid_post"
    d_sae = 4

    def encode(self, resid):
        return torch.zeros(resid.shape[0], self.d_sae)


def _runtime(*, foreign: bool = False, decoding: bool = False) -> ModelRuntime:
    rt = ModelRuntime.__new__(ModelRuntime)
    rt._decoding = threading.Event()
    if decoding:
        rt._decoding.set()
    rt.epoch = 1
    rt.last_ids_epoch = 1
    rt.last_ids = torch.arange(N_TOKENS)
    rt._feats = None
    rt.sae = StubSAE()
    rt.model = Tiny(foreign=foreign)
    rt.device = "cpu"
    rt._block = lambda layer: rt.model.block
    return rt


def test_features_refuse_while_a_generation_decodes():
    """PREVENT. `generate_stream` runs `model.generate` on a daemon thread
    holding no lock, so the hook installed here catches its decode steps."""
    rt = _runtime(decoding=True)
    with pytest.raises(Refusal) as caught:
        rt._compute_features()
    said = caught.value.sentence
    assert "generation is still running" in said
    assert "Wait for the run to finish" in said


def test_a_foreign_pass_in_the_collector_is_refused():
    """AND DETECT, because refusing is a race and this is not.

    A COUNT, not a shape: this method's own pass calls the hook exactly once,
    so a second entry means the tensor at index 0 may not be this
    measurement's at all.
    """
    rt = _runtime(foreign=True)
    with pytest.raises(Refusal) as caught:
        rt._compute_features()
    said = caught.value.sentence
    assert "Another forward pass ran through this model" in said
    assert "2 pass(es)" in said
    assert str(N_TOKENS) in said


def test_a_refused_reading_is_not_cached():
    """The whole reason this site is worse than its siblings. A poisoned
    matrix left in `_feats` would be served to every later request until the
    next generation."""
    rt = _runtime(foreign=True)
    with pytest.raises(Refusal):
        rt._compute_features()
    assert rt._feats is None


def test_a_clean_reading_still_works_and_caches():
    """The guard must not cost the ordinary path."""
    rt = _runtime()
    feats = rt._compute_features()
    assert feats.shape == (N_TOKENS, StubSAE.d_sae)
    assert rt._feats is not None
    assert rt._compute_features() is feats  # served from the cache


# ---------------------------------------------------------- the custom model


def _handle() -> CustomHandle:
    """Only the two locks. `__init__` builds status and adapter state, none of
    which is what is under test."""
    h = CustomHandle.__new__(CustomHandle)
    h._lock = threading.Lock()
    h._measure_lock = threading.Lock()
    return h


def test_one_measurement_at_a_time_on_a_custom_model():
    h = _handle()
    with h.measuring("map this model's layers"):
        with pytest.raises(AdapterError) as caught:
            with h.measuring("sweep this model causally"):
                pass
    said = str(caught.value)
    assert "another measurement is already running" in said
    assert "Wait for the one in flight" in said


def test_the_custom_slot_is_released_when_the_measurement_raises():
    """A sweep that refuses -- an adapter with no samples, a task it cannot
    read -- must not leave the model unmeasurable until restart."""
    h = _handle()

    def fails():
        raise ValueError("the sweep itself failed")

    with pytest.raises(ValueError):
        with h.measuring("sweep this model causally"):
            fails()
    with h.measuring("map this model's layers"):
        pass


def test_the_custom_slot_is_not_the_field_lock():
    """Separate, deliberately: `run` drops `_lock` for the forward pass so the
    status route stays answerable, and that release is what left the hooks
    uncovered. Sharing one lock would undo the reason for the release."""
    h = _handle()
    with h.measuring("map this model's layers"):
        assert h._lock.acquire(timeout=0) is True
        h._lock.release()


def test_a_second_custom_measurement_refuses_rather_than_queueing():
    """`timeout=0`. A causal sweep runs a forward pass per module and can take
    minutes; a queued caller holds a default-executor thread for all of it."""
    import time

    h = _handle()
    started = threading.Event()
    done = threading.Event()

    def hold():
        with h.measuring("sweep this model causally"):
            started.set()
            done.wait(timeout=5)

    worker = threading.Thread(target=hold, daemon=True)
    worker.start()
    assert started.wait(timeout=5), "the first measurement never started"
    try:
        t0 = time.perf_counter()
        with pytest.raises(AdapterError):
            with h.measuring("map this model's layers"):
                pass
        assert time.perf_counter() - t0 < 0.5
    finally:
        done.set()
        worker.join(timeout=5)
