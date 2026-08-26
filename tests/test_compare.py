"""Comparing two runs is only honest when the tokens are the same.

A cell-by-cell difference of two attention matrices means something exactly
when index i is the same token on both sides. That is guaranteed here by
construction — both sides are forward passes over one `last_ids`, not two
generations — and these tests hold the construction in place, because the
version that looks equivalent and is not would ship a smooth, plausible,
entirely fictitious picture.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from modelmri import runtime as runtime_mod  # noqa: E402


class Recorder:
    """A runtime stub that records which variants were captured."""

    def __init__(self, layers=2, heads=2, size=3):
        self.calls: list[str] = []
        self._shape = (layers, heads, size)

    def capture(self, variant: str):
        self.calls.append(variant)
        layers, heads, size = self._shape
        # A different constant per variant, so a difference is predictable.
        fill = {"live": 0.5, "steered": 0.3}.get(variant, 0.1)
        return [torch.full((heads, size, size), fill) for _ in range(layers)]


def runtime_with(recorder: Recorder) -> runtime_mod.ModelRuntime:
    rt = runtime_mod.ModelRuntime()
    rt.last_ids = torch.tensor([1, 2, 3])
    rt.last_ids_epoch = rt.epoch
    rt._attn_tokens = ["a", "b", "c"]
    rt._capture = recorder.capture  # type: ignore[assignment]
    rt.model = object()  # `loaded` only checks for not-None
    return rt


def test_a_difference_is_two_passes_over_one_sequence():
    """Not two generations. With temperature above zero the sampled tokens
    diverge, and a different chat template shifts every index — subtracting
    misaligned sequences produces a picture of nothing."""
    rec = Recorder()
    rt = runtime_with(rec)
    out = rt.attention_diff(0, 0, "live", "steered")

    assert rec.calls == ["live", "steered"], rec.calls
    # One token list, because there is one token sequence.
    assert out["tokens"] == ["a", "b", "c"]
    assert out["a"] == "live" and out["b"] == "steered"


def test_the_difference_is_signed():
    """Half the answer is where attention moved AWAY. A magnitude-only diff
    would drop it silently."""
    rec = Recorder()
    rt = runtime_with(rec)
    out = rt.attention_diff(0, 0, "steered", "live")  # 0.3 - 0.5
    assert all(v < 0 for row in out["matrix"] for v in row)
    assert out["max_abs"] == pytest.approx(0.2, abs=1e-3)


def test_it_reports_how_much_actually_moved():
    rec = Recorder()
    rt = runtime_with(rec)
    out = rt.attention_diff(0, 0, "live", "steered")
    assert out["cells"] == 9
    assert out["moved"] == 9  # every cell differs by 0.2
    assert out["max_abs"] == pytest.approx(0.2, abs=1e-3)


def test_an_identical_pair_says_so_rather_than_drawing_nothing():
    rec = Recorder()
    rt = runtime_with(rec)
    out = rt.attention_diff(0, 0, "live", "live")
    assert out["max_abs"] == 0.0
    assert "identical" in out["note"]


def test_ablating_a_head_cannot_change_its_own_layer_and_the_note_says_why():
    """The most confusing possible zero. Ablation removes a head's OUTPUT,
    so the layer it lives in is computed from an unchanged input — its
    attention is bit-identical, every time, by construction. Without this
    note a user concludes the intervention did nothing."""

    class Same(Recorder):
        def capture(self, variant):
            self.calls.append(variant)
            return [torch.full((2, 3, 3), 0.5) for _ in range(4)]

    rt = runtime_with(Same())
    out = rt.attention_diff(0, 0, "live", "ablate:0.1")
    assert out["max_abs"] == 0.0
    assert "has to be" in out["note"]
    assert "downstream" in out["note"]

    # ...and at a layer that CAN differ, no such claim is made.
    quiet = rt.attention_diff(2, 0, "live", "ablate:0.1")
    assert "has to be" not in quiet["note"]


def test_comparing_while_viewing_a_recording_is_refused():
    """A `.mri` carries no model, so there is no second run to make."""
    from modelmri import session

    rt = runtime_with(Recorder())
    rt.replay = session.Session(tokens=["a"], n_layers=1, n_heads=1)
    with pytest.raises(RuntimeError, match="viewing a recording"):
        rt.attention_diff(0, 0)


def test_an_out_of_range_head_is_refused():
    rt = runtime_with(Recorder())
    with pytest.raises(ValueError, match="head in"):
        rt.attention_diff(0, 99, "live", "steered")


def test_steering_needs_something_to_steer():
    rt = runtime_mod.ModelRuntime()
    rt.last_ids = torch.tensor([1, 2])
    rt.last_ids_epoch = rt.epoch
    rt.model = object()
    rt._steer = None
    with pytest.raises(RuntimeError, match="Nothing is being steered"):
        rt._capture("steered")


def test_an_unknown_variant_is_refused_not_silently_treated_as_live():
    rt = runtime_mod.ModelRuntime()
    rt.last_ids = torch.tensor([1, 2])
    rt.last_ids_epoch = rt.epoch
    rt.model = object()
    with pytest.raises(ValueError, match="unknown variant"):
        rt._capture("something-else")


def test_a_malformed_ablation_spec_says_what_it_expected():
    rt = runtime_mod.ModelRuntime()
    rt.last_ids = torch.tensor([1, 2])
    rt.last_ids_epoch = rt.epoch
    rt.model = object()
    with pytest.raises(ValueError, match="ablate:LAYER.HEAD"):
        rt._capture("ablate:nonsense")


def test_variants_are_cached_separately_and_cleared_together():
    """A stale 'live' beside a fresh 'steered' would render a difference
    between two different generations.

    THE STEERED KEY CARRIES THE INTERVENTION. It used to be the bare string
    "steered", and that was wrong twice: `set_steering` clears nothing and
    every site that clears `_attn_variants` is about the MODEL changing, so
    one feature's map was served under another feature's name -- and the
    cache hit came BEFORE the "nothing is being steered" guard, which made
    that refusal unreachable once any steered map had been cached.

    This test used to seed `"steered"` directly, which is why it pinned the
    old key. `test_steered_attention_cache.py` drives the whole path; what is
    left here is the original property: two variants, two entries, cleared
    together.
    """
    rec = Recorder()
    rt = runtime_with(rec)
    rt._capture = runtime_mod.ModelRuntime._capture.__get__(rt)
    # An SAE and a live steering, because a steered capture now refuses
    # without them rather than answering from a cache.
    rt.sae = object()
    rt._steer = (100, 5.0)

    rt._attn_variants["live"] = ["sentinel-live"]
    rt._attn_variants["steered:100:5.0"] = ["sentinel-steered"]
    assert rt._capture("live") == ["sentinel-live"]
    assert rt._capture("steered") == ["sentinel-steered"]

    # Change the intervention and that entry is no longer what is asked for.
    rt._steer = (100, -3.0)
    assert "steered:100:-3.0" not in rt._attn_variants

    rt._attn_variants.clear()
    assert rt._attn_variants == {}


def test_the_steering_installer_is_shared_with_generation():
    """Two implementations of 'what steering does' would drift, and the
    comparison would then be between a real run and an approximation."""
    import inspect

    source = inspect.getsource(runtime_mod.ModelRuntime.generate_stream)
    assert "self._steer_handle()" in source
    # The old inline copy must not come back.
    assert "def _steer_post" not in source
    assert "def _steer_hook" not in source
