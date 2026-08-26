"""A cached attention map has to be keyed on everything it depends on.

`_capture` caches each variant's maps under the variant string. That is sound
for `ablate:L.H`, which carries its own parameters in the name -- a different
ablation is a different key by construction. It was not sound for "steered",
whose parameters live in `self._steer`: `set_steering` writes them and clears
nothing, and every place that clears `_attn_variants` is about the MODEL
changing (load, unload, generate, adopt_step, GGUF).

So a map measured under one feature and scale went on being served, labelled
"steered", after the feature had been changed -- and `/api/attention/diff`
reported a movement for an intervention that was never run.

The second half is the refusal. It used to sit in the cache-MISS path, which
made it unreachable in the one case it exists for: once a steered map had been
cached, switching steering off could no longer reach it, and the route kept
answering with a steered map for a model that was no longer steered.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from modelmri.errors import Refusal  # noqa: E402
from modelmri.runtime import ModelRuntime  # noqa: E402

N_TOKENS = 4
N_HEADS = 2


class Handle:
    def remove(self) -> None:
        pass


class Out:
    def __init__(self, fill: float) -> None:
        self.attentions = [torch.full((1, N_HEADS, N_TOKENS, N_TOKENS), fill)]


class StubSAE:
    d_sae = 16


def _runtime() -> ModelRuntime:
    """A runtime whose forward pass reports WHICH steering was live for it.

    The map's fill value is the feature id, so a stale map is identifiable
    rather than merely equal-shaped -- which is the whole failure here: the
    shapes always agreed.
    """
    rt = ModelRuntime.__new__(ModelRuntime)
    rt._attn_variants = {}
    rt._steer = None
    rt.sae = StubSAE()
    rt.last_ids = torch.arange(N_TOKENS)
    rt.device = "cpu"
    rt._attn_tokens = ["a", "b", "c", "d"]
    rt._steer_handle = lambda: Handle()
    rt.model = lambda ids, output_attentions=False: Out(
        float(rt._steer[0]) if rt._steer else -1.0
    )
    return rt


def _fill(maps) -> float:
    return float(maps[0].flatten()[0])


def test_changing_the_feature_gives_a_map_measured_under_it():
    """THE DEFECT. The second read used to return the first feature's map."""
    rt = _runtime()
    rt._steer = (100, 5.0)
    assert _fill(rt._capture("steered")) == 100.0
    rt._steer = (200, -3.0)
    assert _fill(rt._capture("steered")) == 200.0


def test_changing_only_the_scale_also_gives_a_new_measurement():
    """Same feature at a different strength is a different intervention, and
    the scale is half of what `set_steering` takes."""
    rt = _runtime()
    rt._steer = (100, 5.0)
    rt._capture("steered")
    rt._steer = (100, -3.0)
    rt._capture("steered")
    assert "steered:100:-3.0" in rt._attn_variants
    assert "steered:100:5.0" in rt._attn_variants


def test_turning_steering_off_refuses_even_after_a_map_was_cached():
    """The refusal moved above the cache lookup. Below it, this call returned
    the stale map instead -- a steered reading for a model with steering
    switched off."""
    rt = _runtime()
    rt._steer = (100, 5.0)
    rt._capture("steered")
    rt._steer = None
    with pytest.raises(Refusal) as caught:
        rt._capture("steered")
    assert "Nothing is being steered" in caught.value.sentence


def test_the_same_steering_is_still_served_from_the_cache():
    """The key exists to avoid a second forward pass, not to defeat it."""
    rt = _runtime()
    rt._steer = (100, 5.0)
    first = rt._capture("steered")
    assert rt._capture("steered") is first


def test_live_is_untouched():
    """`live` and `ablate:L.H` already carried everything they depend on."""
    rt = _runtime()
    first = rt._capture("live")
    assert rt._capture("live") is first
    assert "live" in rt._attn_variants
