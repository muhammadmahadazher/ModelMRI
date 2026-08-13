"""A damage report that drops what it cannot measure is a flattering one.

The failure mode here is quiet and one-directional: every tensor this cannot
pair up, dequantise or line up by shape is a tensor missing from the
denominator, and a worse quantisation looks better for it. So most of this file
is about tensors ending up in `not_compared` with a reason rather than
disappearing.

The other half is the four statistics. RMS alone is misleading in both
directions — a tensor of large weights absorbs a large RMS without moving, a
tensor of tiny weights can be mostly noise at a small one — so cosine and sign
flips are checked against cases constructed to have known answers.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from modelmri import quantdiff  # noqa: E402
from modelmri.errors import BadRequest  # noqa: E402


# ------------------------------------------------------------ name mapping


@pytest.mark.parametrize(
    "gguf,hf",
    [
        ("blk.0.attn_q.weight", "model.layers.0.self_attn.q_proj.weight"),
        ("blk.31.ffn_down.weight", "model.layers.31.mlp.down_proj.weight"),
        ("blk.7.attn_norm.weight", "model.layers.7.input_layernorm.weight"),
        ("token_embd.weight", "model.embed_tokens.weight"),
        ("output.weight", "lm_head.weight"),
        ("output_norm.weight", "model.norm.weight"),
    ],
)
def test_known_names_map(gguf, hf):
    assert quantdiff.hf_name(gguf) == hf


@pytest.mark.parametrize(
    "unknown",
    ["blk.0.some_future_thing.weight", "rope_freqs.weight", "", "blk.x.attn_q.weight"],
)
def test_an_unknown_name_returns_none_rather_than_a_guess(unknown):
    """Pairing two tensors that are not the same tensor produces a damage
    number about nothing, and it looks exactly like a real one."""
    assert quantdiff.hf_name(unknown) is None


# --------------------------------------------------------- the statistics


def test_an_identical_tensor_has_no_damage():
    a = np.array([[1.0, -2.0], [3.0, 0.5]], dtype=np.float32)
    s = quantdiff.compare_tensor(a, a)
    assert s["rms"] == 0.0
    assert s["max_abs"] == 0.0
    assert s["cosine"] == pytest.approx(1.0)
    assert s["sign_flips"] == 0.0
    assert s["relative_rms"] == 0.0


def test_a_negated_tensor_is_maximally_wrong():
    a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    s = quantdiff.compare_tensor(-a, a)
    assert s["cosine"] == pytest.approx(-1.0)
    assert s["sign_flips"] == pytest.approx(1.0)


def test_cosine_ignores_scale_but_rms_does_not():
    """Two tensors can differ a lot in magnitude and still point the same way,
    which is what a good quantiser preserves."""
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    s = quantdiff.compare_tensor(a * 2, a)
    assert s["cosine"] == pytest.approx(1.0)
    assert s["rms"] > 0


def test_sign_flips_counts_only_the_weights_that_turned_round():
    a = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    b = np.array([1.0, -1.0, -1.0, 1.0], dtype=np.float32)
    assert quantdiff.compare_tensor(b, a)["sign_flips"] == pytest.approx(0.5)


def test_relative_rms_makes_tensors_of_different_scales_comparable():
    small = np.array([0.001, -0.002, 0.003], dtype=np.float32)
    large = small * 1000
    err_s = quantdiff.compare_tensor(small * 1.1, small)
    err_l = quantdiff.compare_tensor(large * 1.1, large)
    assert err_l["rms"] > err_s["rms"] * 100          # absolute error explodes
    assert err_l["relative_rms"] == pytest.approx(err_s["relative_rms"], rel=1e-5)


def test_an_all_zero_original_has_undefined_direction_not_zero():
    """0.0 would read as "completely orthogonal", a strong and false claim."""
    zeros = np.zeros(4, dtype=np.float32)
    s = quantdiff.compare_tensor(np.ones(4, dtype=np.float32), zeros)
    assert s["cosine"] is None
    assert s["relative_rms"] is None


def test_a_shape_mismatch_is_refused_rather_than_broadcast():
    with pytest.raises(BadRequest, match="not the same tensor"):
        quantdiff.compare_tensor(np.ones((2, 3)), np.ones((3, 2)))


def test_max_abs_finds_the_single_worst_weight():
    a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.01, 0.01, 5.0], dtype=np.float32)
    s = quantdiff.compare_tensor(b, a)
    assert s["max_abs"] == pytest.approx(5.0)
    assert s["rms"] < s["max_abs"]  # the mean hides it; max is why it is here


# ------------------------------------------------------------- the report


def _damage(name, elements, rel, flips, cos=0.99):
    return quantdiff.TensorDamage(
        name=name, hf_name="x", quant_type="Q4_K", elements=elements,
        rms=0.1, max_abs=0.5, cosine=cos, sign_flips=flips, relative_rms=rel,
    )


def test_the_summary_is_element_weighted_not_tensor_weighted():
    """One tiny norm tensor must not dominate an average over a model whose
    real mass is in a few enormous matrices."""
    r = quantdiff.Report(quantised="q", original="o")
    r.tensors = [_damage("big", 1_000_000, 0.01, 0.001), _damage("tiny", 10, 0.9, 0.9)]
    s = r.summary()
    assert s["mean_relative_rms"] < 0.02, s["mean_relative_rms"]


def test_the_summary_names_the_worst_of_each_kind():
    r = quantdiff.Report(quantised="q", original="o")
    r.tensors = [
        _damage("a", 100, 0.5, 0.01, cos=0.99),
        _damage("b", 100, 0.1, 0.30, cos=0.60),
    ]
    s = r.summary()
    assert s["worst_relative_rms"]["name"] == "a"
    assert s["worst_sign_flips"]["name"] == "b"
    assert s["worst_cosine"]["name"] == "b"


def test_uncompared_tensors_are_counted_and_excluded_not_dropped():
    """Silently omitting them shrinks the denominator of every aggregate and
    makes a worse quantisation look better."""
    r = quantdiff.Report(quantised="q", original="o")
    r.tensors = [_damage("a", 100, 0.1, 0.0)]
    r.not_compared = [{"name": "b", "why": "unknown type"}]
    s = r.summary()
    assert s["compared"] == 1
    assert s["not_compared"] == 1
    assert s["elements_compared"] == 100
    assert "excluded from every average" in s["means"]


def test_a_report_with_nothing_paired_says_so_rather_than_averaging_nothing():
    r = quantdiff.Report(quantised="q", original="o")
    r.not_compared = [{"name": "a", "why": "no mapping"}]
    s = r.summary()
    assert s["compared"] == 0
    assert "nothing to report" in s["means"]
    assert "mean_relative_rms" not in s


def test_the_summary_disclaims_what_it_does_not_measure():
    """Users will read this as "what llama.cpp does". It measures stored
    weights, not the kernels that later consume them."""
    r = quantdiff.Report(quantised="q", original="o")
    r.tensors = [_damage("a", 10, 0.1, 0.0)]
    assert "NOT what llama.cpp's kernels" in r.summary()["means"]


def test_the_report_is_json_safe():
    import json

    r = quantdiff.Report(quantised="q", original="o")
    r.tensors = [_damage("a", 10, 0.1, 0.0)]
    r.not_compared = [{"name": "b", "why": "x"}]
    json.dumps(r.to_dict())


def test_a_cap_is_recorded_rather_than_leaving_a_partial_report_looking_whole(tmp_path):
    r = quantdiff.Report(quantised="q", original="o")
    r.notes.append("compared the first 5 tensors only")
    assert any("first 5" in n for n in r.to_dict()["notes"])
