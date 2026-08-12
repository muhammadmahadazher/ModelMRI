"""The fit calculator has to be checkable by hand, or it is just a verdict.

Every number it prints is arithmetic over values it read, so these tests do
the arithmetic independently and compare. The ones that matter most are the
refusals: an architecture whose KV cache does not match the formula must be
named and declined, because the failure mode of approximating is a confident
"it fits" for a model that then dies on the first forward pass.
"""

from __future__ import annotations

import json
import struct

import pytest

from modelmri import fit
from modelmri.errors import BadRequest, Refusal


def write_safetensors(path, tensors: dict) -> None:
    """A real safetensors file: 8-byte LE length, JSON header, then the data.

    `tensors` maps name -> (dtype, shape). Bytes are zeros; nothing here reads
    the payload, only the table that describes it.
    """
    header, offset = {}, 0
    for name, (dtype, shape) in tensors.items():
        count = 1
        for dim in shape:
            count *= dim
        span = count * fit.DTYPE_BYTES[dtype]
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [offset, offset + span]}
        offset += span
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * offset)


CONFIG = {
    "model_type": "llama",
    "num_hidden_layers": 4,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "hidden_size": 512,
}


@pytest.fixture
def model_dir(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
    write_safetensors(
        tmp_path / "model.safetensors",
        {
            "model.layers.0.self_attn.k_proj.weight": ("BF16", [256, 512]),
            "model.embed_tokens.weight": ("BF16", [1000, 512]),
        },
    )
    return tmp_path


# --------------------------------------------------------- reading the header


def test_weights_are_summed_from_offsets_not_file_size(model_dir):
    w = fit.weights_bytes(model_dir)
    expected = (256 * 512 + 1000 * 512) * 2
    assert w.disk_bytes == expected
    assert w.card_bytes == expected  # no dtype given: loaded as stored
    assert w.by_dtype == {"BF16": expected}
    assert "1 safetensors shard" in w.source
    # The file is strictly larger than the payload — header plus prefix.
    assert (model_dir / "model.safetensors").stat().st_size > w.disk_bytes


def test_shards_are_summed(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
    write_safetensors(tmp_path / "model-00001-of-00002.safetensors", {"a": ("F32", [10])})
    write_safetensors(tmp_path / "model-00002-of-00002.safetensors", {"b": ("F32", [20])})
    w = fit.weights_bytes(tmp_path)
    assert w.disk_bytes == 30 * 4
    assert w.elements == 30
    assert "2 safetensors shards" in w.source


# ------------------------------------------- the dtype the load actually uses


def test_a_float32_checkpoint_loaded_bf16_halves_on_the_card(tmp_path):
    """Measured on gpt2: 548.1 MB on disk, 255.3 MB allocated. 2x, silently."""
    (tmp_path / "config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
    write_safetensors(tmp_path / "model.safetensors", {"w": ("F32", [1000, 512])})
    w = fit.weights_bytes(tmp_path, dtype_bytes=2)
    assert w.disk_bytes == 1000 * 512 * 4
    assert w.card_bytes == 1000 * 512 * 2
    assert w.converted is True


def test_integer_tensors_are_not_repriced_by_a_float_dtype(tmp_path):
    """A quantised checkpoint's int weights keep their width on the card."""
    (tmp_path / "config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
    write_safetensors(
        tmp_path / "model.safetensors",
        {"q": ("I8", [1000, 512]), "scale": ("F32", [1000])},
    )
    w = fit.weights_bytes(tmp_path, dtype_bytes=2)
    assert w.card_bytes == 1000 * 512 * 1 + 1000 * 2


def test_plan_prices_weights_at_the_load_dtype_and_says_so(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
    write_safetensors(tmp_path / "model.safetensors", {"w": ("F32", [1000, 512])})
    f = fit.plan(tmp_path, seq_len=64, dtype_bytes=2)
    assert f.weights_bytes == 1000 * 512 * 2
    assert any("on disk" in n for n in f.notes)
    assert any("re-priced" in t.formula for t in f.terms if t.name == "weights")


def test_an_unknown_dtype_is_refused_not_guessed(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
    blob = json.dumps(
        {"w": {"dtype": "F4_SECRET", "shape": [4], "data_offsets": [0, 4]}}
    ).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(blob)) + blob)
    with pytest.raises(Refusal, match="F4_SECRET"):
        fit.weights_bytes(tmp_path)


def test_a_truncated_file_is_refused(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"\x01\x02")
    with pytest.raises(BadRequest, match="too short"):
        fit.read_header(tmp_path / "model.safetensors")


def test_an_absurd_header_length_is_refused(tmp_path):
    p = tmp_path / "model.safetensors"
    p.write_bytes(struct.pack("<Q", 10**12) + b"{}")
    with pytest.raises(BadRequest, match="header"):
        fit.read_header(p)


def test_no_safetensors_refuses_rather_than_estimating(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
    (tmp_path / "pytorch_model.bin").write_bytes(b"x" * 999)
    with pytest.raises(Refusal, match="does not estimate from file sizes"):
        fit.weights_bytes(tmp_path)


# -------------------------------------------------------------- KV geometry


def test_grouped_query_attention_is_not_widened_to_mha(model_dir):
    geo = fit.kv_geometry(CONFIG, None)
    assert geo.n_kv_heads == 2  # not 8
    assert geo.n_layers == 4


def test_a_missing_layer_count_is_named_not_defaulted():
    with pytest.raises(BadRequest, match="num_hidden_layers"):
        fit.kv_geometry({"model_type": "llama", "num_attention_heads": 8}, None)


def test_head_dim_prefers_what_the_config_states():
    cfg = dict(CONFIG, head_dim=128)
    geo = fit.kv_geometry(cfg, None)
    assert geo.head_dim == 128  # not 512 // 8 == 64
    assert "config.json" in geo.head_dim_source


def test_head_dim_is_read_off_k_proj_when_the_config_omits_it():
    """The same width ablate.head_geometry reads off a loaded model."""
    header = {
        "model.layers.0.self_attn.k_proj.weight": {"shape": [256, 512]},
    }
    geo = fit.kv_geometry(CONFIG, header)
    assert geo.head_dim == 128  # 256 rows / 2 kv heads, NOT 512/8 = 64
    assert "k_proj" in geo.head_dim_source


def test_the_quotient_fallback_says_it_is_a_fallback():
    geo = fit.kv_geometry(CONFIG, None)
    assert geo.head_dim == 64
    assert "fallback" in geo.head_dim_source


# ------------------------------------------------- architectures we refuse


def test_multi_head_latent_attention_is_refused_by_name():
    cfg = dict(CONFIG, model_type="deepseek_v3", kv_lora_rank=512)
    with pytest.raises(fit.UnsupportedArchitecture, match="latent attention"):
        fit.kv_geometry(cfg, None)


def test_sliding_window_is_refused_because_the_cache_stops_growing():
    cfg = dict(CONFIG, model_type="mistral", sliding_window=4096)
    with pytest.raises(fit.UnsupportedArchitecture, match="sliding-window"):
        fit.kv_geometry(cfg, None)


def test_sliding_window_explicitly_disabled_is_fine():
    cfg = dict(CONFIG, sliding_window=4096, use_sliding_window=False)
    assert fit.kv_geometry(cfg, None).n_layers == 4


def test_hybrid_ssm_is_refused_because_some_layers_have_no_kv():
    cfg = dict(CONFIG, model_type="jamba")
    with pytest.raises(fit.UnsupportedArchitecture, match="state-space"):
        fit.kv_geometry(cfg, None)


def test_unsupported_architecture_is_a_refusal_so_the_server_answers_409():
    assert issubclass(fit.UnsupportedArchitecture, Refusal)


# --------------------------------------------------------------- arithmetic


def test_kv_cache_matches_the_formula_on_the_page():
    geo = fit.KVGeometry(n_layers=4, n_heads=8, n_kv_heads=2, head_dim=64,
                         head_dim_source="test")
    assert fit.kv_cache_bytes(geo, 1024, 2) == 2 * 4 * 2 * 64 * 1024 * 2


def test_attention_buffer_is_quadratic_in_sequence_length():
    geo = fit.KVGeometry(4, 8, 2, 64, "test")
    at1k = fit.attention_bytes(geo, 1024, 2)
    at2k = fit.attention_bytes(geo, 2048, 2)
    assert at2k == at1k * 4


def test_the_readme_figure_reproduces():
    """12 layers x 12 heads x 4096^2 x 2 bytes = 4.6 GB, as the roadmap claims."""
    geo = fit.KVGeometry(12, 12, 12, 64, "test")
    assert fit.attention_bytes(geo, 4096, 2) / 1e9 == pytest.approx(4.83, abs=0.05)


def test_plan_total_is_the_sum_of_its_own_terms(model_dir):
    f = fit.plan(model_dir, seq_len=512, dtype_bytes=2)
    assert f.total_bytes == f.weights_bytes + f.kv_bytes + f.attention_bytes
    assert sum(t.bytes for t in f.terms) == f.total_bytes


def test_every_term_carries_a_formula_you_can_check(model_dir):
    f = fit.plan(model_dir, seq_len=512)
    assert {t.name for t in f.terms} == {"weights", "kv_cache", "eager_attention"}
    for term in f.terms:
        assert term.formula
    assert "activations and workspace" in f.excluded


def test_fits_is_none_when_no_budget_was_given(model_dir):
    assert fit.plan(model_dir, seq_len=128).fits is None


def test_fits_tracks_the_budget(model_dir):
    f = fit.plan(model_dir, seq_len=128, budget_bytes=10**12)
    assert f.fits is True
    assert fit.plan(model_dir, seq_len=128, budget_bytes=1).fits is False


def test_seq_len_must_be_positive(model_dir):
    with pytest.raises(BadRequest):
        fit.plan(model_dir, seq_len=0)


def test_missing_config_is_refused(tmp_path):
    write_safetensors(tmp_path / "model.safetensors", {"a": ("F32", [4])})
    with pytest.raises(Refusal, match="config.json"):
        fit.plan(tmp_path, seq_len=8)


# ---------------------------------------------------------- longest context


def test_longest_context_is_the_boundary(model_dir):
    budget = fit.plan(model_dir, seq_len=256).total_bytes
    best = fit.longest_context(model_dir, budget_bytes=budget)
    assert best >= 256
    assert fit.plan(model_dir, seq_len=best).total_bytes <= budget
    assert fit.plan(model_dir, seq_len=best + 1).total_bytes > budget


def test_longest_context_is_zero_when_even_one_token_does_not_fit(model_dir):
    assert fit.longest_context(model_dir, budget_bytes=1) == 0


# ----------------------------------------------------------------- grading


def test_grade_names_the_gap_rather_than_tuning_it_away(model_dir):
    f = fit.plan(model_dir, seq_len=128)
    graded = fit.grade(f, f.total_bytes + 500_000)
    assert graded["gap_bytes"] == 500_000
    assert "unpredicted runtime overhead" in graded["gap_is"]
    assert graded["predicted_bytes"] == f.total_bytes


def test_grade_handles_measuring_less_than_predicted(model_dir):
    f = fit.plan(model_dir, seq_len=128)
    graded = fit.grade(f, f.total_bytes - 1000)
    assert graded["gap_bytes"] == -1000
    assert "less than predicted" in graded["gap_is"]


def test_plan_is_json_safe(model_dir):
    json.dumps(fit.plan(model_dir, seq_len=64, budget_bytes=10**10).to_dict())
