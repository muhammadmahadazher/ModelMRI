# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

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
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + span],
        }
        offset += span
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * offset)


def write_raw_header(path, header) -> None:
    """A safetensors file whose header is EXACTLY what it is handed.

    `write_safetensors` builds a well-formed table from dtype/shape pairs,
    which is the wrong tool for asking what happens to a damaged one. This
    writes the header verbatim — including a header that is not a table at
    all — so the reader is tested against bytes a corrupt download really
    produces rather than against a mock.
    """
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * 64)


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
    write_safetensors(
        tmp_path / "model-00001-of-00002.safetensors", {"a": ("F32", [10])}
    )
    write_safetensors(
        tmp_path / "model-00002-of-00002.safetensors", {"b": ("F32", [20])}
    )
    w = fit.weights_bytes(tmp_path)
    assert w.disk_bytes == 30 * 4
    assert w.elements == 30
    assert "2 safetensors shards" in w.source


# ------------------------------------------- the dtype the load actually uses


def test_a_float32_checkpoint_loaded_bf16_halves_on_the_card(tmp_path):
    """A float32 checkpoint loaded in bfloat16 takes half its on-disk bytes on
    the card. 2x, silently."""
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
    geo = fit.KVGeometry(
        n_layers=4, n_heads=8, n_kv_heads=2, head_dim=64, head_dim_source="test"
    )
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


# ------------------------------------- regression from the pre-push audit


def test_a_nested_text_config_is_refused_like_a_flat_one():
    """Multimodal configs nest the language model under `text_config`.
    `need()` fell through to it for layer/head counts, but the guards read the
    top level only — so a nested gemma-3 was NOT refused for its sliding
    window, and its KV came out 5.8x too big at 32k while the identical FLAT
    config was correctly refused."""
    nested = {
        "model_type": "gemma3",
        "text_config": {
            "model_type": "gemma3_text",
            "num_hidden_layers": 26,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "hidden_size": 1152,
            "head_dim": 256,
            "sliding_window": 512,
        },
    }
    with pytest.raises(fit.UnsupportedArchitecture, match="sliding-window"):
        fit.kv_geometry(nested, None)


def test_nested_geometry_is_read_from_the_text_config():
    nested = {
        "model_type": "qwen2_vl",
        "text_config": {
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "hidden_size": 1536,
            "head_dim": 128,
        },
    }
    geo = fit.kv_geometry(nested, None)
    assert geo.n_layers == 28 and geo.n_kv_heads == 2 and geo.head_dim == 128


def test_a_nested_mla_config_is_refused_too():
    nested = {"model_type": "wrapper", "text_config": dict(CONFIG, kv_lora_rank=512)}
    with pytest.raises(fit.UnsupportedArchitecture, match="latent attention"):
        fit.kv_geometry(nested, None)


# ------------------- regression: audit 1.7, malformed checkpoints (b, c, d)


@pytest.mark.parametrize("body", [[], 5, "abc", True, None])
def test_a_header_that_is_valid_json_but_not_a_table_is_refused(tmp_path, body):
    """`json.loads` succeeds on `[]`, `5`, `"abc"`, `true` and `null`, and
    `read_header` then called `.pop()` on whatever came back. Measured on a
    two-byte header holding `[]`: `TypeError: pop expected at most 1 argument,
    got 2` — a raw Python exception escaping `weights_bytes`, `plan` and
    `weights_table.table_from_safetensors` alike, past the `except BadRequest`
    that exists to turn exactly this into a sentence."""
    p = tmp_path / "model.safetensors"
    write_raw_header(p, body)
    with pytest.raises(BadRequest, match="not a tensor table"):
        fit.read_header(p)


def test_the_header_refusal_says_what_the_file_held_in_json_words(tmp_path):
    """ "a int" and "a NoneType" name a Python type at somebody looking at their
    own JSON. The sentence has to match the file they can open."""
    p = tmp_path / "model.safetensors"
    write_raw_header(p, 5)
    with pytest.raises(BadRequest, match="a number"):
        fit.read_header(p)


@pytest.mark.parametrize("shape", ["abc", [None, 4], 5, {"rows": 4}])
def test_a_shape_that_is_not_numbers_is_refused_by_the_arm_written_for_it(
    tmp_path, shape
):
    """`weights_bytes` caught KeyError/TypeError/ValueError around the three
    `spec[...]` lookups and then ran `int(start)` and `int(dim)` three lines
    BELOW that arm's reach. A header that names its fields and fills them with
    rubbish therefore crashed the reader instead of being refused by it — the
    same defect `image_fit._price` carried, where it was reachable as a 500."""
    (tmp_path / "config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
    write_raw_header(
        tmp_path / "model.safetensors",
        {"w": {"dtype": "F32", "shape": shape, "data_offsets": [0, 16]}},
    )
    with pytest.raises(BadRequest, match="does not recognise"):
        fit.weights_bytes(tmp_path)


def test_offsets_that_are_not_numbers_are_refused_too(tmp_path):
    """`data_offsets: ["a", 4]` unpacks into two names perfectly happily and
    dies on the `int()` below. The pair being present is not the same question
    as the pair being readable."""
    write_raw_header(
        tmp_path / "model.safetensors",
        {"w": {"dtype": "F32", "shape": [4], "data_offsets": ["a", 4]}},
    )
    with pytest.raises(BadRequest, match="does not recognise"):
        fit.weights_bytes(tmp_path)


@pytest.mark.parametrize(
    "spec",
    [
        {"dtype": "F32", "shape": [-4, 4], "data_offsets": [0, 16]},
        {"dtype": "F32", "shape": [4], "data_offsets": [16, 0]},
    ],
)
def test_a_negative_extent_never_subtracts_from_the_checkpoint_total(tmp_path, spec):
    """A negative dimension or a reversed offset pair makes a tensor weigh less
    than nothing, so a damaged shard would make the whole checkpoint look
    smaller than it is. This module's docstring names that as the dangerous
    direction — wrong towards "it fits" — so it is refused rather than summed."""
    write_raw_header(tmp_path / "model.safetensors", {"w": spec})
    with pytest.raises(BadRequest, match="negative size"):
        fit.weights_bytes(tmp_path)


def test_a_half_written_config_is_refused_rather_than_raising_a_json_error(tmp_path):
    """This project's own Drive-backed cache publishes `config.json` mid-write,
    so this is not hypothetical. Measured on one caught that way:
    `JSONDecodeError: Unterminated string starting at: line 1 column 26 (char
    25)`, straight out of `plan()`, with nothing in it naming the file."""
    write_safetensors(tmp_path / "model.safetensors", {"a": ("F32", [4])})
    (tmp_path / "config.json").write_text(
        '{"num_hidden_layers": 4, "num_attention', encoding="utf-8"
    )
    with pytest.raises(BadRequest, match="not valid JSON") as caught:
        fit.plan(tmp_path, seq_len=8)
    # The next step as well as the cause: a truncated config is re-fetched,
    # not repaired by hand.
    assert "again" in caught.value.sentence


@pytest.mark.parametrize("body", ["[1, 2, 3]", "5", '"abc"', "null"])
def test_a_config_that_is_not_a_set_of_fields_is_refused(tmp_path, body):
    """A `config.json` holding `[1, 2, 3]` parses fine and dies one frame
    later: measured `AttributeError: 'list' object has no attribute 'get'`,
    out of `_merged`. Valid JSON and a config are two different claims."""
    write_safetensors(tmp_path / "model.safetensors", {"a": ("F32", [4])})
    (tmp_path / "config.json").write_text(body, encoding="utf-8")
    with pytest.raises(BadRequest, match="rather than a set of fields"):
        fit.plan(tmp_path, seq_len=8)


def test_a_parsed_config_that_is_not_a_mapping_is_refused_at_the_entry():
    """`kv_geometry` is called directly by anything that parsed a config
    itself, so the guard cannot live only in the file reader."""
    with pytest.raises(BadRequest, match="rather than a set of fields"):
        fit.kv_geometry([1, 2, 3], None)


def test_zero_attention_heads_is_named_rather_than_divided_by():
    """Measured: `{"num_hidden_layers": 2, "num_attention_heads": 0,
    "hidden_size": 16}` raised `ZeroDivisionError: integer division or modulo
    by zero` from `_head_dim`. `need()` refused an ABSENT key and accepted any
    present one — one guarded case beside one bare one."""
    with pytest.raises(BadRequest, match="num_attention_heads"):
        fit.kv_geometry(
            {"num_hidden_layers": 2, "num_attention_heads": 0, "hidden_size": 16}
        )


def test_zero_heads_with_a_stated_head_dim_is_worse_than_the_crash():
    """The same config plus `head_dim` did not crash at all. It returned
    `KVGeometry(n_heads=0, n_kv_heads=0)` and `kv_cache_bytes(...)` came back
    as 0 bytes at 4096 tokens — a cache that costs nothing, for a model that
    then dies on the first forward pass. Nothing about that number looks
    wrong, which is what makes it the more dangerous half."""
    cfg = {
        "num_hidden_layers": 2,
        "num_attention_heads": 0,
        "hidden_size": 16,
        "head_dim": 64,
    }
    with pytest.raises(BadRequest, match="zero bytes"):
        fit.kv_geometry(cfg)


def test_a_negative_layer_count_never_shrinks_the_prediction():
    """Measured on `num_hidden_layers: -5`: `kv_cache_bytes` returned
    -10,485,760 and `attention_bytes` -1,342,177,280 at 4096 tokens, both
    SUBTRACTING from `plan()`'s total and moving it towards "it fits" — which
    `longest_context` promises in its own docstring never to do ("never a
    negative or a fabricated minimum")."""
    with pytest.raises(BadRequest, match="negative bytes"):
        fit.kv_geometry(dict(CONFIG, num_hidden_layers=-5), None)


def test_a_stated_zero_kv_head_count_is_not_swallowed_by_the_mha_default():
    """`int(config.get("num_key_value_heads") or n_heads)` treated a stated 0
    as absent. Measured on an 8-head config: `n_kv_heads=8` — the config's own
    number thrown away for one four times larger, by the line whose docstring
    promises never to substitute a default."""
    with pytest.raises(BadRequest, match="num_key_value_heads"):
        fit.kv_geometry(dict(CONFIG, num_key_value_heads=0), None)


def test_an_absent_kv_head_count_still_means_mha():
    """The guard above must not turn the ABSENT case into a refusal: absent is
    the definition of MHA, not a missing value."""
    cfg = {k: v for k, v in CONFIG.items() if k != "num_key_value_heads"}
    assert fit.kv_geometry(cfg, None).n_kv_heads == 8


def test_a_stated_zero_head_dim_is_refused_rather_than_falling_through():
    """`if stated:` sent `head_dim: 0` down the quotient path, where a config
    that also stated 0 heads produced a KV cache of 0 bytes. A stated zero is a
    broken config; an omitted key is a different sentence."""
    with pytest.raises(BadRequest, match="head_dim"):
        fit.kv_geometry(dict(CONFIG, head_dim=0), None)


def test_a_hidden_size_smaller_than_the_head_count_is_refused():
    """`16 // 32` is 0, and a head_dim of 0 makes the whole KV term vanish —
    reported as free rather than as unknown. Both numbers are stated and they
    contradict each other, which is worth saying rather than flooring."""
    with pytest.raises(BadRequest, match="less than one element per head"):
        fit.kv_geometry(dict(CONFIG, hidden_size=16, num_attention_heads=32), None)


@pytest.mark.parametrize("value", [True, "abc", 4.7, [4]])
def test_a_count_that_is_not_a_count_is_named_not_coerced(value):
    """`int(True)` is 1 and `int(4.7)` is 4, so a flag and a fraction both read
    as plausible layer counts. A figure this calculator invented is exactly
    what its docstring says it will never show you as one it read."""
    with pytest.raises(BadRequest, match="num_hidden_layers"):
        fit.kv_geometry(dict(CONFIG, num_hidden_layers=value), None)


def test_a_dtype_width_of_zero_would_price_every_term_at_nothing(model_dir):
    """Every term in `plan` is multiplied by `dtype_bytes`. At 0 the weights,
    the cache and the attention buffer all come out at nothing and `fits` is
    True against any budget — a verdict, and the wrong one, from a number
    nobody supplied on purpose."""
    with pytest.raises(BadRequest, match="dtype_bytes"):
        fit.plan(model_dir, seq_len=8, dtype_bytes=0)


def test_no_malformed_config_can_make_plan_smaller_than_the_weights(tmp_path):
    """The property the guards above exist to hold, checked as one: whatever a
    damaged config says, `plan` either answers with a total that is at least
    the weights it actually read, or it refuses. It never publishes a shrunken
    one, and it never publishes a free cache."""
    write_safetensors(tmp_path / "model.safetensors", {"w": ("F32", [64, 64])})
    floor = 64 * 64 * 4
    damaged = [
        dict(CONFIG, num_hidden_layers=-5),
        dict(CONFIG, num_hidden_layers=0),
        dict(CONFIG, num_attention_heads=0),
        dict(CONFIG, num_key_value_heads=0),
        dict(CONFIG, head_dim=0),
        dict(CONFIG, head_dim=-8),
        dict(CONFIG, hidden_size=0),
    ]
    for cfg in damaged:
        (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        try:
            f = fit.plan(tmp_path, seq_len=4096, dtype_bytes=1)
        except BadRequest:
            continue
        assert f.kv_bytes > 0, cfg
        assert f.attention_bytes > 0, cfg
        assert f.total_bytes >= floor, cfg
