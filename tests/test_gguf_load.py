# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""Loading a GGUF costs three times its file size, and the tool must say so.

The single belief this module exists to correct is that a 4-bit GGUF loads as
a 4-bit model. Transformers has no kernels for these quantised types, so it
dequantises everything on the way in. Measured on this repo's own machine:

    Qwen3-0.6B-Q4_K_M.gguf     0.397 GB on disk -> 1.192 GB resident (3.00x)
    SmolLM2-135M-Q4_K_M.gguf   0.105 GB on disk -> 0.269 GB resident (2.55x)

The resident figure is `parameters x dtype bytes` and it is exact — error
0.000000 against the built module on both. The transit figure is
`parameters x 4`, which is a PREDICTION: sampled RSS came in at -3.5% on the
first and +8.6% on the second, so it is good to about ten percent and not to
the digit. The tests below assert the arithmetic, not the sampling.

More important than either: the refusals. A preflight that is optimistic is
worse than none, because the failure it fails to predict arrives twenty
minutes into a download.

Nothing here downloads. The load path itself is exercised by
`scripts/measure_docs.py --gguf` against a real file; what a unit test can
own is the decision, and the decision is made from numbers.
"""

from __future__ import annotations

import struct

import pytest
from test_gguf_read import build  # the same synthetic-header builder

from modelmri import gguf_load
from modelmri.errors import BadRequest, Refusal

# A gguf_read type id and how many elements a tensor of that shape holds.
F32 = 0


def _model(tmp_path, *, arch="llama", elements=1_000_000, name="model.gguf"):
    """A header describing a model of a known parameter count."""
    path = build(
        tmp_path,
        metadata={"general.architecture": (0x8, arch)},  # _STRING
        tensors=[("token_embd.weight", F32, [elements], 0)],
    )
    return path.rename(path.parent / name)


# ------------------------------------------------------------- picking a file


def test_a_missing_path_is_named(tmp_path):
    with pytest.raises(BadRequest, match="no such file"):
        gguf_load.find_file(tmp_path / "nope.gguf")


def test_a_directory_with_one_gguf_resolves_to_it(tmp_path):
    p = _model(tmp_path, name="only.gguf")
    assert gguf_load.find_file(tmp_path) == p


def test_a_directory_with_several_is_refused_rather_than_guessed(tmp_path):
    """Repos ship Q4_K_M beside Q8_0 beside BF16. Picking one for the user is
    picking which quantisation their measurements describe."""
    _model(tmp_path, name="a-Q4_K_M.gguf")
    _model(tmp_path, name="b-Q8_0.gguf")
    with pytest.raises(BadRequest, match="2 GGUF files"):
        gguf_load.find_file(tmp_path)


def test_the_refusal_lists_the_candidates_in_a_stable_order(tmp_path):
    for n in ("c.gguf", "a.gguf", "b.gguf"):
        _model(tmp_path, name=n)
    with pytest.raises(BadRequest) as err:
        gguf_load.find_file(tmp_path)
    assert "a.gguf, b.gguf, c.gguf" in str(err.value)


def test_an_empty_directory_says_so(tmp_path):
    with pytest.raises(BadRequest, match="no .gguf file"):
        gguf_load.find_file(tmp_path)


@pytest.mark.parametrize(
    "name,expect",
    [
        ("mmproj-gemma-4-E2B-it-BF16.gguf", "multimodal projector"),
        ("gemma-4-E2B-it-mmproj.gguf", "multimodal projector"),
        # Underscore and dot separators. Both are real spellings and both
        # slipped through the first rule, which only knew about `-`.
        ("mmproj_model_f16.gguf", "multimodal projector"),
        ("Qwen2.5-VL-7B.mmproj-f16.gguf", "multimodal projector"),
        ("mtp-gemma-4-E2B-it-BF16.gguf", "multi-token-prediction"),
        # A leading `mtp` token is how llama.cpp names these, so this is a
        # refusal rather than a false positive.
        ("mtp-tuned-7b.gguf", "multi-token-prediction"),
    ],
)
def test_the_companion_files_in_a_gguf_repo_are_refused_by_name(tmp_path, name, expect):
    """These sit next to the model in every modern GGUF repo and are not
    language models. Loading one produces a stack trace from four frames deep
    in transformers about a missing block_count."""
    p = _model(tmp_path, name=name)
    with pytest.raises(gguf_load.Unsupported, match=expect):
        gguf_load.find_file(p)


@pytest.mark.parametrize(
    "name",
    [
        # The marker is a PREFIX of a longer token, not a token. This is the
        # direction that genuinely needs protecting.
        "mtpc-7b-Q4_K_M.gguf",
        "llama-mtpx-7b.gguf",
        "mmprojector-notes-7b.gguf",
    ],
)
def test_a_name_that_merely_starts_with_the_marker_is_not_refused(tmp_path, name):
    """Token-bounded, not substring. The first rule refused `llama-mtpx-7b`
    for its spelling."""
    p = _model(tmp_path, name=name)
    assert gguf_load.find_file(p) == p


@pytest.mark.parametrize(
    "name",
    ["mixture-of-experts-8x7b.gguf", "chain-of-thought-7b.gguf"],
)
def test_a_name_containing_of_is_not_mistaken_for_a_shard(tmp_path, name):
    """`gguf-split` emits `-NNNNN-of-NNNNN`, so the digits are the signal. The
    bare `-of-` substring told mixture-of-experts models to run
    `gguf-split --merge` on a file that is not split."""
    p = _model(tmp_path, name=name)
    assert gguf_load.find_file(p) == p


def test_a_split_gguf_shard_is_refused_with_the_merge_command(tmp_path):
    p = _model(tmp_path, name="model-00001-of-00003.gguf")
    with pytest.raises(gguf_load.Unsupported, match="gguf-split --merge"):
        gguf_load.find_file(p)


def test_a_non_gguf_file_is_refused(tmp_path):
    q = tmp_path / "model.safetensors"
    q.write_bytes(b"\0" * 16)
    with pytest.raises(BadRequest, match="not a .gguf"):
        gguf_load.find_file(q)


# ------------------------------------------------------ what the numbers mean


def test_resident_is_parameters_times_dtype_not_the_file_size(tmp_path):
    """The whole point. A 4 MB file of 1M float32 parameters is 2 MB at
    bfloat16 — and a 4-bit file of the same 1M parameters is ALSO 2 MB at
    bfloat16, because the quantisation is gone by then."""
    p = _model(tmp_path, elements=1_000_000)
    plan = gguf_load.plan(p, dtype="bfloat16", device_kind="cpu")
    assert plan.parameters == 1_000_000
    assert plan.resident_bytes == 2_000_000
    assert plan.resident_bytes != plan.file_bytes


def test_the_peak_is_float32_even_when_bfloat16_was_asked_for(tmp_path):
    """Transformers materialises the whole dequantised checkpoint as float32
    before casting, so asking for bfloat16 does not avoid it. The prediction
    for a 596M model is 2.384 GB against a 1.192 GB result; sampled RSS came
    in at 2.30 GB, -3.5%."""
    p = _model(tmp_path, elements=1_000_000)
    plan = gguf_load.plan(p, dtype="bfloat16", device_kind="cpu")
    assert plan.peak_host_bytes == 4_000_000
    assert plan.peak_host_bytes == 2 * plan.resident_bytes


def test_float32_asks_for_no_transit_beyond_itself(tmp_path):
    p = _model(tmp_path, elements=1_000_000)
    plan = gguf_load.plan(p, dtype="float32", device_kind="cpu")
    assert plan.peak_host_bytes == plan.resident_bytes


def test_the_expansion_ratio_is_reported(tmp_path):
    p = _model(tmp_path, elements=1_000_000)
    plan = gguf_load.plan(p, dtype="bfloat16", device_kind="cpu")
    assert plan.expansion == pytest.approx(
        plan.resident_bytes / plan.file_bytes, rel=1e-9
    )


def test_the_dict_explains_which_number_is_which(tmp_path):
    p = _model(tmp_path, elements=1_000)
    d = gguf_load.plan(p, dtype="bfloat16", device_kind="cpu").to_dict()
    assert "parameters x dtype bytes" in d["means"]
    assert "float32" in d["means"]


def test_the_notes_say_the_measurements_describe_the_quantised_model(tmp_path):
    p = _model(tmp_path, elements=1_000)
    plan = gguf_load.plan(p, dtype="bfloat16", device_kind="cpu")
    assert any("QUANTISED" in n for n in plan.notes)


def test_the_notes_name_the_expansion_and_its_cause(tmp_path):
    p = _model(tmp_path, elements=1_000_000)
    plan = gguf_load.plan(p, dtype="bfloat16", device_kind="cpu")
    assert any("dequantises" in n for n in plan.notes)


# --------------------------------------------------------------- the verdict


def _v(**kw):
    base = dict(
        resident=1_000_000_000,
        peak=2_000_000_000,
        device_kind="cuda",
        device_free=8_000_000_000,
        host_free=16_000_000_000,
        host_total=32_000_000_000,
    )
    base.update(kw)
    return gguf_load._verdict(**base)


def test_a_comfortable_load_fits():
    verdict, why = _v()
    assert verdict == "fits"
    assert "GB" in why


def test_weights_larger_than_the_device_will_not_fit():
    verdict, why = _v(resident=9_000_000_000, device_free=8_000_000_000)
    assert verdict == "will not fit"
    assert "file being smaller" in why


def test_a_transit_larger_than_total_ram_says_closing_things_cannot_help():
    """The measured Gemma 4 E2B case: 18.51 GB of float32 transit on a machine
    with 16.94 GB of RAM. Reporting only "3 GB free" sends someone off to quit
    Chrome for twenty minutes to no effect."""
    verdict, why = _v(
        peak=18_510_000_000, host_free=3_000_000_000, host_total=16_940_000_000
    )
    assert verdict == "will not fit"
    assert "cannot change that" in why
    assert "16.94 GB" in why


def test_a_transit_that_merely_exceeds_free_ram_reports_both_numbers():
    verdict, why = _v(
        peak=20_000_000_000, host_free=8_000_000_000, host_total=32_000_000_000
    )
    assert verdict == "will not fit"
    assert "8.00 GB" in why and "32.00 GB" in why


def test_the_total_ram_check_comes_first():
    """Both arms are true when total is exceeded; the one giving useful advice
    has to win."""
    verdict, why = _v(
        peak=40_000_000_000, host_free=1_000_000_000, host_total=32_000_000_000
    )
    assert "cannot change that" in why


def test_a_near_miss_on_the_device_is_tight_not_fits():
    verdict, why = _v(resident=7_800_000_000, device_free=8_000_000_000)
    assert verdict == "tight"
    assert "activations" in why


def test_an_unreported_device_is_unknown_never_zero():
    """A guard that refuses because it could not measure locks out every
    platform it cannot see."""
    verdict, why = _v(device_free=None)
    assert verdict == "unknown"
    assert "does not report" in why


def test_an_unreported_host_on_cpu_is_unknown_too():
    verdict, _ = _v(device_kind="cpu", host_free=None, host_total=None)
    assert verdict == "unknown"


def test_on_cpu_the_transit_is_the_whole_test():
    """Transit and residency share one pool there, and the transit is larger."""
    verdict, why = _v(
        device_kind="cpu",
        peak=4_000_000_000,
        resident=2_000_000_000,
        host_free=20_000_000_000,
        host_total=32_000_000_000,
    )
    assert verdict == "fits"
    assert "4.00 GB peak" in why


def test_cpu_at_ninety_percent_of_free_is_tight():
    verdict, why = _v(
        device_kind="cpu",
        peak=19_000_000_000,
        host_free=20_000_000_000,
        host_total=32_000_000_000,
    )
    assert verdict == "tight"
    assert "thrash" in why


# ------------------------------------------------------------- architectures


def test_the_supported_list_is_read_from_transformers_not_hardcoded():
    """It grows every release, and a copy here would refuse models the library
    in front of it handles."""
    archs = gguf_load.supported_architectures()
    assert "llama" in archs and "qwen3" in archs


def test_the_metadata_sections_are_not_offered_as_architectures():
    """`general` and `tokenizer` are sections in that table. Listing them in a
    refusal sends people looking for a "general" model."""
    archs = gguf_load.supported_architectures()
    assert "general" not in archs and "tokenizer" not in archs


def test_an_unsupported_architecture_is_refused_with_the_list(tmp_path):
    p = _model(tmp_path, arch="rwkv7")
    with pytest.raises(gguf_load.Unsupported) as err:
        gguf_load.plan(p, device_kind="cpu")
    assert "rwkv7" in str(err.value)
    assert "llama" in str(err.value)


def test_the_refusal_says_the_header_still_reads(tmp_path):
    """Refusing the forward pass is not refusing the file. Bits per weight and
    quantisation damage work on any GGUF this reader can parse."""
    p = _model(tmp_path, arch="rwkv7")
    with pytest.raises(gguf_load.Unsupported, match="header still reads"):
        gguf_load.plan(p, device_kind="cpu")


def test_a_file_with_no_architecture_is_noted_not_refused(tmp_path):
    """Transformers may still infer one. Refusing here would be this module
    deciding a question it does not own."""
    path = build(
        tmp_path,
        metadata={},
        tensors=[("token_embd.weight", F32, [1000], 0)],
    )
    plan = gguf_load.plan(path, device_kind="cpu")
    assert plan.architecture is None
    assert any("no general.architecture" in n for n in plan.notes)


# -------------------------------------------------------------- the refusals


def test_load_refuses_a_model_that_will_not_fit_without_downloading(
    tmp_path, monkeypatch
):
    p = _model(tmp_path, elements=10_000_000_000)
    monkeypatch.setattr(gguf_load, "_host_total", lambda: 8_000_000_000)
    monkeypatch.setattr(gguf_load, "_host_free", lambda: 6_000_000_000)
    monkeypatch.setattr(gguf_load, "_require_gguf", lambda: None)
    with pytest.raises(Refusal) as err:
        gguf_load.load(p, dtype="bfloat16", device="cpu")
    assert "will not load here" in str(err.value)
    # And it names the alternative rather than just saying no.
    assert "ollama" in str(err.value)


def test_confirm_does_not_override_arithmetic(tmp_path, monkeypatch):
    """ "tight" is a preference. "will not fit" is that the RAM needed exceeds
    the RAM that exists, and confirming does not create any."""
    p = _model(tmp_path, elements=10_000_000_000)
    monkeypatch.setattr(gguf_load, "_host_total", lambda: 8_000_000_000)
    monkeypatch.setattr(gguf_load, "_host_free", lambda: 6_000_000_000)
    monkeypatch.setattr(gguf_load, "_require_gguf", lambda: None)
    with pytest.raises(Refusal, match="will not load here"):
        gguf_load.load(p, dtype="bfloat16", device="cpu", confirm=True)


def test_a_tight_load_is_refused_until_confirmed(tmp_path, monkeypatch):
    p = _model(tmp_path, elements=1_000_000_000)  # 4 GB transit
    monkeypatch.setattr(gguf_load, "_host_total", lambda: 32_000_000_000)
    monkeypatch.setattr(gguf_load, "_host_free", lambda: 4_200_000_000)
    monkeypatch.setattr(gguf_load, "_require_gguf", lambda: None)
    with pytest.raises(Refusal, match="confirm=True"):
        gguf_load.load(p, dtype="bfloat16", device="cpu")


def test_the_missing_dependency_names_the_extra_and_what_does_not_need_it():
    """Reading the header needs nothing. Only unpacking blocks does, and the
    message has to keep those apart or people install a package to read a
    table they could already read."""
    import builtins

    real = builtins.__import__

    def no_gguf(name, *a, **kw):
        if name == "gguf":
            raise ImportError("no gguf")
        return real(name, *a, **kw)

    builtins.__import__ = no_gguf
    try:
        with pytest.raises(Refusal) as err:
            gguf_load._require_gguf()
    finally:
        builtins.__import__ = real
    assert "modelmri[gguf]" in str(err.value)
    assert "gguf_read" in str(err.value)


# ------------------------------------------------------------ the load report


def test_the_report_carries_the_prediction_error(tmp_path):
    """Measured, not trusted. The prediction is arithmetic on a header and a
    header can describe a file that does not load the way it says. On
    Qwen3-0.6B-Q4_K_M the error was exactly 0."""
    p = _model(tmp_path, elements=1_000_000)
    plan = gguf_load.plan(p, dtype="bfloat16", device_kind="cpu")
    loaded = gguf_load.Loaded(
        model=None,
        tokenizer=None,
        plan=plan,
        measured_resident_bytes=plan.resident_bytes,
        load_seconds=1.0,
    )
    assert loaded.prediction_error == 0.0
    assert loaded.to_dict()["prediction_error"] == 0.0


def test_a_prediction_that_missed_is_reported_signed(tmp_path):
    p = _model(tmp_path, elements=1_000_000)
    plan = gguf_load.plan(p, dtype="bfloat16", device_kind="cpu")
    loaded = gguf_load.Loaded(
        model=None,
        tokenizer=None,
        plan=plan,
        measured_resident_bytes=int(plan.resident_bytes * 1.5),
        load_seconds=1.0,
    )
    assert loaded.prediction_error == pytest.approx(0.5)


def test_the_report_is_json_safe(tmp_path):
    import json

    p = _model(tmp_path, elements=1_000)
    plan = gguf_load.plan(p, dtype="bfloat16", device_kind="cpu")
    json.dumps(
        gguf_load.Loaded(
            model=None,
            tokenizer=None,
            plan=plan,
            measured_resident_bytes=plan.resident_bytes,
            load_seconds=0.5,
        ).to_dict()
    )


# --------------------------------------------------------------- block lookup


def test_blocks_are_found_at_each_known_path():
    class Node:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    layers = [1, 2, 3]
    assert gguf_load.blocks_of(Node(model=Node(layers=layers))) is layers
    assert gguf_load.blocks_of(Node(transformer=Node(h=layers))) is layers
    assert gguf_load.blocks_of(Node(gpt_neox=Node(layers=layers))) is layers


def test_an_unreachable_block_list_says_what_still_works():
    class Bare:
        pass

    with pytest.raises(Refusal, match="Logit lens and generation still work"):
        gguf_load.blocks_of(Bare())


def test_struct_is_available():
    """The synthetic builder is shared with test_gguf_read; if its import
    shape changes this file fails here rather than in twenty places."""
    assert struct.calcsize("<Q") == 8
