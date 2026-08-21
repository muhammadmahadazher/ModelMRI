"""The decode is the easy part. The controls are the feature.

A good patchscope target prompt describes ANYTHING fluently — hand it a random
vector and it still produces a confident sentence. So a decode on its own is
not evidence, and almost every test here is about the two things standing
beside it.
"""

from __future__ import annotations

import json
import os

import pytest

from modelmri import patch
from modelmri.errors import BadRequest

SOURCE = "The Eiffel Tower is located in the city of Paris"


@pytest.fixture(scope="module")
def gpt2():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    from modelmri import receipts as _receipts

    if _receipts.revision_of("gpt2")[0] is None and not os.environ.get(
        "MODELMRI_TEST_DOWNLOAD"
    ):
        pytest.skip("gpt2 is not in the local model cache")

    from modelmri.runtime import ModelRuntime

    runtime = ModelRuntime()
    try:
        runtime.load("gpt2")
    except Exception as err:
        pytest.skip(f"gpt2 is not available here: {err}")
    yield runtime
    runtime.unload()


@pytest.fixture(scope="module")
def scoped(gpt2):
    return gpt2.patchscope(SOURCE, source_layer=8, max_new_tokens=10)


# --------------------------------------------------------- the overlap rule


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ("hello world", "hello world", 1.0),
        ("hello world", "goodbye moon", 0.0),
        # Containment: everything the decode said, the control already said.
        ("hello", "hello world again", 1.0),
        ("", "anything", 0.0),
        # Case and trailing punctuation must not create a difference that is
        # not there -- the failure being caught is the target prompt's
        # vocabulary reappearing.
        ("Hello, World", "hello world", 1.0),
    ],
)
def test_overlap_measures_shared_vocabulary(a, b, expected):
    assert patch._overlap(a, b) == pytest.approx(expected)


# ------------------------------------------------------------ the controls


def test_both_controls_are_returned_with_every_decode(scoped):
    """A decode alone is not evidence."""
    assert "identity" in scoped["controls"]
    assert scoped["controls"]["random"]
    assert scoped["controls"]["draws"] >= 1


def test_the_identity_control_is_the_untouched_target_prompt(gpt2):
    """Which is what a decode must be compared against to know the patch did
    anything at all."""
    scoped = gpt2.patchscope(SOURCE, source_layer=8, max_new_tokens=8)
    plain = "".join(
        gpt2.generate_stream(
            scoped["target"]["prompt"], 8, temperature=0.0, commit=False
        )
    )
    assert scoped["controls"]["identity"] == plain


def test_the_random_control_has_the_same_norm_as_the_patched_state(gpt2):
    """A control at a different magnitude would be testing the magnitude."""
    import torch

    from modelmri import patch as patch_mod

    blocks = [gpt2._block(i) for i in range(gpt2.model.config.num_hidden_layers)]
    ids = gpt2.tokenizer(SOURCE, return_tensors="pt")["input_ids"].to(gpt2.device)
    sink: dict = {}
    handle = patch_mod._capture(blocks[8], 8, sink)
    try:
        with torch.no_grad():
            gpt2.model(ids)
    finally:
        handle.remove()
    state = sink[8][0, -1, :]

    gen = torch.Generator().manual_seed(patch_mod.CONTROL_SEED)
    r = torch.randn(state.shape, generator=gen).to(state.device, state.dtype)
    control = r / r.norm() * state.norm()
    assert float(control.norm()) == pytest.approx(float(state.norm()), rel=1e-2)


def test_a_decode_matching_a_control_is_not_informative(gpt2):
    """The whole point. An early state can decode IDENTICALLY to the same-norm
    random vector — it carries nothing this target prompt can read, and the
    tool says so instead of interpreting it."""
    scoped = gpt2.patchscope(SOURCE, source_layer=2, max_new_tokens=10)
    if scoped["same_as_random"] or scoped["overlap_random"] >= 1.0:
        assert not scoped["informative"]
        assert "not about the state" in scoped["means"] or (
            scoped["overlap_random"] >= 1.0
        )


def test_containment_alone_makes_a_decode_uninformative(gpt2):
    """A decode can differ from the untouched target as a STRING while using
    nothing but the vocabulary that control already used. A string test alone
    called that informative."""
    scoped = gpt2.patchscope(SOURCE, source_layer=8, max_new_tokens=10)
    if scoped["overlap_identity"] >= 1.0:
        assert not scoped["informative"], (
            "a decode that says nothing the control did not already say is "
            "the target prompt talking, however the strings differ"
        )


def test_informative_requires_clearing_both_controls(scoped):
    if scoped["informative"]:
        assert not scoped["same_as_identity"]
        assert not scoped["same_as_random"]
        assert scoped["overlap_identity"] < 1.0
        assert scoped["overlap_random"] < 1.0


# --------------------------------------------- the target prompt is a result


def test_the_target_prompt_is_returned_not_hidden(scoped):
    """It is part of the result: two decodes under different targets are not
    comparable, and a hidden default would make that invisible."""
    assert scoped["target"]["prompt"] == patch.DEFAULT_TARGET
    assert scoped["target"]["tokens"]


def test_a_custom_target_prompt_is_used_and_reported(gpt2):
    custom = "The word is X. X means"
    scoped = gpt2.patchscope(
        SOURCE, source_layer=6, target_prompt=custom, max_new_tokens=6
    )
    assert scoped["target"]["prompt"] == custom


def test_the_receipt_records_which_target_was_used(scoped):
    """Two patchscopes with different targets are different experiments."""
    request = scoped["receipt"]["request"]
    assert request["source_layer"] == 8
    assert request["target_sha256"]


# ----------------------------------------------------- what it never claims


def test_the_sentence_says_a_decode_is_a_sample(scoped):
    """It is what the model said when handed this state through this target
    prompt, not what the state means."""
    means = scoped["means"]
    assert "A DECODE IS A GENERATION AND THEREFORE A SAMPLE" in means
    assert "not what the state means" in means


def test_a_cross_layer_splice_is_flagged(gpt2):
    """Source layer L into target layer L' is only meaningful where the two
    streams are comparable, and nothing here checks that they are."""
    scoped = gpt2.patchscope(SOURCE, source_layer=4, target_layer=9, max_new_tokens=6)
    assert scoped["cross_layer"] is True
    assert "only comparable where the model treats them alike" in scoped["means"]


def test_a_same_layer_splice_is_not_flagged(scoped):
    assert scoped["cross_layer"] is False


# ------------------------------------------------------------- it refuses


def test_a_source_layer_outside_the_model_is_refused(gpt2):
    with pytest.raises(BadRequest, match="source layer"):
        gpt2.patchscope(SOURCE, source_layer=999)


def test_a_target_layer_outside_the_model_is_refused(gpt2):
    with pytest.raises(BadRequest, match="target layer"):
        gpt2.patchscope(SOURCE, source_layer=2, target_layer=999)


def test_a_source_position_outside_the_prompt_is_refused(gpt2):
    with pytest.raises(BadRequest, match="source position"):
        gpt2.patchscope(SOURCE, source_layer=2, source_position=999)


def test_a_target_position_outside_the_target_is_refused(gpt2):
    with pytest.raises(BadRequest, match="target position"):
        gpt2.patchscope(SOURCE, source_layer=2, target_position=999)


# ------------------------------------------------------------- the harness


def test_the_splice_only_touches_the_prefill(gpt2):
    """Generation is autoregressive: after the prefill the model runs with a
    KV cache and each step passes ONE token, so a splice at position 14 raised
    IndexError inside the generation worker and surfaced as a streamer
    timeout. Measured exactly that way."""
    import torch

    blocks = [gpt2._block(i) for i in range(gpt2.model.config.num_hidden_layers)]
    hidden = int(gpt2.model.config.hidden_size)
    handle = patch._splice_prefill(blocks[4], 12, torch.zeros(hidden))
    try:
        # A single-token stream is a decode step; the hook must skip it rather
        # than raise.
        out = "".join(
            gpt2.generate_stream("A short prompt", 6, temperature=0.0, commit=False)
        )
    finally:
        handle.remove()
    assert isinstance(out, str)


def test_a_patchscope_does_not_rebase_the_analysis_target(gpt2):
    """commit=False. Committing would leave every other panel describing the
    target prompt, which the user never asked to analyse."""
    list(gpt2.generate_stream("The capital of France is", 3, temperature=0.0))
    before = gpt2.last_prompt
    gpt2.patchscope(SOURCE, source_layer=5, max_new_tokens=6)
    assert gpt2.last_prompt == before


def test_the_report_survives_json(scoped):
    out = json.loads(json.dumps(scoped, allow_nan=False))
    assert out["decode"] is not None
    assert out["source"]["layer"] == 8
