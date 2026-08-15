"""A translator fitted to agree with the model will agree with the model.

That is the method, not a flaw in it — which is why the only number this
module is allowed to report is HELD-OUT KL, and why the plain lens never
leaves the screen. Most of these tests are about those two rules holding.
"""

from __future__ import annotations

import json
import os

import pytest

from modelmri import tuned_lens as tl
from modelmri.errors import BadRequest, Refusal

CORPUS = [
    "The capital of France is Paris, on the Seine.",
    "The capital of Italy is Rome, on the Tiber.",
    "Water boils at one hundred degrees at sea level.",
    "The train leaves every twenty minutes on weekdays.",
    "A dog walked slowly along the wet path.",
    "Coffee grown at altitude tastes brighter.",
    "The weather turned cold and the rain continued.",
    "She opened the book and read the first chapter.",
    "The bridge was built from stone from the valley.",
    "Mount Everest sits on the border of Nepal.",
    "The Eiffel Tower was completed in 1889.",
    "Tokyo is the largest city in the world.",
]


# ------------------------------------------------------------ the corpus id


def test_the_same_corpus_hashes_the_same_way_in_any_order():
    """The same sequences read from a `.txt` and from the trace store must
    produce one lens, not two caches of the same thing."""
    assert tl.corpus_hash(CORPUS) == tl.corpus_hash(list(reversed(CORPUS)))


def test_a_different_corpus_hashes_differently():
    assert tl.corpus_hash(CORPUS) != tl.corpus_hash(CORPUS[:-1])


def test_the_cache_key_carries_all_four_things_that_change_the_lens(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("MODELMRI_HOME", str(tmp_path))
    a = tl.cache_path("gpt2", "float32", "abc", 1000)
    for changed in (
        tl.cache_path("gpt2-medium", "float32", "abc", 1000),
        tl.cache_path("gpt2", "bfloat16", "abc", 1000),
        tl.cache_path("gpt2", "float32", "def", 1000),
        tl.cache_path("gpt2", "float32", "abc", 2000),
    ):
        assert changed != a, "each of the four must change the cache key"


def test_a_model_id_with_a_slash_does_not_escape_the_cache_directory(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("MODELMRI_HOME", str(tmp_path))
    path = tl.cache_path("org/model", "float32", "abc", 10)
    assert path.parent.name == "tuned-lenses"
    assert "/" not in path.name and "\\" not in path.name


# ------------------------------------------------------------- the split


def test_the_split_is_deterministic():
    """A cached lens and a freshly trained one must report the same held-out
    number for the same corpus, or there is no telling which to believe."""
    assert tl._split(CORPUS) == tl._split(list(reversed(CORPUS)))


def test_the_split_holds_some_back_and_they_do_not_overlap():
    train, held = tl._split(CORPUS)
    assert held, "something must be held back"
    assert train, "and something must be left to train on"
    assert not (set(train) & set(held))
    assert set(train) | set(held) == set(CORPUS)


def test_too_few_sequences_is_refused_rather_than_split_anyway():
    """One held-out sequence gives a held-out KL with no spread, which is the
    number this module exists to make trustworthy."""

    class _Model:
        pass

    with pytest.raises(BadRequest, match="at least"):
        tl.train(_Model(), None, ["a", "b"], corpus_label="tiny")


# --------------------------------------------------------------- the report


def _info(**over) -> tl.TunedLensInfo:
    kw = dict(
        model_id="gpt2",
        dtype="float32",
        n_layers=2,
        d_model=768,
        corpus_label="12 sentences",
        corpus_sha256="abc",
        n_sequences=12,
        n_tokens=200,
        n_held_out=3,
        n_held_out_scored=3,
        steps=100,
        lr=1e-3,
        seconds=5.0,
        layers=[
            tl.LayerFit(layer=0, plain_kl=10.0, tuned_kl=2.0),
            tl.LayerFit(layer=1, plain_kl=1.0, tuned_kl=1.5),
        ],
    )
    kw.update(over)
    return tl.TunedLensInfo(**kw)


def test_a_layer_the_translator_made_worse_reports_a_negative_gain():
    """Not clamped to zero. A translator that hurt a layer is a finding."""
    info = _info()
    assert info.layers[0].gain == 8.0
    assert info.layers[1].gain == -0.5
    assert len(info.helped) == 1


def test_the_report_says_how_many_layers_actually_improved():
    out = _info().to_dict()
    assert out["n_layers_improved"] == 1
    assert "1 of 2 layers" in out["means"]


def test_the_report_never_mentions_training_loss():
    """A translator's training KL is a statement about the translator."""
    out = json.dumps(_info().to_dict()).lower()
    assert "train_kl" not in out and "training_kl" not in out
    assert "held-out" in out


def test_a_corpus_far_too_small_for_the_fit_says_so():
    """12 sentences fitting 590K parameters per layer is three orders of
    magnitude under-determined. The held-out number is still real; how narrow
    the text was is a separate fact and both are reported."""
    info = _info(n_tokens=200)
    assert info.tokens_per_parameter < 1
    assert "lens for that corpus" in info.caution


def test_a_corpus_large_enough_raises_no_caution():
    info = _info(n_tokens=1_000_000)
    assert info.caution == ""


def test_the_means_sentence_names_the_corpus():
    """The corpus is part of the measurement: the same prompt through a lens
    fitted to different text is a different reading."""
    assert "12 sentences" in _info().means()


# ------------------------------------------------------- loading a saved one


def test_a_lens_for_another_model_is_refused(tmp_path):
    """A translator is only meaningful for the model it was fitted to.
    Loading one across would produce a confident, plausible, wrong reading."""
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    path = tmp_path / "lens.safetensors"
    save_file(
        {"A.0": torch.eye(4), "b.0": torch.zeros(4)},
        str(path),
        metadata={"modelmri": json.dumps({"model_id": "gpt2", "dtype": "float32"})},
    )
    with pytest.raises(Refusal, match="fitted to gpt2"):
        tl.load(path, model_id="gpt2-medium", dtype="float32")


def test_a_lens_fitted_in_another_dtype_is_refused(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    path = tmp_path / "lens.safetensors"
    save_file(
        {"A.0": torch.eye(4), "b.0": torch.zeros(4)},
        str(path),
        metadata={"modelmri": json.dumps({"model_id": "gpt2", "dtype": "float32"})},
    )
    with pytest.raises(Refusal, match="fitted in float32"):
        tl.load(path, model_id="gpt2", dtype="bfloat16")


def test_a_missing_lens_says_so(tmp_path):
    with pytest.raises(BadRequest, match="no tuned lens"):
        tl.load(tmp_path / "nope.safetensors", model_id="gpt2", dtype="float32")


# --------------------------------------------------- against a real model


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
def trained(gpt2):
    if gpt2.tokenizer.pad_token is None:
        gpt2.tokenizer.pad_token = gpt2.tokenizer.eos_token
    return tl.train(
        gpt2.model, gpt2.tokenizer, CORPUS, corpus_label="12 sentences", steps=40
    )


def test_the_projection_is_made_before_training(gpt2):
    projected = tl.plan(gpt2.model, CORPUS)
    assert projected["d_model"] == 768
    assert projected["params_per_layer"] == 768 * 768 + 768
    assert projected["n_held_out"] >= 1
    assert projected["affordable"] is True


def test_training_leaves_no_gradient_on_the_model(gpt2, trained):
    """Gradient has to flow THROUGH the norm and unembedding to reach the
    translator, but accumulating it into the model's weights was a second copy
    of the model in memory for numbers no optimiser here reads."""
    assert not any(p.grad is not None for p in gpt2.model.parameters())
    assert all(p.requires_grad for p in gpt2.model.parameters()), (
        "and requires_grad is restored — this model is the runtime's and is "
        "still loaded afterwards"
    )


def test_the_translator_starts_as_the_plain_lens(gpt2):
    """Identity at initialisation, so an untrained translator IS the plain
    lens. Random init would make step 0 worse than doing nothing."""
    torch = pytest.importorskip("torch")

    info, state = tl.train(
        gpt2.model, gpt2.tokenizer, CORPUS, corpus_label="none", steps=0
    )
    assert torch.allclose(state["A.0"], torch.eye(state["A.0"].shape[0]))
    assert torch.count_nonzero(state["b.0"]) == 0
    # And with no training, the two lenses agree to the last digit.
    for row in info.layers:
        assert row.plain_kl == row.tuned_kl


def test_a_trained_lens_reports_held_out_numbers_per_layer(trained):
    info, _ = trained
    assert len(info.layers) == 12
    assert all(row.plain_kl > 0 for row in info.layers)
    assert info.n_held_out >= 1
    # The early layers are where the plain lens is worst and where a
    # translator has the most to recover.
    assert info.layers[0].plain_kl > info.layers[-1].plain_kl


def test_a_saved_lens_round_trips(gpt2, trained, tmp_path):
    info, state = trained
    info.model_id = "gpt2"
    path = tl.save(info, state, tmp_path / "l.safetensors")
    assert path.with_suffix(".json").is_file(), "readable without torch"

    loaded_info, loaded_state = tl.load(path, model_id="gpt2", dtype=info.dtype)
    assert loaded_info["corpus_sha256"] == info.corpus_sha256
    assert set(loaded_state) == set(state)


def test_reading_a_run_through_the_tuned_lens(gpt2, trained):
    _, state = trained
    list(
        gpt2.generate_stream(
            "The capital of France is", max_new_tokens=3, temperature=0.0
        )
    )
    out = tl.read(gpt2.model, gpt2.tokenizer, gpt2.last_ids, state, top_k=3)
    assert out["kind"] == "tuned"
    assert len(out["layers"]) == 12
    row = out["layers"][0]
    assert len(row["tokens"]) == 3 and len(row["probs"]) == 3
    assert "align these rows to the plain lens by `layer`" in out["align"]


def test_the_plain_reading_is_never_replaced_by_the_tuned_one(gpt2, trained):
    """`layers` is the plain lens on every kind. A caller that ignored `tuned`
    would otherwise get translated rows it never asked for, with no way to
    tell the model from the fit."""
    _, state = trained
    gpt2._tuned = state
    gpt2._tuned_info = {}
    list(
        gpt2.generate_stream(
            "The capital of France is", max_new_tokens=3, temperature=0.0
        )
    )
    plain = gpt2.logit_lens(3, "plain")
    both = gpt2.logit_lens(3, "both")

    assert "tuned" not in plain
    assert both["layers"] == plain["layers"], "the plain rows are identical"
    assert both["tuned"] and both["tuned"] != both["layers"]
    gpt2._tuned, gpt2._tuned_info = {}, {}


def test_asking_for_a_tuned_lens_before_training_one_refuses_offline(gpt2):
    """Nothing is fetched. A downloaded lens would break the offline promise
    the rest of this package keeps."""
    gpt2._tuned = {}
    list(
        gpt2.generate_stream(
            "The capital of France is", max_new_tokens=3, temperature=0.0
        )
    )
    with pytest.raises(Refusal, match="Train one on your own text"):
        gpt2.logit_lens(3, "tuned")


def test_an_unknown_kind_is_refused(gpt2):
    list(
        gpt2.generate_stream(
            "The capital of France is", max_new_tokens=3, temperature=0.0
        )
    )
    with pytest.raises(BadRequest, match="unknown lens kind"):
        gpt2.logit_lens(3, "sharper")


# ------------- the numbers describe what was measured, not what was there


def test_the_sentence_says_how_many_held_out_sequences_were_actually_scored():
    """`hidden_and_target(held_texts[:8])` caps the evaluation at one batch
    while `n_held_out` reported the whole held-out set.

    On a 200-sequence corpus `_split` holds back 50 and the sentence read
    "Held-out KL on 50 sequences the translator never saw" over a measurement
    taken on 8 — and always the same 8, because `_split` sorts before
    striding. That is a silent cap on the one number this module's docstring
    calls the only one that counts.
    """
    said = _info(n_held_out=50, n_held_out_scored=8).means()
    assert "8 of the 50 sequences" in said
    assert "on 50 sequences" not in said


def test_no_cut_is_claimed_when_the_whole_held_out_set_was_scored():
    said = _info(n_held_out=6, n_held_out_scored=6).means()
    assert "on 6 sequences" in said
    assert " of the " not in said


def test_the_cap_is_a_named_constant_rather_than_a_literal():
    """It was `[:8]` inline, which is how it stayed out of the output."""
    import inspect

    source = inspect.getsource(tl.train)
    assert "HELD_OUT_SCORED" in source
    assert "held_texts[:8]" not in source


def test_tokens_are_counted_over_the_text_the_fit_actually_consumed():
    """`n_tokens` tokenized every text untruncated, counting the held-out
    quarter and everything past `max_length`.

    MEASURED on a 200-sequence corpus of 500-token sequences at max_length
    128: 80,699 tokens counted against 19,200 consumed, a 4.2x over-count.
    That ratio feeds `tokens_per_parameter`, which gates `caution` at 1.0 —
    so on a 5,200-sequence corpus the old count reported 3.5569 tokens per
    parameter and stayed silent where the true 0.8453 should have warned.
    """
    import inspect

    source = inspect.getsource(tl.train)
    counted = source[source.index("n_tokens = sum(") :][:300]
    assert "train_texts" in counted, "counted the held-out text as well"
    assert "truncation=True" in counted, "counted tokens past max_length"
    assert "max_length=max_length" in counted


def test_the_caution_fires_on_a_ratio_below_one():
    """The gate itself, so the fix above has something to protect."""
    per_layer = 768 * 768 + 768
    quiet = _info(n_tokens=per_layer * 2)
    loud = _info(n_tokens=per_layer // 2)
    assert quiet.caution == ""
    assert "tokens per parameter" in loud.caution
