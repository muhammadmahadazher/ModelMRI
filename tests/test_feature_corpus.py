"""A top activation is a top activation IN THIS CORPUS.

The dashboards this competes with show features from a model and an SAE
somebody else chose, on text somebody else picked. The numbers here are about
the text you handed it — so most of these tests are about the corpus staying
attached to the claim, and about what the module refuses to say.
"""

from __future__ import annotations

import json
import os

import pytest

from modelmri import feature_corpus as fc
from modelmri.errors import BadRequest, Refusal

CORPUS = [
    "The lawyer cited Brown v. Board of Education in her brief.",
    "The court ruled in Roe v. Wade that the statute was invalid.",
    "A dog ran quickly along the wet path beside the canal.",
    "Coffee grown at high altitude tastes brighter and more acidic.",
    "The train leaves the station every twenty minutes on weekdays.",
    "In Miranda v. Arizona the Supreme Court held otherwise.",
    "The weather turned cold and the rain continued until evening.",
    "She opened the book and read the first chapter twice over.",
]


# ------------------------------------------------------------ the corpus


def test_a_corpus_loads_from_the_same_reader_the_sweep_uses(tmp_path):
    """One format across the three features that take a corpus, not three
    nearly-identical ones."""
    path = tmp_path / "c.txt"
    path.write_text("first line\nsecond line\n", encoding="utf-8")
    texts, label = fc.load_corpus(path)
    assert texts == ["first line", "second line"]
    assert label == "c.txt"


def test_a_missing_corpus_says_so(tmp_path):
    with pytest.raises(BadRequest, match="could not be read"):
        fc.load_corpus(tmp_path / "nope.txt")


def test_the_same_corpus_hashes_the_same_way_in_any_order():
    assert fc.corpus_hash(CORPUS) == fc.corpus_hash(list(reversed(CORPUS)))


# ------------------------------------------------- what the stats refuse to say


def _stats(**over) -> fc.CorpusStats:
    kw = dict(
        corpus_label="notes.txt",
        corpus_sha256="abc",
        n_sequences=8,
        n_tokens=98,
        n_features=24576,
        n_never_fired=21801,
        layer=8,
    )
    kw.update(over)
    return fc.CorpusStats(**kw)


def test_a_feature_that_never_fired_is_not_seen_here_never_dead():
    """Dead means the feature does nothing. Not seen means you did not show it
    anything it responds to, and only one of those is about the model."""
    means = _stats().means()
    assert "NOT SEEN IN THIS CORPUS, not dead" in means
    assert "dead" in means


def test_the_corpus_and_its_size_travel_with_every_number():
    means = _stats().means()
    assert "notes.txt" in means
    assert "98 tokens" in means
    assert "top activation IN THIS TEXT" in means


def test_the_never_fired_share_is_reported():
    stats = _stats()
    assert stats.never_fired_share == pytest.approx(21801 / 24576)
    assert "88.7%" in stats.means()


def test_a_truncated_sweep_says_what_it_did_not_read():
    """Silence about a cut reads as having read everything."""
    means = _stats(truncated=True).means()
    assert "was cut at" in means and "not read" in means


def test_the_stats_survive_json():
    out = json.loads(json.dumps(_stats().to_dict(), allow_nan=False))
    assert out["n_never_fired"] == 21801
    assert "means" in out


# ------------------------------------------------------ against a real SAE


@pytest.fixture(scope="module")
def sae_runtime():
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
        runtime.load_sae(
            "jbloom/GPT2-Small-SAEs-Reformatted", "blocks.8.hook_resid_pre"
        )
    except Exception as err:
        pytest.skip(f"the gpt2 SAE is not available here: {err}")
    yield runtime
    runtime.unload()


@pytest.fixture(scope="module")
def swept(sae_runtime):
    block = sae_runtime._block(8)
    return fc.sweep(
        sae_runtime.model,
        block,
        sae_runtime.tokenizer,
        sae_runtime.sae,
        CORPUS,
        device=sae_runtime.device,
        layer=8,
        corpus_label="test corpus",
    )


def test_a_sweep_counts_firing_without_holding_every_activation(swept):
    """[n_tokens, d_sae] for a 24,576-feature SAE is why this accumulates
    per-feature statistics and keeps no history."""
    stats, per_feature = swept
    assert stats.n_features == 24576
    assert stats.n_tokens > 0
    assert per_feature, "some features fired"
    assert len(per_feature) < stats.n_features, "and most did not"
    assert stats.n_never_fired == stats.n_features - len(per_feature)


def test_the_evidence_shows_spans_with_context(sae_runtime, swept):
    _, per_feature = swept
    busiest = max(per_feature.items(), key=lambda kv: kv[1][0])[0]
    out = fc.evidence(
        sae_runtime.model,
        sae_runtime._block(8),
        sae_runtime.tokenizer,
        sae_runtime.sae,
        CORPUS,
        busiest,
        device=sae_runtime.device,
        top_k=3,
    )
    assert len(out["spans"]) == 3
    assert out["spans"][0]["activation"] >= out["spans"][-1]["activation"]
    assert out["spans"][0]["token"] in out["spans"][0]["text"]
    assert out["histogram"] and len(out["histogram"]) == fc.HISTOGRAM_BINS


def test_every_span_says_where_in_it_the_feature_fired(sae_runtime, swept):
    """The offset, not the first match.

    A span is a window either side of the firing position, so the same word
    can appear in it twice. MEASURED on gpt2 layer 8 with a corpus of court
    sentences: " appeals court disagreed with the trial court's reading."
    fired at the SECOND "court", at character 39 -- `text.index(token)` would
    have pointed at character 8, and highlighting every match would have
    claimed two firings where there was one.
    """
    _, per_feature = swept
    for feature in list(per_feature)[:20]:
        out = fc.evidence(
            sae_runtime.model,
            sae_runtime._block(8),
            sae_runtime.tokenizer,
            sae_runtime.sae,
            CORPUS,
            feature,
            device=sae_runtime.device,
        )
        for span in out["spans"]:
            at = span["offset"]
            assert span["text"][at : at + len(span["token"])] == span["token"], (
                "the offset must land exactly on the token that fired"
            )


def test_the_offset_picks_the_right_one_when_a_word_repeats(sae_runtime):
    """The case the first-match version got wrong, made deliberately."""
    corpus = ["The appeals court disagreed with the trial court's reading."]
    stats, per_feature = fc.sweep(
        sae_runtime.model,
        sae_runtime._block(8),
        sae_runtime.tokenizer,
        sae_runtime.sae,
        corpus,
        device=sae_runtime.device,
        layer=8,
        corpus_label="repeat",
    )
    assert stats.n_tokens > 0
    repeated = 0
    for feature in per_feature:
        out = fc.evidence(
            sae_runtime.model,
            sae_runtime._block(8),
            sae_runtime.tokenizer,
            sae_runtime.sae,
            corpus,
            feature,
            device=sae_runtime.device,
        )
        for span in out["spans"]:
            if span["text"].count(span["token"]) < 2:
                continue
            repeated += 1
            at = span["offset"]
            assert span["text"][at : at + len(span["token"])] == span["token"]
    assert repeated, "this corpus repeats ' court', so some span must contain it twice"


def test_no_natural_language_label_is_produced(sae_runtime, swept):
    """Naming the concept is the reader's job. A generated label would be the
    one thing on the page nothing measured."""
    _, per_feature = swept
    feature = next(iter(per_feature))
    out = fc.evidence(
        sae_runtime.model,
        sae_runtime._block(8),
        sae_runtime.tokenizer,
        sae_runtime.sae,
        CORPUS,
        feature,
        device=sae_runtime.device,
    )
    assert out["label"] is None


def test_a_feature_firing_on_most_tokens_is_flagged_as_unselective(sae_runtime, swept):
    """MEASURED on gpt2 layer 8: the most frequently firing feature fired on
    68% of tokens and promoted an unrelated scatter of vocabulary. That is not
    a concept, and reading its top spans as one would be the mistake."""
    _, per_feature = swept
    busiest = max(per_feature.items(), key=lambda kv: kv[1][0])[0]
    out = fc.evidence(
        sae_runtime.model,
        sae_runtime._block(8),
        sae_runtime.tokenizer,
        sae_runtime.sae,
        CORPUS,
        busiest,
        device=sae_runtime.device,
    )
    if out["firing_rate"] >= 0.2:
        assert out["selective"] is False
        assert "NOT A CONCEPT" in out["means"]


def test_a_feature_that_never_fired_reports_not_seen(sae_runtime, swept):
    stats, per_feature = swept
    quiet = next(i for i in range(stats.n_features) if i not in per_feature)
    out = fc.evidence(
        sae_runtime.model,
        sae_runtime._block(8),
        sae_runtime.tokenizer,
        sae_runtime.sae,
        CORPUS,
        quiet,
        device=sae_runtime.device,
    )
    assert out["spans"] == []
    assert out["n_fired"] == 0
    assert "NOT SEEN IN THIS CORPUS, not dead" in out["means"]


# --------------------------------------------------- the exact half


def test_the_logit_weights_need_no_corpus_and_are_exact(sae_runtime):
    """Pure weight math. No sampling, nothing to be a sample of — and the
    same every time it runs."""
    first = fc.logit_weights(
        sae_runtime.model, sae_runtime.tokenizer, sae_runtime.sae, 7451, top_k=5
    )
    again = fc.logit_weights(
        sae_runtime.model, sae_runtime.tokenizer, sae_runtime.sae, 7451, top_k=5
    )
    assert first["promotes"] == again["promotes"]
    assert first["exact"] is True
    assert len(first["promotes"]) == 5 and len(first["suppresses"]) == 5


def test_the_logit_weights_are_relative_not_absolute(sae_runtime):
    """The norm's real scale depends on the stream this direction would be
    added to, so these rank tokens rather than predict logit amounts."""
    out = fc.logit_weights(
        sae_runtime.model, sae_runtime.tokenizer, sae_runtime.sae, 100
    )
    assert "rank tokens rather than predict logit amounts" in out["means"]
    assert "NO CORPUS AND NO SAMPLING" in out["means"]


def test_promoted_and_suppressed_are_opposite_ends(sae_runtime):
    out = fc.logit_weights(
        sae_runtime.model, sae_runtime.tokenizer, sae_runtime.sae, 100, top_k=3
    )
    assert out["promotes"][0]["logit"] > out["suppresses"][0]["logit"]


def test_a_feature_outside_the_sae_is_refused(sae_runtime):
    with pytest.raises(BadRequest, match="outside this SAE"):
        fc.logit_weights(
            sae_runtime.model, sae_runtime.tokenizer, sae_runtime.sae, 999_999
        )


def test_no_sae_is_a_refusal_not_an_empty_dashboard(sae_runtime):
    with pytest.raises(Refusal, match="No SAE loaded"):
        fc.logit_weights(sae_runtime.model, sae_runtime.tokenizer, None, 0)


# ------------------------------------------------------------ persistence


def test_a_sweep_is_findable_after_the_process_ends(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELMRI_HOME", str(tmp_path))
    stats = _stats(corpus_sha256="deadbeef")
    written = fc.save(stats, {7: (12, 3.5), 9: (4, 1.25)}, model="gpt2", sae="repo")
    assert written == 2

    rows = fc.stored("deadbeef", model="gpt2", sae="repo", layer=8)
    assert rows[0] == {"feature": 7, "n_fired": 12, "max_activation": 3.5}


def test_re_sweeping_the_same_corpus_replaces_rather_than_duplicates(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("MODELMRI_HOME", str(tmp_path))
    stats = _stats(corpus_sha256="same")
    fc.save(stats, {7: (12, 3.5)}, model="gpt2", sae="repo")
    fc.save(stats, {7: (99, 9.9)}, model="gpt2", sae="repo")
    rows = fc.stored("same", model="gpt2", sae="repo", layer=8)
    assert len(rows) == 1 and rows[0]["n_fired"] == 99


# ---------------------------------------------------------- through the API


def test_the_route_returns_the_corpus_beside_the_features(
    sae_runtime, tmp_path, monkeypatch
):
    monkeypatch.setenv("MODELMRI_HOME", str(tmp_path))
    out = sae_runtime.feature_evidence(CORPUS, corpus_label="test corpus")
    assert out["corpus"]["n_tokens"] > 0
    assert out["corpus"]["n_never_fired"] > 0
    assert out["top_by_firing_rate"]
    assert out["receipt"]["op"] == "feature_evidence"
    assert out["receipt"]["request"]["sae_repo"]


def test_asking_for_one_feature_returns_all_three_readouts(
    sae_runtime, tmp_path, monkeypatch
):
    """What it fires on, what it promotes — and the pointer to what removing
    it does, which `feature_ablate` already measures."""
    monkeypatch.setenv("MODELMRI_HOME", str(tmp_path))
    out = sae_runtime.feature_evidence(CORPUS, feature_id=7451, corpus_label="c")
    assert out["evidence"]["feature"] == 7451
    assert out["logit_weights"]["feature"] == 7451
    assert out["logit_weights"]["exact"] is True


# ------------------------------ the SAE's own side of the block


def _rig(point: str):
    """A block whose output is unmistakably not its input, so the two sides
    cannot be confused for one another by coincidence."""
    import torch
    import torch.nn as nn

    class Block(nn.Module):
        def forward(self, x):
            return x * 3.0 + 1.0

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = Block()

        def forward(self, ids):
            return self.block(torch.ones(1, ids.shape[-1], 4))

    class Tok:
        def __call__(self, text, return_tensors=None):
            return {"input_ids": torch.tensor([[1, 2, 3]])}

        def decode(self, t):
            return "x"

    class SAE:
        def __init__(self):
            self.point = point

        def encode(self, x):
            # Identity, so the test reads the tensor the sweep actually fed it.
            return x

    model = Model()
    return model, model.block, Tok(), SAE()


def test_the_corpus_sweep_reads_the_hook_point_the_sae_was_trained_at():
    """It called `patch._capture` — a forward PRE hook, the block's INPUT —
    for every SAE regardless of `point`.

    `saes.py` records this same bug being found and fixed, and the fix reached
    `runtime.py` and `feature_ablate.py` and not this module. A resid_post SAE
    was encoded from the block's input with no error and no warning, so the
    never-fired count, the firing-rate table, the evidence spans and histogram
    and the rows written to `feature_activation` all described activations the
    SAE was never trained on.
    """
    pytest.importorskip("torch")

    seen = {}
    for point in ("resid_pre", "resid_post"):
        model, block, tok, sae = _rig(point)
        rows = list(fc._activations(model, block, tok, sae, ["hello"], "cpu"))
        seen[point] = rows[0][2][0].tolist()

    # The block computes x*3+1 over a tensor of ones.
    assert seen["resid_pre"] == [1.0, 1.0, 1.0, 1.0]
    assert seen["resid_post"] == [4.0, 4.0, 4.0, 4.0]
    assert seen["resid_pre"] != seen["resid_post"], (
        "both points captured the same tensor, so one of them is the wrong "
        "side of the block"
    )


def test_no_sae_capture_site_ignores_the_hook_point():
    """The structural version, because this bug has now been fixed twice in
    two different modules and reintroduced by a third.

    Anything that installs a capture for an SAE must route through
    `feature_ablate._register_capture`, which branches on the point, rather
    than `patch._capture`, which is unconditionally resid_pre.
    """
    import inspect

    from modelmri import feature_corpus

    source = inspect.getsource(feature_corpus._activations)
    code = "\n".join(
        line.split("#", 1)[0] for line in source.splitlines() if line.strip()
    )
    assert "_register_capture" in code
    assert "sae.point" in code
    assert "patch import _capture" not in code


def test_the_evidence_pass_stops_where_the_sweep_stops(sae_runtime, monkeypatch):
    """`sweep` cut at MAX_TOKENS and `evidence` walked the whole corpus, so
    one response reported two different sizes for "this corpus" — and the
    firing rate shown for a feature was over a different denominator from the
    rates it sits beside in the sweep's table. Two numbers that look
    comparable and are not.
    """
    monkeypatch.setattr(fc, "MAX_TOKENS", 4)
    block = sae_runtime._block(8)

    stats, _ = fc.sweep(
        sae_runtime.model,
        block,
        sae_runtime.tokenizer,
        sae_runtime.sae,
        CORPUS,
        device=sae_runtime.device,
        layer=8,
    )
    shown = fc.evidence(
        sae_runtime.model,
        block,
        sae_runtime.tokenizer,
        sae_runtime.sae,
        CORPUS,
        0,
        device=sae_runtime.device,
    )

    assert stats.n_tokens == shown["n_tokens"], "the two passes disagree on size"
    assert shown["truncated"] is True
    # Either branch of `means` must disclose the cut. The never-fired one
    # matters most: "this text never showed it anything" is a claim about the
    # WHOLE corpus, and an unread tail is exactly where the missing
    # activation would be.
    said = shown["means"]
    assert ("cut at" in said) or ("READ ONLY TO" in said), said


def test_an_uncapped_corpus_claims_no_cut(sae_runtime, monkeypatch):
    monkeypatch.setattr(fc, "MAX_TOKENS", 1_000_000)
    block = sae_runtime._block(8)
    shown = fc.evidence(
        sae_runtime.model,
        block,
        sae_runtime.tokenizer,
        sae_runtime.sae,
        CORPUS,
        0,
        device=sae_runtime.device,
    )
    assert shown["truncated"] is False
    assert "cut at" not in shown["means"]
