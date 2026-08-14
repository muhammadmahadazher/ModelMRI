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
