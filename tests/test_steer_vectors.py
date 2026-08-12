"""A difference of means always returns a direction. That is the problem.

Feed this two arbitrary sets of sentences and it produces a vector with a norm,
a layer and a confident sweep — and adding any large vector to a residual
stream changes the output. Nothing about the result looks different when there
was no signal to find.

So the tests that matter here are the ones where there IS no signal: random
labels over structureless activations must come back `beats_null=False` with a
sentence saying the separation is what the estimator produces regardless of
labels. The rest is arithmetic — held-out scoring, matched pairs, and a store
that refuses a direction which cannot belong to the model it is being loaded
onto.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from modelmri import steer_vectors as sv  # noqa: E402
from modelmri.errors import BadRequest, Refusal  # noqa: E402

D = 32


def separated(n=24, gap=6.0, seed=0):
    """Two clouds pushed apart along one axis — a direction really is there."""
    g = torch.Generator().manual_seed(seed)
    axis = torch.zeros(D)
    axis[3] = gap
    pos = torch.randn(n, D, generator=g) + axis
    neg = torch.randn(n, D, generator=g) - axis
    return pos, neg


def structureless(n=24, seed=1):
    """One cloud, arbitrarily labelled. There is nothing to find."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, D, generator=g), torch.randn(n, D, generator=g)


# ------------------------------------------------- the null does its job


@pytest.mark.parametrize("method", ["caa", "repe"])
def test_a_real_direction_beats_its_shuffled_null(method):
    judged, vec = sv.fit_direction(separated(), 5, method=method)
    assert judged.beats_null is True
    assert abs(judged.effect) > judged.null_max
    assert vec.shape == (D,)
    assert float(vec.norm()) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("method", ["caa", "repe"])
def test_labels_over_noise_are_reported_as_no_result(method):
    """The whole point. A direction is always returned; this says it is empty."""
    judged, _ = sv.fit_direction(structureless(), 5, method=method)
    assert judged.beats_null is False
    assert any("does not beat its own label-shuffled" in n for n in judged.notes)
    assert any("regardless of labels" in n for n in judged.notes)


def test_the_null_is_reported_even_when_the_direction_wins():
    """A reader must be able to see what the null was, not just the verdict."""
    judged, _ = sv.fit_direction(separated(), 0)
    assert judged.null_mean >= 0.0
    assert judged.null_max >= judged.null_mean


def test_a_bigger_gap_gives_a_bigger_effect():
    weak, _ = sv.fit_direction(separated(gap=1.0), 0)
    strong, _ = sv.fit_direction(separated(gap=8.0), 0)
    assert abs(strong.effect) > abs(weak.effect)


# --------------------------------------------------------- honest scoring


def test_scoring_is_on_held_out_pairs():
    """A direction scored on its own fitting set separates it by construction."""
    judged, _ = sv.fit_direction(separated(n=24), 0)
    assert judged.n_fit == 12
    assert judged.n_score == 12
    assert judged.n_fit + judged.n_score == judged.n_pairs


def test_too_few_pairs_is_refused_with_the_reason():
    with pytest.raises(Refusal, match="at least 8"):
        sv.fit_direction(separated(n=4), 0)


def test_unmatched_sets_are_refused():
    pos, neg = separated(n=12)
    with pytest.raises(BadRequest, match="must be matched"):
        sv.fit_direction((pos, neg[:8]), 0)


def test_identical_sets_have_no_direction_and_say_so():
    same = torch.randn(12, D)
    with pytest.raises(Refusal, match="no direction between them"):
        sv.fit_direction((same, same.clone()), 0)


def test_an_unknown_method_is_a_bad_request():
    with pytest.raises(BadRequest, match="unknown method"):
        sv.fit_direction(separated(), 0, method="banana")


def test_repe_is_sign_aligned_with_caa():
    """PCA has no sign convention; 'positive' must mean the same in both."""
    caa, cv = sv.fit_direction(separated(), 0, method="caa")
    repe, rv = sv.fit_direction(separated(), 0, method="repe")
    assert float(cv @ rv) > 0
    assert caa.effect > 0 and repe.effect > 0


def test_the_result_is_reproducible_from_its_seed():
    a, _ = sv.fit_direction(separated(), 0, seed=3)
    b, _ = sv.fit_direction(separated(), 0, seed=3)
    assert a.effect == b.effect and a.null_max == b.null_max


def test_residual_norm_is_reported_so_a_coefficient_can_mean_something():
    """A scale of 5 is meaningless across models and layers; relative to the
    stream's own norm it travels."""
    judged, _ = sv.fit_direction(separated(), 0)
    assert judged.residual_norm > 0


# ----------------------------------------------------------------- the store


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(sv, "store_dir", lambda: tmp_path)
    return tmp_path


META = {
    "model": "gpt2",
    "layer": 6,
    "hidden_size": D,
    "method": "caa",
    "dtype": "bfloat16",
    "beats_null": True,
}


def test_a_saved_direction_round_trips(store):
    _, vec = sv.fit_direction(separated(), 6)
    sv.save("politeness", vec, META)
    back, payload, warnings = sv.load("politeness", hidden_size=D, model="gpt2")
    assert torch.allclose(back.float(), vec.float(), atol=1e-6)
    assert payload["layer"] == 6 and payload["method"] == "caa"
    assert warnings == []


def test_saving_without_provenance_is_refused(store):
    _, vec = sv.fit_direction(separated(), 6)
    with pytest.raises(BadRequest, match="hidden_size"):
        sv.save("x", vec, {"model": "gpt2", "layer": 1, "method": "caa", "dtype": "f32"})


def test_a_wrong_shaped_direction_is_refused_by_name(store):
    _, vec = sv.fit_direction(separated(), 6)
    sv.save("politeness", vec, META)
    with pytest.raises(Refusal, match="Refusing rather than reshaping"):
        sv.load("politeness", hidden_size=D * 2, model="gpt2")


def test_a_different_model_warns_loudly_rather_than_blocking(store):
    """Cross-checkpoint transfer is a legitimate experiment when the person
    running it knows that is what they are doing."""
    _, vec = sv.fit_direction(separated(), 6)
    sv.save("politeness", vec, META)
    _, _, warnings = sv.load("politeness", hidden_size=D, model="gpt2-medium")
    assert any("equal size is not equal basis" in w for w in warnings)


def test_loading_a_direction_that_failed_its_null_says_so(store):
    _, vec = sv.fit_direction(structureless(), 6)
    sv.save("nothing", vec, dict(META, beats_null=False))
    _, _, warnings = sv.load("nothing", hidden_size=D, model="gpt2")
    assert any("never evidence of anything" in w for w in warnings)


def test_a_missing_direction_refuses_in_words(store):
    with pytest.raises(Refusal, match="no saved direction"):
        sv.load("absent", hidden_size=D)


def test_a_name_cannot_escape_the_store(store):
    _, vec = sv.fit_direction(separated(), 6)
    out = sv.save("../../etc/passwd", vec, META)
    assert ".." not in out["path"]
    assert out["path"].startswith(str(store))


def test_an_empty_name_is_refused(store):
    _, vec = sv.fit_direction(separated(), 6)
    with pytest.raises(BadRequest, match="at least one letter"):
        sv.save("///", vec, META)


def test_the_catalogue_omits_values_but_keeps_provenance(store):
    _, vec = sv.fit_direction(separated(), 6)
    sv.save("politeness", vec, META)
    rows = sv.catalogue()
    assert len(rows) == 1
    assert "values" not in rows[0]
    assert rows[0]["model"] == "gpt2" and rows[0]["layer"] == 6


def test_an_unreadable_file_is_listed_as_damaged_not_dropped(store):
    """A vector silently missing from its own catalogue is worse than one that
    says it is damaged."""
    (store / "broken.json").write_text("{not json", encoding="utf-8")
    rows = sv.catalogue()
    assert rows and rows[0]["unreadable"] is True


def test_saved_files_are_plain_json(store):
    _, vec = sv.fit_direction(separated(), 6)
    path = store / "politeness.json"
    sv.save("politeness", vec, META)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload["values"]) == D
