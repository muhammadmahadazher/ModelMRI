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


@pytest.mark.parametrize("method", ["caa", "repe"])
def test_identical_sets_have_no_direction_and_say_so(method):
    """PARAMETRIZED OVER BOTH METHODS, and that is the whole test.

    It ran on the default (`caa`) only, and `caa` was never the method with the
    problem. `repe` takes the first principal component of the paired
    DIFFERENCES, and the SVD of an all-zero matrix returns zero singular values
    beside an arbitrary orthonormal V — so `vh[0]` came back a unit vector with
    a norm of exactly 1.0, the `norm == 0.0` guard could not fire, and the
    estimator handed out a basis vector as a fitted direction. Measured on
    twelve identical pairs: `e_0`, and downstream a row reading `effect 0.0,
    p_value 1.0` under the note "this direction does not beat its own
    label-shuffled refits" — a verdict about eight shuffles that had each
    estimated a basis vector too.

    A `NoDirection` specifically, not any refusal: `sweep` tells this apart
    from "not enough pairs" to decide whether one layer or the whole request is
    what has no answer, and a plain `Refusal` here would abort the sweep.
    """
    same = torch.randn(12, D)
    with pytest.raises(sv.NoDirection, match="no direction between them"):
        sv.fit_direction((same, same.clone()), 0, method=method)


@pytest.mark.parametrize("method", ["caa", "repe"])
def test_neither_estimator_invents_a_direction_from_nothing(method):
    """One level below the refusal: `_fit` itself must not return a vector.

    The refusal above is the behaviour; this is the reason it is safe. A
    returned unit vector is not merely a wrong answer, it is one that would
    have been stored — `sweep` keeps `vectors[layer]` for the caller to steer
    with — so what is checked here is that nothing comes back at all.
    """
    same = torch.randn(12, D)
    with pytest.raises(sv.NoDirection):
        sv._fit(same, same.clone(), method)


def test_an_unknown_method_is_a_bad_request():
    with pytest.raises(BadRequest, match="unknown method"):
        sv.fit_direction(separated(), 0, method="banana")


def test_repe_is_sign_aligned_with_caa():
    """PCA has no sign convention; 'positive' must mean the same in both."""
    caa, cv = sv.fit_direction(separated(), 0, method="caa")
    repe, rv = sv.fit_direction(separated(), 0, method="repe")
    assert float(cv @ rv) > 0
    assert caa.effect > 0 and repe.effect > 0


@pytest.mark.parametrize("method", ["caa", "repe"])
def test_the_result_is_reproducible_from_its_seed(method):
    """Parametrized over BOTH methods, which is the whole reason this caught
    nothing for so long: it ran on the default (`caa`) only, and `caa` was
    never the method with a problem."""
    a, _ = sv.fit_direction(separated(), 0, seed=3, method=method)
    b, _ = sv.fit_direction(separated(), 0, seed=3, method=method)
    assert a.effect == b.effect and a.null_max == b.null_max


@pytest.mark.parametrize("method", ["caa", "repe"])
def test_the_seed_is_the_only_randomness_there_is(method):
    """A seed that does not reproduce its own result is not a seed.

    `repe` used `torch.pca_lowrank`, which is a RANDOMISED algorithm taking no
    `generator` — so it drew from the GLOBAL torch RNG, and every `repe`
    direction was a function of whatever had last touched it. Measured over 400
    global states on identical data with identical `seed`: the effect ranged
    0.289 to 0.519, `null_max` 0.777 to 0.982, and the margin deciding
    `beats_null` came within 0.0014 of flipping. `caa` over the same sweep was
    identical to the last digit.

    Perturbing the global RNG between fits is the test the old one was not: a
    result that moves when nothing it was given has moved is not reproducible,
    whatever the receipt says.
    """
    seen = set()
    for global_seed in (0, 1, 7, 99, 12345):
        torch.manual_seed(global_seed)
        judged, vec = sv.fit_direction(structureless(), 5, seed=3, method=method)
        seen.add((judged.beats_null, judged.effect, judged.null_max, float(vec[0])))
    assert len(seen) == 1, (
        f"{method} produced {len(seen)} different results from one seed — the "
        f"only thing that changed between them was the global torch RNG"
    )


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
    "model": "Qwen/Qwen3-1.7B",
    "layer": 6,
    "hidden_size": D,
    "method": "caa",
    "dtype": "bfloat16",
    "beats_null": True,
}


def test_a_saved_direction_round_trips(store):
    _, vec = sv.fit_direction(separated(), 6)
    sv.save("politeness", vec, META)
    back, payload, warnings = sv.load(
        "politeness", hidden_size=D, model="Qwen/Qwen3-1.7B"
    )
    assert torch.allclose(back.float(), vec.float(), atol=1e-6)
    assert payload["layer"] == 6 and payload["method"] == "caa"
    assert warnings == []


def test_saving_without_provenance_is_refused(store):
    _, vec = sv.fit_direction(separated(), 6)
    with pytest.raises(BadRequest, match="hidden_size"):
        sv.save(
            "x",
            vec,
            {
                "model": "Qwen/Qwen3-1.7B",
                "layer": 1,
                "method": "caa",
                "dtype": "f32",
            },
        )


def test_a_wrong_shaped_direction_is_refused_by_name(store):
    _, vec = sv.fit_direction(separated(), 6)
    sv.save("politeness", vec, META)
    with pytest.raises(Refusal, match="Refusing rather than reshaping"):
        sv.load("politeness", hidden_size=D * 2, model="Qwen/Qwen3-1.7B")


def test_a_different_model_warns_loudly_rather_than_blocking(store):
    """Cross-checkpoint transfer is a legitimate experiment when the person
    running it knows that is what they are doing."""
    _, vec = sv.fit_direction(separated(), 6)
    sv.save("politeness", vec, META)
    _, _, warnings = sv.load("politeness", hidden_size=D, model="Qwen/Qwen3-4B")
    assert any("equal size is not equal basis" in w for w in warnings)


def test_loading_a_direction_that_failed_its_null_says_so(store):
    _, vec = sv.fit_direction(structureless(), 6)
    sv.save("nothing", vec, dict(META, beats_null=False))
    _, _, warnings = sv.load("nothing", hidden_size=D, model="Qwen/Qwen3-1.7B")
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
    assert rows[0]["model"] == "Qwen/Qwen3-1.7B" and rows[0]["layer"] == 6


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


# ---------------------------- regressions from the round-2 audit


def test_the_p_value_is_the_textbook_permutation_form():
    """`beats_null` is a boolean and a boolean hides how close the call was —
    1/9 and 9/9 both read as "no"."""
    judged, _ = sv.fit_direction(separated(), 0)
    assert judged.p_value == pytest.approx(1 / (sv.NULL_REFITS + 1), abs=1e-4)
    assert judged.beats_null is True

    empty, _ = sv.fit_direction(structureless(), 0)
    assert empty.p_value > 1 / (sv.NULL_REFITS + 1)
    assert empty.beats_null is False


def test_the_gate_cannot_assert_better_than_one_over_draws_plus_one():
    """With K draws the smallest attainable p-value is 1/(K+1). Publishing a
    smaller one would be claiming a resolution the estimator does not have."""
    judged, _ = sv.fit_direction(separated(gap=50.0), 0)
    # `round(1/9, 4)` is 0.1111, a hair BELOW 1/9 — compare against the value
    # as reported rather than against the unrounded fraction.
    assert judged.p_value >= round(1 / (sv.NULL_REFITS + 1), 4)


def test_the_false_positive_rate_has_not_silently_got_worse():
    """Measured at the time of writing: caa 32/200 = 16.0% on structureless
    data. This pins the order of magnitude so a change to the null, the split
    or the draw count cannot quietly make the gate useless."""
    hits = sum(
        1
        for s in range(120)
        if sv.fit_direction(structureless(seed=s), 0, method="caa")[0].beats_null
    )
    rate = hits / 120
    assert rate < 0.30, f"false-positive rate {rate:.2f} — the gate has degraded"


def test_a_real_direction_is_still_found_after_the_p_value_change():
    found = sum(
        1
        for s in range(30)
        if sv.fit_direction(separated(gap=4.0, seed=s), 0)[0].beats_null
    )
    assert found >= 27, f"only {found}/30"


# ------------------------- the store learns to be read, not only written
#
# `catalogue()` sorted on `saved_at` and promised "newest first"; nothing ever
# wrote that key, so every row sorted equal on an empty string and the order a
# reader saw was whatever `glob` returned. `load()` and `save()` were the only
# doors in and out, so a direction could be created and never removed. And
# `store_dir()` created its directory on being ASKED, which is fine while the
# only caller is a writer and is a rule violation the moment a GET reads it.


def test_a_saved_direction_stamps_when_it_was_saved(store):
    _, vec = sv.fit_direction(separated(), 6)
    sv.save("politeness", vec, META)
    payload = json.loads((store / "politeness.json").read_text(encoding="utf-8"))
    assert payload["saved_at"], "catalogue() sorts on this and promises an order"
    assert payload["saved_at"].startswith("20")


def test_the_catalogue_really_is_newest_first(store):
    _, vec = sv.fit_direction(separated(), 6)
    sv.save("older", vec, dict(META, saved_at="2020-01-01T00:00:00+00:00"))
    sv.save("newer", vec, dict(META, saved_at="2030-01-01T00:00:00+00:00"))
    assert [r["name"] for r in sv.catalogue()] == ["newer", "older"]


def test_a_caller_can_carry_an_original_timestamp_through(store):
    """Re-saving an imported vector should not restamp it as new."""
    _, vec = sv.fit_direction(separated(), 6)
    sv.save("imported", vec, dict(META, saved_at="2021-06-01T12:00:00+00:00"))
    assert sv.catalogue()[0]["saved_at"] == "2021-06-01T12:00:00+00:00"


def test_removing_a_direction_takes_it_out_of_the_catalogue(store):
    _, vec = sv.fit_direction(separated(), 6)
    sv.save("politeness", vec, META)
    out = sv.remove("politeness")
    assert out["removed"] == "politeness"
    assert sv.catalogue() == []
    assert not (store / "politeness.json").exists()


def test_removing_one_that_was_never_there_refuses_by_name(store):
    with pytest.raises(Refusal, match="no saved direction called 'absent'"):
        sv.remove("absent")


def test_a_name_cannot_escape_the_store_on_the_way_out(tmp_path, monkeypatch):
    """Same rule as `save`: the name arrives from a text field, and this one
    unlinks files.

    THE FILE IS REAL, and that is the whole point. Asserting only that a
    Refusal came back proves nothing: `../passwd` names a path that is not in
    the store either way, so the same Refusal is raised whether `_slug` strips
    the separators or does nothing at all. A planted file outside the store is
    the only assertion that can tell those two apart, and `remove` unlinks.
    """
    outside = tmp_path / "passwd.json"
    outside.write_text("not a direction, and not yours to delete", encoding="utf-8")
    inside = tmp_path / "vectors"
    inside.mkdir()
    monkeypatch.setattr(sv, "store_dir", lambda: inside)

    with pytest.raises(Refusal, match="no saved direction"):
        sv.remove("../passwd")
    assert outside.exists(), "the name escaped the store and unlinked a file"
    for separator in ("/", "\\", ".."):
        assert separator not in sv._slug("../../etc/passwd")


def test_every_path_the_store_builds_is_proven_to_be_inside_it(tmp_path, monkeypatch):
    """The guarantee stated where the delete happens, not inferred from `_slug`.

    `_slug` is a whitelist and the test above proves it holds against the
    payload that matters. What it does not do is SAY so anywhere a reader — or
    an analyser — can see: CodeQL read `store_dir() / f"{_slug(name)}.json"`
    followed by `path.unlink()` and reported `py/path-injection` at high
    severity on both lines, correctly, because a generator expression is not a
    sanitiser it can recognise. `_store_path` resolves and checks containment,
    which is the same property in a form that is checkable rather than
    reasoned about.

    Every payload below is one CodeQL's taint path would carry: the route is
    `DELETE /api/steer/directions/{name}`, so `name` is a URL segment.
    """
    inside = tmp_path / "vectors"
    inside.mkdir()
    monkeypatch.setattr(sv, "store_dir", lambda: inside)
    root = inside.resolve()

    for payload in (
        "../../etc/passwd",
        r"..\..\Windows\System32\config\SAM",
        "/etc/shadow",
        "C:/Windows/win.ini",
        "....//....//x",
        "a/../../../b",
        "~/.ssh/id_rsa",
        "%2e%2e%2fetc",
        "x\x00.json",
    ):
        assert sv._store_path(payload).is_relative_to(root), payload

    # And a name with nothing usable in it is refused rather than resolved to
    # some default file, which is the other way a path can go somewhere the
    # caller did not name.
    for empty in ("..", ".", "///", "---"):
        with pytest.raises(BadRequest, match="at least one letter or digit"):
            sv._store_path(empty)


def test_the_containment_check_is_load_bearing_and_not_decoration(
    tmp_path, monkeypatch
):
    """The guard has to fail on its own, or the test above is proving `_slug`.

    Every assertion in `…proven_to_be_inside_it` passes with the containment
    check DELETED, because `_slug` already makes the escape impossible — which
    is exactly what makes that test weak as a check on the new code. The only
    way to see the guard work is to take away the thing that makes it
    redundant. `_slug` is monkeypatched to the identity here, standing in for
    the future edit that loosens it, and the guard is then the one thing
    between a URL segment and `unlink`.
    """
    inside = tmp_path / "vectors"
    inside.mkdir()
    monkeypatch.setattr(sv, "store_dir", lambda: inside)
    monkeypatch.setattr(sv, "_slug", lambda name: name)

    with pytest.raises(BadRequest, match="inside the direction store"):
        sv._store_path("../../escaped")

    # A benign name still resolves under the identity slug, so the guard is
    # refusing the escape rather than refusing everything.
    assert sv._store_path("ordinary").is_relative_to(inside.resolve())


def test_asking_where_the_store_is_does_not_create_it(tmp_path, monkeypatch):
    """`paths.py`: "nothing here creates a directory as a side effect of being
    asked a question". The catalogue is a read, and a read that writes into
    somebody's data folder is the thing that rule forbids."""
    monkeypatch.setenv("MODELMRI_HOME", str(tmp_path))
    assert not sv.store_dir().exists()
    assert sv.catalogue() == []
    assert not sv.store_dir().exists()
    _, vec = sv.fit_direction(separated(), 6)
    sv.save("politeness", vec, META)
    assert sv.store_dir().is_dir(), "writing DOES create it"


def test_the_width_refusal_names_the_model_it_is_being_pushed_into(store):
    """The old sentence said "this model's residual stream is 64" — which end
    is wrong is exactly what a reader cannot work out from that."""
    _, vec = sv.fit_direction(separated(), 6)
    sv.save("politeness", vec, META)
    with pytest.raises(Refusal) as caught:
        sv.load("politeness", hidden_size=D * 2, model="meta-llama/Llama-3.2-1B")
    said = caught.value.sentence
    assert "Qwen/Qwen3-1.7B" in said
    assert "meta-llama/Llama-3.2-1B" in said


def test_the_width_refusal_still_reads_when_no_model_was_named(store):
    _, vec = sv.fit_direction(separated(), 6)
    sv.save("politeness", vec, META)
    with pytest.raises(Refusal, match="this model's residual stream is 64"):
        sv.load("politeness", hidden_size=D * 2)


# ------------------------------------------------- the strength that travels


def test_a_strength_is_reported_against_the_stream_it_is_added_to():
    """The module's own rule: "0.5x the stream's own norm" travels and "5.0"
    does not. One function, because this quotient is published in the status,
    the receipt and the slider's label."""
    assert sv.relative_strength(30.0, 60.0) == pytest.approx(0.5)
    assert sv.relative_strength(-30.0, 60.0) == pytest.approx(-0.5)


def test_an_unmeasured_norm_gives_no_relative_strength_rather_than_zero():
    """0.0 says the push is negligible. That is the one thing an absent
    measurement must never say."""
    assert sv.relative_strength(30.0, None) is None
    assert sv.relative_strength(30.0, 0.0) is None


# ------------------------------------------------- one filename, several names
#
# `_slug` maps every character outside `[A-Za-z0-9_-]` to `-` and truncates to
# 80. That is the right shape for "a name becomes a filename", and it is
# many-to-one: three separate ways for two directions a user thinks of as
# distinct to land on the same file, where `save` wrote straight over whatever
# was already there and returned success.
#
# The tests below are the three ways, plus the case that must keep working
# (re-saving under the same name), plus the one where the store cannot answer
# the question at all.


def _saved_names(store):
    """Every original name the store is actually holding, read off disk."""
    return sorted(
        json.loads(p.read_text(encoding="utf-8"))["name"] for p in store.glob("*.json")
    )


def test_two_names_that_punctuate_differently_do_not_share_one_file(store):
    """`"sycophancy v2"` and `"sycophancy-v2"` both slug to `sycophancy-v2`."""
    _, vec = sv.fit_direction(separated(), 6)
    sv.save("sycophancy v2", vec, META)

    with pytest.raises(Refusal, match="sycophancy v2"):
        sv.save("sycophancy-v2", vec, META)

    # The first one is still there, unedited, and still answers to its own name.
    assert _saved_names(store) == ["sycophancy v2"]
    _, payload, _ = sv.load("sycophancy v2", hidden_size=D, model="Qwen/Qwen3-1.7B")
    assert payload["name"] == "sycophancy v2"


def test_a_case_variant_does_not_overwrite_the_direction_it_shadows(store):
    """`Sycophancy` and `sycophancy` are two files on Linux and one on Windows
    and macOS, and a store people copy between machines cannot mean two
    different things depending on where it is sitting."""
    _, vec = sv.fit_direction(separated(), 6)
    sv.save("Sycophancy", vec, META)

    with pytest.raises(Refusal, match="Sycophancy"):
        sv.save("sycophancy", vec, META)

    assert _saved_names(store) == ["Sycophancy"]


def test_two_long_names_sharing_a_prefix_do_not_collapse_into_one(store):
    """The truncation at 80 characters is the third way in: two names that
    differ only after the cut are one filename."""
    stem = (
        "refusal-behaviour-under-a-very-long-and-carefully-described-experimental-condi"
    )
    first, second = f"{stem}tion-alpha", f"{stem}tion-beta"
    assert sv._slug(first) == sv._slug(second)  # the premise of the test

    _, vec = sv.fit_direction(separated(), 6)
    sv.save(first, vec, META)

    with pytest.raises(Refusal, match="alpha"):
        sv.save(second, vec, META)

    assert _saved_names(store) == [first]


def test_re_saving_the_same_name_replaces_it_and_says_so(store):
    """The collision check must not make a direction unfixable. Saving over
    your own name is the ordinary way to correct one, and it stays allowed —
    it just stops being silent about what it did."""
    _, first = sv.fit_direction(separated(seed=0), 6)
    out = sv.save("politeness", first, META)
    assert out["replaced"] is False

    _, second = sv.fit_direction(separated(seed=1), 6)
    again = sv.save("politeness", second, META)
    assert again["replaced"] is True

    back, _, _ = sv.load("politeness", hidden_size=D, model="Qwen/Qwen3-1.7B")
    assert torch.allclose(back.float(), second.float(), atol=1e-6)
    assert _saved_names(store) == ["politeness"]


def test_a_stored_file_whose_name_cannot_be_read_is_not_overwritten(store):
    """`catalogue()` already treats an unreadable file as damaged rather than
    dropping it, so damaged files exist. One sitting on the slug we are about
    to write cannot be asked whose it is — and a direction is somebody's
    measurement, so the answer is to say so, not to guess and overwrite."""
    (store / "politeness.json").write_text("{not json", encoding="utf-8")

    _, vec = sv.fit_direction(separated(), 6)
    with pytest.raises(Refusal, match="politeness.json"):
        sv.save("politeness", vec, META)

    assert (store / "politeness.json").read_text(encoding="utf-8") == "{not json"


@pytest.mark.parametrize(
    "content",
    [
        "{not json",  # not JSON at all
        "[]",  # valid JSON, but not an object, so it has no .get
        '"politeness"',  # valid JSON, a bare string
        "{}",  # an object that never says what it is
        '{"name": 7}',  # a name that is not a name
    ],
    ids=["broken", "array", "string", "no-name", "name-not-a-string"],
)
def test_a_file_that_cannot_name_itself_is_refused_not_overwritten(store, content):
    """Every shape that survives the read but cannot answer "whose are you?".

    Only the first of these raises inside `json.loads`. The rest do not, and an
    earlier draft called `.get` on whatever came back — an AttributeError on a
    list, a TypeError on a string. A crash where a refusal belongs is still a
    direction lost, only noisier.
    """
    (store / "politeness.json").write_text(content, encoding="utf-8")

    _, vec = sv.fit_direction(separated(), 6)
    with pytest.raises(Refusal, match="politeness.json"):
        sv.save("politeness", vec, META)

    assert (store / "politeness.json").read_text(encoding="utf-8") == content
