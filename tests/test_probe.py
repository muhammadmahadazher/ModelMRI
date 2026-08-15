"""A probe curve that goes up is easy. Knowing when it means nothing is not.

The value of this module is the two references it draws behind the curve, so
almost every test here is about a case where the curve looks good and the
answer is still "no". A probe module that could not report "nothing is
readable anywhere" would not be worth having.
"""

from __future__ import annotations

import json
import os

import pytest

from modelmri import probe
from modelmri.errors import BadRequest, Refusal

torch = pytest.importorskip("torch")


def _states(
    n_examples: int,
    d: int,
    layers: int,
    *,
    signal_from: int | None = None,
    labels=None,
    strength: float = 4.0,
):
    """Random residual streams, optionally with a planted linear signal."""
    torch.manual_seed(0)
    direction = torch.randn(d)
    out = {}
    for layer in range(layers):
        x = torch.randn(n_examples, d)
        if signal_from is not None and layer >= signal_from:
            x = x + (labels.float().unsqueeze(1) - 0.5) * strength * direction
        out[layer] = x
    return out


def _balanced(n: int):
    return torch.cat([torch.zeros(n // 2), torch.ones(n - n // 2)]).long()


# --------------------------------------------------- the minimums are enforced


def test_too_few_examples_per_class_is_refused_not_warned_about():
    """A linear fit on four examples of a 768-dimensional stream separates
    them perfectly and means nothing."""
    labels = _balanced(10)
    states = _states(10, 32, 2)
    with pytest.raises(BadRequest, match="at least"):
        probe.sweep(states, labels)


def test_a_held_out_set_too_small_to_have_resolution_is_refused():
    """Accuracy on n examples has a resolution of 100/n percentage points."""
    labels = _balanced(24)  # 8 per class survives MIN_PER_CLASS, test is small
    states = _states(24, 32, 2)
    with pytest.raises(BadRequest, match="resolution"):
        probe.sweep(states, labels)


def test_three_classes_are_refused_rather_than_collapsed():
    labels = torch.tensor([0, 1, 2] * 12)
    states = _states(36, 32, 2)
    with pytest.raises(BadRequest, match="two classes"):
        probe.sweep(states, labels)


# ------------------------------------------------------------ the split


def test_the_split_is_stratified():
    """An unstratified split of an imbalanced set can put every example of the
    rarer class on one side, and the probe then scores the majority rate while
    looking like it was tested."""
    labels = torch.cat([torch.zeros(40), torch.ones(20)]).long()
    train, test = probe._split(torch.randn(60, 8), labels, seed=0)
    assert (labels[test] == 0).sum() > 0
    assert (labels[test] == 1).sum() > 0
    assert set(train.tolist()) & set(test.tolist()) == set()


def test_the_split_is_deterministic():
    labels = _balanced(60)
    a = probe._split(torch.randn(60, 8), labels, seed=0)
    b = probe._split(torch.randn(60, 8), labels, seed=0)
    assert a[0].tolist() == b[0].tolist()
    assert a[1].tolist() == b[1].tolist()


# ------------------------------------------------- the null does its job


def test_noise_reads_as_nothing_readable_anywhere():
    """The result this module exists to be able to report."""
    torch.manual_seed(1)
    labels = _balanced(80)
    states = {layer: torch.randn(80, 64) for layer in range(5)}
    report = probe.sweep(states, labels[torch.randperm(80)], n_permutations=12)

    # NOT "exactly zero". The band is a 95th percentile and the sweep asks
    # every layer, so chance alone clears about 0.05 per layer -- this test
    # asserting zero is what surfaced that the module was not accounting for
    # the multiple comparison at all.
    assert len(report.readable) <= report.expected_false_positives + 1, (
        "noise should clear no more often than chance predicts"
    )
    if report.readable:
        assert "read this as noise unless it repeats" in report.means()
    else:
        assert "NO LAYER read this concept better than noise" in report.means()


def test_a_planted_signal_is_found_at_the_layer_it_was_planted():
    labels = _balanced(80)
    states = _states(80, 64, 6, signal_from=3, labels=labels)
    report = probe.sweep(states, labels, n_permutations=12)

    readable = {row.layer for row in report.readable}
    assert readable >= {3, 4, 5}, "the layers carrying the signal"
    # A layer without the signal can still clear by chance -- 0.05 per layer,
    # and this sweep asks six. What must hold is that the signal layers are
    # found and that they outnumber what chance explains.
    assert len(readable) > report.expected_false_positives + 1
    for row in report.layers:
        if row.layer >= 3:
            assert row.accuracy > report.majority


def test_a_layer_inside_the_null_is_reported_as_inside_it():
    labels = _balanced(80)
    states = _states(80, 64, 4, signal_from=3, labels=labels)
    report = probe.sweep(states, labels, n_permutations=12)

    early = next(row for row in report.layers if row.layer == 0)
    assert early.inside_null
    assert early.accuracy <= early.null_high


def test_the_majority_class_line_is_measured_on_the_held_out_set():
    """The probe and the majority rate must be scored on the same examples or
    they are not comparable."""
    labels = torch.cat([torch.zeros(60), torch.ones(20)]).long()
    states = _states(80, 32, 2)
    report = probe.sweep(states, labels, n_permutations=6)
    assert 0.5 < report.majority < 1.0
    assert report.n_test >= probe.MIN_TEST


def test_a_saturated_null_is_a_third_state_not_a_verdict():
    """When a shuffled fit reaches the top of the scale, NO accuracy could
    have cleared it. Measured: at six held-out examples the null hit 1.00 at
    five of gpt2's twelve layers, so READABLE was decided by which shuffles
    happened to fit."""
    labels = _balanced(80)
    states = _states(80, 512, 3, signal_from=0, labels=labels, strength=0.0)
    report = probe.sweep(states, labels, n_permutations=8)
    for row in report.layers:
        if row.null_high >= 1.0:
            assert row.null_saturated
    if report.underpowered:
        assert "THE NULL SATURATED" in report.means()


# ------------------------------------------------------ what it never claims


def test_the_sentence_says_readable_is_not_used():
    """A direction can be linearly present and play no part in the answer.
    The ablation follow-up is the only thing that upgrades the claim."""
    labels = _balanced(80)
    states = _states(80, 64, 3, signal_from=0, labels=labels)
    means = probe.sweep(states, labels, n_permutations=6).means()
    assert "READABLE IS NOT USED" in means
    assert "not whether the model reads it" in means


def test_the_sweep_reports_its_own_false_positive_rate():
    """Sweeping every layer against a 95th-percentile band is a multiple
    comparison, and one readable layer out of twelve is roughly what noise
    produces."""
    labels = _balanced(80)
    states = _states(80, 64, 12, signal_from=6, labels=labels)
    report = probe.sweep(states, labels, n_permutations=8)
    assert report.expected_false_positives == pytest.approx(0.6)


def test_the_report_survives_json():
    labels = _balanced(80)
    states = _states(80, 64, 3, signal_from=1, labels=labels)
    out = json.loads(
        json.dumps(
            probe.sweep(states, labels, n_permutations=6).to_dict(), allow_nan=False
        )
    )
    assert out["n_readable_layers"] >= 0
    assert len(out["layers"]) == 3
    assert "best_layer" in out


# --------------------------------------------------------- the direction


def test_the_exported_direction_is_a_unit_vector_in_the_streams_own_space():
    labels = _balanced(80)
    states = _states(80, 64, 3, signal_from=0, labels=labels)
    direction = probe.direction_at(states, labels, 0)
    assert direction.shape == (64,)
    assert abs(float(direction.norm()) - 1.0) < 1e-4


def test_the_direction_separates_the_classes_it_was_fitted_on():
    labels = _balanced(80)
    states = _states(80, 64, 3, signal_from=0, labels=labels)
    direction = probe.direction_at(states, labels, 0)
    projected = states[0].float() @ direction
    assert projected[labels == 1].mean() != projected[labels == 0].mean()


# ------------------------------------------------- against a real model


@pytest.fixture(scope="module")
def gpt2():
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


SINGULAR = [
    "The dog runs",
    "The cat sleeps",
    "The man walks",
    "The bird sings",
    "The child plays",
    "The horse jumps",
    "The woman reads",
    "The boy writes",
    "The tree grows",
    "The river flows",
    "The stone falls",
    "The lamp shines",
    "The door opens",
    "The clock ticks",
    "The wind blows",
    "The fire burns",
    "The ship sails",
    "The bell rings",
    "The star fades",
    "The road bends",
]
PLURAL = [
    "The dogs run",
    "The cats sleep",
    "The men walk",
    "The birds sing",
    "The children play",
    "The horses jump",
    "The women read",
    "The boys write",
    "The trees grow",
    "The rivers flow",
    "The stones fall",
    "The lamps shine",
    "The doors open",
    "The clocks tick",
    "The winds blow",
    "The fires burn",
    "The ships sail",
    "The bells ring",
    "The stars fade",
    "The roads bend",
]


def _examples():
    return [{"text": t, "label": 0} for t in SINGULAR] + [
        {"text": t, "label": 1} for t in PLURAL
    ]


def test_a_real_concept_is_readable_through_the_runtime(gpt2):
    report = gpt2.probe_layers(_examples(), n_permutations=10)
    assert report["n_readable_layers"] > 0
    assert report["best_layer"] is not None
    assert report["receipt"]["op"] == "probe_layers"
    assert report["n_test"] >= probe.MIN_TEST


def test_an_example_without_an_integer_label_is_refused(gpt2):
    bad = [{"text": "a", "label": "singular"}] * 20
    with pytest.raises(BadRequest, match="not an integer"):
        gpt2.probe_layers(bad)


def test_saving_a_direction_is_refused_when_nothing_was_readable(
    gpt2, tmp_path, monkeypatch
):
    """A vector fitted where the probe never beat shuffled labels is fitted to
    noise, and the store is the one place it would later be used with none of
    this context beside it."""
    monkeypatch.setenv("MODELMRI_HOME", str(tmp_path))
    import random

    scrambled = _examples()
    random.Random(0).shuffle(scrambled)
    # Keep the texts, destroy the correspondence between text and label.
    labels = [row["label"] for row in scrambled]
    noise = [
        {"text": row["text"], "label": labels[(i + 7) % len(labels)]}
        for i, row in enumerate(_examples())
    ]
    report = gpt2.probe_layers(noise, n_permutations=10)
    if report["best_layer"] is None:
        with pytest.raises(Refusal, match="fitted to noise"):
            gpt2.probe_layers(noise, n_permutations=10, save_as="junk")


def test_a_saved_direction_carries_the_probe_evidence(gpt2, tmp_path, monkeypatch):
    monkeypatch.setenv("MODELMRI_HOME", str(tmp_path))
    from modelmri import steer_vectors

    out = gpt2.probe_layers(_examples(), n_permutations=10, save_as="plurality")
    assert out["saved"]["dims"] > 0

    stored = json.loads(
        (steer_vectors.store_dir() / "plurality.json").read_text(encoding="utf-8")
    )
    assert stored["method"] == "probe"
    assert stored["accuracy"] > stored["null_high"], (
        "a saved direction comes from a layer that beat its own null"
    )
    assert "READABLE IS NOT USED" in stored["note"]


# --------------------------- the ceiling is the 95th, not the best shuffle


def test_the_null_ceiling_is_not_the_best_of_the_shuffles():
    """The comment beside NULL_HIGH rules this out in as many words: "with 20
    refits the single best shuffle is the best of 20 draws and using it as the
    ceiling would make the null wider every time more permutations were run."

    `band[min(n - 1, round(n * 95 / 100))]` is one rank too high at every n,
    and at the default 20 permutations it is index 19 of 0..19 — the maximum.
    """
    band = [i / 100 for i in range(probe.N_PERMUTATIONS)]
    ceiling = probe._percentile(band, probe.NULL_HIGH)
    assert ceiling < band[-1], "the ceiling is the best shuffle of the batch"
    assert ceiling == band[18]


@pytest.mark.parametrize("n", [20, 50, 100, 200])
def test_the_ceiling_never_lands_on_the_maximum_at_any_permutation_count(n):
    """The failure mode the comment predicts: a band that widens as more
    permutations are run, because the estimator keeps picking the extreme."""
    band = [i / n for i in range(n)]
    assert probe._percentile(band, probe.NULL_HIGH) < band[-1]


def test_the_reported_rate_matches_the_band_it_describes():
    """`expected_false_positives` is `n_layers * (100 - NULL_HIGH) / 100`,
    which assumes the ceiling is genuinely the 95th percentile. Against the
    max of 20 draws the true rate is 1/21 — the two agreed only by coincidence
    at that one permutation count, and raising it would have moved the real
    rate and left the reported one alone."""
    band = [i / 1000 for i in range(1000)]
    ceiling = probe._percentile(band, probe.NULL_HIGH)
    above = sum(1 for v in band if v > ceiling)
    assert above / len(band) == pytest.approx((100 - probe.NULL_HIGH) / 100, abs=0.002)


def test_the_floor_is_the_low_percentile_and_not_the_worst_shuffle():
    band = [i / 100 for i in range(100)]
    floor = probe._percentile(band, probe.NULL_LOW)
    assert floor == band[4]
    assert floor > band[0]


def test_a_single_permutation_still_answers():
    """n=1 has one rank; the estimator must not index past it or below zero."""
    assert probe._percentile([0.7], probe.NULL_HIGH) == 0.7
    assert probe._percentile([0.7], probe.NULL_LOW) == 0.7


def test_a_percentile_of_nothing_is_refused_rather_than_invented():
    with pytest.raises(ValueError):
        probe._percentile([], probe.NULL_HIGH)


# ------------------ one label vector, one answer, whichever two values


def _separable(seed: int = 0, d: int = 32, n: int = 16):
    torch = pytest.importorskip("torch")
    torch.manual_seed(seed)
    low = torch.randn(n, d)
    low[:, 0] -= 3.0
    high = torch.randn(n, d)
    high[:, 0] += 3.0
    return {7: torch.cat([low, high])}, n


def test_the_exported_direction_does_not_depend_on_which_two_labels_were_used():
    """`sweep` remapped any two distinct integers to 0/1 and `direction_at`
    did not, so one label vector got two answers.

    `runtime.probe_layers` only checks that a label is an int, so 1 and 2 are
    legal. `sweep` remapped them and scored the layer correctly; `direction_at`
    handed the raw 1s and 2s to `_fit`, whose
    `binary_cross_entropy_with_logits` does not validate its target — a target
    of 2.0 gives a finite loss whose gradient never changes sign. The fit ran
    to completion and the vector went into the steering store carrying the
    accuracy sweep had measured.

    MEASURED before the fix: cosine 0.883 between the two directions, with the
    largest-weight dimension moving from 20 to 3. Close enough to look right.
    """
    torch = pytest.importorskip("torch")
    states, n = _separable()

    zero_one = torch.tensor([0] * n + [1] * n)
    one_two = torch.tensor([1] * n + [2] * n)

    a = probe.direction_at(states, zero_one, 7)
    b = probe.direction_at(states, one_two, 7)
    assert float(a @ b) == pytest.approx(1.0, abs=1e-6)


def test_the_remap_sends_the_same_class_to_one_on_both_paths():
    """Beyond the remap itself: if the two paths disagreed about WHICH class
    becomes 1, the exported direction would point opposite to the accuracy
    reported next to it. One helper, so they cannot."""
    torch = pytest.importorskip("torch")

    for pair in ((0, 1), (1, 2), (3, 9), (-1, 1)):
        labels = torch.tensor([pair[0]] * 4 + [pair[1]] * 4)
        mapped = probe._binary(labels)
        assert mapped.tolist() == [0, 0, 0, 0, 1, 1, 1, 1], (
            f"{pair} did not map the larger value to 1"
        )


def test_a_direction_for_three_classes_is_refused():
    """A separating direction between three classes is not a thing."""
    torch = pytest.importorskip("torch")
    states, n = _separable()
    labels = torch.tensor([0] * n + [1] * (n - 4) + [2] * 4)
    with pytest.raises(BadRequest, match="two classes"):
        probe.direction_at(states, labels, 7)
