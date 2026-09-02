"""A saved direction that nothing can apply is a write with no reader.

`steer_vectors.py` has fitted, scored and persisted contrastive directions
since it was written, and `runtime.probe_layers(save_as=...)` has been filling
that store from the probe panel. Nothing ever loaded one back. This file is
the other half: the runtime's second steering arm, and the two things it must
not get wrong.

**It must refuse a direction that cannot belong to this model, by name.** A
768-dimensional vector added to a 2048-dimensional stream is a crash; a
2048-dimensional vector fitted on a DIFFERENT 2048-wide model is worse,
because it steers, plausibly, and nothing about the output says the basis was
somebody else's. The refusal has to name both models — the one it came from
and the one in front of you — because "shapes disagree" is a sentence about
tensors and this is a question about provenance.

**And it must report the strength honestly.** `steer_vectors.py`'s own
docstring: "a scale of 5 means nothing across models or even across layers".
The coefficient applied is constant alpha, which is what CAA does and what
keeps two runs comparable — but what is REPORTED beside it is alpha divided by
the residual norm this machine just measured at that layer, and when there is
nothing to measure it on the relative figure is absent with a reason rather
than a zero.

The model here is a three-layer GPT-2 built from a config, so nothing is
downloaded and every assertion below is about arithmetic this process just
did.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from modelmri import steer_vectors as sv  # noqa: E402
from modelmri.errors import BadRequest, Refusal  # noqa: E402
from modelmri.runtime import ModelRuntime  # noqa: E402

N_LAYERS = 3
D_MODEL = 32
VOCAB = 64


class Tok:
    """Just enough tokenizer for `fit_steering_direction`'s capture loop.

    A real GPT-2 tokenizer needs its vocabulary files on disk, and what the
    code under test asks of it is one call and one key. Deterministic from the
    text, so two identical prompts give identical ids.
    """

    def __call__(self, text: str, return_tensors: str = "pt"):
        ids = [(ord(c) % (VOCAB - 1)) + 1 for c in text[:24]] or [1]
        return {"input_ids": torch.tensor([ids])}

    def decode(self, ids):
        return "".join(f"<{i}>" for i in ids)


@pytest.fixture
def rt(tmp_path, monkeypatch):
    """A runtime holding a real (tiny, untrained) causal LM on the CPU.

    `MODELMRI_HOME` rather than a monkeypatch of `steer_vectors.store_dir`,
    because the runtime imports that module inside the method — the same
    reason `tests/test_probe.py` isolates the store this way.
    """
    import transformers

    monkeypatch.setenv("MODELMRI_HOME", str(tmp_path))
    cfg = transformers.GPT2Config(
        n_layer=N_LAYERS, n_head=2, n_embd=D_MODEL, vocab_size=VOCAB, n_positions=32
    )
    torch.manual_seed(0)
    runtime = ModelRuntime()
    runtime.model = transformers.AutoModelForCausalLM.from_config(cfg).eval()
    runtime.tokenizer = Tok()
    runtime.hf_id = "tiny/gpt2-under-test"
    runtime.backend = "hf"
    runtime.device = "cpu"
    runtime.last_ids = torch.tensor([1, 5, 9, 13])
    runtime.last_ids_epoch = runtime.epoch
    return runtime


def _direction(seed: int = 0, dims: int = D_MODEL):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(dims, generator=g)
    return v / v.norm()


def _save(
    name: str,
    *,
    layer: int = 1,
    dims: int = D_MODEL,
    model: str = "tiny/gpt2-under-test",
    seed: int = 0,
):
    return sv.save(
        name,
        _direction(seed, dims),
        {
            "model": model,
            "layer": layer,
            "hidden_size": dims,
            "method": "caa",
            "dtype": "float32",
            "beats_null": True,
        },
    )


def _logits(runtime) -> torch.Tensor:
    """The next-token distribution for the fixed input, under whatever hook
    is installed right now. Greedy decoding reads the argmax of this row, so
    two runs that agree here agree on every token they would produce."""
    ids = runtime.last_ids.unsqueeze(0)
    handle = None
    if runtime._steer_dir is not None or runtime._steer is not None:
        handle = runtime._steer_handle()
    try:
        with torch.no_grad():
            return runtime.model(ids).logits[0, -1].clone()
    finally:
        if handle is not None:
            handle.remove()


# ------------------------------------------------------- it actually steers


def test_applying_a_saved_direction_moves_the_next_token_logits(rt):
    _save("politeness", layer=1)
    baseline = _logits(rt)
    rt.set_steering_direction("politeness", 6.0)
    steered = _logits(rt)
    assert not torch.equal(baseline, steered)


def test_the_same_direction_at_the_same_strength_is_reproducible(rt):
    """Deterministic, or the A/B beside it is comparing two samples rather
    than two interventions."""
    _save("politeness", layer=1)
    rt.set_steering_direction("politeness", 6.0)
    first = _logits(rt)
    rt.set_steering_direction("politeness", 6.0)
    assert torch.equal(first, _logits(rt))


def test_clearing_restores_the_baseline_byte_for_byte(rt):
    _save("politeness", layer=1)
    baseline = _logits(rt)
    rt.set_steering_direction("politeness", 6.0)
    rt.clear_steering()
    assert torch.equal(baseline, _logits(rt))
    assert rt.steering_status() == {"active": False}


def test_it_is_applied_at_the_layer_the_direction_was_saved_at(rt):
    """The layer is provenance, not a parameter. The same vector pushed into
    a different block is a different intervention and would report the same
    name and the same strength for it."""
    _save("early", layer=0, seed=3)
    _save("late", layer=2, seed=3)
    rt.set_steering_direction("early", 5.0)
    early = _logits(rt)
    rt.set_steering_direction("late", 5.0)
    assert not torch.equal(early, _logits(rt))
    assert rt.steering_status()["layer"] == 2


def test_a_layer_outside_this_model_is_refused_rather_than_clamped(rt):
    _save("from-a-deeper-model", layer=40)
    with pytest.raises(BadRequest, match="outside this model"):
        rt.set_steering_direction("from-a-deeper-model", 1.0)


# -------------------------------------------------------------- it refuses


def test_a_direction_of_the_wrong_width_names_both_models(rt):
    """The sentence has to carry the model it came FROM and the model it is
    being pushed INTO. One of the two alone leaves the reader guessing which
    end is wrong."""
    _save("from-a-wider-model", dims=D_MODEL * 2, model="Qwen/Qwen3-1.7B")
    with pytest.raises(Refusal) as caught:
        rt.set_steering_direction("from-a-wider-model", 1.0)
    said = caught.value.sentence
    assert "Qwen/Qwen3-1.7B" in said, said
    assert "tiny/gpt2-under-test" in said, said
    assert "Refusing rather than reshaping" in said


def test_a_direction_that_is_not_saved_refuses_by_name(rt):
    """BY NAME is the guarantee, so the name is what is asserted. Matching
    only the prefix would pass against a sentence that had stopped carrying
    it — and a store full of near-identical names is exactly when the reader
    needs to know which one was not found."""
    with pytest.raises(Refusal) as caught:
        rt.set_steering_direction("never-fitted", 1.0)
    assert "no saved direction called 'never-fitted'" in caught.value.sentence


def test_a_direction_from_another_checkpoint_of_the_same_width_warns(rt):
    """Deliberately not a refusal — `steer_vectors.load` already argues that
    lifting a direction onto a finetune is a real experiment. What it must
    never be is silent, so the warning rides on the status."""
    _save("borrowed", layer=1, model="Qwen/Qwen3-1.7B")
    status = rt.set_steering_direction("borrowed", 2.0)
    assert any("equal size is not equal basis" in w for w in status["warnings"])


def test_a_recording_cannot_be_steered(rt):
    _save("politeness", layer=1)
    rt.replay = object()
    with pytest.raises(Refusal, match="This is a recording"):
        rt.set_steering_direction("politeness", 1.0)


# --------------------------------------------------- the strength is honest


def test_the_reported_strength_is_the_alpha_over_a_measured_norm(rt):
    """Hand-computed against the same definition `fit_direction` records: the
    mean L2 norm of the residual stream entering that layer, at the last
    token."""
    _save("politeness", layer=1)
    states = sv._last_token_states(rt.model, rt._block, [rt.last_ids.unsqueeze(0)], [1])
    expected = float(states[1].norm(dim=-1).mean())

    status = rt.set_steering_direction("politeness", 4.0)
    strength = status["strength"]
    assert strength["alpha"] == pytest.approx(4.0)
    # Published to three decimals, the same precision `fit_direction` records
    # its own `residual_norm` at — one number for one quantity, so a card can
    # put the fitted norm beside the applied one without them disagreeing in
    # a digit neither of them means.
    assert strength["residual_norm"] == pytest.approx(expected, abs=1e-3)
    assert strength["relative"] == pytest.approx(
        4.0 / strength["residual_norm"], rel=1e-9
    )
    assert strength["relative"] == pytest.approx(4.0 / expected, rel=1e-2)
    assert strength["layer"] == 1
    assert strength["measured"], "the status has to say HOW the norm was taken"
    assert strength["unmeasured"] == ""


def test_with_nothing_generated_the_relative_strength_is_unknown_not_zero(rt):
    """There is no prompt to measure the stream on, so there is no relative
    figure. Reporting 0.0 would say the push is negligible, which is the one
    thing an unmeasured number must never say."""
    _save("politeness", layer=1)
    rt.last_ids = None
    status = rt.set_steering_direction("politeness", 4.0)
    strength = status["strength"]
    assert strength["relative"] is None
    assert strength["residual_norm"] is None
    assert strength["unmeasured"], "an absent measurement has to say why"


def test_a_norm_measured_on_an_earlier_generation_stops_claiming_this_one(rt):
    """The number is measured ONCE, at apply time, and the direction is meant
    to outlive the generation it was applied during — running the model under
    it is the whole point. So the norm is still a real measurement afterwards,
    but the sentence beside it stops being true: `residual_norm_at` writes
    "the current generation ... just now", and the panel re-reads the status
    on every generation and prints that sentence verbatim under the slider.

    A measured number wearing another prompt's provenance is worse than an
    absent one, because it reads as freshly taken. The number is kept — it
    was really measured — and the sentence is replaced with one that names
    the generation it belongs to.
    """
    _save("politeness", layer=1)
    fresh = rt.set_steering_direction("politeness", 4.0)["strength"]
    assert "current generation" in fresh["measured"]
    assert fresh["residual_norm"] is not None

    # What `generate_stream` does on the next run: a new tensor of ids for the
    # same model. `_steer_dir` deliberately survives it, and nothing re-measures.
    rt.last_ids = torch.tensor([2, 6, 10, 14, 18])
    rt.last_ids_epoch = rt.epoch

    later = rt.steering_status()["strength"]
    assert later["residual_norm"] == fresh["residual_norm"], "still a real number"
    assert later["relative"] == pytest.approx(fresh["relative"])
    assert "just now" not in later["measured"], later["measured"]
    assert "when this direction was applied" in later["measured"], later["measured"]
    assert "re-apply" in later["measured"], "say what to do about it"


def test_the_norm_still_reads_as_current_while_it_is_current(rt):
    """The other half of the pair above: the freshly-measured sentence must
    not be replaced on every poll, or the honest case would carry the caveat
    written for the stale one."""
    _save("politeness", layer=1)
    rt.set_steering_direction("politeness", 4.0)
    for _ in range(3):
        said = rt.steering_status()["strength"]["measured"]
        assert "just now" in said, said
        assert "when this direction was applied" not in said, said


def test_the_relative_strength_helper_is_the_one_definition():
    """Both halves of the arithmetic live in one place, so the panel's label
    and the receipt cannot drift apart."""
    assert sv.relative_strength(4.0, 50.0) == pytest.approx(0.08)
    assert sv.relative_strength(-4.0, 50.0) == pytest.approx(-0.08)
    assert sv.relative_strength(4.0, None) is None
    assert sv.relative_strength(4.0, 0.0) is None


# ------------------------------------------------------- the tagged union


def test_setting_a_direction_clears_a_feature_steer_and_the_reverse(rt):
    """One slot, two arms. Two live interventions at once would each be
    reported without the other, and the A/B would name one of them."""

    class StubSAE:
        d_sae = 8
        layer = 1
        point = "resid_pre"

        def steering_vector(self, fid):
            return _direction(fid)

    rt.sae = StubSAE()
    _save("politeness", layer=1)

    rt.set_steering(3, 5.0)
    assert rt.steering_status()["kind"] == "feature"
    rt.set_steering_direction("politeness", 2.0)
    status = rt.steering_status()
    assert status["kind"] == "direction" and rt._steer is None
    rt.set_steering(3, 5.0)
    assert rt._steer_dir is None


def test_the_feature_arm_keeps_every_field_it_published(rt):
    """`FeaturesPanel` and `demo.ts` both read this shape. New keys are free;
    a missing or renamed one is a broken panel with nothing on screen saying
    so."""

    class StubSAE:
        d_sae = 8
        layer = 1
        point = "resid_pre"

        def steering_vector(self, fid):
            return _direction(fid)

    rt.sae = StubSAE()
    status = rt.set_steering(3, 5.0)
    assert status["active"] is True
    assert status["feature_id"] == 3
    assert status["scale"] == 5.0


def test_clearing_through_the_old_route_clears_a_direction_too(rt):
    """`FeaturesPanel.onSteerTest` calls `setSteer(null)` three times per A/B,
    including in its catch arm, on the promise that it "always leaves the
    model clean". A direction it could not see would survive that."""
    _save("politeness", layer=1)
    rt.set_steering_direction("politeness", 2.0)
    assert rt.set_steering(None) == {"active": False}
    assert rt._steer_dir is None


def test_a_model_swap_drops_the_direction(rt):
    """The vector is sized and based on the model that was loaded. Left
    behind, its handle would add a d_model-shaped tensor to whatever loaded
    next."""
    _save("politeness", layer=1)
    rt.set_steering_direction("politeness", 2.0)
    rt.unload()
    assert rt._steer_dir is None


# ------------------------------------------------------------- fitting one


def _pairs():
    """Contrast pairs whose LAST token differs.

    The residual stream entering block 0 is the embedding, so two sets that
    end on the same token are identical there — and `_fit` refuses that in
    words rather than returning a zero vector. A fixture whose pairs agreed
    on their final character made the sweep refuse at layer 0 before it
    reached the layers it was written to exercise, which is the estimator
    working exactly as designed on a badly chosen fixture.
    """
    positive = [f"pair {i} is a yes" for i in range(10)]
    negative = [f"pair {i} is a no" for i in range(10)]
    return positive, negative


def test_fitting_reports_the_whole_null_table_not_just_a_verdict(rt):
    positive, negative = _pairs()
    out = rt.fit_steering_direction(positive, negative, method="caa")
    assert out["ran"] is True
    assert len(out["layers"]) == N_LAYERS
    for row in out["layers"]:
        assert "p_value" in row and "null_max" in row and "beats_null" in row
        assert "residual_norm" in row
    assert out["passes"] == len(positive) + len(negative)
    assert out["receipt"]["op"] == "fit_direction"


def test_asking_only_for_the_estimate_spends_nothing_beyond_the_probe(rt):
    positive, negative = _pairs()
    out = rt.fit_steering_direction(positive, negative, estimate_only=True)
    assert out["ran"] is False
    assert out["layers"] == []
    assert out["estimate"]["passes"] == len(positive) + len(negative)
    assert out["estimate"]["basis"]


def _refusing_projection(monkeypatch):
    """Make `budget.project` return a verdict of "refuse" without needing a
    full accelerator to be genuinely full.

    The numbers are fabricated and say so in their own `basis`; what is under
    test is the ORDER of the price quote and the guard, which is invisible on
    a CPU because `budget.check` is silent on `verdict="unknown"`.
    """
    from modelmri import budget

    def over_budget(probe, passes, *, retained_bytes=0):
        return budget.Estimate(
            passes=passes,
            seconds=probe.seconds * passes,
            peak_bytes=90_000_000_000,
            retained_bytes=retained_bytes,
            free_bytes=1_000_000_000,
            fraction_of_free=90.0,
            verdict="refuse",
            basis="a fabricated projection, to put the guard in front of the quote",
        )

    monkeypatch.setattr(budget, "project", over_budget)
    return budget


def test_a_price_the_budget_would_refuse_is_still_quoted(rt, monkeypatch):
    """`estimate_only` exists to say what a run would cost BEFORE it is spent,
    and the expensive case is the only one a reader needs the number for. A
    guard that fires before the quote is returned answers "how much?" with a
    refusal to say, which is the opposite of what the panel asked."""
    budget = _refusing_projection(monkeypatch)
    positive, negative = _pairs()

    out = rt.fit_steering_direction(positive, negative, estimate_only=True)
    assert out["ran"] is False
    assert out["estimate"]["verdict"] == "refuse"
    assert out["estimate"]["peak_bytes"] == 90_000_000_000
    assert out["probe"]["seconds"] >= 0

    # The guard itself is untouched: pricing is not permission.
    with pytest.raises(budget.TooCostly, match="fitting a direction over"):
        rt.fit_steering_direction(positive, negative)
    assert rt.fit_steering_direction(positive, negative, confirm=True)["ran"] is True


def test_the_catalogue_reads_the_live_model_once(rt):
    """`direction_catalogue` runs without the lock on purpose, so a load or an
    unload can land in the middle of it. Reading `self.model` twice — once to
    check it is there and once for its config — is a 500 in that window, on
    the one route whose contract is to answer with a list rather than a
    refusal. Bound once, the window closes."""
    real = rt.model
    reads = {"n": 0}

    class Swapping(type(rt)):
        @property
        def model(self):
            reads["n"] += 1
            # The second read is the race: an unload between the guard and
            # `.config` used to reach `None.config`.
            return real if reads["n"] == 1 else None

        @model.setter
        def model(self, value):  # nothing under test assigns it
            pass

    _save("politeness", layer=1)
    rt.__class__ = Swapping
    out = rt.direction_catalogue()
    assert reads["n"] == 1, f"the model was read {reads['n']} times, not once"
    assert out["hidden_size"] == D_MODEL
    assert out["directions"][0]["compatible"] is True


@pytest.fixture
def no_passes(monkeypatch):
    """Fail loudly if anything runs the model.

    BEFORE A FORWARD PASS is the whole claim of the checks below, and asserting
    only that "some refusal was raised" would pass against the IDENTICAL
    sentence raised by `fit_direction` deep inside the sweep — after 2n passes
    have already been spent. `probe_layers` learned this the expensive way: its
    per-row validation was thorough and never ran.
    """

    def never(*args, **kwargs):
        raise AssertionError("a forward pass was spent before the refusal")

    monkeypatch.setattr(sv, "_last_token_states", never)


def test_too_few_pairs_is_refused_before_a_forward_pass(rt, no_passes):
    with pytest.raises(Refusal, match="at least 8"):
        rt.fit_steering_direction(["a", "b"], ["c", "d"])


def test_unmatched_sets_are_refused_before_a_forward_pass(rt, no_passes):
    positive, negative = _pairs()
    with pytest.raises(BadRequest, match="must be matched"):
        rt.fit_steering_direction(positive, negative[:-1])


def test_an_empty_set_is_refused_in_words(rt, no_passes):
    with pytest.raises(BadRequest, match="contrast pairs"):
        rt.fit_steering_direction([], [])


def test_a_probe_saved_direction_carries_a_verdict_and_a_norm(rt):
    """`steer_vectors.load` checks `payload.get("beats_null") is False`, and a
    missing key is not False — so for every direction the product could
    produce, that warning was unreachable code. The probe writer now records
    both the verdict and the norm the panel reads a strength against."""
    # 26 a class, because `probe.MIN_TEST` is 12 held out and the split keeps
    # a quarter: an accuracy measured on eight examples has a resolution of
    # twelve percentage points, and probe.py refuses it in those words.
    #
    # NO try/except pytest.skip AROUND THIS CALL. It was written with one, and
    # a skip here turns the regression this test exists to catch — the probe
    # writer dropping `beats_null` again, or `MIN_TEST` moving under the split
    # below — into a green run with nothing on screen. The two sets are
    # lexically separable ("yes" against "no"), so a refusal from here is news.
    examples = [{"text": f"pair {i} is a yes", "label": 0} for i in range(26)] + [
        {"text": f"pair {i} is a no", "label": 1} for i in range(26)
    ]
    out = rt.probe_layers(examples, n_permutations=8, save_as="from-the-probe")
    assert out["saved"]["name"] == "from-the-probe"
    _, payload, _ = sv.load(
        "from-the-probe", hidden_size=D_MODEL, model="tiny/gpt2-under-test"
    )
    assert payload["beats_null"] is True
    assert payload["residual_norm"] > 0
    assert payload["saved_at"]


def test_saving_a_fit_that_beat_nothing_is_refused(rt, monkeypatch):
    """The store is the one place a direction is later picked up with none of
    this context beside it."""
    positive, negative = _pairs()

    def nothing_survived(*args, **kwargs):
        rows, vectors = _real_sweep(*args, **kwargs)
        for row in rows["layers"]:
            row["beats_null"] = False
        rows["best_layer"] = None
        rows["survived"] = 0
        return rows, vectors

    _real_sweep = sv.sweep
    monkeypatch.setattr(sv, "sweep", nothing_survived)
    with pytest.raises(Refusal, match="worth saving"):
        rt.fit_steering_direction(positive, negative, save_as="nothing")


def test_a_saved_fit_carries_the_evidence_that_judged_it(rt):
    """The evidence travels with the vector, whenever there is a vector.

    WHETHER A RANDOM MODEL'S DIRECTION BEATS ITS OWN SHUFFLED NULL IS A CLAIM
    ABOUT ACTIVATIONS, NOT ABOUT CODE. These weights are untrained and the
    pairs differ in one token, so the separation is whatever that
    initialisation happens to give — and the permutation test is doing its job
    when it says that is not enough. Measured: it beats the null on
    windows/py3.13 and does not on macos/py3.11 or py3.12, which is how this
    arrived, as a red CI against a green local run.

    The guard this replaces read `if out["best_layer"] is None`, which cannot
    happen on this call: `save_as` was passed, and the runtime REFUSES rather
    than returning a fit with nothing to save — `test_saving_a_fit_that_beat_
    nothing_is_refused` above pins exactly that sentence. So the guard was dead
    code and the path that actually occurs was unguarded.

    Only that one refusal is skipped, by its own words, and everything else is
    re-raised: a fit that fell over for any other reason is still a failure
    here, and the assertions below still run on every machine where a layer
    wins.
    """
    positive, negative = _pairs()
    try:
        out = rt.fit_steering_direction(positive, negative, save_as="loved-vs-hated")
    except Refusal as err:
        if "worth saving" not in str(err):
            raise
        pytest.skip(f"no layer beat its null on this untrained model: {err}")
    assert out["saved"]["name"] == "loved-vs-hated"
    _, payload, _ = sv.load(
        "loved-vs-hated", hidden_size=D_MODEL, model="tiny/gpt2-under-test"
    )
    for key in ("beats_null", "p_value", "effect", "residual_norm", "saved_at"):
        assert key in payload, key


# ------------------------------------------- the panel that reads them back

# EVERY WRITER NEEDS A READER, and the reader for the numbers above is
# `SteeringPanel.tsx`. There is no test runner for the frontend in this repo
# — `npm` corrupts `node_modules` on the mount and the type-check happens
# from a clean copy — so what can be pinned here is text. Two things are worth
# pinning, because both are cases of the panel quietly turning an honest
# backend number into a dishonest screen, and neither is a type error.

PANEL = (
    __import__("pathlib").Path(__file__).resolve().parents[1]
    / "frontend"
    / "src"
    / "SteeringPanel.tsx"
)


def _panel_code() -> str:
    """The panel WITHOUT its comments.

    Two of the pins below assert that a phrase is gone, and this file's own
    convention is to explain a defect where it was fixed — so the sentence
    being banned is quoted verbatim in the comment sitting above its
    replacement, and a plain substring search finds it there and fails. The
    comment is the receipt; searching the code means searching the code.
    """
    import re

    return re.sub(r"/\*.*?\*/", "", PANEL.read_text(encoding="utf-8"), flags=re.S)


def test_the_panel_never_launders_an_absent_number_through_a_zero():
    """`measured()` already takes `number | null | undefined` and answers "—".
    Its own docstring says the `?? 0` at the call sites was the defect and the
    widened signature is the fix: "a `?? 0` is how 'not measured' becomes
    'measured, and it was zero'". On the one band that tells a reader an
    intervention is live, "alpha 0.00" is the worst available lie."""
    import re

    source = PANEL.read_text(encoding="utf-8")
    offenders = re.findall(r"measured\([^)]*\?\?[^)]*\)", source)
    assert offenders == [], offenders


def test_the_sliders_relative_label_is_divided_from_the_slider():
    """The headline beside the strength slider is alpha over a measured norm.
    The APPLIED alpha's quotient (`strength.relative`) is a different number
    the moment the slider moves off the applied value, and printing it over
    "alpha +4" made the two halves of one control disagree. The applied
    measurement is still what supplies the denominator — it is the numerator
    that has to be live."""
    source = PANEL.read_text(encoding="utf-8")
    assert "strength / s.residual_norm" in source
    assert "scaled(s.relative)" not in source, (
        "the slider's own label is being read off the applied coefficient"
    )


# --------------------------- a layer with no direction is a ROW, not a crash

# THE DEFECT THESE PIN. Open the panel, leave its own default contrast pairs
# alone, click "Fit the direction": 409, "the positive and negative sets have
# identical mean activations at this layer". Nothing fitted, anywhere, on a
# 30-layer model where 29 layers had a perfectly good direction.
#
# The estimator was right. `_last_token_states` captures the stream ENTERING
# each block, and entering block 0 that is the last token's own embedding, so
# any two sets whose prompts end with the same token — a full stop, which is
# how sentences end — are literally the same vector at layer 0. `_fit` says so
# and refuses, which is correct: there is no direction there.
#
# What was wrong is that `sweep` treated that one ordinary answer as a failed
# measurement and threw away the other 29. A layer with nothing between the
# two sets is a ROW WITH NO RESULT, in exactly the way this package already
# reports a layer that does not beat its null.


def _shared_final_token_pairs(n: int = 12):
    """Twelve pairs whose prompts all end on the same token, at the same
    position — the ordinary case for real sentences, and the one that used to
    abort the whole sweep.

    Both halves matter. `Tok` is deterministic from the text and reads
    `text[:24]`, so a shared FINAL CHARACTER is a shared final token; and this
    model is a GPT-2, whose embedding is `wte + wpe`, so a shared final
    position is needed too or the positional term alone would separate the two
    sets at layer 0 and the fixture would not reproduce anything. Every string
    here is exactly 24 characters and every one of them ends in a full stop.

    The bodies differ as loudly as this vocabulary allows, so that the layers
    which DO have something to fit clear their nulls by a wide margin rather
    than by a coin flip: measured on this fixture, layer 1 separates 68.4
    against a worst shuffle of 1.72.
    """
    positive = [f"yes yes yes yes yes {i}".ljust(23)[:23] + "." for i in range(n)]
    negative = [f"no no no no no no {i}".ljust(23)[:23] + "." for i in range(n)]
    return positive, negative


@pytest.mark.parametrize("method", list(sv.METHODS))
def test_a_layer_with_no_direction_is_a_row_not_a_failed_sweep(rt, method):
    """29 measured layers are not discarded because the 30th had nothing.

    PARAMETRIZED OVER EVERY SHIPPED METHOD, because the first version of this
    ran on `caa` alone and the whole path was dead under `repe`. `repe` reads
    PC1 of the paired differences, and an SVD of all-zero differences returns
    zero singular values with an arbitrary orthonormal V — so it produced a
    unit basis vector instead of refusing, `sweep` never reached its
    no-direction branch, and layer 0 came back `p_value 1.0` with a bar on the
    chart and the words "inside its null" beside it. The method is a dropdown
    on the panel, so that was one click from the reproduction this whole
    section exists for.
    """
    positive, negative = _shared_final_token_pairs()
    out = rt.fit_steering_direction(positive, negative, method=method)

    assert out["ran"] is True
    assert len(out["layers"]) == N_LAYERS

    zero = out["layers"][0]
    assert zero["layer"] == 0
    assert zero["effect"] == 0
    assert zero["beats_null"] is False
    # AN UNKNOWN IS NOT A ZERO. No labels were shuffled and no refit was
    # scored at this layer, so there is no permutation p-value to publish, no
    # mean of the nulls and no worst of them — and 0.0 is the most confident
    # number in each of those ranges. Absent from the row, not present and
    # lying: `null_max` in particular is what the chart's own tooltip renders
    # as "the worst label-shuffled refit reached 0.000".
    for absent in ("p_value", "null_mean", "null_max"):
        assert absent not in zero, (absent, zero)

    note = " ".join(zero["notes"])
    # THE SUBJECT of the sentence, not a mention of the layer somewhere in it:
    # a note that had stopped naming its own layer still passed `"layer 0" in
    # note`, because the explanation that follows says "entering layer 0 the
    # residual stream is...". That is the mutation this line exists to catch.
    assert note.startswith("no direction at layer 0"), note
    assert "identical mean activations" in note, note

    # The whole point: the layers that had something are untouched, and one of
    # them is the answer.
    assert out["best_layer"] is not None
    assert out["best_layer"] > 0
    assert out["survived"] >= 1
    for row in out["layers"][1:]:
        assert isinstance(row["p_value"], float), row
        assert isinstance(row["null_max"], float), row


def test_asking_only_for_a_layer_with_no_direction_refuses_and_names_it(rt):
    """One degenerate layer out of thirty is a row. One out of one is a no.

    The sentence is the deliverable here. "identical mean activations at this
    layer" — the estimator's own words, which the OLD refusal also used — does
    not tell a reader WHICH layer, WHY, or what to do instead, and a test that
    matched on that phrase would pass against the defect it was written to
    stop. So every assertion below is on wording the old sentence never had.
    """
    positive, negative = _shared_final_token_pairs()
    with pytest.raises(Refusal) as caught:
        rt.fit_steering_direction(positive, negative, layers=[0])

    sentence = caught.value.sentence
    assert "the only layer this fit asked for" in sentence, sentence
    assert "layer 0" in sentence, sentence
    # The cause, named: a shared final token, and what layer 0 is.
    assert "same token" in sentence, sentence
    assert "embedding" in sentence, sentence
    # And what to do instead.
    assert "layer 1" in sentence, sentence


def test_a_fit_with_nothing_between_the_two_sets_names_every_layer(rt):
    """The other half of the rule: refuse only when EVERY layer is degenerate.

    Two identical sets have nothing between them anywhere, not merely at the
    embedding, and a sweep of rows that all say "no result" is a table with no
    reading in it. That is a refusal — and it names the layers it tried, so
    the reader can see it was all of them and not the first one.
    """
    positive, _ = _shared_final_token_pairs()
    with pytest.raises(Refusal) as caught:
        rt.fit_steering_direction(positive, list(positive))

    sentence = caught.value.sentence
    assert "every layer this fit asked for" in sentence, sentence
    for layer in range(N_LAYERS):
        assert str(layer) in sentence, sentence
    # AND IT DOES NOT PRESCRIBE WHAT WAS JUST DONE. A fit with no `layers` in
    # the body sweeps every layer the model has, so this refusal's reader has
    # just swept the whole stack — and the first remedy the sentence offered
    # was "Sweep the whole stack rather than these layers", which reproduces
    # this refusal word for word. A refusal has to name what to do.
    assert "Sweep the whole stack" not in sentence, sentence
    assert "differ in more than their labels" in sentence, sentence


def test_a_refusal_above_layer_zero_explains_itself_without_a_back_reference():
    """A back-reference needs something to refer back to, and the clause it
    points at is only ever printed when layer 0 was asked for.

    The cause clause is additive by design — the embedding argument is true of
    block 0 and of nothing else — so a request that excludes layer 0 prints
    only the above-zero half. That half opened with "Above layer 0 identical
    means say more than that", pointing at a sentence the reader was never
    shown. Reachable from the route: `layers` arrives from the request body
    and is only bounds-checked.
    """
    dangling = "say more than that"
    premise = "no longer the raw embedding"

    for layers in ([7], [1, 2], list(range(1, N_LAYERS))):
        sentence = sv._nothing_to_fit(layers)
        assert dangling not in sentence, (layers, sentence)
        assert premise in sentence, (layers, sentence)
        for layer in layers:
            assert str(layer) in sentence, (layers, sentence)

    # And the back-reference is kept where it has something to refer to, which
    # is the only reason the shorter wording is worth having at all.
    with_zero = sv._nothing_to_fit([0, 1, 2])
    assert dangling in with_zero, with_zero
    assert premise not in with_zero, with_zero


def test_the_no_direction_note_is_about_the_layer_it_is_on():
    """A note travels away from its row — the panel prints it under the chart,
    a saved fit carries it — so it names its own layer, and it does not repeat
    the layer-0 explanation at a layer that explanation is not true of. The
    residual stream is the embedding entering block 0 and nowhere else."""
    zero = sv._no_direction_note(0)
    deep = sv._no_direction_note(7)
    assert zero.startswith("no direction at layer 0"), zero
    assert deep.startswith("no direction at layer 7"), deep
    assert "embedding" in zero, zero
    assert "embedding" not in deep, deep


class TailTok(Tok):
    """`Tok`, keeping the sentence's LAST character.

    The fixture tokenizer reads `text[:24]`, and every one of the panel's
    defaults is longer than that — so the full stop all 24 of them end with is
    truncated away, and with it the exact property that made the shipped
    defaults refuse on a real model. A real tokenizer sees the whole sentence
    and its final token IS that full stop.

    Keeping the last character, with 23 before it so the model's 32 positions
    still fit, reproduces on this fixture what SmolLM2-135M does for real. The
    test below is worth nothing without it: under plain `Tok` the defaults
    truncate to 24 different final characters and never reach the branch.
    """

    def __call__(self, text: str, return_tensors: str = "pt"):
        clipped = text[:23] + text[-1] if len(text) > 24 else text
        return super().__call__(clipped, return_tensors)


def _panel_defaults() -> tuple[list[str], list[str]]:
    """The two default contrast sets, READ OUT OF THE PANEL.

    Copied strings would pin a fossil: the point of the test below is that the
    text the reader is handed on their first click still fits, so it has to be
    the text the panel actually ships today. Same reasoning, and the same
    file, as the two tests above that read `PANEL` for its arithmetic.
    """
    import re

    source = PANEL.read_text(encoding="utf-8")
    sets = []
    for name in ("POSITIVE_DEFAULT", "NEGATIVE_DEFAULT"):
        block = re.search(rf"const {name} = \[(.*?)\]\.join", source, re.S)
        assert block, f"{name} is not where this test looks for it in {PANEL.name}"
        sets.append(re.findall(r'"((?:[^"\\]|\\.)*)"', block.group(1)))
    return sets[0], sets[1]


def test_the_panels_own_defaults_fit_rather_than_refuse(rt):
    """The first click on a fresh panel must not be a 409.

    This is the reproduction from the review, in process: the panel's twelve
    shipped pairs, every one of them ending in a full stop, through the same
    route the button calls. It refused on a real model and it refuses on this
    fixture for the identical reason, which is what makes a three-layer GPT-2
    a fair stand-in for a thirty-layer Llama here.
    """
    positive, negative = _panel_defaults()
    assert len(positive) == len(negative) >= sv.MIN_PAIRS, (
        f"the panel ships {len(positive)} against {len(negative)} lines"
    )

    rt.tokenizer = TailTok()
    out = rt.fit_steering_direction(positive, negative, method="caa")
    assert out["ran"] is True
    assert len(out["layers"]) == N_LAYERS
    # Whatever else this untrained model does with them, no row may carry a
    # p-value it did not measure.
    for row in out["layers"]:
        assert "p_value" not in row or isinstance(row["p_value"], float), row
    # A COVERAGE GUARD, not a product requirement. Every assertion above is
    # satisfied by a fit where nothing degenerate happened at all, so this test
    # could go on passing while quietly ceasing to exercise the branch it was
    # written for — by someone shortening a default below `TailTok`'s 24
    # characters, or by someone reading the new note and "fixing" the defaults
    # to end on different words. If that happens the right move is to delete
    # this line and say so, not to leave a green test standing guard over
    # nothing.
    assert any("p_value" not in row for row in out["layers"]), (
        "the panel's shipped defaults no longer collide at layer 0, so this "
        "test has stopped reproducing the defect it exists for — see TailTok"
    )


def test_the_panel_draws_a_layer_with_no_direction_as_no_data(rt):
    """Every writer needs a reader, and the reader for the row above is the
    chart. A degenerate row has `effect` 0, and a 0.4%-wide bar at the axis is
    how "there was nothing to measure" gets drawn as "measured, and weak" —
    the same lie in pixels that a `p_value` of 0.0 would be in JSON. The
    hatched no-data treatment already in this stylesheet (`.tl-block.no-dur`)
    is the house recipe for it, and "inside its null" is the wrong label for a
    row that never had a null."""
    source = PANEL.read_text(encoding="utf-8")
    assert "st-nodir" in source, "the chart has no no-data treatment"
    assert "no direction here" in source
    css = (PANEL.parent / "styles.css").read_text(encoding="utf-8")
    assert ".st-nodir" in css
    rule = css.split(".st-nodir {")[1].split("}")[0]
    assert "repeating-linear-gradient" in rule
    # NO ELEMENT OPACITY ON THE PLATE. The recipe was lifted from
    # `.tl-block.no-dur`, where `opacity: 0.75` is a step down from the
    # block's own 0.9 over an OPAQUE fill. Over a translucent one the two
    # dims compound: measured over the track, the hatch came out 1.15:1
    # stripe-against-gap in light standard, against 2.18:1 for the same
    # gradient in the timeline, and the whole "no result" plate read fainter
    # (1.29:1) than the ordinary null band it stands in for (1.42:1).
    assert "opacity" not in rule, rule


def test_the_chart_does_not_count_a_never_nulled_layer_as_one_that_failed():
    """The headline over the chart is the sentence a reader reads first.

    "{survived} of {layers.length} layers beat their own null" put the layers
    that never HAD a null into the population that failed to clear one — so a
    30-layer sweep with a degenerate layer 0 said "12 of 30", which tells the
    reader 18 layers were tested and lost when 17 were and one was never
    entered. That is the same claim the row verdict had to stop making, one
    element higher up. Text-pinned, like the two above: there is no frontend
    runner in this repo.
    """
    source = _panel_code()
    assert "of {fit.layers.length} layers beat their own null" not in source, (
        "the denominator counts layers that never had a null to beat"
    )
    # The denominator is the layers that were actually scored, and the rest
    # are counted and named rather than folded in.
    assert 'fit.layers.filter((l) => typeof l.p_value === "number").length' in source
    assert "had no direction to test" in source
    # "No layer's separation beat its own label-shuffled refits" is the same
    # error in the other branch: it is printed whenever `best_layer` is null,
    # which a sweep containing degenerate rows reaches.
    assert "No layer's separation beat its own" not in source, source


def test_the_note_a_degenerate_row_carries_is_printed_and_not_only_hovered():
    """Every writer needs a reader.

    `_no_direction_note` is three sentences carrying the whole explanation —
    why layer 0 is the last token's own embedding, why the p-value is absent
    rather than zero. The panel printed notes only for `fit.best_layer`, and a
    degenerate row is `beats_null: false` by construction so it can never BE
    the best layer. The note therefore reached the reader through one channel:
    `title` on a non-interactive span. Mouse-only — not keyboard-reachable,
    not in the accessible name, invisible on touch.
    """
    source = _panel_code()
    notes = source.split(".flatMap((l) => l.notes)")[0]
    filtered = notes[notes.rindex("fit.layers") :]
    assert 'typeof l.p_value !== "number"' in filtered, filtered
    assert "l.layer === fit.best_layer" in filtered, filtered
