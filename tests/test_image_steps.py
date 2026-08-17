"""Where the denoiser committed, and every step of that this refuses to claim.

A commit step is a number somebody will act on by dragging a slider down, so
the ways it can be quietly wrong are the ways it costs them an image. It can be
wrong because the threshold that produced it was not attached to it; because a
step nobody measured was reported as a zero; because the run was unseeded and
another run commits elsewhere; because a fraction of path length was read as a
fraction of the distance travelled; or because "the latent stopped moving" was
read as "the picture stopped changing". Each test below names the one it stops.

The pipeline here is small and REAL: a genuine `UNet2DConditionModel` with
genuine cross-attention blocks, a real `AutoencoderKL`, a real `CLIPTextModel`,
built in-process so the suite needs no network and no gigabytes. The weights are
random, so nothing here is a claim about where a trained model commits — what is
under test is that the step axis survives, that the arithmetic is the arithmetic
it says it is, that no latent is ever decoded, and that the refusals fire.

The arithmetic half needs none of that and is tested without a pipeline at all,
which is the point of keeping it pure.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("transformers")

from modelmri import image_steps as ist  # noqa: E402
from modelmri.errors import BadRequest  # noqa: E402


@pytest.fixture(scope="module")
def pipe():
    """A real Stable Diffusion pipeline, small enough to run on CPU."""
    from diffusers import (
        AutoencoderKL,
        DDIMScheduler,
        StableDiffusionPipeline,
        UNet2DConditionModel,
    )
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

    torch.manual_seed(0)
    unet = UNet2DConditionModel(
        block_out_channels=(32, 64),
        layers_per_block=1,
        sample_size=32,
        in_channels=4,
        out_channels=4,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim=32,
        norm_num_groups=8,
    )
    vae = AutoencoderKL(
        block_out_channels=[32],
        in_channels=3,
        out_channels=3,
        down_block_types=["DownEncoderBlock2D"],
        up_block_types=["UpDecoderBlock2D"],
        latent_channels=4,
        norm_num_groups=8,
    )
    text_encoder = CLIPTextModel(
        CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
        )
    )
    # `else`-shaped rather than a bare fall-through, for the reason
    # `test_image_attention.py` records: `pytest.skip` raises, so the name is
    # never really unbound, but neither a reader nor a static analyser can see
    # that from the shape.
    try:
        tokenizer = CLIPTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-clip"
        )
    except Exception:
        pytest.skip("the tiny CLIP tokenizer is not cached and there is no network")
        raise  # unreachable; makes the control flow explicit

    built = StableDiffusionPipeline(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=DDIMScheduler(),
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    built.set_progress_bar_config(disable=True)
    return built


PROMPT = "a red cube on a table"


# ------------------------------------------------------------ the step axis


def test_every_step_is_kept_rather_than_collapsed_to_one_number(pipe):
    """A commit step with no curve under it is a claim nobody can check, and it
    cannot tell a run that tailed off smoothly from one that fell off a cliff."""
    run = ist.trace(pipe, PROMPT, seed=7, steps=6)
    assert len(run.steps) == 6
    assert [s.step for s in run.steps] == [0, 1, 2, 3, 4, 5]
    # Timesteps must DECREASE — that is what denoising is, and a constant or
    # increasing sequence would mean the callback is not seeing the schedule.
    times = [s.timestep for s in run.steps]
    assert times == sorted(times, reverse=True), times


def test_the_first_step_carries_none_rather_than_a_change_of_zero(pipe):
    """The starting noise is never handed to the callback, so the movement
    during step 0 is UNMEASURED. A zero there would claim the largest move most
    runs make never happened, and it would draw as a flat first bar."""
    run = ist.trace(pipe, PROMPT, seed=7, steps=6)
    assert run.steps[0].rms_change is None
    assert run.steps[0].cumulative is None
    assert all(s.rms_change is not None for s in run.steps[1:])
    assert "STEP 0 CARRIES NO CHANGE" in run.means()
    assert "rather than as zero" in run.means()


def test_the_cumulative_share_ends_at_exactly_one(pipe):
    """Not 0.9999999999999999. A threshold of 1.0 has to be reachable, and a
    curve that stops short of its own total reads as movement unaccounted for."""
    run = ist.trace(pipe, PROMPT, seed=7, steps=6)
    assert run.steps[-1].cumulative == 1.0
    assert run.commit_at(1.0) == 5


def test_the_share_never_goes_backwards(pipe):
    """A cumulative fraction that dips would mean a step moved the latent a
    negative distance, which is not a thing a distance does."""
    run = ist.trace(pipe, PROMPT, seed=7, steps=6)
    shares = [s.cumulative for s in run.steps if s.cumulative is not None]
    assert shares == sorted(shares)


def test_the_distance_to_the_final_latent_is_zero_only_at_the_end(pipe):
    """`rms_to_final` is the column that stops "95% of the movement" being read
    as "95% of the way there"; it has to be a real measured distance."""
    run = ist.trace(pipe, PROMPT, seed=7, steps=6)
    assert run.steps[-1].rms_to_final == 0.0
    assert run.steps[0].rms_to_final > 0.0


def test_a_change_is_reported_beside_the_scale_of_the_thing_that_changed(pipe):
    """An RMS change of 0.4 is enormous or invisible depending on the latent it
    happened to, and a distance with no scale beside it is not a size."""
    run = ist.trace(pipe, PROMPT, seed=7, steps=4)
    assert all(s.latent_rms is not None and s.latent_rms > 0 for s in run.steps)


# ------------------------------------------------------------- the threshold


def test_the_commit_step_never_travels_without_its_threshold(pipe):
    """ "Committed at step 4" is meaningless. The threshold that produced it has
    to be in the same sentence and in the same payload."""
    run = ist.trace(pipe, PROMPT, seed=7, steps=6)
    said = run.means()
    assert "95% threshold" in said
    assert "THE THRESHOLD IS A CONVENTION, NOT A FINDING" in said
    payload = run.to_dict()
    assert payload["threshold"] == 0.95
    assert payload["commit_step"] == run.commit_at(0.95)


def test_the_step_at_other_thresholds_is_reported_so_none_is_read_alone(pipe):
    """One threshold's answer, alone, looks like a property of the model. The
    distance between the 50% step and the 99% step is the actual finding."""
    run = ist.trace(pipe, PROMPT, seed=7, steps=6)
    commits = run.to_dict()["commits"]
    assert [c["threshold"] for c in commits] == list(ist.REPORTED_THRESHOLDS)
    assert "50% at step" in run.means()


def test_a_stricter_threshold_never_commits_earlier(pipe):
    """If it could, the cumulative curve would not be cumulative and every
    reading off it would be arbitrary."""
    run = ist.trace(pipe, PROMPT, seed=7, steps=6)
    steps = [run.commit_at(t) for t in (0.5, 0.9, 0.95, 0.99)]
    assert all(a is not None for a in steps)
    assert steps == sorted(steps)


def test_the_threshold_is_an_argument_not_a_constant(pipe):
    """A different threshold has to actually change the answer, or it is
    decoration on a hardcoded number."""
    lax = ist.trace(pipe, PROMPT, seed=7, steps=8, threshold=0.25)
    strict = ist.trace(pipe, PROMPT, seed=7, steps=8, threshold=0.99)
    assert lax.commit_at(0.25) < strict.commit_at(0.99)
    assert "25% threshold" in lax.means()
    assert "99% threshold" in strict.means()


# -------------------------------------------------------------------- the seed


def test_a_run_cannot_be_left_unseeded_by_forgetting(pipe):
    """An unseeded trace looks exactly like a seeded one in the chart. The
    keyword has no default so the decision has to be made out loud."""
    with pytest.raises(TypeError, match="seed"):
        ist.trace(pipe, PROMPT, steps=4)


def test_the_same_seed_reproduces_the_same_trajectory(pipe):
    """Without this the commit step is not a measurement of anything, because
    nobody can run it again and get the same number."""
    a = ist.trace(pipe, PROMPT, seed=7, steps=5)
    b = ist.trace(pipe, PROMPT, seed=7, steps=5)
    assert [s.rms_change for s in a.steps] == [s.rms_change for s in b.steps]
    assert a.commit_at(0.95) == b.commit_at(0.95)


def test_another_seed_is_another_trajectory(pipe):
    """The evidence for the caveat: a commit step belongs to a trajectory, not
    to the model. If two seeds gave one answer the caveat would be wrong."""
    a = ist.trace(pipe, PROMPT, seed=7, steps=5)
    b = ist.trace(pipe, PROMPT, seed=8, steps=5)
    assert [s.rms_change for s in a.steps] != [s.rms_change for s in b.steps]


def test_an_unseeded_run_says_it_cannot_be_reproduced(pipe):
    """ "No seed" and "seed 0" are different runs and only one of them can be
    checked or compared against another."""
    run = ist.trace(pipe, PROMPT, seed=None, steps=4)
    assert run.seed is None
    assert "NO SEED WAS FIXED" in run.means()
    assert "cannot be reproduced" in run.means()
    assert "Seed 7." in ist.trace(pipe, PROMPT, seed=7, steps=4).means()


def test_a_seed_that_is_not_a_whole_number_is_refused(pipe):
    with pytest.raises(BadRequest, match="whole number"):
        ist.trace(pipe, PROMPT, seed=7.5, steps=4)


# ------------------------------------------------------------ latents not pixels


def test_the_vae_is_never_asked_to_decode_anything(pipe):
    """Measured, not promised. Decoding a frame per step would cost a full pass
    through the decoder per step and would make the commit step a property of
    the decoder as much as of the denoiser."""
    calls = []
    original = pipe.vae.decode

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    pipe.vae.decode = counting
    try:
        run = ist.trace(pipe, PROMPT, seed=7, steps=5)
    finally:
        # `del`, not a re-assignment: putting the bound method back would leave
        # an instance attribute shadowing the class's own method on a fixture
        # every other test in this file shares.
        del pipe.vae.decode

    assert calls == [], f"{len(calls)} VAE decode(s) happened"
    assert run.vae_decodes == 0
    assert "0 VAE decodes" in run.means()


def test_the_result_says_latent_distance_is_not_visible_difference(pipe):
    """The reader is looking at a chart that says the image stopped changing.
    The decoder is non-linear and that is not what was measured."""
    said = ist.trace(pipe, PROMPT, seed=7, steps=4).means()
    assert "LATENT DISTANCE IS NOT VISIBLE DIFFERENCE" in said
    assert "not the same sentence as when the picture stopped changing" in said


def test_the_result_says_the_fractions_are_path_length_not_distance(pipe):
    """A trajectory that wanders can be 95% along its own path and still a long
    way from where it ends, and the chart cannot show the difference."""
    said = ist.trace(pipe, PROMPT, seed=7, steps=5).means()
    assert "THESE ARE PATH LENGTHS, NOT DISTANCES" in said
    assert "rms_to_final" in said


def test_the_result_says_one_trajectory_is_not_a_property_of_the_model(pipe):
    said = ist.trace(pipe, PROMPT, seed=7, steps=4).means()
    assert "ONE TRAJECTORY, NOT A PROPERTY OF THE MODEL" in said
    assert "measured once is a sample, not a property" in said


def test_the_scheduler_is_named_because_it_decides_where_the_steps_go(pipe):
    """The same model on another scheduler redistributes the movement entirely,
    so a commit step compared across schedulers compares two different runs."""
    run = ist.trace(pipe, PROMPT, seed=7, steps=4)
    assert run.scheduler == "DDIMScheduler"
    assert "DDIMScheduler" in run.means()


def test_the_denominator_of_every_fraction_is_published(pipe):
    """A fraction whose denominator is hidden is not a number anybody can
    check, and this one excludes step 0 — which the sentence has to admit."""
    run = ist.trace(pipe, PROMPT, seed=7, steps=5)
    assert run.total_change > 0
    assert f"{run.total_change:,.4f} of movement from step 1 onward" in run.means()


def test_the_whole_result_survives_json(pipe):
    """It crosses a wire to a panel. A payload with a tuple or a NaN in it is
    not a payload."""
    payload = ist.trace(pipe, PROMPT, seed=7, steps=4).to_dict()
    assert json.loads(json.dumps(payload))["steps"][0]["rms_change"] is None


# -------------------------------------------------------------- pricing it first


def test_the_shape_priced_before_the_run_is_the_shape_the_run_produced(pipe):
    """A predicted cost that does not match the real run is not a prediction.
    This is the only check that `latent_shape_of` reads the right fields."""
    predicted = ist.latent_shape_of(pipe)
    run = ist.trace(pipe, PROMPT, seed=7, steps=4)
    assert predicted == run.latent_shape
    assert run.bytes_held == 0, "the latents are released once measured"


def test_the_memory_priced_before_the_run_is_the_memory_the_run_held(pipe):
    """The bound is only a bound if the arithmetic matches what was allocated."""
    priced = ist.plan(4, latent_shape=ist.latent_shape_of(pipe))
    held = []

    def watch(index, _total):
        held.append(index)

    run = ist.trace(pipe, PROMPT, seed=7, steps=4, on_step=watch)
    assert held == [0, 1, 2, 3], "the progress callback sees every step"
    assert priced["total_bytes"] == 4 * priced["latent_bytes"]
    # 4 channels x 32 x 32 x 4 bytes, on the pipeline this fixture built.
    assert priced["latent_bytes"] == 16384
    assert run.latent_shape == (4, 32, 32)


def test_a_pipeline_whose_shape_cannot_be_read_is_priced_as_unknown():
    """None, never 0. A run whose memory could not be priced is not a run that
    costs nothing."""

    class Opaque:
        unet = object()

    assert ist.latent_shape_of(Opaque()) is None
    priced = ist.plan(20)
    assert priced["latent_bytes"] is None
    assert priced["total_bytes"] is None
    assert priced["fits"] is None
    assert "cannot be priced" in priced["means"]
    assert "rather than as zero" in priced["means"]


def test_the_cost_is_known_before_it_is_spent():
    """No model needed — which is the point of pricing before loading."""
    priced = ist.plan(50, latent_shape=(4, 64, 64))
    assert priced["denoiser_passes"] == 50
    assert priced["vae_decodes"] == 0
    assert priced["latents_kept"] == 50
    assert priced["latent_bytes"] == 65536
    assert priced["total_bytes"] == 3276800
    assert priced["fits"] is True
    assert "No seconds are quoted" in priced["means"]
    assert "NO VAE DECODES AT ALL" in priced["means"]


def test_a_trace_that_would_not_fit_says_so_before_it_is_run():
    """A latent with a time axis is the shape this was not written for, and
    finding out by running out of memory is not a bound."""
    priced = ist.plan(50, latent_shape=(16, 25, 60, 104))
    assert priced["fits"] is False
    assert "past the" in priced["means"]
    assert "smaller height and width" in priced["means"]


def test_a_shape_with_a_bool_in_it_is_not_priced_as_a_one():
    """isinstance(True, int) is True, so a typo would otherwise price a
    one-channel latent and report that it fits."""
    assert ist.plan(20, latent_shape=(True, 64, 64))["latent_bytes"] is None
    assert ist.plan(20, latent_shape=(4, 0, 64))["latent_bytes"] is None


def test_plan_refuses_exactly_what_the_run_would():
    """A preflight that accepts a run the run refuses is worse than no
    preflight — it is a promise the next call breaks."""
    with pytest.raises(BadRequest, match="too few"):
        ist.plan(2)
    with pytest.raises(BadRequest, match="past the"):
        ist.plan(ist.MAX_STEPS + 1)


def test_a_trace_too_large_to_hold_is_refused_one_step_in(pipe, monkeypatch):
    """After ONE step, not after all of them, and with the arithmetic and the
    two parameters that would fix it in the sentence."""
    monkeypatch.setattr(ist, "MAX_TRACE_BYTES", 1024)
    with pytest.raises(BadRequest) as caught:
        ist.trace(pipe, PROMPT, seed=7, steps=4)
    said = str(caught.value)
    assert "(4, 32, 32) latent" in said
    assert "smaller height and width" in said


# ------------------------------------------------------------------- refusals


def test_a_two_step_run_is_refused_because_every_threshold_answers_the_same(pipe):
    """One measured change means step 1 at 50%, at 95% and at 99% alike — an
    answer that looks like a measurement of the model and is a fact about the
    step count."""
    with pytest.raises(BadRequest) as caught:
        ist.trace(pipe, PROMPT, seed=7, steps=2)
    said = str(caught.value)
    assert "EVERY threshold reports step 1" in said
    assert "distilled few-step model" in said


def test_too_many_steps_is_refused_rather_than_truncated(pipe):
    """Running 50 of a requested 500 would not measure the first tenth of the
    500-step run — a scheduler puts its steps somewhere else entirely — so a
    cap here would label a different run with the number that was asked for."""
    with pytest.raises(BadRequest) as caught:
        ist.trace(pipe, PROMPT, seed=7, steps=ist.MAX_STEPS + 1)
    said = str(caught.value)
    assert "different run wearing the number you asked for" in said


def test_a_step_count_that_is_a_bool_is_refused(pipe):
    """isinstance(True, int) is True, and `steps=True` is one step."""
    with pytest.raises(BadRequest, match="whole number of denoising steps"):
        ist.trace(pipe, PROMPT, seed=7, steps=True)


def test_a_pipeline_with_no_denoiser_is_refused():
    class Nothing:
        pass

    with pytest.raises(ist.NotSupported, match="no `unet` or `transformer`"):
        ist.trace(Nothing(), PROMPT, seed=7, steps=4)


def test_a_pipeline_that_hides_its_latents_is_refused():
    """Rebuilding them from the denoiser's input would mean undoing whatever
    scaling the scheduler applied, which is a different quantity."""

    class Hidden:
        unet = object()

        def __call__(self, prompt, *, callback_on_step_end, **kwargs):
            callback_on_step_end(self, 0, 9.0, {})
            return None

    with pytest.raises(ist.NotSupported, match="does not hand its latents"):
        ist.trace(Hidden(), PROMPT, seed=7, steps=4)


def test_a_batch_of_images_is_refused_rather_than_averaged():
    """Averaging several images' movement into one curve reports a commit step
    that belongs to none of them."""

    class Batched:
        unet = object()

        def __call__(self, prompt, *, callback_on_step_end, **kwargs):
            callback_on_step_end(self, 0, 9.0, {"latents": torch.zeros(3, 4, 8, 8)})
            return None

    with pytest.raises(ist.NotSupported, match="belongs to none of them"):
        ist.trace(Batched(), PROMPT, seed=7, steps=4)


def test_a_run_that_hands_over_one_latent_cannot_be_measured():
    """One latent is a position, not a movement. An empty chart would read as
    "nothing changed"."""

    class Once:
        unet = object()

        def __call__(self, prompt, *, callback_on_step_end, **kwargs):
            callback_on_step_end(self, 0, 9.0, {"latents": torch.zeros(1, 4, 8, 8)})
            return None

    with pytest.raises(ist.NotSupported, match="needs at least"):
        ist.trace(Once(), PROMPT, seed=7, steps=4)


def test_a_failed_run_does_not_poison_the_next_one(pipe):
    """Nothing is attached to the pipeline, and a run that raises has to leave
    it exactly as usable as it was."""
    before = dict(pipe.unet.attn_processors)
    # A size the VAE cannot downsample. diffusers raises `ValueError: "height"
    # and "width" have to be divisible by 8 but are 7 and 7` — named exactly
    # rather than caught as a blind `Exception`, so this keeps testing the
    # recovery rather than quietly passing on some future unrelated error.
    with pytest.raises(ValueError, match="divisible by 8"):
        ist.trace(pipe, PROMPT, seed=7, steps=4, height=7, width=7)
    assert set(before) == set(pipe.unet.attn_processors)
    assert len(ist.trace(pipe, PROMPT, seed=7, steps=4).steps) == 4


# ------------------------------------------------------------- the held latents


def test_a_scheduler_that_writes_in_place_cannot_rewrite_a_held_latent():
    """`.to()` returns the SAME storage when nothing needs converting, so
    without the copy a scheduler reusing its buffer would rewrite every step
    already held — and every change would read as zero, drawing a chart that
    says the model committed at step 1."""
    store = ist._Trace(steps=3)
    latent = torch.zeros(1, 4, 4, 4)
    store.add(0, 9.0, latent)
    latent.add_(5.0)
    store.add(1, 8.0, latent)
    assert float(store.latents[0].max()) == 0.0
    assert float(store.latents[1].max()) == 5.0


def test_a_released_trace_holds_nothing(pipe):
    """A traceback keeps every frame's locals alive, so a 50-step run that
    failed at step 49 would hold 49 latents through every handler above it.
    "It will be collected eventually" is not a memory bound."""
    store = ist._Trace(steps=2)
    store.add(0, 9.0, torch.zeros(1, 4, 8, 8))
    assert store.bytes_held == 1024
    store.release()
    assert store.latents == []
    assert store.bytes_held == 0
    store.release()  # idempotent: it runs on a path that may already have run


def test_a_float16_latent_is_widened_before_it_is_differenced():
    """A float16 difference between two nearly identical late-step latents
    rounds away exactly the small values a commit reading depends on."""
    store = ist._Trace(steps=2)
    store.add(0, 9.0, torch.zeros(1, 4, 8, 8, dtype=torch.float16))
    assert store.latents[0].dtype.name == "float32"
    assert store.bytes_held == 1024


# ------------------------------------------------------------- the arithmetic
#
# No pipeline, no torch, no GPU. This is the half whose correctness can be
# checked on any machine, which is why it is pure.


def test_an_unmeasured_step_stays_unmeasured_through_the_arithmetic():
    """The one that matters most. If `None` became `0.0` anywhere in here, the
    chart would show a real first bar of zero height and the fractions would be
    shares of a total that never existed."""
    fractions, total = ist.cumulative_fractions([None, 1.0, 1.0, 2.0])
    assert fractions[0] is None
    assert fractions[1:] == [0.25, 0.5, 1.0]
    assert total == 4.0


def test_the_last_fraction_is_exactly_one_so_a_full_threshold_is_reachable():
    """0.1 + 0.2 is 0.30000000000000004. Dividing by a separately summed total
    gives 0.9999999999999999, which a threshold of 1.0 never reaches."""
    fractions, _ = ist.cumulative_fractions([None, 0.1, 0.2])
    assert fractions[-1] == 1.0
    assert ist.commit_step(fractions, 1.0) == 2


def test_the_commit_step_is_the_first_past_the_threshold_not_the_nearest():
    """ "Nearest" would let a step that has not reached the threshold be
    reported as the step at which it was reached."""
    fractions, _ = ist.cumulative_fractions([None, 1.0, 1.0, 2.0])
    assert ist.commit_step(fractions, 0.3) == 2
    assert ist.commit_step(fractions, 0.25) == 1


def test_a_run_that_never_moved_refuses_rather_than_committing_at_step_zero():
    """A total of zero makes every fraction 0/0. Reporting step 0 would read as
    the model deciding everything immediately, which is the opposite finding."""
    with pytest.raises(ist.NotMeasurable) as caught:
        ist.cumulative_fractions([None, 0.0, 0.0])
    assert "share of zero is not a number" in str(caught.value)


def test_a_run_with_nothing_measured_is_refused_as_a_run_nobody_looked_at():
    """Different from a run that did not move, and the message says so."""
    with pytest.raises(BadRequest, match="a run nothing looked at"):
        ist.cumulative_fractions([None, None])


def test_a_non_finite_change_is_refused_rather_than_drawn_as_a_gap():
    """A NaN draws as a break in the curve, which reads as the denoiser holding
    still — indistinguishable from a genuine plateau."""
    with pytest.raises(ist.NotMeasurable, match="not finite"):
        ist.cumulative_fractions([None, 1.0, float("nan")])


def test_a_negative_change_is_refused_because_a_distance_cannot_be_negative():
    with pytest.raises(BadRequest, match="cannot be"):
        ist.cumulative_fractions([None, -1.0])


def test_a_bool_change_is_not_a_change_of_one():
    """isinstance(True, int) is True, so `True` would otherwise sail through as
    a movement of 1.0."""
    with pytest.raises(BadRequest, match="not a number"):
        ist.cumulative_fractions([None, True])


def test_a_threshold_outside_the_share_range_is_refused_by_name():
    """At 0 every run "commits" at its first measured step, which is a fact
    about the threshold rather than about the model."""
    fractions, _ = ist.cumulative_fractions([None, 1.0, 1.0])
    for bad in (0, -0.5, 1.5, float("nan")):
        with pytest.raises(BadRequest, match="share of the"):
            ist.commit_step(fractions, bad)
    with pytest.raises(BadRequest, match="between 0 and 1"):
        ist.commit_step(fractions, True)


def test_a_threshold_no_step_reaches_is_none_rather_than_the_last_step():
    """ "The run never got there" and "the run got there at the end" are
    different answers about a trajectory."""
    assert ist.commit_step([None, 0.4, 0.6], 0.9) is None


def test_the_default_threshold_is_stated_rather_than_buried():
    """A magic 0.95 inside a function is a number nobody can change or cite."""
    assert ist.DEFAULT_COMMIT_THRESHOLD == 0.95
    assert ist.DEFAULT_COMMIT_THRESHOLD in ist.REPORTED_THRESHOLDS
