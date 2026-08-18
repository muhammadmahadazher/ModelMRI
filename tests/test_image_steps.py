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

import base64
import io
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


# ============================================================== the filmstrip
#
# The companion that DOES decode, and every way a strip of frames can lie about
# the run it came from: by hiding that it is a subset, by claiming a decode
# count it did not measure, by shrinking a picture without saying so, by being
# read as the model's guess at the finished image, or by leaving a wrapper
# attached to somebody else's VAE. Each test below names the one it stops.


def _counting_decode(pipe, calls):
    """Wrap `pipe.vae.decode` so a test can count what really happened."""
    original = pipe.vae.decode

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    return counting


class _Scheduler:
    """A named scheduler for the stub pipelines, because the strip reports it."""


class _Stub:
    """A pipeline whose denoiser sits in the `transformer` slot — a DiT shape.

    Real where it matters: the `vae` handed in is a genuine `AutoencoderKL`, so
    every stub test below exercises the real decode path. What is faked is the
    denoising loop, which is exactly the part that would need a download.

    `**kwargs` in the signature on purpose: a `__call__` that swallows anything
    proves nothing about what it accepts, and `_call_surface` must read that as
    "cannot say" rather than as a refusal.
    """

    def __init__(self, vae, latents):
        self.transformer = object()
        self.vae = vae
        self.scheduler = _Scheduler()
        self.latents = latents

    def __call__(self, prompt, *, num_inference_steps, callback_on_step_end, **kwargs):
        for index, latent in enumerate(self.latents):
            callback_on_step_end(self, index, float(100 - index), {"latents": latent})
        return None


@pytest.fixture(scope="module")
def tiny_vae():
    """A real AutoencoderKL — small, random, and genuinely a decoder."""
    from diffusers import AutoencoderKL

    torch.manual_seed(0)
    return AutoencoderKL(
        block_out_channels=[32],
        in_channels=3,
        out_channels=3,
        down_block_types=["DownEncoderBlock2D"],
        up_block_types=["UpDecoderBlock2D"],
        latent_channels=4,
        norm_num_groups=8,
    )


# ------------------------------------------------- the subset, stated exactly


def test_only_the_named_steps_are_decoded_and_the_count_is_the_real_one(pipe):
    """The whole feature in one test: a subset chosen by the caller, and a decode
    count that was counted rather than intended. Fifty decodes of an SDXL latent
    do not fit beside the pipeline, so "decode them all" is not an option that
    exists — and a claimed count is not a count."""
    calls = []
    pipe.vae.decode = _counting_decode(pipe, calls)
    try:
        strip = ist.filmstrip(pipe, PROMPT, seed=7, steps=8, every=3)
    finally:
        del pipe.vae.decode

    assert strip.decoded_steps == [0, 3, 6, 7]
    assert len(strip.frames) == 4
    assert len(calls) == 4, f"{len(calls)} decode(s) really happened"
    assert strip.vae_decodes == 4
    assert strip.vae_decodes_for_frames == 4


def test_the_strip_says_which_steps_ran_and_were_never_looked_at(pipe):
    """The requirement that makes the strip honest. Without `skipped_steps` an
    8-frame strip and an 8-step run are the same object to a reader."""
    strip = ist.filmstrip(pipe, PROMPT, seed=7, steps=8, every=3)
    payload = strip.to_dict()
    assert payload["skipped_steps"] == [1, 2, 4, 5]
    assert payload["decoded_steps"] == [0, 3, 6, 7]
    assert payload["steps_run"] == 8
    assert payload["frames_decoded"] == 4
    # Every step the run took is on exactly one of the two lists.
    assert sorted(payload["decoded_steps"] + payload["skipped_steps"]) == list(range(8))
    said = strip.means()
    assert "THIS IS A SUBSET, AND THIS IS EXACTLY WHICH" in said
    assert "Run and never looked at: 4 step(s)" in said
    assert "the strip is not a 4-step run" in said


def test_the_gap_between_two_frames_is_not_one_step_of_work(pipe):
    """The reading everybody will make off a filmstrip. Two neighbouring frames
    six steps apart show six steps of change, and the sentence has to say so or
    the picture argues otherwise."""
    said = ist.filmstrip(pipe, PROMPT, seed=7, steps=8, every=3).means()
    assert "everything that happened across the gap between their step numbers" in said


def test_a_selected_step_the_run_never_reached_is_its_own_list(tiny_vae):
    """Different from a step deliberately skipped: one is a choice and the other
    is a gap in what could be seen. Folding them together would hide a pipeline
    whose callback does not reach every step."""
    latents = [torch.randn(1, 4, 8, 8) for _ in range(2)]
    strip = ist.filmstrip(
        _Stub(tiny_vae, latents),
        PROMPT,
        seed=7,
        steps=6,
        at=[0, 4],
        include_final=False,
    )
    assert strip.decoded_steps == [0]
    assert strip.steps_never_reached == [4]
    assert strip.steps_run == 2
    assert "SELECTED STEP(S) NEVER ARRIVED" in strip.means()
    assert "gap in what this could see" in strip.means()


# --------------------------------------------------------------- the frames


def test_every_frame_is_a_real_png_at_the_size_it_reports(pipe):
    """A frame is only evidence if it opens. The reported size has to be the
    size of the bytes, or the resolution fields are decoration."""
    from PIL import Image

    strip = ist.filmstrip(pipe, PROMPT, seed=7, steps=4, at=[0, 3])
    for frame in strip.frames:
        assert frame.png[:8] == b"\x89PNG\r\n\x1a\n"
        opened = Image.open(io.BytesIO(frame.png))
        assert opened.size == (frame.width, frame.height)
        assert (frame.decoded_width, frame.decoded_height) == (32, 32)
        assert frame.latent_rms is not None and frame.latent_rms > 0


def test_the_frames_cross_a_wire_as_base64_without_becoming_bytes(pipe):
    """The payload goes to a browser. `bytes` is not JSON, and an empty string
    where a picture should be renders as a broken image rather than as an
    absence."""
    payload = ist.filmstrip(pipe, PROMPT, seed=7, steps=4, at=[0, 3]).to_dict()
    revived = json.loads(json.dumps(payload))
    first = revived["frames"][0]
    assert first["png"].startswith("data:image/png;base64,")
    raw = base64.b64decode(first["png"].split(",", 1)[1])
    assert raw[:4] == b"\x89PNG"
    assert len(raw) == first["png_bytes"]
    assert revived["png_bytes_total"] == sum(f["png_bytes"] for f in revived["frames"])


def test_a_frame_with_no_bytes_is_null_rather_than_an_empty_picture():
    """`""` in an `<img src>` is a broken-image icon, which reads as a decode
    that produced black rather than one that never happened."""
    assert ist.Frame(step=0).to_dict()["png"] is None
    assert ist.Frame(step=0).to_dict()["png_bytes"] == 0


def test_the_emitted_resolution_is_bounded_and_the_shrink_is_reported(pipe):
    """A frame silently shrunk is a picture of a resolution the model never
    worked at — and `frame_pixels` bounds the RESPONSE, never the decode, which
    the sentence has to say before somebody reaches for it to fix an OOM."""
    strip = ist.filmstrip(
        pipe, PROMPT, seed=7, steps=4, at=[0, 3], height=64, width=64, frame_pixels=32
    )
    frame = strip.frames[0]
    assert (frame.decoded_width, frame.decoded_height) == (64, 64)
    assert (frame.width, frame.height) == (32, 32)
    assert frame.downsampled is True
    assert frame.to_dict()["downsampled"] is True
    said = strip.means()
    assert "the decoder produced 64x64 and the frames are emitted at 32x32" in said
    assert "bounds this RESPONSE and not the decode" in said


def test_a_frame_that_was_not_shrunk_does_not_claim_it_was(pipe):
    """The other half of the same honesty: `downsampled: true` on an untouched
    picture would send somebody hunting for detail that is all there."""
    strip = ist.filmstrip(pipe, PROMPT, seed=7, steps=4, at=[3], frame_pixels=384)
    assert strip.frames[0].downsampled is False
    assert "Nothing was resized" in strip.means()


def test_the_frames_can_be_written_to_disk_named_by_their_step(pipe, tmp_path):
    """ "Saves of steps" is the ask. The filename is the STEP, not the position
    in the strip, so a directory listing of 000, 003, 006 is visibly a subset
    where 000, 001, 002 would look like a three-step run."""
    strip = ist.filmstrip(pipe, PROMPT, seed=7, steps=8, every=3)
    written = strip.save(tmp_path / "strip")
    names = sorted(p.name for p in (tmp_path / "strip").iterdir())
    assert names == ["step_000.png", "step_003.png", "step_006.png", "step_007.png"]
    assert len(written) == 4
    assert (tmp_path / "strip" / "step_000.png").read_bytes() == strip.frames[0].png


def test_saving_onto_a_file_is_refused_rather_than_overwriting_it(pipe, tmp_path):
    """A directory is what this writes into. Told to write into a file, the
    honest answer is no rather than four failed writes."""
    target = tmp_path / "not-a-directory"
    target.write_text("mine")
    strip = ist.filmstrip(pipe, PROMPT, seed=7, steps=4, at=[3])
    with pytest.raises(BadRequest, match="rather than a directory"):
        strip.save(target)
    assert target.read_text() == "mine"


# ------------------------------------------------------- the sibling's claim


def test_the_trace_still_decodes_nothing_now_that_the_filmstrip_exists(pipe):
    """The claim this feature had to leave intact. `vae_decodes: 0` up there is
    checkable by counting, and a filmstrip that reached into `trace` — or a
    wrapper left behind by one — would quietly turn it into a lie."""
    calls = []
    pipe.vae.decode = _counting_decode(pipe, calls)
    try:
        run = ist.trace(pipe, PROMPT, seed=7, steps=4)
        assert calls == [], "the trace decoded something"
        strip = ist.filmstrip(pipe, PROMPT, seed=7, steps=4, at=[3])
        after = ist.trace(pipe, PROMPT, seed=7, steps=4)
    finally:
        del pipe.vae.decode

    assert run.vae_decodes == 0
    assert after.vae_decodes == 0
    assert strip.vae_decodes == 1
    assert len(calls) == 1, "only the strip's single frame was decoded"


def test_the_strip_does_not_leave_a_wrapper_on_somebody_elses_vae(pipe):
    """A counter left attached would keep counting for every later generation in
    the process, and the next strip would report double."""
    ist.filmstrip(pipe, PROMPT, seed=7, steps=4, at=[3])
    assert "decode" not in vars(pipe.vae)
    again = ist.filmstrip(pipe, PROMPT, seed=7, steps=4, at=[3])
    assert again.vae_decodes == 1, "a stacked wrapper would count twice"


def test_a_failed_strip_leaves_the_pipeline_exactly_as_it_found_it(pipe):
    """The recovery path. A run that raised halfway must not poison the next
    one, and the wrapper is on a module the caller owns."""
    with pytest.raises(ValueError, match="divisible by 8"):
        ist.filmstrip(pipe, PROMPT, seed=7, steps=4, at=[3], height=7, width=7)
    assert "decode" not in vars(pipe.vae)
    assert len(ist.filmstrip(pipe, PROMPT, seed=7, steps=4, at=[3]).frames) == 1


def test_a_wrapper_that_was_already_there_is_put_back_not_deleted(pipe):
    """Something else may already be wrapping `decode` — a profiler, a test.
    Restoring state means restoring THAT, not removing it."""
    calls = []
    mine = _counting_decode(pipe, calls)
    pipe.vae.decode = mine
    try:
        ist.filmstrip(pipe, PROMPT, seed=7, steps=4, at=[3])
        assert pipe.vae.decode is mine
        assert calls == [1]
    finally:
        del pipe.vae.decode


# ------------------------------------------------------------- what it is not


def test_the_caveat_travels_with_the_frames(pipe):
    """Now that frames exist somebody WILL compare them, and a pixel difference
    between two decoded frames is a property of the VAE as much as of the
    denoiser — which is the whole argument `trace` makes for not decoding."""
    said = ist.filmstrip(pipe, PROMPT, seed=7, steps=4, at=[0, 3]).means()
    assert "LATENT DISTANCE IS NOT VISIBLE DIFFERENCE, AND NOW BOTH EXIST" in said
    assert "a property of this VAE as much as of the denoiser" in said
    assert "per-step movement is its job and is not recomputed here" in said


def test_the_result_says_a_frame_is_not_the_models_guess_at_the_finished_image(pipe):
    """The reading that makes an early frame look like a broken model. This
    decodes the RUNNING latent, noise and all; the scheduler's prediction of the
    clean sample is a different tensor and is not what is drawn here."""
    said = ist.filmstrip(pipe, PROMPT, seed=7, steps=4, at=[0, 3]).means()
    assert "WHAT A FRAME AT STEP k IS NOT" in said
    assert "RUNNING latent" in said
    assert "is the latent, not the model failing" in said


def test_the_decode_arithmetic_is_published_rather_than_assumed(pipe):
    """A frame's contrast comes from this VAE's own scaling factor. Hard-coding
    Stable Diffusion's 0.18215 would mis-decode every other VAE, so the number
    used is read off the config and reported beside the picture."""
    strip = ist.filmstrip(pipe, PROMPT, seed=7, steps=4, at=[3])
    assert strip.latent_scaling == pytest.approx(pipe.vae.config.scaling_factor)
    assert strip.latent_shift is None
    assert "scaling factor 0.18215" in strip.means()
    assert "(x / 2 + 0.5)" in strip.means()


def test_a_run_that_could_not_be_reproduced_says_so(pipe):
    """ "No seed" and "seed 0" are different runs, and only one of them will
    ever produce these pictures again."""
    strip = ist.filmstrip(pipe, PROMPT, seed=None, steps=4, at=[3])
    assert strip.seed is None
    assert "NO SEED WAS FIXED" in strip.means()
    with pytest.raises(TypeError, match="seed"):
        ist.filmstrip(pipe, PROMPT, steps=4, at=[3])


def test_the_same_seed_gives_the_same_pictures(pipe):
    """Without this a filmstrip is an illustration rather than a measurement."""
    a = ist.filmstrip(pipe, PROMPT, seed=7, steps=4, at=[0, 3])
    b = ist.filmstrip(pipe, PROMPT, seed=7, steps=4, at=[0, 3])
    assert [f.png for f in a.frames] == [f.png for f in b.frames]
    c = ist.filmstrip(pipe, PROMPT, seed=8, steps=4, at=[0, 3])
    assert [f.png for f in c.frames] != [f.png for f in a.frames]


# ------------------------------------------------------------------- memory


def test_the_strip_holds_one_latent_per_frame_and_releases_them_as_it_decodes(pipe):
    """Never all the frames as tensors. The host figure is the peak the held
    latents really reached, not the zero the object holds afterwards."""
    strip = ist.filmstrip(pipe, PROMPT, seed=7, steps=8, every=3)
    # 4 frames x 4 channels x 32 x 32 x 4 bytes, on the pipeline this fixture
    # built — the same arithmetic `filmstrip_plan` prices without running.
    assert strip.host_latent_bytes == 4 * 16384
    assert strip.latent_shape == (4, 32, 32)


def test_a_strip_too_large_to_hold_is_refused_at_its_first_selected_step(
    pipe, monkeypatch
):
    """After one held latent, not after all of them, and with the two parameters
    that would fix it in the sentence."""
    monkeypatch.setattr(ist, "MAX_TRACE_BYTES", 1024)
    with pytest.raises(BadRequest) as caught:
        ist.filmstrip(pipe, PROMPT, seed=7, steps=8, every=3)
    said = str(caught.value)
    assert "(4, 32, 32) latent" in said
    assert "fewer frames" in said
    assert "filmstrip_plan()" in said


def test_the_peak_is_null_rather_than_zero_when_it_cannot_be_measured(pipe):
    """The CPU has no allocator high-water mark. Reporting 0 would claim the
    decodes were free, which is the opposite of what was measured."""
    strip = ist.filmstrip(pipe, PROMPT, seed=7, steps=4, at=[3])
    assert strip.peak_device_bytes is None
    assert strip.peak_source is None
    assert "no allocator high-water mark" in strip.peak_unmeasured
    assert "Reported as null rather than as zero" in strip.means()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA on this machine")
def test_the_peak_is_a_real_number_where_the_allocator_publishes_one(pipe):
    """The other half: on a card, the peak has to be observed and reported —
    "how much did this cost me" is the question a filmstrip provokes first."""
    pipe.to("cuda")
    try:
        strip = ist.filmstrip(pipe, PROMPT, seed=7, steps=4, at=[0, 3])
    finally:
        pipe.to("cpu")
    assert strip.peak_device_bytes > 0
    assert strip.peak_source == "torch.cuda.max_memory_allocated"
    assert strip.peak_device.startswith("cuda")
    assert strip.peak_unmeasured is None
    assert "high-water mark on cuda" in strip.means()
    assert "not the driver's" in strip.means()


# --------------------------------------------------------- choosing the subset
#
# Pure: no pipeline, no torch, no download. Which steps get decoded is the most
# important thing the strip reports, so it is checkable on any machine.


def test_the_caller_chooses_the_subset_and_this_module_never_does():
    """A default subset would be this module deciding what a reader sees, and
    "all of them" is the cost the whole feature exists not to pay."""
    with pytest.raises(BadRequest, match="either both ways of choosing it or neither"):
        ist.select_steps(20)
    with pytest.raises(BadRequest, match="either both ways of choosing it or neither"):
        ist.select_steps(20, every=4, at=[1])


def test_every_nth_step_carries_the_last_one_unless_told_not_to():
    """A strip that stops one step short of the finished picture is missing the
    one frame everybody assumes is there — and the decision is an argument, not
    a rule, so it can be turned off."""
    assert ist.select_steps(20, every=4) == [0, 4, 8, 12, 16, 19]
    assert ist.select_steps(20, every=4, include_final=False) == [0, 4, 8, 12, 16]
    assert ist.select_steps(21, every=4) == [0, 4, 8, 12, 16, 20]


def test_an_explicit_list_is_sorted_and_deduplicated_rather_than_taken_twice():
    """Asking for step 5 twice is one frame, and decoding it twice would pay for
    a picture the caller already has."""
    assert ist.select_steps(6, at=[5, 0, 0, 5]) == [0, 5]
    assert ist.select_steps(6, at=[3], include_final=False) == [3]


def test_a_step_outside_the_run_is_refused_by_name():
    """ "Step 40 of a 20-step run" is a request nothing can satisfy, and the
    refusal names the range rather than clamping into it."""
    with pytest.raises(BadRequest, match="is not in a 20-step run"):
        ist.select_steps(20, at=[40])
    with pytest.raises(BadRequest, match="is not in a 20-step run"):
        ist.select_steps(20, at=[-1])


def test_the_step_before_the_first_one_does_not_exist_here():
    """The run's starting noise is never handed over, so there is no index for
    it — the same gap `trace` reports as a null first change."""
    with pytest.raises(BadRequest) as caught:
        ist.select_steps(20, at=[-1])
    assert "no index here" in str(caught.value)


def test_a_bool_is_not_a_step_index_and_not_a_spacing():
    """isinstance(True, int) is True, so `at=[True]` would decode step 1 and
    `every=True` would decode every step."""
    with pytest.raises(BadRequest, match="a whole number"):
        ist.select_steps(20, at=[True])
    with pytest.raises(BadRequest, match="whole number of steps between frames"):
        ist.select_steps(20, every=True)
    with pytest.raises(BadRequest, match="yes or no"):
        ist.select_steps(20, every=4, include_final=1)


def test_an_empty_selection_is_refused_rather_than_returning_an_empty_strip():
    """An empty strip is not a cheaper strip — it is a call with nothing in
    it."""
    with pytest.raises(BadRequest, match="named no steps"):
        ist.select_steps(20, at=[])
    with pytest.raises(BadRequest, match="list of step indices"):
        ist.select_steps(20, at=5)


def test_too_many_frames_is_refused_with_the_spacing_that_would_fit():
    """Truncating to twelve would produce exactly the thing this feature must
    never produce: a strip whose gaps nobody was told about. So it refuses, and
    the sentence carries the `every` that fits."""
    with pytest.raises(BadRequest) as caught:
        ist.select_steps(50, every=1)
    said = str(caught.value)
    assert f"past the {ist.MAX_FRAMES} this decodes" in said
    assert "Raise `every` to at least 5" in said
    assert len(ist.select_steps(50, every=5)) <= ist.MAX_FRAMES

    with pytest.raises(BadRequest, match=f"Name at most {ist.MAX_FRAMES}"):
        ist.select_steps(50, at=list(range(ist.MAX_FRAMES + 1)))


def test_the_filmstrip_floor_is_one_step_and_the_ceiling_is_the_traces():
    """A two-step distilled run is a perfectly watchable two frames — the trace
    refuses it because a COMMIT STEP needs three, which is arithmetic this does
    not do. The ceiling is shared, because a truncated schedule is a different
    run in both."""
    assert ist.select_steps(2, every=1) == [0, 1]
    with pytest.raises(BadRequest, match="at least one denoising step"):
        ist.select_steps(0, every=1)
    with pytest.raises(BadRequest, match="different run wearing the number"):
        ist.select_steps(ist.MAX_STEPS + 1, every=100)
    with pytest.raises(BadRequest, match="whole number of denoising steps"):
        ist.select_steps(True, every=1)


def test_the_emitted_resolution_is_refused_outside_its_range():
    """Both ends are wrong for a stated reason, and neither is clamped."""
    assert ist.check_frame_pixels(ist.DEFAULT_FRAME_PIXELS) == 384
    for bad in (0, 8, ist.MAX_FRAME_PIXELS + 1, True):
        with pytest.raises(BadRequest):
            ist.check_frame_pixels(bad)
    with pytest.raises(BadRequest, match="bounds the response either way"):
        ist.check_frame_pixels(4096)


def test_a_run_of_ranges_is_compressed_without_dropping_a_step():
    """The prose says "1-3, 5" so a 190-step gap does not become an ellipsis
    that hides which steps were skipped."""
    assert ist._ranges([1, 2, 3, 5]) == "1-3, 5"
    assert ist._ranges([]) == "none"
    assert ist._ranges([4, 4, 0]) == "0, 4"


# --------------------------------------------------------------- pricing it


def test_the_strip_is_priced_before_it_is_spent():
    """No model needed — the point of pricing before loading."""
    priced = ist.filmstrip_plan(50, every=8, latent_shape=(4, 64, 64))
    assert priced["frames"] == 8
    assert priced["decoded_steps"] == [0, 8, 16, 24, 32, 40, 48, 49]
    assert len(priced["skipped_steps"]) == 42
    assert priced["denoiser_passes"] == 50
    assert priced["vae_decodes"] == 8
    assert priced["vae_decodes_if_pipeline_also_decodes"] == 9
    assert priced["latent_bytes"] == 65536
    assert priced["total_bytes"] == 8 * 65536
    assert priced["fits"] is True
    assert "No seconds are quoted" in priced["means"]


def test_the_preflight_prices_the_frames_the_run_actually_decoded(pipe):
    """A predicted cost that does not match the real run is not a prediction."""
    priced = ist.filmstrip_plan(8, every=3, latent_shape=ist.latent_shape_of(pipe))
    strip = ist.filmstrip(pipe, PROMPT, seed=7, steps=8, every=3)
    assert priced["frames"] == len(strip.frames)
    assert priced["decoded_steps"] == strip.decoded_steps
    assert priced["skipped_steps"] == strip.skipped_steps
    assert priced["total_bytes"] == strip.host_latent_bytes
    assert priced["vae_decodes"] == strip.vae_decodes


def test_the_preflight_refuses_exactly_what_the_run_would():
    """A preflight that accepts a call the call rejects is a promise the next
    request breaks."""
    with pytest.raises(BadRequest, match="past the"):
        ist.filmstrip_plan(50, every=1)
    with pytest.raises(BadRequest, match="is not in a 20-step run"):
        ist.filmstrip_plan(20, at=[99])
    with pytest.raises(BadRequest, match="different run wearing the number"):
        ist.filmstrip_plan(ist.MAX_STEPS + 1, every=100)
    with pytest.raises(BadRequest, match="either both ways"):
        ist.filmstrip_plan(20)
    with pytest.raises(BadRequest, match="outside the 32 to 768"):
        ist.filmstrip_plan(20, every=4, frame_pixels=4096)


def test_what_the_preflight_cannot_know_is_null_rather_than_invented():
    """PNG size depends on the picture and the peak depends on the machine.
    Neither is a zero, and neither is a guess."""
    priced = ist.filmstrip_plan(20, every=4)
    assert priced["png_bytes"] is None
    assert priced["peak_device_bytes"] is None
    assert priced["latent_bytes"] is None
    assert priced["total_bytes"] is None
    assert priced["fits"] is None
    assert "cannot be priced" in priced["means"]
    assert "null rather than invented" in priced["means"]


def test_the_preflight_says_the_frame_bound_is_not_a_memory_bound():
    """The mistake this sentence stops: reaching for `frame_pixels` to make a
    decode fit that did not. It shrinks the response, after the decode."""
    said = ist.filmstrip_plan(20, every=4)["means"]
    assert "BOUNDS THE RESPONSE, NOT THE DECODE" in said
    assert "smaller height and width" in said


def test_a_strip_that_would_not_fit_in_host_memory_says_so_first():
    """A latent with a time axis is the shape this was not written for, and
    finding out by running out of memory is not a bound. (16, 61, 90, 160) is
    53.6 MiB a step, so eight of them is 429 MiB against the 256 this holds.)"""
    priced = ist.filmstrip_plan(50, every=8, latent_shape=(16, 61, 90, 160))
    assert priced["fits"] is False
    assert "past the" in priced["means"]
    assert "fewer frames" in priced["means"]


# ------------------------------------------------------------------ refusals


def test_a_transformer_denoiser_is_filmed_exactly_like_a_unet(tiny_vae):
    """`imaging.py` distinguishes the two families, and this must not. The
    denoiser lives in the `transformer` slot on a DiT and the latents are the
    same four-dimensional thing, so there is nothing here to special-case."""
    torch.manual_seed(1)
    latents = [torch.randn(1, 4, 8, 8) for _ in range(4)]
    strip = ist.filmstrip(_Stub(tiny_vae, latents), PROMPT, seed=7, steps=4, every=2)
    assert strip.decoded_steps == [0, 2, 3]
    assert strip.scheduler == "_Scheduler"
    assert len(strip.frames) == 3
    assert all(f.png[:4] == b"\x89PNG" for f in strip.frames)
    assert strip.vae_decodes == 3


def test_a_packed_latent_is_refused_rather_than_decoded_as_a_scrambled_picture(
    tiny_vae,
):
    """A packed-sequence latent has no height and width until the pipeline's own
    arithmetic unpacks it, and a wrong guess decodes to something that still
    looks like a frame — which is worse than no frame at all."""
    latents = [torch.randn(1, 64, 16) for _ in range(2)]
    with pytest.raises(ist.NotSupported) as caught:
        ist.filmstrip(_Stub(tiny_vae, latents), PROMPT, seed=7, steps=2, every=1)
    said = str(caught.value)
    assert "packed into a sequence" in said
    assert "`trace()` measures this run" in said


def test_a_video_latent_is_refused_because_a_step_is_a_clip_not_a_picture(tiny_vae):
    """ "The picture at step k" has no answer when the latent has a time axis,
    and picking one of its frames would be this module choosing which."""
    latents = [torch.randn(1, 4, 3, 8, 8) for _ in range(2)]
    with pytest.raises(ist.NotSupported, match="time axis"):
        ist.filmstrip(_Stub(tiny_vae, latents), PROMPT, seed=7, steps=2, every=1)


def test_a_batch_of_images_is_refused_rather_than_tiled(tiny_vae):
    """Every step would decode to several pictures, and a filmstrip has one
    axis."""
    latents = [torch.randn(3, 4, 8, 8) for _ in range(2)]
    with pytest.raises(ist.NotSupported, match="one image at a time"):
        ist.filmstrip(_Stub(tiny_vae, latents), PROMPT, seed=7, steps=2, every=1)


def test_a_diverged_latent_is_refused_rather_than_decoded_into_a_picture(tiny_vae):
    """A NaN latent decodes to something, and that something would be read as
    the model's own output. The refusal happens before any of it is drawn."""
    latents = [torch.full((1, 4, 8, 8), float("nan")) for _ in range(2)]
    with pytest.raises(ist.NotMeasurable, match="not finite"):
        ist.filmstrip(_Stub(tiny_vae, latents), PROMPT, seed=7, steps=2, every=1)


def test_a_filmstrip_of_a_pipeline_that_hides_its_latents_is_refused(tiny_vae):
    """Rebuilding them from the denoiser's input would mean undoing whatever
    scaling the scheduler applied — and a picture made of the wrong tensor still
    looks like a picture."""

    class Hidden(_Stub):
        def __call__(self, prompt, *, callback_on_step_end, **kwargs):
            callback_on_step_end(self, 0, 9.0, {})
            return None

    with pytest.raises(ist.NotSupported, match="nothing here to decode"):
        ist.filmstrip(Hidden(tiny_vae, []), PROMPT, seed=7, steps=4, at=[0])


def test_a_pipeline_with_no_vae_is_refused_by_name():
    """A pixel-space diffusion model denoises the image directly and needs a
    different reader — and `trace` still measures it without decoding."""

    class NoVae:
        unet = object()

    with pytest.raises(ist.NotSupported, match="no VAE to decode with"):
        ist.filmstrip(NoVae(), PROMPT, seed=7, steps=4, at=[3])


def test_a_filmstrip_of_a_pipeline_with_no_denoiser_is_refused():
    class Nothing:
        pass

    with pytest.raises(ist.NotSupported, match="no `unet` or `transformer`"):
        ist.filmstrip(Nothing(), PROMPT, seed=7, steps=4, at=[3])


def test_a_pipeline_that_exposes_no_step_callback_is_refused_before_it_runs():
    """A class-conditioned DiT is the real example: no `callback_on_step_end`
    anywhere in its signature, so the intermediate latents are never exposed and
    there is nothing to film. It refuses by name rather than letting somebody
    else's TypeError reach the reader as "something broke"."""

    class NoCallback:
        transformer = object()
        vae = object()

        def __call__(self, class_labels, num_inference_steps=50):
            return None

    with pytest.raises(ist.NotSupported) as caught:
        ist.filmstrip(NoCallback(), PROMPT, seed=7, steps=4, at=[3])
    said = str(caught.value)
    assert "does not accept `callback_on_step_end`" in said
    assert "call surface rather than the architecture" in said


def test_a_pipeline_that_takes_no_prompt_is_refused_rather_than_mis_conditioned():
    """Handing a prompt to a class-conditioned model conditions on nothing the
    caller chose, and the frames would be of some other image entirely."""

    class NoPrompt:
        transformer = object()
        vae = object()

        def __call__(
            self, class_labels, num_inference_steps=50, callback_on_step_end=None
        ):
            return None

    with pytest.raises(ist.NotSupported, match="not conditioned on text"):
        ist.filmstrip(NoPrompt(), PROMPT, seed=7, steps=4, at=[3])


def test_a_pipeline_that_swallows_its_arguments_is_not_refused_on_a_guess(tiny_vae):
    """A `__call__` taking `**kwargs` proves nothing about what it accepts, so
    absence is not evidence of absence — refusing there would turn every wrapper
    around a working pipeline into a "not supported"."""
    assert ist._call_surface(_Stub(tiny_vae, [])) is None
    assert ist._call_surface(object()) is None
    latents = [torch.randn(1, 4, 8, 8) for _ in range(2)]
    assert (
        len(
            ist.filmstrip(
                _Stub(tiny_vae, latents), PROMPT, seed=7, steps=2, every=1
            ).frames
        )
        == 2
    )


def test_a_run_that_hands_over_nothing_cannot_be_filmed(tiny_vae):
    """An empty strip returned as a success would read as a model that produced
    nothing, rather than as a callback that was never reached."""
    with pytest.raises(ist.NotSupported, match="without handing over a single step"):
        ist.filmstrip(_Stub(tiny_vae, []), PROMPT, seed=7, steps=4, at=[0])


def test_a_filmstrip_seed_that_is_not_a_whole_number_is_refused(pipe):
    with pytest.raises(BadRequest, match="whole number"):
        ist.filmstrip(pipe, PROMPT, seed=7.5, steps=4, at=[3])
