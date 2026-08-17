"""Cross-attention out of a real diffusion model.

The pipeline here is small and REAL: a genuine `UNet2DConditionModel` with
genuine `CrossAttnDownBlock2D` / `CrossAttnUpBlock2D` blocks, a real
`AutoencoderKL`, a real `CLIPTextModel`. It is built in-process rather than
downloaded, so the tests need no network and no gigabytes — but every module
under test is the one a real Stable Diffusion checkpoint uses, and the
attention is computed by diffusers' own code.

The weights are random, so the maps mean nothing about the world. That is
fine and deliberate: what is under test is that the capture reaches the right
blocks, keeps the step axis, labels the columns with real tokens, finds where
the padding starts, restores the pipeline afterwards, and reproduces under a
seed. None of those is a claim about what a trained model attends to.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("transformers")

from modelmri import image_attention as ia  # noqa: E402
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
    try:
        tokenizer = CLIPTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-clip"
        )
    except Exception:
        pytest.skip("the tiny CLIP tokenizer is not cached and there is no network")

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


# ------------------------------------------------------------- the capture


def test_every_step_is_captured_separately(pipe):
    """The step axis is the whole point. Early steps decide layout and late
    steps decide texture; one averaged map hides that completely."""
    run = ia.capture(pipe, PROMPT, steps=4, seed=7)
    assert len(run.steps) == 4
    assert [s.step for s in run.steps] == [0, 1, 2, 3]
    # Timesteps must DECREASE — that is what denoising is, and an increasing
    # or constant sequence would mean the callback is not seeing the schedule.
    times = [s.timestep for s in run.steps]
    assert times == sorted(times, reverse=True), times


def test_only_cross_attention_blocks_contribute(pipe):
    """This UNet has one CrossAttnDownBlock2D and one CrossAttnUpBlock2D. Self-
    attention among latents answers a different question, and averaging the
    two together would combine quantities that are not the same quantity."""
    run = ia.capture(pipe, PROMPT, steps=2, seed=7)
    for step in run.steps:
        assert step.blocks > 0
    # Every step sees the same blocks, because the architecture does not change
    # between steps. A varying count means captures are being dropped.
    assert len({s.blocks for s in run.steps}) == 1


def test_the_columns_are_real_tokens_not_indices(pipe):
    """An attention map whose columns are integers is not a map of words."""
    run = ia.capture(pipe, PROMPT, steps=2, seed=7)
    assert run.tokens[0].startswith("<|startoftext|>")
    assert any("ed</w>" in t or "red" in t for t in run.tokens[:8]), run.tokens[:8]
    assert len(run.steps[0].per_token) == len(run.tokens)


def test_padding_is_reported_rather_than_plotted_as_words(pipe):
    """CLIP pads to 77 and the padding attracts real attention mass. That is a
    genuine finding and a terrible chart, so it travels as an index."""
    run = ia.capture(pipe, PROMPT, steps=2, seed=7)
    assert 0 < run.padding_from < len(run.tokens)
    assert "padding" in run.means()


def test_the_same_seed_reproduces_the_same_maps(pipe):
    a = ia.capture(pipe, PROMPT, steps=3, seed=7)
    b = ia.capture(pipe, PROMPT, steps=3, seed=7)
    assert a.steps[0].per_token == b.steps[0].per_token
    assert a.steps[-1].per_token == b.steps[-1].per_token


def test_an_unseeded_run_says_it_is_unseeded(pipe):
    run = ia.capture(pipe, PROMPT, steps=2)
    assert run.seed is None
    assert "No seed was fixed" in run.means()


def test_the_result_says_attention_is_not_a_cause(pipe):
    """The project's own standard. A heatmap is correlational and the sentence
    that says so has to travel with it, not sit in the docs."""
    said = ia.capture(pipe, PROMPT, steps=2, seed=7).means()
    assert "ATTENTION IS NOT A CAUSE" in said
    assert "knockout" in said


def test_the_pipeline_is_restored_after_a_capture(pipe):
    """A pipeline left with capturing processors attached keeps allocating
    maps for every later generation in the process — a memory leak whose
    symptom is a slow OOM nobody connects to a panel they opened once."""
    before = dict(pipe.unet.attn_processors)
    ia.capture(pipe, PROMPT, steps=2, seed=7)
    after = pipe.unet.attn_processors
    assert set(before) == set(after)
    for name, processor in after.items():
        assert not isinstance(processor, ia._Capturing), name


def test_the_pipeline_is_restored_even_when_the_run_fails(pipe):
    """`finally`, not a happy-path cleanup. A generation that raises must not
    leave the model instrumented."""
    before = dict(pipe.unet.attn_processors)
    # A size the VAE cannot downsample. diffusers raises `ValueError:
    # "height" and "width" have to be divisible by 8 but are 7 and 7` — named
    # exactly rather than caught as a blind `Exception`, so this keeps testing
    # the restore rather than quietly passing on some future unrelated error.
    with pytest.raises(ValueError, match="divisible by 8"):
        ia.capture(pipe, PROMPT, steps=2, seed=7, height=7, width=7)
    for name, processor in pipe.unet.attn_processors.items():
        assert not isinstance(processor, ia._Capturing), name
    assert set(before) == set(pipe.unet.attn_processors)


def test_more_steps_than_the_ceiling_are_reported_not_silently_dropped(pipe):
    """A cap that does not say what it dropped reads as "this is the whole
    run"."""
    run = ia.capture(pipe, PROMPT, steps=ia.MAX_STEPS + 10, seed=7)
    assert run.steps_requested == ia.MAX_STEPS + 10
    assert len(run.steps) <= ia.MAX_STEPS
    assert "were not captured" in run.means()


def test_zero_steps_is_a_refusal(pipe):
    with pytest.raises(BadRequest, match="at least one"):
        ia.capture(pipe, PROMPT, steps=0)


# ------------------------------------------------------------- refusals


def test_a_pipeline_with_no_denoiser_is_refused():
    class Nothing:
        pass

    with pytest.raises(ia.NotSupported, match="no `unet` or `transformer`"):
        ia.capture(Nothing(), PROMPT, steps=1)


def test_a_denoiser_that_cannot_be_instrumented_is_refused():
    """Reconstructing attention from hidden states afterwards is a DIFFERENT
    quantity from the one the model used, so this refuses rather than
    approximating."""

    class Pipe:
        unet = object()

    with pytest.raises(ia.NotSupported, match="set_attn_processor"):
        ia.capture(Pipe(), PROMPT, steps=1)


# ------------------------------------------------------------- knockout


def test_a_one_word_prompt_cannot_be_knocked_out(pipe):
    """Removing the only word measures the unconditional model, not the effect
    of a word."""
    with pytest.raises(BadRequest, match="nothing to knock out"):
        ia.knockout(pipe, "cube", tokens=[], seed=7, steps=2)


def test_knockout_runs_every_arm_at_the_identical_seed(pipe):
    """The seed is doing the work. At a different seed per arm these numbers
    would be sampling noise with a word's name on them."""
    out = ia.knockout(pipe, "a red cube", tokens=[], seed=7, steps=2)
    assert {a["word"] for a in out["arms"]} == {"a", "red", "cube"}
    assert out["seed"] == 7
    assert "THE SEED IS DOING THE WORK" in out["means"]
    assert "identical seed" in out["means"]


def test_knockout_says_removing_a_word_is_not_silencing_it(pipe):
    """The remaining prompt is a different, shorter sentence and the model
    conditions on all of it — so a big number is not quite 'that word caused
    this'."""
    out = ia.knockout(pipe, "a red cube", tokens=[], seed=7, steps=2)
    assert "NOT SILENCING IT" in out["means"]


def test_knockout_rows_are_sorted_by_effect(pipe):
    out = ia.knockout(pipe, "a red cube", tokens=[], seed=7, steps=2)
    distances = [a["distance"] for a in out["arms"]]
    assert distances == sorted(distances, reverse=True)


# ------------------------------------------------------------- planning


def test_the_cost_is_known_before_it_is_spent():
    """No model needed — which is the point of pricing before loading."""
    plan = ia.plan(steps=20, words=6)
    assert plan["arms"] == 7
    assert plan["passes"] == 140
    assert "No seconds are quoted" in plan["means"]
