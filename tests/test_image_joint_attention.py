# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""Joint-attention denoisers refuse before the generation, not after it.

`imaging` calls SD3, Flux, AuraFlow and CogVideoX DiT-shaped, which is true,
and DiT-shaped carried `cross_attention` on the strength of the family name.
But `capture` only instruments blocks named `attn2` — diffusers' name for the
block whose KEYS are the prompt — and an MM-DiT has none: it concatenates the
text and the image into one sequence and attends over the pair. So every one
of those checkpoints ran a full generation, captured nothing, and was told
"this denoiser may attend to its conditioning somewhere this does not reach"
after the reader had paid for twenty denoising passes.

The order is the whole fix, and it is the order every other refusal on this
side already uses: `attn_processors` is a walk of `named_children`, so the
fact that decides this is free and available the instant the pipeline object
exists. Nothing here spends a pass to learn something a lookup answers.

## Why the check cannot be "does an attn2 key exist"

SD 3.5 Large sets `dual_attention_layers`, and those blocks DO carry an
`attn2` — a plain SELF-attention over the image stream, `cross_attention_dim
= None`, called with no `encoder_hidden_states`. A name-only test says yes,
the capability survives, and the reader pays for the same empty capture. So
the site test asks the MODULE whether it is a cross-attention, which is
diffusers' own `is_cross_attention` flag, and falls back to keeping the
capability whenever it cannot look.

## The fakes are the real key strings

No downloads. The processor keys here — `transformer_blocks.0.attn.processor`
and friends — are the strings `AttentionMixin.attn_processors` really
produces, and `test_the_verified_table_still_matches_diffusers` builds every
class in the table that can be built and checks that claim against the
installed library rather than against this file's memory of it.

## The tokenizer on the fake pipeline is load-bearing

`capture` refuses for want of a tokenizer several lines BELOW the preflight,
so a fake without one can never reach a generation and "zero passes were
spent" is true of it no matter what the implementation does. Every fixture
that asserts a cost therefore carries a `_Tokenizer`, and
`test_a_pipeline_with_a_tokenizer_really_does_spend_a_pass` is the control
that proves the fixture can be reached.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from modelmri import image_attention as ia


class _Attention:
    """One attention module, as far as the site walk is concerned.

    `is_cross_attention` is diffusers' own flag, set in `Attention.__init__`
    as `cross_attention_dim is not None`. Nothing else about the module
    matters to the walk.
    """

    def __init__(self, cross: bool) -> None:
        self.is_cross_attention = cross


class _Denoiser:
    """A denoiser with a stated set of attention blocks and nothing else.

    `attn_processors` and `named_modules` are the two surfaces the site walk
    reads, and both are real diffusers shapes: keys are module paths with
    `.processor` appended, and `named_modules` yields `(path, module)`.
    """

    def __init__(self, blocks: dict) -> None:
        # {module path: is_cross_attention}
        self._blocks = dict(blocks)
        self.installed = None
        #: EVERY call, in order. `capture` installs the wrapped processors and
        #: restores the originals in a `finally`, so the last value says
        #: nothing about whether the pipeline was ever instrumented — which is
        #: the fact `…before_the_pipeline_is_instrumented` is about.
        self.installs: list[dict] = []

    @property
    def attn_processors(self) -> dict:
        return {f"{path}.processor": object() for path in self._blocks}

    def named_modules(self):
        return [(path, _Attention(cross)) for path, cross in self._blocks.items()]

    def set_attn_processor(self, processors) -> None:
        self.installed = processors
        self.installs.append(processors)


class _Tokenizer:
    """Enough of a tokenizer for `_tokenize` to succeed on.

    WITHOUT this the fake pipeline cannot reach a generation at all: `capture`
    refuses for want of a tokenizer several lines below the preflight, so
    `__call__` below is never entered and `passes == []` holds no matter where
    the preflight sits — or whether it exists. An assertion that cannot fail
    is not a receipt, and "the refusal costs zero passes" is the whole claim
    of this workstream. `test_a_pipeline_with_a_tokenizer_really_does_spend_a_pass`
    is the control that proves this fake is adequate to make it bite.

    The two shapes `_tokenize` asks for: a call with `padding="max_length"`
    whose `input_ids` are padded to `model_max_length`, and a bare call with
    `truncation=True` whose length says where the real tokens stop.
    """

    model_max_length = 8

    def __call__(self, prompt, **kwargs) -> dict:
        ids = list(range(1, len(prompt.split()) + 1))
        if kwargs.get("padding") == "max_length":
            ids = (ids + [0] * self.model_max_length)[: self.model_max_length]
        return {"input_ids": ids}

    def convert_ids_to_tokens(self, ids):
        return [f"w{i}" if i else "<pad>" for i in ids]


class _Pipe:
    """A pipeline that counts the passes anybody spends on it.

    There is no step counter in this codebase, so this is the same shape
    `test_image_steps.py::_counting_decode` uses for the VAE: a delegate that
    appends to a list. An empty list is the assertion the whole workstream
    exists to make true — but only on a pipeline that would otherwise have
    generated, which is what `tokenizer=` is for.
    """

    def __init__(self, denoiser, *, slot: str = "transformer", tokenizer=None) -> None:
        setattr(self, slot, denoiser)
        self.passes: list[dict] = []
        #: `_tokenize` reads this with `getattr(pipe, "tokenizer", None)`, so
        #: `None` is exactly the "no tokenizer this can read" pipeline.
        self.tokenizer = tokenizer

    def __call__(self, prompt, **kwargs):
        self.passes.append({"prompt": prompt, **kwargs})


#: The key an MM-DiT block really produces: one joint `attn`, no `attn2`.
JOINT = {"transformer_blocks.0.attn": False, "transformer_blocks.1.attn": False}

#: What a UNet or a cross-attention DiT produces: `attn1` self, `attn2` cross.
CROSS = {
    "transformer_blocks.0.attn1": False,
    "transformer_blocks.0.attn2": True,
}

#: SD 3.5 Large. The joint block, plus a dual `attn2` that is SELF-attention.
DUAL = {
    "transformer_blocks.0.attn": False,
    "transformer_blocks.0.attn2": False,
}


# --------------------------------------------------- which blocks are sites


def test_the_one_rule_lives_in_one_place():
    """`_wrap` installs on the answer this gives, and `_measurable` withholds
    on it. A list of names re-typed on the other side of a wire is the
    defect."""
    assert ia.is_capture_site("down_blocks.1.attentions.0.attn2.processor")
    assert ia.is_capture_site("transformer_blocks.0.attn2.processor")
    assert not ia.is_capture_site("transformer_blocks.0.attn.processor")
    assert not ia.is_capture_site("transformer_blocks.0.attn1.processor")


def test_a_joint_attention_denoiser_has_no_capture_sites():
    seen = ia.capture_sites(_Denoiser(JOINT))
    assert seen is not None
    assert seen.sites == ()
    assert seen.blocks == 2
    assert seen.denoiser == "_Denoiser"


def test_a_cross_attention_denoiser_has_them():
    seen = ia.capture_sites(_Denoiser(CROSS))
    assert seen is not None
    assert seen.sites == ("transformer_blocks.0.attn2.processor",)
    assert seen.blocks == 2


def test_an_attn2_that_is_self_attention_is_not_a_capture_site():
    """SD 3.5 Large's dual attention. The NAME matches and the block is a
    plain self-attention over the image stream, called with no
    `encoder_hidden_states` — so `_Capturing` would record nothing from it,
    and counting it as a site buys the reader the same empty capture the
    name-only test was supposed to prevent."""
    seen = ia.capture_sites(_Denoiser(DUAL))
    assert seen is not None
    assert seen.sites == ()


# ------------------------------------------- absence of evidence is not zero


def test_no_denoiser_at_all_is_could_not_look_rather_than_none():
    """`None` keeps the capability. A wrapper pipeline whose denoiser this
    cannot reach is not a pipeline without cross-attention."""
    assert ia.capture_sites(None) is None


def test_a_denoiser_with_no_attn_processors_is_could_not_look():
    assert ia.capture_sites(object()) is None


def test_a_walk_that_raises_is_could_not_look():
    """`attn_processors` is a third-party recursive walk over somebody else's
    modules. It raising says nothing about whether cross-attention exists."""

    class Exploding:
        @property
        def attn_processors(self):
            raise RuntimeError("a module in here has no `named_children`")

    assert ia.capture_sites(Exploding()) is None


def test_a_walk_that_finds_no_attention_at_all_is_could_not_look():
    """An empty processor map says something about this SURFACE, not about
    the architecture — and the sentence it would license claims the denoiser
    mixes text and image in joint attention blocks, which would be a
    diagnosis invented out of nothing. The refusal at the end of a run still
    covers such a denoiser, and it is honest about what it knows."""

    class Empty:
        attn_processors: dict = {}

    assert ia.capture_sites(Empty()) is None


def test_the_site_walk_asks_the_one_rule_and_nothing_else():
    """The membership test lives in `is_capture_site`, and `capture_sites`
    may not carry a second one of its own.

    Found by mutation: widening the walk to `attn1` as well passed every
    other test in this file and every real model in the table, because the
    `is_cross_attention` refinement one line below undoes it — an `attn1` is
    self-attention and diffusers says so. The refinement is not a substitute.
    It is only consulted on denoisers this build can walk the modules of, and
    on the ones it cannot — the case the whole three-valued walk exists for —
    a widened name rule silently hands the capability back to a joint
    denoiser while `_wrap`, which still asks `is_capture_site`, installs on
    nothing. Two rules that agree on every model anybody tested is how the
    original defect got in.
    """

    class NamesOnly:
        attn_processors = {
            "transformer_blocks.0.attn1.processor": object(),
            "transformer_blocks.0.attn.processor": object(),
        }

    seen = ia.capture_sites(NamesOnly())
    assert seen is not None
    assert seen.sites == (), "only `is_capture_site` decides what a site is"
    assert seen.blocks == 2, "and everything walked is still counted"


def test_blocks_this_cannot_ask_about_are_kept_rather_than_dropped():
    """No `named_modules`, so `is_cross_attention` cannot be consulted. The
    name matched and that is the only evidence there is, so the site counts —
    withholding on a question nobody could ask is a guess."""

    class NamesOnly:
        attn_processors = {"transformer_blocks.0.attn2.processor": object()}

    seen = ia.capture_sites(NamesOnly())
    assert seen is not None
    assert seen.sites == ("transformer_blocks.0.attn2.processor",)


# ------------------------------------------------------------- the sentence


def test_the_sentence_names_the_architecture_and_never_blames_the_model():
    said = ia.no_capture_site_sentence(ia.capture_sites(_Denoiser(JOINT)))
    assert "_Denoiser" in said, "the architecture is what this is about"
    assert "joint" in said.lower()
    # Not a fault in the checkpoint, and said in as many words. A reader told
    # only that a measurement is unavailable goes looking for a newer build.
    assert "not at fault" in said.lower()
    for blame in ("broken", "is at fault", "invalid", "corrupt"):
        assert blame not in said.lower(), said
    # The measurement that DOES answer the same question here.
    assert "knockout" in said.lower()


def test_the_sentence_says_how_many_blocks_were_looked_at():
    """The DENOMINATOR, and it is the difference between a receipt and an
    assertion. "None of them is a cross-attention block" out of fifty-seven
    is a finding; out of a number nobody states it is a claim the reader has
    to take on trust. The count was previously asserted as `"2" in said`,
    which the literal `attn2` in the same sentence satisfies on its own — so
    deleting the count entirely left the test green.
    """
    many = ia.no_capture_site_sentence(ia.capture_sites(_Denoiser(JOINT)))
    assert "2 attention blocks" in many, many

    one = ia.no_capture_site_sentence(
        ia.capture_sites(_Denoiser({"transformer_blocks.0.attn": False}))
    )
    assert "1 attention block and" in one, "and it counts in English"


def test_the_sentence_does_not_repeat_the_post_hoc_hedge():
    """`no_capture_site_sentence`'s own docstring forbids this and nothing
    enforced it.

    The refusal at the END of `capture` says the denoiser "may attend to its
    conditioning somewhere this does not reach, which is a gap in coverage" —
    honest for a denoiser that has capture sites and still produced nothing,
    and exactly the vagueness that sends a reader looking for a newer build.
    This sentence is a fact established before a pass was spent. Borrowing
    the hedge would throw that away and make the two refusals unreadable
    against each other.
    """
    said = ia.no_capture_site_sentence(ia.capture_sites(_Denoiser(JOINT)))
    for hedge in ("may attend", "gap in coverage", "somewhere this does not"):
        assert hedge not in said.lower(), said


# ------------------------------------------------ where the wrapping lands


def test_wrap_installs_on_exactly_the_capture_sites():
    """The other half of "the one rule lives in one place", and the half that
    actually installs.

    `is_capture_site` is asserted directly above, but `_wrap` is where the
    condition historically lived, and a widening applied HERE rather than
    there fails nothing else in this file: on a UNet a wrapped
    self-attention block records nothing anyway (`_Capturing` returns early
    with no `encoder_hidden_states`), so the existing capture tests stay
    green while every joint denoiser silently gets its capability back.

    EVERY key comes back, wrapped or not: `set_attn_processor` raises when
    the dict it is handed is shorter than the module's own processor count.
    """
    keys = {f"{path}.processor": object() for path in {**JOINT, **CROSS}}
    wrapped = ia._wrap(keys, ia._Collector())

    assert set(wrapped) == set(keys), "a short dict breaks every pipeline"
    for name, processor in wrapped.items():
        assert isinstance(processor, ia._Capturing) is ia.is_capture_site(name), name
    assert any(isinstance(p, ia._Capturing) for p in wrapped.values()), (
        "the CROSS half of the fixture has a site, so something must be wrapped"
    )


# ----------------------------------------------- the preflight, in `capture`


def test_a_pipeline_with_a_tokenizer_really_does_spend_a_pass():
    """THE CONTROL, and without it every "zero passes" assertion below is a
    statement about a fake that could never have generated in the first
    place.

    A cross-attention denoiser on a pipeline with a tokenizer clears every
    refusal above the generation, so `capture` instruments the denoiser and
    calls it. What comes back is the post-hoc refusal — these fake processors
    are never invoked, so nothing lands in the store — which is the sentence
    the preflight exists to keep MM-DiT readers from ever paying for.
    """
    denoiser = _Denoiser(CROSS)
    pipe = _Pipe(denoiser, tokenizer=_Tokenizer())

    with pytest.raises(ia.NotSupported, match="without capturing a single"):
        ia.capture(pipe, "a red cube", steps=3, seed=7)

    assert len(pipe.passes) == 1, "the fake pipeline can be reached"
    assert pipe.passes[0]["num_inference_steps"] == 3
    # And it was instrumented on the way in and restored on the way out.
    assert any(isinstance(p, ia._Capturing) for p in denoiser.installs[0].values())
    assert not any(isinstance(p, ia._Capturing) for p in denoiser.installed.values()), (
        "a pipeline left with capturing processors leaks a map per generation"
    )


def test_a_joint_attention_pipeline_is_refused_before_a_single_pass():
    """THE ACCEPTANCE TEST. Zero passes, on a pipeline that would otherwise
    have spent twenty.

    The tokenizer is the load-bearing half of the fixture and it is here on
    purpose. Without it `capture` refuses for want of one several lines below
    the preflight, so `_Pipe.__call__` is unreachable and `passes == []`
    holds even with the preflight deleted — the exact shape of assertion this
    project's own review culture calls a receipt for nothing.
    `…really_does_spend_a_pass` above proves this fixture reaches a
    generation when it is allowed to.
    """
    denoiser = _Denoiser(JOINT)
    pipe = _Pipe(denoiser, tokenizer=_Tokenizer())
    with pytest.raises(ia.NotSupported, match="joint"):
        ia.capture(pipe, "a red cube", steps=20, seed=7)
    assert pipe.passes == [], "a pass was spent on a refusal that needed none"


def test_the_refusal_arrives_before_the_pipeline_is_instrumented():
    """A preflight below `set_attn_processor` costs no generation and still
    leaves the pipeline wearing capturing processors: the refusal raises
    past the `finally` that would have restored them, and every later
    generation in the process allocates a map nobody reads. The symptom is a
    slow OOM in a panel somebody opened once.

    `installs` rather than `installed` — `capture` restores the originals on
    its way out, so the LAST value says nothing about whether anything was
    ever installed.
    """
    denoiser = _Denoiser(JOINT)
    with pytest.raises(ia.NotSupported, match="joint"):
        ia.capture(_Pipe(denoiser, tokenizer=_Tokenizer()), "a red cube", steps=20)
    assert denoiser.installs == [], "the pipeline was instrumented anyway"
    assert denoiser.installed is None


def test_the_refusal_says_nothing_was_generated():
    """The old one arrived after twenty denoising passes and read as a
    property of the model. This one has to say what it cost, or a reader who
    has seen the old sentence will assume it cost the same."""
    with pytest.raises(ia.NotSupported) as caught:
        ia.capture(
            _Pipe(_Denoiser(JOINT), tokenizer=_Tokenizer()), "a red cube", steps=20
        )
    assert "Nothing was generated" in caught.value.sentence


def test_a_dual_attention_denoiser_is_refused_too():
    """SD 3.5 Large again, this time through the route the reader takes."""
    denoiser = _Denoiser(DUAL)
    pipe = _Pipe(denoiser, tokenizer=_Tokenizer())
    with pytest.raises(ia.NotSupported, match="joint"):
        ia.capture(pipe, "a red cube", steps=20)
    assert pipe.passes == []
    assert denoiser.installs == []


def test_a_cross_attention_pipeline_still_gets_past_the_gate():
    """The regression the fix must not cause. This fake has a real capture
    site, so it reaches the next refusal down — the tokenizer one — rather
    than this one."""
    pipe = _Pipe(_Denoiser(CROSS))
    with pytest.raises(ia.NotSupported, match="no tokenizer"):
        ia.capture(pipe, "a red cube", steps=2)


def test_a_denoiser_this_cannot_look_at_still_gets_past_the_gate():
    """`None` from the walk must not become a refusal, or every pipeline this
    cannot introspect loses a measurement it may well support."""

    class Opaque:
        def set_attn_processor(self, processors):
            raise AssertionError("never reached")

    pipe = _Pipe(Opaque())
    with pytest.raises(ia.NotSupported, match="no tokenizer"):
        ia.capture(pipe, "a red cube", steps=2)


# ------------------------------------------------- what the LOADED status says


def test_the_loaded_status_withholds_the_map_and_keeps_the_knockout():
    """`knockout` never touches `attn_processors`: it removes a word,
    regenerates at the same seed and RMS-differences the images. It is the
    CAUSAL half of this section and it works perfectly on an MM-DiT, so the
    gate that removes the map must not take it as well."""
    from modelmri import image_runtime as ir

    offered = ("cross_attention", "token_knockout", "step_commit", "latent_trace")
    kept, withheld = ir._measurable(_Pipe(_Denoiser(JOINT)), offered)

    assert "cross_attention" not in kept
    assert "token_knockout" in kept
    assert "step_commit" in kept and "latent_trace" in kept
    assert "joint" in withheld["cross_attention"].lower()
    assert "token_knockout" not in withheld


def test_a_cross_attention_pipeline_keeps_everything_it_had():
    from modelmri import image_runtime as ir

    offered = ("cross_attention", "token_knockout", "step_commit", "latent_trace")
    kept, withheld = ir._measurable(_Pipe(_Denoiser(CROSS)), offered)
    assert kept == list(offered)
    assert withheld == {}


def test_a_denoiser_the_status_cannot_walk_keeps_its_capabilities():
    """ "Could not look" reaches `_measurable` too, and it must land the same
    way there as it does in `capture`.

    A denoiser can expose `set_attn_processor` — so it clears the check above
    this one — and still offer nothing to walk. The map is then withheld on
    the strength of a question nobody could put, which is the failure mode
    the three-valued walk exists to prevent, and the control disappears from
    a pipeline that may well support it.
    """
    from modelmri import image_runtime as ir

    class Opaque:
        def set_attn_processor(self, processors):
            raise AssertionError("never reached")

    kept, withheld = ir._measurable(
        _Pipe(Opaque()), ("cross_attention", "token_knockout")
    )
    assert kept == ["cross_attention", "token_knockout"]
    assert withheld == {}


def test_a_pipeline_with_no_denoiser_keeps_its_capabilities():
    """The `_stub_load` case, and the rule behind it: a capability removed
    because nothing could be looked at is a guess, not a measurement."""
    from modelmri import image_runtime as ir

    kept, withheld = ir._measurable(object(), ("cross_attention", "token_knockout"))
    assert kept == ["cross_attention", "token_knockout"]
    assert withheld == {}


# ------------------------------------------ the table, against the real library


#: The classes `_ATTENTION_STYLES` states a style for that expose no
#: `attn_processors` to walk at all, so there is nothing for a built instance
#: to be asked. Each was settled by reading diffusers' own source, and the
#: reading is recorded in the comment above `_ATTENTION_STYLES` — this set is
#: what stops that being a place a class can be quietly parked: a class that
#: HAS the property and is listed here fails the completeness test below.
VERIFIED_BY_SOURCE = frozenset(
    {
        # `BasicTransformerBlock` WITH `cross_attention_dim` when the config
        # states one — PixArt-Alpha's older checkpoints are exactly this.
        "Transformer2DModel",
        # `BasicTransformerBlock` built with NO `cross_attention_dim`, so its
        # `attn2` is `None`; class-conditioned through ada-norm.
        "DiTTransformer2DModel",
        # `LuminaNextDiTBlock.attn2` is built with `cross_attention_dim=` and
        # called with `encoder_hidden_states`.
        "LuminaNextDiT2DModel",
        # Unconditional. No cross-attention anywhere.
        "UNet2DModel",
    }
)


def _built_denoisers() -> dict:
    """Every class in the table this build can construct, at toy sizes.

    No weights are downloaded and nothing is run: these are `__init__` calls
    costing single-digit milliseconds each, and the whole point is that the
    claim in `_ATTENTION_STYLES` is a claim about somebody ELSE's library and
    so has to be checked against that library rather than against this file's
    memory of it. Two of the four families this workstream exists for —
    AuraFlow and CogVideoX — were in the table on the strength of a reading
    alone, and a table entry nothing re-checks is how the original defect got
    in.
    """
    from diffusers import (
        AuraFlowTransformer2DModel,
        CogVideoXTransformer3DModel,
        FluxTransformer2DModel,
        HunyuanDiT2DModel,
        PixArtTransformer2DModel,
        SanaTransformer2DModel,
        UNet2DConditionModel,
        UNet3DConditionModel,
        UNetSpatioTemporalConditionModel,
    )

    return {
        "UNet2DConditionModel": UNet2DConditionModel(
            sample_size=8,
            in_channels=4,
            out_channels=4,
            down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
            block_out_channels=(8, 16),
            layers_per_block=1,
            norm_num_groups=4,
            cross_attention_dim=16,
            attention_head_dim=4,
        ),
        "UNet3DConditionModel": UNet3DConditionModel(
            sample_size=8,
            in_channels=4,
            out_channels=4,
            down_block_types=("CrossAttnDownBlock3D", "DownBlock3D"),
            up_block_types=("UpBlock3D", "CrossAttnUpBlock3D"),
            block_out_channels=(8, 16),
            layers_per_block=1,
            norm_num_groups=4,
            cross_attention_dim=16,
            attention_head_dim=4,
        ),
        # `SpatioTemporalResBlock` hard-codes 32 GroupNorm groups, so this one
        # cannot be shrunk below 32 channels the way the others can.
        "UNetSpatioTemporalConditionModel": UNetSpatioTemporalConditionModel(
            sample_size=8,
            in_channels=4,
            out_channels=4,
            down_block_types=(
                "CrossAttnDownBlockSpatioTemporal",
                "DownBlockSpatioTemporal",
            ),
            up_block_types=("UpBlockSpatioTemporal", "CrossAttnUpBlockSpatioTemporal"),
            block_out_channels=(32, 64),
            layers_per_block=1,
            cross_attention_dim=16,
            num_attention_heads=(2, 2),
            addition_time_embed_dim=8,
            projection_class_embeddings_input_dim=24,
        ),
        "PixArtTransformer2DModel": PixArtTransformer2DModel(
            num_attention_heads=2,
            attention_head_dim=8,
            in_channels=4,
            num_layers=2,
            cross_attention_dim=32,
            sample_size=8,
            caption_channels=32,
            num_embeds_ada_norm=10,
            patch_size=2,
            norm_num_groups=4,
        ),
        "HunyuanDiT2DModel": HunyuanDiT2DModel(
            num_attention_heads=2,
            attention_head_dim=8,
            in_channels=4,
            patch_size=2,
            sample_size=8,
            hidden_size=16,
            num_layers=2,
            cross_attention_dim=32,
            cross_attention_dim_t5=32,
            pooled_projection_dim=16,
            text_len=8,
            text_len_t5=8,
        ),
        "SanaTransformer2DModel": SanaTransformer2DModel(
            in_channels=4,
            out_channels=4,
            num_attention_heads=2,
            attention_head_dim=8,
            num_layers=1,
            num_cross_attention_heads=2,
            cross_attention_head_dim=8,
            cross_attention_dim=16,
            caption_channels=32,
            sample_size=8,
            patch_size=1,
        ),
        "SD3Transformer2DModel": _sd3(),
        "FluxTransformer2DModel": FluxTransformer2DModel(
            patch_size=1,
            in_channels=4,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=8,
            num_attention_heads=2,
            joint_attention_dim=32,
            pooled_projection_dim=16,
            axes_dims_rope=(2, 3, 3),
        ),
        "AuraFlowTransformer2DModel": AuraFlowTransformer2DModel(
            sample_size=8,
            patch_size=2,
            in_channels=4,
            num_mmdit_layers=1,
            num_single_dit_layers=1,
            attention_head_dim=8,
            num_attention_heads=2,
            joint_attention_dim=32,
            caption_projection_dim=16,
            out_channels=4,
            pos_embed_max_size=16,
        ),
        "CogVideoXTransformer3DModel": CogVideoXTransformer3DModel(
            num_attention_heads=2,
            attention_head_dim=8,
            in_channels=4,
            out_channels=4,
            time_embed_dim=16,
            text_embed_dim=32,
            num_layers=1,
            sample_width=8,
            sample_height=8,
            sample_frames=9,
            patch_size=2,
            temporal_compression_ratio=4,
            max_text_seq_length=8,
        ),
    }


def _sd3(**extra):
    from diffusers import SD3Transformer2DModel

    return SD3Transformer2DModel(
        sample_size=8,
        patch_size=2,
        in_channels=4,
        num_layers=2,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=32,
        caption_projection_dim=16,
        pooled_projection_dim=16,
        out_channels=4,
        **extra,
    )


def test_the_verified_table_still_matches_diffusers():
    """The static table is a claim about somebody else's library, so it is
    checked against that library rather than against this file.

    Ten genuine denoisers, built in-process at toy sizes: no weights are
    downloaded and nothing is run. What is asserted is that every class
    `imaging` calls joint really does produce no capture site, and every one
    it calls cross really does — which is exactly the claim the picker's
    badge makes before a reader spends a download on it.

    The site NAMES are checked, not just that there are some. "PixArt has
    sites" stays true if the walk starts returning `attn1` as well, and a map
    averaged over self-attention and cross-attention together is two
    quantities added up.
    """
    pytest.importorskip("torch")
    pytest.importorskip("diffusers")

    from modelmri import imaging

    for name, denoiser in _built_denoisers().items():
        seen = ia.capture_sites(denoiser)
        assert seen is not None, name
        style = imaging._ATTENTION_STYLES[name]
        if style == imaging.ATTENTION_CROSS:
            assert seen.sites, f"{name} is called cross-attention and has no attn2"
            assert all(s.endswith("attn2.processor") for s in seen.sites), (
                f"{name} contributed a site that is not an attn2: {seen.sites}"
            )
        else:
            assert not seen.sites, f"{name} is called {style} and has an attn2"

    # And the case the name-only test gets wrong on a real model.
    dual = _sd3(dual_attention_layers=(0,), qk_norm="rms_norm")
    assert any(ia.is_capture_site(k) for k in dual.attn_processors), (
        "SD 3.5's dual attention really does put an attn2 in the key list"
    )
    assert ia.capture_sites(dual).sites == (), "and it is still self-attention"


def test_every_class_in_the_table_was_checked_against_the_library_or_read():
    """No entry in `_ATTENTION_STYLES` gets in without a receipt.

    The style decides whether a reader is offered the map at all, so a class
    added to the table with nobody checking it is precisely the original
    defect: SD3 and Flux carried `cross_attention` for a year on the strength
    of "DiT-shaped", and it took a full generation to find out.

    The dividing line is not a taste: a class that exposes `attn_processors`
    can be BUILT and asked, so it must be, and one that does not expose the
    property has nothing to ask and is settled by a reading recorded beside
    the table. A class in `VERIFIED_BY_SOURCE` that grows the property in a
    future diffusers fails here rather than quietly staying unchecked.
    """
    pytest.importorskip("torch")
    pytest.importorskip("diffusers")

    import diffusers

    from modelmri import imaging

    built = set(_built_denoisers())
    for name in imaging._ATTENTION_STYLES:
        assert name in built or name in VERIFIED_BY_SOURCE, (
            f"{name} states an attention style nothing checks. Build it in "
            f"`_built_denoisers` or record the reading in VERIFIED_BY_SOURCE."
        )
    for name in VERIFIED_BY_SOURCE:
        cls = getattr(diffusers, name, None)
        assert cls is not None, f"{name} is no longer a diffusers class"
        assert not hasattr(cls, "attn_processors"), (
            f"{name} now exposes `attn_processors`, so it can be built and "
            f"asked rather than read. Move it into `_built_denoisers`."
        )
    assert built <= set(imaging._ATTENTION_STYLES), (
        "this test builds a class the table says nothing about"
    )


# ------------------------------------- what the panel prices, once they part


PANEL = Path(__file__).resolve().parents[1] / "frontend" / "src" / "ImagePanel.tsx"

#: Each preflight cost line in the panel, and the capability whose absence
#: makes that line a number about nothing.
#:
#: `renderCost` prices ONE CAPTURE, so it belongs to `cross_attention`;
#: `armsCost` prices a knockout; `traceCost` prices keeping a latent per step.
COST_GUARDS = {
    "renderCost": "canCapture",
    "armsCost": "canKnock",
    "traceCost": "canTrace",
}


def test_every_cost_line_in_the_panel_is_guarded_by_what_it_prices():
    """A cost quoted for a measurement this checkpoint cannot make is a
    number about nothing — and one left over from the previous checkpoint is
    a number about a different model.

    None of the three cost states is ever cleared: the effect that fills them
    is gated on the capability, there is no `setRenderCost(null)` anywhere,
    and `ImagePanel` is mounted once for the session, so unloading a model
    and loading another leaves the old numbers sitting in state. That was
    unreachable while the map and the knockout were always withheld together:
    `canCapture` false implied `canKnock` false and the whole block was
    absent. On an MM-DiT they part company on purpose — the knockout works,
    the map does not — so an SDXL followed by a Flux printed "one capture ·
    20 denoising passes" directly under the server's own sentence saying no
    map can be captured here, with no capture button on screen.

    Read off the source the same way `test_image_families` reads
    `OWNED_FAMILIES`: this is a fact about the panel, and there is no way to
    assert it from Python other than to look.
    """
    src = PANEL.read_text("utf-8", errors="replace")
    block = re.search(r'<div className="image-cost">(.*?)\n\s*</div>', src, re.S)
    assert block, "the cost block is not where this test expects it"
    body = block.group(1)

    for state, guard in COST_GUARDS.items():
        rendered = re.search(rf"\{{{state} && ([^(]*)\(", body)
        assert rendered, f"{state} is no longer rendered in the cost block"
        assert guard in rendered.group(1), (
            f"the {state} line renders without checking `{guard}`, so it "
            f"quotes a cost for a measurement this checkpoint may not offer "
            f"— and these states are never cleared, so the number can be the "
            f"previous checkpoint's."
        )
