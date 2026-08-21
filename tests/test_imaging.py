"""Deciding what kind of image model a checkpoint is, before opening it.

Everything on the image side depends on this answer. A UNet has
cross-attention over prompt tokens; a DiT has it somewhere else; a ViT has no
prompt at all; a detector emits boxes. Draw a word-to-pixel map for a model
with no cross-attention and you get a picture of something that does not
exist — and it looks exactly like a picture of something that does.

So the tests are about two things: reading real structures correctly, and
refusing to name a family this does not know.

The fixtures are the real shapes. A diffusers pipeline really is a
`model_index.json` of `["library", "ClassName"]` pairs with the denoiser's own
`config.json` in a subfolder; a safetensors header really is a
little-endian u64 followed by JSON. Where a real checkpoint is on this disk,
there is a test that reads it.

No torch, no downloads, no loading — this module reads JSON and tensor
headers, and that is what makes it cheap enough to run before anything is
spent.
"""

from __future__ import annotations

import json
import struct

from modelmri import imaging


def _pipeline(tmp_path, denoiser_class="UNet2DConditionModel", slot="unet", **cfg):
    """A diffusers pipeline directory, in the real shape."""
    root = tmp_path / "pipe"
    root.mkdir(exist_ok=True)
    (root / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "StableDiffusionPipeline",
                "_diffusers_version": "0.31.0",
                slot: ["diffusers", denoiser_class],
                "vae": ["diffusers", "AutoencoderKL"],
                "text_encoder": ["transformers", "CLIPTextModel"],
                # A pipeline may legitimately record a component it does not
                # use. It must not become a phantom entry.
                "safety_checker": [None, None],
            }
        ),
        encoding="utf-8",
    )
    sub = root / slot
    sub.mkdir(exist_ok=True)
    (sub / "config.json").write_text(
        json.dumps({"_class_name": denoiser_class, **cfg}), encoding="utf-8"
    )
    return root


def _transformers(tmp_path, **config):
    root = tmp_path / "hf"
    root.mkdir(exist_ok=True)
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return root


def _safetensors(tmp_path, names, name="model.safetensors"):
    """A real safetensors header: u64 length, then that many bytes of JSON."""
    header = json.dumps(
        {n: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]} for n in names}
    ).encode("utf-8")
    p = tmp_path / name
    p.write_bytes(struct.pack("<Q", len(header)) + header + b"\x00\x00\x00\x00")
    return p


# ------------------------------------------------------------- pipelines


def test_a_unet_pipeline_is_read_from_its_index_and_subconfig(tmp_path):
    root = _pipeline(tmp_path, cross_attention_dim=768, sample_size=64)
    m = imaging.detect(root)
    assert m.family == imaging.UNET_DIFFUSION
    assert m.architecture == "UNet2DConditionModel"
    assert m.pipeline == "StableDiffusionPipeline"
    assert m.cross_attention_dim == 768
    assert m.image_size == 64


def test_a_dit_pipeline_is_a_different_family_from_a_unet(tmp_path):
    """They denoise the same thing and are not the same architecture; the
    panels that apply differ, so conflating them draws the wrong one."""
    root = _pipeline(
        tmp_path,
        denoiser_class="FluxTransformer2DModel",
        slot="transformer",
        cross_attention_dim=4096,
    )
    m = imaging.detect(root)
    assert m.family == imaging.DIT_DIFFUSION
    assert m.cross_attention_dim == 4096


def test_a_component_the_pipeline_does_not_use_is_not_a_phantom_entry(tmp_path):
    """`[null, null]` means "this pipeline has no safety checker". Listing it
    as a component would report a part that is not there."""
    m = imaging.detect(_pipeline(tmp_path))
    assert "safety_checker" not in m.components
    assert m.components["vae"] == "AutoencoderKL"


def test_an_unconditional_denoiser_says_there_are_no_word_maps_to_draw(tmp_path):
    """The substantive one. No cross-attention means no word-to-pixel map, and
    drawing one anyway would be inventing it."""
    root = _pipeline(tmp_path, denoiser_class="UNet2DModel", cross_attention_dim=0)
    m = imaging.detect(root)
    assert m.family == imaging.UNET_DIFFUSION
    assert m.cross_attention_dim == 0
    assert "UNCONDITIONAL" in m.means()
    assert "inventing them" in m.means()


def test_an_unknown_denoiser_in_the_transformer_slot_says_which_evidence_was_used(
    tmp_path,
):
    """The slot is evidence and evidence is not identification. Using it is
    fine; using it silently is not."""
    root = _pipeline(
        tmp_path, denoiser_class="SomeBrandNewTransformer", slot="transformer"
    )
    m = imaging.detect(root)
    assert m.family == imaging.DIT_DIFFUSION
    assert "not a class this knows" in m.reason
    assert "transformer` slot" in m.reason


def test_an_unknown_denoiser_in_the_unet_slot_is_refused(tmp_path):
    """No slot evidence, no known class — so no family. A guess here opens the
    wrong panel on a real model."""
    root = _pipeline(tmp_path, denoiser_class="SomethingElseEntirely", slot="unet")
    m = imaging.detect(root)
    assert m.family == imaging.UNKNOWN
    assert "SomethingElseEntirely" in m.reason
    assert "does not exist" in m.means()


# ---------------------------------------------------------- transformers


def test_a_detector_and_a_classifier_are_not_the_same_family(tmp_path):
    assert (
        imaging.detect(_transformers(tmp_path, model_type="detr")).family
        == imaging.DETECTION
    )
    assert (
        imaging.detect(_transformers(tmp_path, model_type="vit")).family == imaging.VIT
    )


def test_the_architecture_suffix_is_used_when_the_model_type_is_unknown(tmp_path):
    """transformers' naming scheme is a real signal — but only after
    `model_type`, which is the field the library itself dispatches on."""
    m = imaging.detect(
        _transformers(
            tmp_path,
            model_type="something_new",
            architectures=["BrandNewForObjectDetection"],
        )
    )
    assert m.family == imaging.DETECTION


def test_a_vision_config_identifies_a_vlm_with_no_name_hardcoded(tmp_path):
    """The rule that made Gemma 4 and Qwen 3.6 identify correctly without this
    module having heard of either. A config carrying a `vision_config` has a
    vision tower BY CONSTRUCTION; growing a name map instead is the hardcoding
    `vla.py` was corrected for, and it does not survive next week's release."""
    m = imaging.detect(
        _transformers(
            tmp_path,
            model_type="a_model_type_from_the_future",
            architectures=["FutureForConditionalGeneration"],
            vision_config={"hidden_size": 1152, "image_size": 448},
        )
    )
    assert m.family == imaging.VLM
    assert "vision_config" in m.reason


def test_a_causal_lm_is_not_an_image_model(tmp_path):
    m = imaging.detect(
        _transformers(tmp_path, model_type="qwen3", architectures=["Qwen3ForCausalLM"])
    )
    assert m.family == imaging.UNKNOWN
    assert "qwen3" in m.reason


# ------------------------------------------------------- tensor names only


def test_a_unet_is_identified_from_tensor_names_when_there_is_no_config(tmp_path):
    p = _safetensors(
        tmp_path,
        ["down_blocks.0.attn.weight", "mid_block.weight", "up_blocks.0.attn.weight"],
    )
    m = imaging.detect(p)
    assert m.family == imaging.UNET_DIFFUSION


def test_a_dit_is_identified_from_tensor_names(tmp_path):
    p = _safetensors(tmp_path, ["transformer_blocks.0.attn.to_q.weight"])
    assert imaging.detect(p).family == imaging.DIT_DIFFUSION


def test_tensor_names_that_match_nothing_are_refused_with_the_first_one(tmp_path):
    p = _safetensors(tmp_path, ["some.unfamiliar.tensor", "another.one"])
    m = imaging.detect(p)
    assert m.family == imaging.UNKNOWN
    assert "some.unfamiliar.tensor" in m.reason


def test_the_safetensors_header_is_read_without_loading_the_file(tmp_path):
    names = imaging.read_tensor_names(_safetensors(tmp_path, ["a.weight", "b.weight"]))
    assert names == ["a.weight", "b.weight"]


def test_a_header_length_the_file_cannot_back_is_refused(tmp_path):
    """The length comes FROM the file, so a corrupt or hostile one can claim
    the header is sixteen exabytes and `read(n)` would try."""
    p = tmp_path / "hostile.safetensors"
    p.write_bytes(struct.pack("<Q", 2**63) + b"{}")
    assert imaging.read_tensor_names(p) == []


def test_a_truncated_file_is_refused_rather_than_crashing(tmp_path):
    p = tmp_path / "short.safetensors"
    p.write_bytes(b"\x01\x02")
    assert imaging.read_tensor_names(p) == []


def test_metadata_is_not_a_tensor(tmp_path):
    header = json.dumps(
        {"__metadata__": {"format": "pt"}, "real.weight": {"dtype": "F32"}}
    ).encode()
    p = tmp_path / "m.safetensors"
    p.write_bytes(struct.pack("<Q", len(header)) + header)
    assert imaging.read_tensor_names(p) == ["real.weight"]


# ------------------------------------------------------------- refusals


def test_a_path_that_does_not_exist_says_so(tmp_path):
    m = imaging.detect(tmp_path / "nowhere")
    assert m.family == imaging.UNKNOWN
    assert "nothing at that path" in m.reason


def test_a_directory_with_nothing_readable_names_all_three_sources(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    m = imaging.detect(empty)
    assert m.family == imaging.UNKNOWN
    assert "model_index.json" in m.reason
    assert "config.json" in m.reason
    assert "safetensors" in m.reason


def test_an_unreadable_config_falls_through_rather_than_concluding(tmp_path):
    """Absent and unreadable both mean "this source cannot answer", and
    neither entitles anybody to a conclusion about the model."""
    root = tmp_path / "broken"
    root.mkdir()
    (root / "config.json").write_text("{not json", encoding="utf-8")
    _safetensors(root, ["down_blocks.0.w", "up_blocks.0.w"])
    assert imaging.detect(root).family == imaging.UNET_DIFFUSION


def test_an_unknown_family_offers_no_capabilities(tmp_path):
    """An unknown family must offer nothing rather than everything — the
    capability list is what a panel asks before drawing."""
    m = imaging.detect(_transformers(tmp_path, model_type="qwen3"))
    assert m.capabilities == ()
    assert m.to_dict()["capabilities"] == []


def test_every_known_family_offers_at_least_one_capability():
    """A family this claims to know and can do nothing with is a family it
    should be reporting as unknown."""
    for family in (
        imaging.UNET_DIFFUSION,
        imaging.DIT_DIFFUSION,
        imaging.VIT,
        imaging.CLIP,
        imaging.DETECTION,
        imaging.SEGMENTATION,
        imaging.VLM,
    ):
        assert imaging._CAPABILITIES[family], family


# --------------------------------------------------- against the real disk


def test_it_reads_whatever_image_models_are_actually_cached_here():
    """Not a fixture. Whatever is on this machine, every entry the scan
    returns must be self-consistent — a family it named, or an unknown that
    says why."""
    for m in imaging.scan_cache(limit=30):
        assert m.path
        if m.known:
            assert m.capabilities, f"{m.path} named a family with no capabilities"
            assert m.means()
        else:
            assert m.reason, f"{m.path} is unknown with no reason"


def test_the_browse_list_excludes_models_with_no_visual_evidence():
    """ "Qwen3 is not an image model" is a determination, not a gap. Listing
    every causal LM as an unidentified image model buries the real ones."""
    for m in imaging.scan_cache(limit=30):
        assert "ForCausalLM" not in m.architecture, (
            f"{m.path} is a causal LM and should not be in the image list"
        )


def test_the_family_label_has_one_home():
    """Three callers were reading `_FAMILY_LABEL` directly and a fourth was
    about to. A second place that decides what a family is CALLED is a second
    place for the name to drift."""
    from modelmri import imaging

    assert imaging.label(imaging.SEGMENTATION) == "a segmentation model"
    assert imaging.label(imaging.UNET_DIFFUSION) == "a UNet diffusion model"


def test_an_unrecognised_family_is_not_echoed_back_at_the_reader():
    """An identifier printed at somebody is not a label. A name this does not
    know falls through to the UNKNOWN sentence rather than appearing raw in
    the middle of a refusal."""
    from modelmri import imaging

    said = imaging.label("something_nobody_added_here")
    assert "something_nobody_added_here" not in said
    assert said == imaging.label(imaging.UNKNOWN)


def test_a_class_conditioned_pipeline_is_not_offered_word_measurements(tmp_path):
    """REPORTED with a screenshot: `facebook/DiT-XL-2-256` loaded, 3.33 GB
    resident, the status bar advertising `cross_attention, token_knockout,
    step_commit, latent_trace` — and every one refusing at the click, after
    the reader had typed a prompt into a pipeline that has no `prompt`
    parameter at all.

    `DiTPipeline`'s components are `scheduler`, `transformer`, `vae`: no text
    encoder, no tokenizer, and an `id2label` of 1000 ImageNet classes. It is
    CLASS-conditioned. There are no words for a picture to have looked at.

    `capabilities` was `_CAPABILITIES.get(self.family)` — a lookup from the
    family NAME — so every DiT-shaped checkpoint was promised the word-based
    measurements because some DiT-shaped checkpoints are text-conditioned.
    A capability list is a promise about THIS checkpoint.
    """
    root = tmp_path / "dit"
    (root / "transformer").mkdir(parents=True)
    (root / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "DiTPipeline",
                "scheduler": ["diffusers", "DDIMScheduler"],
                "transformer": ["diffusers", "Transformer2DModel"],
                "vae": ["diffusers", "AutoencoderKL"],
                "id2label": {str(i): f"class {i}" for i in range(1000)},
            }
        ),
        encoding="utf-8",
    )
    (root / "transformer" / "config.json").write_text(
        json.dumps({"_class_name": "Transformer2DModel", "sample_size": 32}),
        encoding="utf-8",
    )

    found = imaging.detect(root)

    assert found.known
    assert found.conditioning == "class"
    assert found.n_classes == 1000
    # The two word-shaped measurements are withheld; the latent-side ones are
    # not, because a class-conditioned run still has steps and a latent.
    assert "cross_attention" not in found.capabilities
    assert "token_knockout" not in found.capabilities
    assert "step_commit" in found.capabilities

    # And the sentence says what to do instead of what is missing.
    assert "class" in found.means().lower()


def test_a_text_conditioned_pipeline_keeps_its_word_measurements(tmp_path):
    """The other side of the same test: withholding must key on what the
    checkpoint HAS, not on the family, or every DiT loses capabilities that
    PixArt and Sana genuinely support."""
    root = tmp_path / "pixart"
    (root / "transformer").mkdir(parents=True)
    (root / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "PixArtAlphaPipeline",
                "transformer": ["diffusers", "Transformer2DModel"],
                "text_encoder": ["transformers", "T5EncoderModel"],
                "tokenizer": ["transformers", "T5Tokenizer"],
                "vae": ["diffusers", "AutoencoderKL"],
            }
        ),
        encoding="utf-8",
    )
    (root / "transformer" / "config.json").write_text(
        json.dumps({"_class_name": "Transformer2DModel", "cross_attention_dim": 1152}),
        encoding="utf-8",
    )

    found = imaging.detect(root)

    assert found.conditioning == "text"
    assert "cross_attention" in found.capabilities
    assert "token_knockout" in found.capabilities


def test_an_unconditional_denoiser_outranks_an_inferred_conditioning(tmp_path):
    """A MEASURED zero beats a component list.

    `cross_attention_dim: 0` is the denoiser's own config saying it attends to
    nothing, which is a stronger and different claim from `null` ("the config
    did not say"). A pipeline that lists a text encoder beside a zero-width
    denoiser must still read as unconditional — the first version of this
    ordering described it as taking a prompt, and drawing word maps for one
    would be inventing them.
    """
    root = tmp_path / "uncond"
    (root / "unet").mkdir(parents=True)
    (root / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "StableDiffusionPipeline",
                "unet": ["diffusers", "UNet2DConditionModel"],
                "text_encoder": ["transformers", "CLIPTextModel"],
                "vae": ["diffusers", "AutoencoderKL"],
            }
        ),
        encoding="utf-8",
    )
    (root / "unet" / "config.json").write_text(
        json.dumps({"_class_name": "UNet2DModel", "cross_attention_dim": 0}),
        encoding="utf-8",
    )

    found = imaging.detect(root)
    assert found.cross_attention_dim == 0
