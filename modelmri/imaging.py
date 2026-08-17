"""What kind of image model is this, before anything tries to open it.

Every panel on the image side depends on this answer. A UNet diffusion model
has cross-attention over prompt tokens and a denoising schedule; a DiT has
neither in the same place; a ViT classifier has no prompt at all; a detector
emits boxes. Drawing a cross-attention map for a model that has no
cross-attention is not a degraded result — it is a picture of something that
does not exist, and it looks exactly like a picture of something that does.

So this reads the checkpoint and reports what it found, and refuses by name
when it does not recognise the architecture. `discover.py` already applies
that rule to text models (`"{archs[0]} is not a causal language model"`); this
is the same rule, on a family tree with many more branches.

## Read, never assumed

Four sources, in the order a real checkpoint answers them:

  `model_index.json`   a diffusers PIPELINE — names every component and the
                       library each comes from
  `config.json`        a transformers model — `model_type` and `architectures`
  safetensors header   tensor names, when there is no config to read
  the directory        subfolders like `unet/`, `vae/`, `text_encoder/`

`vla.py` was corrected once for hardcoding three SmolVLA values — the tensor
prefix, the vision config repo, the module class — which made a general viewer
into a one-policy viewer. Nothing here is allowed to know a model name.

## What this does not do

It does not load anything. Every function reads JSON and tensor headers, so
the whole module runs in milliseconds on a machine with no accelerator and no
torch, and a caller can price the work before spending it.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

# What a model DOES, which is what decides the panels rather than its name.
#
# `unknown` is a first-class member, not a failure: a family this cannot name
# is reported as unknown WITH what was found, because a wrong family silently
# opens the wrong analysis.
UNET_DIFFUSION = "unet_diffusion"
DIT_DIFFUSION = "dit_diffusion"
VIT = "vit"
CLIP = "clip"
DETECTION = "detection"
SEGMENTATION = "segmentation"
VLM = "vlm"
UNKNOWN = "unknown"

# Diffusers pipeline classes → the denoiser architecture underneath. Read from
# `model_index.json`'s component list rather than the pipeline name where
# possible; this map is the fallback for a pipeline whose components were not
# readable.
_DENOISER_CLASSES = {
    "UNet2DConditionModel": UNET_DIFFUSION,
    "UNet2DModel": UNET_DIFFUSION,
    "UNet3DConditionModel": UNET_DIFFUSION,
    "UNetSpatioTemporalConditionModel": UNET_DIFFUSION,
    "Transformer2DModel": DIT_DIFFUSION,
    "DiTTransformer2DModel": DIT_DIFFUSION,
    "PixArtTransformer2DModel": DIT_DIFFUSION,
    "SD3Transformer2DModel": DIT_DIFFUSION,
    "FluxTransformer2DModel": DIT_DIFFUSION,
    "HunyuanDiT2DModel": DIT_DIFFUSION,
    "AuraFlowTransformer2DModel": DIT_DIFFUSION,
    "LuminaNextDiT2DModel": DIT_DIFFUSION,
    "CogVideoXTransformer3DModel": DIT_DIFFUSION,
    "SanaTransformer2DModel": DIT_DIFFUSION,
}

# transformers `model_type` → family. Only entries whose family genuinely
# changes which panels apply; a model type this does not list is `unknown`
# with its type reported, which is more useful than a guess.
_BY_MODEL_TYPE = {
    "vit": VIT,
    "deit": VIT,
    "beit": VIT,
    "swin": VIT,
    "swinv2": VIT,
    "convnext": VIT,
    "convnextv2": VIT,
    "dinov2": VIT,
    "dinov3": VIT,
    "siglip": CLIP,
    "siglip2": CLIP,
    "clip": CLIP,
    "chinese_clip": CLIP,
    "altclip": CLIP,
    "owlvit": DETECTION,
    "owlv2": DETECTION,
    "detr": DETECTION,
    "conditional_detr": DETECTION,
    "deformable_detr": DETECTION,
    "yolos": DETECTION,
    "rt_detr": DETECTION,
    "rt_detr_v2": DETECTION,
    "grounding-dino": DETECTION,
    "table-transformer": DETECTION,
    "segformer": SEGMENTATION,
    "maskformer": SEGMENTATION,
    "mask2former": SEGMENTATION,
    "oneformer": SEGMENTATION,
    "upernet": SEGMENTATION,
    "sam": SEGMENTATION,
    "sam2": SEGMENTATION,
    "sam3": SEGMENTATION,
    "sam3_video": SEGMENTATION,
    "llava": VLM,
    "llava_next": VLM,
    "idefics2": VLM,
    "idefics3": VLM,
    "smolvlm": VLM,
    "qwen2_vl": VLM,
    "qwen2_5_vl": VLM,
    "paligemma": VLM,
    "pixtral": VLM,
}

# What each family lets a user SEE. Keyed by family so a panel can ask rather
# than infer, and so an unknown family offers nothing instead of everything.
_CAPABILITIES = {
    UNET_DIFFUSION: (
        "cross_attention",
        "token_knockout",
        "step_commit",
        "latent_trace",
    ),
    DIT_DIFFUSION: ("cross_attention", "token_knockout", "step_commit", "latent_trace"),
    VIT: ("patch_attention", "attribution", "layer_readout"),
    CLIP: ("patch_attention", "text_image_similarity", "layer_readout"),
    DETECTION: ("patch_attention", "attribution", "box_confidence"),
    SEGMENTATION: ("patch_attention", "attribution"),
    VLM: ("patch_attention", "cross_attention", "layer_readout"),
    UNKNOWN: (),
}

# Human names, so a panel never prints an identifier at somebody.
_FAMILY_LABEL = {
    UNET_DIFFUSION: "a UNet diffusion model",
    DIT_DIFFUSION: "a transformer (DiT) diffusion model",
    VIT: "a vision transformer",
    CLIP: "an image-text embedding model",
    DETECTION: "an object detector",
    SEGMENTATION: "a segmentation model",
    VLM: "a vision-language model",
    UNKNOWN: "an architecture this does not recognise",
}


def label(family: str) -> str:
    """The family in prose, for anything that has a family but no `ImageModel`.

    `ImageModel.to_dict` and `means` already read this table; a third caller
    reaching into `_FAMILY_LABEL` would be a third place that decides what a
    family is called, and they drift. An unrecognised name falls through to
    the UNKNOWN sentence rather than being echoed back at somebody — an
    identifier printed at a reader is not a label.
    """
    return _FAMILY_LABEL.get(family, _FAMILY_LABEL[UNKNOWN])


# The safetensors header is a little-endian u64 length followed by that many
# bytes of JSON. Bounded because the length is read from the file: a corrupt
# or hostile one can claim the header is 16 exabytes, and `read(n)` would
# happily try.
_MAX_HEADER_BYTES = 100 * 1024 * 1024


@dataclass
class ImageModel:
    """What was found, and what may be done with it."""

    path: str = ""
    # WHERE it was read from, which is not always what it is CALLED.
    # `scan_cache` renames `path` to the repo id, because that is the name a
    # reader recognises and the one `load` takes — and the snapshot directory
    # underneath it was being thrown away with it. Anything that needs to
    # measure the files (their size on disk, whether the download finished)
    # needs the directory, and reconstructing one from a repo id means
    # rebuilding the cache's own layout in a second place.
    directory: str = ""
    family: str = UNKNOWN
    # The class the checkpoint names for itself. Reported even when the family
    # is unknown -- "PixArtSigmaPipeline, which this does not recognise" is
    # actionable and "unknown" alone is not.
    architecture: str = ""
    pipeline: str = ""
    # Components of a diffusers pipeline, as {name: class}. Empty for a plain
    # transformers model, which is a fact rather than a gap.
    components: dict = field(default_factory=dict)
    # Whether the denoiser attends to prompt tokens at all. `None` when it
    # could not be determined -- an unconditional UNet has no cross-attention
    # and a map drawn for one would be a picture of nothing.
    cross_attention_dim: int | None = None
    image_size: int | None = None
    # Set when nothing here could be read.
    reason: str = ""

    @property
    def known(self) -> bool:
        return self.family != UNKNOWN

    @property
    def capabilities(self) -> tuple:
        return _CAPABILITIES.get(self.family, ())

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "family": self.family,
            "label": label(self.family),
            "architecture": self.architecture,
            "pipeline": self.pipeline,
            "components": dict(self.components),
            "cross_attention_dim": self.cross_attention_dim,
            "image_size": self.image_size,
            "capabilities": list(self.capabilities),
            "known": self.known,
            "reason": self.reason,
            "means": self.means(),
        }

    def means(self) -> str:
        if not self.known:
            found = self.architecture or self.pipeline
            named = f" It calls itself `{found}`." if found else ""
            return (
                f"This is not an image architecture ModelMRI recognises, so "
                f"nothing here will open it.{named} {self.reason} A panel "
                f"drawn for the wrong family is a picture of something that "
                f"does not exist, and it looks exactly like a picture of "
                f"something that does."
            ).strip()

        name = label(self.family)
        detail = ""
        if self.family in (UNET_DIFFUSION, DIT_DIFFUSION):
            if self.cross_attention_dim:
                detail = (
                    f" It attends to prompt tokens through a "
                    f"{self.cross_attention_dim}-wide cross-attention, so the "
                    f"word-to-pixel maps apply."
                )
            elif self.cross_attention_dim == 0:
                detail = (
                    " It is UNCONDITIONAL — no cross-attention to a prompt — "
                    "so there are no word-to-pixel maps to draw, and drawing "
                    "any would be inventing them."
                )
        return (
            f"{self.architecture or self.pipeline or 'This'} is {name}."
            f"{detail} What ModelMRI can measure on it: "
            f"{', '.join(self.capabilities) or 'nothing yet'}."
        )


def detect(path: str | Path) -> ImageModel:
    """Read a checkpoint directory and say what it is. Loads nothing."""
    p = Path(path)
    found = ImageModel(path=str(p), directory=str(p))
    if not p.exists():
        found.reason = "There is nothing at that path."
        return found
    if p.is_file():
        return _detect_file(p, found)

    index = _read_json(p / "model_index.json")
    if index is not None:
        return _detect_pipeline(p, index, found)

    config = _read_json(p / "config.json")
    if config is not None:
        return _detect_transformers(config, found)

    # No config at all. The tensor names are the last honest source: a UNet
    # has `down_blocks.*`, a ViT has `encoder.layer.*`, and neither is a guess.
    weights = sorted(p.glob("*.safetensors"))
    if weights:
        return _detect_from_tensors(weights[0], found)

    found.reason = (
        "It has no model_index.json, no config.json and no safetensors to "
        "read tensor names from, so there is nothing here that says what it is."
    )
    return found


def _detect_file(p: Path, found: ImageModel) -> ImageModel:
    if p.suffix.lower() == ".safetensors":
        return _detect_from_tensors(p, found)
    found.reason = (
        f"'{p.suffix or 'no extension'}' is a single file this cannot read a "
        f"structure out of. Point at the model's directory instead."
    )
    return found


def _detect_pipeline(root: Path, index: dict, found: ImageModel) -> ImageModel:
    """A diffusers pipeline names every component and the library it is from."""
    found.pipeline = str(index.get("_class_name") or "")
    components = {}
    for name, value in index.items():
        if name.startswith("_"):
            continue
        # Each component is ["library", "ClassName"], and a pipeline may
        # legitimately record `[null, null]` for one it does not use.
        if isinstance(value, (list, tuple)) and len(value) == 2 and value[1]:
            components[name] = str(value[1])
    found.components = components

    denoiser_name, denoiser_class = "", ""
    for name in ("unet", "transformer", "denoiser"):
        if name in components:
            denoiser_name, denoiser_class = name, components[name]
            break

    found.architecture = denoiser_class or found.pipeline
    found.family = _DENOISER_CLASSES.get(denoiser_class, UNKNOWN)
    if found.family == UNKNOWN and denoiser_class:
        # A class this does not know, in the denoiser slot. The slot itself is
        # evidence -- `transformer` means DiT-shaped -- but evidence is not
        # identification, and the response says which was used.
        found.family = DIT_DIFFUSION if denoiser_name == "transformer" else UNKNOWN
        if found.family != UNKNOWN:
            found.reason = (
                f"`{denoiser_class}` is not a class this knows, but it sits in "
                f"the `transformer` slot, so it is treated as DiT-shaped."
            )
    if found.family == UNKNOWN:
        found.reason = found.reason or (
            f"Its denoiser is `{denoiser_class or 'not named'}`, which is not "
            f"an architecture this can read."
        )
        return found

    # The denoiser's own config, for the two numbers a panel needs.
    sub = _read_json(root / denoiser_name / "config.json") or {}
    dim = sub.get("cross_attention_dim")
    found.cross_attention_dim = int(dim) if isinstance(dim, int) else None
    size = sub.get("sample_size")
    found.image_size = int(size) if isinstance(size, int) else None
    return found


def _detect_transformers(config: dict, found: ImageModel) -> ImageModel:
    archs = config.get("architectures") or []
    found.architecture = str(archs[0]) if archs else ""
    model_type = str(config.get("model_type") or "")

    family = _BY_MODEL_TYPE.get(model_type, UNKNOWN)
    if family == UNKNOWN and archs:
        # The architecture SUFFIX is a real signal in transformers' naming
        # scheme and is checked only after model_type, which is the field the
        # library itself dispatches on.
        name = found.architecture
        if name.endswith("ForObjectDetection"):
            family = DETECTION
        elif name.endswith(("ForImageClassification", "ForImageSegmentation")):
            family = SEGMENTATION if "Segmentation" in name else VIT
        elif name.endswith("ForSemanticSegmentation"):
            family = SEGMENTATION

    if family == UNKNOWN and config.get("vision_config"):
        # STRUCTURAL, not another name in the map. A config that carries a
        # `vision_config` has a vision tower by construction, whatever the
        # model is called — which is how Gemma 4 and Qwen 3.6 identify
        # correctly without this file having heard of either.
        #
        # `_BY_MODEL_TYPE` exists for the cases where the type alone is
        # decisive; growing it model by model is the hardcoding `vla.py` was
        # corrected for, and it does not scale past whatever shipped last week.
        family = VLM
        found.reason = (
            f"`{model_type or 'this model type'}` is not in the known list, "
            f"but its config carries a `vision_config`, so it has a vision "
            f"tower and is read as a vision-language model."
        )

    found.family = family
    if family == UNKNOWN:
        found.reason = (
            f"Its `model_type` is `{model_type or 'absent'}`, which is not an "
            f"image architecture this reads."
        )
        return found

    for key in ("image_size", "sample_size"):
        value = config.get(key) or (config.get("vision_config") or {}).get(key)
        if isinstance(value, int):
            found.image_size = value
            break
    return found


def _detect_from_tensors(weights: Path, found: ImageModel) -> ImageModel:
    """Tensor names, when there is no config. Read, never inferred from size."""
    names = read_tensor_names(weights)
    if not names:
        found.reason = (
            "Its safetensors header could not be read, so there are no tensor "
            "names to identify it by."
        )
        return found

    joined = "\n".join(names)
    if "down_blocks." in joined and "up_blocks." in joined:
        found.family = UNET_DIFFUSION
        found.architecture = "a UNet (identified by its tensor names)"
    elif "transformer_blocks." in joined or "double_blocks." in joined:
        found.family = DIT_DIFFUSION
        found.architecture = "a DiT (identified by its tensor names)"
    elif "encoder.layer." in joined or "vision_model." in joined:
        found.family = VIT
        found.architecture = "a vision transformer (identified by its tensor names)"
    else:
        found.reason = (
            f"Its {len(names)} tensor names match no architecture this knows. "
            f"The first is `{names[0]}`."
        )
    return found


def read_tensor_names(path: str | Path, limit: int = 4000) -> list[str]:
    """Tensor names from a safetensors header. No torch, no full read.

    The header is a little-endian u64 length then that many bytes of JSON, so
    this touches a few kilobytes of a file that may be twenty gigabytes.

    Bounded twice, because both numbers come from the file: the header length
    is checked before `read`, and the name list is capped. A hostile header
    can claim to be sixteen exabytes and `read(n)` would try.
    """
    p = Path(path)
    try:
        with p.open("rb") as fh:
            raw = fh.read(8)
            if len(raw) < 8:
                return []
            (length,) = struct.unpack("<Q", raw)
            if not 0 < length <= _MAX_HEADER_BYTES:
                return []
            header = json.loads(fh.read(length).decode("utf-8"))
    except (OSError, ValueError, struct.error):
        return []
    if not isinstance(header, dict):
        return []
    return [k for k in list(header)[:limit] if k != "__metadata__"]


def _read_json(path: Path):
    """`None` for absent OR unreadable, and the caller distinguishes.

    Both mean "this source cannot answer", and every caller falls through to
    the next source rather than concluding anything from the absence.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def scan_cache(hub: str | Path | None = None, limit: int = 200) -> list[ImageModel]:
    """Every image model already on this disk. Downloads nothing.

    Same rule as `discover.scan`: what is here, before asking anybody to type
    a name. An entry that cannot be identified is INCLUDED as unknown rather
    than dropped, so the list is what is on the disk rather than what this
    module happens to understand.
    """
    from . import paths

    root = Path(hub) if hub else Path(paths.hf_hub_cache())
    if not root.is_dir():
        return []
    out: list[ImageModel] = []
    for entry in sorted(root.glob("models--*")):
        if len(out) >= limit:
            break
        snaps = entry / "snapshots"
        if not snaps.is_dir():
            continue
        for snap in sorted(snaps.iterdir()):
            if not snap.is_dir():
                continue
            if not _looks_visual(snap):
                # "Qwen3 is not an image model" is a DETERMINATION, not a gap,
                # and listing every causal LM as an unidentified image model
                # would bury the one pipeline on the disk under twenty of
                # them. `detect()` on a path somebody named still reports
                # unknown-with-a-reason; this is the browse list.
                break
            found = detect(snap)
            found.path = entry.name[len("models--") :].replace("--", "/")
            out.append(found)
            break
    out.sort(key=lambda m: (not m.known, m.path))
    return out


def _looks_visual(snap: Path) -> bool:
    """Is there any evidence at all that this consumes or emits pixels?

    Cheap and structural — a pipeline index, a vision sub-config, or a known
    image `model_type`. Deliberately generous: a false positive shows up as
    `unknown` with its architecture named, which is informative, while a false
    negative hides a model the user has.
    """
    if (snap / "model_index.json").is_file():
        return True
    config = _read_json(snap / "config.json")
    if not isinstance(config, dict):
        return False
    if config.get("vision_config") or config.get("image_size"):
        return True
    if str(config.get("model_type") or "") in _BY_MODEL_TYPE:
        return True
    return any(
        str(a).endswith(
            (
                "ForObjectDetection",
                "ForImageClassification",
                "ForImageSegmentation",
                "ForSemanticSegmentation",
            )
        )
        for a in (config.get("architectures") or [])
    )
