# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""VLA (Vision-Language-Action) introspection — looking inside a robot policy.

What this does: loads the **vision tower of a real policy checkpoint** and
runs real robot-camera frames through it with eager attention, so every patch
of the frame gets an attention value we can draw back onto the image.

**Any policy, not one policy.** Three things used to be hardcoded to SmolVLA:
the tensor prefix (`model.vlm_with_expert.vlm.model.vision_model.`), the repo
its vision config came from, and the module class. Every other checkpoint
found zero tensors and was told its "layout is not supported" — true only
because nothing had looked. Now all three come from the checkpoint:

  * the prefix is DISCOVERED by scanning the tensor names for a vision-shaped
    path segment, and the busiest candidate wins,
  * the config is read from the checkpoint's own `vision_config`, or from the
    VLM it names (SmolVLA's config carries `vlm_model_name`),
  * the module is built by `AutoModel.from_config`, which on SmolVLM's vision
    config produces exactly the class this used to name.

A checkpoint that genuinely has no recognisable vision tower is refused with
the top-level names it *does* have, which is a report rather than a verdict.

**Not every tower is a transformer.** ACT — the Action Chunking Transformer,
and the most common architecture in this category — sees through a torchvision
ResNet under `model.backbone.`, described by a draccus config that has `type:
"act"` and no `model_type` at all. Two things follow, and the second is the
one that matters:

  * `_vision_config` returns a `ConvBackboneSpec` for such a config instead of
    a transformers config, and `VLAHandle.load` builds the torchvision module
    from it. Pointing `/api/vla/load` at an ACT repo used to reach
    `AutoConfig.from_pretrained`, whose bare `ValueError` came back as HTTP
    500 — "Something inside ModelMRI failed rather than refusing" — one frame
    away from a good 409 that nothing could reach.
  * **A ResNet has no attention, and this module says so rather than drawing
    one.** `n_layers` and `n_heads` are `None` for a convolutional backbone
    because the concepts do not apply to it, not because nothing was measured;
    `grid` and `patch_size` are real, and MEASURED by running one frame
    through the stack and reading the feature-map shape it returns.
    `analyse()` refuses, and its sentence names the thing that DOES work:
    the occlusion sweep needs a pooled embedding and nothing else, so a causal
    map on an ACT policy is a real measurement rather than a consolation.

Honesty about scope ("perception" mode):
  * These are the policy's own weights — not a stand-in model.
  * SmolVLA freezes its vision encoder during training, so its tower is
    architecturally SmolVLM2's; it is nonetheless the exact module the
    policy sees the world through.
  * The action expert (which turns those features into motor commands)
    needs the `lerobot` package, whose torch/numpy pins conflict with the
    core runtime. That is `full` mode — deliberately opt-in, never a
    requirement for `pip install modelmri`.

Attention is reduced inside the forward pass: the raw tensor is
[1, 12 heads, 1024, 1024] (~50 MB per layer in fp32). We keep the mean
attention *received* per patch, reshaped to the 32x32 grid, per head.
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import paths
from .errors import BadRequest, Refusal

DEFAULT_VLA_REPO = "lerobot/smolvla_base"

# The name segments a vision tower goes by across published policies. This is
# a list of naming conventions, not a list of supported models: the prefix is
# DISCOVERED from the checkpoint's own tensor names, and the module is built
# from the checkpoint's own config.
#
# This used to be one hardcoded string —
# `model.vlm_with_expert.vlm.model.vision_model.` — SmolVLA's exact layout,
# alongside a hardcoded VLM repo for the config and a hardcoded module class.
# Any other policy found zero tensors and was told its "checkpoint layout is
# not supported", which was true only because nothing had looked.
VISION_HINTS = (
    "vision_model",
    "vision_tower",
    "vision_encoder",
    "image_encoder",
    "visual",
    # ACT keeps its torchvision ResNet under `model.backbone.`, and that is
    # the only name it has — nothing in an ACT checkpoint contains the word
    # "vision" anywhere in a tensor path.
    #
    # LAST ON PURPOSE, and the order is load-bearing. The loop below takes the
    # FIRST HINT that matches a key, not the leftmost match inside it, and
    # Qwen-VL-style checkpoints name their tower `backbone.visual.`. With
    # "backbone" ahead of "visual" that whole checkpoint collapses to the
    # prefix `backbone.`, which also covers the language model — so the tower
    # would come back with the wrong tensors under it and the config would
    # still describe a vision encoder. Measured on the synthetic key list in
    # tests/test_vla_discovery.py: 4 tensors either way, but the prefix flips
    # from `backbone.visual.` to `backbone.`.
    "backbone",
)

# The config keys a lerobot-style policy names its torchvision backbone in.
# A list of naming conventions exactly like VISION_HINTS above, and for the
# same reason: the VALUE is read out of the checkpoint's own config, and it is
# checked against torchvision's registry before anything is built from it, so
# this is not a list of supported models.
CONV_BACKBONE_KEYS = ("vision_backbone",)

# How a draccus config labels a camera input inside `input_features`.
# lerobot serialises `FeatureType.VISUAL` here, and it is the only place a
# convolutional policy says what size it was trained at — a ResNet has no
# `image_size` of its own, it takes whatever it is given.
VISUAL_FEATURE_TYPE = "VISUAL"

# How many config keys a refusal quotes back before it starts counting instead.
CONFIG_KEYS_SHOWN = 12


def prepare_frame(rgb, size: int, device):
    """One camera frame as a vision tower's normalised [1,3,S,S] input.

    THE ONLY normalisation in this package, and it is shared rather than
    copied for the reason `VLA.occlude` already depends on: two of them would
    mean the causal map and the attention map beside it describe different
    images, with nothing on screen saying so. `checkpoints.compare` needs the
    identical transform to put two policies' embeddings on one scale, so it
    calls this rather than growing a second one.
    """
    import numpy as np
    import torch

    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    img = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    img = torch.nn.functional.interpolate(
        img, size=(int(size), int(size)), mode="bilinear", align_corners=False
    )
    return (img * 2.0 - 1.0).to(device)


def discover_vision_prefix(keys) -> tuple[str, int]:
    """The state-dict prefix of this checkpoint's vision tower.

    Returns (prefix, tensor_count) for the busiest candidate — the tower is
    the largest group of tensors under any one vision-shaped name, and a
    stray `visual_proj` weight elsewhere cannot outvote 197 of them.

    Raises if nothing matches, naming what WAS found so the answer is
    actionable rather than "unsupported".
    """
    counts: dict[str, int] = {}
    for key in keys:
        for hint in VISION_HINTS:
            marker = f"{hint}."
            idx = key.find(marker)
            # A path segment, not a substring: `supervision_model` must not
            # match `vision_model`.
            if idx < 0 or (idx and key[idx - 1] != "."):
                continue
            prefix = key[: idx + len(marker)]
            counts[prefix] = counts.get(prefix, 0) + 1
            break
    if not counts:
        roots = sorted({k.split(".")[0] for k in keys})[:8]
        # A Refusal: the checkpoint is fine, we just cannot find a tower in it,
        # and the message is the report that lets someone tell us the name.
        raise Refusal(
            "No vision tower found in this checkpoint. Looked for a tensor "
            f"path containing any of {', '.join(VISION_HINTS)}; the top-level "
            f"names present are {', '.join(roots)}. If this policy keeps its "
            "vision encoder under another name, open an issue with that name "
            "and it becomes one line here."
        )
    prefix, n = max(counts.items(), key=lambda kv: kv[1])
    return prefix, n


@dataclass(frozen=True)
class ConvBackboneSpec:
    """A torchvision CNN, as the policy's own config names it.

    `_vision_config` returns one of these in place of a transformers config
    when the checkpoint names a convolutional backbone, and it is deliberately
    NOT shaped like a transformers config. It carries no `num_attention_heads`
    and no `patch_size`, because a ResNet has neither — a struct that answered
    those with a number would be the fabrication this module exists to avoid,
    and every reader downstream would inherit it.
    """

    #: The torchvision builder name, verbatim from the config: `resnet18`.
    builder: str
    #: Which config key named it, so a refusal can quote the file rather than
    #: assert something about it.
    named_by: str
    #: The config's own `type`, e.g. `act`. `None` when it does not say — this
    #: arm is about the shape of the config, not about one policy family.
    policy_type: str | None
    #: `replace_final_stride_with_dilation` as the config gives it. `None`
    #: means the config is silent and the builder's own default stands, which
    #: is a different fact from `False`, a choice somebody wrote down.
    dilate_final_stride: bool | None
    #: The square edge to feed, derived from the camera shapes below. `None`
    #: when the config declares none, and then the caller has to say.
    image_size: int | None
    #: Every camera shape the config declared, verbatim, so the sentence that
    #: reports the square can name what it was squared from.
    declared_shapes: tuple[tuple[int, ...], ...] = ()


class _TowerOutput:
    """What `ConvVisionTower` hands back, named after what transformers calls it.

    `last_hidden_state` because that is the attribute `vla_occlude._pooled`
    reads, and one contract shared by both tower kinds is why the occlusion
    sweep works on ACT without vla_occlude knowing ACT exists.
    """

    __slots__ = ("last_hidden_state", "spatial_shape")

    def __init__(self, last_hidden_state, spatial_shape: tuple[int, int]) -> None:
        self.last_hidden_state = last_hidden_state
        #: (rows, cols) of the feature map the tokens were flattened from.
        #: Carried rather than recomputed because the caller that needs it —
        #: `_measure_feature_grid` — must not have to know how the flatten
        #: was ordered.
        self.spatial_shape = spatial_shape


class ConvVisionTower:
    """A torchvision CNN behind the interface the rest of this module calls.

    NOT an `nn.Module` subclass, and the reason is mechanical: this file
    imports torch inside functions so that `import modelmri` costs nothing on
    a machine that only wants the CLI, and a class statement inheriting from
    `nn.Module` at module scope would undo that for every import of the
    package. It delegates the four things anything here asks of a module —
    `eval`, `to`, `load_state_dict`, and being called — to the real
    `nn.ModuleDict` it holds.

    WHAT IT RETURNS. `last_hidden_state` of shape [B, rows*cols, channels]:
    the cells of the final feature map, flattened in row-major order. Those
    cells are not tokens and this does not call them tokens anywhere the
    reader can see — they are the spatial positions of a convolutional
    activation, and the only thing downstream does with them is take a mean,
    which is defined the same way for both.

    WHAT IT REFUSES. `output_attentions=True`. There is no attention here to
    report, and the sentence says so along with the measurement that does
    work on this policy.
    """

    #: Read by `VLAHandle.analyse` before it spends a forward pass. An
    #: attribute rather than an isinstance check at every call site, so a
    #: second non-attention tower kind has one place to declare itself.
    has_attention = False

    def __init__(self, body, spec: ConvBackboneSpec) -> None:
        self.body = body
        self.spec = spec

    # ---------- the module surface ----------

    def eval(self):
        self.body.eval()
        return self

    def to(self, device):
        self.body.to(device)
        return self

    def load_state_dict(self, state, strict: bool = False):
        return self.body.load_state_dict(state, strict=strict)

    def __call__(self, pixel_values, output_attentions: bool = False):
        import torch

        if output_attentions:
            raise Refusal(self.no_attention_sentence())
        x = pixel_values
        for child in self.body.values():
            x = child(x)
        if x.dim() != 4:
            # Not a shape check for its own sake: `_pooled` would happily mean
            # over whatever came out and the sweep would report numbers from
            # it. A stack whose body does not end in a [B, C, H, W] map is one
            # this wrapper does not understand, and saying so is the only
            # honest option.
            raise Refusal(
                f"torchvision {self.spec.builder} produced a "
                f"{x.dim()}-dimensional output rather than the [batch, "
                f"channels, rows, cols] feature map this reads a spatial grid "
                f"from, so ModelMRI cannot say which part of the frame any of "
                f"it corresponds to. Only backbones ending in a convolutional "
                f"feature map can be opened here."
            )
        rows, cols = int(x.shape[-2]), int(x.shape[-1])
        hidden = torch.flatten(x, 2).transpose(1, 2)
        return _TowerOutput(hidden, (rows, cols))

    # ---------- the sentence ----------

    def no_attention_sentence(self, repo: str | None = None) -> str:
        """Why there is no attention map here, and what to run instead.

        Lives on the tower rather than at each raise site so the three places
        that refuse — `analyse`, `attention`, and this class's own `__call__`
        — cannot drift into saying three different things about one policy.
        """
        who = repo or self.spec.policy_type or "this policy"
        target = repo or "<repo>"
        return (
            f"{who} sees through a convolutional backbone (torchvision "
            f"{self.spec.builder}, named by `{self.spec.named_by}` in its own "
            f"config), and a convolutional stack has no attention at all — no "
            f"heads, no per-layer attention weights, nothing that could be "
            f"reshaped into a grid. ModelMRI will not paint an activation map "
            f"where an attention map goes, because it would be read as "
            f"attention. What DOES work on this policy is the occlusion sweep "
            f"(POST /api/vla/occlude): it hides each block of the frame, "
            f"re-runs this backbone, and reports how far its pooled feature "
            f"map moved — an interventional measure the attention map only "
            f"approximates. `modelmri policy start --repo {target}` opens the "
            f"action half."
        )


def _positive_int(value) -> int | None:
    """`value` as a positive int, or `None` if it is not one.

    `isinstance(True, int)` is True, so the bool guard goes first: a config
    that wrote `shape: [3, true, 640]` must not contribute a `1`.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = int(value)
    return number if number > 0 else None


def _declared_camera_shapes(raw: dict) -> tuple[tuple[int, ...], ...]:
    """Every camera shape a draccus config declares, verbatim and in order."""
    features = raw.get("input_features")
    if not isinstance(features, dict):
        return ()
    shapes: list[tuple[int, ...]] = []
    for feature in features.values():
        if not isinstance(feature, dict):
            continue
        # `VISUAL` and `FeatureType.VISUAL` are both in the wild depending on
        # how draccus serialised the enum, and neither is a fact worth
        # refusing a checkpoint over.
        if not str(feature.get("type", "")).upper().endswith(VISUAL_FEATURE_TYPE):
            continue
        shape = feature.get("shape")
        if not isinstance(shape, (list, tuple)) or len(shape) < 2:
            continue
        dims = tuple(d for d in (_positive_int(v) for v in shape) if d is not None)
        if len(dims) >= 2:
            shapes.append(dims)
    return tuple(shapes)


def _conv_backbone_spec(raw: dict) -> ConvBackboneSpec | None:
    """The torchvision backbone this config names, or `None` if it names none.

    Reads only; nothing is built here and nothing is validated against
    torchvision yet, because the sentence for "that is not a torchvision
    model" belongs beside the build where the alternatives can be listed.
    """
    builder = named_by = None
    for key in CONV_BACKBONE_KEYS:
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            builder, named_by = value.strip(), key
            break
    if builder is None:
        return None

    shapes = _declared_camera_shapes(raw)
    # The SMALLER edge of the smallest camera. Upsampling to the larger one
    # invents pixels the camera never had, and the sweep's cost is quadratic
    # in the side — this is a choice, so it is reported in `declared_shapes`
    # beside the square it produced rather than made silently.
    edges = [min(shape[-2:]) for shape in shapes]
    dilate = raw.get("replace_final_stride_with_dilation")
    return ConvBackboneSpec(
        builder=builder,
        named_by=named_by,
        policy_type=(raw["type"] if isinstance(raw.get("type"), str) else None) or None,
        # Only a real bool counts. A config that omits the key is silent, and
        # silence is not `False` — the builder's own default is what stands.
        dilate_final_stride=dilate if isinstance(dilate, bool) else None,
        image_size=min(edges) if edges else None,
        declared_shapes=shapes,
    )


def _no_vision_config_sentence(
    raw: dict, cfg_file: Path, *, model_type: str | None = None
) -> str:
    """Why this checkpoint's vision half cannot be opened, and what still can.

    Named because two arms raise it — a config transformers cannot parse at
    all, and a config it parses into something with no vision in it — and
    they differ by exactly one clause, which `model_type` supplies. Passing
    the parsed model type rather than reusing one sentence for both keeps the
    "no `model_type`" clause from being said about a config that has one.
    """
    if not cfg_file.is_file():
        found = "there is no config.json in the snapshot at all"
    elif not raw:
        found = "its config.json is not readable as a JSON object"
    else:
        keys = sorted(raw)
        shown = ", ".join(keys[:CONFIG_KEYS_SHOWN])
        # The cap travels with the true count. The reader is looking for a key
        # this sentence did not name, so a list silently cut at twelve is the
        # one thing it must not be.
        more = (
            ""
            if len(keys) <= CONFIG_KEYS_SHOWN
            else f" ({CONFIG_KEYS_SHOWN} of {len(keys)} keys)"
        )
        found = f"its config.json has {shown}{more}"
    kind = raw.get("type") if isinstance(raw.get("type"), str) else None
    told = f" It calls itself `{kind}`." if kind else ""
    transformers_says = (
        f"transformers reads it as a `{model_type}` config, which carries no "
        f"vision half"
        if model_type
        else "no `model_type` for transformers to build from"
    )
    return (
        "This checkpoint does not say what its vision encoder is, so ModelMRI "
        f"will not guess one: {transformers_says}, "
        "no `vision_config` block, no `vlm_model_name`, and none of "
        f"{', '.join(CONV_BACKBONE_KEYS)} naming a torchvision backbone — "
        f"{found}.{told} The ACTION half may still open: `modelmri policy "
        "start --repo <this repo>` loads a policy through lerobot's own "
        "registry, which needs no transformers config, and /api/vla/actions/* "
        "then asks it what it would DO. Opening the vision half here needs "
        "the encoder named in the config."
    )


def _vision_config(policy_snap: Path, hf_home: str | Path | None):
    """The vision config for this policy, from the policy itself.

    Four sources, in order of directness:

    1. a `vision_config` block in the checkpoint's own config,
    2. the VLM it names — SmolVLA's config carries
       `vlm_model_name: HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, which is
       where the hardcoded constant came from, except now it is read rather
       than assumed,
    3. a torchvision backbone it names — ACT's draccus config carries
       `vision_backbone: resnet18`, which is a `ConvBackboneSpec` and not a
       transformers config at all,
    4. the checkpoint's config as a whole, if it *is* a vision config.

    Returns a transformers config for 1, 2 and 4, and a `ConvBackboneSpec`
    for 3. `VLAHandle.load` is the only caller and branches on the type.
    """
    import json

    from transformers import AutoConfig

    raw = {}
    cfg_file = policy_snap / "config.json"
    if cfg_file.is_file():
        try:
            raw = json.loads(cfg_file.read_text("utf-8"))
        except ValueError:
            raw = {}

    if isinstance(raw.get("vision_config"), dict):
        from transformers import AutoConfig as _AC

        return _AC.for_model(
            raw["vision_config"].get("model_type", "clip_vision_model"),
            **raw["vision_config"],
        )

    named = raw.get("vlm_model_name") or raw.get("vlm_repo")
    if named:
        try:
            snap = _snapshot(str(named), hf_home)
        except FileNotFoundError as err:
            # The published sentence names the repo and the command rather
            # than pasting `err`, whose text carries the cache directory on
            # this machine — a Refusal's message goes to the browser verbatim
            # (see errors.py). `from err` keeps the directory on the traceback
            # for whoever is debugging, which is where it belongs.
            raise Refusal(
                f"This policy's vision config comes from {named}, which is not "
                f"in your HuggingFace cache. Download it with "
                f"`huggingface-cli download {named}`."
            ) from err
        return AutoConfig.from_pretrained(str(snap)).vision_config

    conv = _conv_backbone_spec(raw)
    if conv is not None:
        return conv

    # THE 500 THAT LIVED HERE. This call had no except arm, and a draccus
    # lerobot config.json has `type: "act"` where transformers looks for
    # `model_type`. MEASURED against a synthetic ACT snapshot, before this
    # arm existed:
    #
    #   ValueError: Unrecognized model in <hub>\models--lerobot--act_fake\
    #   snapshots\deadbeef. Should have a `model_type` key in its config.json.
    #
    # `except BadRequest` in server.py does not catch a plain ValueError, so
    # POST /api/vla/load answered 500 "Something inside ModelMRI failed rather
    # than refusing" — while `discover_vision_prefix` one frame away already
    # had a 409 with a sentence naming the fix, unreachable because the config
    # lookup runs first. The exception's own text also carried the snapshot
    # path on this machine into the log line, which is the second reason not
    # to relay it: the sentence below is written here, and `from err` keeps
    # the path on the traceback where it belongs.
    try:
        cfg = AutoConfig.from_pretrained(str(policy_snap))
    except (ValueError, OSError, KeyError) as err:
        raise Refusal(_no_vision_config_sentence(raw, cfg_file)) from err
    if hasattr(cfg, "vision_config"):
        return cfg.vision_config
    if hasattr(cfg, "patch_size") and hasattr(cfg, "num_hidden_layers"):
        return cfg
    raise Refusal(
        _no_vision_config_sentence(
            raw, cfg_file, model_type=getattr(cfg, "model_type", None)
        )
    )


def _spatial_children(net) -> list[tuple[str, object]]:
    """The children of a torchvision model up to the head that collapses space.

    A rule about torch's own module TYPES rather than about layer names:
    everything before the first `AdaptiveAvgPool2d` / `AvgPool2d` / `Linear` /
    `Flatten` is the part that still has rows and columns, and everything from
    there on has thrown them away. Measured across torchvision 0.26: resnet
    breaks at `avgpool` (keeping conv1..layer4), vgg and convnext at `avgpool`,
    densenet at `classifier` (keeping `features`).

    Names matter as much as the split does — the kept children go into an
    `nn.ModuleDict` under their original names, so the state-dict keys are
    `conv1.weight`, `layer4.1.bn2.running_var` and so on: exactly what sits
    under `model.backbone.` in an ACT checkpoint. Dropping the classifier this
    way is also what makes the missing-keys check below meaningful, since a
    kept `fc` would report two missing tensors on every checkpoint that
    (correctly) does not carry one.
    """
    from torch import nn

    collapses = (nn.AdaptiveAvgPool2d, nn.AvgPool2d, nn.Linear, nn.Flatten)
    kept: list[tuple[str, object]] = []
    for name, child in net.named_children():
        if isinstance(child, collapses):
            break
        kept.append((name, child))
    return kept


def _build_conv_tower(spec: ConvBackboneSpec, keys) -> ConvVisionTower:
    """The torchvision backbone this config names, built to match these keys.

    `keys` are the checkpoint's own tensor names with the tower prefix already
    stripped, and they decide one thing that cannot be read from the config:
    WHICH NORMALISATION LAYER. lerobot builds ACT's ResNet with torchvision's
    `FrozenBatchNorm2d`, whose buffers are weight/bias/running_mean/running_var
    and nothing else, while a stock `BatchNorm2d` also carries
    `num_batches_tracked`. MEASURED on torchvision 0.26: `resnet18` has 102
    state-dict keys frozen and 122 unfrozen, the 20 extra being exactly those
    counters. Build the wrong one and 20 keys come back "missing" from a
    checkpoint that is perfectly fine, and the refusal below fires on it.

    NOTHING IS DOWNLOADED. `weights=None`, deliberately, even though an ACT
    config names `pretrained_backbone_weights: ResNet18_Weights.IMAGENET1K_V1`
    — that string describes what the backbone was initialised from before the
    policy was trained, and the trained tensors are in the checkpoint we are
    about to load. Honouring it would fetch ImageNet weights over the network
    and then overwrite every one of them.
    """
    import torchvision

    # Membership in torchvision's own registry, not a bare `getattr` on the
    # module: `builder` is a string out of a file, and `getattr` on a module
    # will hand back `__loader__` or `os` as happily as `resnet18`.
    available = torchvision.models.list_models(module=torchvision.models)
    if spec.builder not in available:
        near = sorted(n for n in available if n.startswith(spec.builder[:6]))[:6]
        raise Refusal(
            f"This policy's config names `{spec.builder}` under "
            f"`{spec.named_by}`, and torchvision {torchvision.__version__} has "
            f"no classification model by that name. It has "
            f"{len(available)} of them"
            + (f", including {', '.join(near)}" if near else "")
            + ". ModelMRI will not substitute a similar architecture for the "
            "one the checkpoint names, because the weights would load into "
            "the wrong shapes or silently into the right ones."
        )

    kwargs: dict = {"weights": None}
    tracked = any(str(k).endswith("num_batches_tracked") for k in keys)
    if not tracked:
        from torchvision.ops.misc import FrozenBatchNorm2d

        kwargs["norm_layer"] = FrozenBatchNorm2d
    if spec.dilate_final_stride is not None:
        kwargs["replace_stride_with_dilation"] = [
            False,
            False,
            bool(spec.dilate_final_stride),
        ]

    factory = getattr(torchvision.models, spec.builder)
    try:
        net = factory(**kwargs)
    except TypeError as err:
        # A real torchvision model that does not take these arguments — not
        # every family accepts `norm_layer` or dilation. The sentence names
        # the two we tried to pass so the reader can tell which one to drop
        # from the config, rather than relaying torch's own argument spelling.
        raise Refusal(
            f"torchvision {spec.builder} does not accept the construction "
            f"ModelMRI derived from this checkpoint (a "
            f"{'BatchNorm2d' if tracked else 'FrozenBatchNorm2d'} norm layer"
            + (", dilation on the final stride" if spec.dilate_final_stride else "")
            + f"). Only backbone families built like a ResNet can be opened "
            f"here; `{spec.builder}` is not one of them."
        ) from err

    from torch import nn

    kept = _spatial_children(net)
    if not kept:
        raise Refusal(
            f"torchvision {spec.builder} has no convolutional body before its "
            f"classifier head, so there is no feature map to read a spatial "
            f"grid from. Only backbones ending in one can be opened here."
        )
    return ConvVisionTower(nn.ModuleDict(dict(kept)), spec)


def _conv_image_size(spec: ConvBackboneSpec, repo: str, asked: int | None) -> int:
    """The square edge to feed a convolutional backbone, or a refusal.

    Two sources and no third: what the caller asked for, then what the
    checkpoint's config declares its cameras to be. There is no default,
    because a CNN has no native input size — 224 is a fact about ImageNet, not
    about this policy, and feeding a 480x640 policy a 224 square would move
    every occlusion block somewhere it was not.
    """
    if asked is not None:
        if isinstance(asked, bool) or not isinstance(asked, (int, float)):
            raise BadRequest(
                f"image_size must be a whole number of pixels, and "
                f"{asked!r} is not one."
            )
        side = int(asked)
        if side < 32:
            # Below the backbone's own total stride the feature map has no
            # rows left, and the sweep would plan blocks on an empty grid.
            raise BadRequest(
                f"image_size {side} is too small for a convolutional backbone "
                f"to produce a feature map with any rows in it — 32 pixels is "
                f"the smallest square this will feed."
            )
        return side
    if spec.image_size is not None:
        return spec.image_size
    raise Refusal(
        f"{repo} names `{spec.builder}` as its vision backbone under "
        f"`{spec.named_by}`, and a convolutional backbone has no input size of "
        f"its own — it takes whatever it is given. This config declares no "
        f"camera shape to read one from, so ModelMRI has nothing to feed it "
        f"and will not invent a square. Send `image_size` with the load: the "
        f"edge in pixels the policy was trained at."
    )


def _conv_reason(
    repo: str,
    spec: ConvBackboneSpec,
    found: int,
    prefix: str,
    side: int,
    rows: int,
    cols: int,
) -> str:
    """What `/api/vla` says about a convolutional tower, in one sentence.

    Says the three things a reader cannot get from the numbers beside it: what
    was actually built, that there is no attention here and why the two null
    fields are null, and which measurement DOES work — because a refusal that
    names the thing that works is worth twice one that does not, and this
    sentence is where a reader meets the limitation first.
    """
    squared = ""
    off = [s for s in spec.declared_shapes if tuple(s[-2:]) != (side, side)]
    if off:
        shapes = "; ".join("x".join(str(d) for d in s) for s in off)
        squared = (
            f" Its config declares camera frames of {shapes}, which this "
            f"resizes to a {side}x{side} square — the occlusion grid is in "
            f"cell coordinates and is unaffected, but the aspect ratio the "
            f"backbone sees here is not the one it was trained on."
        )
    return (
        f"vision backbone of the real {repo} checkpoint: torchvision "
        f"{spec.builder} ({found} tensors under '{prefix}'), fed {side}x{side} "
        f"and returning a {rows}x{cols} feature map. It is a convolutional "
        f"stack, so it has no attention: n_layers and n_heads are null because "
        f"the concepts do not apply to it, and /api/vla/analyse refuses rather "
        f"than reshaping activations into a square. The occlusion sweep does "
        f"work — it measures a shift in this backbone's pooled feature map, "
        f"which needs no attention at all.{squared} The action expert needs "
        f"the optional lerobot extra."
    )


def _measure_feature_grid(model, image_size: int, device) -> tuple[int, int]:
    """The spatial grid this tower actually produces, by running one frame.

    NOT `image_size // 32`. A ResNet's output grid is the product of its own
    strides, `replace_final_stride_with_dilation` changes it, and a different
    family changes it again — so the only honest source is the shape the
    module returns for the size it is about to be given. MEASURED on
    torchvision 0.26: resnet18 fed 480x480 returns a 15x15 map.

    One zero frame, no gradients. It costs `load` a second forward pass on top
    of its warmup, which is the price of not reciting a stride — measured on
    this machine, a 480x480 resnet18 probe is a few milliseconds against the
    4.0 s the CUDA warmup takes anyway.
    """
    import torch

    probe = torch.zeros(1, 3, int(image_size), int(image_size), device=device)
    with torch.no_grad():
        out = model(pixel_values=probe, output_attentions=False)
    return out.spatial_shape


@dataclass
class VLAStatus:
    loaded: bool = False
    #: `unavailable` until a tower is loaded, `perception` after. `data` and
    #: `full` were in this comment as if they were states this reports, and
    #: neither is ever assigned anywhere in the codebase — a documented enum
    #: whose two extra values could only ever mislead a reader grepping for
    #: where they come from. `full` is the opt-in policy sidecar, which
    #: `/api/policy` answers for; the dataset half is `/api/vla/episodes`.
    #:
    #: STILL `perception` FOR A CONVOLUTIONAL BACKBONE, and that is a
    #: decision. `mode` names WHICH HALF of the policy is open — the eyes or
    #: the action expert — not what the eyes are made of. A separate
    #: `perception_conv` value would make every `mode === "perception"` check
    #: quietly false for an ACT policy and hide the occlusion panel, which is
    #: the half that genuinely works on it. What the tower is made of is a
    #: different question, and `n_heads` answers it: see below.
    mode: str = "unavailable"  # unavailable | perception
    reason: str = ""
    repo: str | None = None
    #: `None`, not `"cpu"`. With nothing loaded there is no placement to
    #: report, and naming a device reads as a decision that was made.
    device: str | None = None
    #: `None`, not 0, for all five. These are read off a checkpoint, so with
    #: nothing loaded they are UNKNOWN — and the sibling fields `repo` and
    #: `warmup_ms` in this same dataclass already said so with `null` while
    #: these five published a confident zero. A resting `/api/vla` reported
    #: `n_layers: 0, n_heads: 0` beside `repo: null`, which reads as a tower
    #: that exists and has no layers.
    #:
    #: AND `None` FOR A CONVOLUTIONAL BACKBONE EVEN WHEN ONE IS LOADED, for
    #: the same rule read the other way: a ResNet has no attention heads and
    #: no layers of attention weights, so there is no number here that is
    #: about this tower. `n_layers` in particular is the range the layer
    #: slider and `attention(layer, …)` index over, and an ACT policy has
    #: nothing to index. A reader wanting "does this policy have an attention
    #: map?" reads `loaded && n_heads !== null`; the sentence is in `reason`,
    #: and /api/vla/analyse refuses with the same one.
    n_layers: int | None = None
    n_heads: int | None = None
    #: REAL for both tower kinds, and measured differently. For a transformer
    #: it is `image_size // patch_size`, straight off the config. For a
    #: convolutional backbone it is the shape of the feature map the stack
    #: actually returns, read from one probe pass — the concept survives the
    #: architecture change because both end in a rectangle of positions over
    #: the frame, which is exactly what the occlusion sweep plans blocks on.
    grid: list[int] = field(default_factory=list)  # [32, 32]
    #: The square edge fed to the tower, and the pixels one grid cell covers.
    #: `patch_size` is a patch edge for a transformer and the backbone's total
    #: stride for a CNN; both are "how far apart two grid cells are on the
    #: frame", which is the only thing anything downstream does with it.
    image_size: int | None = None
    patch_size: int | None = None
    warmup_ms: int | None = None
    # Tokens this tower prepends before the patches -- a class token, and
    # registers on top of that in DINOv2-style towers. 0 for SigLIP, which is
    # what SmolVLA uses and why `reshape(n_heads, grid, grid)` worked here for
    # as long as it did. Reported so a reader knows the map covers the patches
    # and not the whole sequence.
    #
    # `None` when nothing is loaded, because a REAL 0 here is a fact about a
    # tower — SigLIP prepends none — and "no tower" is not that fact.
    n_prefix_tokens: int | None = None
    # `dataset: dict | None` used to sit here. It was declared, serialised on
    # every `/api/vla` response, and assigned NOWHERE in the codebase — so it
    # was always `null`, and the TypeScript that declared it said so too. A
    # field that can only ever hold one value is not a field; removed rather
    # than given an invented assignment. `/api/vla/episodes` is where the open
    # dataset is described, and `/api/vla` already names the configured one as
    # `dataset_repo`.

    def to_dict(self) -> dict:
        return asdict(self)


def hub_root(hf_home: str | Path | None = None) -> Path:
    """The HuggingFace hub cache these checkpoints are downloaded into.

    An explicit `hf_home` keeps its literal meaning (root, so `hub/` under
    it); otherwise defer to the resolver. Computing this as `hf_home()/hub`
    ignored HF_HUB_CACHE, which is the variable the HuggingFace docs tell
    people to set — so the panel reported the checkpoint missing while it sat
    in the real cache, and the suggested `huggingface-cli download` fix
    re-downloaded it into the same directory we were not looking at.
    """
    return (Path(hf_home) / "hub") if hf_home else paths.hf_hub_cache()


def _snapshot(repo: str, hf_home: str | Path | None = None) -> Path:
    # A SLASH IS NOT ENOUGH. This tested `"/" not in repo`, which `pusht`
    # fails and `../../etc/passwd` passes — so a traversal string reached the
    # not-cached arm below and came back as "Download it first
    # (`huggingface-cli download ../../etc/passwd`)": a command that cannot
    # run, for a string that is not an id at all. That arm's own comment
    # scopes it to "a WELL-FORMED id that is not cached", and nothing was
    # enforcing the well-formed half.
    #
    # Shared with `vla_data.snapshot_path`, which had no check whatsoever.
    repo = paths.validate_repo_id(repo)
    owner, name = repo.split("/", 1)
    base = hub_root(hf_home) / f"models--{owner}--{name}"
    snaps = sorted((base / "snapshots").glob("*")) if base.is_dir() else []
    if not snaps:
        # Was a FileNotFoundError answered 409-with-its-own-text by server.py,
        # an arm that could not tell this sentence from safetensors failing to
        # open a file. Same words, a type that cannot be confused with a
        # library's. The snapshot directory comes out with it: vla.py's house
        # rule (see the model.safetensors refusal below) is that the repo id
        # and the download command are the actionable part.
        raise Refusal(
            f"{repo} is not cached. Download it first "
            f"(huggingface-cli download {repo})."
        )
    return snaps[-1]


class VLAHandle:
    """The loaded vision tower of a VLA policy, plus cached attention."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.model = None
        self.accel = None
        self.status_ = VLAStatus()
        # attention for the last analysed frame: list per layer of [heads, G, G]
        self._attn: list = []
        self._attn_key: tuple | None = None

    # ---------- loading ----------

    def load(
        self,
        repo: str = DEFAULT_VLA_REPO,
        hf_home: str | Path | None = None,
        *,
        image_size: int | None = None,
    ) -> VLAStatus:
        """Load a policy's vision tower. Blocking — run in a worker thread.

        Architecture-agnostic by construction: the tensor prefix is discovered
        from the checkpoint, the config comes from the checkpoint, and the
        module is built from that config — by `AutoModel` for a transformer
        tower, by `_build_conv_tower` for a torchvision one. Nothing here
        names SmolVLA and nothing here names ACT.

        `image_size` is the square edge in pixels to feed the tower, and it is
        only ever consulted for a convolutional backbone: a transformer tower
        states its own input size in its config, while a CNN takes whatever it
        is given and the config's declared camera shape is the only thing that
        knows. Passing it overrides that shape; omitting it when the config
        declares none is refused rather than guessed.
        """
        import torch
        from safetensors.torch import load_file
        from transformers import AutoModel

        t0 = time.time()
        policy_snap = _snapshot(repo, hf_home)
        # SHARDS TOO. `model.safetensors` was the only filename this module
        # looked for, and any policy above safetensors' 5 GB threshold ships
        # `model-0000N-of-0000M.safetensors` plus an index instead. The refusal
        # then said the repo had "no weights here" and told the reader to
        # re-download — which fetches the identical layout, so the reader
        # loops. The check also fires before `_vision_config`, which explicitly
        # supports the Qwen2-VL / LLaVA / PaliGemma shapes that are exactly the
        # ones large enough to shard.
        #
        # `discover.py`, `fit.py` and `quantdiff.py` are all shard-aware; this
        # module was the outlier.
        shards = sorted(policy_snap.glob("model-*-of-*.safetensors"))
        weights_file = policy_snap / "model.safetensors"
        if not weights_file.is_file() and not shards:
            # Names the repo and the fix rather than the snapshot directory:
            # this sentence is published at 409, and a Refusal does not put a
            # path from this machine in front of the reader.
            raise Refusal(
                f"{repo} is cached but has no safetensors weights in it — "
                f"neither `model.safetensors` nor a `model-0000N-of-0000M` "
                f"shard set — so there is nothing here to load a vision tower "
                f"from. Re-download it "
                f"(huggingface-cli download {repo})."
            )

        from . import devices

        accel = devices.detect()
        cfg = _vision_config(policy_snap, hf_home)
        conv = cfg if isinstance(cfg, ConvBackboneSpec) else None
        if conv is None:
            cfg._attn_implementation = "eager"  # sdpa returns no attention weights
        else:
            # Resolved BEFORE the safetensors file is opened, because the
            # answer to "you have to tell me the input size" should not cost
            # the reader a multi-gigabyte read first.
            edge = _conv_image_size(conv, repo, image_size)

        if shards and not weights_file.is_file():
            # Merged in shard order. The index maps tensor names to files and
            # every shard is disjoint, so a plain update is the whole job —
            # what mattered was looking for them at all.
            state = {}
            for shard in shards:
                state.update(load_file(str(shard)))
        else:
            state = load_file(str(weights_file))
        prefix, found = discover_vision_prefix(state.keys())
        vision_state = {
            k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)
        }
        # BUILT AFTER THE KEYS ARE IN HAND, which is a reorder rather than a
        # rewrite: `AutoModel.from_config` does not care when it runs, and
        # `_build_conv_tower` does — the checkpoint's own key set is what says
        # whether its ResNet was frozen-batchnorm or not, and there is no
        # other place to read it from.
        if conv is None:
            # From the config, not from a class name. `AutoModel.from_config`
            # on SmolVLM's vision config produces exactly the
            # SmolVLMVisionTransformer this used to hardcode — same 197
            # parameters, same key set — and it produces the right module for
            # a SigLIP or CLIP tower too.
            model = AutoModel.from_config(cfg)
        else:
            model = _build_conv_tower(conv, vision_state.keys())
        missing, unexpected = model.load_state_dict(vision_state, strict=False)
        if missing:
            raise Refusal(
                f"Found {found} vision tensors under '{prefix}' in {repo}, but "
                f"{len(missing)} the module needs are absent (e.g. "
                f"{', '.join(missing[:3])}). That means this tower is a "
                f"different architecture from the config describing it — "
                "refusing to measure anything through a partially-initialised "
                "model."
            )
        model.eval()
        try:
            model.to(accel.torch_device)
        except Exception:
            accel = devices.detect(prefer="cpu")
            model.to("cpu")
        self.accel = accel

        if conv is None:
            side = int(cfg.image_size)
            grid = [side // int(cfg.patch_size)] * 2
            patch = int(cfg.patch_size)
            n_layers: int | None = int(cfg.num_hidden_layers)
            n_heads: int | None = int(cfg.num_attention_heads)
            reason = (
                f"vision tower of the real {repo} checkpoint "
                f"({found} tensors under '{prefix}'); the action expert "
                "needs the optional lerobot extra"
            )
        else:
            side = edge
            rows, cols = _measure_feature_grid(model, side, accel.torch_device)
            grid = [rows, cols]
            # The stride between grid cells on the frame, derived from the two
            # numbers beside it rather than recited: `//` and not `/` because
            # `vla_occlude.sweep` indexes pixels with it, and it clamps the
            # last block to the frame edge, so a floor here loses nothing.
            patch = side // rows
            # THE HONEST PART. Not 0 — a ResNet does not have zero attention
            # heads, it has no such thing as an attention head, and a 0 in
            # this field is a measurement somebody took.
            n_layers = n_heads = None
            reason = _conv_reason(repo, conv, found, prefix, side, rows, cols)

        with self._lock:
            self.model = model
            self._attn = []
            self._attn_key = None
            self.status_ = VLAStatus(
                loaded=True,
                mode="perception",
                reason=reason,
                repo=repo,
                device=accel.torch_device,
                n_layers=n_layers,
                n_heads=n_heads,
                grid=grid,
                image_size=side,
                patch_size=patch,
                warmup_ms=None,
            )

        # warm up so the user's first click isn't the lazy-init click
        dummy = torch.zeros(1, 3, side, side, device=accel.torch_device)
        with torch.no_grad():
            model(pixel_values=dummy, output_attentions=False)
        self.status_.warmup_ms = int((time.time() - t0) * 1000)
        return self.status_

    def status(self) -> VLAStatus:
        return self.status_

    # ---------- inference ----------

    def analyse(self, rgb, key: tuple) -> dict:
        """Run one camera frame through the tower; cache per-layer attention.

        `rgb` is an HxWx3 uint8 ndarray. Returns shape metadata; the maps are
        served by `attention()`.
        """
        import torch

        if self.model is None:
            raise Refusal("No VLA policy loaded. POST /api/vla/load first.")
        # BEFORE THE FORWARD PASS, not after the reshape. A convolutional
        # backbone has activations of exactly the right shape to be reshaped
        # into a [heads, G, G] map, which is the whole danger: the picture
        # would come out looking like every other attention map in this panel
        # and be read as one. Refusing here also spends no GPU on an answer
        # that cannot be given.
        if not getattr(self.model, "has_attention", True):
            raise Refusal(self.model.no_attention_sentence(self.status_.repo))

        grid = self.status_.grid[0]
        # Through `_prepare`, which the occlusion sweep also uses. Two copies
        # of this normalisation would mean the causal map and the attention
        # map beside it describe different images, and nothing on screen
        # would say so.
        img = self._prepare(rgb)

        t0 = time.time()
        with self._lock, torch.no_grad():
            out = self.model(pixel_values=img, output_attentions=True)

            # NOT EVERY TOWER IS ALL PATCHES. `reshape(n_heads, grid, grid)`
            # assumes the token axis is exactly grid x grid, which is true of
            # SigLIP (attention pooling, no class token) and false of ViT,
            # CLIP and DINOv2 -- a class token, and registers on top of that.
            # Those prepend to the sequence, so the row is grid*grid + k long
            # and reshape raised a bare RuntimeError: a 500 out of a module
            # that otherwise goes to real lengths to load an architecture
            # nobody here has downloaded.
            #
            # The prefix is dropped from BOTH axes, so the map stays
            # patch-attends-to-patch rather than mixing in a token that is not
            # anywhere on the image. Identical to the old behaviour when there
            # is no prefix.
            patches = grid * grid
            n_tokens = int(out.attentions[0].shape[-1])
            prefix = n_tokens - patches
            if prefix < 0:
                raise Refusal(
                    f"this tower produced {n_tokens} attention tokens for a "
                    f"{grid}x{grid} patch grid ({patches} patches). ModelMRI "
                    f"cannot say which of them are patches, and drawing the "
                    f"wrong ones on the frame would be worse than not drawing."
                )
            self.status_.n_prefix_tokens = prefix
            maps = [
                a[0][:, prefix:, prefix:]
                .mean(dim=-2)
                .reshape(self.status_.n_heads, grid, grid)
                .float()
                .cpu()
                for a in out.attentions
            ]
            self._attn = maps
            self._attn_key = key
        return {
            "layers": len(maps),
            "heads": self.status_.n_heads,
            "grid": [grid, grid],
            "n_prefix_tokens": self.status_.n_prefix_tokens,
            "latency_ms": int((time.time() - t0) * 1000),
        }

    def _prepare(self, rgb):
        """One camera frame as the tower's normalised [1,3,S,S] input."""
        return prepare_frame(rgb, self.status_.image_size, self.status_.device)

    def occlude(
        self,
        rgb,
        scale_rgb: list,
        *,
        baseline: str = "episode_mean",
        stride: int = 0,
        layer: int = -1,
        head: int = -1,
        key: tuple | None = None,
        camera: str = "",
    ) -> dict:
        """What the tower's representation DEPENDED on, not what it looked at.

        `scale_rgb` are other frames of the same episode; they set the unit —
        the score is a shift in the tower's own embedding spread across them,
        because a raw L2 in embedding space means nothing on its own.

        PERCEPTION ONLY, and the returned sentence says so: without the action
        expert there is no action to affect.
        """
        from . import vla_occlude

        if self.model is None:
            raise Refusal("No VLA policy loaded. POST /api/vla/load first.")

        # The attention map for the same frame, when one has been taken, so
        # the two can be ranked against each other without a second request.
        #
        # A layer outside the tower is REFUSED rather than quietly dropped. It
        # used to fall through to attention_map=None, and the panel then said
        # "no attention map for this frame, run the policy on it first" — which
        # sends somebody to do a thing that cannot help, for a mistake that is
        # in their layer number.
        #
        # And the cached map has to be THIS frame's. The cache holds whatever
        # was analysed last — a cross-episode sweep overwrites it wholesale —
        # so without this check the headline Spearman could rank one frame's
        # causal map against a different frame's attention and report it as
        # this one's. Comparing the two maps is the whole point of the
        # measurement, and it is only a comparison when they share a frame.
        attention_map = None
        index = None
        stale = (
            key is not None
            and self._attn_key is not None
            and tuple(key) != tuple(self._attn_key)
        )
        if self._attn and not stale:
            layers = len(self._attn)
            index = layers - 1 if layer < 0 else layer
            if not 0 <= index < layers:
                raise BadRequest(
                    f"layer must be in [0,{layers}) to compare against "
                    f"attention, or -1 for the last."
                )
            attention_map = self.attention(index, head)["heat"]

        with self._lock:
            return vla_occlude.sweep(
                self.model,
                self.status_.device,
                self._prepare(rgb),
                grid=self.status_.grid,
                patch=self.status_.patch_size,
                scale_frames=[self._prepare(f) for f in scale_rgb],
                baseline=baseline,
                stride=stride or vla_occlude.DEFAULT_STRIDE,
                attention_map=attention_map,
                compared_layer=index,
                compared_head=head,
                # `key` already identifies the frame -- it is what the
                # staleness check above compares against -- so the result can
                # say which frame it is of instead of leaving that to whoever
                # happens to call this.
                episode=None if key is None else int(key[0]),
                timestep=None if key is None else int(key[1]),
                camera=camera,
            ).to_dict()

    def occlusion_cost(self, stride: int = 0) -> dict:
        """What the sweep would cost, before anybody waits for it."""
        from . import vla_occlude

        if not self.status_.loaded:
            raise Refusal("No VLA policy loaded.")
        return vla_occlude.estimate(
            self.status_.grid, stride or vla_occlude.DEFAULT_STRIDE
        )

    def attention_meta(self) -> dict:
        if not self.status_.loaded:
            return {"available": False, "reason": "no VLA policy loaded"}
        # Before "analyse a frame first", which for a convolutional backbone
        # is an instruction that cannot succeed — it sends the reader to a
        # button that refuses, for a limitation that is in the architecture.
        if not getattr(self.model, "has_attention", True):
            return {
                "available": False,
                "reason": self.model.no_attention_sentence(self.status_.repo),
            }
        if not self._attn:
            return {"available": False, "reason": "analyse a frame first"}
        return {
            "available": True,
            "reason": "",
            "n_layers": len(self._attn),
            "n_heads": self.status_.n_heads,
            "grid": self.status_.grid,
            "key": list(self._attn_key) if self._attn_key else None,
        }

    def attention(self, layer: int, head: int = -1) -> dict:
        """A [G][G] heatmap, normalised to [0,1]. head=-1 means mean over heads."""
        if self.model is not None and not getattr(self.model, "has_attention", True):
            # Same reason as in `analyse`: "analyse a frame first" would be a
            # next step that cannot work, and this route is reachable without
            # going through `analyse` at all.
            raise Refusal(self.model.no_attention_sentence(self.status_.repo))
        if not self._attn:
            # Ordering refusal: nothing is wrong with the request, there is
            # simply nothing measured yet to answer it from.
            raise Refusal("Analyse a frame first (POST /api/vla/analyse).")
        n_layers = len(self._attn)
        if not 0 <= layer < n_layers:
            raise BadRequest(f"layer must be in [0,{n_layers})")
        n_heads = self.status_.n_heads
        if head < -1 or head >= n_heads:
            raise BadRequest(f"head must be -1 (mean) or in [0,{n_heads})")

        m = self._attn[layer]
        m = m.mean(dim=0) if head < 0 else m[head]
        lo = float(m.min())
        hi = float(m.max())
        span = hi - lo
        norm = (m - lo) / span if span > 1e-12 else m * 0.0
        return {
            "layer": layer,
            "head": head,
            "grid": self.status_.grid,
            "heat": [[round(float(v), 4) for v in row] for row in norm.tolist()],
            # The SAME grid before normalisation. `heat` is stretched to [0,1]
            # so it can be drawn, and that subtracts this frame's own minimum
            # -- fine for a picture, wrong for any statistic. Anything that
            # computes over the distribution reads this instead; `vla_sweep`'s
            # entropy read `heat` and reported a frame's spread as a function
            # of its own darkest patch.
            "values": [[float(v) for v in row] for row in m.tolist()],
            "min": lo,
            "max": hi,
        }


# The frame travels at the resolution the POLICY SAW, not at the camera's.
# Anything larger is downsampled and the section says so -- a causal map is
# drawn over the frame, and a frame silently shrunk puts every block in the
# wrong place, which looks exactly like a finding.
MAX_SHARE_EDGE = 512


def attention_key(episode, timestep, camera: str = "") -> tuple:
    """What a cached attention map is ABOUT.

    THE CAMERA BELONGS IN HERE. `raw_frame` reads through the reader's
    process-wide current camera, so on a multi-camera dataset the same
    (episode, timestep) names a DIFFERENT picture -- and the panel refetches
    the frame when the camera changes while leaving `heat` and `heatKey`
    exactly where they were. With the camera missing from the key, neither
    side could tell that the map and the picture had come apart: `stale`
    stayed false, the "heatmap is from another frame" pill stayed hidden, and
    one tower's grid was drawn over another view's frame while
    `/api/vla/attention/meta` reported the pair as current.

    One function because three call sites built this tuple by hand, and a key
    assembled in three places is a key that drifts.
    """
    return (int(episode), int(timestep), str(camera or ""))


def share_payload(
    handle,
    reader,
    *,
    episode: int,
    timestep: int,
    layer: int = -1,
    head: int = -1,
    occlusion: dict | None = None,
) -> dict:
    """Everything a robot finding needs to be readable on somebody else's laptop.

    The camera frame, the per-layer attention, the causal map with its control
    band, and exactly which policy revision, dataset, episode, timestep and
    camera produced them. `session._vla` refuses this shape if any of those is
    missing, so the section cannot be written without them.
    """
    import numpy as np

    from . import vla_data

    status = handle.status()
    if not status.loaded:
        raise Refusal("No VLA policy loaded, so there is nothing to share.")

    rgb = np.asarray(reader.raw_frame(episode, timestep))
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    downsampled = False
    note = ""
    if max(height, width) > MAX_SHARE_EDGE:
        from PIL import Image

        scale = MAX_SHARE_EDGE / max(height, width)
        new = (max(1, int(width * scale)), max(1, int(height * scale)))
        rgb = np.asarray(Image.fromarray(rgb).resize(new, Image.BILINEAR))
        note = (
            f"the camera frame was {width}x{height} and is stored at "
            f"{new[0]}x{new[1]}; the occlusion grid is in PATCH coordinates "
            f"and is unaffected, but do not measure pixels off this image"
        )
        downsampled = True
        height, width = int(rgb.shape[0]), int(rgb.shape[1])

    # The cache holds the LAST analysed frame, which a cross-episode sweep
    # overwrites. Shipping it beside a different frame's picture would put two
    # frames in one file with nothing saying so, and the whole point of the
    # file is that somebody else can trust what is in it.
    maps = []
    fresh = handle._attn_key is None or tuple(handle._attn_key) == attention_key(
        episode, timestep, getattr(reader, "camera", "")
    )
    if handle._attn and fresh:
        wanted = range(len(handle._attn)) if layer < 0 else [layer]
        for index in wanted:
            if 0 <= index < len(handle._attn):
                maps.append(handle.attention(index, head)["heat"])

    payload: dict = {
        "provenance": {
            # `repo` and not a friendly name: two checkpoints of the same
            # policy are different models, and a finding attributed to the
            # wrong one is worse than an unattributed one.
            "policy": status.repo or "",
            # The REAL commit, read from the local cache the same way every
            # receipt in this project reads one — never the network. Two
            # checkpoints of the same repo are different models, and a
            # finding attributed to "lerobot/smolvla_base" with no commit is
            # attributed to whichever copy the reader happens to have.
            #
            # Falls back to the repo id when the cache cannot say, because an
            # unknown revision is still better than refusing to share.
            "revision": _revision_of(status.repo) or (status.repo or ""),
            "dataset": getattr(reader, "repo_id", ""),
            "camera": getattr(reader, "camera", ""),
            "episode": int(episode),
            "timestep": int(timestep),
        },
        "frame": vla_data.encode_png(rgb),
        "frame_size": [width, height],
        "frame_downsampled": downsampled,
    }
    if note:
        payload["frame_note"] = note
    if maps:
        payload["attention"] = maps
    if occlusion:
        payload["occlusion"] = {
            k: v
            for k, v in occlusion.items()
            if k
            in (
                "baseline",
                "grid",
                "stride",
                "blocks",
                "n_blocks",
                "n_controlled",
                "passes",
                "scale",
                "attention_agreement",
                # The agreement is layer-dependent, so a shared .mri carrying
                # the Spearman without the layer it came from is not a
                # reproducible claim.
                "compared_layer",
                "compared_head",
                "means",
            )
        }
    return payload


def _revision_of(repo: str | None) -> str:
    """The cached commit for this policy, or "" when it cannot be read."""
    if not repo:
        return ""
    try:
        from . import receipts

        commit, _ = receipts.revision_of(repo)
    except Exception:
        # Deliberately broad and silent: every failure here means "the cache
        # cannot say", and the caller already has a fallback. A share button
        # that raises because a ref file is missing would be worse than one
        # that records an unknown revision.
        return ""
    return commit or ""
