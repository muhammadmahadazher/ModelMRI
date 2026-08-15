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
)


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


def _vision_config(policy_snap: Path, hf_home: str | Path | None):
    """The vision config for this policy, from the policy itself.

    Three sources, in order of directness:

    1. a `vision_config` block in the checkpoint's own config,
    2. the VLM it names — SmolVLA's config carries
       `vlm_model_name: HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, which is
       where the hardcoded constant came from, except now it is read rather
       than assumed,
    3. the checkpoint's config as a whole, if it *is* a vision config.
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

    cfg = AutoConfig.from_pretrained(str(policy_snap))
    if hasattr(cfg, "vision_config"):
        return cfg.vision_config
    if hasattr(cfg, "patch_size") and hasattr(cfg, "num_hidden_layers"):
        return cfg
    raise Refusal(
        "This checkpoint does not say what its vision encoder is: no "
        "`vision_config` block, no `vlm_model_name`, and its own config is "
        "not a vision config."
    )


@dataclass
class VLAStatus:
    loaded: bool = False
    mode: str = "unavailable"  # unavailable | data | perception | full
    reason: str = ""
    repo: str | None = None
    device: str = "cpu"
    n_layers: int = 0
    n_heads: int = 0
    grid: list[int] = field(default_factory=list)  # [32, 32]
    image_size: int = 0
    patch_size: int = 0
    warmup_ms: int | None = None
    dataset: dict | None = None

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
        self, repo: str = DEFAULT_VLA_REPO, hf_home: str | Path | None = None
    ) -> VLAStatus:
        """Load a policy's vision tower. Blocking — run in a worker thread.

        Architecture-agnostic by construction: the tensor prefix is discovered
        from the checkpoint, the config comes from the checkpoint, and the
        module is built from that config by `AutoModel`. Nothing here names
        SmolVLA.
        """
        import torch
        from safetensors.torch import load_file
        from transformers import AutoModel

        t0 = time.time()
        policy_snap = _snapshot(repo, hf_home)
        weights_file = policy_snap / "model.safetensors"
        if not weights_file.is_file():
            # Names the repo and the fix rather than the snapshot directory:
            # this sentence is published at 409, and a Refusal does not put a
            # path from this machine in front of the reader.
            raise Refusal(
                f"{repo} is cached but has no model.safetensors, so there are "
                f"no weights here to load a vision tower from. Re-download it "
                f"(huggingface-cli download {repo})."
            )

        from . import devices

        accel = devices.detect()
        cfg = _vision_config(policy_snap, hf_home)
        cfg._attn_implementation = "eager"  # sdpa returns no attention weights
        # From the config, not from a class name. `AutoModel.from_config` on
        # SmolVLM's vision config produces exactly the SmolVLMVisionTransformer
        # this used to hardcode — same 197 parameters, same key set — and it
        # produces the right module for a SigLIP or CLIP tower too.
        model = AutoModel.from_config(cfg)

        state = load_file(str(weights_file))
        prefix, found = discover_vision_prefix(state.keys())
        vision_state = {
            k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)
        }
        missing, unexpected = model.load_state_dict(vision_state, strict=False)
        if missing:
            raise Refusal(
                f"Found {found} vision tensors under '{prefix}' in {repo}, but "
                f"{len(missing)} the module needs are absent (e.g. "
                f"{', '.join(missing[:3])}). That means this tower is a "
                f"different architecture from the config describing it — "
                "refusing to show attention from a partially-initialised model."
            )
        model.eval()
        try:
            model.to(accel.torch_device)
        except Exception:
            accel = devices.detect(prefer="cpu")
            model.to("cpu")
        self.accel = accel

        grid = cfg.image_size // cfg.patch_size
        with self._lock:
            self.model = model
            self._attn = []
            self._attn_key = None
            self.status_ = VLAStatus(
                loaded=True,
                mode="perception",
                reason=(
                    f"vision tower of the real {repo} checkpoint "
                    f"({found} tensors under '{prefix}'); the action expert "
                    "needs the optional lerobot extra"
                ),
                repo=repo,
                device=accel.torch_device,
                n_layers=int(cfg.num_hidden_layers),
                n_heads=int(cfg.num_attention_heads),
                grid=[grid, grid],
                image_size=int(cfg.image_size),
                patch_size=int(cfg.patch_size),
                warmup_ms=None,
            )

        # warm up so the user's first click isn't the lazy-init click
        dummy = torch.zeros(
            1, 3, cfg.image_size, cfg.image_size, device=accel.torch_device
        )
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
        import numpy as np
        import torch

        if self.model is None:
            raise Refusal("No VLA policy loaded. POST /api/vla/load first.")

        grid = self.status_.grid[0]
        # Through `_prepare`, which the occlusion sweep also uses. Two copies
        # of this normalisation would mean the causal map and the attention
        # map beside it describe different images, and nothing on screen
        # would say so.
        img = self._prepare(rgb)

        t0 = time.time()
        with self._lock, torch.no_grad():
            out = self.model(pixel_values=img, output_attentions=True)
            # attention RECEIVED per patch: mean over queries -> [heads, G, G]
            maps = [
                a[0]
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
            "latency_ms": int((time.time() - t0) * 1000),
        }

    def _prepare(self, rgb):
        """One camera frame as the tower's normalised [1,3,S,S] input.

        Lifted out of `analyse` so the occlusion sweep feeds the tower exactly
        what the attention path feeds it. Two normalisations would mean the
        causal map and the attention map beside it describe different images,
        and nothing on screen would say so.
        """
        import numpy as np
        import torch

        size = self.status_.image_size
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
        img = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        img = torch.nn.functional.interpolate(
            img, size=(size, size), mode="bilinear", align_corners=False
        )
        return (img * 2.0 - 1.0).to(self.status_.device)

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
        stale = key is not None and self._attn_key is not None and tuple(key) != tuple(self._attn_key)
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
            "min": lo,
            "max": hi,
        }


# The frame travels at the resolution the POLICY SAW, not at the camera's.
# Anything larger is downsampled and the section says so -- a causal map is
# drawn over the frame, and a frame silently shrunk puts every block in the
# wrong place, which looks exactly like a finding.
MAX_SHARE_EDGE = 512


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
    fresh = handle._attn_key is None or tuple(handle._attn_key) == (episode, timestep)
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
                "baseline", "grid", "stride", "blocks", "n_blocks",
                "n_controlled", "passes", "scale", "attention_agreement",
                # The agreement is layer-dependent, so a shared .mri carrying
                # the Spearman without the layer it came from is not a
                # reproducible claim.
                "compared_layer", "compared_head",
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
