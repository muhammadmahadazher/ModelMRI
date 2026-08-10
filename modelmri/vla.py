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

        size = self.status_.image_size
        grid = self.status_.grid[0]
        # letterbox to the square input the tower expects, then normalise to [-1,1]
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
        img = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        img = torch.nn.functional.interpolate(
            img, size=(size, size), mode="bilinear", align_corners=False
        )
        img = (img * 2.0 - 1.0).to(self.status_.device)

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
