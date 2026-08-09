"""VLA (Vision-Language-Action) introspection — looking inside a robot policy.

What this does: loads the **vision tower of the actual SmolVLA checkpoint**
(`lerobot/smolvla_base`, 197 tensors, verified byte-for-byte from its
model.safetensors) and runs real robot-camera frames through it with eager
attention, so every patch of the frame gets an attention value we can draw
back onto the image.

Honesty about scope (v0.4, "perception" mode):
  * These are SmolVLA's own weights — not a stand-in model.
  * SmolVLA freezes its vision encoder during training, so this tower is
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

DEFAULT_VLA_REPO = "lerobot/smolvla_base"
VLM_REPO = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
VISION_PREFIX = "model.vlm_with_expert.vlm.model.vision_model."


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
        raise FileNotFoundError(
            f"{repo} is not cached under {base}. Download it first "
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
        """Load the policy's vision tower. Blocking — run in a worker thread."""
        import torch
        from safetensors.torch import load_file
        from transformers import AutoConfig
        from transformers.models.smolvlm.modeling_smolvlm import (
            SmolVLMVisionTransformer,
        )

        t0 = time.time()
        policy_snap = _snapshot(repo, hf_home)
        weights_file = policy_snap / "model.safetensors"
        if not weights_file.is_file():
            raise RuntimeError(f"{repo} has no model.safetensors at {policy_snap}")

        try:
            vlm_snap = _snapshot(VLM_REPO, hf_home)
        except FileNotFoundError as err:
            raise RuntimeError(
                f"SmolVLA's vision config comes from {VLM_REPO}, which is not cached. {err}"
            ) from err

        from . import devices

        accel = devices.detect()
        cfg = AutoConfig.from_pretrained(str(vlm_snap)).vision_config
        cfg._attn_implementation = "eager"  # sdpa returns no attention weights
        model = SmolVLMVisionTransformer(cfg)

        state = load_file(str(weights_file))
        vision_state = {
            k[len(VISION_PREFIX) :]: v
            for k, v in state.items()
            if k.startswith(VISION_PREFIX)
        }
        if not vision_state:
            raise RuntimeError(
                f"No vision-tower tensors ('{VISION_PREFIX}*') found in {repo} — "
                "this checkpoint layout is not supported."
            )
        missing, unexpected = model.load_state_dict(vision_state, strict=False)
        if missing:
            raise RuntimeError(
                f"{len(missing)} vision tensors missing from {repo} (e.g. {missing[:3]}) — refusing "
                "to show attention from a partially-initialised model."
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
                    "vision tower of the real SmolVLA checkpoint; the action expert "
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
            raise RuntimeError("No VLA policy loaded. POST /api/vla/load first.")

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
            raise RuntimeError("Analyse a frame first (POST /api/vla/analyse).")
        n_layers = len(self._attn)
        if not 0 <= layer < n_layers:
            raise ValueError(f"layer must be in [0,{n_layers})")
        n_heads = self.status_.n_heads
        if head < -1 or head >= n_heads:
            raise ValueError(f"head must be -1 (mean) or in [0,{n_heads})")

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
