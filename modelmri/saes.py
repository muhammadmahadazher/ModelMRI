"""Sparse autoencoder loading and math.

Loads SAELens-format SAEs directly from the HuggingFace Hub (cfg.json +
sae_weights.safetensors) without depending on the sae-lens package — the
format is stable and the math is small:

    features = relu((x [- b_dec]) @ W_enc + b_enc)
    recon    = features @ W_dec + b_dec

Default release: jbloom/GPT2-Small-SAEs-Reformatted, one SAE per GPT-2
residual-stream hook point (e.g. blocks.8.hook_resid_pre, d_sae=24576).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

DEFAULT_SAE_REPO = "jbloom/GPT2-Small-SAEs-Reformatted"
DEFAULT_SAE_HOOK = "blocks.8.hook_resid_pre"


@dataclass
class SAEStatus:
    loaded: bool
    repo: str | None = None
    hook: str | None = None
    layer: int | None = None
    d_in: int | None = None
    d_sae: int | None = None


class SAEHandle:
    """One loaded SAE: weights on CPU float32, encode/decode helpers."""

    def __init__(
        self,
        repo: str,
        hook: str,
        layer: int,
        W_enc: torch.Tensor,
        b_enc: torch.Tensor,
        W_dec: torch.Tensor,
        b_dec: torch.Tensor,
        apply_b_dec_to_input: bool,
    ) -> None:
        self.repo, self.hook, self.layer = repo, hook, layer
        self.W_enc, self.b_enc = W_enc, b_enc
        self.W_dec, self.b_dec = W_dec, b_dec
        self.apply_b_dec_to_input = apply_b_dec_to_input
        self.d_in, self.d_sae = W_enc.shape

    @classmethod
    def load(
        cls, repo: str = DEFAULT_SAE_REPO, hook: str = DEFAULT_SAE_HOOK
    ) -> "SAEHandle":
        m = re.search(r"blocks\.(\d+)\.", hook)
        if not m:
            raise ValueError(f"Cannot parse layer index from hook name: {hook!r}")
        layer = int(m.group(1))

        cfg_path = hf_hub_download(repo, f"{hook}/cfg.json")
        weights_path = hf_hub_download(repo, f"{hook}/sae_weights.safetensors")
        cfg = json.loads(open(cfg_path, encoding="utf-8").read())
        tensors = load_file(weights_path)

        return cls(
            repo=repo,
            hook=hook,
            layer=layer,
            W_enc=tensors["W_enc"].float(),
            b_enc=tensors["b_enc"].float(),
            W_dec=tensors["W_dec"].float(),
            b_dec=tensors["b_dec"].float(),
            apply_b_dec_to_input=bool(cfg.get("apply_b_dec_to_input", False)),
        )

    def status(self) -> SAEStatus:
        return SAEStatus(
            loaded=True,
            repo=self.repo,
            hook=self.hook,
            layer=self.layer,
            d_in=self.d_in,
            d_sae=self.d_sae,
        )

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """[S, d_in] residual activations -> [S, d_sae] feature activations."""
        x = x.float()
        if self.apply_b_dec_to_input:
            x = x - self.b_dec
        return torch.relu(x @ self.W_enc + self.b_enc)

    def steering_vector(self, feature_id: int) -> torch.Tensor:
        """Unit-norm decoder direction for one feature ([d_in])."""
        v = self.W_dec[feature_id]
        return v / (v.norm() + 1e-8)
