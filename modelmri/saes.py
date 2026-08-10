"""Sparse autoencoder loading and math.

Loads SAELens-format SAEs directly from the HuggingFace Hub (cfg.json +
sae_weights.safetensors) without depending on the sae-lens package — the
format is stable and the math is small:

    features = relu(prepare(x) @ W_enc + b_enc)
    recon    = features @ W_dec + b_dec

`prepare` is the interesting part, and it is measured rather than assumed. An
SAE is trained on activations in one particular convention — centered along
d_model or not, with b_dec subtracted from the input or not — and feeding it
the other kind does not error. It returns features, in the right shape, for a
vector the SAE never saw.

That is what was happening. cfg.json for the default release has no
`apply_b_dec_to_input` key and nothing anywhere declares centering, so the
panel fed gpt2's raw HuggingFace residual stream to an SAE trained on
TransformerLens activations, which are centered. Measured on gpt2
blocks.8.hook_resid_pre, prompt "The Eiffel Tower is located in the city of",
fraction of variance unexplained went from 13579.24 to 0.0010 once the
convention was right, and the number of features firing per token from 7491.5
to 60.5. The panel's top-8 overlapped the correct top-8 two of eight.

So `calibrate` runs all four conventions against the model the SAE is actually
attached to and keeps the one that reconstructs, because "which convention"
has no universal answer — Gemma Scope SAEs are trained on raw HuggingFace
activations and must not be centered. See CONVENTIONS below.

Default release: jbloom/GPT2-Small-SAEs-Reformatted, one SAE per GPT-2
residual-stream hook point (e.g. blocks.8.hook_resid_pre, d_sae=24576).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

DEFAULT_SAE_REPO = "jbloom/GPT2-Small-SAEs-Reformatted"
DEFAULT_SAE_HOOK = "blocks.8.hook_resid_pre"


# The four ways an SAE's input can be prepared, and the reason there are four.
#
# `b_dec` is declared by a cfg key that the default release does not have, so
# `cfg.get("apply_b_dec_to_input", False)` resolved to False for an SAE whose
# training forward always subtracted it. Centering is not declared at all:
# SAELens SAEs are trained on TransformerLens activations, and TransformerLens
# loads GPT-2 with center_writing_weights=True, which makes the residual stream
# mean-zero along d_model. HuggingFace's stream is not centered — measured
# d_model mean 0.507 on gpt2 layer 8 — so an SAE trained on a centered stream
# was being asked about vectors it never saw.
#
# Measured, gpt2 blocks.8.hook_resid_pre, prompt "The Eiffel Tower is located
# in the city of", 11 tokens, float32, fraction of variance unexplained:
#
#     raw, no b_dec  (what shipped)   FVU 13579.24    L0 7491.5
#     raw + b_dec                     FVU 12908.35    L0 2745.4
#     centered, no b_dec              FVU     0.4219  L0 1344.0
#     centered + b_dec                FVU     0.0010  L0   60.5
#
# The top-8 features the panel plots overlapped the correct ones 2 of 8.
#
# Neither convention is universally right — Gemma Scope SAEs are trained on raw
# HuggingFace activations and must not be centered — so this is not a new
# default to hardcode. It is measured per SAE against the model it is actually
# attached to, which is the only form of the answer that is true on someone
# else's machine with someone else's SAE.
# Ordered LEAST-transforming first, and the order is load-bearing. `sorted` is
# stable, so conventions that score identically keep this order and the winner
# is the one that touched the activations least. An SAE the data cannot
# distinguish should not be silently centered: applying a transform that buys
# no measurable reconstruction is a change with nothing behind it.
CONVENTIONS: tuple[tuple[str, bool, bool], ...] = (
    ("raw", False, False),
    ("b_dec", False, True),
    ("centered", True, False),
    ("centered+b_dec", True, True),
)

# Worse than predicting the mean vector. Not a taste threshold: at FVU >= 1 the
# "reconstruction" carries less of the activation than a constant would, so the
# features are not a decomposition of anything and must not be plotted.
FVU_UNUSABLE = 1.0


@dataclass
class SAECalibration:
    """Which input convention this SAE turned out to want, and how well it did.

    `declared` is what cfg.json claimed about b_dec — None when the key is
    absent, which is the case that started this. Kept beside the measured
    answer so a disagreement is visible rather than silently overridden.
    """

    convention: str
    center: bool
    subtract_b_dec: bool
    fvu: float
    l0: float
    rel_err: float
    n_tokens: int
    declared_b_dec: bool | None
    ranked: list[tuple[str, float]]  # every convention tried, by FVU
    # Fields, not properties: the server returns this through
    # `dataclasses.asdict`, which skips properties. As a property `usable`
    # would never reach the browser and the panel would have to carry its own
    # copy of the threshold — putting the decision about whether a measurement
    # can be trusted in the one place that cannot measure it.
    usable: bool
    unusable_at: float = FVU_UNUSABLE


@dataclass
class SAEStatus:
    loaded: bool
    repo: str | None = None
    hook: str | None = None
    layer: int | None = None
    d_in: int | None = None
    d_sae: int | None = None
    # Absent until the first encode, because calibration needs real activations
    # from the model the SAE is attached to.
    calibration: SAECalibration | None = None


class SAEHandle:
    """One loaded SAE: weights on CPU float32, encode/decode helpers."""

    def __init__(
        self,
        repo: str,
        hook: str,
        point: str,
        layer: int,
        W_enc: torch.Tensor,
        b_enc: torch.Tensor,
        W_dec: torch.Tensor,
        b_dec: torch.Tensor,
        apply_b_dec_to_input: bool | None,
    ) -> None:
        self.repo, self.hook, self.layer = repo, hook, layer
        self.point = point  # resid_pre | resid_post
        self.W_enc, self.b_enc = W_enc, b_enc
        self.W_dec, self.b_dec = W_dec, b_dec
        # What cfg.json declared. None means the key was absent — which is not
        # the same as "False", and treating it as False is the bug this class
        # now measures its way out of.
        self.declared_b_dec = apply_b_dec_to_input
        self.calibration: SAECalibration | None = None
        self.d_in, self.d_sae = W_enc.shape

    @classmethod
    def load(
        cls, repo: str = DEFAULT_SAE_REPO, hook: str = DEFAULT_SAE_HOOK
    ) -> "SAEHandle":
        m = re.search(r"blocks\.(\d+)\.hook_(\w+)", hook)
        if not m:
            raise ValueError(f"Cannot parse layer index from hook name: {hook!r}")
        layer = int(m.group(1))
        point = m.group(2)
        # The hook POINT was previously discarded, so every SAE was fed the
        # residual stream ENTERING the block. For a resid_post SAE that is the
        # wrong side of the block: it produces plausible-looking features that
        # describe activations the SAE was never trained on. Reject what we
        # cannot place rather than quietly using the wrong tensor.
        if point not in ("resid_pre", "resid_post"):
            raise ValueError(
                f"Unsupported hook point {point!r} in {hook!r}. ModelMRI reads the "
                f"residual stream: use a hook_resid_pre or hook_resid_post SAE."
            )

        cfg_path = hf_hub_download(repo, f"{hook}/cfg.json")
        weights_path = hf_hub_download(repo, f"{hook}/sae_weights.safetensors")
        # read_text rather than open(...).read(): the latter leaves the handle
        # to the garbage collector, which is fine on CPython and not guaranteed
        # anywhere else.
        cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
        tensors = load_file(weights_path)

        return cls(
            repo=repo,
            hook=hook,
            point=point,
            layer=layer,
            W_enc=tensors["W_enc"].float(),
            b_enc=tensors["b_enc"].float(),
            W_dec=tensors["W_dec"].float(),
            b_dec=tensors["b_dec"].float(),
            # None when absent. `cfg.get(key, False)` is what made an undeclared
            # SAE look like one that had declined the subtraction.
            apply_b_dec_to_input=(
                bool(cfg["apply_b_dec_to_input"])
                if "apply_b_dec_to_input" in cfg
                else None
            ),
        )

    def status(self) -> SAEStatus:
        return SAEStatus(
            loaded=True,
            repo=self.repo,
            hook=self.hook,
            layer=self.layer,
            d_in=self.d_in,
            d_sae=self.d_sae,
            calibration=self.calibration,
        )

    def _prepare(self, x: torch.Tensor, center: bool, subtract: bool) -> torch.Tensor:
        if center:
            x = x - x.mean(-1, keepdim=True)
        if subtract:
            x = x - self.b_dec
        return x

    @torch.no_grad()
    def calibrate(self, x: torch.Tensor) -> SAECalibration:
        """Decide which input convention this SAE wants, by reconstructing.

        An SAE is a claim that these activations decompose into these features.
        That claim is checkable: encode, decode, and see how much of the
        variance comes back. So rather than trusting a config key that may be
        absent — or a convention inherited from whichever library trained it —
        run all four and keep the one that reconstructs.

        On the default SAE the margin is five orders of magnitude — FVU 0.0010
        against 13579.24 — so the choice is never close in practice. It can be
        exactly tied in principle, though: an SAE that reconstructs the identity
        does so under every convention. `CONVENTIONS` is therefore ordered
        least-transforming first and the sort is stable, so a tie resolves to
        the convention that touched the activations least rather than to
        whichever happened to be listed first.

        Note which tensor is the target: when a convention centers the input,
        the SAE is reconstructing the CENTERED stream, because that is the
        stream it was trained on. Scoring it against the raw one would fail
        every centered SAE for being correct.
        """
        x = x.float()
        if x.ndim != 2 or x.shape[-1] != self.d_in:
            raise ValueError(
                f"Calibration needs [S, {self.d_in}] activations, got {tuple(x.shape)}"
            )

        scored: list[tuple[str, float, bool, bool, float, float]] = []
        for name, center, subtract in CONVENTIONS:
            target = x - x.mean(-1, keepdim=True) if center else x
            feats = torch.relu(
                self._prepare(x, center, subtract) @ self.W_enc + self.b_enc
            )
            recon = feats @ self.W_dec + self.b_dec
            resid = target - recon
            denom = (target - target.mean(0)).pow(2).sum()
            fvu = (resid.pow(2).sum() / denom).item() if denom > 0 else float("inf")
            scored.append(
                (
                    name,
                    fvu,
                    center,
                    subtract,
                    feats.gt(0).float().sum(-1).mean().item(),
                    (resid.norm() / (target.norm() + 1e-12)).item(),
                )
            )

        scored.sort(key=lambda row: row[1])
        name, fvu, center, subtract, l0, rel_err = scored[0]
        self.calibration = SAECalibration(
            convention=name,
            center=center,
            subtract_b_dec=subtract,
            fvu=round(fvu, 6),
            l0=round(l0, 2),
            rel_err=round(rel_err, 6),
            n_tokens=int(x.shape[0]),
            declared_b_dec=self.declared_b_dec,
            ranked=[(row[0], round(row[1], 6)) for row in scored],
            usable=fvu < FVU_UNUSABLE,
        )
        return self.calibration

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """[S, d_in] residual activations -> [S, d_sae] feature activations.

        Calibrates on first use. The activations passed here are exactly the
        ones calibration needs, so there is nothing for a caller to arrange and
        no way to forget.
        """
        x = x.float()
        if self.calibration is None:
            self.calibrate(x)
        cal = self.calibration
        assert cal is not None
        return torch.relu(
            self._prepare(x, cal.center, cal.subtract_b_dec) @ self.W_enc + self.b_enc
        )

    @torch.no_grad()
    def decode(self, feats: torch.Tensor) -> torch.Tensor:
        """[S, d_sae] feature activations -> [S, d_in] reconstruction."""
        return feats.float() @ self.W_dec + self.b_dec

    def steering_vector(self, feature_id: int) -> torch.Tensor:
        """Unit-norm decoder direction for one feature ([d_in])."""
        v = self.W_dec[feature_id]
        return v / (v.norm() + 1e-8)
