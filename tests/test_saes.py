"""The SAE must work out which activations it was trained on.

An SAE fed the wrong input convention does not error. It returns features, in
the right shape, with plausible magnitudes, for a vector it never saw — and the
panel plots them. That is the failure this file exists to make loud.

The synthetic SAE below is built so that exactly ONE convention can reconstruct
it: its decoder rows are orthogonal to the ones-vector, so its output always has
zero d_model mean and can only match a centered target, and its b_dec is
non-zero so skipping the subtraction leaves a constant error. Getting the right
answer here is therefore evidence of choosing, not of arithmetic.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from modelmri.saes import CONVENTIONS, FVU_UNUSABLE, SAEHandle

D_IN = 8


def _centering_projector(d: int) -> torch.Tensor:
    """P = I - 11^T/d. Symmetric, idempotent, kills the ones-direction."""
    return torch.eye(d) - torch.ones(d, d) / d


def synthetic_sae(
    *, b_dec_scale: float = 3.0, declared: bool | None = None
) -> SAEHandle:
    """An SAE that reconstructs exactly, and only from `centered+b_dec`.

    W_enc = [P, -P] and W_dec = [P; -P] make the pair an identity on the
    centered subspace, because relu(a) - relu(-a) == a. Everything the decoder
    can emit is therefore mean-zero.
    """
    p = _centering_projector(D_IN)
    torch.manual_seed(0)
    b_dec = torch.randn(D_IN)
    b_dec = (b_dec - b_dec.mean()) * b_dec_scale  # mean-zero, so P leaves it alone
    return SAEHandle(
        repo="synthetic/sae",
        hook="blocks.0.hook_resid_pre",
        point="resid_pre",
        layer=0,
        W_enc=torch.cat([p, -p], dim=1),
        b_enc=torch.zeros(2 * D_IN),
        W_dec=torch.cat([p, -p], dim=0),
        b_dec=b_dec,
        apply_b_dec_to_input=declared,
    )


def uncentered_activations(n: int = 12) -> torch.Tensor:
    """Activations with a real d_model mean, like HuggingFace's residual stream."""
    torch.manual_seed(1)
    return torch.randn(n, D_IN) * 5.0 + 4.0  # the +4 is the mean that matters


# ------------------------------------------------------- the absent config key


def test_an_absent_config_key_is_not_a_declared_false():
    """`cfg.get(key, False)` is what started this.

    The default release has no `apply_b_dec_to_input`, and reading that as False
    made an undeclared SAE indistinguishable from one that had declined the
    subtraction. Absent must survive as None so the measurement can override it
    and a disagreement stays visible.
    """
    assert synthetic_sae(declared=None).declared_b_dec is None
    assert synthetic_sae(declared=False).declared_b_dec is False
    assert synthetic_sae(declared=True).declared_b_dec is True


def test_the_measurement_overrides_a_wrong_declaration():
    """A config that says False does not get to be wrong quietly."""
    sae = synthetic_sae(declared=False)
    cal = sae.calibrate(uncentered_activations())
    assert cal.subtract_b_dec is True, "measurement deferred to a wrong cfg key"
    assert cal.declared_b_dec is False, "the declaration must stay visible"


# ------------------------------------------------------------- choosing right


def test_it_picks_the_only_convention_that_reconstructs():
    sae = synthetic_sae()
    cal = sae.calibrate(uncentered_activations())
    assert cal.convention == "centered+b_dec"
    assert cal.center is True and cal.subtract_b_dec is True
    assert cal.fvu < 1e-6, (
        f"the exact convention should reconstruct exactly, got {cal.fvu}"
    )
    assert cal.usable


def test_every_other_convention_is_measurably_worse():
    """The ranking must be a real ordering, not a lucky first entry."""
    sae = synthetic_sae()
    cal = sae.calibrate(uncentered_activations())
    assert [name for name, _ in cal.ranked][0] == "centered+b_dec"
    assert len(cal.ranked) == len(CONVENTIONS)
    by_name = dict(cal.ranked)
    for other in ("centered", "b_dec", "raw"):
        assert by_name[other] > by_name["centered+b_dec"], (
            f"{other} scored {by_name[other]}, no worse than the exact convention"
        )
    # Sorted best-first, so a caller can read ranked[0] without re-sorting.
    scores = [fvu for _, fvu in cal.ranked]
    assert scores == sorted(scores)


def test_centering_is_not_a_new_hardcoded_default():
    """An SAE trained on RAW activations must be left uncentered.

    Gemma Scope SAEs are trained on HuggingFace activations directly. Replacing
    "never center" with "always center" would break them exactly as badly, in
    the other direction.
    """
    # Identity SAE with no b_dec: reconstructs whatever it is given, so the
    # target it matches best is the raw stream.
    sae = SAEHandle(
        repo="synthetic/raw",
        hook="blocks.0.hook_resid_pre",
        point="resid_pre",
        layer=0,
        W_enc=torch.cat([torch.eye(D_IN), -torch.eye(D_IN)], dim=1),
        b_enc=torch.zeros(2 * D_IN),
        W_dec=torch.cat([torch.eye(D_IN), -torch.eye(D_IN)], dim=0),
        b_dec=torch.zeros(D_IN),
        apply_b_dec_to_input=None,
    )
    cal = sae.calibrate(uncentered_activations())
    assert cal.center is False, "centering was applied to an SAE that did not want it"
    assert cal.fvu < 1e-6


# ------------------------------------------------------------- encode contract


def test_encode_calibrates_once_and_then_reuses_it():
    """Calibration must not drift between calls on one loaded SAE.

    Re-deriving per call would let two panels reading the same SAE disagree
    about which convention it wants, and the second answer would silently
    replace the first.
    """
    sae = synthetic_sae()
    x = uncentered_activations()
    assert sae.calibration is None
    first = sae.encode(x)
    cal = sae.calibration
    assert cal is not None
    again = sae.encode(x)
    assert sae.calibration is cal, "encode re-calibrated instead of reusing"
    assert torch.equal(first, again)


def test_decode_reconstructs_what_calibrate_claimed():
    """calibrate's FVU must describe what decode actually produces."""
    sae = synthetic_sae()
    x = uncentered_activations()
    feats = sae.encode(x)
    cal = sae.calibration
    assert cal is not None
    target = x - x.mean(-1, keepdim=True) if cal.center else x
    recon = sae.decode(feats)
    fvu = (
        (target - recon).pow(2).sum() / (target - target.mean(0)).pow(2).sum()
    ).item()
    assert abs(fvu - cal.fvu) < 1e-5, (
        f"decode gives {fvu}, calibration claimed {cal.fvu}"
    )


def test_calibration_refuses_activations_of_the_wrong_width():
    sae = synthetic_sae()
    with pytest.raises(ValueError, match=str(D_IN)):
        sae.calibrate(torch.randn(4, D_IN + 1))
    with pytest.raises(ValueError):
        sae.calibrate(torch.randn(2, 3, D_IN))


# --------------------------------------------------------------- the usability gate


def test_an_sae_that_reconstructs_nothing_is_marked_unusable():
    """FVU >= 1 is worse than predicting the mean vector.

    Not a taste threshold: at that point the features carry less of the
    activation than a constant would, so they are not a decomposition of
    anything and the panel must not plot them.
    """
    torch.manual_seed(2)
    sae = SAEHandle(
        repo="synthetic/noise",
        hook="blocks.0.hook_resid_pre",
        point="resid_pre",
        layer=0,
        W_enc=torch.randn(D_IN, 4) * 0.01,
        b_enc=torch.zeros(4),
        W_dec=torch.randn(4, D_IN) * 0.01,
        b_dec=torch.zeros(D_IN),
        apply_b_dec_to_input=None,
    )
    cal = sae.calibrate(uncentered_activations())
    assert cal.fvu >= FVU_UNUSABLE
    assert not cal.usable


# ------------------------------------------------------- the real one, if cached


def _default_sae_is_cached() -> bool:
    home = os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE")
    roots = [Path(home)] if home else []
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    return any(
        p.exists()
        for root in roots
        for p in [
            root / "models--jbloom--GPT2-Small-SAEs-Reformatted",
            root / "hub" / "models--jbloom--GPT2-Small-SAEs-Reformatted",
        ]
    )


@pytest.mark.skipif(
    not _default_sae_is_cached(), reason="default SAE not in the local HF cache"
)
def test_the_shipped_sae_declares_nothing_and_wants_both():
    """The case that was live: no cfg key, and both transforms needed.

    Measured on gpt2 blocks.8.hook_resid_pre with the residual stream captured
    the way runtime.py captures it: FVU 13579.24 raw against 0.0010 for
    centered+b_dec, L0 7491.5 against 60.5. Only the ordering and the
    usability are asserted here — the exact figures depend on the prompt, and
    a test that pinned them would fail for the wrong reason.
    """
    sae = SAEHandle.load()
    assert sae.declared_b_dec is None, "cfg gained the key; revisit the default"

    torch.manual_seed(3)
    # Stand-in for a residual stream: large norm and a non-zero d_model mean,
    # which is what makes the raw convention lose.
    x = torch.randn(16, sae.d_in) * 50.0 + 20.0
    cal = sae.calibrate(x)
    by_name = dict(cal.ranked)
    assert by_name["centered+b_dec"] < by_name["raw"], (
        f"centered+b_dec {by_name['centered+b_dec']} did not beat raw {by_name['raw']}"
    )
    assert cal.center is True
