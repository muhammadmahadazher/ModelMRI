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

import json
import math
import os
from pathlib import Path

import pytest
import torch

from modelmri.errors import BadRequest, Refusal
from modelmri.saes import (
    ACT_GATED,
    ACT_JUMPRELU,
    ACT_RELU,
    ACT_TOPK,
    ARCHITECTURE_ACTIVATION,
    CONVENTIONS,
    FVU_UNUSABLE,
    SAEHandle,
)

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

    Measured at blocks.8.hook_resid_pre with the residual stream captured the
    way runtime.py captures it: raw scored an FVU in the thousands against a
    small fraction of one for centered+b_dec, with an L0 to match. Only the
    ordering and the usability are asserted here — the exact figures depend on
    the prompt, and a test that pinned them would fail for the wrong reason.
    """
    # Named explicitly. There is no module-level default any more:
    # it pointed at this release, and a default that names one model
    # is what the SAE route stopped doing. The loader still opens any
    # SAELens repo — only the registry's recommendation changed — so
    # this still exercises that reader for anyone whose cache has it.
    sae = SAEHandle.load(
        "jbloom/GPT2-Small-SAEs-Reformatted", "blocks.8.hook_resid_pre"
    )
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


def test_encode_feature_is_encode_restricted_to_one_column(hub):
    """One column of W_enc, not all 24,576 — and the same number.

    `feature_ablate` asks this once per scored row, to report how much of a
    feature the SAE's encoder still reads after that feature's contribution has
    been subtracted from the stream. A full `encode` per row would turn a free
    check into 19 million multiply-adds a row; this is 768. If the two ever
    disagreed, the per-row honesty column would be measuring something other
    than the activations the ranking was taken from.

    Every architecture, not just the ReLU this was written against. The
    equality is the contract, and it is the contract for a different reason per
    gate: ReLU, JumpReLU and Gated are elementwise in the feature index, so one
    column really is one column; TopK is a rank statistic over the whole row
    and cannot be, so `encode_feature` pays for the full row there rather than
    returning a number the full encode would have zeroed. A version of this
    test that only ever exercised ReLU would keep passing while the TopK path
    was wrong, which is the failure it exists to catch. The loaders it uses
    live in the architecture section at the foot of this file.
    """
    sae = synthetic_sae()
    torch.manual_seed(11)
    x = torch.randn(6, sae.d_in) * 4.0 + 2.0
    full = sae.encode(x)
    for f in (0, 1, sae.d_sae - 1):
        assert torch.allclose(sae.encode_feature(x, f), full[:, f], atol=1e-6)

    for name, loaded in every_architecture(hub).items():
        full = loaded.encode(HAND_X)
        for f in range(HAND_D_SAE):
            assert torch.allclose(
                loaded.encode_feature(HAND_X, f), full[:, f], atol=1e-6
            ), f"{name}: feature {f} through one column is not what encode gave"


def test_encode_feature_refuses_to_calibrate_from_one_feature():
    """A convention chosen on one column is a convention chosen on nothing.

    `calibrate` compares four input conventions by how much of the stream comes
    back; a single feature's activation cannot make that comparison. Refusing
    is the difference between "you called these in the wrong order" and a
    silently wrong convention on every row that follows.
    """
    sae = synthetic_sae()
    assert sae.calibration is None
    with pytest.raises(ValueError, match="needs a calibration"):
        sae.encode_feature(torch.zeros(2, sae.d_in), 0)


# ====================================================================== JumpReLU
#
# Gemma Scope ships a fifth tensor, `threshold`, and running those SAEs through
# a plain ReLU is the same failure the top of this file is about: features, in
# the right shape, with plausible magnitudes, from a gate that was never
# applied. Measured on the real release, that is 1795 features firing per token
# instead of 67 — so it does not look broken, it looks dense.

D_SAE = 2 * D_IN


def gated_sae(threshold: float | torch.Tensor) -> SAEHandle:
    """The synthetic SAE with a JumpReLU gate bolted on.

    Same weights as `synthetic_sae`, so any difference in what it emits is the
    gate and nothing else.

    Both attributes are set, and that is the point of the second one: since the
    activation function became an explicit descriptor rather than "whatever
    `threshold is None` implies", handing this SAE a threshold no longer opens
    the JumpReLU branch on its own. A fixture that set only the tensor would
    quietly go on testing a ReLU.
    """
    sae = synthetic_sae()
    sae.activation = ACT_JUMPRELU
    sae.threshold = (
        threshold
        if isinstance(threshold, torch.Tensor)
        else torch.full((D_SAE,), float(threshold))
    )
    return sae


def test_a_threshold_gates_features_a_relu_would_have_passed():
    """The whole point of the fifth tensor.

    A feature whose pre-activation is positive but below its own threshold is
    zero, not small. Asserted against the ungated SAE on the same activations
    so the comparison isolates the gate.
    """
    x = uncentered_activations()
    ungated = synthetic_sae().encode(x)
    fired = ungated[ungated > 0]
    assert fired.numel(), "the fixture must fire something for this to mean anything"

    # A threshold above the median firing activation must silence some of them
    # and leave the rest at exactly the value the ReLU gave.
    cut = float(fired.median())
    gated = gated_sae(cut).encode(x)
    assert gated.gt(0).sum() < ungated.gt(0).sum(), "the gate passed everything"
    survivors = gated > 0
    assert torch.equal(gated[survivors], ungated[survivors]), (
        "JumpReLU must pass the pre-activation through, not rescale it"
    )
    assert (ungated[~survivors] <= cut).all(), "something above its threshold was cut"


def test_a_gate_that_never_closes_is_exactly_a_relu():
    """Same weights, threshold below every pre-activation -> identical output.

    This is what makes one encode path safe to share: the JumpReLU branch is a
    generalisation of the ReLU one rather than a second implementation that
    happens to agree today.
    """
    x = uncentered_activations()
    plain = synthetic_sae()
    wide_open = gated_sae(-1e9)
    assert torch.equal(plain.encode(x), wide_open.encode(x))


def test_encode_feature_applies_that_feature_s_own_threshold():
    """One column of W_enc has to mean one entry of `threshold`.

    The gate is per feature. Restricting the encoder to one column while
    comparing against all 16,384 thresholds would judge a feature by another
    feature's bar, and `feature_ablate` calls this once per scored row.
    """
    torch.manual_seed(21)
    sae = gated_sae(torch.rand(D_SAE) * 3.0)
    x = uncentered_activations()
    full = sae.encode(x)
    for f in (0, 1, D_SAE - 1):
        assert torch.allclose(sae.encode_feature(x, f), full[:, f], atol=1e-6)


def test_no_threshold_is_not_a_threshold_of_zero():
    """`None` and a zero vector are different answers about the weights.

    A SAELens release has no gate. Reporting one whose bar happens to be zero
    would be inventing a tensor the publisher never shipped, and `status` would
    then have to quote a span of [0.0, 0.0] as if it had been measured.
    """
    plain = synthetic_sae().status()
    assert plain.activation == "relu"
    assert plain.threshold_span is None

    gated = gated_sae(torch.linspace(1.0, 4.0, D_SAE)).status()
    assert gated.activation == "jumprelu"
    assert gated.threshold_span == [1.0, 4.0]


# ============================================ two layouts, one in-memory object
#
# Nothing below touches the network. The Gemma Scope release is a real .npz
# written into tmp_path — the format is a zip of .npy arrays and building one
# is three lines, so stubbing the BYTES rather than the reader keeps
# `_read_npz` under test instead of mocked out.
#
# The layer/width/L0 spread mirrors what google/gemma-scope-2b-pt-res actually
# publishes at layer 20 (two widths, five sparsities), including the
# `embedding/` directory that must never be offered as a layer.

FAKE_FILES = (
    "README.md",
    "embedding/width_4k/average_l0_6/params.npz",
    "layer_3/width_16k/average_l0_14/params.npz",
    "layer_20/width_16k/average_l0_22/params.npz",
    "layer_20/width_16k/average_l0_71/params.npz",
    "layer_20/width_16k/average_l0_294/params.npz",
    "layer_20/width_65k/average_l0_20/params.npz",
    "layer_20/width_65k/average_l0_61/params.npz",
)
HOOK_20 = "blocks.20.hook_resid_post"


def _weights() -> dict:
    """The five arrays a Gemma Scope file holds, at this file's toy width."""
    p = _centering_projector(D_IN)
    torch.manual_seed(0)
    b_dec = torch.randn(D_IN)
    b_dec = (b_dec - b_dec.mean()) * 3.0
    return {
        "W_enc": torch.cat([p, -p], dim=1).numpy(),
        "b_enc": torch.zeros(D_SAE).numpy(),
        "W_dec": torch.cat([p, -p], dim=0).numpy(),
        "b_dec": b_dec.numpy(),
        # Below every pre-activation these fixtures produce, so the gate is
        # open and the two layouts are comparable arithmetic. The gate itself
        # is tested above, on the same weights.
        "threshold": torch.full((D_SAE,), -1e9).numpy(),
    }


class FakeHub:
    """A repo that exists only as a list of names and files written on demand."""

    def __init__(self, tmp_path, files, *, sae_lens_hook=None):
        self.tmp_path = tmp_path
        self.files = list(files)
        self.sae_lens_hook = sae_lens_hook
        self.requested: list[str] = []
        self.listings = 0
        self.listing_fails = False
        #: What `{hook}/cfg.json` holds. The empty object is not a placeholder
        #: — it is close to what a pre-v3 SAELens release actually ships, and
        #: it is the case that must keep loading as a plain ReLU.
        self.cfg: dict = {}
        #: What `{hook}/sae_weights.safetensors` holds. None means the four
        #: tensors every architecture shares; an architecture that carries
        #: more (a JumpReLU threshold, a Gated release's three) sets this.
        self.weights: dict | None = None

    # --- the HfApi half
    def list_repo_files(self, repo):
        self.listings += 1
        if self.listing_fails:
            raise ConnectionError("no network in this test")
        return list(self.files)

    def __call__(self, *_args, **_kw):  # HfApi() -> self
        return self

    # --- the hf_hub_download half
    def download(self, repo, name, **_kw):
        import numpy as np
        from huggingface_hub.errors import EntryNotFoundError
        from safetensors.torch import save_file

        self.requested.append(name)
        if name.endswith("params.npz"):
            if name not in self.files:
                raise EntryNotFoundError(f"{name} is not in {repo}")
            path = self.tmp_path / name.replace("/", "__")
            if not path.exists():
                np.savez(path, **_weights())
            return str(path)
        if self.sae_lens_hook and name.startswith(f"{self.sae_lens_hook}/"):
            leaf = name.rsplit("/", 1)[-1]
            path = self.tmp_path / f"{self.sae_lens_hook.replace('.', '_')}__{leaf}"
            if leaf == "cfg.json":
                path.write_text(json.dumps(self.cfg), encoding="utf-8")
            elif self.weights is not None:
                save_file(dict(self.weights), path)
            else:
                arrays = _weights()
                save_file(
                    {
                        k: torch.from_numpy(arrays[k])
                        for k in ("W_enc", "b_enc", "W_dec", "b_dec")
                    },
                    path,
                )
            return str(path)
        raise EntryNotFoundError(f"{name} is not in {repo}")


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """A stubbed Hub, and a cleared index cache.

    `saes._INDEX_CACHE` is module level and keyed by repo id, which is right
    for a server that loads two SAEs from one release and wrong for a suite
    where every test uses the same fake repo id. One test's listing must not
    become the next test's answer.
    """
    from modelmri import saes

    saes._INDEX_CACHE.clear()
    fake = FakeHub(tmp_path, FAKE_FILES)
    monkeypatch.setattr(saes, "HfApi", fake)
    monkeypatch.setattr(saes, "hf_hub_download", fake.download)
    yield fake
    saes._INDEX_CACHE.clear()


def test_the_npz_reader_reads_the_keys_the_real_release_ships(hub):
    """W_enc [d_in, d_sae], W_dec [d_sae, d_in], and a threshold.

    The orientation is the load-bearing half: `encode` is `x @ W_enc` and
    `decode` is `feats @ W_dec`, so a reader that transposed either one would
    still produce a tensor of plausible numbers.
    """
    sae = SAEHandle.load("fake/gemma-scope", HOOK_20)
    assert sae.W_enc.shape == (D_IN, D_SAE)
    assert sae.W_dec.shape == (D_SAE, D_IN)
    assert sae.b_enc.shape == (D_SAE,)
    assert sae.b_dec.shape == (D_IN,)
    assert sae.threshold is not None and sae.threshold.shape == (D_SAE,)
    assert (sae.d_in, sae.d_sae) == (D_IN, D_SAE)
    assert sae.W_enc.dtype is torch.float32


def test_both_layouts_produce_the_same_object_and_the_same_encode(hub):
    """Two formats, one interface — asserted on the output, not the plumbing.

    Same weights written twice, once as a Gemma Scope .npz and once as a
    SAELens .safetensors, must encode identically. If the two ever went through
    different code, the convention search that makes this module trustworthy
    would exist in two copies and only one of them would be the one that runs.
    """
    hub.sae_lens_hook = HOOK_20
    from_safetensors = SAEHandle.load("fake/sae-lens", HOOK_20)
    hub.sae_lens_hook = None
    from_npz = SAEHandle.load("fake/gemma-scope", HOOK_20)

    assert type(from_npz) is type(from_safetensors)
    x = uncentered_activations()
    assert torch.equal(from_safetensors.encode(x), from_npz.encode(x))
    assert from_safetensors.calibration.convention == from_npz.calibration.convention, (
        "the same weights chose different input conventions through the two readers"
    )


def test_a_gemma_scope_repo_is_found_without_being_named(hub):
    """The layout comes from the repo, not from its id.

    A repo called "gemma-scope-anything" that shipped safetensors would be
    opened by the wrong reader if the choice were made on the name, and this
    fixture's repo id says nothing about either layout.
    """
    sae = SAEHandle.load("acme/whatever", HOOK_20)
    assert sae.release.layout == "gemma_scope"
    assert f"{HOOK_20}/cfg.json" in hub.requested, "the SAELens layout was never tried"


# ------------------------------------------------- the default has to say so


def test_the_default_release_names_the_rule_and_what_it_beat(hub):
    """A silently chosen sparsity is the defect this feature exists to avoid.

    layer 20 publishes two widths and five sparsities at the narrower one.
    Loading without naming either must report which it took, why, and what the
    alternatives were — otherwise the panel shows a number that looks like a
    property of the model and is a property of a directory name.
    """
    sae = SAEHandle.load("fake/gemma-scope", HOOK_20)
    rel = sae.release
    assert rel.width == "width_16k", "the narrowest published width was not chosen"
    assert rel.advertised_l0 == 71, "the median of 22/71/294 was not chosen"
    assert rel.file == "layer_20/width_16k/average_l0_71/params.npz"

    assert "default" in rel.chosen_by["width"]
    assert "width_65k" in rel.chosen_by["width"], "the rejected width is not named"
    assert "default" in rel.chosen_by["average_l0"]
    for value in ("22", "71", "294"):
        assert value in rel.chosen_by["average_l0"], (
            f"{value} not offered to the reader"
        )


def test_the_index_offered_is_the_one_for_this_layer(hub):
    """`available` is what a picker renders, so it has to be the real list."""
    rel = SAEHandle.load("fake/gemma-scope", HOOK_20).release
    assert rel.available == {
        "width_16k": [22, 71, 294],
        "width_65k": [20, 61],
    }


def test_the_embedding_release_is_never_offered_as_a_layer(hub):
    """`embedding/width_4k/...` exists in the real repo and has no hook name.

    Nothing here can address it — `blocks.N.hook_*` is the only vocabulary the
    loader has — so listing it as a layer would offer a release that cannot be
    loaded.
    """
    from modelmri.saes import release_index

    index = release_index("fake/gemma-scope")
    assert sorted(index) == [3, 20]


def test_the_caller_can_pick_and_picking_is_recorded_as_picking(hub):
    """The reader who chose must be distinguishable from the reader who did not."""
    sae = SAEHandle.load("fake/gemma-scope", HOOK_20, width="width_65k", average_l0=20)
    rel = sae.release
    assert (rel.width, rel.advertised_l0) == ("width_65k", 20)
    assert rel.chosen_by["width"] == "caller"
    assert rel.chosen_by["average_l0"] == "caller"
    assert rel.file == "layer_20/width_65k/average_l0_20/params.npz"


def test_a_bare_width_label_is_accepted(hub):
    """A panel shows "65k"; the repo spells it "width_65k".

    Rejecting the shorter form would 404 on the exact string the reader was
    just shown.
    """
    rel = SAEHandle.load("fake/gemma-scope", HOOK_20, width="65k").release
    assert rel.width == "width_65k"
    assert rel.advertised_l0 == 20, "the median of [20, 61] is the lower middle"


def test_naming_only_one_coordinate_defaults_the_other_out_loud(hub):
    """Half a choice is still a choice, and the other half still has to speak."""
    rel = SAEHandle.load("fake/gemma-scope", HOOK_20, average_l0=294).release
    assert rel.chosen_by["average_l0"] == "caller"
    assert "default" in rel.chosen_by["width"]


# ----------------------------------------------------- refusing, in sentences


def test_a_layer_nobody_published_names_the_ones_that_exist(hub):
    with pytest.raises(BadRequest) as caught:
        SAEHandle.load("fake/gemma-scope", "blocks.99.hook_resid_post")
    assert "3, 20" in caught.value.sentence


def test_a_width_nobody_published_names_the_ones_that_exist(hub):
    with pytest.raises(BadRequest) as caught:
        SAEHandle.load("fake/gemma-scope", HOOK_20, width="width_999k")
    assert "width_16k" in caught.value.sentence
    assert "width_65k" in caught.value.sentence


def test_an_l0_nobody_published_names_the_ones_that_exist(hub):
    with pytest.raises(BadRequest) as caught:
        SAEHandle.load("fake/gemma-scope", HOOK_20, average_l0=1234)
    assert "22, 71, 294" in caught.value.sentence


def test_a_bool_is_not_an_average_l0(hub):
    """`isinstance(True, int)` is True, so True would load average_l0 1.

    And it would be reported as `chosen_by: caller`, which is the part that
    makes it worse than an error: a sparsity nobody asked for, labelled as a
    deliberate choice.
    """
    with pytest.raises(BadRequest, match="not a flag"):
        SAEHandle.load("fake/gemma-scope", HOOK_20, average_l0=True)


def test_gemma_scope_refuses_the_entering_side_of_the_block(hub):
    """`layer_20` is the stream LEAVING block 20; there is no resid_pre release.

    Serving one against the block's input is the failure the hook-point split
    was written for, one layout further along.
    """
    with pytest.raises(BadRequest) as caught:
        SAEHandle.load("fake/gemma-scope", "blocks.20.hook_resid_pre")
    assert "hook_resid_post" in caught.value.sentence


def test_a_params_file_missing_a_key_is_refused_by_name(hub, tmp_path):
    """A .npz with four of the five arrays is not a Gemma Scope release."""
    import numpy as np

    from modelmri.saes import _read_npz

    arrays = _weights()
    del arrays["threshold"]
    path = tmp_path / "short.npz"
    np.savez(path, **arrays)
    with pytest.raises(Refusal, match="threshold"):
        _read_npz(str(path), "fake/gemma-scope", "layer_20/.../params.npz")


# ------------------------------------------------- unknown is not empty


def test_an_unread_index_is_not_an_empty_one(hub):
    """`None` and `{}` are different answers and must stay different.

    `{}` means the listing was read and holds no releases — how a SAELens repo
    answers. `None` means nobody reached the Hub. Flattening them would let a
    picker render "this repo publishes nothing" about a repo it never asked.
    """
    from modelmri.saes import release_index

    hub.listing_fails = True
    assert release_index("fake/gemma-scope") is None

    hub.listing_fails = False
    hub.files = ["README.md"]
    assert release_index("fake/gemma-scope") == {}


def test_an_unreadable_index_refuses_to_invent_a_default(hub):
    """No listing means no alternatives, and a default with nothing behind it."""
    hub.listing_fails = True
    with pytest.raises(Refusal) as caught:
        SAEHandle.load("fake/gemma-scope", HOOK_20)
    assert "width" in caught.value.sentence


def test_an_unreadable_index_still_loads_a_fully_named_release(hub):
    """Both coordinates given determines the path, so the listing is not needed.

    `available` then comes back None — unknown — rather than an empty dict that
    would read as "this layer has only the one you asked for".
    """
    hub.listing_fails = True
    rel = SAEHandle.load(
        "fake/gemma-scope", HOOK_20, width="width_16k", average_l0=71
    ).release
    assert rel.available is None
    assert "unknown rather than empty" in rel.chosen_by["available"]


def test_a_repo_with_neither_layout_says_so(hub):
    hub.files = ["README.md", "model.bin"]
    with pytest.raises(Refusal) as caught:
        SAEHandle.load("fake/nothing", HOOK_20)
    assert "SAELens" in caught.value.sentence


# ------------------------------------------------------ the default rules alone


def test_the_width_default_is_the_narrowest_and_unreadable_labels_sort_last():
    """An unparseable label must not become the default by scoring zero."""
    from modelmri.saes import _pick_width

    chosen, why = _pick_width(
        {"width_65k": [1], "width_16k": [1], "width_1m": [1], "width_wat": [1]}, 20
    )
    assert chosen == "width_16k"
    assert "4 widths" in why


def test_the_l0_default_is_the_lower_middle_when_the_count_is_even():
    """Deterministic, so two loads of the same release agree."""
    from modelmri.saes import _pick_l0

    assert _pick_l0([22, 38, 71, 139, 294], "width_16k")[0] == 71
    assert _pick_l0([20, 34, 61, 114, 221, 400], "width_65k")[0] == 61
    assert _pick_l0([9], "width_4k")[0] == 9


# --------------------------------------------------------------- the registry


def test_the_registry_does_not_bake_in_a_sparsity():
    """A hardcoded L0 in the table is the thing this feature exists to prevent.

    The table may say a release is indexed by width and average L0; it may not
    say which. Those come off the Hub at load time so they cannot go stale in
    a source file, and so nobody's sparsity is chosen by this repository.
    """
    from modelmri import sae_registry

    for entry in sae_registry.catalogue():
        if entry["layout"] != "gemma_scope":
            continue
        assert entry["indexed_by"] == ["width", "average_l0"]
        for key, value in entry.items():
            assert "average_l0_" not in str(value), (
                f"{entry['repo']} pins a release in {key}"
            )


# ======================================== the gate the cfg.json declares
#
# `_read_sae_lens` used to return `threshold=None` for every SAELens release,
# on the stated assumption that they are all plain ReLU. That was true of the
# 2023 releases and is false of the modern ones: SAELens registers standard,
# gated, jumprelu and topk inference classes, and cfg.json names which. Loading
# a TopK SAE without its gate is the same failure as the input convention at
# the top of this file — right shape, plausible magnitudes, a rule that was
# never applied — except that TopK hides better, because it ships the same four
# tensors a standard release does and nothing in the weight file betrays it.
#
# Everything below is hand-computed. The fixture is four inputs by six features
# with small exact values; its b_dec is zero and its inputs are already
# mean-zero along d_model, so all four conventions prepare the input
# identically, the tie resolves to `raw` (stable sort, least-transforming
# first), and `prepare` is therefore the identity. What `encode` returns is
# exactly `_activate(x @ W_enc)`, which fits on paper.

HAND_D_IN, HAND_D_SAE = 4, 6

#: [d_in, d_sae]. Columns 0-3 each select one input dimension, column 4 is
#: minus dimension 0, and column 5 is minus HALF of dimension 1. The half is
#: deliberate: it keeps every pre-activation in the fixture distinct, and
#: torch.topk does not specify which of two exactly equal values it keeps.
HAND_W_ENC = torch.tensor(
    [
        [1.0, 0.0, 0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, -0.5],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    ]
)

#: Two tokens whose d_model mean is exactly zero, so centering is exactly the
#: identity rather than nearly it and the hand arithmetic is exact.
HAND_X = torch.tensor([[1.0, -1.0, 2.0, -2.0], [3.0, 1.0, -2.0, -2.0]])

#: HAND_X @ HAND_W_ENC, done by hand. Asserted below before anything relies on
#: it, because every expectation in this section is derived from it.
HAND_PRE = torch.tensor(
    [
        [1.0, -1.0, 2.0, -2.0, -1.0, 0.5],
        [3.0, 1.0, -2.0, -2.0, -3.0, -0.5],
    ]
)

#: relu(HAND_PRE). The baseline every gated architecture must differ from,
#: because "loads as a plain ReLU" is precisely the defect.
HAND_RELU = torch.tensor(
    [
        [1.0, 0.0, 2.0, 0.0, 0.0, 0.5],
        [3.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    ]
)

#: Per feature, and chosen so the gate closes on some features a ReLU passed.
HAND_THRESHOLD = torch.tensor([0.5, 0.5, 1.5, 0.5, 0.5, 1.5])

#: What the JumpReLU and top-2 fixtures both emit, for different reasons.
HAND_GATED_TO_TWO = torch.tensor(
    [
        [1.0, 0.0, 2.0, 0.0, 0.0, 0.0],
        [3.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    ]
)


def _hand_weights(**replaced) -> dict:
    """The four tensors every architecture ships, plus whatever a gate needs."""
    out = {
        "W_enc": HAND_W_ENC.clone(),
        "b_enc": torch.zeros(HAND_D_SAE),
        "W_dec": HAND_W_ENC.t().contiguous(),
        "b_dec": torch.zeros(HAND_D_IN),
    }
    out.update(replaced)
    return {k: v for k, v in out.items() if v is not None}


def load_sae_lens(hub, cfg, *, weights=None, hook=HOOK_20, repo="fake/sae-lens"):
    """Load a SAELens-layout release with this cfg.json and these tensors."""
    hub.sae_lens_hook = hook
    hub.cfg = cfg
    hub.weights = _hand_weights() if weights is None else weights
    return SAEHandle.load(repo, hook)


def gated_weights() -> dict:
    """A Gated release: no b_enc, and three tensors a standard release lacks.

    `r_mag` is log 2 on feature 1, so `exp(r_mag)` doubles that column of the
    magnitude encoder and nothing else. `b_gate` is -2 on feature 0, which
    closes its gate on a token whose magnitude path is positive — so this
    fixture exercises both halves of the gated encode, in opposite directions.
    """
    return _hand_weights(
        b_enc=None,
        b_gate=torch.tensor([-2.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        r_mag=torch.tensor([0.0, math.log(2.0), 0.0, 0.0, 0.0, 0.0]),
        b_mag=torch.zeros(HAND_D_SAE),
    )


def every_architecture(hub) -> dict:
    """One loaded SAE per architecture this module can encode.

    Each comes from its own fake repo id so one FakeHub can answer for all
    four: the hook path is what the loader asks for, and the repo id is the
    only thing distinguishing the four cfg.json bodies it is handed.
    """
    out = {}
    for name, cfg, weights in (
        (ACT_RELU, {"architecture": "standard"}, None),
        (
            ACT_JUMPRELU,
            {"architecture": "jumprelu"},
            _hand_weights(threshold=HAND_THRESHOLD.clone()),
        ),
        (ACT_TOPK, {"architecture": "topk", "k": 2}, None),
        (ACT_GATED, {"architecture": "gated"}, gated_weights()),
    ):
        out[name] = load_sae_lens(hub, cfg, weights=weights, repo=f"fake/{name}")
    return out


#: The signature phrase of each FALLBACK refusal in modelmri/saes.py — the
#: branch that fires when a value is none of the ones named above it. Every
#: one of the four QUOTES the value it is refusing, which is exactly why "the
#: offending value appears in the sentence" proves nothing: delete the by-name
#: branch and the fallback catches the same input and echoes the same word, on
#: a sentence that now says something FALSE about it ("tanh-relu is not an
#: activation function SAELens ever defined" — v5 defined it).
#:
#: Measured rather than reasoned. Every entry here was found by deleting a
#: branch and watching a green test stay green.
FALLBACK_REFUSALS = {
    "unknown architecture": ", ".join(sorted(ARCHITECTURE_ACTIVATION)),
    "unknown normalize_activations": "is not one of the values SAELens defines",
    "unknown reshape_activations": "is not a reshaping this knows",
    "unknown activation_fn_str": "is not an activation function SAELens ever defined",
}


def refused_by_name(sentence: str) -> str:
    """The sentence back, having asserted it is not one of the four fallbacks.

    Returns it so a test reads as one line —
    `assert "b_gate" in refused_by_name(caught.value.sentence)` — and so the
    thing being asserted about is visibly the thing that was checked.
    """
    for what, phrase in FALLBACK_REFUSALS.items():
        assert phrase not in sentence, (
            f"refused as an {what}, which may well be true and is not the "
            f"reason this test exists:\n{sentence}"
        )
    return sentence


def test_the_hand_computed_pre_activations_are_the_ones_the_matmul_gives():
    """Every expectation below is derived from HAND_PRE, so check it first.

    If the fixture drifts this fails once and says so, rather than every
    architecture test failing separately with numbers nobody can source.
    """
    assert torch.equal(HAND_X @ HAND_W_ENC, HAND_PRE)
    assert torch.equal(torch.relu(HAND_PRE), HAND_RELU)
    assert torch.equal(HAND_X.mean(-1), torch.zeros(2)), (
        "the inputs must be exactly mean-zero or centering is not the identity"
    )


def test_the_hand_fixture_ties_every_convention_so_prepare_is_the_identity(hub):
    """The premise of every hand-computed expectation in this section.

    b_dec is zero and the input is already centered, so all four conventions
    prepare the same tensor and score the same FVU. The tie resolves to `raw`
    because CONVENTIONS is ordered least-transforming first and the sort is
    stable — documented behaviour, asserted here because the hand arithmetic
    depends on it.
    """
    sae = load_sae_lens(hub, {"architecture": "standard"})
    cal = sae.calibrate(HAND_X)
    assert cal.convention == "raw"
    assert len({fvu for _, fvu in cal.ranked}) == 1, "the conventions did not tie"


# ---------------------------------------------------- one architecture at a time


def test_a_standard_release_still_loads_as_a_plain_relu(hub):
    """The path that was already right must not move."""
    sae = load_sae_lens(hub, {"architecture": "standard"})
    assert sae.status().activation == ACT_RELU
    assert sae.status().k is None
    assert sae.threshold is None
    assert torch.equal(sae.encode(HAND_X), HAND_RELU)


def test_a_cfg_that_names_nothing_is_the_legacy_standard_release(hub):
    """Pre-v3 SAELens wrote no architecture key and had only one architecture.

    Absent is not guessed here — it is READ, off a schema that predates the
    key. The sentence saying so travels with the release, so a reader can tell
    "this file says standard" from "this file is old enough that standard is
    the only thing it could be".
    """
    sae = load_sae_lens(hub, {"hook_point": "blocks.20.hook_resid_post"})
    assert sae.status().activation == ACT_RELU
    assert torch.equal(sae.encode(HAND_X), HAND_RELU)
    why = sae.release.chosen_by["architecture"]
    assert "standard" in why and "no architecture" in why


def test_a_modern_cfg_with_no_architecture_is_refused(hub):
    """SAELens 6 writes `architecture` unconditionally, so its absence is wrong.

    Defaulting it to standard here would be the same mistake as reading an
    absent `apply_b_dec_to_input` as False: a claim the file never made,
    resolved in favour of the gate that does nothing.
    """
    with pytest.raises(Refusal) as caught:
        load_sae_lens(hub, {"metadata": {"sae_lens_version": "6.27.3"}})
    sentence = refused_by_name(caught.value.sentence)
    assert "architecture" in sentence
    # The reason, not just the word. The unknown-architecture fallback also
    # says "architecture" — it would quote the None — so the assertion above
    # on its own would pass on a sentence claiming this file named something
    # unreadable rather than that it named nothing at all.
    assert "SAELens 6 or later" in sentence


def test_a_jumprelu_release_loads_the_threshold_it_ships(hub):
    """The cheapest half of the fix: the tensor was there and nobody looked."""
    sae = load_sae_lens(
        hub,
        {"architecture": "jumprelu"},
        weights=_hand_weights(threshold=HAND_THRESHOLD.clone()),
    )
    status = sae.status()
    assert status.activation == ACT_JUMPRELU
    assert status.threshold_span == [0.5, 1.5]
    assert torch.equal(sae.encode(HAND_X), HAND_GATED_TO_TWO)
    assert not torch.equal(sae.encode(HAND_X), HAND_RELU), (
        "the gate passed everything a ReLU would have"
    )


def test_a_pre_activation_exactly_equal_to_its_threshold_does_not_fire(hub):
    """The JumpReLU comparison is STRICTLY greater, and nothing pinned it.

    HAND_THRESHOLD is never equal to a HAND_PRE entry, so `pre > threshold`
    written as `pre >= threshold` passed every other test in this file. One
    threshold moved onto a pre-activation is the whole difference: feature 0's
    first token sits exactly on its bar and must be zero, and the same run has
    to keep the features that clear theirs so the test cannot pass by gating
    everything shut.
    """
    threshold = HAND_THRESHOLD.clone()
    threshold[0] = 1.0  # exactly HAND_PRE[0, 0]
    sae = load_sae_lens(
        hub, {"architecture": "jumprelu"}, weights=_hand_weights(threshold=threshold)
    )
    assert torch.equal(
        sae.encode(HAND_X),
        torch.tensor([[0.0, 0.0, 2.0, 0.0, 0.0, 0.0], [3.0, 1.0, 0.0, 0.0, 0.0, 0.0]]),
    )


def test_a_gating_pre_activation_of_exactly_zero_keeps_the_gate_shut(hub):
    """The Gated gate is STRICTLY greater than zero, and nothing pinned that
    either — `gated_weights` puts no gating pre-activation on the boundary.

    Feature 0's first token lands exactly on 0 here, so `> 0` zeroes a
    magnitude of 1.0 and `>= 0` passes it. Feature 5 on the same token clears
    its gate, so a gate stuck shut fails too.
    """
    weights = _hand_weights(
        b_enc=None,
        b_gate=torch.tensor([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        r_mag=torch.zeros(HAND_D_SAE),
        b_mag=torch.zeros(HAND_D_SAE),
    )
    sae = load_sae_lens(hub, {"architecture": "gated"}, weights=weights)
    assert torch.allclose(
        sae.encode(HAND_X),
        torch.tensor([[0.0, 0.0, 2.0, 0.0, 0.0, 0.5], [3.0, 1.0, 0.0, 0.0, 0.0, 0.0]]),
        atol=1e-6,
    )


def test_a_jumprelu_release_without_its_threshold_is_refused(hub):
    """No threshold, no JumpReLU. Falling back to ReLU is the whole defect."""
    with pytest.raises(Refusal) as caught:
        load_sae_lens(hub, {"architecture": "jumprelu"})
    sentence = refused_by_name(caught.value.sentence)
    assert "jumprelu" in sentence
    assert "threshold" in sentence


def test_a_topk_release_is_never_loaded_as_a_plain_relu(hub):
    """The regression that matters, and the one no weight file betrays.

    A TopK release ships exactly the four tensors a standard one ships. The
    only evidence it is gated at all is cfg.json, so a loader that parses
    cfg.json for nothing loads it wide open — here that is feature 5 on the
    first token, which a ReLU passes at 0.5 and the top-2 gate zeroes.
    """
    sae = load_sae_lens(hub, {"architecture": "topk", "k": 2})
    status = sae.status()
    assert status.activation == ACT_TOPK
    assert status.k == 2
    assert torch.equal(sae.encode(HAND_X), HAND_GATED_TO_TWO)
    assert not torch.equal(sae.encode(HAND_X), HAND_RELU), (
        "a top-2 gate returned what a ReLU would have"
    )


def test_topk_selects_on_pre_activations_so_fewer_than_k_can_fire(hub):
    """L0 <= k, and it is allowed to be less. Do not "fix" that.

    SAELens takes the top k of the PRE-activations and applies the ReLU to the
    winners afterwards, so a selected feature whose pre-activation is negative
    is written as exactly zero. Rounding that up to k firing features would
    invent activations the SAE never emits.
    """
    sae = load_sae_lens(hub, {"architecture": "topk", "k": 4})
    feats = sae.encode(HAND_X)
    # Row 0 has three positive pre-activations and row 1 has two; k is four.
    assert feats.gt(0).sum(-1).tolist() == [3, 2]
    # Both rows have an exact tie at the k-th place (-1.0 twice in the first,
    # -2.0 twice in the second), and torch.topk does not specify which of two
    # equal values it keeps. It does not matter here and it never matters:
    # whichever loses is written as a zero and whichever wins is relu'd to a
    # zero, so the returned activations are the same either way.
    assert torch.equal(feats, HAND_RELU)


def test_a_topk_release_with_no_k_is_refused_by_name(hub):
    """There is no defensible default k: SAELens's dataclass default of 100 is
    a placeholder, not a property of anybody's release.

    `"k" in sentence` is worth nothing on its own and this test used to rest
    on it: the hook every sentence here names is `blocks.20.hook_resid_post`,
    so the letter k is in every refusal this module can raise. The phrase
    below is only in the branch this test is about.
    """
    with pytest.raises(Refusal) as caught:
        load_sae_lens(hub, {"architecture": "topk"})
    sentence = refused_by_name(caught.value.sentence)
    assert "topk" in sentence
    assert "k IS the gate" in sentence


def test_a_topk_k_wider_than_the_dictionary_is_refused(hub):
    """k is how many of d_sae features may fire, so k > d_sae describes a
    different weight file than the one that shipped beside this cfg."""
    with pytest.raises(Refusal) as caught:
        load_sae_lens(hub, {"architecture": "topk", "k": HAND_D_SAE + 1})
    assert str(HAND_D_SAE) in caught.value.sentence


def test_a_topk_k_of_zero_is_refused(hub):
    """Zero features may fire is not a sparsity, it is an SAE that emits
    nothing — and torch.topk would answer it without complaint.

    The value is asserted into the sentence, not merely the raising: `if k is
    None` written as `if not k` refuses a zero too, with the sentence "names
    no k", which is false about a cfg that named 0. A bare `pytest.raises`
    cannot tell those two apart.
    """
    with pytest.raises(Refusal) as caught:
        load_sae_lens(hub, {"architecture": "topk", "k": 0})
    assert "k=0" in refused_by_name(caught.value.sentence)


def test_a_topk_k_that_is_a_boolean_is_refused(hub):
    """`isinstance(True, int)` is True, so a stray bool is a top-1 gate.

    json.load produces one from a cfg that wrote `"k": true`, and a top-1 gate
    is a plausible-looking SAE: one feature per token, an L0 of 1.0 in the
    panel, and a published sparsity that nobody published. The same hazard on
    the Gemma Scope side already had a test; this side did not.
    """
    with pytest.raises(Refusal) as caught:
        load_sae_lens(hub, {"architecture": "topk", "k": True})
    assert "not a count of features" in refused_by_name(caught.value.sentence)


def test_a_topk_k_equal_to_the_dictionary_is_exactly_a_relu(hub):
    """Every feature is in the top d_sae, so the selection is a no-op.

    Arithmetic, not a special case: the scatter writes every column and the
    ReLU is applied to all of them. Asserted rather than assumed because a
    reader meeting `k == d_sae` deserves to know it degenerates rather than
    raises, and torch.topk is happy to be asked for a whole row.
    """
    sae = load_sae_lens(hub, {"architecture": "topk", "k": HAND_D_SAE})
    assert torch.equal(sae.encode(HAND_X), HAND_RELU)


def test_the_older_schema_promotes_activation_fn_str_topk(hub):
    """v3-v5 wrote `activation_fn_str` + `activation_fn_kwargs`, not `k`.

    SAELens's own migration promotes exactly that pair to architecture "topk",
    and a release written by v5 is not a rarity — reading only the modern key
    would load every one of them as a plain ReLU.
    """
    sae = load_sae_lens(
        hub,
        {
            "architecture": "standard",
            "activation_fn_str": "topk",
            "activation_fn_kwargs": {"k": 2},
        },
    )
    assert sae.status().activation == ACT_TOPK
    assert sae.status().k == 2
    assert torch.equal(sae.encode(HAND_X), HAND_GATED_TO_TWO)


def test_the_older_schema_topk_without_a_k_is_refused(hub):
    """SAELens leaves this one as standard and drops the activation function.

    That is a silent wrong load in SAELens itself, and copying it would be
    copying the defect this file exists to remove.
    """
    with pytest.raises(Refusal) as caught:
        load_sae_lens(hub, {"activation_fn_str": "topk", "activation_fn_kwargs": {}})
    # Not `"k" in sentence`, which is what this asserted and which is true of
    # every refusal in the module — the hook name alone contains three of
    # them. `activation_fn_kwargs` is named by this branch and by no other.
    assert "activation_fn_kwargs" in refused_by_name(caught.value.sentence)


def test_tanh_relu_is_refused_because_the_two_libraries_disagree(hub):
    """v5 computed tanh(relu(pre)); v6 drops the value and computes relu(pre).

    Neither answer can be trusted without knowing which version trained the
    release, so this names the disagreement rather than picking a side.
    """
    with pytest.raises(Refusal) as caught:
        load_sae_lens(hub, {"activation_fn_str": "tanh-relu"})
    sentence = refused_by_name(caught.value.sentence)
    assert "tanh-relu" in sentence
    # Quoting the value is not enough: the unknown-activation_fn_str fallback
    # quotes it too, on a sentence saying SAELens never defined it — which is
    # false, v5 defined it, and is the opposite of the reason here.
    assert "tanh(relu(pre))" in sentence


def test_a_gated_release_encodes_through_both_of_its_paths(hub):
    """One shared W_enc, two biases, and a per-feature exp(r_mag) on one path.

    Hand-computed, and the two entries that matter pull in opposite
    directions. Feature 0 on the first token has a positive magnitude and a
    closed gate, so it must be 0.0 rather than the 1.0 a ReLU gives; feature 1
    on the second token has an open gate and r_mag = log 2, so it must be 2.0
    rather than 1.0. Getting both right is evidence of the gated encode rather
    than of arithmetic that happens to agree.

    allclose rather than equal, for the doubled entry alone: exp(log 2) in
    float32 is 2.0 to within an ulp, not necessarily exactly 2.0.
    """
    sae = load_sae_lens(hub, {"architecture": "gated"}, weights=gated_weights())
    assert sae.status().activation == ACT_GATED
    feats = sae.encode(HAND_X)
    expected = torch.tensor(
        [
            [0.0, 0.0, 2.0, 0.0, 0.0, 0.5],
            [3.0, 2.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    assert torch.allclose(feats, expected, atol=1e-6), feats
    assert feats[0, 0] == 0.0, "a closed gate must zero a positive magnitude"
    assert not torch.allclose(feats, HAND_RELU, atol=1e-6)


def test_a_gated_release_missing_its_gate_tensors_is_refused_by_name(hub):
    """A Gated file ships b_gate, r_mag and b_mag, and no b_enc at all.

    One that declares gated and ships the standard four is not a file this can
    encode, and loading it as a standard SAE — which is what asking only for
    W_enc/b_enc/W_dec/b_dec would have done — is the silent wrong load.
    """
    with pytest.raises(Refusal) as caught:
        load_sae_lens(hub, {"architecture": "gated"})
    sentence = refused_by_name(caught.value.sentence)
    for key in ("b_gate", "r_mag", "b_mag"):
        assert key in sentence


def test_batchtopk_as_an_architecture_string_is_refused_by_name(hub):
    """SAELens registers batchtopk for TRAINING only.

    There is no inference class and SAELens's own registry lookup raises on it.
    A released BatchTopK SAE is saved as jumprelu with a distilled threshold;
    a cfg literally saying batchtopk is a training checkpoint, whose gate ranks
    activations across a whole batch and so is not a function of one token.
    """
    with pytest.raises(Refusal) as caught:
        load_sae_lens(hub, {"architecture": "batchtopk", "k": 2})
    sentence = refused_by_name(caught.value.sentence)
    assert "batchtopk" in sentence
    assert "jumprelu" in sentence


def test_a_batchtopk_trained_release_loads_as_the_jumprelu_it_was_saved_as(hub):
    """The other half of the same fact, and it must NOT refuse.

    A BatchTopK SAE is released with architecture "jumprelu" and a threshold
    that is one distilled scalar broadcast across every feature. The encode is
    JumpReLU exactly; what a reader deserves is being told the threshold is
    constant by construction, so a flat span is not read as a bug.
    """
    sae = load_sae_lens(
        hub,
        {
            "architecture": "jumprelu",
            "metadata": {
                "sae_lens_version": "6.27.3",
                "training_architecture": "batchtopk",
            },
        },
        weights=_hand_weights(threshold=torch.full((HAND_D_SAE,), 0.75)),
    )
    assert sae.status().activation == ACT_JUMPRELU
    assert sae.status().threshold_span == [0.75, 0.75]
    assert "batchtopk" in sae.release.chosen_by["architecture"]


def test_an_unknown_architecture_string_is_refused_and_quoted(hub):
    """Quoted, so the reader can search for it. Named with the repo and hook,
    so nobody has to guess which of two loaded SAEs it is about."""
    with pytest.raises(Refusal) as caught:
        load_sae_lens(hub, {"architecture": "quantum_relu"})
    assert "quantum_relu" in caught.value.sentence
    assert HOOK_20 in caught.value.sentence
    assert "fake/sae-lens" in caught.value.sentence
    # This is the one test that WANTS the fallback, so it pins the phrase the
    # others assert the absence of. If that sentence stops listing what it can
    # read, `refused_by_name` quietly stops discriminating and every by-name
    # test above goes back to passing on the wrong branch.
    assert FALLBACK_REFUSALS["unknown architecture"] in caught.value.sentence


def test_a_transcoder_is_refused_with_its_own_reason(hub):
    """A transcoder maps between two hook points, so it does not reconstruct
    the stream it reads and this module's FVU contract does not apply to it.
    "Unknown architecture" would be true and would not be the reason."""
    with pytest.raises(Refusal) as caught:
        load_sae_lens(hub, {"architecture": "skip_transcoder"})
    sentence = refused_by_name(caught.value.sentence)
    assert "skip_transcoder" in sentence
    assert "TWO hook points" in sentence


# ------------------------------------------------- what happens before the gate


def test_expected_average_only_in_is_refused_because_the_factor_is_elsewhere(hub):
    """SAELens keeps that scaling factor in its own bundled release table,
    keyed by release name and SAE id, and folds it into the weights at load
    time. Loading from a repo path alone cannot recover it, and without it
    every feature magnitude is off by an unknown constant. SAELens itself only
    warns here; a warning nobody reads is how wrong numbers get plotted."""
    with pytest.raises(Refusal) as caught:
        load_sae_lens(
            hub,
            {
                "architecture": "standard",
                "normalize_activations": "expected_average_only_in",
            },
        )
    sentence = refused_by_name(caught.value.sentence)
    assert "expected_average_only_in" in sentence
    # The unknown-value fallback LISTS all four legal values, so the string
    # above is in that sentence too and this test passed against it — on a
    # refusal claiming SAELens never defined a value SAELens defines.
    assert "bundled release table" in sentence


def test_a_normalisation_that_has_to_be_undone_is_refused(hub):
    """constant_norm_rescale and layer_norm rescale the input and have to be
    undone on the way out. encode and decode are separate calls here with no
    state between them, and implementing half of it would return a
    reconstruction in the wrong units."""
    for value in ("constant_norm_rescale", "layer_norm"):
        with pytest.raises(Refusal) as caught:
            load_sae_lens(
                hub, {"architecture": "standard", "normalize_activations": value}
            )
        sentence = refused_by_name(caught.value.sentence)
        assert value in sentence
        # Same shadowing as above: the fallback names all four values, so
        # `value in sentence` alone held for a sentence saying SAELens does
        # not define one of its own.
        assert "undone on the way out" in sentence


def test_an_unknown_normalize_activations_value_is_refused(hub):
    with pytest.raises(Refusal) as caught:
        load_sae_lens(
            hub, {"architecture": "standard", "normalize_activations": "whitened"}
        )
    assert "whitened" in caught.value.sentence
    # The one test that wants this fallback, pinning the phrase the others
    # assert the absence of. See test_an_unknown_architecture_string_...
    assert FALLBACK_REFUSALS["unknown normalize_activations"] in caught.value.sentence


def test_a_boolean_normalize_activations_reads_as_the_value_it_migrates_to(hub):
    """Old configs wrote a bool. SAELens migrates False to "none" and True to
    "expected_average_only_in", so True must refuse and False must load."""
    sae = load_sae_lens(
        hub, {"architecture": "standard", "normalize_activations": False}
    )
    assert torch.equal(sae.encode(HAND_X), HAND_RELU)
    with pytest.raises(Refusal) as caught:
        load_sae_lens(hub, {"architecture": "standard", "normalize_activations": True})
    sentence = refused_by_name(caught.value.sentence)
    assert "expected_average_only_in" in sentence
    # Without `refused_by_name` this passed with the migration deleted: an
    # unmigrated `True` reaches the unknown-value fallback, which lists
    # expected_average_only_in among the four it names. The refusal has to be
    # the one about the missing factor, not one about a value it cannot read.
    assert "bundled release table" in sentence


def test_a_hook_z_sae_is_refused_rather_than_fed_the_residual_stream(hub):
    """A hook_z SAE expects [..., n_heads, d_head], flattened before the
    encoder. This module addresses the residual stream, whose vectors are
    [..., d_model] — the right shape and the wrong content."""
    with pytest.raises(Refusal) as caught:
        load_sae_lens(
            hub, {"architecture": "standard", "reshape_activations": "hook_z"}
        )
    sentence = refused_by_name(caught.value.sentence)
    assert "hook_z" in sentence
    # The unknown-reshape fallback quotes 'hook_z' too, so the line above
    # passed with this branch deleted — on a sentence saying nothing here
    # knows what hook_z is, when the point of the branch is that it does.
    assert "n_heads" in sentence


def test_an_unknown_reshape_activations_value_is_refused(hub):
    """The other half of the reshape gate, which had no test at all.

    hook_z is the reshaping this module can NAME. Anything else rearranges the
    encoder's input in a way nothing here can reproduce, and the sentence has
    to say that rather than let the value through as "none".
    """
    with pytest.raises(Refusal) as caught:
        load_sae_lens(
            hub, {"architecture": "standard", "reshape_activations": "hook_q"}
        )
    assert "hook_q" in caught.value.sentence
    assert FALLBACK_REFUSALS["unknown reshape_activations"] in caught.value.sentence


def test_apply_b_dec_to_input_is_read_when_it_is_declared(hub):
    """Declared true and declared false stay two different answers, and both
    stay visible beside the convention the calibration actually measured."""
    sae = load_sae_lens(hub, {"architecture": "standard", "apply_b_dec_to_input": True})
    assert sae.declared_b_dec is True
    assert "true" in sae.release.chosen_by["apply_b_dec_to_input"].lower()

    sae = load_sae_lens(
        hub, {"architecture": "standard", "apply_b_dec_to_input": False}
    )
    assert sae.declared_b_dec is False
    assert "false" in sae.release.chosen_by["apply_b_dec_to_input"].lower()


def test_an_absent_apply_b_dec_stays_absent_and_records_the_saelens_default(hub):
    """Two facts, and they are different facts.

    The file declares nothing, so `declared_b_dec` is None — reading it as
    False is the bug this module was rewritten around. But SAELens's own
    loaders default an absent key to TRUE in every schema they have written,
    so the release behaves as though it said true, and a reader comparing the
    declaration against the measured convention has to be told that or they
    will read "undeclared" as "declined".
    """
    sae = load_sae_lens(hub, {"architecture": "standard"})
    assert sae.declared_b_dec is None
    why = sae.release.chosen_by["apply_b_dec_to_input"]
    assert "true" in why.lower(), "the SAELens default for an absent key is unrecorded"


# ---------------------------------------------------------- decoder-norm rescale


def test_rescaling_by_the_decoder_norm_changes_which_features_win(hub):
    """TopK selects on pre-activations scaled by the decoder row norms, so the
    flag decides the winners and not only their size.

    Folded at load, exactly as SAELens folds it when it saves an inference
    model: W_enc and b_enc multiplied by the norms, W_dec divided by them.
    After that the plain encode/decode arithmetic IS the rescaled one, and
    nothing downstream — feature_ablate subtracting act x W_dec[f], the feature
    corpus reading W_dec[f] as a direction — has to know the flag existed.
    """
    W_dec = HAND_W_ENC.t().contiguous().clone()
    W_dec[5] = torch.tensor([0.0, -3.0, 0.0, 0.0])  # norm 3; the others norm 1
    weights = _hand_weights(W_dec=W_dec)

    plain = load_sae_lens(
        hub, {"architecture": "topk", "k": 2}, weights=weights, repo="fake/plain"
    )
    assert torch.equal(plain.encode(HAND_X), HAND_GATED_TO_TWO)

    scaled = load_sae_lens(
        hub,
        {"architecture": "topk", "k": 2, "rescale_acts_by_decoder_norm": True},
        weights=weights,
        repo="fake/scaled",
    )
    # Feature 5's pre-activation becomes 0.5 x 3 = 1.5 on the first token,
    # which beats feature 0's 1.0 and takes its place in the top two.
    assert torch.equal(
        scaled.encode(HAND_X),
        torch.tensor([[0.0, 0.0, 2.0, 0.0, 0.0, 1.5], [3.0, 1.0, 0.0, 0.0, 0.0, 0.0]]),
    )
    assert torch.allclose(scaled.W_dec[5].norm(), torch.tensor(1.0), atol=1e-6), (
        "the fold must leave the decoder rows unit-norm"
    )
    # The contribution of a feature is its activation times its decoder row,
    # and the fold must leave that invariant — it is what feature_ablate
    # subtracts and what the reconstruction is built from.
    assert torch.allclose(
        scaled.decode(scaled.encode(HAND_X))[0],
        torch.tensor([0.0, -1.5, 2.0, 0.0]),
        atol=1e-6,
    )
    # The fold rewrote three published tensors, so it owes the reader a
    # sentence. Without one, `W_dec[5].norm() == 1.0` in the panel contradicts
    # the 3.0 in the release and nothing accounts for the difference.
    why = scaled.release.chosen_by["rescale_acts_by_decoder_norm"]
    assert "W_dec divided by them" in why
    assert "rescale_acts_by_decoder_norm" not in plain.release.chosen_by, (
        "a release that folded nothing must not report a fold"
    )


def test_a_rescale_flag_that_is_not_a_boolean_is_refused(hub):
    """The one cfg value this module coerced instead of checking.

    `bool("false")` is True, and this flag rewrites W_enc, b_enc and W_dec —
    so a string a person reads as "off" turned the fold ON and moved which
    features win the selection. k, average_l0 and normalize_activations were
    all type-checked with a comment saying why; this one was not.
    """
    with pytest.raises(Refusal) as caught:
        load_sae_lens(
            hub,
            {"architecture": "topk", "k": 2, "rescale_acts_by_decoder_norm": "false"},
        )
    sentence = refused_by_name(caught.value.sentence)
    assert "'false'" in sentence
    assert "not true or false" in sentence


def test_a_zero_decoder_row_refuses_the_rescale_rather_than_dividing_by_it(hub):
    """The fold divides W_dec by its own row norms, and a dead feature's row
    can be exactly zero.

    Nothing else in this file has a zero decoder row, so this guard shipped
    untested. Without it the division puts `inf` in W_dec, `torch.topk` is
    free to select a zero pre-activation, and the reconstruction is NaN in
    every dimension — with no refusal and no number to trace it to.
    """
    W_dec = HAND_W_ENC.t().contiguous().clone()
    W_dec[5] = torch.zeros(HAND_D_IN)
    with pytest.raises(Refusal) as caught:
        load_sae_lens(
            hub,
            {"architecture": "topk", "k": 2, "rescale_acts_by_decoder_norm": True},
            weights=_hand_weights(W_dec=W_dec),
        )
    sentence = refused_by_name(caught.value.sentence)
    assert "1 of its 6" in sentence, "the count and the total are the evidence"
    assert "NaN" in sentence


# --------------------------------------- the weight file, and the cfg beside it


def test_a_scaling_factor_that_is_not_all_ones_is_refused(hub):
    """Older releases ship a per-feature finetuning scale, and this drops it.

    Only ABSENT weight keys were checked, so a fifth tensor was read off disk
    and thrown away. SAELens's own reader deletes an all-ones scaling_factor
    and applies a real one — it raises outright rather than ignore it — so a
    release from a finetuned decoder loaded here with every feature magnitude
    off by a factor the publisher trained, and nothing said so.
    """
    with pytest.raises(Refusal) as caught:
        load_sae_lens(
            hub,
            {"architecture": "standard"},
            weights=_hand_weights(scaling_factor=torch.full((HAND_D_SAE,), 7.0)),
        )
    sentence = refused_by_name(caught.value.sentence)
    assert "scaling_factor" in sentence
    assert "not all ones" in sentence


def test_an_all_ones_scaling_factor_loads_and_is_named_as_dropped(hub):
    """The value SAELens itself deletes must not be refused.

    Refusing it would reject releases that are fine — an all-ones scale is
    arithmetically nothing. But it was still a tensor in the file that this
    does not apply, so it is written down rather than left as silence.
    """
    sae = load_sae_lens(
        hub,
        {"architecture": "standard"},
        weights=_hand_weights(scaling_factor=torch.ones(HAND_D_SAE)),
    )
    assert torch.equal(sae.encode(HAND_X), HAND_RELU)
    why = sae.release.chosen_by["weights"]
    assert "scaling_factor" in why and "all ones" in why


def test_a_weight_file_with_nothing_extra_reports_no_extras(hub):
    """ "Nothing was dropped" is the ordinary case and says nothing.

    Asserted so the key cannot quietly become a sentence on every release,
    where it would stop meaning "something in this file was not applied".
    """
    sae = load_sae_lens(hub, {"architecture": "standard"})
    assert "weights" not in sae.release.chosen_by


def test_a_cfg_that_names_another_hook_is_refused(hub):
    """cfg.json states the hook it reads, and nothing compared it to the one
    it was fetched from.

    resid_pre against resid_post is the dangerous pair: the two streams differ
    by one block's output, so an SAE fed the wrong side still reconstructs
    well enough to pass the usability bar and be plotted. A whole layer apart
    is the same failure with more room in it.
    """
    with pytest.raises(Refusal) as caught:
        load_sae_lens(
            hub,
            {
                "architecture": "standard",
                "metadata": {
                    "sae_lens_version": "6.27.3",
                    "hook_name": "blocks.3.hook_resid_post",
                },
            },
        )
    sentence = refused_by_name(caught.value.sentence)
    assert "blocks.3.hook_resid_post" in sentence
    assert HOOK_20 in sentence


def test_a_cfg_that_names_the_hook_it_was_fetched_from_loads(hub):
    """The agreeing case, in both spellings the schemas use.

    `hook_point` is the pre-v3 name and `metadata.hook_name` the modern one,
    and a check that only read one of them would refuse half the releases it
    was meant to protect — or, worse, pass them.
    """
    for cfg in (
        {"hook_point": HOOK_20},
        {
            "architecture": "standard",
            "metadata": {"sae_lens_version": "6.27.3", "hook_name": HOOK_20},
        },
    ):
        sae = load_sae_lens(hub, cfg)
        assert torch.equal(sae.encode(HAND_X), HAND_RELU)


def test_a_hook_name_this_cannot_parse_is_not_a_disagreement(hub):
    """An unfamiliar spelling is not evidence of a mis-addressed release.

    Refusing on a hook name this module's regex cannot read would reject
    releases for being written in a convention nobody here has met, which is
    the opposite of the guarantee: only a declaration that PARSES and
    DISAGREES is a contradiction.
    """
    sae = load_sae_lens(hub, {"hook_point": "transformer.h.20.output"})
    assert torch.equal(sae.encode(HAND_X), HAND_RELU)


def test_a_cfg_whose_dimensions_disagree_with_its_weight_file_is_refused(hub):
    """d_in and d_sae are stated in cfg.json and measured off the tensors.

    `_refuse_impossible_k` cannot catch this: it checks k against
    `W_enc.shape[1]`, not against the cfg that named it. So a cfg describing a
    65,536-feature dictionary loaded beside a six-feature weight file, and the
    panel reported six — from the file — while every gate came off the cfg.
    """
    with pytest.raises(Refusal) as caught:
        load_sae_lens(hub, {"architecture": "topk", "k": 2, "d_in": 4, "d_sae": 65536})
    sentence = refused_by_name(caught.value.sentence)
    assert "d_sae=65536" in sentence
    assert "d_sae=6" in sentence


def test_dimensions_that_agree_load_and_a_boolean_is_not_a_dimension(hub):
    """Two halves of the same guard.

    A cfg that states the right shapes must load — the check is against
    contradiction, not against declaring anything at all — and a bool is not a
    width: `isinstance(True, int)` is True, so an unchecked one would compare
    equal to a d_in of 1 and refuse a file that is fine.
    """
    sae = load_sae_lens(
        hub, {"architecture": "standard", "d_in": HAND_D_IN, "d_sae": HAND_D_SAE}
    )
    assert torch.equal(sae.encode(HAND_X), HAND_RELU)
    sae = load_sae_lens(hub, {"architecture": "standard", "d_in": True})
    assert torch.equal(sae.encode(HAND_X), HAND_RELU)


def test_the_model_a_release_names_is_carried_through_in_both_states(hub):
    """The one declaration this module can repeat and cannot check.

    An SAE is attached to whichever model the session has open, and nothing
    compares that against the model it was trained on — the only check
    anywhere is d_in, which two unrelated models of the same width share. So
    the name is recorded, and its ABSENCE is recorded too: "this cfg does not
    say" and "this cfg says gpt2" are different things to know when the
    features look wrong.
    """
    sae = load_sae_lens(hub, {"architecture": "standard", "model_name": "gpt2-small"})
    assert "gpt2-small" in sae.release.chosen_by["model"]

    sae = load_sae_lens(hub, {"architecture": "standard"})
    assert "not declared" in sae.release.chosen_by["model"]


# ------------------------------------------------------------------- edge cases


def test_every_architecture_returns_nothing_for_a_zero_vector(hub):
    """A zero input has nothing to decompose, so no feature may fire.

    TopK is the one worth asserting: its gate selects k features whatever the
    row holds, so an implementation that dropped the ReLU would report k zeros
    as k firing features.
    """
    for name, sae in every_architecture(hub).items():
        sae.encode(HAND_X)  # calibrate on something with content in it
        feats = sae.encode(torch.zeros(2, HAND_D_IN))
        assert feats.abs().max() == 0.0, f"{name} fired on a zero vector"


def test_a_threshold_in_another_dtype_still_gates_the_same_way(hub):
    """Releases are float32 today, and the gate must not depend on that.

    A float16 threshold compared against float32 pre-activations promotes
    silently and would gate at slightly the wrong bar; worse, carrying it as
    float16 would halve the precision of a number `status` quotes as measured.
    """
    sae = load_sae_lens(
        hub,
        {"architecture": "jumprelu"},
        weights=_hand_weights(threshold=HAND_THRESHOLD.to(torch.float16)),
    )
    assert sae.threshold.dtype is torch.float32
    assert torch.equal(sae.encode(HAND_X), HAND_GATED_TO_TWO)


def test_the_gemma_scope_reader_still_loads_exactly_as_it_did(hub):
    """The .npz path carries its own JumpReLU thresholds and must not move.

    Asserted on the tensors and on the encode rather than on the plumbing:
    this is the reader that was already correct, and the whole risk of making
    the SAELens path explicit is that it disturbs this one.
    """
    sae = SAEHandle.load("fake/gemma-scope", HOOK_20)
    assert sae.status().activation == ACT_JUMPRELU
    assert sae.threshold is not None
    assert torch.equal(sae.threshold, torch.full((D_SAE,), -1e9))
    assert sae.release.layout == "gemma_scope"
    x = uncentered_activations()
    feats = sae.encode(x)
    cal = sae.calibration
    # Every threshold is below every pre-activation, so the gate is open and
    # the answer is the plain ReLU of the same weights in the same convention
    # — which is what the layout-equivalence test above compares the SAELens
    # reader against.
    prepared = sae._prepare(x, cal.center, cal.subtract_b_dec)
    assert torch.equal(feats, torch.relu(prepared @ sae.W_enc + sae.b_enc))
    assert cal.convention == "centered+b_dec"
