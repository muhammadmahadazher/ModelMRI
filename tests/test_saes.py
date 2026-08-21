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

from modelmri.errors import BadRequest, Refusal
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


def test_encode_feature_is_encode_restricted_to_one_column():
    """One column of W_enc, not all 24,576 — and the same number.

    `feature_ablate` asks this once per scored row, to report how much of a
    feature the SAE's encoder still reads after that feature's contribution has
    been subtracted from the stream. A full `encode` per row would turn a free
    check into 19 million multiply-adds a row; this is 768. If the two ever
    disagreed, the per-row honesty column would be measuring something other
    than the activations the ranking was taken from.
    """
    sae = synthetic_sae()
    torch.manual_seed(11)
    x = torch.randn(6, sae.d_in) * 4.0 + 2.0
    full = sae.encode(x)
    for f in (0, 1, sae.d_sae - 1):
        assert torch.allclose(sae.encode_feature(x, f), full[:, f], atol=1e-6)


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
    """
    sae = synthetic_sae()
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
                path.write_text("{}", encoding="utf-8")
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
