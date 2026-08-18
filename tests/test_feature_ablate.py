"""Feature ranking must be a measurement, not the bar chart in causal clothing.

The features panel already produces an ordered list; this adds one that claims
to be causal, and a causal list is read as truth. Every test here guards one of
the ways it could lie: ranking an SAE that decomposes nothing, padding the list
with features that never fired, claiming an edit removed something it did not,
hiding a truncated candidate set, or reporting arithmetic as signal.

The synthetic SAE below is built so that both of the properties the module
depends on are exact rather than approximate. Its encoder columns and decoder
rows are one orthonormal basis of the mean-zero subspace, so (i) it
reconstructs exactly, and only from `centered+b_dec` — everything it can emit
has zero d_model mean, and its b_dec is non-zero so skipping the subtraction
leaves a constant error — and (ii) subtracting `act * W_dec[f]` drives feature
f's pre-activation to exactly zero, which is what the mechanism check asserts
about real SAEs and must be true of the stand-in for the test to mean anything.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from modelmri import feature_ablate  # noqa: E402
from modelmri.errors import BadRequest  # noqa: E402
from modelmri.saes import SAEHandle  # noqa: E402

D_IN = 8
SEQ = 6
VOCAB = 5
POSITION = SEQ - 1

PROMPT = "The Eiffel Tower is located in the city of"


# ------------------------------------------------------------- the stand-ins


def _centered_basis(d: int) -> torch.Tensor:
    """[d, d-1] orthonormal columns spanning the mean-zero subspace."""
    g = torch.Generator().manual_seed(0)
    p = torch.eye(d) - torch.ones(d, d) / d
    q, _ = torch.linalg.qr(p @ torch.randn(d, d, generator=g))
    q = q[:, : d - 1]
    assert torch.allclose(q.T @ q, torch.eye(d - 1), atol=1e-5)
    assert q.sum(0).abs().max() < 1e-5, "basis is not mean-zero"
    return q


def synthetic_sae(b_dec_scale: float = 3.0) -> SAEHandle:
    """Reconstructs exactly, only from `centered+b_dec`, with aligned encoder.

    W_enc = [Q, -Q] and W_dec = [Q; -Q]^T: relu(a) - relu(-a) == a makes the
    pair an identity on the centered subspace, and each feature's encoder
    direction IS its decoder direction, with unit norm — so removing
    act*W_dec[f] takes that feature's pre-activation to exactly 0.

    `b_dec_scale=0` gives up the b_dec half of the convention test and buys an
    SAE whose pre-activations can be driven to EXACTLY zero, which the
    nothing-fires test needs — see the comment there.
    """
    q = _centered_basis(D_IN)
    g = torch.Generator().manual_seed(1)
    b_dec = torch.randn(D_IN, generator=g)
    b_dec = (b_dec - b_dec.mean()) * b_dec_scale  # mean-zero, centering-proof
    return SAEHandle(
        repo="synthetic/sae",
        hook="blocks.0.hook_resid_pre",
        point="resid_pre",
        layer=0,
        W_enc=torch.cat([q, -q], dim=1),
        b_enc=torch.zeros(2 * (D_IN - 1)),
        W_dec=torch.cat([q.T, -q.T], dim=0),
        b_dec=b_dec,
        apply_b_dec_to_input=None,
    )


def unusable_sae() -> SAEHandle:
    """Reconstructs less than a constant vector would: FVU >= 1."""
    g = torch.Generator().manual_seed(2)
    return SAEHandle(
        repo="synthetic/noise",
        hook="blocks.0.hook_resid_pre",
        point="resid_pre",
        layer=0,
        W_enc=torch.randn(D_IN, 4, generator=g) * 0.01,
        b_enc=torch.zeros(4),
        W_dec=torch.randn(4, D_IN, generator=g) * 0.01,
        b_dec=torch.zeros(D_IN),
        apply_b_dec_to_input=None,
    )


class ToyBlock(torch.nn.Module):
    """Identity, so hooks are the only thing that changes the stream."""

    def forward(self, x):
        return x


class _Out:
    def __init__(self, logits):
        self.logits = logits


class ToyModel:
    """Logits from a CAUSAL running mean of the stream at the block.

    The mixing is what makes `scope="prompt"` testable: without it a linear
    readout at one position would make every earlier-token edit score exactly
    zero, and a test that passed would be testing nothing.

    `double` calls the block twice on different tensors. That stands in for
    the failure the write-back check exists to catch — the edit landing
    somewhere other than where the capture came from — because the capture
    records the first call and the edit replaces both.
    """

    def __init__(self, block, resid, readout, double: bool = False):
        self.block, self.resid, self.readout, self.double = (
            block,
            resid,
            readout,
            double,
        )

    def __call__(self, ids):
        h = self.block(self.resid.unsqueeze(0))
        if self.double:
            h = h + self.block(self.resid.unsqueeze(0) * 0.5)
        n = torch.arange(1, h.shape[1] + 1, dtype=h.dtype).view(1, -1, 1)
        return _Out((h.cumsum(1) / n) @ self.readout)


def toy_stream() -> torch.Tensor:
    """Activations with a real d_model mean, like HuggingFace's stream."""
    g = torch.Generator().manual_seed(3)
    return torch.randn(SEQ, D_IN, generator=g) * 5.0 + 4.0


def toy(
    *, sae: SAEHandle | None = None, resid: torch.Tensor | None = None, **kw
) -> dict:
    sae = sae or synthetic_sae()
    resid = toy_stream() if resid is None else resid
    g = torch.Generator().manual_seed(4)
    block = ToyBlock()
    model = ToyModel(
        block,
        resid,
        torch.randn(D_IN, VOCAB, generator=g),
        double=kw.pop("double", False),
    )
    return feature_ablate.rank_features(
        model,
        block,
        torch.zeros(1, SEQ, dtype=torch.long),
        sae,
        position=kw.pop("position", POSITION),
        **kw,
    )


# ------------------------------------------------- refusing what cannot be ranked


def test_an_sae_that_decomposes_nothing_is_refused_with_its_fvu():
    """FVU >= 1 means the features are not a decomposition of anything.

    Not hypothetical: the shipped SAE scored an FVU in the thousands, with
    thousands of features firing at once, before saes.py calibrated its input
    convention. Ranking that many simultaneously-firing features by causal
    effect would have been ranking arbitrary directions with a confident
    number attached — so the refusal carries the measured FVU rather than the
    word "unusable".
    """
    with pytest.raises(feature_ablate.FeatureAblationError) as err:
        toy(sae=unusable_sae())
    message = str(err.value)
    assert "variance unexplained" in message
    # The number itself, not a verdict about it.
    assert any(ch.isdigit() for ch in message.split("unexplained")[1][:20])


def test_a_position_where_nothing_fires_is_refused_not_padded():
    """24576 rows of the noise floor is not a ranking.

    The row is a CONSTANT vector and the SAE's b_dec is zero, so its centered
    input is exactly 0 and every pre-activation is exactly 0. The obvious
    fixture — `b_dec + c` against a non-zero b_dec — does not work, and finding
    that out is worth a line: `x - mean(x) - b_dec` came back at 2.4e-07 rather
    than 0 there, and 7 features "fired" at 2.4e-07. The module's candidate
    rule is `activation > 0` and stays that way, because a feature contributing
    2.4e-07 of a decoder direction genuinely is in the stream and its score
    comes back flagged `below_resolution`; inventing a minimum activation would
    be a threshold nobody measured.
    """
    resid = toy_stream()
    sae = synthetic_sae(b_dec_scale=0.0)
    resid[POSITION] = torch.full((D_IN,), 4.0)
    assert int((sae.encode(resid)[POSITION] > 0).sum()) == 0, "fixture does not fire 0"
    with pytest.raises(feature_ablate.FeatureAblationError, match="nothing to remove"):
        toy(sae=sae, resid=resid)


def test_an_edit_that_does_not_land_where_the_capture_came_from_is_refused():
    """The floor pass is the plumbing check, not a formality.

    If the tensor the edit replaces is not the tensor the capture read, every
    score is the difference between two places in the network rather than the
    effect of a feature — and the scores would still look like scores.
    """
    with pytest.raises(feature_ablate.FeatureAblationError, match="unchanged moves"):
        toy(double=True)


# ------------------------------------------------- which features are on trial


def test_only_features_that_fire_are_candidates():
    """Removing zero times a decoder direction is the identity.

    A feature with activation 0 contributes nothing to the stream, so its
    score is the floor by construction. Listing it would pad the ranking with
    zeros indistinguishable from measurements.
    """
    sae, resid = synthetic_sae(), toy_stream()
    feats = sae.encode(resid)
    firing = int((feats[POSITION] > 0).sum())
    silent = feats.shape[1] - firing
    assert silent > 0, "the fixture proves nothing if every feature fires"

    out = toy(sae=sae, resid=resid)
    assert out["n_candidates"] == firing
    assert all(row["activation"] > 0 for row in out["ranked"])
    tested = {row["feature_id"] for row in out["ranked"]}
    assert tested == {int(f) for f in (feats[POSITION] > 0).nonzero().flatten()}


def test_prompt_scope_puts_features_from_earlier_tokens_on_trial():
    """Measured: features in the global top-8 fire only at earlier tokens.

    The panel cannot show those at all today, and a position-local ranking
    cannot either — they reach the prediction through attention. Prompt scope
    puts far more candidates on trial than position scope does.
    """
    here = toy(scope="position")
    across = toy(scope="prompt")
    assert across["n_candidates"] > here["n_candidates"]
    assert across["scope"] == "prompt"
    # A feature removed wherever it fires must name those positions, so a
    # reader can tell a token-local claim from a whole-prompt one.
    assert any(len(row["positions"]) > 1 for row in across["ranked"])


def test_truncation_is_reported_three_ways_and_never_silent():
    """ "Not asked" and "asked, and the answer was nothing" are different.

    The cap orders by peak activation because nothing else is available before
    the measurement — and that ordering is exactly the one this module exists
    to say is not the causal one (6 of 8 agreement at a position, 3 of 8 across
    the prompt), so a dropped feature may well have ranked.
    """
    out = toy(max_candidates=2)
    assert out["n_tested"] == 2 < out["n_candidates"]
    assert out["truncated"] is True
    assert len(out["ranked"]) == 2
    assert "NOT TESTED" in out["coverage"]

    whole = toy()
    assert whole["truncated"] is False
    assert whole["n_tested"] == whole["n_candidates"]


# ------------------------------------------------- does the edit do what it says


def test_the_mechanism_check_is_about_the_edit_landing_and_says_so():
    """`removal_verified` is one claim: the model received the intended edit.

    It used to be a different claim — that re-encoding the edited stream shows
    the feature gone — taken on ONE row and reported as a property of the edit
    and the SAE. Measured on the real SAE at blocks.8.hook_resid_pre, that
    claim is false for most of the features firing at the attributed token,
    and the few rows that do pass pass because relu clamped an overshoot
    rather than because the feature left. So the tick is now about the edit,
    which really is a property of the edit and the dtype, and what the SAE
    still reads is per row.
    """
    out = toy()
    assert out["removal_verified"] is True
    assert out["edit_deviation"] == pytest.approx(0.0, abs=1e-6)
    assert "The edit landed" in out["removal_check"]
    # And it refuses to be read as the other claim.
    assert "does NOT mean the feature left" in out["removal_check"]
    assert "encoder_residual" in out["removal_check"]
    assert "residual_activation" not in out, "the one-row verdict is gone"


def test_every_scored_row_reports_what_the_encoder_still_reads():
    """Per row, because it varies per row: 0% to 60.3% on the real SAE.

    The synthetic SAE here is the ideal case — its encoder columns ARE its
    decoder rows, so subtracting act*W_dec drives the pre-activation to exactly
    zero and every row reads 0.0. That is the point of asserting it here: the
    field is present and correct even where it is boring, and the real-SAE test
    below is where it is not.
    """
    out = toy()
    assert out["ranked"], "nothing to check"
    for row in out["ranked"]:
        assert "encoder_residual" in row
        assert row["encoder_residual"] == pytest.approx(0.0, abs=1e-5)
    assert out["n_encoder_residual"] == 0
    assert out["encoder_residual_max"] == pytest.approx(0.0, abs=1e-5)


def test_the_edit_touches_one_direction_and_the_mean_moves_with_it():
    """What separates this from replacing the stream with the reconstruction.

    Measured: substituting the SAE's reconstruction while removing NOTHING
    costs more at the attribution position than almost every feature firing
    there scores in total. This edit's no-op costs nothing because it changes
    nothing.

    The edited token's d_model MEAN does move, and it is asserted here because
    the module used to claim it did not. `act*W_dec[f]` has a non-zero mean —
    measured on the real SAE, not assumed — and holding `mu` at the value the
    decomposition was taken with is exactly why this edit equals "zero the
    feature, decode, re-add the error".
    """
    sae, resid = synthetic_sae(), toy_stream()
    feats = sae.encode(resid)
    f = int(feats[POSITION].argmax())
    contribution = float(feats[POSITION, f]) * sae.W_dec[f]
    edited = resid.clone()
    edited[POSITION] -= contribution

    delta = edited - resid
    assert delta[:POSITION].abs().max() == 0.0, "an edit leaked to another token"
    # The change is exactly the feature's own contribution, parallel to its
    # decoder direction.
    direction = sae.W_dec[f] / sae.W_dec[f].norm()
    residual = delta[POSITION] - (delta[POSITION] @ direction) * direction
    assert residual.abs().max() < 1e-5
    # No OTHER token's mean moves.
    assert (edited.mean(-1) - resid.mean(-1))[:POSITION].abs().max() == 0.0
    # The edited one's moves by exactly the contribution's own mean — zero for
    # this SAE, whose decoder rows are mean-zero by construction, and not zero
    # for a real one, which is the case the module docstring carries numbers
    # for.
    moved = float(edited[POSITION].mean() - resid[POSITION].mean())
    assert moved == pytest.approx(-float(contribution.mean()), abs=1e-6)


def test_a_no_op_edit_scores_the_measured_floor():
    """Hook installed, captured stream written back unchanged, nothing removed.

    Measured on gpt2 (fp32/cuda, 11 tokens): exactly 0.0 against the base
    distribution on four repeats, equal to a no-hook replay. The floor is the
    thing that proves the write-back is inert; the RESOLUTION is a different
    number, because two real scores in that run came back negative (-1e-08,
    -3e-08) — impossible for a KL, and float32 summation over 50257 vocabulary
    entries.
    """
    out = toy()
    assert out["noise_floor_kl"] == pytest.approx(0.0, abs=1e-9)
    assert out["replay_kl"] == pytest.approx(0.0, abs=1e-9)
    assert out["resolution_kl"] > out["noise_floor_kl"]
    assert out["resolution_kl"] == feature_ablate.RESOLUTION_KL


# ------------------------------------------------- what the answer says


def test_the_answer_names_its_intervention():
    out = toy()
    assert out["intervention"] == feature_ablate.SUBTRACT
    assert out["intervention"] in out["means"] or "W_dec" in out["means"]


def test_an_unknown_intervention_or_scope_is_a_bad_request():
    """`?scope=vibes` is a malformed call, not a measurement we declined.

    runtime.py turns a FeatureAblationError into 409 "ModelMRI decided not to
    answer", which is the wrong sentence for a parameter the caller can fix.
    BadRequest is a ValueError and FeatureAblationError a RuntimeError, so no
    existing handler can confuse them.
    """
    with pytest.raises(BadRequest, match="unknown intervention"):
        toy(intervention="replace_with_reconstruction")
    with pytest.raises(BadRequest, match="unknown scope"):
        toy(scope="vibes")
    with pytest.raises(BadRequest, match="outside a sequence"):
        toy(position=SEQ + 3)
    assert not isinstance(BadRequest(""), feature_ablate.FeatureAblationError)


def test_the_answer_says_the_scores_under_count_rather_than_over():
    """Direction matters and the head panel's wording does not transfer.

    Features: 43 singles sum to 0.66446 against 2.135221 for one joint
    ablation — 3.2x UNDER. Heads on gpt2 layer 0: 1.995 against 0.208 — 8x
    over. Copying ablate.py's sentence here would invert the caveat.
    """
    means = toy()["means"].lower()
    assert "not" in means and "add up" in means
    assert "under" in means
    assert "sum_of_singles" in means and "joint_kl" in means


def test_the_answer_carries_a_per_position_reconstruction_error():
    """The calibration's FVU is not this token's accuracy, or even its units.

    Measured: FVU 0.000984 aggregated over 11 tokens, dominated by token 0
    whose stream norm is 3077.3 against ~100 elsewhere — while at the
    attribution position the SAE misses 20.36% of the norm, worth 0.0775 nats,
    more than 41 of the 43 features there. Quoting the aggregate beside a
    ranking would claim a decomposition that is 99.9% complete.

    And `fvu` cannot be the aggregate it is compared against: it is a squared
    fraction against `residual_share`'s norm fraction, so the honest pairing is
    `rel_err` (0.029397 against 0.203571, 7x) — which is why it is returned.
    """
    out = toy()
    assert 0.0 <= out["residual_share"] <= 1.0
    assert out["residual_kl"] >= 0.0
    # Both are reported, and they are different quantities. The synthetic SAE
    # reconstructs exactly, so here they happen to agree at 0 — which is why
    # what is asserted is that the response says which is which rather than
    # that the two numbers differ.
    assert isinstance(out["fvu"], float) and isinstance(out["residual_share"], float)
    assert isinstance(out["rel_err"], float)
    assert "SQUARED-error fraction" in out["residual_means"]
    assert "rel_err" in out["residual_means"]
    assert (
        str(out["fvu"]) in out["residual_means"] or "aggregate" in out["residual_means"]
    )


def test_the_reconstruction_baseline_follows_the_scope():
    """A one-token baseline understates what a prompt-scope ranking edits.

    Measured on gpt2: substituting the reconstruction at position 10 costs
    0.077530 nats, and over positions 0-10 — the window a prompt-scope ranking
    actually edits — 0.221217, 2.85x more. Against the first, 2 of 43 features
    clear; against the second, 1 of 256. The panel printed the first beside a
    prompt-scope ranking, so a feature was shown as clearing the SAE's own
    error when it did not.
    """
    at_pos = toy(scope="position")
    over_prompt = toy(scope="prompt")
    assert at_pos["residual_window"] == [POSITION, POSITION]
    # The prompt-scope window reaches back before the attributed token.
    assert over_prompt["residual_window"][0] < POSITION
    assert over_prompt["residual_window"][1] == POSITION
    # And the share it reports is the worst in that window, not this token's.
    assert over_prompt["residual_share"] >= over_prompt["residual_share_at_position"]
    assert at_pos["residual_share"] == at_pos["residual_share_at_position"]


def test_every_score_is_paired_with_a_same_size_random_control():
    """A score is partly the size of the edit, and the response says how much.

    Measured on gpt2 at the attributed token: a random Gaussian direction at
    feature 5856's norm of 35.5 costs 0.0666-0.1093 nats over five draws
    against that feature's own 0.417461 — so the top row clears its control by
    about 4x, not by everything, and 9 of the 43 rows do not clear theirs at
    all. Two of those nine, #22852 and #1288, are in the bar chart's plotted
    top-8.
    """
    out = toy()
    for row in out["ranked"]:
        assert "control_kl" in row
        assert row["clears_control"] == (row["kl"] > row["control_kl"])
    assert out["n_clearing_control"] == sum(
        1 for row in out["ranked"] if row["clears_control"]
    )
    assert "control_kl" in out["control_means"]
    # Seeded, so the same request twice gives the same controls — otherwise a
    # row could cross its own control between two identical runs.
    again = toy()
    assert [row["control_kl"] for row in again["ranked"]] == [
        row["control_kl"] for row in out["ranked"]
    ]


def test_the_ranking_is_sorted_and_every_row_reports_the_top_token():
    out = toy()
    scores = [row["kl"] for row in out["ranked"]]
    assert scores == sorted(scores, reverse=True)
    for row in out["ranked"]:
        assert {
            "feature_id",
            "activation",
            "kl",
            "flips_top",
            "p_top_before",
            "p_top_after",
        } <= set(row)
        assert 0.0 <= row["p_top_before"] <= 1.0
        assert 0.0 <= row["p_top_after"] <= 1.0
    # TWO passes per row, not one: the feature's own edit and its same-norm
    # control. Checked against real runs — 43 features tested came back as 92
    # passes on gpt2, 256 as 518.
    assert out["passes"] == 2 * out["n_tested"] + 6
    assert out["position"] == POSITION


def test_scores_below_the_resolution_are_flagged_not_dressed_up():
    out = toy()
    for row in out["ranked"]:
        assert row["below_resolution"] == (abs(row["kl"]) < out["resolution_kl"])


# ------------------------------------------------------- the real one, if cached


def _cached(*names: str) -> bool:
    home = os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE")
    roots = [Path(home)] if home else []
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    return all(
        any((root / name).exists() or (root / "hub" / name).exists() for root in roots)
        for name in names
    )


@pytest.mark.skipif(
    not _cached("models--gpt2", "models--jbloom--GPT2-Small-SAEs-Reformatted"),
    reason="gpt2 or the default SAE is not in the local HF cache",
)
def test_the_real_ranking_is_not_the_bar_chart():
    """The whole premise, on the run every number in the module was taken from.

    gpt2, fp32, eager, SAE blocks.8.hook_resid_pre, prompt "The Eiffel Tower
    is located in the city of", attributing at position 10 where the top token
    is " Paris" at p=0.06378. Feature 5856 leads by 25x, which is why it is
    safe to pin across devices; the exact KL is given a 2% band because this
    may run on CPU where the measurement was taken on cuda.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2", torch_dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    try:
        ids = tok(PROMPT, return_tensors="pt").input_ids.to(device)
        sae = SAEHandle.load()
        out = feature_ablate.rank_features(
            model,
            model.transformer.h[sae.layer],
            ids,
            sae,
            position=int(ids.shape[1]) - 1,
            decode=lambda t: tok.decode([t]),
        )
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    assert out["target_token"] == " Paris"
    assert out["convention"] == "centered+b_dec"
    assert out["ranked"][0]["feature_id"] == 5856
    assert out["ranked"][0]["kl"] == pytest.approx(0.4174529, rel=0.02)
    # The EDIT lands exactly. That is the whole of what this flag claims.
    assert out["removal_verified"] is True
    assert out["edit_deviation"] == pytest.approx(0.0, abs=1e-5)

    # …and on a REAL SAE the feature does not leave the SAE's reading of the
    # stream, which is the claim the flag used to be read as making. Measured
    # here: 38 of 43 rows above the 1% tolerance, the worst at 60.26%
    # (feature 5926), because W_enc[:,f] and W_dec[f] are not dual — their dot
    # product has mean 0.8387 over d_sae, min -0.3819, max 1.3072. A run where
    # this came back 0 for every row would mean the check had stopped looking.
    assert out["n_encoder_residual"] >= 30, out["n_encoder_residual"]
    assert out["encoder_residual_max"] > 0.5
    by_id = {row["feature_id"]: row for row in out["ranked"]}
    assert by_id[5856]["encoder_residual"] == pytest.approx(0.0, abs=1e-3)
    assert by_id[2194]["encoder_residual"] == pytest.approx(0.1818, abs=0.02)
    assert by_id[21062]["encoder_residual"] == pytest.approx(0.3014, abs=0.02)

    # A same-norm random direction is not free, so most rows clear their own
    # control and a real minority do not. Measured 34 of 43 clearing, with
    # #22852 and #1288 — 5th and 6th in the activation chart — among the nine
    # that do not. Pinned as a band rather than a number because the draws are
    # seeded but the model may be running on a different device here.
    #
    # The band is also the honest width of the claim, not just device slack.
    # With 8 draws per row on this setup the same 43 rows give 34 clearing one
    # draw, 21 the 95th percentile and 20 all 8, and the shipped count moved
    # 34 -> 36 between cpu and cuda on identical draws. A tighter assertion
    # here would be pinning one sample of a random variable.
    assert 20 <= out["n_clearing_control"] <= 42, out["n_clearing_control"]
    assert out["ranked"][0]["control_kl"] > 0.02, "a same-size edit is not free"
    assert out["ranked"][0]["kl"] > 3 * out["ranked"][0]["control_kl"]

    # The finding, and the reason this module exists: the causal order is not
    # the activation order the panel plots. Measured overlap 6 of 8, with
    # features 6807 (act 1.328) and 8628 (1.561) entering the causal top-6
    # from outside the plotted set.
    by_kl = [row["feature_id"] for row in out["ranked"][:8]]
    by_act = [
        row["feature_id"]
        for row in sorted(out["ranked"], key=lambda r: -r["activation"])[:8]
    ]
    assert by_kl != by_act
    assert len(set(by_kl) & set(by_act)) < 8

    # And they under-count, in the opposite direction from heads.
    assert out["sum_of_singles"] < out["joint_kl"]
