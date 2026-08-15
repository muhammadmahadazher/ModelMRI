"""Causal occlusion on a camera frame, tested against a tower with a known blind spot.

The tower here reads ONE region of the image and ignores the rest by
construction, so the sweep can be wrong in a way no structural test would
catch. Everything else in this file is about the wording, because the wording
is the feature: a perception-only shift labelled "caused the action" would be
the exact overclaim the module exists to avoid.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from modelmri import vla_occlude as occ  # noqa: E402
from modelmri.errors import BadRequest  # noqa: E402


class OneRegionTower(nn.Module):
    """Reads rows 4-7, cols 4-7 of an 8x8 patch grid. Ignores everything else."""

    def __init__(self, size=64, patch=8):
        super().__init__()
        self.size, self.patch = size, patch

    def forward(self, pixel_values, output_attentions=False):
        region = pixel_values[..., 32:64, 32:64]
        v = region.mean(dim=(1, 2, 3))
        n_patch = (self.size // self.patch) ** 2
        hidden = torch.stack([v * (i + 1) * 0.01 for i in range(n_patch)], dim=1)
        return type("O", (), {"last_hidden_state": hidden.unsqueeze(-1).expand(-1, -1, 16)})()


class BlindTower(nn.Module):
    """Returns the same embedding whatever it is shown."""

    def forward(self, pixel_values, output_attentions=False):
        n = int(pixel_values.shape[0])
        return type(
            "O", (), {"last_hidden_state": torch.ones(n, 64, 16)}
        )()


@pytest.fixture
def frames():
    torch.manual_seed(0)
    return [torch.randn(1, 3, 64, 64) for _ in range(8)]


# ------------------------------------------------------------ what it finds


def test_the_sweep_finds_the_region_the_tower_actually_reads(frames):
    """Ground truth: this tower reads a 4x4 block of the 8x8 grid and nothing
    else. MEASURED: all 16 of the top 16 blocks land inside it."""
    out = occ.sweep(
        OneRegionTower().eval(), "cpu", frames[0], grid=[8, 8], patch=8,
        scale_frames=frames, baseline="midpoint", stride=1, max_controlled=8,
    )
    top = out.blocks[:16]
    inside = [b for b in top if 4 <= b.row <= 7 and 4 <= b.col <= 7]
    assert len(inside) == 16, "the sweep ranked blocks the tower cannot see"


def test_blocks_outside_the_read_region_score_zero(frames):
    out = occ.sweep(
        OneRegionTower().eval(), "cpu", frames[0], grid=[8, 8], patch=8,
        scale_frames=frames, baseline="midpoint", stride=1, max_controlled=1,
    )
    outside = [b for b in out.blocks if b.row < 4 or b.col < 4]
    assert outside
    for block in outside:
        assert block.shift == 0.0, f"r{block.row}c{block.col} is not read and moved"


# ----------------------------------------------------------- the two fills


def test_both_fill_baselines_are_named_and_reported(frames):
    """Occlusion is out of distribution — a grey box is itself a stimulus — so
    two baselines ship and the reader keeps what survives both."""
    for baseline in occ.BASELINES:
        out = occ.sweep(
            OneRegionTower().eval(), "cpu", frames[0], grid=[8, 8], patch=8,
            scale_frames=frames, baseline=baseline, stride=2, max_controlled=2,
        )
        assert out.baseline == baseline
        assert baseline in out.means()
    assert set(occ.BASELINES) == {"episode_mean", "midpoint"}


def test_an_unknown_fill_is_refused_rather_than_defaulted(frames):
    with pytest.raises(BadRequest, match="unknown fill baseline"):
        occ.sweep(
            OneRegionTower().eval(), "cpu", frames[0], grid=[8, 8], patch=8,
            scale_frames=frames, baseline="black", stride=4,
        )


def test_the_summary_always_says_occlusion_is_off_distribution(frames):
    out = occ.sweep(
        OneRegionTower().eval(), "cpu", frames[0], grid=[8, 8], patch=8,
        scale_frames=frames, stride=4, max_controlled=1,
    )
    assert "OUT OF DISTRIBUTION" in out.means()
    assert "keep what survives both" in out.means()


# ------------------------------------------------------------- the control


def test_the_control_occludes_an_area_and_not_a_random_tensor(frames):
    """The treatment occludes an AREA, so the null has to occlude an area of
    the same size somewhere else. A same-norm random tensor would compare an
    occlusion against something that is not one."""
    out = occ.sweep(
        OneRegionTower().eval(), "cpu", frames[0], grid=[8, 8], patch=8,
        scale_frames=frames, baseline="midpoint", stride=1, max_controlled=6,
    )
    tested = [b for b in out.blocks if b.control_max is not None]
    assert tested, "nothing was controlled"
    for block in tested:
        assert block.clears_control == (block.shift > block.control_max)


def test_an_untested_block_has_no_verdict_rather_than_a_failing_one(frames):
    """False would read as 'a random occlusion did as much', which nothing
    measured for a block nobody controlled."""
    out = occ.sweep(
        OneRegionTower().eval(), "cpu", frames[0], grid=[8, 8], patch=8,
        scale_frames=frames, stride=1, max_controlled=2,
    )
    untested = [b for b in out.blocks if b.control_max is None]
    assert untested
    for block in untested:
        assert block.clears_control is None
        assert block.control_draws == 0


def test_the_control_seed_is_fixed_so_a_run_repeats(frames):
    kwargs = dict(
        grid=[8, 8], patch=8, scale_frames=frames, baseline="midpoint",
        stride=1, max_controlled=4,
    )
    a = occ.sweep(OneRegionTower().eval(), "cpu", frames[0], **kwargs)
    b = occ.sweep(OneRegionTower().eval(), "cpu", frames[0], **kwargs)
    assert [x.control_max for x in a.blocks] == [x.control_max for x in b.blocks]


# ------------------------------------------------------- the two maps


def test_the_rank_agreement_between_the_maps_is_reported(frames):
    """The two disagreeing on your own checkpoint is the finding, and it is
    not visible from either map alone."""
    attention = [[float((r * 8 + c) % 5) for c in range(8)] for r in range(8)]
    out = occ.sweep(
        OneRegionTower().eval(), "cpu", frames[0], grid=[8, 8], patch=8,
        scale_frames=frames, stride=1, attention_map=attention, max_controlled=2,
    )
    assert out.attention_agreement is not None
    assert -1.0 <= out.attention_agreement <= 1.0
    assert "SPEARMAN" in out.means()


def test_the_agreement_names_the_layer_it_was_measured_against(frames):
    """A Spearman with no layer beside it is not a reportable number: measured
    on a real SmolVLA checkpoint the same causal map agreed at -0.053 against
    layer 0 and -0.103 against layer 11."""
    attention = [[float((r * 8 + c) % 5) for c in range(8)] for r in range(8)]
    out = occ.sweep(
        OneRegionTower().eval(), "cpu", frames[0], grid=[8, 8], patch=8,
        scale_frames=frames, stride=1, attention_map=attention,
        compared_layer=11, compared_head=-1, max_controlled=2,
    )
    assert out.compared_layer == 11
    assert "against layer 11" in out.means()
    assert "averaged over its heads" in out.means()


def test_a_named_head_is_said_to_be_one_head(frames):
    attention = [[float((r * 8 + c) % 5) for c in range(8)] for r in range(8)]
    out = occ.sweep(
        OneRegionTower().eval(), "cpu", frames[0], grid=[8, 8], patch=8,
        scale_frames=frames, stride=1, attention_map=attention,
        compared_layer=3, compared_head=7, max_controlled=2,
    )
    assert "against layer 3 head 7" in out.means()


def test_an_inverted_ranking_is_not_worded_as_an_absence_of_one(frames):
    """-0.9 and -0.05 are opposite findings. A strong negative rank
    correlation says the blocks attention ranked HIGHEST are the ones the
    representation depended on LEAST, which is a stronger claim than the two
    maps merely disagreeing."""
    out = occ.Occlusion(baseline="episode_mean", grid=[8, 8], stride=1)
    out.attention_agreement = -0.91
    assert "RANKINGS ARE INVERTED" in out.means()
    assert "depended on LEAST" in out.means()

    out.attention_agreement = -0.05
    assert "RANKINGS ARE INVERTED" not in out.means()
    assert "ranking the blocks differently" in out.means()

    out.attention_agreement = 0.85
    assert "largely rank the same blocks" in out.means()


def test_the_three_readings_are_exhaustive(frames):
    """Every value in [-1,1] gets exactly one of the three, including the two
    boundaries themselves."""
    out = occ.Occlusion(baseline="episode_mean", grid=[8, 8], stride=1)
    for value in (-1.0, -0.6, -0.599, 0.0, 0.6, 0.601, 1.0):
        out.attention_agreement = value
        means = out.means()
        hits = sum(
            phrase in means
            for phrase in (
                "RANKINGS ARE INVERTED",
                "ranking the blocks differently",
                "largely rank the same blocks",
            )
        )
        assert hits == 1, f"{value} matched {hits} readings"


def test_a_layer_with_nothing_compared_carries_no_layer(frames):
    """Otherwise a caller passing a layer but supplying no map would read as
    'layer 11 agreed on nothing', which is not what happened."""
    out = occ.sweep(
        OneRegionTower().eval(), "cpu", frames[0], grid=[8, 8], patch=8,
        scale_frames=frames, stride=4, compared_layer=11, max_controlled=1,
    )
    assert out.attention_agreement is None
    assert out.compared_layer is None
    assert "layer 11" not in out.means()


def test_no_attention_map_reports_none_rather_than_zero_agreement(frames):
    """Zero agreement is a measurement. No map is the absence of one."""
    out = occ.sweep(
        OneRegionTower().eval(), "cpu", frames[0], grid=[8, 8], patch=8,
        scale_frames=frames, stride=4, max_controlled=1,
    )
    assert out.attention_agreement is None
    assert "No attention map was supplied" in out.means()


def test_spearman_uses_ranks_because_the_two_maps_have_no_shared_unit():
    """Attention is a probability and a causal shift is in embedding-spread
    units; a Pearson correlation between them would be arithmetic across
    incompatible scales."""
    # Perfectly monotonic but wildly different magnitudes.
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [0.001, 0.002, 0.5, 90.0, 9000.0]
    assert occ.spearman(a, b) == pytest.approx(1.0)
    assert occ.spearman(a, list(reversed(b))) == pytest.approx(-1.0)


def test_ties_share_a_rank_rather_than_taking_storage_order():
    """An attention map with many equal values would otherwise produce a
    correlation about the order they happened to be stored in."""
    flat = [1.0, 1.0, 1.0, 1.0]
    assert occ.spearman(flat, [4.0, 3.0, 2.0, 1.0]) is None


def test_too_few_points_to_correlate_is_none():
    assert occ.spearman([1.0, 2.0], [1.0, 2.0]) is None


# ----------------------------------------------------------- the refusals


def test_a_scale_over_one_frame_is_refused():
    """A spread measured over one frame is zero, and every score would be a
    division by nothing."""
    with pytest.raises(BadRequest, match="at least two frames"):
        occ.scale_from([torch.ones(16)])


def test_a_tower_with_no_spread_is_refused_and_points_at_the_audit(frames):
    """The episode may be static, or every frame may be decoding to the same
    picture — which `modelmri audit` checks for by name."""
    with pytest.raises(BadRequest, match="modelmri.*audit"):
        occ.sweep(
            BlindTower().eval(), "cpu", frames[0], grid=[8, 8], patch=8,
            scale_frames=frames, stride=4,
        )


def test_too_many_blocks_is_refused_rather_than_truncated():
    """A map missing its bottom half, presented as a map, is worse than a
    refusal that names the number."""
    with pytest.raises(BadRequest, match="each is one tower pass"):
        occ.plan([32, 32], 1, max_blocks=100)


def test_the_refusal_names_the_block_count_and_the_cap():
    with pytest.raises(BadRequest) as caught:
        occ.plan([32, 32], 1, max_blocks=100)
    assert "1024 blocks" in str(caught.value)
    assert "cap is 100" in str(caught.value)


def test_a_zero_stride_is_refused():
    with pytest.raises(BadRequest, match="at least 1 patch"):
        occ.plan([8, 8], 0)


def test_a_tower_returning_no_hidden_states_is_refused(frames):
    class Empty(nn.Module):
        def forward(self, pixel_values, output_attentions=False):
            return type("O", (), {})()

    with pytest.raises(BadRequest, match="no hidden states"):
        occ.sweep(
            Empty().eval(), "cpu", frames[0], grid=[8, 8], patch=8,
            scale_frames=frames, stride=4,
        )


# ---------------------------------------------------------------- the cost


def test_the_cost_is_answerable_before_the_run():
    """A 32x32 sweep at stride 1 is over a thousand tower passes. Nobody
    should discover that by waiting."""
    fine = occ.estimate([32, 32], 1)
    assert fine["blocks"] == 1024
    assert fine["passes"] > 1024
    coarse = occ.estimate([32, 32], occ.DEFAULT_STRIDE)
    assert coarse["blocks"] == 64
    assert coarse["passes"] < fine["passes"] / 4


def test_the_default_stride_is_coarse_and_fine_is_opt_in():
    assert occ.DEFAULT_STRIDE > 1
    assert occ.estimate([32, 32], occ.DEFAULT_STRIDE)["blocks"] == 64


# ------------------------------------------------- what it must never claim


def test_it_never_says_the_occlusion_caused_the_action(frames):
    """Without the action expert there is no action to affect. The sentence
    says so in those words rather than leaving it to a caption somebody might
    drop."""
    out = occ.sweep(
        OneRegionTower().eval(), "cpu", frames[0], grid=[8, 8], patch=8,
        scale_frames=frames, stride=4, max_controlled=1,
    )
    means = out.means()
    assert "PERCEPTION ONLY" in means
    assert "not an effect on the action" in means
    assert "caused the action" not in means.replace(
        "It must not be read as 'this caused the robot to do that'.", ""
    )


def test_the_report_survives_json(frames):
    out = occ.sweep(
        OneRegionTower().eval(), "cpu", frames[0], grid=[8, 8], patch=8,
        scale_frames=frames, stride=4, max_controlled=1,
    )
    doc = json.loads(json.dumps(out.to_dict(), allow_nan=False))
    assert doc["baseline"] in occ.BASELINES
    assert "means" in doc
    assert len(doc["blocks"]) == doc["n_blocks"]
