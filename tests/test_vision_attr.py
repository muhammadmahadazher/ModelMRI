"""Occlusion attribution, tested against networks whose blind spots are known.

Three real `nn.Module`s, none of them mocks: a two-convolution network that
reads only the top-left quadrant of the image, a linear head over the image's
mean, and one that never looks at the pixels at all. Each has a ground truth
the map either finds or does not, which is the only way to tell a correct
sweep from one that draws a plausible picture of the wrong windows.

Everything else here is about wording and refusals, because a saliency map is
the most persuasive-looking thing this project can draw: a 14x14 array
upsampled over a photograph reads as an explanation, and every sentence these
tests pin exists to stop it claiming more than an occlusion sweep measured.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from modelmri import vision_attr as va  # noqa: E402
from modelmri.errors import BadRequest  # noqa: E402


class CornerNet(nn.Module):
    """Two convolutions and a linear head, reading ONLY the top-left 16x16.

    A real forward pass with a known blind spot: everything outside that
    quadrant is sliced away before the first convolution, so a sweep that
    scores anything out there has invented it.
    """

    def __init__(self, classes: int = 3) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.conv1 = nn.Conv2d(3, 4, 3, padding=1)
        self.conv2 = nn.Conv2d(4, 4, 3, padding=1)
        self.head = nn.Linear(4, classes)

    def forward(self, x):
        h = torch.relu(self.conv1(x[..., :16, :16]))
        h = self.conv2(h)
        return self.head(h.mean(dim=(-2, -1)))


class MeanNet(nn.Module):
    """A real linear head over one measured number: the image's mean pixel.

    Class 0 rises with the mean and class 1 falls with it, so on an image with
    a bright half and a dark half the two halves push class 0 in opposite
    directions — which is what makes a signed map testable.
    """

    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(1, 2, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(torch.tensor([[10.0], [-10.0]]))

    def forward(self, x):
        return self.head(x.mean(dim=(1, 2, 3)).unsqueeze(1))


class BlindNet(nn.Module):
    """A real linear head that never reads a pixel."""

    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(1, 3)

    def forward(self, x):
        ones = torch.ones(int(x.shape[0]), 1, dtype=x.dtype, device=x.device)
        return self.head(ones)


class OneScoreNet(nn.Module):
    """A single output, the way an objectness or regression head has one."""

    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(1, 1)

    def forward(self, x):
        return self.head(x.mean(dim=(1, 2, 3)).unsqueeze(1))


class BoxNet(nn.Module):
    """`[batch, boxes, scores]` — the shape a detector actually returns."""

    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(1, 6)

    def forward(self, x):
        flat = self.head(x.mean(dim=(1, 2, 3)).unsqueeze(1))
        return flat.reshape(int(x.shape[0]), 3, 2)


@pytest.fixture
def photo():
    """32x32 with a bright left half and a dark right half.

    Deliberately straddling the grey fill: occluding the bright side lowers
    the mean and occluding the dark side raises it, so a signed map has to
    carry both signs on one picture.
    """
    image = torch.full((1, 3, 32, 32), 0.1)
    image[..., :, :16] = 0.9
    return image


@pytest.fixture
def noise():
    torch.manual_seed(1)
    return torch.rand(1, 3, 32, 32)


# ------------------------------------------------------------- the geometry


def test_a_stride_wider_than_the_patch_leaves_pixels_no_window_ever_covers():
    """The map would have holes in it and still be shaped like a complete map
    of the image, so a region nobody measured reads as a region that did
    nothing."""
    with pytest.raises(BadRequest) as caught:
        va.plan_windows(64, 64, patch=8, stride=16)
    said = str(caught.value)
    assert "50% of the pixels under no window" in said
    assert "Set the stride to 8 or less" in said


def test_the_last_window_is_clamped_to_the_edge_so_no_strip_goes_unmeasured():
    """Without the clamp a 30-pixel axis at patch 8 stride 8 stops at 16 and
    the final six pixel columns are never occluded — the map would be silent
    about a strip of the image while being presented as a map of it."""
    grid = va.plan_windows(30, 30, patch=8, stride=8)
    assert grid.tops == [0, 8, 16, 22]
    assert grid.edge_row_clamped is True
    covered = set()
    for top in grid.tops:
        covered.update(range(top, top + grid.patch))
    assert covered == set(range(30)), "a pixel row went unmeasured"


def test_an_unclamped_axis_does_not_claim_to_have_been_clamped():
    """The flag is a fact about this map. A grid that divides exactly has no
    extra overlap and must not say it does."""
    grid = va.plan_windows(32, 32, patch=8, stride=8)
    assert grid.edge_row_clamped is False
    assert grid.edge_col_clamped is False
    assert grid.rows == grid.cols == 4


def test_a_patch_that_covers_the_whole_image_is_refused_as_not_a_map():
    """Occluding everything measures whether the model can see at all, which
    is a different question from where in the image it was looking."""
    with pytest.raises(BadRequest, match="whether the model can see at all"):
        va.plan_windows(16, 16, patch=16, stride=16)


def test_a_patch_larger_than_the_image_is_refused_before_the_arithmetic():
    with pytest.raises(BadRequest, match="does not fit inside"):
        va.plan_windows(16, 16, patch=32)


def test_a_boolean_patch_is_refused_because_isinstance_true_int_is_true():
    """`patch=True` would sail through an isinstance check and become a patch
    of 1 — a per-pixel map and fifty thousand forward passes that look like
    somebody asked for them."""
    with pytest.raises(BadRequest) as caught:
        va.plan_windows(32, 32, patch=True)
    assert "isinstance(True, int)` is True" in str(caught.value)


def test_a_zero_stride_is_refused_rather_than_looping_forever():
    with pytest.raises(BadRequest, match="same place forever"):
        va.plan_windows(32, 32, patch=8, stride=0)


def test_the_default_stride_is_a_plain_tiling_that_covers_each_pixel_once():
    grid = va.plan_windows(32, 32, patch=8)
    assert grid.stride == 8
    assert grid.overlap == 0
    assert grid.n_windows == 16


# ----------------------------------------------------------------- the cost


def test_the_cost_is_answerable_before_a_single_pass_is_taken():
    """A 224 image at stride 1 is over forty thousand forward passes. Nobody
    should discover that by waiting."""
    fine = va.estimate(224, 224, patch=16, stride=1)
    coarse = va.estimate(224, 224, patch=16, stride=16)
    assert fine["n_windows"] == 209 * 209
    assert coarse["n_windows"] == 196
    assert coarse["passes"] == 197
    assert fine["passes"] > coarse["passes"] * 100


def test_estimate_never_refuses_the_run_it_is_pricing():
    """A caller about to be refused needs the number that got them refused. An
    estimate that declined to produce it would leave them guessing at the
    stride."""
    priced = va.estimate(224, 224, patch=16, stride=1)
    assert priced["within_ceiling"] is False
    assert "PAST THE CEILING" in priced["means"]
    assert "would be" in priced["means"] and "and fits" in priced["means"]


def test_a_run_past_the_ceiling_is_refused_and_names_a_stride_that_fits():
    """A refusal that only says no leaves the caller guessing at the parameter
    the arithmetic already knows."""
    with pytest.raises(BadRequest) as caught:
        va.plan_windows(224, 224, patch=16, stride=1)
    said = str(caught.value)
    assert "past the ceiling of 4096" in said
    assert "would be" in said and "and fits" in said
    assert "Nothing was run" in said


def test_an_unmeasured_pass_time_is_none_rather_than_zero_seconds():
    """None is "nobody measured" and 0 is "instant", and only one of them is
    true. A default per-pass time would be a forecast this tool made up."""
    priced = va.estimate(64, 64, patch=16)
    assert priced["seconds"] is None
    assert "No per-pass time was measured" in priced["means"]
    timed = va.estimate(64, 64, patch=16, seconds_per_pass=0.25)
    assert timed["seconds"] == pytest.approx(priced["passes"] * 0.25)


def test_a_negative_pass_time_is_refused_rather_than_forecast():
    with pytest.raises(BadRequest, match="finite, non-negative"):
        va.estimate(64, 64, patch=16, seconds_per_pass=-1.0)


def test_the_batch_memory_is_the_part_that_can_be_computed_exactly():
    """Activations are a multiple of this that nothing here can know without
    running the model, so they are absent rather than guessed at."""
    priced = va.estimate(224, 224, patch=16, batch=32)
    assert priced["input_bytes_per_call"] == 32 * 3 * 224 * 224 * 4
    assert "activations behind them are a multiple" in priced["means"]


# ------------------------------------------------------- what the sweep finds


def test_the_sweep_finds_the_quadrant_the_network_actually_reads(noise):
    """Ground truth: this network slices away everything but the top-left
    16x16 before its first convolution. A map that ranks anything else has
    invented it."""
    out = va.sweep(CornerNet().eval(), noise, patch=8, batch=8)
    assert out.grid.rows == out.grid.cols == 4
    top4 = sorted(out.windows, key=lambda w: -abs(w.logit_drop))[:4]
    assert all(w.row < 2 and w.col < 2 for w in top4), (
        "the sweep ranked windows the network cannot see"
    )


def test_windows_the_network_cannot_see_score_zero_and_not_a_small_number(noise):
    """Anything non-zero out there is arithmetic noise being drawn as
    evidence, and on a heatmap noise is indistinguishable from a faint
    finding."""
    out = va.sweep(CornerNet().eval(), noise, patch=8, batch=8)
    outside = [w for w in out.windows if w.row >= 2 or w.col >= 2]
    assert len(outside) == 12
    for w in outside:
        assert w.logit_drop == 0.0, f"row {w.row} col {w.col} is not read and moved"


def test_a_model_that_ignores_the_image_yields_no_peak_rather_than_the_first(photo):
    """A flat map is a result — it says the answer did not depend on any part
    of this picture — and picking a strongest window out of an exact tie would
    invent one out of storage order."""
    out = va.sweep(BlindNet().eval(), photo, patch=8)
    assert out.spread == 0.0
    assert out.strongest is None
    assert out.to_dict()["strongest"] is None
    assert "NOTHING IN THIS IMAGE MOVED THE OUTPUT" in out.means()


def test_a_window_that_argued_against_the_class_keeps_its_negative_sign(photo):
    """Covering the dark half RAISES this class. An absolute value would draw
    a region that argues against the class as evidence for it."""
    out = va.sweep(MeanNet().eval(), photo, patch=8, target=0)
    bright = [w for w in out.windows if w.left < 16]
    dark = [w for w in out.windows if w.left >= 16]
    assert all(w.logit_drop > 0 for w in bright)
    assert all(w.logit_drop < 0 for w in dark)
    assert out.most_negative is not None
    assert "arguing against the class" in out.means()


def test_a_class_nothing_argued_against_reports_none_and_not_a_zero(photo):
    """None is "no region opposed this class"; 0.0 would be a window that was
    measured and moved nothing. Different answers."""
    out = va.sweep(MeanNet().eval(), photo, patch=8, target=0, fill="black")
    assert min(w.logit_drop for w in out.windows) == 0.0
    assert out.most_negative is None
    assert out.to_dict()["most_negative"] is None


def test_the_peak_is_stated_as_relative_to_this_image_and_not_absolute(photo):
    """A "most important region" with no spread beside it reads as a property
    of the model rather than a rank among sixteen windows of one picture."""
    out = va.sweep(MeanNet().eval(), photo, patch=8, target=0)
    said = out.means()
    assert "only a peak relative to the other 15 windows OF THIS IMAGE" in said
    assert "ONE IMAGE IS A SAMPLE, NOT A PROPERTY OF THE MODEL" in said


# ----------------------------------------------------------------- the fill


def test_the_fill_changes_the_answer_so_a_result_that_hid_it_would_be_wrong(photo):
    """Grey and black are different baselines, not different renderings of one
    measurement — the same image and the same model give different maps."""
    grey = va.sweep(MeanNet().eval(), photo, patch=8, target=0, fill="grey")
    black = va.sweep(MeanNet().eval(), photo, patch=8, target=0, fill="black")
    assert grey.map_rows() != black.map_rows()
    assert "grey fill" in grey.means()
    assert "black fill" in black.means()


def test_mean_substitution_is_never_described_as_removal(photo):
    out = va.sweep(MeanNet().eval(), photo, patch=8, fill="image_mean")
    said = out.means()
    assert "MEAN SUBSTITUTION IS A SPECIFIC BASELINE, NOT REMOVAL" in said
    assert "it is not absence" in said
    assert "only replace it" in said


def test_every_fill_says_occlusion_is_out_of_distribution(photo):
    """A flat square is itself a stimulus the model has never seen, so part of
    every score is the square rather than the missing content."""
    for fill in va.FILLS:
        out = va.sweep(MeanNet().eval(), photo, patch=8, fill=fill)
        assert out.fill == fill
        assert "OCCLUSION IS OUT OF DISTRIBUTION" in out.means()
        assert "keep what survives both" in out.means()


def test_an_unknown_fill_is_refused_rather_than_quietly_defaulted(photo):
    with pytest.raises(BadRequest) as caught:
        va.sweep(MeanNet().eval(), photo, patch=8, fill="blurred")
    assert "The fill changes the answer" in str(caught.value)


def test_the_fill_is_read_from_the_value_range_rather_than_assumed(photo):
    """ "Grey is 0.5" is wrong for an ImageNet-normalised tensor and for a
    [-1,1] one. The number that was actually used travels in the result."""
    out = va.sweep(MeanNet().eval(), photo, patch=8, value_range=(-1.0, 1.0))
    assert out.value_range == (-1.0, 1.0)
    assert out.fill_value == [0.0]
    assert out.value_range_inferred is False
    assert "INFERRED FROM THIS ONE IMAGE" not in out.means()


def test_an_inferred_range_says_it_was_inferred_from_one_image(photo):
    """One image's observed extremes are a lower bound on what the model
    accepts, not its input range."""
    out = va.sweep(MeanNet().eval(), photo, patch=8)
    assert out.value_range_inferred is True
    assert out.value_range == (0.1, 0.9)
    assert out.fill_value == [pytest.approx(0.5)]
    assert "INFERRED FROM THIS ONE IMAGE" in out.means()


def test_an_image_with_no_contrast_is_refused_rather_than_mapped_as_zeros():
    """Every fill drawn from its own range would be the pixel already there,
    so the map would be a flat zero — which looks exactly like a model that
    ignores its input."""
    flat = torch.full((1, 3, 32, 32), 0.4)
    with pytest.raises(va.NotAttributable) as caught:
        va.sweep(MeanNet().eval(), flat, patch=8)
    assert "no range to draw a fill from" in str(caught.value)
    assert "ignores its input" in str(caught.value)


def test_a_backwards_value_range_is_refused():
    with pytest.raises(BadRequest, match="high greater than low"):
        va.sweep(
            MeanNet().eval(),
            torch.rand(1, 3, 32, 32),
            patch=8,
            value_range=(1.0, 0.0),
        )


# --------------------------------------------------------------- the target


def test_the_class_the_model_chose_is_distinguished_from_one_you_supplied(photo):
    """Explaining the model's own answer and auditing a label you named are
    different questions that produce the same picture."""
    chosen = va.sweep(MeanNet().eval(), photo, patch=8)
    assert chosen.target_chosen_by_model is True
    assert "the model chose itself" in chosen.means()
    assert "nobody said it is the right answer" in chosen.means()

    given = va.sweep(MeanNet().eval(), photo, patch=8, target=1)
    assert given.target_chosen_by_model is False
    assert "which you named" in given.means()
    assert "whether or not the model predicted it" in given.means()


def test_a_target_outside_the_class_count_is_refused_naming_the_count(photo):
    with pytest.raises(BadRequest) as caught:
        va.sweep(MeanNet().eval(), photo, patch=8, target=9)
    assert "this head's 2 classes (0 to 1)" in str(caught.value)


def test_a_boolean_target_is_refused_rather_than_becoming_class_one(photo):
    """`target=True` is class 1 to isinstance and a mistake to everyone
    else."""
    with pytest.raises(BadRequest, match="isinstance"):
        va.sweep(MeanNet().eval(), photo, patch=8, target=True)


def test_class_names_of_the_wrong_length_are_dropped_entirely(photo):
    """A name list of the wrong length mislabels at least one class, and a map
    captioned with the wrong class name is worse than one captioned with a
    number."""
    out = va.sweep(
        MeanNet().eval(), photo, patch=8, target=0, class_names=["cat", "dog", "fox"]
    )
    assert out.class_names_dropped is True
    assert out.target_label == ""
    assert "ALL of them were dropped" in out.means()


def test_class_names_that_do_match_are_used(photo):
    out = va.sweep(
        MeanNet().eval(), photo, patch=8, target=0, class_names=["bright", "dark"]
    )
    assert out.target_label == "bright"
    assert out.class_names_dropped is False
    assert "bright" in out.means()


# ---------------------------------------------------------- the probability


def test_softmax_confidence_is_never_called_the_probability_of_correctness(photo):
    """It is high when the model is confidently wrong, it moves when an
    unrelated class moves, and nothing here has calibrated it."""
    out = va.sweep(MeanNet().eval(), photo, patch=8, target=0)
    said = out.means()
    assert "SOFTMAX CONFIDENCE IS NOT THE PROBABILITY OF BEING RIGHT" in said
    assert "confidently wrong" in said
    assert out.base_prob is not None
    assert all(w.prob_drop is not None for w in out.windows)


def test_a_single_output_head_reports_no_probability_rather_than_one(photo):
    """A softmax over one number is 1.0 by construction. Printing "confidence
    1.00" would be a number with no information in it."""
    out = va.sweep(OneScoreNet().eval(), photo, patch=8)
    assert out.classes == 1
    assert out.base_prob is None
    assert all(w.prob_drop is None for w in out.windows)
    assert "softmax over one number is 1.0 by construction" in out.means()


def test_the_logit_is_the_primary_score_because_a_probability_is_shared(photo):
    """A probability moves when any other class moves, so a map drawn from it
    confounds "this supported the class" with "this suppressed another"."""
    out = va.sweep(MeanNet().eval(), photo, patch=8, target=0)
    assert "The logit movement above is the primary score" in out.means()


# --------------------------------------------------------------- the batching


def test_batching_changes_the_schedule_and_nothing_else(photo):
    """A batching bug that mixed windows up would draw a perfectly plausible
    map of the wrong regions, and nothing on screen would say so."""
    one = va.sweep(MeanNet().eval(), photo, patch=8, target=0, batch=1)
    many = va.sweep(MeanNet().eval(), photo, patch=8, target=0, batch=64)
    assert one.map_rows() == many.map_rows()
    assert one.forward_calls > many.forward_calls


def test_the_pass_count_is_every_window_plus_the_unoccluded_reference(photo):
    out = va.sweep(MeanNet().eval(), photo, patch=8, batch=5)
    assert out.passes == out.grid.n_windows + 1 == 17
    # 16 windows in batches of five is four calls, plus the reference run.
    assert out.forward_calls == 5
    assert f"{out.passes} forward passes in 5 batched calls" in out.means()


def test_a_batch_larger_than_the_bound_is_reported_and_not_silently_capped(photo):
    """A silently reduced batch is a silently different run time, and somebody
    will time it."""
    out = va.sweep(MeanNet().eval(), photo, patch=8, batch=4096)
    assert out.batch_requested == 4096
    assert out.batch_used == va.MAX_BATCH
    assert "A batch of 4096 was asked for and 64 was used" in out.means()


def test_a_batch_within_the_bound_says_nothing_about_being_capped(photo):
    out = va.sweep(MeanNet().eval(), photo, patch=8, batch=8)
    assert "was asked for and" not in out.means()


def test_a_zero_batch_is_refused(photo):
    with pytest.raises(BadRequest, match="at least 1 occluded copy"):
        va.sweep(MeanNet().eval(), photo, patch=8, batch=0)


# --------------------------------------------------------------- the refusals


def test_a_model_in_training_mode_is_refused_because_dropout_is_not_occlusion():
    """The same input twice is two different answers in training mode, so
    every score in the map would be that noise as much as the occlusion."""
    model = CornerNet()
    assert model.training is True
    with pytest.raises(va.NotAttributable) as caught:
        va.sweep(model, torch.rand(1, 3, 32, 32), patch=8)
    said = str(caught.value)
    assert "`model.eval()`" in said
    assert "This will not do it for you" in said
    assert model.training is True, "the refusal must not mutate the caller's model"


def test_a_non_finite_logit_is_refused_rather_than_left_as_a_hole_in_the_map(photo):
    """A gap in a heatmap reads as "this region did nothing", which is not
    what happened."""

    class NaNNet(nn.Module):
        def forward(self, x):
            out = torch.zeros(int(x.shape[0]), 2)
            out[:, 0] = float("nan")
            return out

    with pytest.raises(va.NotAttributable, match="no reference"):
        va.sweep(NaNNet().eval(), photo, patch=8)


def test_a_non_finite_pixel_is_refused_before_anything_runs(photo):
    broken = photo.clone()
    broken[0, 0, 0, 0] = float("nan")
    with pytest.raises(va.NotAttributable, match="not numbers"):
        va.sweep(MeanNet().eval(), broken, patch=8)


def test_a_batch_of_images_is_refused_because_one_map_cannot_cover_several():
    with pytest.raises(BadRequest, match="Attribution is about ONE image"):
        va.sweep(MeanNet().eval(), torch.rand(4, 3, 32, 32), patch=8)


def test_an_integer_image_tensor_is_refused_because_the_fill_would_truncate():
    """A grey of 0.5 truncates to 0 in an integer tensor, which is black — the
    map would be of a different experiment from the one requested."""
    with pytest.raises(BadRequest, match="truncates to 0, which is black"):
        va.sweep(
            MeanNet().eval(), torch.randint(0, 255, (1, 3, 32, 32), dtype=torch.uint8)
        )


def test_something_that_is_not_a_tensor_is_refused_by_name():
    with pytest.raises(BadRequest, match="does no image loading of its own"):
        va.sweep(MeanNet().eval(), [[0.0]], patch=8)


def test_a_bare_three_dimensional_image_is_accepted(photo):
    """[C, H, W] is what a transform pipeline hands back, and refusing it
    would be pedantry rather than a safeguard."""
    out = va.sweep(MeanNet().eval(), photo[0], patch=8)
    assert out.grid.height == 32


# ------------------------------------------------------- reading the output


def test_an_output_this_cannot_read_points_at_the_forward_hook(photo):
    """Guessing which of several returned tensors is "the" output would pick
    what the map is about by accident."""
    with pytest.raises(va.NotAttributable, match="forward="):
        va.sweep(BoxNet().eval(), photo, patch=8)


def test_a_forward_hook_lets_a_detector_be_measured_as_a_black_box(photo):
    """The whole point of treating the model as a black box: anything that can
    be reduced to [batch, classes] is measurable, detector included."""
    out = va.sweep(
        BoxNet().eval(),
        photo,
        patch=8,
        forward=lambda model, x: model(x).max(dim=1).values,
    )
    assert out.classes == 2
    assert len(out.windows) == 16


def test_a_dict_output_without_logits_says_which_keys_it_found(photo):
    class DictNet(nn.Module):
        def forward(self, x):
            return {"boxes": torch.zeros(int(x.shape[0]), 4)}

    with pytest.raises(va.NotAttributable) as caught:
        va.sweep(DictNet().eval(), photo, patch=8)
    assert "['boxes']" in str(caught.value)


def test_a_logits_attribute_is_read_the_way_transformers_returns_one(photo):
    class HFNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Linear(1, 2)

        def forward(self, x):
            flat = self.head(x.mean(dim=(1, 2, 3)).unsqueeze(1))
            return type("Output", (), {"logits": flat})()

    out = va.sweep(HFNet().eval(), photo, patch=8)
    assert out.classes == 2
    assert len(out.windows) == 16


# ------------------------------------------------------------- the reporting


def test_the_map_resolution_is_reported_and_said_to_be_coarser(photo):
    """A 4x4 array upsampled over a 32x32 picture is the most convincing lie
    this module could tell."""
    out = va.sweep(MeanNet().eval(), photo, patch=8)
    said = out.means()
    assert "THE MAP IS 4x4, ONE CELL PER WINDOW" in said
    assert "coarser than the image by a factor of 8" in said
    assert "whatever an upsampled heatmap appears to show" in said


def test_the_map_is_rows_by_columns_with_every_cell_filled(photo):
    """The grid covers the image completely by construction, so a caller never
    has to decide what a missing cell means."""
    out = va.sweep(MeanNet().eval(), photo, patch=8)
    table = out.map_rows()
    assert len(table) == 4
    assert all(len(row) == 4 for row in table)
    assert sorted(w.logit_drop for w in out.windows) == sorted(
        v for row in table for v in row
    )


def test_overlapping_windows_say_their_scores_are_not_independent(photo):
    out = va.sweep(MeanNet().eval(), photo, patch=8, stride=4)
    assert out.grid.overlap == 4
    assert "not independent of one another" in out.means()


def test_a_clamped_edge_is_reported_rather_than_hidden():
    """The final window overlaps its neighbour by more than the stride, which
    is a fact about this map."""
    image = torch.rand(1, 3, 30, 30)
    out = va.sweep(MeanNet().eval(), image, patch=8, stride=8)
    assert out.grid.edge_row_clamped and out.grid.edge_col_clamped
    assert "pulled back to the edge" in out.means()
    assert "row and column" in out.means()


def test_the_report_survives_json(photo):
    out = va.sweep(MeanNet().eval(), photo, patch=8, target=0)
    doc = json.loads(json.dumps(out.to_dict(), allow_nan=False))
    assert doc["fill"] in va.FILLS
    assert len(doc["windows"]) == doc["grid"]["n_windows"]
    assert len(doc["map"]) == doc["grid"]["map_rows"]
    assert "means" in doc


def test_the_model_name_travels_when_it_is_given(photo):
    out = va.sweep(MeanNet().eval(), photo, patch=8, model_name="google/vit-base")
    assert "google/vit-base" in out.means()
