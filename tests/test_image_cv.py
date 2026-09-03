# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""The computer-vision instrument, tested against nets whose answers are known.

Every model here is a real `nn.Module` with a real forward pass — a tiny
transformer that computes genuine softmax attention, a two-region classifier
whose two classes read two different corners of the image, a detector whose
query slots read different corners, a segmenter that labels a grid. None is a
mock, because the thing being checked is whether the instrument finds what is
actually there: a map that is drawn correctly and a map that is drawn
convincingly look identical, and only a network with a known blind spot tells
them apart.

The rest is about wording, and it is not decoration. A prediction panel is the
most quotable thing this project produces — a class name and a percentage — so
every sentence pinned here exists to stop one of them claiming more than a
forward pass measured: that the names came from this checkpoint, that a
softmax is not a probability of being right, that a query slot is fixed while
the box it draws is not, and that `None` for "could not be measured" never
becomes `0`.

Nothing here downloads. Nothing here needs a GPU.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from modelmri import image_cv as cv  # noqa: E402
from modelmri.errors import BadRequest  # noqa: E402


class Out(dict):
    """A stand-in for a transformers `ModelOutput`: attributes AND keys.

    Both accesses are real on the thing this module is pointed at, and both
    are read by it — `_tensor_of` prefers the attribute and `_output_keys`
    needs the keys to be able to name what a model returned in a refusal. A
    test double that offered only one would leave the other path unexercised.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


class Config:
    """A config object, which is what transformers hands back."""

    def __init__(self, **fields) -> None:
        self.__dict__.update(fields)


def _labels(count: int, prefix: str = "thing") -> dict:
    """`id2label` with STRING keys, which is what a JSON round-trip gives."""
    return {str(i): f"{prefix} {i}" for i in range(count)}


# --------------------------------------------------------------- the models


class TwoRegionNet(nn.Module):
    """Class 0 reads only the top-left quadrant, class 1 only the bottom-right.

    A real forward pass with two known blind spots, and the point of having
    two: attributing class 0 and attributing class 1 must produce different
    maps on the same image, or the module is drawing the argmax whatever it
    was asked for.
    """

    def __init__(self, size: int = 32) -> None:
        super().__init__()
        self.half = size // 2
        self.config = Config(id2label=_labels(2, "corner"))

    def forward(self, x):
        half = self.half
        top_left = x[..., :half, :half].mean(dim=(1, 2, 3))
        bottom_right = x[..., half:, half:].mean(dim=(1, 2, 3))
        return Out(logits=torch.stack([top_left, bottom_right], dim=-1) * 10.0)


class MultiLabelNet(nn.Module):
    """A head the checkpoint declares MULTI-LABEL, which is a sigmoid."""

    def __init__(self) -> None:
        super().__init__()
        self.config = Config(
            id2label=_labels(3, "tag"),
            problem_type="multi_label_classification",
        )

    def forward(self, x):
        mean = x.mean(dim=(1, 2, 3)).unsqueeze(1)
        return Out(logits=(mean * torch.tensor([[6.0, 5.0, 4.0]])))


class RegressionNet(nn.Module):
    """A head the checkpoint declares a REGRESSION, which has no classes."""

    def __init__(self) -> None:
        super().__init__()
        self.config = Config(problem_type="regression", id2label={"0": "score"})

    def forward(self, x):
        return Out(logits=x.mean(dim=(1, 2, 3)).unsqueeze(1))


class TinyDetector(nn.Module):
    """Three query slots; slot 0 reads the top-left, slot 1 the bottom-right.

    Four class columns for three labels, which is the DETR shape: the last
    column is "no object" and the module has to work that out from the widths
    rather than from anything this class is called.
    """

    def __init__(self, size: int = 32, columns: int = 4, labels: int = 3) -> None:
        super().__init__()
        self.half = size // 2
        self.columns = columns
        self.config = Config(id2label=_labels(labels, "object"))

    def forward(self, x):
        half = self.half
        batch = int(x.shape[0])
        top_left = x[..., :half, :half].mean(dim=(1, 2, 3))
        bottom_right = x[..., half:, half:].mean(dim=(1, 2, 3))
        quiet = torch.full_like(top_left, -4.0)
        # Slot 0 votes class 0 on the top-left, slot 1 votes class 1 on the
        # bottom-right, slot 2 votes for nothing at all.
        logits = torch.full((batch, 3, self.columns), -4.0)
        logits[:, 0, 0] = top_left * 10.0
        logits[:, 1, 1] = bottom_right * 10.0
        logits[:, 2, self.columns - 1] = quiet + 8.0
        boxes = torch.tensor([[[0.25, 0.25, 0.5, 0.5]] * 3]).expand(batch, 3, 4)
        return Out(logits=logits, pred_boxes=boxes.contiguous())


class TinySegmenter(nn.Module):
    """A per-pixel head on a 4x4 grid: the left half is class 1, the rest 0."""

    def __init__(self, classes: int = 3, grid: int = 4) -> None:
        super().__init__()
        self.classes = classes
        self.grid = grid
        self.config = Config(id2label=_labels(classes, "stuff"))

    def forward(self, x):
        batch = int(x.shape[0])
        grid = self.grid
        # The score for class 1 on the left half is driven by the image's own
        # left half, so occluding there is measurable rather than constant.
        left = x[..., :, : x.shape[-1] // 2].mean(dim=(1, 2, 3))
        logits = torch.zeros(batch, self.classes, grid, grid)
        logits[:, 1, :, : grid // 2] = left.reshape(batch, 1, 1) * 10.0 + 1.0
        logits[:, 0, :, grid // 2 :] = 5.0
        return Out(logits=logits)


class TinyMaskHead(nn.Module):
    """A mask-query head: a class vector and a mask per query slot."""

    def __init__(self, grid: int = 4) -> None:
        super().__init__()
        self.grid = grid
        self.config = Config(id2label=_labels(2, "region"))

    def forward(self, x):
        batch = int(x.shape[0])
        grid = self.grid
        classes = torch.full((batch, 2, 3), -6.0)
        classes[:, 0, 0] = 6.0
        classes[:, 1, 1] = 6.0
        masks = torch.full((batch, 2, grid, grid), -6.0)
        # Query 0 claims the top half; query 1 claims one corner; the rest of
        # the grid is claimed by nobody, which must come back as -1.
        masks[:, 0, : grid // 2, :] = 6.0
        masks[:, 1, grid - 1, grid - 1] = 6.0
        return Out(class_queries_logits=classes, masks_queries_logits=masks)


class PromptedNet(nn.Module):
    """A promptable segmenter: masks for a prompt, and no prompt was given."""

    def forward(self, x):
        return Out(
            pred_masks=torch.zeros(int(x.shape[0]), 1, 4, 4),
            iou_scores=torch.zeros(int(x.shape[0]), 1),
        )


class TinyViT(nn.Module):
    """A real transformer: patch embedding, a class token, genuine attention.

    `nn.MultiheadAttention` computes the same softmax a ViT does, and asking
    it for un-averaged weights gives exactly the `[batch, heads, q, k]` the
    readout reduces. Nothing is faked — the rows sum to one because a softmax
    put them there.
    """

    def __init__(self, size: int = 32, patch: int = 16, heads: int = 2) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.dim = 8
        self.embed = nn.Conv2d(3, self.dim, patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, self.dim))
        self.blocks = nn.ModuleList(
            nn.MultiheadAttention(self.dim, heads, batch_first=True) for _ in range(3)
        )
        self.head = nn.Linear(self.dim, 4)
        self.config = Config(
            patch_size=patch,
            image_size=size,
            num_attention_heads=heads,
            num_hidden_layers=3,
            id2label=_labels(4, "class"),
        )

    def forward(self, x, output_hidden_states=False, output_attentions=False):
        h = self.embed(x).flatten(2).transpose(1, 2)
        h = torch.cat([self.cls.expand(int(h.shape[0]), -1, -1), h], dim=1)
        hidden = [h]
        attention = []
        for block in self.blocks:
            done, weights = block(
                h, h, h, need_weights=True, average_attn_weights=False
            )
            h = h + done
            hidden.append(h)
            attention.append(weights)
        found = Out(logits=self.head(h[:, 0]))
        if output_hidden_states:
            found["hidden_states"] = tuple(hidden)
        if output_attentions:
            found["attentions"] = tuple(attention)
        return found


class SdpaViT(TinyViT):
    """The same transformer under a kernel that never builds the matrix.

    Measured behaviour, not invented: on transformers 5.14.1 an SDPA model
    asked for `output_attentions=True` returns an EMPTY TUPLE and warns to a
    log. The empty tuple is the case that must not be reported as a model
    with no attention.
    """

    def forward(self, x, output_hidden_states=False, output_attentions=False):
        found = super().forward(x, output_hidden_states=output_hidden_states)
        if output_attentions:
            found["attentions"] = ()
        return found


class TinyCNN(nn.Module):
    """A convolutional stack: spatial hidden states and no attention at all."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(1)
        self.one = nn.Conv2d(3, 4, 3, stride=2, padding=1)
        self.two = nn.Conv2d(4, 4, 3, stride=2, padding=1)
        self.head = nn.Linear(4, 3)
        self.config = Config(id2label=_labels(3, "kind"))

    def forward(self, x, output_hidden_states=False, output_attentions=False):
        a = torch.relu(self.one(x))
        b = torch.relu(self.two(a))
        found = Out(logits=self.head(b.mean(dim=(-2, -1))))
        if output_hidden_states:
            found["hidden_states"] = (a, b)
        return found


class SilentNet(nn.Module):
    """A model that returns logits and nothing else to look inside."""

    def __init__(self) -> None:
        super().__init__()
        self.config = Config(id2label=_labels(2))

    def forward(self, x, output_hidden_states=False, output_attentions=False):
        return Out(logits=x.mean(dim=(1, 2, 3)).unsqueeze(1).expand(-1, 2))


# ------------------------------------------------------------------ images


@pytest.fixture
def corners():
    """32x32, bright top-left, medium bottom-right, dark elsewhere."""
    image = torch.full((1, 3, 32, 32), 0.1)
    image[..., :16, :16] = 0.9
    image[..., 16:, 16:] = 0.6
    return image


UNIT = (0.0, 1.0)


def _eval(model):
    model.eval()
    return model


# ------------------------------------------------------------- class names


def test_class_names_are_ordered_by_the_integer_key_not_the_string_one():
    """transformers hands `id2label` back with string keys, and sorting those
    as text puts "10" immediately after "1" — a list built that way puts every
    name against the wrong class and looks entirely reasonable doing it."""
    model = _eval(TwoRegionNet())
    model.config.id2label = {str(i): f"class {i}" for i in range(12)}
    assert cv.label_names(model)[10] == "class 10"
    assert cv.label_names(model)[1] == "class 1"


def test_a_checkpoint_with_no_id2label_gets_indices_and_says_so(corners):
    """Borrowing another model's class list for a head that happens to be the
    same width is how a fine-tune answers with somebody else's vocabulary, and
    it would look exactly as authoritative as the truth."""
    model = _eval(TwoRegionNet())
    del model.config.id2label
    found = cv.predict(model, corners).to_dict()

    assert found["labels_read"] is False
    assert found["labels_published"] is None
    assert all(row["label"] == "" for row in found["classes_top"])
    assert "reported as INDICES" in found["means"]
    assert "will borrow another" in found["means"]


def test_names_of_the_wrong_length_are_dropped_rather_than_applied(corners):
    """A name list that does not match the head mislabels at least one class,
    and a wrong name is worse than an index."""
    model = _eval(TwoRegionNet())
    model.config.id2label = _labels(5)
    found = cv.predict(model, corners).to_dict()

    assert found["labels_read"] is False
    assert found["labels_published"] == 5
    assert "the names are NOT applied" in found["means"]


# ------------------------------------------------------- device and dtype


def test_a_tensor_on_the_wrong_device_is_refused_before_any_work(corners):
    """Copying it here instead would move the whole batch across the bus on
    every forward call of the sweep, which is most of its run time, and
    nothing in the result would say so."""
    model = _eval(TinyCNN()).to("meta")
    with pytest.raises(cv.NotMeasurable) as caught:
        cv.predict(model, corners)
    said = str(caught.value)
    assert "meta" in said and "cpu" in said
    assert "device=" in said


def test_a_narrow_dtype_is_cast_at_the_forward_and_the_narrowing_is_reported(
    corners,
):
    """A bfloat16 model cannot multiply a float32 tensor at all, so the cast
    has to happen — and a logit printed at six decimals when the weights carry
    three is four digits of measurement and two of formatting."""
    model = _eval(TinyCNN()).to(torch.bfloat16)
    found = cv.predict(model, corners).to_dict()

    assert found["dtype"] == "torch.bfloat16"
    assert "carries far fewer significant digits" in found["means"]
    # The tensor handed in is untouched: the occluder's arithmetic stays exact.
    assert corners.dtype == torch.float32


# ---------------------------------------------------------- what task it is


def test_the_task_is_read_from_the_output_shape_and_not_from_any_name(corners):
    """Nothing here is allowed to know a model name — a checkpoint published
    next week that follows these conventions has to work unchanged."""
    assert cv.predict(_eval(TwoRegionNet()), corners).task == cv.CLASSIFY
    assert cv.predict(_eval(TinyDetector()), corners).task == cv.DETECT
    assert cv.predict(_eval(TinySegmenter()), corners).task == cv.SEMANTIC
    assert cv.predict(_eval(TinyMaskHead()), corners).task == cv.MASK_QUERIES


def test_an_output_this_cannot_read_is_refused_by_name_of_what_it_returned():
    """Reading it as whichever branch came first would answer a question the
    model did not ask, and the refusal has to be actionable."""
    with pytest.raises(cv.NotMeasurable) as caught:
        cv.task_of(Out(embeddings=torch.zeros(1, 4), pooled=torch.zeros(1, 4)))
    said = str(caught.value)
    assert "embeddings" in said and "pooled" in said
    assert "no `logits`" in said


def test_three_dimensional_logits_without_boxes_are_refused_not_reduced():
    """Choosing an axis here would pick what the prediction is about by
    accident, and the picture would look the same either way."""
    with pytest.raises(cv.NotMeasurable) as caught:
        cv.task_of(Out(logits=torch.zeros(1, 5, 7)))
    assert "(1, 5, 7)" in str(caught.value)


def test_a_promptable_segmenter_is_refused_rather_than_run_on_no_prompt(corners):
    """It segments what you point at; with no prompt there is no prediction,
    and inventing one would be inventing the answer."""
    with pytest.raises(cv.NotMeasurable) as caught:
        cv.predict(_eval(PromptedNet()), corners)
    assert "no point, box or text prompt" in str(caught.value)


# --------------------------------------------------------- the prediction


def test_a_classifier_reports_top_k_with_the_checkpoints_own_names(corners):
    model = _eval(TwoRegionNet())
    found = cv.predict(model, corners, top_k=2, model_name="two-region").to_dict()

    assert found["task"] == "classification"
    assert found["labels_read"] is True
    assert [row["label"] for row in found["classes_top"]] == ["corner 0", "corner 1"]
    # Ground truth: the top-left is 0.9 and the bottom-right 0.6, times ten.
    assert found["classes_top"][0]["logit"] == pytest.approx(9.0, abs=1e-4)
    assert found["classes_top"][1]["logit"] == pytest.approx(6.0, abs=1e-4)
    probabilities = [row["probability"] for row in found["classes_top"]]
    assert probabilities[0] + probabilities[1] == pytest.approx(1.0, abs=1e-6)
    assert "SOFTMAX CONFIDENCE IS NOT THE PROBABILITY OF BEING RIGHT" in found["means"]


def test_a_multi_label_head_is_scored_by_sigmoid_because_it_says_it_is(corners):
    """A softmax over a multi-label head produces confident-looking numbers
    that sum to one across classes the model scores independently."""
    found = cv.predict(_eval(MultiLabelNet()), corners, top_k=3).to_dict()

    assert found["scoring"] == "sigmoid"
    total = sum(row["probability"] for row in found["classes_top"])
    assert total > 1.0
    assert "do not sum to one" in found["means"]


def test_a_regression_head_reports_no_probability_at_all(corners):
    """`None` and `0.0` are different answers, and a softmax over a quantity
    with no classes is a confident number about nothing."""
    found = cv.predict(_eval(RegressionNet()), corners).to_dict()

    assert found["scoring"] == "raw"
    assert found["classes_top"][0]["probability"] is None
    assert "REGRESSION" in found["means"]


def test_top_k_as_a_boolean_is_refused_rather_than_becoming_one(corners):
    """`isinstance(True, int)` is True in Python, so this would quietly become
    a one-row prediction that looks like somebody asked for it."""
    with pytest.raises(BadRequest) as caught:
        cv.predict(_eval(TwoRegionNet()), corners, top_k=True)
    assert "isinstance(True, int)" in str(caught.value)


def test_a_model_in_training_mode_is_refused_rather_than_switched(corners):
    """Dropout makes the same image two different answers, and flipping
    somebody's model as a side effect of measuring it is not this module's
    business."""
    model = TwoRegionNet()
    model.train()
    with pytest.raises(cv.NotMeasurable) as caught:
        cv.predict(model, corners)
    assert "model.eval()" in str(caught.value)
    assert "will not do it for you" in str(caught.value)


# ----------------------------------------------------------- the detector


def test_the_no_object_column_is_found_from_the_head_width_not_a_model_list(
    corners,
):
    """A head one column wider than its label list keeps that column for "no
    object" — read from this checkpoint's own widths, because a list of model
    names stops working the week after it is written."""
    found = cv.predict(_eval(TinyDetector()), corners, top_k=3).to_dict()

    assert found["scoring"] == "softmax_no_object"
    assert "4 columns for 3 labels" in found["scoring_reason"]
    # The quiet slot votes for the no-object column, and that must never be
    # reported as a detection of a class.
    assert all(box["index"] < 3 for box in found["boxes"])


def test_a_head_as_wide_as_its_labels_is_read_as_sigmoid_scored(corners):
    model = _eval(TinyDetector(columns=3, labels=3))
    found = cv.predict(model, corners, top_k=3).to_dict()

    assert found["scoring"] == "sigmoid"
    assert "do not sum to one and are not meant to" in found["scoring_reason"]


def test_without_labels_a_detector_reports_raw_logits_and_refuses_to_guess(
    corners,
):
    """The two conventions are different arithmetic, and a probability
    computed under the wrong one is a confident number about the wrong sum."""
    model = _eval(TinyDetector())
    del model.config.id2label
    found = cv.predict(model, corners, top_k=2).to_dict()

    assert found["scoring"] == "unknown"
    assert "RAW LOGITS ARE REPORTED RATHER THAN SCORES" in found["scoring_reason"]


def test_the_top_box_is_the_one_the_image_actually_supports(corners):
    """Slot 0 reads the bright top-left and slot 1 the dimmer bottom-right, so
    slot 0 has to come first or the ranking is not reading the image."""
    found = cv.predict(_eval(TinyDetector()), corners, top_k=3).to_dict()

    assert found["boxes"][0]["query"] == 0
    assert found["boxes"][0]["index"] == 0
    assert found["boxes"][0]["label"] == "object 0"
    assert found["boxes"][1]["query"] == 1


def test_boxes_carry_both_the_heads_own_convention_and_this_tensors_pixels(
    corners,
):
    """`pred_boxes` is centre-x, centre-y, width, height normalised, and the
    corners beside it are that convention applied to this tensor — labelled,
    because the convention is transformers' rather than this checkpoint's."""
    found = cv.predict(_eval(TinyDetector()), corners, top_k=1).to_dict()
    box = found["boxes"][0]

    assert box["box_cxcywh"] == [0.25, 0.25, 0.5, 0.5]
    # 32-pixel tensor: centre 0.25 with width 0.5 spans -0.25 to 0.5.
    assert box["box_xyxy"] == [0.0, 0.0, 16.0, 16.0]
    assert "transformers convention" in found["scoring_reason"]


def test_the_queries_a_detector_did_not_list_are_counted(corners):
    """A list capped at k that says nothing is a list claiming to be
    complete."""
    found = cv.predict(_eval(TinyDetector()), corners, top_k=1).to_dict()

    assert found["queries_total"] == 3
    assert len(found["boxes"]) == 1
    assert "3 box queries" in found["means"]
    assert "A query is a SLOT, not an object" in found["means"]


# --------------------------------------------------------- the segmenters


def test_a_per_pixel_head_reports_its_own_grid_and_says_it_is_coarser(corners):
    found = cv.predict(_eval(TinySegmenter()), corners).to_dict()

    assert found["map_height"] == 4 and found["map_width"] == 4
    assert found["map_stride"] == 1
    assert len(found["label_map"]) == 4
    # Left half is class 1, right half class 0, so both are present and the
    # areas are equal.
    by_index = {segment["index"]: segment for segment in found["segments"]}
    assert set(by_index) == {0, 1}
    assert by_index[1]["cells"] == 8 and by_index[0]["cells"] == 8
    assert by_index[1]["bbox"] == [0, 0, 4, 2]
    assert "its own internal resolution, NOT the 32x32" in found["means"]


def test_a_per_pixel_margin_is_the_gap_to_the_runner_up_and_says_so(corners):
    """The same field means something else on a mask head, so the quantity
    travels with the number rather than being assumed from its name."""
    found = cv.predict(_eval(TinySegmenter()), corners).to_dict()

    assert (
        "gap in logits between the winning class and the runner-up"
        in (found["margin_kind"])
    )
    by_index = {segment["index"]: segment for segment in found["segments"]}
    assert by_index[0]["mean_margin"] == pytest.approx(5.0, abs=1e-4)


def test_a_mask_head_leaves_unclaimed_cells_unclaimed(corners):
    """ "Nothing was segmented here" is an answer; filling it in with the
    least-bad query would draw a segment nobody predicted."""
    found = cv.predict(_eval(TinyMaskHead()), corners).to_dict()

    flat = [cell for row in found["label_map"] for cell in row]
    assert -1 in flat
    unclaimed = next(s for s in found["segments"] if s["index"] == -1)
    # The bottom half minus the one corner query 1 claims.
    assert unclaimed["cells"] == 7
    assert unclaimed["label"] == ""


def test_a_mask_heads_margin_is_named_as_a_different_quantity(corners):
    found = cv.predict(_eval(TinyMaskHead()), corners).to_dict()

    assert "ABOVE the 0.5 threshold" in found["margin_kind"]
    assert "NOT a gap between classes" in found["margin_kind"]
    assert found["mask_threshold"] == 0.5
    assert "a threshold somebody chose" in found["means"]


def test_a_mask_threshold_outside_zero_to_one_is_refused(corners):
    with pytest.raises(BadRequest) as caught:
        cv.predict(_eval(TinyMaskHead()), corners, mask_threshold=1.0)
    assert "strictly between 0 and 1" in str(caught.value)


def test_a_map_too_large_to_carry_is_subsampled_and_the_stride_is_reported(
    corners,
):
    """A silently thinned map is a map claiming a resolution it does not
    have, and every boundary in it would be drawn too coarse with nothing
    saying so."""
    side = 400  # 160,000 cells, past the 65,536 this carries
    found = cv.predict(_eval(TinySegmenter(grid=side)), corners).to_dict()

    assert found["map_height"] == side and found["map_width"] == side
    assert found["map_stride"] > 1
    assert len(found["label_map"]) < side
    assert "SUBSAMPLED" in found["means"]
    # The counts are from the FULL map, not the thinned one.
    assert sum(segment["cells"] for segment in found["segments"]) == side * side


# ------------------------------------------------------------ the readout


def test_a_transformer_gets_per_layer_attention_on_its_own_patch_grid(corners):
    model = _eval(TinyViT())
    found = cv.layer_readout(model, corners, model_name="tiny-vit").to_dict()

    assert found["kind"] == "attention"
    assert found["n_layers"] == 3
    assert found["tokens"] == 5  # one class token plus a 2x2 patch grid
    assert found["prefix_tokens"] == 1
    assert (found["grid_rows"], found["grid_cols"]) == (2, 2)
    assert found["heads"] == 2
    assert found["forward_passes"] == 2
    assert "ATTENTION IS NOT A CAUSE" in found["means"]


def test_the_attention_a_layer_spends_off_the_grid_is_reported_not_dropped(
    corners,
):
    """The class token attending to itself is real mass excluded from the map,
    and a map that quietly renormalised it away would overstate every patch."""
    model = _eval(TinyViT())
    found = cv.layer_readout(model, corners).to_dict()

    layer = found["layers"][0]
    total = sum(v for row in layer["values"] for v in row) + layer["off_grid_mass"]
    assert total == pytest.approx(1.0, abs=1e-4)
    assert "reported rather than normalised away" in found["means"]


def test_the_spread_across_heads_travels_with_the_mean_over_them(corners):
    """A mean over heads hides head-level disagreement, so the size of what it
    hides is on every layer."""
    found = cv.layer_readout(_eval(TinyViT()), corners).to_dict()

    assert all(layer["head_disagreement"] is not None for layer in found["layers"])
    assert "hides head-level disagreement" in found["means"]


def test_a_convolutional_stack_gets_feature_maps_named_as_not_attention(corners):
    """A CNN has no attention. Producing something shaped like an attention
    map for one would be a picture of something that does not exist."""
    found = cv.layer_readout(_eval(TinyCNN()), corners).to_dict()

    assert found["kind"] == "feature_map"
    assert found["n_layers"] == 2
    assert (found["layers"][0]["rows"], found["layers"][0]["cols"]) == (16, 16)
    assert "THIS IS NOT ATTENTION" in found["means"]
    assert "convolutional stack" in found["reason"]


def test_an_empty_attention_capture_is_a_fact_about_the_kernel_not_the_model(
    corners,
):
    """Measured on transformers 5.14.1: an SDPA model asked for attentions
    returns an EMPTY TUPLE and warns to a log. Reporting that as zero layers
    of attention states a fact about the kernel as a fact about the
    architecture."""
    found = cv.layer_readout(_eval(SdpaViT()), corners).to_dict()

    assert found["kind"] != "attention"
    assert "never materialises the probability matrix" in found["reason"]
    assert "NOT about the architecture" in found["reason"]
    assert found["n_layers"] == 4  # the activations are still there


def test_a_model_with_nothing_to_look_inside_says_so_rather_than_drawing_blank(
    corners,
):
    """A blank heatmap and a model with nothing to read look identical, and
    only one of them is a measurement."""
    found = cv.layer_readout(_eval(SilentNet()), corners).to_dict()

    assert found["kind"] == "none"
    assert found["layers"] == []
    assert "no hidden states" in found["reason"]
    assert "NO PER-LAYER READOUT WAS PRODUCED" in found["means"]


def test_an_unaffordable_attention_capture_is_refused_before_it_allocates(
    corners, monkeypatch
):
    """Finding out by waiting for an out-of-memory is the failure the ceiling
    exists to prevent, and the reason names the number that got you refused."""
    monkeypatch.setattr(cv, "MAX_ATTENTION_BYTES", 16)
    found = cv.layer_readout(_eval(TinyViT()), corners).to_dict()

    assert found["kind"] != "attention"
    assert "past the" in found["reason"]
    assert found["forward_passes"] == 1  # the second pass never happened
    assert found["attention_bytes"] == 3 * 2 * 5 * 5 * 4


def test_a_head_count_that_cannot_be_read_stops_the_capture_being_priced(
    corners,
):
    """An unpriced capture is an out-of-memory in the middle of a measurement
    rather than a refusal before one."""
    model = _eval(TinyViT())
    del model.config.num_attention_heads
    found = cv.layer_readout(model, corners).to_dict()

    assert found["kind"] != "attention"
    assert found["attention_bytes"] is None
    assert "could not be priced" in found["reason"]


# --------------------------------------------------------- the attribution


def test_attribution_defaults_to_the_models_own_answer_and_says_the_model_chose(
    corners,
):
    """Class 0 reads only the top-left quadrant, so a sweep that scores
    anything else strongest has invented it."""
    found = cv.attribute(
        _eval(TwoRegionNet()),
        corners,
        patch=16,
        stride=16,
        value_range=UNIT,
        model_name="two-region",
    ).to_dict()

    assert found["region_chosen_by"] == "model"
    assert found["target_label"] == "corner 0"
    strongest = found["attribution"]["strongest"]
    assert (strongest["row"], strongest["col"]) == (0, 0)
    # Ground truth: the quadrant means 0.9 and grey is 0.5, times ten.
    assert strongest["logit_drop"] == pytest.approx(4.0, abs=1e-3)
    assert "the model's own strongest answer" in found["means"]


def test_naming_another_class_attributes_that_class_and_not_the_argmax(corners):
    """ "Why did it pick that" and "what supports this other class" are
    different questions with the same picture, and a module that always drew
    the argmax would answer the first while claiming the second."""
    found = cv.attribute(
        _eval(TwoRegionNet()),
        corners,
        target=1,
        patch=16,
        stride=16,
        value_range=UNIT,
    ).to_dict()

    assert found["region_chosen_by"] == "caller"
    assert found["target_label"] == "corner 1"
    strongest = found["attribution"]["strongest"]
    assert (strongest["row"], strongest["col"]) == (1, 1)
    assert strongest["logit_drop"] == pytest.approx(1.0, abs=1e-3)
    assert "You named it" in found["means"]


def test_a_detector_is_attributed_over_a_query_slot_and_the_caveat_travels(
    corners,
):
    """Slot 0 reads the top-left quadrant. The slot is fixed across the sweep;
    the box it draws is not, and that is the whole caveat of the measurement."""
    found = cv.attribute(
        _eval(TinyDetector()), corners, patch=16, stride=16, value_range=UNIT
    ).to_dict()

    assert found["task"] == "detection"
    assert found["query"] == 0
    assert found["region_chosen_by"] == "model"
    strongest = found["attribution"]["strongest"]
    assert (strongest["row"], strongest["col"]) == (0, 0)
    assert "A QUERY SLOT IS FIXED, THE BOX IT DRAWS IS NOT" in found["means"]


def test_naming_a_different_query_moves_the_map_to_that_slot(corners):
    found = cv.attribute(
        _eval(TinyDetector()),
        corners,
        query=1,
        patch=16,
        stride=16,
        value_range=UNIT,
    ).to_dict()

    assert found["query"] == 1
    strongest = found["attribution"]["strongest"]
    assert (strongest["row"], strongest["col"]) == (1, 1)


def test_a_segmenter_is_attributed_over_a_region_of_its_own_output_grid(corners):
    """The occluder works in image pixels and the region is in map cells; a
    region read as the wrong one measures somewhere else entirely."""
    found = cv.attribute(
        _eval(TinySegmenter()),
        corners,
        region=(0, 0, 4, 2),
        target=1,
        patch=16,
        stride=16,
        value_range=UNIT,
    ).to_dict()

    assert found["task"] == "semantic_segmentation"
    assert found["region"] == [0, 0, 4, 2]
    assert found["map_height"] == 4 and found["map_width"] == 4
    assert "output grid" in found["means"]
    # The left half of the image drives class 1 there, so the left windows
    # move it and the right ones do not.
    table = found["attribution"]["map"]
    assert table[0][0] > table[0][1]


def test_a_region_outside_the_models_grid_is_refused_naming_that_grid(corners):
    with pytest.raises(BadRequest) as caught:
        cv.attribute(
            _eval(TinySegmenter()), corners, region=(0, 0, 40, 40), patch=16, stride=16
        )
    said = str(caught.value)
    assert "4x4 output grid" in said
    assert "not the image's" in said


def test_a_region_that_is_not_four_numbers_is_refused(corners):
    with pytest.raises(BadRequest) as caught:
        cv.attribute(_eval(TinySegmenter()), corners, region=(0, 0), patch=16)
    assert "four whole numbers" in str(caught.value)


def test_a_classifier_has_no_query_slots_and_says_so(corners):
    """Accepting the parameter and ignoring it would read as a promise."""
    with pytest.raises(BadRequest) as caught:
        cv.attribute(_eval(TwoRegionNet()), corners, query=2, patch=16)
    assert "no box queries and no mask regions" in str(caught.value)


def test_a_per_pixel_segmenter_has_no_query_slots_either(corners):
    with pytest.raises(BadRequest) as caught:
        cv.attribute(_eval(TinySegmenter()), corners, query=1, patch=16)
    assert "no query slots, only a grid" in str(caught.value)


def test_the_occlusion_sweeps_own_caveats_survive_unchanged(corners):
    """There is one occluder in this project and this is not a second: its
    fill caveat and its resolution limit have to arrive intact."""
    found = cv.attribute(
        _eval(TwoRegionNet()), corners, patch=16, stride=16, value_range=UNIT
    ).to_dict()

    inner = found["attribution"]["means"]
    assert "OCCLUSION IS OUT OF DISTRIBUTION" in inner
    assert "ONE CELL PER WINDOW" in inner
    assert "the occlusion sweep's own report, unchanged" in found["means"]


def test_a_promptable_segmenter_cannot_be_attributed_either(corners):
    with pytest.raises(cv.NotMeasurable) as caught:
        cv.attribute(_eval(PromptedNet()), corners, patch=16)
    assert "explaining an answer this tool invented" in str(caught.value)


# ---------------------------------------------------------------- the cost


def test_the_preflight_prices_all_three_measurements_without_a_model():
    found = cv.plan(224, 224, layers=12, heads=12, tokens=196, patch=16, batch=32)

    assert found["predict"]["forward_passes"] == 1
    assert found["readout"]["forward_passes"] == 2
    assert found["readout"]["attention_bytes"] == 12 * 12 * 196 * 196 * 4
    assert found["attribution"]["passes"] == 197
    assert found["total_forward_passes"] == 1 + 2 + 197


def test_an_unpriceable_attention_capture_is_unknown_and_never_zero():
    """A capture whose memory could not be computed is not a capture that
    costs nothing."""
    found = cv.plan(224, 224, patch=16)

    assert found["readout"]["attention_bytes"] is None
    assert found["readout"]["forward_passes"] == 1
    assert "UNKNOWN rather than zero" in found["means"]


def test_the_token_count_is_stated_as_a_floor_because_prefixes_are_not_in_it():
    """`readout_shape_of` counts patches, and a class token is not a patch —
    a memory figure quoted as exact and short by two tokens squared is worse
    than one labelled as a floor."""
    model = _eval(TinyViT())
    shape = cv.readout_shape_of(model)
    assert shape == {"layers": 3, "heads": 2, "tokens": 4}

    found = cv.plan(32, 32, patch=16, **shape)
    assert "floor rather than an exact size" in found["means"]


def test_the_preflight_never_refuses_the_run_it_is_pricing():
    """A caller about to be refused needs the number that got them refused."""
    found = cv.plan(224, 224, patch=16, stride=1)

    assert found["attribution"]["within_ceiling"] is False
    assert "PAST THE CEILING" in found["attribution"]["means"]


def test_no_seconds_are_invented_when_nobody_measured_this_machine():
    found = cv.plan(224, 224, patch=16)

    assert found["seconds"] is None
    assert "an invented one would be a number this tool made up" in found["means"]


def test_the_sweep_is_priced_by_the_sweeps_own_estimator_not_a_second_one():
    """Two functions answering "how many forward passes" would be free to
    disagree, and this panel and the attribution panel would then price the
    same sweep differently."""
    from modelmri import vision_attr

    mine = cv.plan(224, 224, patch=16, stride=8, batch=8)["attribution"]
    theirs = vision_attr.estimate(224, 224, patch=16, stride=8, batch=8)
    assert mine == theirs


def test_both_occlusion_routes_ask_the_processor_for_their_grey():
    """MEASURED on google/vit-base-patch16-224, one 224x224 picture, patch and
    stride 112, target 902 "whistle", base logit 4.625 on both routes:

        /api/image/cv/attribute  fill 0.019608, range [-0.686, 0.725], INFERRED
        /api/image/attribution   fill 0.0,      range [-1.0, 1.0], from the processor

    and cell [0][1] of the map came back -0.21875 against -0.25. Same model,
    same picture, same target — two different greys and two different answers.

    `/api/image/attribution` already carried the reasoning in a comment: a
    range inferred from one photograph is a fact about that photograph, and a
    picture of a bright sky never reaches the bottom of the model's input
    range, so its "grey" is not the model's neutral. `cv/attribute` had the
    same processor in hand two lines above and never asked it.

    Asserted on the SIGNATURE and the wiring rather than by running a model,
    because the comparison needs a real classifier resident; the end-to-end
    equality was verified live before this was written.
    """
    import inspect

    from modelmri import image_cv, server

    # The parameter has always existed on the function.
    assert "value_range" in inspect.signature(image_cv.attribute).parameters

    # And the route passes it now. A source check, deliberately: the defect
    # was an argument that was not passed, which no unit test of either side
    # alone can see.
    source = inspect.getsource(server.create_app)
    start = source.index('@app.post("/api/image/cv/attribute")')
    route = source[start : source.index("@app.", start + 10)]
    assert "value_range=image_input.value_range_of(processor)" in route


# ------------------------------------- a processor that re-ranks its output


class _ReRankingProcessor:
    """A post-processor of the RT-DETR family.

    It flattens the (query x class) grid, sorts it descending, and hands back
    the top `queries` of it — so its rows are ranked, not query-aligned. Real
    ones do this: `PekingU/rtdetr_r50vd`, and `conditional_detr` and
    `deformable_detr` at exactly 100 queries.
    """

    def post_process_object_detection(self, output, threshold=0.0, target_sizes=None):
        logits = output.logits[0]
        flat = logits.sigmoid().flatten()
        ranked, _ = torch.sort(flat, descending=True)
        n = int(logits.shape[0])
        return [{"scores": ranked[:n], "labels": torch.zeros(n), "boxes": None}]


class _AlignedProcessor:
    """One that really does answer per query, in QUERY order.

    Reversed deliberately. Query order is arbitrary — a detector's later slots
    are as likely to be the confident ones as its earlier slots — and this
    fixture has only three queries, where the odds of landing in descending
    order by chance are one in six. A real head has 100 to 300, which is what
    makes the sortedness test safe in production and unusable at this size.
    """

    def post_process_object_detection(self, output, threshold=0.0, target_sizes=None):
        logits = output.logits[0]
        per_query = logits.softmax(dim=-1).max(dim=-1).values
        return [
            {
                "scores": per_query.flip(0),
                "labels": logits.argmax(dim=-1).flip(0),
                "boxes": None,
            }
        ]


def test_a_re_ranking_processor_is_not_read_as_query_aligned(corners):
    """LENGTH IS NOT ALIGNMENT, and the guard was `len(scores) == queries`.

    MEASURED on `PekingU/rtdetr_r50vd`: background slots with logit around -11
    were reported as 99% detections carrying their own boxes, the three real
    detections never appeared, and every row contradicted itself — `score`
    from the sorted tensor beside a `logit` that is query-aligned.
    """
    found = cv.predict(
        _eval(TinyDetector()), corners, top_k=3, processor=_ReRankingProcessor()
    ).to_dict()

    assert found["scoring"] != "checkpoint_post_processor", (
        "a sorted result cannot be matched back to the queries the boxes come from"
    )
    assert "re-ranks" in found["scoring_reason"]

    # And every row stays about ONE query: the score and the logit agree in
    # ordering, which is what was contradicting itself before.
    boxes = found["boxes"]
    ranked_by_score = [b["query"] for b in sorted(boxes, key=lambda b: -b["score"])]
    ranked_by_logit = [b["query"] for b in sorted(boxes, key=lambda b: -b["logit"])]
    assert ranked_by_score == ranked_by_logit


def test_a_genuinely_query_aligned_processor_is_still_preferred(corners):
    """The guard must not throw away the checkpoint's own answer — reading
    beats inferring, which is the rule the rest of this module follows."""
    found = cv.predict(
        _eval(TinyDetector()), corners, top_k=3, processor=_AlignedProcessor()
    ).to_dict()

    assert found["scoring"] == "checkpoint_post_processor"
    assert "OWN" in found["scoring_reason"]


def test_running_out_of_memory_is_not_reported_as_needing_a_prompt():
    """`_run`'s bare `except Exception` routed everything to one sentence, so a
    CUDA out-of-memory told the reader to go looking for a prompt API on a
    model that has none — on the 8 GB card this project targets, which is
    exactly where that happens."""

    class _OutOfMemoryError(Exception):
        pass

    _OutOfMemoryError.__name__ = "OutOfMemoryError"

    said = cv._forward_refusal(
        _eval(TwoRegionNet()), torch.zeros(1, 3, 8, 8), _OutOfMemoryError("cuda")
    ).sentence

    assert "ran out of memory" in said
    assert "prompt" not in said
    assert "Unload" in said, "and say what to do about it"


def test_a_missing_package_is_not_reported_as_needing_a_prompt():
    err = ImportError("No module named 'timm'")
    err.name = "timm"

    said = cv._forward_refusal(
        _eval(TwoRegionNet()), torch.zeros(1, 3, 8, 8), err
    ).sentence

    assert "pip install timm" in said
    assert "weights are fine" in said


def test_an_unrecognised_failure_hedges_rather_than_asserting():
    """The residual arm is a guess and now says so — it used to assert a
    prompt-shaped cause for every exception a third-party forward can raise."""
    said = cv._forward_refusal(
        _eval(TwoRegionNet()), torch.zeros(1, 3, 8, 8), RuntimeError("something")
    ).sentence

    assert "MAY be" in said
    assert "modelmri serve" in said, "and point at where the real cause is"


def test_a_query_past_the_end_of_the_head_is_refused_by_name(corners):
    """`CVAttributeRequest` validates `ge=0` and `_as_int` is a type check, so
    `query=999` reached `logits[:, 999, :]` and raised IndexError on the very
    first, unoccluded pass — neither Refusal nor BadRequest, so a 500. Passing
    `query=` also skips `predict()`, so nothing upstream had looked at the
    shape."""
    with pytest.raises(BadRequest) as caught:
        cv.attribute(_eval(TinyDetector()), corners, query=999, patch=8, stride=8)

    said = caught.value.sentence
    assert "999" in said
    assert "numbered 0 to" in said, "name the range that does exist"


def test_a_query_inside_the_head_still_runs(corners):
    """So the bound cannot become "refuse every named query"."""
    got = cv.attribute(_eval(TinyDetector()), corners, query=1, patch=8, stride=8)

    assert got.to_dict()["region_chosen_by"] == "caller"
