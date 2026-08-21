"""What a computer-vision model actually answered, and what supported it.

`imaging.py` has been able to NAME a classifier, a detector and a segmenter
since the image side existed, and `image_catalog` lists the ones on this disk.
Nothing could measure one. Loading `google/vit-base-patch16-224` gave you a
model and no instrument — the diffusion panels do not apply (there is no
prompt and no denoising schedule) and the VLM panels do not apply (there is no
text to attend to). This is the third family's instrument.

Three questions, in the order somebody actually asks them:

  `predict`        what did it say — classes, boxes or masks, with the
                   checkpoint's OWN label names
  `layer_readout`  what did each layer look at — attention where the
                   architecture has it, feature maps where it does not
  `attribute`      what in the picture holds that answer up — the occlusion
                   sweep, run over the prediction rather than over the argmax

## The task is read off the OUTPUT, never off a name

`vla.py` was corrected once for hardcoding three SmolVLA values and `imaging`
carries the rule: nothing here is allowed to know a model name. So `predict`
runs the model once and dispatches on the SHAPE of what came back —
`pred_boxes` beside `logits` is a detector, `masks_queries_logits` is a
mask-query segmenter, four-dimensional `logits` are a per-pixel head, and
two-dimensional ones are a classifier. A checkpoint released next week that
follows those conventions works here without this file being edited, and one
that does not is refused BY NAME of what it returned rather than silently read
as whichever branch came first.

## The label names come from the checkpoint or not at all

`id2label` is the model's own answer to "what is class 285". A checkpoint that
publishes none gets integers and a sentence saying so. Borrowing ImageNet's
thousand names for a head that happens to be a thousand wide is how a fine-tune
on a thousand medical classes comes back saying "Egyptian cat", and it would
look exactly as authoritative as the true answer.

## Attribution is over the PREDICTION, and the prediction is a choice

"Why did it pick that" and "what supports this other class" are different
questions with the same picture, so `attribute` takes the thing being explained
explicitly: a class for a classifier, a query slot for a detector, a region of
the mask grid for a segmenter. Where the caller names nothing, the model's own
top answer is used AND the result says the model chose it — `vision_attr`
already carries that distinction and this does not add a second one.

The occluder itself is `vision_attr.sweep`, unmodified. There is exactly one
occluder in this project and this is not a second: what this module adds is the
`forward=` reduction that turns a detector's `[batch, queries, classes]` or a
segmenter's `[batch, classes, h, w]` into the `[batch, classes]` the sweep
measures. That reduction is the whole content of "attribute over a box" — the
rest is the same measurement, with the same refusals and the same caveats.

## The dtype is the model's, and it is reported

A checkpoint loaded in bfloat16 — which is what this project's own loader picks
on an accelerator that prefers it — cannot be fed a float32 tensor at all: the
matrix multiply refuses. So the cast happens at the forward boundary, the
occlusion arithmetic stays in float32 where the fill value is exact, and the
dtype travels in every result. It has to: bfloat16 carries about three decimal
digits, and a score printed at six is then four digits of measurement and two
of format.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from . import fmt, vision_attr
from .errors import BadRequest, Refusal
from .vision_attr import SCORE_DECIMALS

log = logging.getLogger(__name__)

# Probabilities are rounded finer than logits because they live in [0, 1] and
# a small class in a thousand-wide head is genuinely 1e-7. The same two
# precisions `vision_attr` reports its drops at, for the same reason.
PROB_DECIMALS = 8

# What a prediction will list. A thousand-class head has a thousand answers and
# 995 of them are noise; the cap is on the REPORT and the count of what was cut
# travels with it.
DEFAULT_TOP_K = 5
MAX_TOP_K = 100

# The most mask segments or box queries a prediction will enumerate. A
# panoptic head has a hundred queries and a semantic one can find fifty labels
# in one photograph; both lists are truncated by size and both say so.
MAX_SEGMENTS = 64

# The most cells a returned label map will carry. A segmentation head emits its
# map at its own internal resolution — often a quarter of the image on each
# axis — and 512x512 of those is a quarter of a million integers in a JSON
# body. Past this the map is SUBSAMPLED and the stride is reported, because a
# silently thinned map is a map claiming a resolution it does not have.
MAX_MAP_CELLS = 65_536

# The most memory an attention capture will be allowed to hold. Every layer's
# full [heads, tokens, tokens] matrix is resident at once when
# `output_attentions=True`, which is 22 MB for a 12-layer ViT at 197 tokens and
# 2.7 GB for a 40-layer model at 1025. The ceiling is memory rather than taste,
# and the refusal names the number that got you refused.
MAX_ATTENTION_BYTES = 2 * 1024 * 1024 * 1024

# Where a mask query's logit becomes "inside the mask". 0.5 on a sigmoid is the
# convention every mask head is trained against, and it is still a threshold
# somebody chose — so it is a parameter with this default, and the value used
# is on every result.
DEFAULT_MASK_THRESHOLD = 0.5

# The task kinds, decided by what the model RETURNED.
CLASSIFY = "classification"
DETECT = "detection"
SEMANTIC = "semantic_segmentation"
MASK_QUERIES = "mask_segmentation"
PROMPTED = "prompted_segmentation"

_TASK_LABEL = {
    CLASSIFY: "a classifier",
    DETECT: "an object detector",
    SEMANTIC: "a per-pixel segmenter",
    MASK_QUERIES: "a mask-query segmenter",
    PROMPTED: "a promptable segmenter",
}


class NotMeasurable(Refusal):
    """This model cannot answer this question, and the sentence says why.

    Its own class for the reason `vision_attr.NotAttributable` has one: the
    caller's fix is neither "fix a parameter" nor "load something else". A
    promptable segmenter with no prompt, a head whose output this cannot read,
    an architecture with no attention to capture — each needs a different next
    move and each gets a different sentence.
    """


# ------------------------------------------------------------------- naming


def label_names(model) -> list[str] | None:
    """The head's own class names in class order, or `None`.

    `id2label` is a dict keyed by index, and transformers hands it back with
    STRING keys after a JSON round-trip — so the order is restored by sorting
    on the INTEGER, not by trusting insertion order and not by sorting the keys
    as text, which puts "10" immediately after "1". A list built in the wrong
    order puts every name against the wrong class and looks entirely reasonable
    doing it.

    `None` rather than `["class 0", "class 1", ...]`: `vision_attr` drops a
    name list whose length does not match the head rather than applying it to
    the wrong classes, and invented names would defeat that check as well as
    being invented.
    """
    config = getattr(model, "config", None)
    table = getattr(config, "id2label", None)
    if not isinstance(table, dict) or not table:
        # Some heads publish it one level down, on the vision half of a
        # composite config. Read rather than assumed absent.
        table = getattr(getattr(config, "vision_config", None), "id2label", None)
    if not isinstance(table, dict) or not table:
        return None
    try:
        return [str(table[k]) for k in sorted(table, key=int)]
    except (TypeError, ValueError):
        # Keys that are not indices at all. Unknown, said as unknown.
        return None


def _named(names: list[str] | None, index: int) -> str:
    """One class's name, or `""` when the checkpoint published none.

    `""` and not `f"class {index}"`: the caller decides how to print an
    unnamed class, and every `means()` here says the names were absent rather
    than quietly printing a number that looks like a name.
    """
    if not names or not 0 <= index < len(names):
        return ""
    return str(names[index])


def _naming_note(names: list[str] | None, classes: int) -> str:
    """The sentence about where the class names came from, or did not."""
    if names is None:
        return (
            f"This checkpoint publishes no `id2label`, so its {classes} classes "
            f"are reported as INDICES. Nothing here will borrow another "
            f"model's class list to fill them in — a thousand-wide head is not "
            f"evidence of ImageNet, and a fine-tune answering 'Egyptian cat' "
            f"about a chest x-ray would look exactly as authoritative as the "
            f"truth."
        )
    if len(names) != classes:
        return (
            f"This checkpoint's `id2label` has {len(names)} entries and its "
            f"head has {classes} outputs, so the names are NOT applied: a list "
            f"of the wrong length mislabels at least one class, and a wrong "
            f"name is worse than an index."
        )
    return (
        f"Class names are this checkpoint's own `id2label`, all {len(names)} of "
        f"them, in class order."
    )


# ------------------------------------------------------- running the model


def _model_dtype(model):
    """The dtype the weights are actually held in, or `None`.

    Read from a parameter rather than from `config.torch_dtype`, which records
    what was requested at save time and not what this process loaded.
    """
    try:
        for parameter in model.parameters():
            return parameter.dtype
    except (AttributeError, TypeError):
        return None
    return None


def _model_device(model):
    """The device the first parameter sits on, or `None` for a model with none."""
    try:
        for parameter in model.parameters():
            return parameter.device
    except (AttributeError, TypeError):
        return None
    return None


def _cast(image, model):
    """The image in the model's own dtype. The DEVICE is not touched.

    Not a convenience. A bfloat16 model — which is what this project's loader
    picks on an accelerator that prefers it — cannot multiply a float32 tensor
    at all; the operation refuses rather than promoting. So the cast happens
    HERE, at the forward boundary, and never on the tensor the occluder does
    its arithmetic on: the fill value stays exact in float32 and only the copy
    the model sees is narrowed.

    The narrowing is real and is reported on every result. bfloat16 carries
    about three decimal digits, so a logit printed at six is four digits of
    measurement and two of formatting.

    The device is deliberately NOT fixed up the same way, and the asymmetry is
    the point. A dtype mismatch has no cheap alternative — the model simply
    cannot be run otherwise — while a device mismatch does: build the tensor
    on the right device once. Doing it here instead would copy the whole batch
    across the bus on every one of the sweep's forward calls, which is most of
    the run time, and nothing in the result would say so. `_require_device`
    refuses it up front instead.
    """
    dtype = _model_dtype(model)
    if dtype is not None and image.dtype != dtype and image.is_floating_point():
        return image.to(dtype)
    return image


def _require_device(model, image) -> None:
    """Refuse a tensor built on a different device from the model's.

    Up front rather than at the first forward, so a sweep that would have
    copied 196 occluded images across the bus is refused before the first one.
    `image_input.to_tensor` takes the device for exactly this reason.
    """
    device = _model_device(model)
    if device is not None and image.device != device:
        raise NotMeasurable(
            f"this model is on {device} and the image tensor is on "
            f"{image.device}. Build the tensor on the model's device — "
            f"`to_tensor(..., device=...)` takes it — because copying it here "
            f"instead would move the whole batch across on every forward call "
            f"of the sweep, which is most of its run time, and nothing in the "
            f"result would say so."
        )


def _run(model, image, **kwargs):
    """One forward pass, under `no_grad`, with a refusal that says what broke.

    Every failure of a third-party forward reaches a reader as the exception
    TYPE and a sentence written here. transformers puts absolute paths from
    this machine into its messages, and a device or dtype mismatch produces a
    sentence about matrix dtypes that helps nobody — so the two mismatches this
    module can detect are named specifically and everything else is named by
    class.
    """
    import torch

    with torch.no_grad():
        try:
            return model(_cast(image, model), **kwargs)
        except Exception as err:
            raise _forward_refusal(model, image, err) from None


def _forward_refusal(model, image, err: Exception) -> Refusal:
    """Why a forward pass failed, in this project's words rather than torch's."""
    device = _model_device(model)
    if device is not None and image.device != device:
        # `_require_device` catches this before any work on every entry point
        # here. It survives as a backstop for a model SHARDED across devices,
        # where the first parameter's device is not every parameter's — the
        # up-front check passes and the mismatch appears deep in the stack.
        return NotMeasurable(
            f"this model's first parameters are on {device} and the image "
            f"tensor is on {image.device}. A model spread across devices needs "
            f"its input where its first layer is."
        )
    # BRANCH ON THE CAUSE. Everything reached this one sentence, so a CUDA
    # out-of-memory told the reader to go looking for a prompt API on a model
    # that has none — on the 8 GB card this project targets, which is where
    # that happens. Reproduced with `torch.OutOfMemoryError`.
    #
    # Read by class NAME rather than by import: `torch.OutOfMemoryError` moved
    # between versions, and this module is reached without torch resident.
    if type(err).__name__ in ("OutOfMemoryError", "OutOfMemoryError_"):
        return NotMeasurable(
            f"this model ran out of memory on {device or 'the accelerator'} "
            f"during its forward pass. Nothing about the image or the "
            f"checkpoint is wrong — there is not enough room for it right now. "
            f"Unload whatever else is resident, or open this on the CPU."
        )
    if isinstance(err, ImportError):
        package = getattr(err, "name", "") or ""
        return NotMeasurable(
            "this checkpoint's forward pass needs a package that is not "
            "installed here"
            + (f" — `pip install {package}`." if package else ".")
            + " The weights are fine; nothing needs re-downloading."
        )
    return NotMeasurable(
        f"this model's forward pass refused the image ({type(err).__name__}). "
        f"This MAY be a head that needs more than pixels — a promptable "
        f"segmenter wants points or boxes, a grounding detector wants text — "
        f"which cannot be run from an image alone, and guessing a prompt would "
        f"be guessing the answer. The terminal running `modelmri serve` "
        f"carries what was actually raised."
    )


def _accepts(model, name: str) -> bool:
    """Does this model's forward take that keyword at all?

    Asked BEFORE the call rather than caught after it. `_run` turns every
    exception a third-party forward raises into one refusal, which is right —
    torch's messages carry paths and helping nobody — but it also means a
    `TypeError` for an unknown keyword arrives indistinguishable from a real
    bug inside the model. This is the question the keyword check is actually
    about, and the signature answers it without running anything.

    A forward taking `**kwargs` accepts everything, so it answers True and the
    call decides — which is the honest reading of a signature that says it
    takes whatever it is given.
    """
    import inspect

    try:
        parameters = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        # A model whose forward cannot be introspected — a compiled or wrapped
        # one. Assume it accepts and let the call answer, which is the same
        # position as `**kwargs`.
        return True
    if name in parameters:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())


def _require_eval(model, what: str) -> None:
    """Refuse a training-mode model, for the reason `vision_attr` refuses one.

    Dropout and batch-norm running statistics make the same input two different
    answers, so a prediction is a sample of noise and every occlusion score is
    that noise as much as the occlusion. Refused rather than switched: changing
    somebody's model as a side effect of measuring it is not this module's
    business, and the fix is one call they can see.
    """
    if getattr(model, "training", False) is True:
        raise NotMeasurable(
            f"this model is in training mode, where dropout and batch-norm make "
            f"the same image give a different answer each time — {what} would "
            f"be that noise as much as the model. Call `model.eval()` first. "
            f"This will not do it for you."
        )


def _tensor_of(output, name: str):
    """One named field of a model output, whatever container it came in."""
    import torch

    found = None
    if hasattr(output, name):
        found = getattr(output, name)
    elif isinstance(output, dict) and name in output:
        found = output[name]
    return found if isinstance(found, torch.Tensor) else None


def _output_keys(output) -> list[str]:
    """What the model actually returned, for a refusal that can name it."""
    if hasattr(output, "keys"):
        try:
            return sorted(str(k) for k in output.keys())
        except Exception:
            return []
    if isinstance(output, (tuple, list)):
        return [f"<{len(output)} unnamed tensors>"]
    return [f"<{type(output).__name__}>"]


def task_of(output) -> str:
    """Which of the three families this output belongs to, from its SHAPE.

    Order matters and it is the order of specificity, not of popularity. A
    promptable segmenter's `pred_masks` is checked first because such a model
    may also carry `logits`; a mask-query head is checked before a plain
    per-pixel one because it has both a class tensor and a mask tensor; boxes
    beside class logits are a detector whatever the checkpoint is called.

    Nothing here reads a model name, a `model_type` or an architecture string.
    A checkpoint published next week that follows these conventions is measured
    without this file changing, and one that does not is REFUSED by name of
    what it returned — never read as whichever branch happened to come first.
    """
    if _tensor_of(output, "pred_masks") is not None:
        return PROMPTED
    if _tensor_of(output, "masks_queries_logits") is not None:
        return MASK_QUERIES
    logits = _tensor_of(output, "logits")
    if _tensor_of(output, "pred_boxes") is not None and logits is not None:
        return DETECT
    if logits is None:
        raise NotMeasurable(
            f"this model returned {', '.join(_output_keys(output)) or 'nothing'} "
            f"and no `logits`, so there is no prediction here this can read. "
            f"Choosing one of those tensors would be choosing what the answer "
            f"is about by accident."
        )
    if logits.ndim == 4:
        return SEMANTIC
    if logits.ndim == 2:
        return CLASSIFY
    raise NotMeasurable(
        f"this model returned logits shaped {tuple(logits.shape)}, which is "
        f"neither a classifier's [batch, classes], a per-pixel head's "
        f"[batch, classes, height, width], nor a detector's boxes-and-classes "
        f"pair. Reducing an axis here would pick what the prediction is about "
        f"by accident."
    )


# ------------------------------------------------------------- predictions


@dataclass
class ClassScore:
    """One class, its raw output, and its probability where there is one."""

    index: int
    # "" when the checkpoint published no `id2label`. A caller printing this
    # sees the absence rather than a number dressed as a name.
    label: str = ""
    logit: float = 0.0
    # `None` — never 0.0 — for a head where no probability is defined: a
    # regression head, or a single output where a softmax is 1.0 by
    # construction and says nothing.
    probability: float | None = None

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "label": self.label,
            "logit": self.logit,
            "probability": self.probability,
        }


@dataclass
class Box:
    """One detector query: where it drew a box and what it called it.

    `query` is the SLOT, and it is the handle attribution needs. A DETR-family
    head has a fixed number of queries and slot 17 is slot 17 on every image
    and every occluded copy of one — but the box it draws moves, which is the
    caveat `attribute` carries.
    """

    query: int
    index: int
    label: str = ""
    score: float = 0.0
    # `None` when the scoring convention could not be established, which is a
    # different answer from a score of zero.
    logit: float | None = None
    # (x0, y0, x1, y1) in the pixels of the tensor the model was given, so a
    # caller can draw it over the same picture it measured.
    box_xyxy: list[float] = field(default_factory=list)
    # (cx, cy, w, h) normalised to [0, 1], which is what the head emits.
    box_cxcywh: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "index": self.index,
            "label": self.label,
            "score": self.score,
            "logit": self.logit,
            "box_xyxy": self.box_xyxy,
            "box_cxcywh": self.box_cxcywh,
        }


@dataclass
class Segment:
    """One label present in a mask, and how much of the picture it claims."""

    index: int
    label: str = ""
    cells: int = 0
    # Of the map's cells, not of the image's pixels — the map is coarser and
    # the fraction is the honest one either way.
    fraction: float = 0.0
    # How decisively this label won the cells it won. The QUANTITY differs by
    # head — a per-pixel head's margin is the gap to the runner-up class, a
    # mask head's is how far past the threshold its mask sat — so the field
    # never travels without `Prediction.margin_kind` saying which one it is.
    # Naming both "margin" and leaving the reader to assume would be two
    # different numbers under one word.
    mean_margin: float | None = None
    # (top, left, height, width) in map cells — the handle `attribute` takes.
    bbox: list[int] = field(default_factory=list)
    # `query` for a mask-query head, `None` for a per-pixel one. The two are
    # attributed through different reductions and the field says which.
    query: int | None = None

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "label": self.label,
            "cells": self.cells,
            "fraction": self.fraction,
            "mean_margin": self.mean_margin,
            "bbox": self.bbox,
            "query": self.query,
        }


@dataclass
class Prediction:
    """What the model said, and every way this report is narrower than it."""

    task: str = CLASSIFY
    model_name: str = ""
    # The dtype the weights are held in, because it bounds every digit below.
    dtype: str = ""
    height: int = 0
    width: int = 0
    classes: int = 0
    # Whether `id2label` was published AND matched the head's width.
    labels_read: bool = False
    labels_published: int | None = None
    # Where the names came from, or why there are none, in one sentence.
    labels_note: str = ""

    classes_top: list[ClassScore] = field(default_factory=list)
    boxes: list[Box] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)

    # How the numbers above became scores. "softmax", "sigmoid", "raw" for a
    # classifier; for a detector, whether the checkpoint's own post-processor
    # did it or this module read the convention off the head's width.
    scoring: str = ""
    scoring_reason: str = ""
    # What `Segment.mean_margin` measures on this head. Empty where no segment
    # carries one — never a default word covering two different quantities.
    margin_kind: str = ""

    # Totals, so a truncated list is visibly truncated.
    queries_total: int = 0
    segments_total: int = 0
    top_k_requested: int = 0

    # The map a segmenter produced, at ITS OWN resolution and possibly
    # subsampled. `map_stride` of 1 means every cell is there.
    label_map: list[list[int]] = field(default_factory=list)
    map_height: int = 0
    map_width: int = 0
    map_stride: int = 1

    mask_threshold: float | None = None
    forward_passes: int = 1
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "task_label": _TASK_LABEL.get(self.task, self.task),
            "model_name": self.model_name,
            "dtype": self.dtype,
            "height": self.height,
            "width": self.width,
            "classes": self.classes,
            "labels_read": self.labels_read,
            "labels_published": self.labels_published,
            "labels_note": self.labels_note,
            "classes_top": [c.to_dict() for c in self.classes_top],
            "boxes": [b.to_dict() for b in self.boxes],
            "segments": [s.to_dict() for s in self.segments],
            "scoring": self.scoring,
            "scoring_reason": self.scoring_reason,
            "margin_kind": self.margin_kind,
            "queries_total": self.queries_total,
            "segments_total": self.segments_total,
            "top_k_requested": self.top_k_requested,
            "label_map": self.label_map,
            "map_height": self.map_height,
            "map_width": self.map_width,
            "map_stride": self.map_stride,
            "mask_threshold": self.mask_threshold,
            "forward_passes": self.forward_passes,
            "seconds": self.seconds,
            "means": self.means(),
        }

    def means(self) -> str:
        parts: list[str] = [
            f"{self.model_name or 'This model'} is "
            f"{_TASK_LABEL.get(self.task, 'a vision model')}, read from the "
            f"SHAPE of what it returned rather than from its name — so a "
            f"checkpoint this tool has never heard of is measured the same way, "
            f"and one whose output does not fit any of these shapes is refused "
            f"rather than forced into the nearest."
        ]

        if self.task == CLASSIFY and self.classes_top:
            top = self.classes_top[0]
            who = top.label or f"class {top.index}"
            confidence = (
                f" at {top.probability:.4f}" if top.probability is not None else ""
            )
            parts.append(
                f"Its top answer on this {self.height}x{self.width} tensor is "
                f"{who}{confidence}, out of {self.classes} classes; "
                f"{len(self.classes_top)} of them are listed."
            )
        elif self.task == DETECT:
            parts.append(
                f"It emitted {self.queries_total} box queries and the "
                f"{len(self.boxes)} highest-scoring are listed. A query is a "
                f"SLOT, not an object: every image gets the same "
                f"{self.queries_total} of them, and a low-scoring one is the "
                f"head saying 'nothing here' rather than a missing detection."
            )
        elif self.task in (SEMANTIC, MASK_QUERIES):
            parts.append(
                f"It labelled a {self.map_height}x{self.map_width} grid — its "
                f"own internal resolution, NOT the {self.height}x{self.width} "
                f"tensor it was given — and {self.segments_total} distinct "
                f"labels appear in it, of which {len(self.segments)} are "
                f"listed. Every boundary in that map is coarser than the image "
                f"by roughly {max(1, self.height // max(1, self.map_height))}x, "
                f"whatever an upsampled overlay appears to show."
            )
            if self.map_stride > 1:
                parts.append(
                    f"THE MAP RETURNED HERE IS SUBSAMPLED, every "
                    f"{self.map_stride}th cell on each axis, because the full "
                    f"one is past the {MAX_MAP_CELLS:,} cells this carries. The "
                    f"counts and areas above are from the FULL map; only the "
                    f"picture is thinned."
                )

        if self.scoring_reason:
            parts.append(self.scoring_reason)
        if self.margin_kind:
            parts.append(self.margin_kind)
        if self.labels_note:
            parts.append(self.labels_note)

        if self.scoring == "softmax":
            parts.append(
                f"SOFTMAX CONFIDENCE IS NOT THE PROBABILITY OF BEING RIGHT. It "
                f"is a normalisation over the {self.classes} classes this head "
                f"happens to have: it moves when an unrelated class moves, it "
                f"is just as high when the model is confidently wrong, and "
                f"nothing in this tool has calibrated it. The raw logit travels "
                f"beside it for that reason."
            )
        if self.mask_threshold is not None:
            parts.append(
                f"A mask cell counts as inside at a sigmoid of "
                f"{self.mask_threshold}, which is a threshold somebody chose "
                f"rather than something the model published. A different one "
                f"gives different areas."
            )
        if self.dtype and self.dtype not in ("torch.float32", "float32", "float64"):
            parts.append(
                f"These weights are held in {self.dtype}, which carries far "
                f"fewer significant digits than the {SCORE_DECIMALS} these "
                f"numbers are printed at — the trailing ones are formatting, "
                f"not measurement."
            )
        parts.append(
            "ONE IMAGE IS A SAMPLE, NOT A PROPERTY OF THE MODEL. This is what "
            "it answered about this picture; the next picture can answer "
            "differently and nothing here has measured accuracy."
        )
        return " ".join(parts)


def _check_top_k(top_k) -> int:
    """A positive whole number of rows, with `bool` refused."""
    value = vision_attr._as_int(top_k, "top_k")
    if value < 1:
        raise BadRequest("top_k must be at least 1 — a prediction of nothing.")
    if value > MAX_TOP_K:
        raise BadRequest(
            f"top_k of {value} is past the {MAX_TOP_K} this lists. A head's "
            f"long tail is normalisation rather than prediction, and a "
            f"thousand rows is not a reading of anything."
        )
    return value


def predict(
    model,
    image,
    *,
    top_k: int = DEFAULT_TOP_K,
    processor=None,
    mask_threshold: float = DEFAULT_MASK_THRESHOLD,
    model_name: str = "",
) -> Prediction:
    """What this model says about this one image.

    `image` is a `[C, H, W]` or `[1, C, H, W]` float tensor built by the
    checkpoint's own preprocessor — `image_input.to_tensor` does that, and this
    module deliberately does no image loading of its own, so what the model is
    shown is exactly what you built.

    `processor` is optional and is used for one thing only: a detector's own
    post-processing, which knows whether its head is scored by softmax or by
    sigmoid. Absent, the convention is read from the head's width against its
    label count and the reading is reported.
    """
    import time

    import torch

    started = time.perf_counter()
    _require_eval(model, "the prediction")
    # `vision_attr` owns what counts as one image tensor. A second validator
    # here would be a second answer to that question, free to drift from the
    # one the occluder enforces two functions later.
    image = vision_attr._one_image(image)
    _require_device(model, image)
    if not isinstance(mask_threshold, float) and not isinstance(mask_threshold, int):
        raise BadRequest("mask_threshold must be a number between 0 and 1.")
    if isinstance(mask_threshold, bool) or not 0.0 < float(mask_threshold) < 1.0:
        raise BadRequest(
            f"mask_threshold must be strictly between 0 and 1, not "
            f"{mask_threshold!r}: at 0 every cell is inside every mask and at 1 "
            f"none is."
        )
    top_k = _check_top_k(top_k)

    height, width = int(image.shape[-2]), int(image.shape[-1])
    output = _run(model, image)
    task = task_of(output)

    found = Prediction(
        task=task,
        model_name=str(model_name or ""),
        dtype=str(_model_dtype(model) or ""),
        height=height,
        width=width,
        top_k_requested=top_k,
        seconds=round(time.perf_counter() - started, 3),
    )
    names = label_names(model)
    found.labels_published = None if names is None else len(names)

    if task == PROMPTED:
        raise NotMeasurable(
            "this is a promptable segmenter: it segments what you point at, so "
            "with no point, box or text prompt there is no prediction to "
            "report. Running it on an image alone returns masks for a prompt "
            "nobody gave, and inventing one here would be inventing the answer."
        )
    if task == CLASSIFY:
        _fill_classification(found, output, names, top_k, model)
    elif task == DETECT:
        _fill_detection(found, output, names, top_k, processor, height, width)
    elif task == SEMANTIC:
        _fill_semantic(found, output, names)
    else:
        found.mask_threshold = float(mask_threshold)
        _fill_mask_queries(found, output, names, float(mask_threshold), torch)

    # After the fill, because the note is about the head's ACTUAL width and
    # only the fill knows it: a detector's class tensor is one column wider
    # than its label list by construction, and a note written before that was
    # read would call a correct label list a mismatched one.
    found.labels_note = (
        f"Class names are this checkpoint's own `id2label`, "
        f"{found.labels_published} of them, in class order."
        if found.labels_read
        else _naming_note(names, found.classes)
    )
    found.seconds = round(time.perf_counter() - started, 3)
    return found


def _problem_type(model) -> str:
    """How this head's outputs are meant to be read, as the checkpoint says it.

    `problem_type` is a real transformers config field with three values, and
    they are three different arithmetics: single-label is a softmax over
    classes, multi-label is an independent sigmoid per class, and regression
    has no probability at all. Applying a softmax to a multi-label head
    produces confident-looking numbers that sum to one across classes the model
    was trained to score independently.
    """
    value = getattr(getattr(model, "config", None), "problem_type", None)
    return str(value) if value else ""


def _fill_classification(found, output, names, top_k, model) -> None:
    import torch

    logits = _tensor_of(output, "logits").float()
    classes = int(logits.shape[-1])
    found.classes = classes
    found.labels_read = names is not None and len(names) == classes

    problem = _problem_type(model)
    if problem == "regression":
        found.scoring = "raw"
        found.scoring_reason = (
            "This checkpoint declares itself a REGRESSION head, so its outputs "
            "are values rather than class scores and no probability is "
            "reported — a softmax over them would be a confident-looking "
            "number about a quantity that has no classes."
        )
        probabilities = None
    elif problem == "multi_label_classification":
        found.scoring = "sigmoid"
        found.scoring_reason = (
            "This checkpoint declares itself MULTI-LABEL, so each class is "
            "scored independently by a sigmoid and the probabilities do not sum "
            "to one — which is correct here and would be wrong for a softmax "
            "head."
        )
        probabilities = torch.sigmoid(logits[0])
    elif classes < 2:
        found.scoring = "raw"
        found.scoring_reason = (
            "This head has a single output, so there is no class probability: a "
            "softmax over one number is 1.0 by construction and would say "
            "nothing. Only the raw output is here, in this model's own units."
        )
        probabilities = None
    else:
        found.scoring = "softmax"
        found.scoring_reason = (
            f"This checkpoint states no `problem_type`, and a {classes}-wide "
            f"head with none is transformers' single-label case — so the "
            f"probabilities are a softmax across the {classes} classes. That "
            f"reading is stated rather than hidden: a multi-label head read this "
            f"way would report numbers summing to one across classes it scores "
            f"independently."
            if not problem
            else f"This checkpoint declares `{problem}`, read as a single-label "
            f"head, so the probabilities are a softmax across its {classes} "
            f"classes."
        )
        probabilities = torch.softmax(logits[0], dim=-1)

    take = min(top_k, classes)
    order = torch.topk(logits[0], take)
    for value, index in zip(order.values.tolist(), order.indices.tolist(), strict=True):
        found.classes_top.append(
            ClassScore(
                index=int(index),
                label=_named(names, int(index)) if found.labels_read else "",
                logit=round(float(value), SCORE_DECIMALS),
                probability=(
                    None
                    if probabilities is None
                    else round(float(probabilities[int(index)]), PROB_DECIMALS)
                ),
            )
        )


def _detection_convention(columns: int, names: list[str] | None) -> tuple[str, str]:
    """How to turn a detector's class logits into scores, read STRUCTURALLY.

    Two conventions are in the wild and they are not interchangeable. A
    DETR-style head has one column more than it has labels — the extra one is
    the "no object" slot — and is scored by a softmax across all of them. A
    sigmoid-style head (the deformable and real-time families) has exactly as
    many columns as labels and scores each independently.

    The head's WIDTH against the checkpoint's own label count is what
    distinguishes them, and that is a fact read from this checkpoint. A list of
    model names would be the hardcoding `vla.py` was corrected for, and it
    stops working the week after it is written.

    Where there are no labels to compare against, this returns `unknown` — and
    the caller reports raw logits rather than a score, because a probability
    computed under the wrong convention is a confident number about the wrong
    arithmetic.
    """
    if names is None:
        return "unknown", (
            f"This checkpoint publishes no `id2label`, so its {columns}-column "
            f"head cannot be told apart from a {columns - 1}-label head with a "
            f"'no object' column — the two are scored by different arithmetic. "
            f"RAW LOGITS ARE REPORTED RATHER THAN SCORES, because a probability "
            f"computed under the wrong convention is a confident number about "
            f"the wrong sum."
        )
    if columns == len(names) + 1:
        return "softmax_no_object", (
            f"This head has {columns} columns for {len(names)} labels, so the "
            f"extra one is the 'no object' slot and the scores are a softmax "
            f"across all {columns} — the DETR convention, read from the head's "
            f"own width rather than from the model's name."
        )
    if columns == len(names):
        return "sigmoid", (
            f"This head has exactly {columns} columns for its {columns} labels "
            f"and no 'no object' slot, so each class is scored independently by "
            f"a sigmoid — read from the head's own width rather than from the "
            f"model's name. These scores do not sum to one and are not meant to."
        )
    return "unknown", (
        f"This head has {columns} columns and this checkpoint publishes "
        f"{len(names)} labels, which is neither the DETR shape "
        f"({len(names) + 1}) nor the sigmoid shape ({len(names)}). RAW LOGITS "
        f"ARE REPORTED RATHER THAN SCORES: this module will not guess which "
        f"arithmetic a head it cannot recognise was trained under."
    )


def _fill_detection(found, output, names, top_k, processor, height, width) -> None:
    import torch

    logits = _tensor_of(output, "logits").float()
    boxes = _tensor_of(output, "pred_boxes").float()
    if logits.ndim != 3 or boxes.ndim != 3:
        raise NotMeasurable(
            f"this detector returned class logits shaped {tuple(logits.shape)} "
            f"and boxes shaped {tuple(boxes.shape)}; both have to be "
            f"[batch, queries, ...] for a query to be a handle on anything."
        )
    queries = int(logits.shape[1])
    columns = int(logits.shape[2])
    found.classes = columns
    found.queries_total = queries
    found.labels_read = names is not None and len(names) in (columns, columns - 1)

    convention, reason = _detection_convention(columns, names)
    found.scoring = convention
    found.scoring_reason = reason

    # The checkpoint's OWN post-processor knows its convention exactly, and
    # preferring it is the same rule the rest of this module follows: read, do
    # not infer. It is used only to confirm the scores — the box geometry is
    # computed here either way, so that a run with a processor and a run
    # without one do not report two different boxes for the same query.
    scores = None
    if processor is not None and hasattr(processor, "post_process_object_detection"):
        try:
            done = processor.post_process_object_detection(
                output, threshold=0.0, target_sizes=[(height, width)]
            )
            if done and "scores" in done[0] and len(done[0]["scores"]) == queries:
                cand = done[0]["scores"].float().cpu()
                # LENGTH IS NOT ALIGNMENT, and this treated it as though it
                # were. Some post-processors RE-RANK: they flatten the
                # (query x class) grid, sort it descending, and hand back the
                # top `queries` of it. The length check passes and row `i` is
                # then somebody else's query entirely.
                #
                # MEASURED on `PekingU/rtdetr_r50vd`: background slots with
                # logit around -11 were reported as 99% detections carrying
                # their own boxes, the three real detections never appeared,
                # and each row contradicted itself on screen — `score=0.9975`
                # printed beside `logit=-10.82`, because `logit` is
                # query-aligned and `score` was not. `conditional_detr` and
                # `deformable_detr` hit it too, at exactly 100 queries.
                #
                # A sorted result is the signature, and it is safe to test
                # for: a genuinely query-aligned output that happens to be
                # monotonically decreasing is vanishingly unlikely, and being
                # wrong about it only costs the structural reading below,
                # which is correct and says that it is a reading.
                sorted_out = bool(queries > 1 and (cand[:-1] >= cand[1:]).all())
                if sorted_out:
                    scores = None
                    found.scoring_reason = (
                        "This checkpoint's `post_process_object_detection` "
                        "re-ranks its output — it returns scores sorted by "
                        "confidence rather than one per query — so its rows "
                        "cannot be matched back to the queries the boxes come "
                        "from. Scores here are derived from the logits "
                        "instead, which keeps every row about one query."
                    )
                else:
                    scores = cand
                    found.scoring = "checkpoint_post_processor"
                    found.scoring_reason = (
                        "Scores come from this checkpoint's OWN "
                        "`post_process_object_detection`, which knows whether "
                        "its head is softmax- or sigmoid-scored — read rather "
                        "than inferred. The boxes below are computed here from "
                        "`pred_boxes` so that they do not depend on whether a "
                        "processor was supplied."
                    )
        except Exception:
            # A post-processor that will not run is not a reason to fail a
            # prediction: the structural reading below answers the same
            # question and says that it is a reading. Deliberately broad —
            # every processor family raises something different here, and the
            # fallback is complete.
            scores = None

    if scores is None:
        if convention == "softmax_no_object":
            probability = torch.softmax(logits[0], dim=-1)[:, : len(names)]
            scores = probability.max(dim=-1).values.cpu()
        elif convention == "sigmoid":
            scores = torch.sigmoid(logits[0]).max(dim=-1).values.cpu()

    # Which class each query voted for, never counting the "no object" column
    # as a class — it is the head's way of saying "nothing here", and reporting
    # it as a detection would be reporting the absence of one.
    real = (
        len(names)
        if (names is not None and convention == "softmax_no_object")
        else columns
    )
    class_index = logits[0, :, :real].argmax(dim=-1).cpu()
    class_logit = logits[0, :, :real].max(dim=-1).values.cpu()

    ranked = (scores if scores is not None else class_logit).argsort(descending=True)
    for query in ranked[: min(top_k, queries)].tolist():
        index = int(class_index[query])
        cx, cy, bw, bh = (float(v) for v in boxes[0, query].tolist())
        found.boxes.append(
            Box(
                query=int(query),
                index=index,
                label=_named(names, index) if found.labels_read else "",
                score=(
                    round(float(scores[query]), PROB_DECIMALS)
                    if scores is not None
                    else round(float(class_logit[query]), SCORE_DECIMALS)
                ),
                logit=round(float(class_logit[query]), SCORE_DECIMALS),
                box_cxcywh=[round(v, 6) for v in (cx, cy, bw, bh)],
                box_xyxy=[
                    round((cx - bw / 2) * width, 2),
                    round((cy - bh / 2) * height, 2),
                    round((cx + bw / 2) * width, 2),
                    round((cy + bh / 2) * height, 2),
                ],
            )
        )

    if scores is None:
        found.scoring_reason += (
            " The rows are ordered by that raw logit, which orders the queries "
            "correctly within this one head and means nothing between models."
        )
    found.scoring_reason += (
        " Boxes are read as `pred_boxes` — centre x, centre y, width, height, "
        "normalised to the tensor — which is the transformers convention for "
        "that field rather than something this checkpoint stated. The pixel "
        "corners beside them are that convention applied to this tensor's own "
        f"{height}x{width}."
    )


def _subsample(rows: list[list[int]]) -> tuple[list[list[int]], int]:
    """A label map thinned to fit, with the stride that thinned it.

    Returned rather than logged. A map silently reduced from 512x512 to 128x128
    is a map claiming a resolution it does not have, and every boundary in it
    would be drawn four times too coarse with nothing saying so.
    """
    height = len(rows)
    width = len(rows[0]) if height else 0
    if height * width <= MAX_MAP_CELLS or not width:
        return rows, 1
    stride = math.ceil(math.sqrt(height * width / MAX_MAP_CELLS))
    return [row[::stride] for row in rows[::stride]], stride


def _segments_from(labels, margins, names, found, *, queries=None) -> None:
    """Turn a per-cell label map into the segments present, largest first."""
    import torch

    height, width = int(labels.shape[0]), int(labels.shape[1])
    total_cells = height * width
    present = torch.unique(labels)
    found.segments_total = int(present.numel())

    rows: list[Segment] = []
    for value in present.tolist():
        index = int(value)
        mask = labels == index
        cells = int(mask.sum())
        where = mask.nonzero()
        top, left = int(where[:, 0].min()), int(where[:, 1].min())
        bottom, right = int(where[:, 0].max()), int(where[:, 1].max())
        rows.append(
            Segment(
                index=index,
                label=_named(names, index) if found.labels_read else "",
                cells=cells,
                fraction=round(cells / total_cells, 6),
                mean_margin=(
                    None
                    if margins is None
                    else round(float(margins[mask].mean()), SCORE_DECIMALS)
                ),
                bbox=[top, left, bottom - top + 1, right - left + 1],
                query=None if queries is None else int(queries[index]),
            )
        )
    rows.sort(key=lambda s: s.cells, reverse=True)
    found.segments = rows[:MAX_SEGMENTS]

    grid = [[int(v) for v in row] for row in labels.tolist()]
    found.label_map, found.map_stride = _subsample(grid)
    found.map_height, found.map_width = height, width


def _fill_semantic(found, output, names) -> None:
    logits = _tensor_of(output, "logits").float()
    if logits.shape[0] != 1:
        raise NotMeasurable(
            f"this head returned a batch of {int(logits.shape[0])} label maps "
            f"for one image, so there is no single map to report."
        )
    classes = int(logits.shape[1])
    found.classes = classes
    found.labels_read = names is not None and len(names) == classes
    found.scoring = "argmax"
    found.scoring_reason = (
        f"Each cell of the map is the class with the highest logit there, out "
        f"of {classes}. No probability is reported per cell: a softmax across "
        f"a per-pixel head's classes is the same normalisation with the same "
        f"caveats, and the MARGIN over the runner-up — which is reported per "
        f"segment — is the quantity that says whether a boundary was close."
    )
    found.margin_kind = (
        "`mean_margin` on each segment is the mean gap in logits between the "
        "winning class and the runner-up, over the cells that segment won. A "
        "segment that won everywhere by 0.001 and one that won by 8 cover the "
        "same area and are not the same finding."
        if classes >= 2
        else "This head has a single class, so there is no runner-up to "
        "measure a margin against and none is reported."
    )

    plane = logits[0]
    if classes >= 2:
        # Two largest per cell, so the margin is the real gap rather than the
        # gap to whichever class happens to sit at index 0.
        top2 = plane.topk(2, dim=0)
        labels = top2.indices[0]
        margins = top2.values[0] - top2.values[1]
    else:
        labels = plane.argmax(dim=0)
        margins = None
    _segments_from(labels, margins, names, found)


def _fill_mask_queries(found, output, names, threshold, torch) -> None:
    """A mask-query head: one class vector and one mask per query slot."""
    masks = _tensor_of(output, "masks_queries_logits").float()
    classes_logits = _tensor_of(output, "class_queries_logits")
    if classes_logits is None:
        raise NotMeasurable(
            "this head returned mask queries with no `class_queries_logits`, so "
            "its masks have no labels and there is nothing to call them. A "
            "mask with an index for a name is not a segmentation."
        )
    classes_logits = classes_logits.float()
    queries = int(masks.shape[1])
    columns = int(classes_logits.shape[-1])
    found.classes = columns
    found.queries_total = queries
    found.labels_read = names is not None and len(names) in (columns, columns - 1)

    # The same structural reading as a detector's, and for the same reason: a
    # head one column wider than its label list keeps its last column for "no
    # object", and counting that column as a class would report the absence of
    # a thing as a thing.
    real = len(names) if (names is not None and columns == len(names) + 1) else columns
    per_query = torch.softmax(classes_logits[0], dim=-1)[:, :real]
    query_class = per_query.argmax(dim=-1)

    # Each cell goes to the query whose mask claims it most strongly, and only
    # where any mask claims it at all. Cells no query claims are left as -1
    # rather than assigned to the least-bad query: "nothing was segmented here"
    # is an answer, and filling it in would draw a segment over it.
    probability = torch.sigmoid(masks[0])
    strength, owner = probability.max(dim=0)
    labels = torch.where(
        strength >= threshold,
        query_class[owner],
        torch.full_like(owner, -1),
    )
    margins = strength - threshold

    found.scoring = "mask_argmax"
    found.scoring_reason = (
        f"Each cell belongs to the query whose mask claims it most strongly, "
        f"and only where that claim passes the {threshold} sigmoid threshold — "
        f"cells no query claims are -1, which is 'nothing was segmented here' "
        f"rather than a segment nobody drew. Each query's label is the argmax "
        f"of its own class vector over {real} classes."
    )
    found.margin_kind = (
        f"`mean_margin` on each segment is how far the winning mask's sigmoid "
        f"sat ABOVE the {threshold} threshold, averaged over that segment's "
        f"cells — NOT a gap between classes, which is what the same field means "
        f"on a per-pixel head. Two different quantities never share one word "
        f"here without this sentence."
    )
    _segments_from(labels.cpu(), margins.cpu(), names, found, queries=None)
    # The query that owns each reported label, so `attribute` has a handle. A
    # label can be owned by several queries; the largest is the one carried,
    # and the segment's cell count is the label's total across all of them.
    for segment in found.segments:
        if segment.index < 0:
            segment.label = ""
            continue
        cells = labels == segment.index
        owners = owner.cpu()[cells]
        if owners.numel():
            segment.query = int(torch.mode(owners).values)


# ------------------------------------------------------------ layer readout


@dataclass
class LayerMap:
    """One layer's readout, already reduced to something that can be drawn."""

    layer: int
    rows: int = 0
    cols: int = 0
    # Row-major `rows x cols`. For attention this is the readout position's
    # attention over the patch grid, averaged over heads; for a feature map it
    # is the mean absolute activation across channels.
    values: list[list[float]] = field(default_factory=list)
    # Mean per-cell standard deviation ACROSS HEADS. A mean over heads hides
    # head-level disagreement exactly as it does on the text side, so the size
    # of what is hidden travels with it. `None` where there are no heads.
    head_disagreement: float | None = None
    # Attention the readout row spent on tokens that are not in the grid — the
    # class token attending to itself, a distillation token. Excluded from the
    # map above, so it is reported rather than silently normalised away.
    off_grid_mass: float | None = None

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "rows": self.rows,
            "cols": self.cols,
            "values": self.values,
            "head_disagreement": self.head_disagreement,
            "off_grid_mass": self.off_grid_mass,
        }


# What a readout can be. `none` is a first-class answer with a reason attached,
# because "this architecture has no attention" and "attention could not be
# captured on this build" are different facts and a single empty list would
# report both as the same nothing.
ATTENTION = "attention"
FEATURE_MAP = "feature_map"
TOKEN_NORM = "token_activation"
NONE = "none"


@dataclass
class Readout:
    """Per-layer maps, and what kind of quantity they are."""

    kind: str = NONE
    reason: str = ""
    model_name: str = ""
    dtype: str = ""
    layers: list[LayerMap] = field(default_factory=list)
    # Tokens the model actually ran on, and how they were split into a grid.
    tokens: int = 0
    prefix_tokens: int = 0
    grid_rows: int = 0
    grid_cols: int = 0
    heads: int | None = None
    # Where the attention was read FROM. A class token is the readout position
    # of a ViT classifier; without one, the map is the mean over every query
    # position, which is a different quantity and is named as one.
    source: str = ""
    forward_passes: int = 0
    attention_bytes: int | None = None
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "reason": self.reason,
            "model_name": self.model_name,
            "dtype": self.dtype,
            "layers": [layer.to_dict() for layer in self.layers],
            "n_layers": len(self.layers),
            "tokens": self.tokens,
            "prefix_tokens": self.prefix_tokens,
            "grid_rows": self.grid_rows,
            "grid_cols": self.grid_cols,
            "heads": self.heads,
            "source": self.source,
            "forward_passes": self.forward_passes,
            "attention_bytes": self.attention_bytes,
            "seconds": self.seconds,
            "means": self.means(),
        }

    def means(self) -> str:
        if self.kind == NONE:
            return (
                f"NO PER-LAYER READOUT WAS PRODUCED for "
                f"{self.model_name or 'this model'}. {self.reason} That is "
                f"stated rather than answered with an empty picture: a blank "
                f"heatmap and a model with nothing to read look identical, and "
                f"only one of them is a measurement."
            )

        parts: list[str] = []
        if self.kind == ATTENTION:
            parts.append(
                f"{len(self.layers)} layers of attention from "
                f"{self.model_name or 'this model'}, each averaged over its "
                f"{self.heads} heads and reshaped onto the "
                f"{self.grid_rows}x{self.grid_cols} patch grid the image was "
                f"cut into. The attention is read from {self.source}."
            )
            parts.append(
                "ATTENTION IS NOT A CAUSE. A patch can be attended to and "
                "change nothing in the answer. `attribute` covers a region up "
                "and re-runs the model, which is the measurement that can say "
                "what a region did."
            )
            worst = max(
                (layer.head_disagreement or 0.0 for layer in self.layers),
                default=0.0,
            )
            parts.append(
                f"The mean over heads hides head-level disagreement, so the "
                f"size of what it hides travels with each layer: the worst "
                f"layer's per-cell spread across heads is {worst:,.6f}. Where "
                f"that is comparable to the map's own values, the averaged "
                f"picture is not what any single head saw."
            )
            if self.prefix_tokens:
                off = max(
                    (layer.off_grid_mass or 0.0 for layer in self.layers), default=0.0
                )
                parts.append(
                    f"{self.prefix_tokens} of the {self.tokens} tokens are not "
                    f"patches — a class token, and a distillation token where "
                    f"there is one — so they have no place on the grid and are "
                    f"excluded from it. They still take real attention mass: up "
                    f"to {fmt.measured(off)} of a row in the layers here, which is "
                    f"reported rather than normalised away."
                )
        elif self.kind == FEATURE_MAP:
            parts.append(
                f"{len(self.layers)} layers of FEATURE MAPS from "
                f"{self.model_name or 'this model'} — the mean absolute "
                f"activation across channels at each spatial position. "
                f"{self.reason}"
            )
            parts.append(
                "THIS IS NOT ATTENTION and must not be read as it. A large "
                "activation is a strong response, not a routing decision, and "
                "nothing here says the answer depended on it — which is exactly "
                "what `attribute` measures."
            )
        else:
            parts.append(
                f"{len(self.layers)} layers of TOKEN ACTIVATION MAGNITUDE from "
                f"{self.model_name or 'this model'} — the L2 norm of each "
                f"patch's hidden state, on the "
                f"{self.grid_rows}x{self.grid_cols} grid. {self.reason}"
            )
            parts.append(
                "THIS IS NOT ATTENTION. It is how large a token's "
                "representation is, which is a different quantity that happens "
                "to draw the same-looking picture."
            )

        parts.append(
            f"{self.forward_passes} forward passes were run to produce this"
            + (
                f", holding {self.attention_bytes / 1e6:,.1f} MB of attention "
                f"matrices at the peak."
                if self.attention_bytes
                else "."
            )
        )
        if self.dtype and self.dtype not in ("torch.float32", "float32", "float64"):
            parts.append(
                f"These weights are held in {self.dtype}, so the trailing "
                f"digits of every number above are formatting rather than "
                f"measurement."
            )
        return " ".join(parts)


def _head_count(model) -> int | None:
    """How many attention heads a layer has, or `None`.

    Read from the config, and from the vision half of a composite one, because
    that is where a checkpoint states it. `None` is the honest answer where it
    is not stated — and it is load-bearing: without it the memory an attention
    capture would hold cannot be priced, and an unpriced capture on a large
    model is an out-of-memory in the middle of somebody's measurement.
    """
    config = getattr(model, "config", None)
    for holder in (config, getattr(config, "vision_config", None)):
        value = getattr(holder, "num_attention_heads", None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return int(value)
    return None


def _patch_grid(model, height: int, width: int) -> tuple[int, int] | None:
    """The rows and columns of patches this image was cut into, or `None`.

    From the checkpoint's own `patch_size`, which is what decides it. A model
    that does not publish one is not a patch model, or is one this cannot lay
    out — and a grid guessed from a token count factorises many ways.
    """
    config = getattr(model, "config", None)
    for holder in (config, getattr(config, "vision_config", None)):
        size = getattr(holder, "patch_size", None)
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            continue
        if height % size or width % size:
            return None
        return height // size, width // size
    return None


def _eager(model):
    """Switch attention to eager if this build can, returning how to undo it.

    SDPA and flash attention never materialise the probability matrix, so
    `output_attentions=True` on them comes back as an EMPTY TUPLE — measured on
    transformers 5.14.1, where the warning goes to a log nobody reads and the
    result looks exactly like a model with no attention. `runtime.py` forces
    eager at load time on the text side for the same reason; this does it for
    a model that is already loaded, and puts it back.

    Returns `(changed, previous)`. `changed` False means this build offers no
    way to switch, which the caller reports rather than papers over.
    """
    setter = getattr(model, "set_attn_implementation", None)
    previous = getattr(getattr(model, "config", None), "_attn_implementation", None)
    if not callable(setter) or previous is None:
        return False, None
    if str(previous) == "eager":
        return False, None
    try:
        setter("eager")
    except Exception:
        # A build that rejects the switch, or a model whose sub-configs
        # disagree. Broad on purpose: this is a best-effort improvement to a
        # measurement that has a complete fallback, and the fallback says which
        # one it took.
        return False, None
    return True, str(previous)


def layer_readout(model, image, *, model_name: str = "") -> Readout:
    """What each layer of this model looked at, where there is such a thing.

    A vision transformer has attention and it is captured. A convolutional
    classifier does not, and it gets feature maps — a real quantity, named as a
    different one. A model with neither gets `kind="none"` and a sentence,
    because a blank heatmap and a model with nothing to read look identical.

    Two forward passes at most: one for the hidden states, which is what says
    how many tokens there are and therefore what an attention capture would
    cost, and one more for the attention itself when that cost is affordable.
    Both are counted in the result.
    """
    import time

    import torch

    started = time.perf_counter()
    _require_eval(model, "a per-layer readout")
    image = vision_attr._one_image(image)
    _require_device(model, image)
    height, width = int(image.shape[-2]), int(image.shape[-1])

    found = Readout(
        model_name=str(model_name or ""),
        dtype=str(_model_dtype(model) or ""),
    )

    if not _accepts(model, "output_hidden_states"):
        raise NotMeasurable(
            "this model's forward pass does not accept `output_hidden_states`, "
            "so there is no per-layer anything to read out of it from outside. "
            "Nothing was run: a pass that could not have returned the layers is "
            "a pass spent on nothing."
        )
    output = _run(model, image, output_hidden_states=True)
    found.forward_passes = 1

    hidden = getattr(output, "hidden_states", None)
    if not hidden:
        found.reason = (
            "It returned no hidden states, so there are no per-layer "
            "activations here at all — not a shallow model, a model this "
            "cannot see inside from its output alone."
        )
        found.seconds = round(time.perf_counter() - started, 3)
        return found

    grid = _patch_grid(model, height, width)
    first = hidden[0]
    if first.ndim == 3:
        found.tokens = int(first.shape[1])
        if grid is not None and grid[0] * grid[1] <= found.tokens:
            found.grid_rows, found.grid_cols = grid
            found.prefix_tokens = found.tokens - grid[0] * grid[1]

    heads = _head_count(model)
    found.heads = heads
    attention = _capture_attention(model, image, hidden, heads, found)
    if attention is not None:
        _fill_attention(found, attention, heads)
        found.seconds = round(time.perf_counter() - started, 3)
        return found

    _fill_features(found, hidden, torch)
    found.seconds = round(time.perf_counter() - started, 3)
    return found


def _capture_attention(model, image, hidden, heads, found):
    """The attention tuple, or `None` with `found.reason` saying why not."""
    if hidden[0].ndim != 3:
        found.reason = (
            "Its hidden states are spatial feature maps rather than sequences "
            "of tokens, which is a convolutional stack: there is no attention "
            "matrix in it to capture."
        )
        return None
    if not found.grid_rows:
        found.reason = (
            f"Its {found.tokens} tokens could not be laid back onto a patch "
            f"grid — this checkpoint publishes no usable `patch_size`, or the "
            f"image does not divide by it — so an attention map would have no "
            f"geometry to be drawn on."
        )
        return None
    if heads is None:
        found.reason = (
            "Its head count could not be read from this checkpoint's config, so "
            "the memory an attention capture would hold could not be priced "
            "first — and an unpriced capture is an out-of-memory in the middle "
            "of a measurement rather than before one."
        )
        return None

    layers = max(0, len(hidden) - 1)
    predicted = layers * heads * found.tokens * found.tokens * 4
    found.attention_bytes = predicted
    if predicted > MAX_ATTENTION_BYTES:
        found.reason = (
            f"Capturing its attention would hold {predicted / 1e9:,.2f} GB at "
            f"once — {layers} layers of {heads} heads over {found.tokens} "
            f"tokens, all resident together — which is past the "
            f"{MAX_ATTENTION_BYTES / 1e9:,.1f} GB this will allocate. Nothing "
            f"was run: finding out by waiting for an out-of-memory is the "
            f"failure this refusal exists to prevent."
        )
        return None

    if not _accepts(model, "output_attentions"):
        found.attention_bytes = None
        found.reason = (
            "Its forward pass does not accept `output_attentions`, so whatever "
            "attention it has is not reachable from outside it — and no second "
            "pass was spent finding that out."
        )
        return None

    changed, previous = _eager(model)
    try:
        output = _run(model, image, output_attentions=True)
    finally:
        if changed and previous:
            # Always put back. A model left in eager attention runs every later
            # generation in this process more slowly, which is a performance
            # regression whose cause is a panel somebody opened once.
            try:
                model.set_attn_implementation(previous)
            except Exception as err:
                # Logged rather than swallowed. The measurement above is
                # already complete and failing it now would throw away a good
                # result over cleanup — but a model stuck in eager attention
                # runs every later pass in this process more slowly, and "the
                # app got slow after I opened that panel" is a bug nobody
                # traces back here without this line.
                log.warning(
                    "could not restore %s attention after a readout (%s)",
                    previous,
                    type(err).__name__,
                )
    found.forward_passes += 1

    attention = getattr(output, "attentions", None)
    if not attention:
        # Measured, not assumed: on transformers 5.14.1 an SDPA model returns
        # an EMPTY TUPLE here and warns to a log. Reporting that as zero layers
        # of attention would be reporting a fact about the kernel as a fact
        # about the architecture.
        found.attention_bytes = None
        found.reason = (
            "It ran under an attention kernel that never materialises the "
            "probability matrix — SDPA or flash attention — and this build "
            "offered no way to switch it to eager, so the capture came back "
            "empty. That is a fact about the kernel, NOT about the "
            "architecture: this model does have attention, and reconstructing "
            "it from hidden states afterwards would be a different quantity "
            "from the one it used."
        )
        return None
    return attention


def _fill_attention(found, attention, heads) -> None:
    """Reduce every layer's [heads, q, k] to one drawable map, on the spot.

    Reduced as it arrives rather than collected and averaged later, for the
    reason `image_attention` gives: the full stack is layers x heads x tokens
    squared, and holding it to reduce afterwards means holding all of it at
    once when the reduction throws away all but a few hundred numbers.
    """
    import torch

    prefix = found.prefix_tokens
    found.source = (
        "the class token, which is the position a classifier reads its answer from"
        if prefix >= 1
        else "the mean over every query position, because this model has no "
        "class token to read from — a different quantity from a classifier's "
        "readout and named as one"
    )
    cells = found.grid_rows * found.grid_cols
    from_config = heads

    for index, layer in enumerate(attention):
        if not isinstance(layer, torch.Tensor) or layer.ndim != 4:
            continue
        # The head count is now MEASURED rather than read: the config's number
        # priced the capture, and this is the tensor that actually came back.
        # Where the two disagree the config was wrong and the memory figure was
        # wrong with it, which is worth saying out loud.
        found.heads = int(layer.shape[1])
        probabilities = layer[0].float()
        row = probabilities[:, 0, :] if prefix >= 1 else probabilities.mean(dim=1)
        patches = row[:, prefix : prefix + cells]
        if int(patches.shape[-1]) != cells:
            continue
        mean = patches.mean(dim=0)
        spread = patches.std(dim=0) if int(patches.shape[0]) > 1 else None
        off = float(row[:, :prefix].sum(dim=-1).mean()) if prefix else None
        found.layers.append(
            LayerMap(
                layer=index,
                rows=found.grid_rows,
                cols=found.grid_cols,
                values=[
                    [round(float(v), SCORE_DECIMALS) for v in line]
                    for line in mean.reshape(found.grid_rows, found.grid_cols).tolist()
                ],
                head_disagreement=(
                    None
                    if spread is None
                    else round(float(spread.mean()), SCORE_DECIMALS)
                ),
                off_grid_mass=None if off is None else round(off, SCORE_DECIMALS),
            )
        )
    found.kind = ATTENTION if found.layers else NONE
    if not found.layers:
        found.reason = (
            f"Its attention came back in a shape this cannot lay on the "
            f"{found.grid_rows}x{found.grid_cols} patch grid, so no layer "
            f"produced a map. Reshaping it anyway would draw a picture of "
            f"whichever axis the indexing happened to pick."
        )
        return
    if from_config is not None and found.heads != from_config:
        found.reason = (
            f"This checkpoint's config says {from_config} attention heads and "
            f"its attention tensors have {found.heads}. The measured number is "
            f"what the maps above are averaged over; the config's is what the "
            f"memory this capture was priced at came from, so that estimate "
            f"was off by the same ratio."
        )


def _fill_features(found, hidden, torch) -> None:
    """Feature maps for a stack that has no attention to capture."""
    too_wide: list[str] = []
    for index, layer in enumerate(hidden):
        if not isinstance(layer, torch.Tensor):
            continue
        if layer.ndim == 4:
            # [batch, channels, h, w] — a convolutional stage. The mean
            # absolute activation across channels is the one reduction that
            # does not privilege a channel nobody chose.
            plane = layer[0].float().abs().mean(dim=0)
            rows, cols = int(plane.shape[0]), int(plane.shape[1])
        elif layer.ndim == 3 and found.grid_rows:
            cells = found.grid_rows * found.grid_cols
            tokens = layer[0].float()[found.prefix_tokens : found.prefix_tokens + cells]
            if int(tokens.shape[0]) != cells:
                continue
            plane = tokens.norm(dim=-1).reshape(found.grid_rows, found.grid_cols)
            rows, cols = found.grid_rows, found.grid_cols
        else:
            continue
        if rows * cols > MAX_MAP_CELLS:
            # An early convolutional stage is close to full resolution, and
            # several of those are a megabyte of JSON each. Dropped rather than
            # thinned — a subsampled feature map is a different picture from
            # the one the layer produced — and every one that was dropped is
            # named below, because a layer list with holes in it reads as a
            # model with fewer layers.
            too_wide.append(f"layer {index} at {rows}x{cols}")
            continue
        found.layers.append(
            LayerMap(
                layer=index,
                rows=rows,
                cols=cols,
                values=[
                    [round(float(v), SCORE_DECIMALS) for v in line]
                    for line in plane.tolist()
                ],
            )
        )

    dropped = (
        f" {len(too_wide)} layers are NOT in this list because their maps are "
        f"past the {MAX_MAP_CELLS:,} cells this carries "
        f"({', '.join(too_wide[:4])}"
        f"{', and more' if len(too_wide) > 4 else ''}) — dropped rather than "
        f"thinned, because a subsampled feature map is a different picture from "
        f"the one the layer produced."
        if too_wide
        else ""
    )

    if not found.layers:
        found.kind = NONE
        found.reason = (
            found.reason
            or "Its hidden states are neither spatial feature maps nor a patch "
            "grid this could lay out, so there is no per-layer picture here "
            "that would be about the image rather than about an axis chosen at "
            "random."
        ) + dropped
        return

    # `reason` arrives carrying WHY there was no attention, which is the more
    # informative half and is kept. The sentence appended says what is here
    # instead, so a reader is never left with an explanation of an absence and
    # no account of the thing they are looking at.
    found.kind = FEATURE_MAP if hidden[0].ndim == 4 else TOKEN_NORM
    found.reason = (
        found.reason
        or "This model has no attention matrix to capture at all, so what is "
        "here is the activation itself."
    ) + dropped


# ------------------------------------------------------------- attribution


@dataclass
class Attributed:
    """One occlusion map, and what exactly it is a map OF.

    The map itself is `vision_attr`'s and is nested unchanged: there is one
    occluder in this project and this does not restate its numbers or its
    caveats. What this adds is the answer to "which of the model's answers is
    this about", which for a detector or a segmenter is a choice somebody made.
    """

    attribution: object = None
    task: str = CLASSIFY
    # "model" when this module took the model's own top answer, "caller" when
    # it was named. The difference is the difference between explaining the
    # answer given and auditing one you supplied.
    region_chosen_by: str = "model"
    what: str = ""
    query: int | None = None
    region: list[int] | None = None
    target_label: str = ""
    # For a segmenter, the region is in the MAP's cells and the occluder works
    # in the IMAGE's pixels. Both resolutions travel so nobody reads one as the
    # other.
    map_height: int = 0
    map_width: int = 0
    dtype: str = ""

    @property
    def names_dropped_by_the_sweep(self) -> bool:
        """Did the occluder decline to caption its own map with the names?

        Read from the sweep's own report rather than recomputed here, so the
        sentence explaining it cannot end up attached to a run where it did
        not happen — two places deciding the same fact is how they drift.
        """
        return bool(getattr(self.attribution, "class_names_dropped", False))

    def to_dict(self) -> dict:
        return {
            "attribution": self.attribution.to_dict() if self.attribution else None,
            "task": self.task,
            "task_label": _TASK_LABEL.get(self.task, self.task),
            "region_chosen_by": self.region_chosen_by,
            "what": self.what,
            "query": self.query,
            "region": self.region,
            "target_label": self.target_label,
            "map_height": self.map_height,
            "map_width": self.map_width,
            "dtype": self.dtype,
            "names_dropped_by_the_sweep": self.names_dropped_by_the_sweep,
            "means": self.means(),
        }

    def means(self) -> str:
        parts = [f"This map is about {self.what}."]
        if self.region_chosen_by == "model":
            parts.append(
                "That was the model's own strongest answer, picked by this tool "
                "rather than named by you — so the map explains the answer "
                "given, not a correct one."
            )
        else:
            parts.append(
                "You named it. This is where the evidence for THAT answer is, "
                "whether or not the model gave it — 'why did it pick that' and "
                "'what supports this other one' are different questions with "
                "the same picture."
            )

        if self.names_dropped_by_the_sweep:
            parts.append(
                "The sweep below reports the class names as dropped, and that "
                "is correct rather than a gap: this head's class vector is one "
                "column wider than the checkpoint's label list — the extra "
                "column is the 'no object' slot — so matching the names by "
                "position would put every one of them against the wrong class. "
                "The label named above comes from the class index instead, "
                "which is the same list read the right way round."
            )

        if self.task == DETECT:
            parts.append(
                f"A QUERY SLOT IS FIXED, THE BOX IT DRAWS IS NOT. Slot "
                f"{self.query} is the same slot on every occluded copy of this "
                f"image, which is what makes the comparison possible — but the "
                f"box that slot emits moves as the image changes, so a large "
                f"drop can mean the object moved out of that slot as much as it "
                f"stopped being visible. The score is the class logit of the "
                f"slot, which is the part that is comparable."
            )
        elif self.task in (SEMANTIC, MASK_QUERIES) and self.region:
            top, left, rows, cols = self.region
            parts.append(
                f"The score is the mean class logit over the "
                f"{rows}x{cols} block of the model's own "
                f"{self.map_height}x{self.map_width} output grid at row {top}, "
                f"column {left} — averaged, because a segmenter has one logit "
                f"per cell and a map of the whole thing would be one occlusion "
                f"sweep per cell. Averaging is a choice: a region whose cells "
                f"disagree reports their mean and looks like a region that "
                f"mildly agrees."
            )
        if self.dtype and self.dtype not in ("torch.float32", "float32", "float64"):
            parts.append(
                f"These weights are held in {self.dtype}, so the last digits of "
                f"every drop below are that precision rather than the model's "
                f"opinion."
            )
        parts.append(
            "Everything below is the occlusion sweep's own report, unchanged — "
            "the same measurement, the same fill caveat and the same resolution "
            "limit as every other attribution in this tool."
        )
        return " ".join(parts)


def _reduce_for(task, *, query, region, columns_note):
    """The `forward=` that turns this head's output into `[batch, classes]`.

    This is the entire content of "attribute over a box" or "over a mask
    region": the sweep is unchanged and measures whatever two-dimensional
    tensor it is handed, so choosing that tensor is choosing what the map is
    about — which is why it is done here, explicitly, and never by an axis
    picked inside a generic reducer.
    """

    def detector(model, x):
        out = model(x)
        logits = _tensor_of(out, "logits")
        if logits is None or logits.ndim != 3:
            raise NotMeasurable(columns_note)
        return logits[:, query, :]

    def mask_query(model, x):
        out = model(x)
        logits = _tensor_of(out, "class_queries_logits")
        if logits is None or logits.ndim != 3:
            raise NotMeasurable(columns_note)
        return logits[:, query, :]

    def semantic(model, x):
        out = model(x)
        logits = _tensor_of(out, "logits")
        if logits is None or logits.ndim != 4:
            raise NotMeasurable(columns_note)
        top, left, rows, cols = region
        return logits[:, :, top : top + rows, left : left + cols].mean(dim=(-2, -1))

    return {DETECT: detector, MASK_QUERIES: mask_query, SEMANTIC: semantic}[task]


def attribute(
    model,
    image,
    *,
    target: int | None = None,
    query: int | None = None,
    region: tuple[int, int, int, int] | None = None,
    processor=None,
    patch: int = vision_attr.DEFAULT_PATCH,
    stride: int | None = None,
    fill: str = vision_attr.DEFAULT_FILL,
    value_range: tuple[float, float] | None = None,
    batch: int = vision_attr.DEFAULT_BATCH,
    max_passes: int = vision_attr.MAX_PASSES,
    model_name: str = "",
) -> Attributed:
    """Occlude the image and measure what it does to THIS prediction.

    `target` names a class, `query` names a detector's box slot or a mask
    head's query, `region` names a `(top, left, height, width)` block of a
    segmenter's own output grid. Everything left out is taken from the model's
    own top answer and the result says so — attributing the argmax is one
    question and "what supports this other class" is another, and this refuses
    to answer the second while claiming to have answered the first.

    The sweep itself is `vision_attr.sweep`, called with a reduction rather
    than reimplemented. Its refusals, its fill caveat and its resolution limit
    all apply here unchanged, which is the point.
    """
    _require_eval(model, "an attribution map")
    image = vision_attr._one_image(image)
    _require_device(model, image)

    probe = _run(model, image)
    task = task_of(probe)
    names = label_names(model)
    dtype = str(_model_dtype(model) or "")

    if task == PROMPTED:
        raise NotMeasurable(
            "this is a promptable segmenter, so there is no unprompted "
            "prediction to attribute. Occluding an image to explain a mask "
            "nobody asked for would be explaining an answer this tool invented."
        )

    if task == CLASSIFY:
        if query is not None or region is not None:
            raise BadRequest(
                "this is a classifier: it has no box queries and no mask "
                "regions, only classes. Pass `target=` to attribute a class "
                "other than its top one."
            )
        logits = _tensor_of(probe, "logits")
        chosen = "caller" if target is not None else "model"
        index = int(target) if target is not None else int(logits[0].argmax())
        # Named only when the checkpoint's list matches the head exactly, the
        # same rule `vision_attr` applies before it captions a map: a name list
        # of the wrong length mislabels at least one class.
        matched = names is not None and len(names) == int(logits.shape[-1])
        label = _named(names, index) if matched else ""
        found = vision_attr.sweep(
            model,
            image,
            target=target,
            patch=patch,
            stride=stride,
            fill=fill,
            value_range=value_range,
            batch=batch,
            forward=lambda m, x: m(_cast(x, m)),
            class_names=names,
            max_passes=max_passes,
            model_name=model_name,
        )
        return Attributed(
            attribution=found,
            task=task,
            region_chosen_by=chosen,
            what=f"the class {label or f'index {index}'}",
            target_label=label,
            dtype=dtype,
        )

    prediction = None
    if query is None and region is None:
        # The model's own strongest answer, computed by the same function that
        # reports predictions — so "the top box" here and "the top box" in the
        # prediction panel are the same box rather than two rankings.
        prediction = predict(
            model, image, top_k=1, processor=processor, model_name=model_name
        )

    if task in (DETECT, MASK_QUERIES):
        if region is not None:
            raise BadRequest(
                f"{_TASK_LABEL[task]} is attributed over a QUERY SLOT, not over "
                f"a region of the map: pass `query=`. A region would be a block "
                f"of a grid this head does not produce."
            )
        if query is None:
            rows = prediction.boxes if task == DETECT else prediction.segments
            if not rows:
                raise NotMeasurable(
                    "this model produced no scored query on this image, so "
                    "there is nothing here to explain. A detector that finds "
                    "nothing is a result; occluding to explain it would be "
                    "explaining an absence."
                )
            picked = rows[0].query
            if picked is None:
                # Either the largest region of this image was segmented by
                # nobody, or it was claimed by several slots at once. Both mean
                # there is no slot to hold fixed across the sweep, and both are
                # answers rather than errors — so the sentence covers the two
                # rather than asserting whichever is more common.
                raise NotMeasurable(
                    "the largest region of this model's own output grid is not "
                    "owned by a single query slot — nothing claimed it, or "
                    "several slots did — so there is no slot to hold fixed "
                    "across the sweep. Name one with `query=`, from the "
                    "prediction's segment list."
                )
            query = int(picked)
            chosen = "model"
            label, called_by_model = rows[0].label, True
        else:
            query = vision_attr._as_int(query, "query")
            chosen = "caller"
            label, called_by_model = "", False
        if target is not None:
            # A named class overrides whatever the slot itself voted for, and
            # the caption has to follow it — captioning the map with the slot's
            # own class while measuring a different one is exactly the
            # mislabelling this module refuses everywhere else.
            label, called_by_model = _named(names, int(target)), False
        note = (
            "this model stopped returning per-query class logits partway "
            "through the sweep, so the remaining windows would be measuring a "
            "different quantity from the first ones."
        )
        reduce = _reduce_for(task, query=query, region=None, columns_note=note)
        slot = "box query" if task == DETECT else "mask query"
        what = f"{slot} {query}" + (
            f", which the model called {label}"
            if label and called_by_model
            else f", scored for the class {label}"
            if label
            else ""
        )
        found = vision_attr.sweep(
            model,
            image,
            target=target,
            patch=patch,
            stride=stride,
            fill=fill,
            value_range=value_range,
            batch=batch,
            forward=lambda m, x, _r=reduce: _r(m, _cast(x, m)),
            class_names=names,
            max_passes=max_passes,
            model_name=model_name,
        )
        return Attributed(
            attribution=found,
            task=task,
            region_chosen_by=chosen if target is None else "caller",
            what=what,
            query=query,
            target_label=label,
            dtype=dtype,
        )

    # Per-pixel segmentation: the handle is a block of the model's own grid.
    logits = _tensor_of(probe, "logits")
    map_height, map_width = int(logits.shape[-2]), int(logits.shape[-1])
    if query is not None:
        raise BadRequest(
            "this is a per-pixel segmenter: it has no query slots, only a grid "
            "of labels. Pass `region=(top, left, height, width)` in that grid's "
            "own cells."
        )
    if region is None:
        segment = prediction.segments[0] if prediction.segments else None
        if segment is None:
            raise NotMeasurable(
                "this model labelled no cell of its own output grid, so there "
                "is no region here to explain."
            )
        region = tuple(segment.bbox)
        chosen = "model"
        label = segment.label
    else:
        chosen = "caller"
        label = ""
    try:
        values = list(region)
    except TypeError:
        values = []
    if len(values) != 4:
        raise BadRequest(
            f"region must be four whole numbers — (top, left, height, width) "
            f"in the model's own {map_height}x{map_width} output grid — not "
            f"{region!r}."
        ) from None
    region = tuple(vision_attr._as_int(v, "region") for v in values)
    top, left, rows, cols = region
    if rows < 1 or cols < 1:
        raise BadRequest(
            f"a region of {rows}x{cols} cells has no area, so there is nothing "
            f"to average a logit over."
        )
    if not (
        0 <= top and 0 <= left and top + rows <= map_height and left + cols <= map_width
    ):
        raise BadRequest(
            f"the region ({top}, {left}, {rows}, {cols}) falls outside this "
            f"model's {map_height}x{map_width} output grid. That grid is the "
            f"model's own resolution, not the image's — a region in image "
            f"pixels would be measuring somewhere else entirely."
        )
    note = (
        "this model stopped returning a per-pixel grid partway through the "
        "sweep, so the remaining windows would be measuring a different "
        "quantity from the first ones."
    )
    reduce = _reduce_for(SEMANTIC, query=None, region=region, columns_note=note)
    found = vision_attr.sweep(
        model,
        image,
        target=target,
        patch=patch,
        stride=stride,
        fill=fill,
        value_range=value_range,
        batch=batch,
        forward=lambda m, x, _r=reduce: _r(m, _cast(x, m)),
        class_names=names,
        max_passes=max_passes,
        model_name=model_name,
    )
    return Attributed(
        attribution=found,
        task=task,
        region_chosen_by=chosen if target is None else "caller",
        what=(
            f"the {rows}x{cols} region of the label map at row {top}, column "
            f"{left}" + (f", which the model labelled {label}" if label else "")
        ),
        region=list(region),
        target_label=label,
        map_height=map_height,
        map_width=map_width,
        dtype=dtype,
    )


# ------------------------------------------------------------------- cost


def readout_shape_of(model) -> dict:
    """What an attention capture on this model would hold, from its config.

    Runs nothing. `tokens` is the PATCH GRID only, which is a lower bound: a
    class token and a distillation token add to it and this cannot know how
    many a checkpoint has without running it. That is said in `plan`'s
    sentence rather than quietly rounded up, because a memory figure quoted as
    exact and short by two tokens squared is worse than one labelled as a floor.
    """
    config = getattr(model, "config", None)
    layers = None
    for holder in (config, getattr(config, "vision_config", None)):
        value = getattr(holder, "num_hidden_layers", None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            layers = int(value)
            break

    size = None
    for holder in (config, getattr(config, "vision_config", None)):
        value = getattr(holder, "image_size", None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            size = int(value)
            break

    tokens = None
    if size is not None:
        grid = _patch_grid(model, size, size)
        if grid is not None:
            tokens = grid[0] * grid[1]

    return {"layers": layers, "heads": _head_count(model), "tokens": tokens}


def plan(
    height: int,
    width: int,
    *,
    layers: int | None = None,
    heads: int | None = None,
    tokens: int | None = None,
    patch: int = vision_attr.DEFAULT_PATCH,
    stride: int | None = None,
    batch: int = vision_attr.DEFAULT_BATCH,
    seconds_per_pass: float | None = None,
    max_passes: int = vision_attr.MAX_PASSES,
) -> dict:
    """What the three measurements cost, before any of them is spent.

    Needs no model — the shape arguments come from `readout_shape_of`, which
    reads a loaded one's config without running it. The attribution half is
    `vision_attr.estimate` itself rather than the same arithmetic written
    twice: two functions answering "how many forward passes" would be free to
    disagree, and this panel and the attribution panel would then price the
    same sweep differently.

    Like `estimate`, this NEVER refuses on the ceiling. A caller about to be
    refused needs the number that got them refused.
    """
    sweep = vision_attr.estimate(
        height,
        width,
        patch=patch,
        stride=stride,
        batch=batch,
        seconds_per_pass=seconds_per_pass,
        max_passes=max_passes,
    )

    attention_bytes = None
    # `isinstance(True, int)` is True, so `layers=True` would price a
    # one-layer capture and look like somebody asked for it.
    if all(
        isinstance(v, int) and not isinstance(v, bool) and v > 0
        for v in (layers, heads, tokens)
    ):
        attention_bytes = layers * heads * tokens * tokens * 4
    readout_passes = 1 if attention_bytes is None else 2

    if attention_bytes is None:
        memory = (
            "The layers, heads or token count could not be read, so the memory "
            "an attention capture would hold is UNKNOWN rather than zero — a "
            "capture that could not be priced is not a capture that costs "
            "nothing. `readout_shape_of(model)` reads the three off a loaded "
            "model's config without running it."
        )
    else:
        fits = attention_bytes <= MAX_ATTENTION_BYTES
        memory = (
            f"An attention capture holds every layer at once: {layers} layers x "
            f"{heads} heads x {tokens} x {tokens} in float32 is "
            f"{attention_bytes / 1e6:,.1f} MB"
            + (
                f", within the {MAX_ATTENTION_BYTES / 1e9:,.1f} GB this will allocate."
                if fits
                else f", past the {MAX_ATTENTION_BYTES / 1e9:,.1f} GB this will "
                f"allocate — the readout will report feature maps instead and "
                f"say why."
            )
            + " That token count is the PATCH GRID only: a class token and a "
            "distillation token are not in it, so this figure is a floor rather "
            "than an exact size."
        )

    total = 1 + readout_passes + sweep["passes"]
    seconds = (
        None if seconds_per_pass is None else round(total * float(seconds_per_pass), 3)
    )

    return {
        "predict": {"forward_passes": 1},
        "readout": {
            "forward_passes": readout_passes,
            "attention_bytes": attention_bytes,
            "layers": layers,
            "heads": heads,
            "tokens": tokens,
            "ceiling_bytes": MAX_ATTENTION_BYTES,
        },
        "attribution": sweep,
        "total_forward_passes": total,
        "seconds": seconds,
        "means": (
            f"Three measurements on one {height}x{width} image: the prediction "
            f"is 1 forward pass, the per-layer readout is {readout_passes} "
            f"(one for the hidden states, which is what says whether an "
            f"attention capture is affordable, and one for the attention "
            f"itself), and the occlusion sweep is {sweep['passes']} — "
            f"{total} passes in all.\n\n{memory}\n\n"
            + (
                f"At the {float(seconds_per_pass):,.4f}s per pass you measured, "
                f"all three are {seconds:,.1f}s."
                if seconds is not None
                else "No per-pass time was measured on this machine, so there "
                "is no forecast here — an invented one would be a number this "
                "tool made up."
            )
            + f"\n\nThe sweep's own half of this: {sweep['means']}"
        ),
    }
