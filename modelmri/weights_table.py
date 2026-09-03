# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""Every tensor in a model, and whether its numbers are still numbers.

Two tools already do half of this, and neither does it to a model that is
loaded and running in front of you.

Netron opens a FILE and draws its graph: per-node op type, shapes, dtypes, and
the tensor data behind them. It is the best thing there is for that, and it is
static — it reads a checkpoint on disk, not the object your process is holding
after `from_pretrained`, after a LoRA merge, after a quantisation, after four
hours of finetuning.

TensorBoard's Debugger V2 has a `FULL_HEALTH` tensor mode that reports, per
tensor, how many of its elements are NaN, +Inf, -Inf and finite. That is the
right measurement and it has to be wired in before the run starts, into a
TensorFlow graph, and it tells you about tensors flowing through ops rather
than about the weights sitting in the model.

ModelMRI is already holding the model. So this walks `named_parameters()` and
`named_buffers()` on whatever `nn.Module` it is given and answers both
questions at once, on the live object, for an architecture this file has never
heard of.

## Nothing here consults a table of dtype widths

A byte count is `element_size() * numel()` from torch itself, or the span
between a tensor's `data_offsets` in a safetensors header. Both are read. This
module has no dtype-to-width map, on purpose: a map is a thing that can be
wrong about a dtype it has not seen, and `fit.py` already has to refuse an
unknown one. Reading the width leaves nothing to be wrong about.

## A NaN count of zero is not the same answer as "nobody looked"

This is the whole discipline of the health half, and it is enforced by shape
rather than by convention: a row carries EITHER a `Health` with real counts in
it OR `health=None` with a sentence saying why nothing was counted. There is
no zeroed `Health` anywhere in this file. A tensor on the `meta` device, a
complex tensor, an empty tensor, a tensor whose storage another row already
scanned, and a tensor the run's budget never reached are five different
reasons, and each one says which it is.

`Health.all_finite` carries the same distinction one level down, as
`True | False | None`:

  `False`  something non-finite was positively found. A finding survives
           sampling — seeing one NaN proves a NaN is there.
  `True`   the whole tensor was read and none was found, OR the dtype has no
           bit pattern for NaN and infinity and the answer is true by
           construction rather than by counting.
  `None`   part of the tensor was read and none was found there. THAT IS NOT
           A CLEAN BILL OF HEALTH, and it must not collapse into `True`.

The same rule runs through the statistics. A tensor that is entirely NaN has
no finite part, so its minimum, maximum, mean and standard deviation are
`None` — not `0.0`, which is a number some tensor somewhere really does have.
And the statistics are never rounded on the way out: a weight of 1e-12 rounded
to nine places is 0.0, and turning things into zero is the one thing this
module exists not to do.

## The cap is on the table, never on the findings

A 70B checkpoint has hundreds of tensors and a mixture-of-experts one has tens
of thousands. So the row list is capped and sorted largest first — and a cap
that quietly cut the list would read as "this is every tensor", which is why
`tensors_dropped`, `dropped_elements` and `dropped_bytes` always travel.

The cap also has a failure mode of its own worth naming: dropping the small
rows is exactly how you would hide the one broken tensor, because a corrupted
bias vector is small. So any tensor whose scan found a NaN or an infinity is
kept regardless of its size, and the totals — parameter counts, byte counts,
the per-dtype breakdown, the number of unhealthy tensors — are computed over
EVERY tensor and are complete even when the rows are not.

## What a health scan costs, before it is spent

Reading every element of a 70B model is 70 billion reads. Measured on
Qwen3-1.7B (2,031,739,904 elements across 311 tensors, bf16, on CPU):
an exhaustive scan reads all of them in 15.6 seconds, at 130 million elements
a second. A 70B model at that rate is about nine minutes.

So there is one number the caller controls, and it is per tensor: an
`allowance`. A tensor at or under the allowance is read COMPLETELY. A tensor
over it is read at a stride, and `scanned`, `stride` and `complete` all travel
so the reader knows which happened and over how many elements. The allowance
is also divided down by the run budget when there are many tensors, so that
every tensor gets looked at rather than the first few getting looked at
thoroughly and the rest not at all — a scan that covered the first 40 layers
and stopped would have a coverage bias nobody asked for.

`scan_cost` prices the whole thing from a list of element counts, with no
torch and no model, so a caller can see the number before spending it.

## The header path, for a model too big to load

`table_from_safetensors` reads the same table out of a safetensors header
without loading anything — the format is a length prefix and a block of JSON,
so this touches a few kilobytes of a file that may be forty gigabytes. It is
the same trick `imaging.read_tensor_names` uses, and it gives an EXACT table:
names, shapes, dtypes, element counts and byte spans, all declared by the file.

It gives no health at all. Not zero NaN — none, `health=None`, with the reason
"the file was never opened past its header". The numbers were not read, so
nothing here has an opinion about them, and this is the case the whole
"zero versus unchecked" rule was written for.

One caveat travels with it: the header names safetensors dtypes (`BF16`,
`F32`) and a live module names torch ones (`bfloat16`, `float32`). They are
not translated into each other, because a translation table is a thing that
can be wrong; `dtype_naming` says which vocabulary a table is written in so
nobody diffs the two per-dtype breakdowns by string.

## Tied weights are counted once and said out loud

`named_parameters()` de-duplicates by default, which means a model whose
`lm_head.weight` is its embedding silently loses one of the two names. This
walks with `remove_duplicate=False` and reports both rows, marking the second
with the name of the first and leaving it out of every total. A reader who
sees one row is not told about the tie; a reader who sees two identical rows
concludes the model has twice the parameters it has. Both rows, one total, and
`shared_tensors` counting them, is the only version of that which is true.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from .errors import BadRequest, Refusal
from .fmt import ordinal as _ordinal

# How many rows a table carries by default. Large enough that every tensor of
# an ordinary dense model fits -- Qwen3-1.7B has 311 -- and small enough that
# a mixture-of-experts checkpoint with tens of thousands does not arrive as one
# response. What the cap dropped is always reported, and an unhealthy tensor is
# never dropped by it.
MAX_ROWS = 512

# Elements one tensor may be read for, by default. At or under this a tensor is
# read completely; over it, at a stride. 1,048,576 is one mebi-element, which
# costs about 8 ms at the 130M elements/second measured on Qwen3-1.7B, so a
# 300-tensor model prices at roughly 2.4 seconds of health scanning.
SAMPLE_ELEMENTS = 1 << 20

# The whole run's element budget, which divides evenly across the tensors so
# that coverage does not tail off through the model. 256 mebi-elements is
# about two seconds at the measured rate.
SCAN_BUDGET = 1 << 28

# Below this, an allowance is not a sample of anything. A model with so many
# tensors that the budget divides to fewer than this per tensor is refused
# with the arithmetic in the message rather than scanned uselessly.
MIN_ALLOWANCE = 1024

# The largest working buffer a scan materialises at once. Every scan converts
# to float64 to count and to accumulate -- exact for every float dtype torch
# has, including the 8-bit ones -- and 1,048,576 float64 elements is 8 MB. The
# tensor itself is never copied; only this window is.
CHUNK_ELEMENTS = 1 << 20

# How many unhealthy tensor names the summary reads out. Every one of them is
# still counted in `unhealthy`; this caps the sentence, not the finding.
NAMED_UNHEALTHY = 8


class NotMeasured(Refusal):
    """Nothing here read a number, and the message says what to change.

    Its own class because the caller's correct response is specific to this
    module: not "load something else" but "point this at a model, raise the
    budget, or accept that a header carries no values".
    """


def _whole(name: str, value, *, minimum: int) -> int:
    """An int, with `True` refused rather than silently meaning 1.

    `isinstance(True, int)` is True, so `limit=True` would arrive here as a
    table of exactly one row and nothing would have said so. Every integer
    knob in this module goes through here.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise BadRequest(
            f"`{name}` must be a whole number and got {type(value).__name__}. "
            f"A boolean is not accepted even though Python counts it as an "
            f"int: `{name}=True` would quietly mean `{name}=1`."
        )
    if value < minimum:
        raise BadRequest(f"`{name}` must be at least {minimum:,} and got {value:,}.")
    return value


# --------------------------------------------------------------- the pricing
#
# Pure arithmetic over element counts. No torch, no model, no file -- so the
# cost of a scan can be shown to somebody before it is spent, and so the
# planning can be tested on a machine with no accelerator.


def plan_scan(elements: int, allowance: int) -> tuple[int, int]:
    """How many elements one tensor's scan reads, and at what stride.

    A tensor at or under the allowance is read completely, at stride 1. Over
    it, the stride is the smallest that brings the count within the allowance,
    so the sample is spread across the whole tensor rather than taken off the
    front — the first 1M elements of an embedding matrix are the first few
    hundred rows of the vocabulary, which is a question about those tokens and
    not about the tensor.

    Returns `(scanned, stride)`. `scanned` is what a strided view really holds,
    `ceil(elements / stride)`, rather than the allowance — those differ (a
    1.5M-element tensor at stride 2 reads 750,000, not 1,048,576) and the
    reported number has to be the one that was read.
    """
    elements = _whole("elements", elements, minimum=0)
    allowance = _whole("allowance", allowance, minimum=1)
    if elements == 0:
        return 0, 1
    if elements <= allowance:
        return elements, 1
    stride = max(2, math.ceil(elements / allowance))
    return (elements + stride - 1) // stride, stride


def allowance_for(
    n_tensors: int,
    *,
    per_tensor_elements: int = SAMPLE_ELEMENTS,
    max_scan_elements: int = SCAN_BUDGET,
) -> int:
    """The per-tensor element allowance, given how many tensors there are.

    An EVEN share, deliberately. The obvious alternative — scan in model order
    until the budget runs out — reads every element of the embedding and the
    first few blocks and never looks at the last forty layers, and then reports
    "no NaN found" about a model most of which nobody opened. Dividing the
    budget means the coverage is uniform and the sentence describing it is true
    of the whole model.
    """
    n_tensors = _whole("n_tensors", n_tensors, minimum=0)
    per_tensor_elements = _whole(
        "per_tensor_elements", per_tensor_elements, minimum=MIN_ALLOWANCE
    )
    max_scan_elements = _whole(
        "max_scan_elements", max_scan_elements, minimum=MIN_ALLOWANCE
    )
    if n_tensors <= 0:
        return per_tensor_elements
    share = max_scan_elements // n_tensors
    if share < MIN_ALLOWANCE:
        raise NotMeasured(
            f"this model has {n_tensors:,} tensors and a budget of "
            f"{max_scan_elements:,} elements, which divides to {share:,} "
            f"elements each — below the {MIN_ALLOWANCE:,} at which a sample "
            f"stops describing anything. Raise `max_scan_elements` to at least "
            f"{n_tensors * MIN_ALLOWANCE:,}, or ask for the table without a "
            f"health scan."
        )
    return min(per_tensor_elements, share)


def scan_cost(
    element_counts,
    *,
    per_tensor_elements: int = SAMPLE_ELEMENTS,
    max_scan_elements: int = SCAN_BUDGET,
    exhaustive: bool = False,
) -> dict:
    """What a health scan would read, before it reads anything.

    `element_counts` is one number per tensor, which `table()` can produce
    without touching a single weight and a safetensors header gives away for
    free. So the price of the expensive half is knowable from the cheap half.

    `exhaustive=True` prices reading every element of every tensor, which is
    the only way `all_finite` comes back `True` for a large tensor.
    """
    counts = [_whole("element count", c, minimum=0) for c in element_counts]
    total = sum(counts)
    if exhaustive:
        allowance = max(counts) if counts else 0
        scanned = total
        stride_by_tensor = [1] * len(counts)
    else:
        allowance = allowance_for(
            len(counts),
            per_tensor_elements=per_tensor_elements,
            max_scan_elements=max_scan_elements,
        )
        planned = [plan_scan(c, allowance) for c in counts]
        scanned = sum(s for s, _ in planned)
        stride_by_tensor = [st for _, st in planned]

    full = sum(1 for st in stride_by_tensor if st == 1)
    sampled = len(counts) - full
    # Rate measured on Qwen3-1.7B, CPU, float64 accumulation: 2,031,739,904
    # elements in 15.6 s. Reported as the measurement it is, with the machine
    # named, because a laptop and a server do not share it -- and it is the
    # only way a caller can turn "reads 268M elements" into "takes 2 seconds".
    return {
        "tensors": len(counts),
        "elements": total,
        "allowance": allowance,
        "scanned": scanned,
        "fully_scanned": full,
        "sampled": sampled,
        "share": round(scanned / total, 6) if total else 0.0,
        "exhaustive": bool(exhaustive),
        "means": (
            f"A health scan of these {len(counts):,} tensors would read "
            f"{scanned:,} of their {total:,} elements"
            + (
                " — every one of them, so a 'no NaN' answer would cover the "
                "whole model."
                if scanned >= total
                else f", {scanned / total * 100:,.1f}% of the model, with "
                f"{sampled:,} tensor(s) read at a stride. A 'no NaN' answer "
                f"from a strided read covers the elements that were read and "
                f"nothing else."
            )
            + f" At the 130 million elements per second measured on Qwen3-1.7B "
            f"on CPU that is about {scanned / 130_000_000:,.1f} seconds; your "
            f"machine is not that machine, so treat it as an order of "
            f"magnitude rather than a promise."
        ),
    }


# ------------------------------------------------------------------- health


@dataclass
class Health:
    """What was found in a tensor's numbers, and over how many of them.

    Only ever constructed when elements were actually read. A tensor nobody
    looked at has `health=None` and a reason, never one of these with zeros
    in it — that is the single rule this whole module is arranged around.
    """

    elements: int
    scanned: int
    stride: int
    nan: int
    pos_inf: int
    neg_inf: int
    zeros: int
    finite: int
    # Over the FINITE part only, and `None` when there is no finite part.
    # A tensor that is entirely NaN has no minimum; 0.0 would be a number some
    # other tensor really has.
    #
    # Never rounded. A bf16 weight of 1e-12 rounded to nine places is 0.0, and
    # a module whose job is to notice numbers going wrong must not be the thing
    # that turns one into zero.
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    # Population, over the elements that were read, and `None` below two of
    # them: a spread over one number is not a spread.
    std: float | None = None
    # True for integer and boolean dtypes, which have no bit pattern for NaN or
    # infinity. Their `all_finite` is then true by CONSTRUCTION rather than by
    # counting, and the sentence says so — the two are different evidence.
    nonfinite_impossible: bool = False

    @property
    def complete(self) -> bool:
        return self.scanned >= self.elements

    @property
    def nonfinite(self) -> int:
        return self.nan + self.pos_inf + self.neg_inf

    @property
    def all_finite(self) -> bool | None:
        """`True`, `False`, or `None` for "read part of it and saw none"."""
        if self.nonfinite:
            # A positive finding survives sampling: one NaN seen is one NaN
            # present, whatever fraction of the tensor was read.
            return False
        if self.nonfinite_impossible or self.complete:
            return True
        return None

    @property
    def all_zero(self) -> bool | None:
        """Is every element exactly zero? `None` when only part was read.

        A dead layer and an uninitialised buffer both look like this, and both
        are worth noticing — but "every element I read was zero" over one
        element in a thousand is not "every element is zero".
        """
        if self.scanned <= 0:
            return None
        if self.zeros < self.scanned:
            return False
        return True if self.complete else None

    def to_dict(self) -> dict:
        return {
            "elements": self.elements,
            "scanned": self.scanned,
            "stride": self.stride,
            "complete": self.complete,
            "nan": self.nan,
            "pos_inf": self.pos_inf,
            "neg_inf": self.neg_inf,
            "nonfinite": self.nonfinite,
            "zeros": self.zeros,
            "finite": self.finite,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.mean,
            "std": self.std,
            "all_finite": self.all_finite,
            "all_zero": self.all_zero,
            "nonfinite_impossible": self.nonfinite_impossible,
            "means": self.means(),
        }

    def means(self) -> str:
        coverage = (
            f"Every one of its {self.elements:,} elements was read."
            if self.complete
            else (
                f"{self.scanned:,} of its {self.elements:,} elements were read, "
                f"every {_ordinal(self.stride)} one. EVERY COUNT BELOW IS A COUNT OVER "
                f"THAT SAMPLE, not over the tensor."
            )
        )
        counts = (
            f"{self.nan:,} NaN, {self.pos_inf:,} +Inf, {self.neg_inf:,} -Inf, "
            f"{self.finite:,} finite, of which {self.zeros:,} are exactly zero."
        )
        if self.finite and self.minimum is not None and self.maximum is not None:
            spread = (
                f" The finite part runs {self.minimum:.6g} to "
                f"{self.maximum:.6g}, mean {self.mean:.6g}"
                + (
                    f", standard deviation {self.std:.6g}."
                    if self.std is not None
                    else ", and one finite element has no standard deviation."
                )
            )
        else:
            spread = (
                " There is no finite part at all, so the minimum, maximum, "
                "mean and standard deviation are None rather than 0 — those "
                "are different answers, and 0.0 is a value some tensor really "
                "holds."
            )

        if self.all_finite is False:
            verdict = (
                f" THIS TENSOR HOLDS {self.nonfinite:,} NON-FINITE VALUE(S). "
                f"Anything computed through it is NaN from here on, and a "
                f"forward pass will not say so."
            )
        elif self.nonfinite_impossible:
            verdict = (
                " Its dtype has no bit pattern for NaN or infinity, so it "
                "holds neither by construction rather than because a scan "
                "went looking."
            )
        elif self.all_finite:
            verdict = " No NaN and no infinity anywhere in it."
        else:
            verdict = (
                " No NaN and no infinity IN THE SAMPLE — which is not the same "
                "claim as none being there, and this reports it as unproven "
                "rather than clean. Scan it exhaustively to settle it."
            )
        zeroed = (
            " Every element read is zero, and the whole tensor was read: this "
            "tensor is entirely zero."
            if self.all_zero
            else ""
        )
        return f"{coverage} {counts}{spread}{verdict}{zeroed}"


# ---------------------------------------------------------------- the table


@dataclass
class TensorRow:
    """One tensor: what it is, how big, and — separately — how it is doing."""

    name: str
    module: str = ""
    leaf: str = ""
    # The class of the module that owns it, read from `named_modules()`. This
    # is the nearest thing here to Netron's op type, and it is read rather than
    # inferred from the name: `imaging.py` was written after `vla.py` had to be
    # corrected for deciding things from name prefixes.
    module_type: str = ""
    kind: str = "parameter"
    shape: list[int] = field(default_factory=list)
    dtype: str = ""
    elements: int = 0
    # `None` would mean nothing measured it. Both sources here can measure it,
    # so it is an int in practice — but the type says what an absence would
    # mean rather than letting it arrive as 0.
    bytes: int | None = None
    trainable: bool | None = None
    device: str = ""
    # The earlier row this tensor shares storage with. Non-empty means this row
    # is an alias and contributes nothing to any total.
    shared_with: str = ""
    health: Health | None = None
    # Non-empty exactly when `health` is None. "Unscanned with no reason" is
    # indistinguishable from "scanned and clean", which is the confusion this
    # whole file is built to prevent.
    health_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "module": self.module,
            "leaf": self.leaf,
            "module_type": self.module_type,
            "kind": self.kind,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "elements": self.elements,
            "bytes": self.bytes,
            "trainable": self.trainable,
            "device": self.device,
            "shared_with": self.shared_with,
            "health": self.health.to_dict() if self.health else None,
            "health_reason": self.health_reason,
            "means": self.means(),
        }

    def means(self) -> str:
        size = (
            f"{self.bytes / 1e6:,.2f} MB"
            if self.bytes is not None
            else "an unmeasured number of bytes"
        )
        where = f" in {self.module_type}" if self.module_type else ""
        shape = "x".join(str(d) for d in self.shape) or "scalar"
        head = (
            f"`{self.name}` is a {self.dtype} {self.kind}{where}, shape "
            f"{shape}, {self.elements:,} elements, {size}, on {self.device}."
        )
        if self.shared_with:
            head += (
                f" It is the SAME STORAGE as `{self.shared_with}` — a tied "
                f"weight, counted once in every total on this table."
            )
        if self.health is not None:
            return f"{head} {self.health.means()}"
        return (
            f"{head} ITS NUMBERS WERE NOT READ: {self.health_reason} That is "
            f"not a NaN count of zero; nothing here has an opinion about what "
            f"is in it."
        )


@dataclass
class WeightTable:
    """The rows that fit, and every total, over every tensor there was."""

    source: str = ""
    # Which vocabulary `dtype` is written in: "torch" or "safetensors". They
    # are not translated into each other, so this says which one a reader is
    # looking at rather than implying the two are comparable.
    dtype_naming: str = "torch"
    rows: list[TensorRow] = field(default_factory=list)

    # Discovery, over EVERY tensor rather than over the rows that fit.
    tensors_total: int = 0
    tensors_unique: int = 0
    shared_tensors: int = 0
    parameters: int = 0
    buffers: int = 0
    meta_tensors: int = 0

    # The cap.
    limit: int = MAX_ROWS
    tensors_shown: int = 0
    tensors_dropped: int = 0
    dropped_elements: int = 0
    dropped_bytes: int = 0

    # Totals, over unique tensors only, so a tied weight is counted once.
    elements_total: int = 0
    bytes_total: int = 0
    trainable_elements: int = 0
    frozen_elements: int = 0
    # The third category, and it exists because the other two do not add up
    # without it. A buffer is not a parameter with `requires_grad=False` -- it
    # is not a parameter at all -- so `rows_from_module` records `trainable`
    # as None for one, deliberately.
    #
    # Without this field a reader summing trainable + frozen gets a number
    # SHORT of `elements_total` with nothing on the page saying why, and a
    # BatchNorm-heavy model can hide a real share of its weights in that gap.
    # `test_the_three_categories_account_for_every_element` holds the identity.
    buffer_elements: int = 0
    by_dtype: dict = field(default_factory=dict)

    # Health. `health_checked` False means every row's `health` is None for one
    # reason — nobody asked — and no count below has been taken.
    health_checked: bool = False
    health_mode: str = ""
    allowance: int = 0
    tensors_scanned: int = 0
    tensors_unscanned: int = 0
    elements_scanned: int = 0
    elements_scannable: int = 0
    nan_total: int = 0
    pos_inf_total: int = 0
    neg_inf_total: int = 0
    zeros_total: int = 0
    unhealthy: int = 0
    unhealthy_names: list[str] = field(default_factory=list)
    all_zero_tensors: int = 0
    unproven: int = 0

    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "dtype_naming": self.dtype_naming,
            "rows": [r.to_dict() for r in self.rows],
            "tensors_total": self.tensors_total,
            "tensors_unique": self.tensors_unique,
            "shared_tensors": self.shared_tensors,
            "parameters": self.parameters,
            "buffers": self.buffers,
            "meta_tensors": self.meta_tensors,
            "limit": self.limit,
            "tensors_shown": self.tensors_shown,
            "tensors_dropped": self.tensors_dropped,
            "dropped_elements": self.dropped_elements,
            "dropped_bytes": self.dropped_bytes,
            "elements_total": self.elements_total,
            "bytes_total": self.bytes_total,
            "trainable_elements": self.trainable_elements,
            "frozen_elements": self.frozen_elements,
            "buffer_elements": self.buffer_elements,
            "by_dtype": {k: dict(v) for k, v in self.by_dtype.items()},
            "health_checked": self.health_checked,
            "health_mode": self.health_mode,
            "allowance": self.allowance,
            "tensors_scanned": self.tensors_scanned,
            "tensors_unscanned": self.tensors_unscanned,
            "elements_scanned": self.elements_scanned,
            "elements_scannable": self.elements_scannable,
            "nan_total": self.nan_total,
            "pos_inf_total": self.pos_inf_total,
            "neg_inf_total": self.neg_inf_total,
            "zeros_total": self.zeros_total,
            "unhealthy": self.unhealthy,
            "unhealthy_names": list(self.unhealthy_names),
            "all_zero_tensors": self.all_zero_tensors,
            "unproven": self.unproven,
            "notes": list(self.notes),
            "means": self.means(),
        }

    def means(self) -> str:
        parts = [
            f"{self.tensors_total:,} tensors in {self.source or 'this model'}: "
            f"{self.parameters:,} parameters and {self.buffers:,} buffers, "
            f"{self.elements_total:,} elements and "
            f"{self.bytes_total / 1e9:,.3f} GB in total. Buffers are counted "
            f"and listed separately from parameters because a rotary cache is "
            f"not a weight, and a NaN in one is as fatal as a NaN in the other."
        ]
        if self.shared_tensors:
            parts.append(
                f"{self.shared_tensors} tensor(s) are TIED — the same storage "
                f"under a second name, listed as their own rows and counted "
                f"once in every total above. Summing the rows yourself would "
                f"double them."
            )
        if self.meta_tensors:
            parts.append(
                f"{self.meta_tensors} tensor(s) are on the `meta` device: they "
                f"have a shape and a dtype and no values at all. Their bytes "
                f"are what they WOULD occupy once materialised, and nothing "
                f"here has read a number out of them."
            )
        if self.tensors_dropped:
            parts.append(
                f"THIS TABLE SHOWS {self.tensors_shown:,} OF "
                f"{self.tensors_total:,} TENSORS. The {self.tensors_dropped:,} "
                f"not shown are the smallest ones and account for "
                f"{self.dropped_elements:,} elements and "
                f"{self.dropped_bytes / 1e6:,.1f} MB. Every total on this "
                f"table is over all of them; only the rows were cut."
            )
        if self.by_dtype:
            widest = sorted(self.by_dtype.items(), key=lambda kv: -kv[1]["elements"])[0]
            parts.append(
                f"{len(self.by_dtype)} dtype(s) present, written in the "
                f"{self.dtype_naming} vocabulary; most of the model is "
                f"{widest[0]} ({widest[1]['elements']:,} elements). A "
                f"safetensors header says `BF16` where a loaded module says "
                f"`bfloat16` and nothing here translates between them, so do "
                f"not diff two breakdowns written in different vocabularies."
            )

        if not self.health_checked:
            parts.append(
                "NO HEALTH SCAN WAS RUN. Nothing on this table says how many "
                "NaN or infinite values any of these tensors holds — not zero, "
                "not few, nothing. It is a table of shapes and sizes, and the "
                "numbers inside were never read."
            )
            if self.notes:
                parts.append(" ".join(self.notes))
            return " ".join(parts)

        coverage = (
            f"{self.tensors_scanned:,} tensor(s) were read, "
            f"{self.elements_scanned:,} of their {self.elements_scannable:,} "
            f"elements, at an allowance of {self.allowance:,} elements each."
        )
        if self.health_mode == "full":
            coverage += " Every one was read completely."
        elif self.health_mode == "sampled":
            coverage += (
                " Every one was read at a stride, so no count below is a count "
                "over a whole tensor."
            )
        else:
            coverage += (
                " Some were read completely and some at a stride; each row "
                "says which, and the two cannot be pooled."
            )
        parts.append(coverage)

        if self.tensors_unscanned:
            parts.append(
                f"{self.tensors_unscanned:,} tensor(s) were NOT read at all "
                f"and carry no counts. Each says why in its own row — a tied "
                f"alias, an empty tensor, a complex one, or one on the `meta` "
                f"device with no values to read."
            )

        if self.unhealthy:
            named = ", ".join(f"`{n}`" for n in self.unhealthy_names)
            more = (
                f" and {self.unhealthy - len(self.unhealthy_names):,} more"
                if self.unhealthy > len(self.unhealthy_names)
                else ""
            )
            parts.append(
                f"{self.unhealthy:,} TENSOR(S) HOLD NON-FINITE VALUES: "
                f"{self.nan_total:,} NaN, {self.pos_inf_total:,} +Inf and "
                f"{self.neg_inf_total:,} -Inf across everything read. They are "
                f"{named}{more}. Every one of them is in the rows above "
                f"whatever its size, because dropping a small broken tensor to "
                f"fit a row cap is exactly how this table would hide the thing "
                f"it exists to find."
            )
        elif self.unproven:
            parts.append(
                f"No NaN and no infinity were found — but {self.unproven:,} "
                f"tensor(s) were only sampled, so for those this is UNPROVEN "
                f"rather than clean. Run it exhaustively to turn that into an "
                f"answer."
            )
        else:
            parts.append(
                "No NaN and no infinity anywhere, and every tensor that could "
                "hold one was read in full."
            )

        if self.all_zero_tensors:
            parts.append(
                f"{self.all_zero_tensors:,} tensor(s) are entirely zero, read "
                f"in full. That is normal for a freshly initialised bias and "
                f"is a dead layer anywhere else; this counts them and does not "
                f"decide which."
            )
        if self.notes:
            parts.append(" ".join(self.notes))
        return " ".join(parts)


def split_name(name: str) -> tuple[str, str]:
    """`model.layers.0.q_proj.weight` -> `model.layers.0.q_proj`, `weight`.

    Structural: the owning module is everything before the last dot, which is
    true of every `nn.Module` because that is how `named_parameters` builds the
    string. Nothing here matches on what the components are called.
    """
    module, _, leaf = name.rpartition(".")
    return module, leaf or name


# --------------------------------------------------------------- the scanner


def scan_tensor(
    tensor, *, allowance: int = SAMPLE_ELEMENTS
) -> tuple[Health | None, str]:
    """Count what is in one tensor. Returns `(health, reason_it_was_not)`.

    Exactly one of the two is meaningful: a `Health` with real counts, or
    `None` with a sentence. There is no third state and no zeroed `Health`.

    Reads in windows of `CHUNK_ELEMENTS` so a 300-million-element embedding is
    never converted whole. The tensor itself is not copied — `reshape(-1)` is a
    view of any contiguous tensor, which every `nn.Parameter` is unless
    somebody transposed one in place, and a strided slice of that view is
    another view. Only the float64 window is allocated, at 8 MB.
    """
    import torch

    allowance = _whole("allowance", allowance, minimum=1)

    if tensor.device.type == "meta":
        return None, (
            "it is on the `meta` device, which carries a shape and a dtype and "
            "no values. There is nothing to read until it is materialised."
        )
    if tensor.is_complex():
        return None, (
            "it is complex, and a complex number has no ordering — there is no "
            "minimum or maximum to report — so this reader does not scan it "
            "rather than reporting half a measurement."
        )
    total = int(tensor.numel())
    if total == 0:
        return None, (
            "it has no elements, so there is nothing to count. A row of zeros "
            "here would read as a tensor that was checked and found clean."
        )

    flat = tensor.detach().reshape(-1)
    _, stride = plan_scan(total, allowance)
    selected = flat if stride == 1 else flat[::stride]
    scanned = int(selected.numel())

    floating = bool(tensor.is_floating_point())
    nan = pos_inf = neg_inf = zeros = finite = 0
    lo: float | None = None
    hi: float | None = None
    # Chan's parallel merge of (count, mean, sum of squared deviations), so the
    # standard deviation is one pass and numerically stable. Accumulating a raw
    # sum of squares instead loses most of its digits on weights whose mean is
    # far from zero, which is exactly the case somebody is investigating.
    agg_n, agg_mean, agg_m2 = 0, 0.0, 0.0

    with torch.no_grad():
        for start in range(0, scanned, CHUNK_ELEMENTS):
            block = selected[start : start + CHUNK_ELEMENTS]
            # float64 for every dtype: exact for f8/f16/bf16/f32/f64 and for
            # every integer width up to 2^53, and it means one code path counts
            # NaN rather than one path per dtype family.
            work = block.double()
            if floating:
                nan_mask = torch.isnan(work)
                inf_mask = torch.isinf(work)
                nan += int(nan_mask.sum())
                pos_inf += int((inf_mask & (work > 0)).sum())
                neg_inf += int((inf_mask & (work < 0)).sum())
                values = work[~(nan_mask | inf_mask)]
            else:
                # An integer or boolean dtype has no bit pattern for NaN or
                # infinity. Not "none were found" — none can exist, which is a
                # stronger statement and is recorded as a different one.
                values = work
            count = int(values.numel())
            finite += count
            if not count:
                continue
            zeros += int((values == 0).sum())
            block_lo = float(values.min())
            block_hi = float(values.max())
            lo = block_lo if lo is None else min(lo, block_lo)
            hi = block_hi if hi is None else max(hi, block_hi)
            var, mean = torch.var_mean(values, correction=0)
            b_n, b_mean, b_m2 = count, float(mean), float(var) * count
            merged = agg_n + b_n
            delta = b_mean - agg_mean
            agg_mean += delta * b_n / merged
            agg_m2 += b_m2 + delta * delta * agg_n * b_n / merged
            agg_n = merged

    return (
        Health(
            elements=total,
            scanned=scanned,
            stride=stride,
            nan=nan,
            pos_inf=pos_inf,
            neg_inf=neg_inf,
            zeros=zeros,
            finite=finite,
            minimum=lo,
            maximum=hi,
            mean=agg_mean if agg_n else None,
            # Population, over what was read. `None` below two elements because
            # a spread over one number is not a spread — and 0.0 there would
            # read as "this tensor is constant", which is a real finding this
            # must not manufacture.
            std=math.sqrt(agg_m2 / agg_n) if agg_n >= 2 else None,
            nonfinite_impossible=not floating,
        ),
        "",
    )


# ---------------------------------------------------------- the live module


def rows_from_module(model, *, include_buffers: bool = True) -> list[tuple]:
    """`[(TensorRow, tensor), ...]` for every parameter and buffer. No reads.

    Walks with `remove_duplicate=False` so a tied weight appears under both of
    its names; the second is marked `shared_with` the first and is left out of
    every total. torch's default hides the second name entirely, which means a
    table built on it cannot tell a reader that `lm_head.weight` IS the
    embedding — and that is a fact about the model, not a detail.
    """
    named = getattr(model, "named_parameters", None)
    if not callable(named):
        raise BadRequest(
            "this needs an object with `named_parameters()` — an `nn.Module`. "
            "What arrived has no such method, so there is no tensor list to "
            "read. A path to a checkpoint goes to `table_from_safetensors` "
            "instead, which reads the header without loading anything."
        )

    types: dict[str, str] = {}
    modules = getattr(model, "named_modules", None)
    if callable(modules):
        types = {name: type(m).__name__ for name, m in modules()}

    out: list[tuple] = []
    seen: dict[tuple, str] = {}

    def add(name: str, tensor, kind: str, trainable: bool | None) -> None:
        module, leaf = split_name(name)
        # Storage identity first, object identity as the fallback. A `meta`
        # tensor's `data_ptr()` is 0 for every one of them, so keying on the
        # pointer alone would call an entire meta-initialised model one tied
        # weight -- which is what `remove_duplicate` itself would have done.
        pointer = 0
        try:
            pointer = int(tensor.data_ptr())
        except (RuntimeError, AttributeError):
            pointer = 0
        key = ("ptr", tensor.device.type, pointer) if pointer else ("id", id(tensor))
        out.append(
            (
                TensorRow(
                    name=name,
                    module=module,
                    leaf=leaf,
                    module_type=types.get(module, ""),
                    kind=kind,
                    shape=[int(d) for d in tensor.shape],
                    dtype=str(tensor.dtype).removeprefix("torch."),
                    elements=int(tensor.numel()),
                    bytes=int(tensor.numel()) * int(tensor.element_size()),
                    trainable=trainable,
                    device=str(tensor.device),
                    shared_with=seen.get(key, ""),
                ),
                tensor,
            )
        )
        seen.setdefault(key, name)

    for name, param in named(remove_duplicate=False):
        add(name, param, "parameter", bool(param.requires_grad))
    if include_buffers:
        buffers = getattr(model, "named_buffers", None)
        if callable(buffers):
            for name, buf in buffers(remove_duplicate=False):
                # `trainable=None`, not False. A buffer is not a parameter that
                # happens to be frozen; the question does not apply to it, and
                # False would put it in the "frozen weights" column.
                add(name, buf, "buffer", None)
    return out


def table(
    model,
    *,
    limit: int = MAX_ROWS,
    include_buffers: bool = True,
    health: bool = False,
    per_tensor_elements: int = SAMPLE_ELEMENTS,
    max_scan_elements: int = SCAN_BUDGET,
    exhaustive: bool = False,
    source: str = "",
) -> WeightTable:
    """The per-tensor table for a loaded `nn.Module`, optionally with health.

    `health=False` by default, and that default is the honest one: the table
    half is free and the health half reads every element it is allowed to. The
    cost is knowable first — `scan_cost([r.elements for r, _ in
    rows_from_module(model)])` prices it without reading a weight.

    `exhaustive=True` reads every element of every tensor and ignores both
    budgets. It is the only setting under which `all_finite` can come back
    `True` for a tensor larger than the allowance, and it is opt-in because on
    a 70B model it is several minutes of pure memory bandwidth.
    """
    limit = _whole("limit", limit, minimum=1)
    pairs = rows_from_module(model, include_buffers=include_buffers)
    if not pairs:
        raise NotMeasured(
            "this module has no parameters and no buffers, so there is no "
            "weight table to build. An `nn.Module` that holds no tensors is "
            "usually a wrapper — pass the model it wraps."
        )

    notes: list[str] = []
    allowance = 0
    if health:
        allowance = _scan_all(
            pairs,
            per_tensor_elements=per_tensor_elements,
            max_scan_elements=max_scan_elements,
            exhaustive=exhaustive,
        )
    else:
        for row, _ in pairs:
            row.health_reason = (
                "no health scan was requested for this run, so its numbers "
                "were never read."
            )

    return _assemble(
        [row for row, _ in pairs],
        source=source or type(model).__name__,
        dtype_naming="torch",
        limit=limit,
        health_checked=bool(health),
        allowance=allowance,
        notes=notes,
    )


def _scan_all(
    pairs: list[tuple],
    *,
    per_tensor_elements: int,
    max_scan_elements: int,
    exhaustive: bool,
) -> int:
    """Scan every scannable tensor, and say why for the ones that are not.

    The allowance is decided ONCE, from how many tensors are actually going to
    be read, so that it is the same for all of them. Deciding it per tensor as
    the budget drained would give the last layers a smaller sample than the
    first, and the coverage of the answer would depend on iteration order.
    """
    scannable = [
        (row, tensor)
        for row, tensor in pairs
        if not row.shared_with and row.elements and row.device != "meta"
    ]
    if exhaustive:
        allowance = max((row.elements for row, _ in scannable), default=MIN_ALLOWANCE)
    else:
        allowance = allowance_for(
            len(scannable),
            per_tensor_elements=per_tensor_elements,
            max_scan_elements=max_scan_elements,
        )

    for row, tensor in pairs:
        if row.shared_with:
            # Not re-read: it is the same storage, so the counts would be
            # identical by construction. Pointed at rather than copied, so no
            # reader can conclude two tensors were independently checked.
            row.health, row.health_reason = (
                None,
                (
                    f"it is the same storage as `{row.shared_with}`, which was "
                    f"read; its counts are on that row and were not taken twice."
                ),
            )
            continue
        row.health, row.health_reason = scan_tensor(tensor, allowance=allowance)
    return allowance


# --------------------------------------------------- the header, without a load


def rows_from_safetensors(path: str | Path) -> tuple[list[TensorRow], list[str]]:
    """Every tensor a safetensors file or directory DECLARES. Loads nothing.

    The header is a little-endian u64 length then that many bytes of JSON, so
    this touches a few kilobytes of a file that may be forty gigabytes — the
    same read `imaging.read_tensor_names` does, through `fit.read_header`,
    which is the one reader in this package for that format.

    Byte counts come from each tensor's `data_offsets` span, which is the
    payload the file actually holds. Not the file size, which also carries the
    header and any alignment padding, and not a parameter count multiplied by
    an assumed dtype width.
    """
    from . import fit

    p = Path(path)
    if p.is_dir():
        shards = sorted(p.glob("*.safetensors"))
        if not shards:
            raise NotMeasured(
                f"there is no .safetensors file in {p.name}, so there is no "
                f"tensor table to read. A pickle checkpoint (.bin, .pt) "
                f"carries its table only inside the pickle, which cannot be "
                f"read without running it — see weights_scan.py for why this "
                f"package will not."
            )
    elif p.is_file():
        shards = [p]
    else:
        raise NotMeasured(f"there is nothing at {p.name} to read a tensor table from.")

    rows: list[TensorRow] = []
    notes: list[str] = []
    seen: dict[str, str] = {}
    for shard in shards:
        try:
            header = fit.read_header(shard)
        except BadRequest as err:
            # `fit.read_header`'s messages are authored for a reader and name
            # only the file, so relaying one is relaying a sentence this
            # package wrote rather than a library's internals.
            raise NotMeasured(str(err)) from err  # leak-ok: authored by fit.py
        for name, spec in header.items():
            try:
                start, end = spec["data_offsets"]
                dtype = str(spec["dtype"])
                shape = [int(d) for d in spec["shape"]]
            except (KeyError, TypeError, ValueError) as err:
                raise NotMeasured(
                    f"{shard.name} describes a tensor called {name!r} without "
                    f"a readable dtype, shape and offset pair, so this table "
                    f"cannot say what is in the file. Refusing rather than "
                    f"listing it with the fields it did have."
                ) from err
            if name in seen:
                # Two shards declaring the same tensor name is a broken index,
                # and silently keeping the last one would produce a table that
                # is quietly about a file nobody has.
                notes.append(
                    f"`{name}` is declared in both {seen[name]} and "
                    f"{shard.name}; both rows are listed and the totals count "
                    f"both, because nothing here knows which one a loader "
                    f"would pick."
                )
            seen[name] = shard.name
            count = 1
            for dim in shape:
                count *= dim
            module, leaf = split_name(name)
            rows.append(
                TensorRow(
                    name=name,
                    module=module,
                    leaf=leaf,
                    kind="stored",
                    shape=shape,
                    dtype=dtype,
                    elements=count,
                    bytes=int(end) - int(start),
                    # Unknown, not False. A header says nothing about whether a
                    # tensor would be trained, and False would put every weight
                    # in the frozen column.
                    trainable=None,
                    device="disk",
                    health_reason=(
                        "the file was never opened past its header, so not one "
                        "of its numbers has been read."
                    ),
                )
            )
    if len(shards) > 1:
        notes.append(
            f"Read from {len(shards)} shards. The totals are over all of them."
        )
    return rows, notes


def table_from_safetensors(path: str | Path, *, limit: int = MAX_ROWS) -> WeightTable:
    """The table for a checkpoint too big to load. No health, and it says so.

    Everything here is exact — the file declares every shape, dtype and byte
    span — and nothing here is a health scan. `health=None` on every row with
    the reason attached, because the alternative is a table of zero NaN counts
    for a file nobody opened.
    """
    limit = _whole("limit", limit, minimum=1)
    rows, notes = rows_from_safetensors(path)
    if not rows:
        raise NotMeasured(
            f"{Path(path).name} has a readable header that declares no "
            f"tensors at all. That is a file with a table in it and nothing "
            f"in the table."
        )
    notes.append(
        "Read from the header alone: nothing was loaded, no element was read, "
        "and a tied weight is invisible here because a header records storage "
        "spans and not Python identity."
    )
    return _assemble(
        rows,
        source=str(Path(path).name),
        dtype_naming="safetensors",
        limit=limit,
        health_checked=False,
        allowance=0,
        notes=notes,
    )


# ------------------------------------------------------------- the assembly


def _assemble(
    rows: list[TensorRow],
    *,
    source: str,
    dtype_naming: str,
    limit: int,
    health_checked: bool,
    allowance: int,
    notes: list[str],
) -> WeightTable:
    """Totals over every row; the cap applied only to which rows are carried."""
    out = WeightTable(
        source=source,
        dtype_naming=dtype_naming,
        limit=limit,
        allowance=allowance,
        health_checked=health_checked,
        notes=list(notes),
    )
    out.tensors_total = len(rows)

    unique = [r for r in rows if not r.shared_with]
    out.tensors_unique = len(unique)
    out.shared_tensors = len(rows) - len(unique)
    out.parameters = sum(1 for r in rows if r.kind == "parameter")
    out.buffers = sum(1 for r in rows if r.kind == "buffer")
    out.meta_tensors = sum(1 for r in rows if r.device == "meta")

    for row in unique:
        out.elements_total += row.elements
        if row.bytes is not None:
            out.bytes_total += row.bytes
        if row.trainable is True:
            out.trainable_elements += row.elements
        elif row.trainable is False:
            out.frozen_elements += row.elements
        else:
            # `None` -- a buffer. Counted rather than dropped, so the three
            # categories sum to `elements_total` exactly.
            out.buffer_elements += row.elements
        bucket = out.by_dtype.setdefault(
            row.dtype, {"tensors": 0, "elements": 0, "bytes": 0}
        )
        bucket["tensors"] += 1
        bucket["elements"] += row.elements
        bucket["bytes"] += row.bytes or 0

    unhealthy_rows = []
    modes = set()
    for row in rows:
        if row.health is None:
            if health_checked:
                out.tensors_unscanned += 1
            continue
        out.tensors_scanned += 1
        out.elements_scanned += row.health.scanned
        out.elements_scannable += row.health.elements
        out.nan_total += row.health.nan
        out.pos_inf_total += row.health.pos_inf
        out.neg_inf_total += row.health.neg_inf
        out.zeros_total += row.health.zeros
        modes.add("full" if row.health.complete else "sampled")
        if row.health.all_finite is False:
            out.unhealthy += 1
            unhealthy_rows.append(row)
        elif row.health.all_finite is None:
            # Sampled, nothing found, nothing proven. Counted separately so the
            # summary can say "clean" only when it is entitled to.
            out.unproven += 1
        if row.health.all_zero:
            out.all_zero_tensors += 1
    if len(modes) == 1:
        out.health_mode = modes.pop()
    elif modes:
        out.health_mode = "mixed"
    out.unhealthy_names = [r.name for r in unhealthy_rows[:NAMED_UNHEALTHY]]

    # Findings first, then the largest. A cap that dropped the small rows would
    # drop a corrupted bias vector, which is small and is the entire point of
    # having looked.
    flagged = sorted(unhealthy_rows, key=lambda r: (-r.elements, r.name))
    # Identity, not equality. `TensorRow` is a dataclass, so two rows with the
    # same shape, dtype and health compare equal — and a model with two
    # identical bias vectors would have had one of them silently treated as the
    # other and dropped from the table by a membership test.
    flagged_ids = {id(r) for r in flagged}
    rest = sorted(
        (r for r in rows if id(r) not in flagged_ids),
        key=lambda r: (-r.elements, r.name),
    )
    kept = flagged[:limit]
    kept += rest[: max(0, limit - len(kept))]
    kept_ids = {id(r) for r in kept}
    dropped = [r for r in rows if id(r) not in kept_ids]

    out.rows = sorted(kept, key=lambda r: (-r.elements, r.name))
    out.tensors_shown = len(out.rows)
    out.tensors_dropped = len(dropped)
    out.dropped_elements = sum(r.elements for r in dropped if not r.shared_with)
    out.dropped_bytes = sum(r.bytes or 0 for r in dropped if not r.shared_with)
    if len(flagged) > limit:
        out.notes.append(
            f"{len(flagged) - limit:,} tensor(s) holding non-finite values did "
            f"not fit in a {limit:,}-row table and are not listed. They are "
            f"still counted in `unhealthy`, so the finding is complete even "
            f"where the rows are not."
        )
    return out
