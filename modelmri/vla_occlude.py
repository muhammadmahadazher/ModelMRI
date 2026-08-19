"""What the policy's vision actually depended on — not what it looked at.

The VLA panel paints attention, and by this project's own standard that is the
weak version. The field has already settled it: Embodied Interpretability
measured that interventional masking beats attention weights on explanation
fidelity, and VLA-Trace uses attention knockout rather than attention viewing.
ModelMRI ships the thing the literature moved away from.

So this occludes each block of the camera frame in turn, re-runs the tower, and
reports how far the representation moved — beside the attention map, with the
rank correlation between them printed for THIS frame. The two disagreeing on
your own checkpoint is the finding, and it is not visible from either map
alone.

WHAT THIS IS NOT, AND THE WORDING IS LOAD-BEARING
-------------------------------------------------
PERCEPTION ONLY. The score is a shift in the vision tower's pooled embedding.
It is not an effect on the action, because without the action expert there is
no action to affect — `vla.py` refuses that today and says why. Nothing here
may be labelled "caused the action", and the sentence this module returns says
so in those words rather than leaving it to a caption somebody might drop.

OCCLUSION IS OUT OF DISTRIBUTION. A grey box is itself a stimulus: the encoder
has never seen one, so part of any shift is the box rather than the missing
content. That is why TWO fill baselines ship rather than one, named on screen
the way `ablate.BASELINES` are — if a block is hot under both, the finding does
not rest on which grey you chose.

THE CONTROL IS A BLOCK SOMEWHERE ELSE. Not a same-norm random tensor: the
treatment here occludes an AREA, so the null has to be occluding an area of
the same size at a random location. Anything else compares an occlusion
against something that is not one.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field

from .errors import BadRequest

# Verbatim from patch.py, and for the same reason: a control is a measurement
# somebody else can repeat rather than a number that moves each refresh.
CONTROL_DRAWS = 8
CONTROL_SEED = 0

# A full 32x32 sweep is 1,024 tower passes. The default steps by 4, which is
# 64 blocks, and the fine grid is opt-in with the pass count on screen first.
DEFAULT_STRIDE = 4

# Above this the run is a job rather than a click, and the caller is told the
# number before it starts rather than after.
MAX_BLOCKS = 1_024

# How many frames of the episode set the scale. The score is in units of the
# embedding's own per-dimension spread across the episode, and a spread over
# one frame is zero.
SCALE_FRAMES = 8

# The two fills, named on screen. `episode_mean` is the average pixel of the
# frames sampled for the scale; `midpoint` is the tower's own normalisation
# centre, which is what a zero tensor means after `(x * 2 - 1)`.
BASELINES = ("episode_mean", "midpoint")


class OcclusionError(BadRequest):
    """This measurement cannot be taken honestly, and we say why."""


@dataclass
class Block:
    """One occluded region of the frame."""

    row: int
    col: int
    # Shift in the pooled embedding, in units of its own per-dimension spread
    # across the episode. Unitless on purpose: a raw L2 in embedding space is
    # a number whose size depends on the tower.
    shift: float
    # The strongest of the same-area occlusions drawn for THIS block at random
    # locations. None when this block was not among the strongest and so was
    # never controlled -- NOT 0.0.
    control_max: float | None = None
    # How many of those draws were usable. Not always `CONTROL_DRAWS`: a draw
    # that lands on the block itself is not a control, and `draws` is a
    # parameter. Whatever reports the result reads this rather than the
    # constant.
    control_draws: int = 0
    clears_control: bool | None = None
    # Mean attention received by the patches under this block, for the layer
    # the caller asked about. Carried so the two maps can be compared without
    # a second request.
    attention: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Occlusion:
    baseline: str
    grid: list[int]
    stride: int
    blocks: list[Block] = field(default_factory=list)
    n_blocks: int = 0
    n_controlled: int = 0
    passes: int = 0
    seconds: float = 0.0
    scale: float = 0.0
    scale_frames: int = 0
    # Spearman between this frame's causal map and its attention map. None
    # when no attention map was supplied — which is a state, not a zero.
    attention_agreement: float | None = None
    # WHICH attention map that was. The agreement is layer-dependent and can
    # change sign across the tower, so a bare Spearman with no layer beside it
    # is not a reportable number. None when nothing was compared.
    compared_layer: int | None = None
    compared_head: int | None = None
    # WHICH frame was occluded. None rather than 0: these were filled in after
    # the fact by the HTTP route and by nothing else, so every other caller --
    # `VLA.occlude` from Python, a test, a script -- got `episode 0, timestep
    # 0`, which is a real frame in every dataset and reads exactly like one.
    # They are set at the source now; None means nobody said.
    episode: int | None = None
    timestep: int | None = None
    camera: str = ""

    def to_dict(self) -> dict:
        out = asdict(self)
        out["means"] = self.means()
        return out

    def means(self) -> str:
        cleared = [b for b in self.blocks if b.clears_control]
        tested = [b for b in self.blocks if b.clears_control is not None]
        parts = [
            f"Each of {self.n_blocks} blocks of the camera frame was replaced "
            f"by the {self.baseline} fill and the vision tower re-run — "
            f"{self.passes} passes at stride {self.stride} on a "
            f"{self.grid[0]}x{self.grid[1]} patch grid. Scores are in units of "
            f"the tower's own embedding spread across {self.scale_frames} "
            f"frames of this episode."
        ]
        if not tested:
            parts.append(
                "NOTHING WAS TESTED AGAINST CHANCE on this run, so every "
                "number above is a shift with no null beside it."
            )
        elif not cleared:
            parts.append(
                f"NONE of the {len(tested)} strongest blocks beat its own "
                f"control: occluding the same area somewhere else moved the "
                f"representation as much or more. On this evidence no region "
                f"of this frame is distinguished from covering up that much "
                f"of it anywhere."
            )
        else:
            # The draws each block actually got, not the module's default.
            # `draws` is a parameter, and a draw that lands on the block
            # itself is not a control and is discarded -- so the count varies
            # per block and printing the constant claimed a null that was
            # never run.
            counts = sorted({b.control_draws for b in cleared})
            beat = (
                f"all {counts[0]} same-area occlusions at random locations"
                if len(counts) == 1
                else (
                    f"every same-area occlusion drawn for it — between "
                    f"{counts[0]} and {counts[-1]}, because a draw that landed "
                    f"on the block itself is not a control and was discarded"
                )
            )
            parts.append(
                f"{len(cleared)} of the {len(tested)} blocks tested against "
                f"chance beat {beat}."
            )
        if self.attention_agreement is None:
            parts.append(
                "No attention map was supplied, so the two cannot be compared "
                "on this frame."
            )
        else:
            agreement = self.attention_agreement
            where = (
                ""
                if self.compared_layer is None
                else f", against layer {self.compared_layer}"
                + (
                    " averaged over its heads"
                    if self.compared_head is None or self.compared_head < 0
                    else f" head {self.compared_head}"
                )
            )
            # Three readings, not two. A strong NEGATIVE rank correlation is a
            # relationship, not the absence of one: it says the blocks
            # attention ranked highest are the ones the representation depended
            # on least. Folding that in with "uncorrelated" would print the
            # same sentence for -0.9 and -0.05, which are opposite findings.
            if agreement > 0.6:
                reading = (
                    "They largely rank the same blocks, which is worth knowing "
                    "and is not the usual result."
                )
            elif agreement < -0.6:
                reading = (
                    "THE RANKINGS ARE INVERTED: the blocks this frame's "
                    "attention ranked highest are the ones its representation "
                    "depended on LEAST. That is a relationship, not an absence "
                    "of one, and it is a stronger claim than the two maps "
                    "merely disagreeing."
                )
            else:
                reading = (
                    "Where the model LOOKED and what its representation "
                    "DEPENDED ON are ranking the blocks differently — which is "
                    "the whole reason this measurement exists, and is not "
                    "visible from either map alone."
                )
            parts.append(
                f"THE TWO MAPS AGREE AT SPEARMAN {agreement:+.3f} on this "
                f"frame{where}. " + reading
            )
        parts.append(
            "PERCEPTION ONLY. This is a shift in the vision tower's pooled "
            "embedding, not an effect on the action: without the action expert "
            "there is no action to affect. It must not be read as 'this caused "
            "the robot to do that'."
        )
        parts.append(
            f"OCCLUSION IS OUT OF DISTRIBUTION — a {self.baseline} patch is "
            f"itself a stimulus the encoder has never seen, so part of every "
            f"shift is the patch rather than the missing content. Run the "
            f"other baseline and keep what survives both."
        )
        return " ".join(parts)


# ------------------------------------------------------------------ helpers


def plan(grid: list[int], stride: int, *, max_blocks: int = MAX_BLOCKS) -> list[tuple]:
    """Which blocks get occluded, in row-major order.

    Refused past the cap rather than truncated: a map missing its bottom half,
    presented as a map, is worse than a refusal that names the number.
    """
    if stride < 1:
        raise OcclusionError("stride must be at least 1 patch")
    rows = list(range(0, int(grid[0]), stride))
    cols = list(range(0, int(grid[1]), stride))
    total = len(rows) * len(cols)
    if total > max_blocks:
        raise OcclusionError(
            f"stride {stride} on a {grid[0]}x{grid[1]} grid is {total} blocks "
            f"and each is one tower pass. The cap is {max_blocks} — raise the "
            f"stride, or raise the cap knowing what it costs. A map cut short "
            f"would be missing regions and still look like a map."
        )
    return [(r, c) for r in rows for c in cols]


def _pooled(model, image, device):
    """The tower's pooled patch embedding for one image, on the CPU.

    Mean over PATCH tokens rather than the CLS token: not every tower has a
    CLS token, and one that does puts a different thing in it than another
    does. The mean is defined the same way everywhere.
    """
    import torch

    with torch.no_grad():
        out = model(pixel_values=image.to(device), output_attentions=False)
    hidden = getattr(out, "last_hidden_state", None)
    if hidden is None:
        raise OcclusionError(
            "this vision tower returned no hidden states, so there is no "
            "representation to measure a shift in."
        )
    return hidden[0].float().mean(dim=0).cpu()


def scale_from(embeddings: list) -> float:
    """The spread of the tower's own embeddings across the sampled frames.

    A raw L2 in embedding space is a number whose size depends on the tower —
    one model's 0.4 is another's 40 — and neither means anything on its own.
    Dividing by the spread the tower produces across frames of THIS episode
    makes the score "this many times the variation the model shows anyway".
    """
    import torch

    if len(embeddings) < 2:
        raise OcclusionError(
            f"the scale needs at least two frames and got {len(embeddings)}. "
            f"A spread measured over one frame is zero, and every score would "
            f"be a division by nothing."
        )
    stacked = torch.stack(embeddings)
    centred = stacked - stacked.mean(dim=0, keepdim=True)
    spread = float(torch.linalg.vector_norm(centred, dim=1).pow(2).mean().sqrt())
    if not math.isfinite(spread) or spread <= 0.0:
        raise OcclusionError(
            "this tower returned the same embedding for every sampled frame, "
            "so there is no spread to measure a shift against. The episode may "
            "be static, or the frames may all be decoding to the same picture "
            "— `modelmri audit` checks for exactly that."
        )
    return spread


def spearman(a: list[float], b: list[float]) -> float | None:
    """Rank correlation, so the two maps can be compared without a shared unit.

    Attention is a probability and a causal shift is in embedding-spread
    units; a Pearson correlation between them would be arithmetic across
    incompatible scales. Ranks have no units.
    """
    n = min(len(a), len(b))
    if n < 3:
        return None

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: values[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and values[order[j + 1]] == values[order[i]]:
                j += 1
            # Ties share the average rank. Without this an attention map with
            # many equal values would produce a correlation about the order
            # they happened to be stored in.
            shared = (i + j) / 2.0
            for k in range(i, j + 1):
                out[order[k]] = shared
            i = j + 1
        return out

    ra, rb = ranks(a[:n]), ranks(b[:n])
    ma, mb = sum(ra) / n, sum(rb) / n
    va = sum((v - ma) ** 2 for v in ra)
    vb = sum((v - mb) ** 2 for v in rb)
    if va <= 0 or vb <= 0:
        return None
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb, strict=True))
    return cov / math.sqrt(va * vb)


def _fill_value(baseline: str, images: list):
    """What goes in the occluded block, in the tower's normalised space.

    Two of them, named, because occlusion is out of distribution and the
    reader should be able to see which findings survive the choice.
    """
    import torch

    if baseline == "midpoint":
        # `analyse` maps [0,1] to [-1,1], so zero IS the tower's midpoint --
        # the "neutral grey" this normalisation was built around.
        return torch.tensor(0.0)
    if baseline == "episode_mean":
        if not images:
            raise OcclusionError("no frames were sampled to average")
        # Mean over the SPATIAL axes, giving one colour per channel: [1,3,1,1],
        # which broadcasts into any block. `dim=(1, 2)` averaged over channels
        # and height instead and produced [1,1,1,S] — a "colour" that varied
        # along the width and had no channel axis, which torch refused to
        # broadcast into the block. The midpoint fill is a scalar and hid it.
        return torch.stack(images).mean(dim=0).mean(dim=(-2, -1), keepdim=True)
    raise OcclusionError(
        f"unknown fill baseline {baseline!r} — expected one of {BASELINES}."
    )


# -------------------------------------------------------------------- sweep


def sweep(
    model,
    device,
    frame,
    *,
    grid: list[int],
    patch: int,
    scale_frames: list,
    baseline: str = "episode_mean",
    stride: int = DEFAULT_STRIDE,
    attention_map: list[list[float]] | None = None,
    compared_layer: int | None = None,
    compared_head: int | None = None,
    draws: int = CONTROL_DRAWS,
    max_controlled: int = 12,
    max_blocks: int = MAX_BLOCKS,
    episode: int | None = None,
    timestep: int | None = None,
    camera: str = "",
) -> Occlusion:
    """Occlude each block of one frame and report how far the tower moved.

    `frame` and `scale_frames` are already-normalised [1, 3, S, S] tensors —
    the same thing `vla.analyse` builds — so this module does no image
    handling of its own and cannot disagree with the panel about what the
    tower was shown.
    """
    import torch

    if baseline not in BASELINES:
        raise OcclusionError(
            f"unknown fill baseline {baseline!r} — expected one of {BASELINES}."
        )
    blocks = plan(grid, stride, max_blocks=max_blocks)

    started = time.perf_counter()
    passes = 0

    base_embeddings = []
    for image in scale_frames:
        base_embeddings.append(_pooled(model, image, device))
        passes += 1
    spread = scale_from(base_embeddings)

    here = _pooled(model, frame, device)
    passes += 1
    fill = _fill_value(baseline, scale_frames)

    def occluded(row: int, col: int):
        """The frame with one block replaced by the fill."""
        edited = frame.clone()
        y0, x0 = row * patch, col * patch
        y1 = min(int(frame.shape[-2]), (row + stride) * patch)
        x1 = min(int(frame.shape[-1]), (col + stride) * patch)
        edited[..., y0:y1, x0:x1] = fill
        return edited

    rows: list[Block] = []
    for row, col in blocks:
        moved = _pooled(model, occluded(row, col), device)
        passes += 1
        shift = float(torch.linalg.vector_norm(moved - here)) / spread
        attention = None
        if attention_map:
            attention = _attention_under(attention_map, row, col, stride)
        rows.append(Block(row=row, col=col, shift=round(shift, 6), attention=attention))

    # Rank agreement BEFORE the controls reorder anything: the comparison is
    # between the two maps as measured, and it is the headline of this whole
    # feature.
    agreement = None
    if attention_map:
        agreement = spearman(
            [b.shift for b in rows], [b.attention or 0.0 for b in rows]
        )
        if agreement is not None:
            agreement = round(agreement, 4)

    # Controls on the strongest blocks only. `draws` passes each, and a block
    # that did not place is not one anybody is about to call hot.
    ranked = sorted(rows, key=lambda b: -b.shift)
    generator = torch.Generator().manual_seed(CONTROL_SEED)
    n_rows, n_cols = (
        len(range(0, int(grid[0]), stride)),
        len(range(0, int(grid[1]), stride)),
    )
    for block in ranked[:max_controlled]:
        if n_rows * n_cols < 2:
            break
        worst = 0.0
        used = 0
        for _ in range(max(1, draws)):
            # A block of the SAME AREA somewhere else. The treatment occludes
            # an area, so the null has to occlude an area -- a same-norm
            # random tensor would be comparing an occlusion against something
            # that is not one.
            r = int(torch.randint(0, n_rows, (1,), generator=generator)) * stride
            c = int(torch.randint(0, n_cols, (1,), generator=generator)) * stride
            if (r, c) == (block.row, block.col):
                continue
            moved = _pooled(model, occluded(r, c), device)
            passes += 1
            used += 1
            worst = max(worst, float(torch.linalg.vector_norm(moved - here)) / spread)
        if used:
            block.control_max = round(worst, 6)
            block.control_draws = used
            block.clears_control = block.shift > worst

    return Occlusion(
        baseline=baseline,
        grid=[int(grid[0]), int(grid[1])],
        stride=stride,
        blocks=ranked,
        n_blocks=len(rows),
        n_controlled=sum(1 for b in rows if b.clears_control is not None),
        passes=passes,
        seconds=round(time.perf_counter() - started, 2),
        scale=round(spread, 6),
        scale_frames=len(base_embeddings),
        attention_agreement=agreement,
        # Only meaningful when something was actually compared; a layer index
        # beside a null agreement would read as "layer 11 agreed on nothing".
        compared_layer=compared_layer if agreement is not None else None,
        compared_head=compared_head if agreement is not None else None,
        episode=None if episode is None else int(episode),
        timestep=None if timestep is None else int(timestep),
        camera=str(camera or ""),
    )


def _attention_under(attention_map, row: int, col: int, stride: int) -> float:
    """Mean attention received by the patches this block covers."""
    total, n = 0.0, 0
    for r in range(row, min(row + stride, len(attention_map))):
        line = attention_map[r]
        for c in range(col, min(col + stride, len(line))):
            total += float(line[c])
            n += 1
    return round(total / n, 8) if n else 0.0


def estimate(
    grid: list[int],
    stride: int,
    *,
    controlled: int = 12,
    draws: int = CONTROL_DRAWS,
    scale_frames: int = SCALE_FRAMES,
) -> dict:
    """What the sweep will cost, before it runs.

    A 32x32 sweep at stride 1 is 1,024 tower passes plus the controls. Nobody
    should discover that by waiting.

    REFUSES WHAT `plan` REFUSES, and for the reason a preflight exists at all.
    MEASURED: `GET /api/vla/occlude/cost?stride=-1` answered 200 with
    `{"blocks": 1024, "passes": 1129, "stride": -1}` while
    `POST /api/vla/occlude {"stride": -1}` answered 422 "stride must be at
    least 1 patch". So the route whose whole job is pricing a run quoted a
    firm figure for one the very next click would refuse.

    And the figure did not describe what it priced: `max(1, stride)` clamped
    the arithmetic to stride 1 while the payload echoed the caller's -1, so
    "1,024 blocks at stride -1" was a price for a run nobody asked for. A
    caller does not reach a negative stride by accident and then want a
    number; they want to know it is not a stride.

    0 never arrives here — `VLAHandle.occlusion_cost` reads it as the
    query-string way of saying "not stated" and substitutes the default — so
    anything below 1 at this point is a value somebody chose.
    """
    if stride < 1:
        raise OcclusionError(
            f"a stride of {stride} is not a number of patches. It must be at "
            f"least 1 — the same rule the run itself enforces, so this refuses "
            f"rather than quoting a price for something the next click would "
            f"turn down."
        )
    blocks = len(range(0, int(grid[0]), stride)) * len(range(0, int(grid[1]), stride))
    return {
        "blocks": blocks,
        "passes": blocks + scale_frames + 1 + min(controlled, blocks) * draws,
        "stride": stride,
        "grid": [int(grid[0]), int(grid[1])],
    }
