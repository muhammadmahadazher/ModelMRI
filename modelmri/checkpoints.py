"""Where two robot checkpoints diverge, over identical frames.

rollout-doctor warns when the harness, config, task set or sample size differ
between two runs, and TRI STEP does sequential A/B. Both treat the policy as
opaque, so neither can say WHERE the finetune changed the model. VLA-Trace did
CKA-style representation drift and shipped no code.

This runs both checkpoints over the same frames of the same episodes and
reports the per-layer representation distance between their vision towers.

    Named `checkpoints.py` and not `compare.py` because `tests/test_compare.py`
    already owns that name, and two modules a grep apart is how the wrong one
    gets edited.

WHAT IT SAYS AND WHAT IT DOES NOT
---------------------------------
DESCRIPTIVE, NOT CAUSAL. It says where two towers differ on your frames. It
does not say which is better, and there is no version of this measurement that
does — a checkpoint that diverges more might be the one that learned the task.
Nothing here may imply a winner.

THE BEHAVIOUR HALF IS ABSENT AND SAYS SO. Predicted-action distance per frame
needs the action expert, which is a sidecar this does not have. The report
names which half ran rather than leaving a reader to assume both did.

TWO LOADS, ONE AT A TIME. 8 GB will not hold two policies, and the ones worth
comparing are the ones near that limit. A is loaded, run over every frame,
released; then B. The per-frame activations live on the CPU in between, which
is the whole reason the sequencing works.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field

from .errors import BadRequest

# Frames per comparison. Each costs one tower pass per side, and the loads
# dominate — so this is about the table and the wait rather than memory.
MAX_FRAMES = 512

DEFAULT_FRAME_STRIDE = 25

# The fields that must match for a per-layer table to line up. A mismatch in
# any of them is refused BY NAME: "the checkpoints are incompatible" sends the
# reader to check both, and "image_size 512 against 384" tells them which one
# they did not mean to pick.
COMPATIBILITY = (
    ("image_size", "the frames are resized differently before the tower sees them"),
    ("patch_size", "the patch grids do not line up"),
    ("num_hidden_layers", "a per-layer table would compare layer 3 with layer 3"),
    ("hidden_size", "there is no cosine between vectors of different lengths"),
)


class CheckpointError(BadRequest):
    """These two checkpoints cannot be compared honestly, and we say why."""


@dataclass
class LayerDistance:
    layer: int
    # Centred kernel alignment: 1.0 means the two layers represent these
    # frames the same way up to rotation and scale, 0.0 means unrelated.
    cka: float
    # Mean cosine between the two towers' pooled outputs, frame by frame.
    # Reported BESIDE the CKA rather than instead of it: they disagree when
    # one tower has rotated its basis without changing what it encodes, and
    # that disagreement is informative.
    cosine: float
    n_frames: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Comparison:
    checkpoint_a: str
    checkpoint_b: str
    dataset: str
    camera: str
    layers: list[LayerDistance] = field(default_factory=list)
    n_frames: int = 0
    n_episodes: int = 0
    frame_stride: int = DEFAULT_FRAME_STRIDE
    seconds: float = 0.0
    # Which halves ran. The behaviour half needs the action expert.
    ran_perception: bool = True
    ran_behaviour: bool = False
    behaviour_absent_because: str = ""

    def to_dict(self) -> dict:
        out = asdict(self)
        out["means"] = self.means()
        return out

    @property
    def most_divergent(self) -> LayerDistance | None:
        """The layer with the LOWEST alignment. Rarely the interesting one."""
        return min(self.layers, key=lambda r: r.cka) if self.layers else None

    @property
    def first_divergent(self) -> LayerDistance | None:
        """Where the alignment first FALLS — no threshold involved.

        `most_divergent` is the lowest CKA, and in a chain that is almost
        always the last layer: once two towers come apart they stay apart and
        the gap compounds. MEASURED on a 6-block tower with drift planted at
        block 3: CKA reads 1.0000 through layer 3, then 0.9273, 0.8831,
        0.8452 — so "most divergent" says layer 6 and the answer is layer 4.
        Where it STARTS is the question somebody is asking.

        Shares `model_diff.steepest_drop` rather than re-deriving it: it is
        the same question about the same shape of curve, and two copies would
        drift.
        """
        if len(self.layers) < 2:
            return None
        from .model_diff import steepest_drop

        index, _ = steepest_drop([r.cka for r in self.layers])
        return self.layers[index] if index is not None else None

    def means(self) -> str:
        parts = [
            f"{self.checkpoint_a} against {self.checkpoint_b} over "
            f"{self.n_frames} frames of {self.n_episodes} episodes, every "
            f"{self.frame_stride}th frame, through the {self.camera} camera "
            f"of {self.dataset}. Both towers saw identical frames."
        ]
        first = self.first_divergent
        worst = self.most_divergent
        if first is not None:
            parts.append(
                f"They first come apart at layer {first.layer} — CKA "
                f"{first.cka:.4f}, mean cosine {first.cosine:+.4f}. That is "
                f"where the alignment FALLS, not where it is lowest: once two "
                f"towers diverge they stay diverged and the gap compounds, so "
                f"the lowest CKA is almost always the last layer and almost "
                f"never the answer"
                + (
                    f" — here it is layer {worst.layer} at {worst.cka:.4f}."
                    if worst is not None and worst.layer != first.layer
                    else "."
                )
            )
            parts.append(
                "BOTH numbers are printed because they disagree when one "
                "tower has rotated its basis without changing what it "
                "encodes: a low cosine beside a high CKA is a rotation, and a "
                "low CKA is a different representation."
            )
        elif worst is not None:
            parts.append(
                f"The alignment never falls between layers — it is "
                f"{worst.cka:.4f} at its lowest. These two towers represent "
                f"your frames the same way all the way through."
            )
        parts.append(
            "THIS SAYS WHERE THEY DIFFER, NOT WHICH IS BETTER. There is no "
            "version of this measurement that says which is better — a "
            "checkpoint that diverges more might be the one that learned the "
            "task."
        )
        if self.ran_behaviour:
            parts.append("Both the perception and behaviour halves ran.")
        else:
            # The reason is optional; the STATEMENT is not. A dangling colon
            # with nothing after it reads as a truncated message rather than
            # as a half that deliberately did not run.
            why = (
                f" {self.behaviour_absent_because.strip()}"
                if self.behaviour_absent_because.strip()
                else ""
            )
            parts.append(
                f"THE PERCEPTION HALF RAN AND THE BEHAVIOUR HALF DID NOT.{why} "
                f"So this compares what the two towers SEE and says nothing "
                f"about what either would DO."
            )
        return " ".join(parts)


# ------------------------------------------------------- the compatibility gate


def check_compatible(a: dict, b: dict, label_a: str, label_b: str) -> None:
    """Refuse a pair whose per-layer table would compare the wrong things.

    NAMES THE FIELD, both values, and why it matters. A refusal that says only
    "incompatible" sends the reader to diff two configs by hand.
    """
    for field_name, why in COMPATIBILITY:
        left, right = a.get(field_name), b.get(field_name)
        if left is None or right is None:
            raise CheckpointError(
                f"{label_a if left is None else label_b} does not state its "
                f"`{field_name}`, and comparing without it would be guessing "
                f"at whether these two are comparable at all."
            )
        if left != right:
            raise CheckpointError(
                f"`{field_name}` is {left} in {label_a} and {right} in "
                f"{label_b} — {why}."
            )


def check_cameras(a: list, b: list, label_a: str, label_b: str) -> None:
    """Both checkpoints must have been shown the same camera keys.

    Separate from the numeric fields because the failure reads differently:
    two policies trained on different camera sets are not a configuration
    mismatch, they are two policies that were never asked the same question.
    """
    if set(a) != set(b):
        only_a = sorted(set(a) - set(b))
        only_b = sorted(set(b) - set(a))
        bits = []
        if only_a:
            bits.append(f"{label_a} has {only_a}")
        if only_b:
            bits.append(f"{label_b} has {only_b}")
        raise CheckpointError(
            "these checkpoints were trained on different cameras — "
            + " and ".join(bits)
            + ". They were never asked the same question, so a per-layer "
            "comparison would not be one."
        )


# ------------------------------------------------------------- the distances


def cka(x, y) -> float:
    """Linear centred kernel alignment between two activation matrices.

    Invariant to rotation and to isotropic scaling, which is exactly what is
    wanted: two towers can encode the same thing in different bases, and a
    raw distance would call that a difference. `[n_frames, d]` each; `d` need
    not match, which is why this and not a plain correlation.
    """
    import torch

    if x.shape[0] != y.shape[0]:
        raise CheckpointError(
            f"the two sides have {x.shape[0]} and {y.shape[0]} frames — they "
            f"must be the same frames in the same order."
        )
    if x.shape[0] < 2:
        raise CheckpointError(
            "CKA over one frame is undefined: centring leaves nothing behind."
        )
    a = (x - x.mean(dim=0, keepdim=True)).double()
    b = (y - y.mean(dim=0, keepdim=True)).double()
    cross = float(torch.linalg.matrix_norm(b.T @ a) ** 2)
    left = float(torch.linalg.matrix_norm(a.T @ a))
    right = float(torch.linalg.matrix_norm(b.T @ b))
    if left <= 0 or right <= 0:
        # One side is constant across every frame. Not a similarity of zero —
        # there is no direction to align with.
        raise CheckpointError(
            "one of these towers returned the same activation for every "
            "frame, so there is nothing for the other to be aligned with."
        )
    return cross / (left * right)


def mean_cosine(x, y) -> float:
    """Mean per-frame cosine, defined only when the widths match."""
    import torch

    if x.shape[1] != y.shape[1]:
        # Not an error and not zero: CKA handles different widths and this
        # cannot, so the caller gets a NaN-free signal that this half is
        # simply not defined here.
        return float("nan")
    sim = torch.nn.functional.cosine_similarity(x.double(), y.double(), dim=1)
    return float(sim.mean())


# ------------------------------------------------------------------- capture


def pooled_layers(model, image, device) -> list:
    """Every layer's pooled patch embedding for one frame, on the CPU.

    On the CPU deliberately: the caller holds these while the other checkpoint
    loads, and that sequencing is the only reason two policies can be compared
    on a machine that fits one.
    """
    import torch

    with torch.no_grad():
        out = model(pixel_values=image.to(device), output_hidden_states=True)
    states = getattr(out, "hidden_states", None)
    if not states:
        raise CheckpointError(
            "this vision tower returned no hidden states, so there is nothing "
            "to compare layer by layer."
        )
    return [h[0].float().mean(dim=0).cpu() for h in states]


def plan(
    reader,
    *,
    frame_stride: int = DEFAULT_FRAME_STRIDE,
    max_frames: int = MAX_FRAMES,
) -> list[tuple[int, int]]:
    """Which frames both checkpoints see. Identical for both, by construction."""
    if frame_stride < 1:
        raise CheckpointError("the frame stride must be at least 1")
    episodes = reader.episodes()
    if not episodes:
        raise CheckpointError("this dataset has no episodes to compare over")
    pairs = [
        (ep.index, t)
        for ep in episodes
        for t in range(0, int(ep.length), frame_stride)
    ]
    if len(pairs) > max_frames:
        raise CheckpointError(
            f"that is {len(pairs):,} frames and each is a tower pass on BOTH "
            f"sides. The cap is {max_frames:,} — raise the stride rather than "
            f"having the set cut short, because a comparison over a silently "
            f"trimmed frame set is not the comparison you asked for."
        )
    return pairs


def compare(
    load_side,
    checkpoint_a: str,
    checkpoint_b: str,
    reader,
    *,
    frame_stride: int = DEFAULT_FRAME_STRIDE,
    max_frames: int = MAX_FRAMES,
    on_stage=None,
    behaviour_absent_because: str = (
        "the action expert needs the optional lerobot extra, which is not "
        "installed."
    ),
) -> Comparison:
    """Load each side once, run identical frames, release, compare.

    `load_side(spec)` returns `(model, device, config, cameras, release)` and
    is supplied by the caller, so this module holds no load policy and can be
    tested without a checkpoint.
    """
    import torch

    if checkpoint_a == checkpoint_b:
        raise CheckpointError(
            "both sides are the same checkpoint, so every distance would be "
            "zero by construction."
        )
    frames = plan(reader, frame_stride=frame_stride, max_frames=max_frames)
    started = time.perf_counter()

    captured: dict[str, list] = {}
    configs: dict[str, dict] = {}
    cameras: dict[str, list] = {}
    for spec in (checkpoint_a, checkpoint_b):
        if on_stage:
            on_stage("load", spec)
        model, device, config, cams, release = load_side(spec)
        try:
            configs[spec] = config
            cameras[spec] = list(cams or [])
            if len(configs) == 2:
                # BEFORE the second side's frames are run, so a mismatched
                # pair costs one load rather than two full sweeps.
                check_compatible(
                    configs[checkpoint_a], configs[checkpoint_b],
                    checkpoint_a, checkpoint_b,
                )
                check_cameras(
                    cameras[checkpoint_a], cameras[checkpoint_b],
                    checkpoint_a, checkpoint_b,
                )
            rows = []
            for index, (episode, timestep) in enumerate(frames):
                if on_stage:
                    on_stage("frame", f"{spec} · {index + 1}/{len(frames)}")
                rows.append(
                    pooled_layers(model, reader.frame_tensor(episode, timestep), device)
                )
            captured[spec] = rows
        finally:
            # In a `finally`: a capture that raises must still give the memory
            # back, or the second side has nowhere to load into and the real
            # error is buried under an out-of-memory.
            release()

    rows_a, rows_b = captured[checkpoint_a], captured[checkpoint_b]
    n_layers = min(len(rows_a[0]), len(rows_b[0]))
    layers: list[LayerDistance] = []
    for layer in range(n_layers):
        x = torch.stack([r[layer] for r in rows_a])
        y = torch.stack([r[layer] for r in rows_b])
        layers.append(
            LayerDistance(
                layer=layer,
                cka=round(cka(x, y), 6),
                cosine=round(mean_cosine(x, y), 6),
                n_frames=int(x.shape[0]),
            )
        )

    return Comparison(
        checkpoint_a=checkpoint_a,
        checkpoint_b=checkpoint_b,
        dataset=getattr(reader, "repo_id", ""),
        camera=getattr(reader, "camera", ""),
        layers=layers,
        n_frames=len(frames),
        n_episodes=len({e for e, _ in frames}),
        frame_stride=frame_stride,
        seconds=round(time.perf_counter() - started, 2),
        ran_perception=True,
        ran_behaviour=False,
        behaviour_absent_because=behaviour_absent_because,
    )
