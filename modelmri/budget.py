"""What will this analysis cost, before you pay for it?

Every ranking in this package is a loop of forward passes, and the loop knows
how many: `ablate.rank_heads` returns `passes`, `patch.trace` returns `passes`,
and both figures are exact. What none of them knew was what a pass costs *here*
— so the honest answer to "should I run the whole model?" was a shrug, and the
dishonest one was a number measured on the maintainer's RTX 4060 that ranged
between 12 and 71 ms/pass across sessions on that same card.

So this module measures one pass on this machine, right now, and multiplies.

**Time multiplies. Peak memory does not.** This is the one thing in here that
is easy to get confidently wrong. `n` sequential forward passes take about `n`
times as long, but they do not hold `n` times the memory: pass `k` frees before
pass `k+1` allocates, so the peak of the loop is the peak of one pass, plus
whatever the loop itself retains between iterations. `patch.trace` caches the
clean run's activations and genuinely does retain; `ablate.rank_heads` does not.
Callers say which they are — `retained_bytes` — rather than this module
guessing, because multiplying a peak by 132 would produce a refusal on every
analysis this tool offers, and a refusal that is always wrong is worse than no
guard at all.

**One pass is one sample.** The first analysis after a load pays CUDA warm-up
and runs several times slower than the rest (Qwen3-0.6B measured 3.05 s, then
0.80, then 0.78 for identical work). An estimate built from one probe is
labelled `basis="one probe pass on this machine"` and is never presented as a
property of the model. It is a better number than a figure from someone else's
card, and it is still a sample.

Measured end to end on gpt2 (bf16, cuda, RTX 4060, "The capital of France is",
the full 146-pass head sweep): the projection called 146 passes exactly, 4.90 s
against 4.46 s actual (1.10x), and 1.1 MB peak against 1.8 MB actual. The peak
runs low — per-iteration allocator churn is not in a single probe — so it is an
estimate to compare against a budget, not a bound to promise anybody. The
refusal threshold is set well under 1.0 partly for that reason.

**A memory reading this module cannot take says so.** CUDA reports both the
allocator's peak and the driver's free bytes. XPU reports the allocator's peak
and, on most builds, nothing about free memory. MPS has no per-region peak at
all on the versions this runs against, and CPU has no accelerator to ask. Those
paths return `None` with a `reason`, and every consumer here is written so that
`None` means "unknown" and never `0` — a budget check that reads unknown as
zero free memory refuses everything, and one that reads unknown as zero cost
approves everything. Both are the `.get(name, 0.0)` bug that made 206 robot
episodes show one video, wearing a different hat.

**The allocator's view is not the driver's.** `max_memory_allocated` counts
what PyTorch handed out, not what the card is holding — the caching allocator
reserves more and returns it slowly. Every number this module produces is
labelled `allocated by PyTorch`, and `free_bytes` comes from the driver via
`mem_get_info`, so the two sides of the comparison are named for what they are
rather than silently mixed.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from .errors import Refusal

# Above this share of the driver's free memory, an analysis is refused rather
# than started. Not 1.0: an allocation that exactly fills the card fails, and
# the eager attention buffer this tool forces is not in the probe's marginal
# reading when the sequence is short. Overridable, because "I will close
# something else" is a real answer and this is not the disk rule.
REFUSE_ABOVE_FRACTION = 0.9

# Below this, the accelerator question is nobody's emergency and the preflight
# only reports. Matches capacity.py's habit of not firing during ordinary work:
# a guard that interrupts a 40 ms job is a guard people learn to click through.
MIN_INTERESTING_SECONDS = 2.0


class TooCostly(Refusal):
    """Refused before the loop started, with both numbers named.

    A `Refusal` rather than its own root: the caller understood the request
    perfectly and this is a deliberate "no" with a sentence for the reader,
    which is exactly what errors.py reserves 409 for. `overridable` carries
    the same meaning as `capacity.TooBig.overridable` — this one always is,
    because unlike free disk, free VRAM can be made by closing something.
    """

    def __init__(self, message: str, *, overridable: bool = True) -> None:
        super().__init__(message)
        self.overridable = overridable


@dataclass
class Memory:
    """What the accelerator would tell us, or why it would not.

    `peak_bytes` is the allocator's high-water mark for the probed region;
    `free_bytes` and `total_bytes` come from the driver where one answers.
    Any of them may be `None`, and `None` is not zero — see the module
    docstring. `reason` is filled exactly when something is `None`.
    """

    peak_bytes: int | None = None
    free_bytes: int | None = None
    total_bytes: int | None = None
    source: str = ""
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Probe:
    """One real forward pass, timed and measured on this machine."""

    seconds: float
    memory: Memory
    device_kind: str

    def to_dict(self) -> dict:
        return {
            "seconds": round(self.seconds, 4),
            "memory": self.memory.to_dict(),
            "device_kind": self.device_kind,
        }


@dataclass
class Estimate:
    """The projected cost of `passes` passes, and what it is built from."""

    passes: int
    seconds: float | None
    peak_bytes: int | None
    retained_bytes: int
    free_bytes: int | None
    fraction_of_free: float | None
    verdict: str  # ok | tight | refuse | unknown
    basis: str
    unmeasured: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.seconds is not None:
            d["seconds"] = round(self.seconds, 2)
        if self.fraction_of_free is not None:
            d["fraction_of_free"] = round(self.fraction_of_free, 3)
        return d


# --------------------------------------------------------------- reading memory


def _torch():
    import torch

    return torch


def _backend(device_kind: str):
    """The torch submodule that answers about this accelerator, or None.

    `rocm` is deliberately mapped onto `torch.cuda`: ROCm reports itself
    through the cuda API surface with `torch.version.hip` set, which is the
    same thing `devices._cuda_like` relies on.
    """
    torch = _torch()
    if device_kind in ("cuda", "rocm"):
        return getattr(torch, "cuda", None)
    if device_kind == "xpu":
        return getattr(torch, "xpu", None)
    if device_kind == "mps":
        return getattr(torch, "mps", None)
    return None


def reset_peak(device_kind: str) -> bool:
    """Zero the allocator's high-water mark. True if it took.

    Returns rather than raises: a machine that cannot reset its counter can
    still be timed, and losing the whole preflight over the memory half would
    trade a useful number for none.
    """
    mod = _backend(device_kind)
    fn = getattr(mod, "reset_peak_memory_stats", None) if mod else None
    if fn is None:
        return False
    try:
        fn()
        return True
    except Exception:
        # Driver states this code cannot enumerate. The caller reads False and
        # reports the memory half as unmeasured, which is the honest outcome.
        return False


def _peak_allocated(device_kind: str) -> int | None:
    mod = _backend(device_kind)
    fn = getattr(mod, "max_memory_allocated", None) if mod else None
    if fn is None:
        return None
    try:
        return int(fn())
    except Exception:
        return None


def _allocated_now(device_kind: str) -> int | None:
    mod = _backend(device_kind)
    for name in ("memory_allocated", "current_allocated_memory"):
        fn = getattr(mod, name, None) if mod else None
        if fn is None:
            continue
        try:
            return int(fn())
        except Exception:
            return None
    return None


def _synchronize(device_kind: str) -> None:
    """Make the timing mean something.

    CUDA and XPU launches are asynchronous: without this, `perf_counter`
    measures how long it took to *queue* the work, which on a warm queue is
    close to zero and would make every projection wildly optimistic.
    """
    mod = _backend(device_kind)
    fn = getattr(mod, "synchronize", None) if mod else None
    if fn is None:
        return
    try:
        fn()
    except Exception:  # noqa: S110 - a sync that fails leaves the timing
        # slightly optimistic, which the caller already treats as a floor. It
        # is not a reason to refuse to estimate.
        pass


def free_memory(device_kind: str) -> Memory:
    """Driver-level free/total for this accelerator, or the reason we cannot.

    Only CUDA/ROCm answer this reliably through torch. `torch.xpu.mem_get_info`
    exists on newer builds and is tried; MPS reports a recommended ceiling
    rather than free memory, which is a different quantity and is not
    substituted for one.
    """
    torch = _torch()

    if device_kind in ("cuda", "rocm"):
        try:
            free, total = torch.cuda.mem_get_info()
            return Memory(
                free_bytes=int(free), total_bytes=int(total), source="torch.cuda"
            )
        except Exception as err:
            return Memory(
                source="torch.cuda",
                reason=f"the driver did not answer mem_get_info ({type(err).__name__})",
            )

    if device_kind == "xpu":
        fn = getattr(getattr(torch, "xpu", None), "mem_get_info", None)
        if fn is not None:
            try:
                free, total = fn()
                return Memory(
                    free_bytes=int(free), total_bytes=int(total), source="torch.xpu"
                )
            except Exception:  # noqa: S110 - the Memory below IS the handling
                pass
        return Memory(
            source="torch.xpu",
            reason="this torch build's XPU backend does not report free memory",
        )

    if device_kind == "mps":
        # `recommended_max_memory` is a ceiling Apple suggests, not free space,
        # and unified memory means the GPU is competing with everything else on
        # the machine for it. Reporting it as `free_bytes` would put a number
        # in a field that means something else.
        return Memory(
            source="torch.mps",
            reason=(
                "Apple Silicon shares one pool with the whole system, so there "
                "is no free-VRAM figure to read"
            ),
        )

    return Memory(reason="no accelerator — there is no VRAM budget to check")


# ------------------------------------------------------------------- the probe


def probe_pass(run, device_kind: str) -> Probe:
    """Run `run()` once, timed, with the allocator's peak for that region.

    `run` must do one iteration of **the loop being projected**, not a bare
    forward pass — hooks installed, softmax taken, everything the real body
    does. This is the easiest thing here to get wrong and it fails quietly.
    Measured on gpt2 (bf16, cuda, "The capital of France is", 146-pass sweep):
    probing `model(ids)` predicted a 0.7 MB peak against 1.8 MB actually used,
    because `ablate._cut` clones the projection's input and `distribution()`
    upcasts the logits to fp32 — neither of which a bare pass does. Probing a
    real iteration predicted 1.1 MB against the same 1.8 MB. A probe that does
    less work than the loop it projects is not a cheap estimate, it is a wrong
    one, and it is wrong in the direction that approves an analysis which then
    runs the card out of memory.

    The real token ids, too, not a synthetic shape: the cost of a pass depends
    on sequence length, so a probe over a different length measures a different
    thing.

    The peak reported is MARGINAL: what the pass needed on top of what was
    already resident. Model weights are already allocated by the time anything
    calls this, and billing them to the analysis would make every estimate look
    like it needed the whole card.
    """
    before = _allocated_now(device_kind)
    could_reset = reset_peak(device_kind)

    _synchronize(device_kind)
    started = time.perf_counter()
    run()
    _synchronize(device_kind)
    seconds = time.perf_counter() - started

    memory = free_memory(device_kind)
    if not could_reset:
        memory.reason = memory.reason or (
            "this backend does not expose the allocator's peak, so the memory "
            "cost of a pass could not be measured"
        )
        return Probe(seconds=seconds, memory=memory, device_kind=device_kind)

    peak = _peak_allocated(device_kind)
    if peak is None or before is None:
        memory.reason = memory.reason or (
            "the allocator did not report a peak for this region"
        )
        return Probe(seconds=seconds, memory=memory, device_kind=device_kind)

    # max(…, 0): a pass that frees more than it takes reads negative against a
    # `before` captured a moment earlier, and a negative cost is not a smaller
    # cost — it is no measurable cost.
    memory.peak_bytes = max(int(peak) - int(before), 0)
    memory.source = memory.source or f"torch.{device_kind}"
    return Probe(seconds=seconds, memory=memory, device_kind=device_kind)


def project(probe: Probe, passes: int, *, retained_bytes: int = 0) -> Estimate:
    """Scale one measured pass up to `passes` of them.

    Time multiplies; peak does not. `retained_bytes` is for loops that hold
    something across iterations (patch.trace caches the clean activations);
    pass 0 for loops that do not (ablate.rank_heads). See the module docstring
    — this distinction is the whole reason the parameter exists rather than
    being inferred.
    """
    if passes < 1:
        raise ValueError("passes must be at least 1")

    seconds = probe.seconds * passes
    mem = probe.memory
    notes: list[str] = []

    peak = None
    fraction = None
    verdict = "unknown"

    if mem.peak_bytes is not None:
        peak = mem.peak_bytes + retained_bytes
        if retained_bytes:
            notes.append(
                "includes what this analysis holds across passes, on top of "
                "one pass's own cost"
            )
        if mem.free_bytes:
            fraction = peak / mem.free_bytes
            verdict = (
                "refuse"
                if fraction > REFUSE_ABOVE_FRACTION
                else "tight"
                if fraction > 0.6
                else "ok"
            )
        else:
            notes.append(
                "the cost is known but the free-memory figure is not, so this "
                "is not checked against a budget"
            )
    elif mem.free_bytes:
        notes.append("free memory is known but the cost of a pass is not")

    if seconds < MIN_INTERESTING_SECONDS and verdict == "unknown":
        notes.append("under two seconds — not worth asking about")

    return Estimate(
        passes=passes,
        seconds=seconds,
        peak_bytes=peak,
        retained_bytes=retained_bytes,
        free_bytes=mem.free_bytes,
        fraction_of_free=fraction,
        verdict=verdict,
        basis="one probe pass on this machine",
        unmeasured=mem.reason,
        notes=notes,
    )


def _gb(n: int | float) -> str:
    return f"{n / 1e9:,.1f} GB"


def check(estimate: Estimate, *, label: str, confirm: bool = False) -> Estimate:
    """Raise `TooCostly` when the projection says this will not fit.

    Returns the estimate otherwise, so a caller can log or return it. Silent on
    `verdict="unknown"` by design: a guard that fires when it could not take
    the measurement is a guard that blocks CPU users out of ignorance.
    """
    if estimate.verdict != "refuse" or confirm:
        return estimate

    assert estimate.peak_bytes is not None and estimate.free_bytes is not None
    raise TooCostly(
        f"{label} would need about {_gb(estimate.peak_bytes)} on top of the "
        f"model, and there is {_gb(estimate.free_bytes)} free on this "
        f"accelerator. That is {estimate.fraction_of_free:.0%} of what is "
        f"left, measured from one probe pass, so it would probably run out "
        f"partway through. Close something on the GPU, use a shorter prompt, "
        f"or run it anyway if you know the estimate is pessimistic.",
        overridable=True,
    )
