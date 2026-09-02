"""Moving a model onto an accelerator, with a meter on it.

Split out of `runtime.py` to break the package's one import cycle, and the
cycle is worth describing because every edge in it was deliberate.

`runtime` reaches for `model_diff`, `gguf_load` and `behavdiff`; `model_diff`
reaches for `behavdiff`; `behavdiff` reaches for `gguf_load`. All forward, all
deferred inside the functions that need them. Then `gguf_load` reached back for
`runtime.move_to_device`, and that ONE EDGE closed the loop — CodeQL reported
`py/cyclic-import` on all seven of those imports, correctly: a deferred import
does not stop a cycle existing, it stops it mattering at load time.

The honest fix was not to suppress seven notes. It was to notice that
`move_to_device` never needed `runtime` at all — it takes a model and a device
and publishes bytes, and its only dependencies are `progress` and the clock. A
leaf module is where it belonged, and with it here the graph is a DAG.

`runtime` imports `move_to_device` because it calls it. It does NOT re-export
`resident_bytes` or `DEVICE_PUBLISH_EVERY_S`: both were re-exported at first to
spare their callers an edit, and both then had no reader in `runtime` at all —
a writer with no reader, which is the defect this project names most often, and
which CodeQL reported as `py/unused-import`. Their one caller is
`tests/test_device_move.py`, and it names this module directly.

That test documents the trap at its own patch site, and it is the reason the
publish interval must be patched here: rebinding it on `runtime` would rebind a
name this module never reads, and the test would go on passing while checking
less.
"""

from __future__ import annotations

import time
from itertools import chain

from . import progress

# How often the device move publishes. A move is a few hundred modules, so
# publishing on every one would take the tracker's lock far more often than
# anything polls it; publishing on none of them is the bug this exists to fix.
DEVICE_PUBLISH_EVERY_S = 0.25


def resident_bytes(model) -> int:
    """Bytes of parameters and buffers `model` holds, or 0 for "cannot say".

    `parameters()` and `buffers()` de-duplicate by default, so a model that
    ties its embedding to its output head is counted once — which is what the
    move actually costs, and what the denominator has to be for the bar to be
    able to reach the end.

    ZERO IS UNKNOWN, NOT EMPTY, and one unsizable tensor makes the whole
    answer unknown. A partial sum would be worse than no number: the bar would
    stop short of its own end on every checkpoint that wraps its weights in
    something other than a plain tensor, and a meter that never finishes is
    the exact report this whole change came from. The caller moves the model
    without a meter instead, which is the module rule for progress — a load
    must never fail because the thing measuring it did.
    """
    total = 0
    try:
        for tensor in chain(model.parameters(), model.buffers()):
            total += tensor.numel() * tensor.element_size()
    except Exception:
        # Deliberately not narrowed, and deliberately around the whole walk
        # rather than around one multiplication. A model can decline this at
        # either end — a quantised or wrapped parameter with no
        # `element_size`, or an object with no `buffers()` at all — and every
        # one of those is a reason to skip the meter rather than to fail the
        # load.
        return 0
    return total


def move_to_device(model, device) -> int:
    """Move `model` onto `device`, publishing how much has landed as it goes.
    Returns the bytes moved.

    `model.to(device)` is one opaque call that can run for minutes, and it was
    the only step of a load with no meter at all. Reported from the browser as
    21 minutes sitting on "Moving to the accelerator" — under a bar drawn full
    at "2.5 GB / 2.5 GB · 0 bytes left · ~0s left", which was the finished
    download's figure still being published into the same snapshot. The
    counters are now scoped to the phase that measured them (see
    `progress._phase`) and this fills the gap that left.

    It is not a small gap. Measured on this machine — RTX 4060 Laptop, torch
    2.11.0+cu128, transformers 5.13.0, `meta-llama/Llama-3.2-1B-Instruct`
    fully cached — `from_pretrained` took 7.1 s and the move took 15.36 s for
    2,471,629,056 bytes. Two thirds of the wall clock was the step with no
    meter, and it is two thirds for a structural reason: safetensors are
    memory-mapped, so `from_pretrained` returns before the weights have been
    read and the move is what actually pulls them off the disk. 160.9 MB/s
    there, against 215.6 MB/s reading a shard of the same cache sequentially
    — which is a filesystem's answer, not a GPU's. This cache is a junction
    onto a synced network drive; when that drive has to fetch rather than
    serve locally, the same step is minutes rather than seconds.

    A copy onto an accelerator is the one part of a load whose exact size is
    known before it starts, so there is no good reason for it to be the part
    that cannot say how far along it is.

    PUBLIC API only, deliberately. `Module.to` is `Module._apply` underneath
    and it would be shorter to hand `_apply` a counting function, but this is
    the model-load critical path: a private traversal that changes shape in a
    torch release would break loading rather than break a number. `.to()` is
    idempotent, so walking `modules()` deepest-first moves every tensor
    exactly once and each later call over the same subtree is a walk over
    tensors already on the device. Those repeat walks are what the approach
    costs, and they cost almost nothing: measured on the same 1B model, with
    every tensor already resident so that nothing is copied, 23 ms across its
    215 modules against 3 ms for one `Module.to`. Twenty milliseconds to put
    a number on a thirteen-second step.

    Bytes are counted per module and de-duplicated by the identity of the
    tensors as they were BEFORE that module moved, so a tied weight reached
    through two modules is counted once. `min` guards the published figure
    anyway, because a bar drawn past its own end is this project's most
    familiar way of being wrong.
    """
    total = resident_bytes(model)
    walk: list = []
    if total > 0:
        try:
            walk = list(model.modules())
        except Exception:
            # Same rule as `resident_bytes`, one step further along: a thing
            # that can be sized but not enumerated still has to move.
            walk = []
    if not walk:
        # Nothing to measure, nothing measurable, or nothing to walk. Either
        # way the move still has to happen; it just happens the way it always
        # did, under an indeterminate bar. A load must never fail because the
        # thing measuring it did.
        model.to(device)
        return 0
    # From here every tensor is known to be sizable, because `resident_bytes`
    # asked all of them before returning a figure at all. That is what lets
    # the loop below add bytes without a guard of its own.

    progress.TRACKER.publish(bytes_done=0, bytes_total=total)
    seen: set[int] = set()
    moved = 0
    last = 0.0
    for module in reversed(walk):
        # Read BEFORE the move. `.to()` rewrites `param.data` in place only
        # when the source and destination types are shallow-copy compatible —
        # true cpu->cuda, false cpu->meta — and otherwise installs a NEW
        # `Parameter`. Identity is what tells a tied weight from a second one,
        # so counting afterwards turned one shared embedding into two and
        # overshot the total. Sizes are the same either side of the move.
        own = (*module.parameters(recurse=False), *module.buffers(recurse=False))
        module.to(device)
        for tensor in own:
            if id(tensor) in seen:
                continue
            seen.add(id(tensor))
            moved += tensor.numel() * tensor.element_size()
        now = time.monotonic()
        if now - last >= DEVICE_PUBLISH_EVERY_S:
            last = now
            progress.TRACKER.publish(bytes_done=min(moved, total), bytes_total=total)
    # The last module is the root and its own parameters are usually none, so
    # the loop above can finish a whole second before its final publish. Said
    # once, plainly, rather than left to the next poll.
    progress.TRACKER.publish(bytes_done=min(moved, total), bytes_total=total)
    return moved
