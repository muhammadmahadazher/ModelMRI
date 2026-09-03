# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""The shared session answer is dropped when the resident model changes.

`RunsOn` caches `/api/session` at module scope so six panels share one request.
It was keyed on `epoch` alone, and `epoch` counts GENERATIONS: `Playground`
bumps it when one finishes, and sets it to the literal 0 after a load. On a
fresh page nothing has generated, so epoch is already 0, `setEpoch(0)` changes
nothing, no `[epoch]` effect re-runs, and the cache hands back the answer
fetched before the model existed.

MEASURED in the browser before the fix: fresh page, press Load, wait for
"Loaded ✓" in the RUN panel — and six panels below it read "Nothing is loaded,
so pick one in Run at the top of the page first." The page telling you to do
the thing you just did.

The mirror is worse. After Unload, `App` bumps `resetKey` and the remount lands
on epoch 0 again, but a module-level cache survives a remount — so the panels
advertised "measures <model> · cuda:0 · bfloat16" with live buttons over freed
memory.

Source-level checks: the browser behaviour is what was driven, and these guard
the wiring, which is what regresses silently.
"""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "frontend" / "src"


def _read(name: str) -> str:
    return (SRC / name).read_text("utf-8", errors="replace")


def test_the_cache_can_be_invalidated_and_the_readers_subscribe():
    """A cleared cache alone is not enough: the effects are keyed on `[epoch]`,
    so nothing would re-run to notice it was cleared."""
    runs_on = _read("RunsOn.tsx")
    assert "export function invalidateSession" in runs_on
    # NO READER KEYED ON EPOCH ALONE. This used to assert that exactly two
    # effects were subscribed, which is a count and not the property: adding a
    # third subscribed reader -- `useModelIdentity`, which exists so a panel
    # holding a measurement can tell a model swap from a generation -- failed
    # a test about the opposite mistake.
    assert "}, [epoch]);" not in runs_on, (
        "a reader is keyed on epoch alone again, so it will not notice a model "
        "change that did not also produce a generation"
    )
    assert runs_on.count("}, [epoch, seen]);") >= 2, (
        "the subscribed readers are gone; a cleared cache would be cleared "
        "with nothing re-running to notice"
    )
    assert "useSessionVersion" in runs_on


def test_every_path_that_changes_the_resident_model_says_so():
    """Load and unload are the two, and both were silent to this cache."""
    play = _read("Playground.tsx")
    app = _read("App.tsx")
    assert "invalidateSession()" in play, "a load no longer invalidates"
    assert "invalidateSession()" in app, "an unload no longer invalidates"

    # The load path specifically: beside the `setEpoch(0)` that cannot signal
    # anything, because it is already 0 on the run that matters.
    load = re.search(r"invalidateSession\(\);\s*\n\s*setEpoch\(0\);", play)
    assert load, (
        "the invalidation is no longer on the load path next to setEpoch(0) — "
        "which is the no-op it exists to compensate for"
    )


def test_epoch_is_not_made_monotonic_instead():
    """The tempting one-line fix, which breaks two other things.

    `epoch > 0` gates the telemetry bar and the attention panel. Bumping it on
    a load would mount both for a run that never happened — a bar of timings
    for a generation nobody made.
    """
    play = _read("Playground.tsx")
    assert "setEpoch(0);" in play, (
        "the load path no longer resets epoch, so a load will mount the "
        "telemetry bar and attention panel for a run that never happened"
    )
    assert "{epoch > 0 && !replay && <TelemetryBar" in play
