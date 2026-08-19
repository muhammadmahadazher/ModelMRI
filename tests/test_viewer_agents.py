"""The zero-install viewer shows the run a `.mri` was built around.

`ShareRun`'s docstring is the promise this file guards:

    Every competitor's share artefact is a link into their hosted trace UI,
    which dies when the account lapses. This is a gzipped file that opens in a
    browser with nothing installed: the recipient sees the failing tool call,
    clicks it, and lands in the attention view of the generation that produced
    the bad argument, on a machine with no GPU.

Until this, `AgentsPanel` sat inside `{!VIEWER && …}` in `App.tsx`, so the
viewer mounted no agents panel at all and a bundle built around a failing step
opened with the run invisible.

These are source-level checks, deliberately. `tests/viewer_check.py` drives the
built viewer with Playwright and is where the rendering is proven; this asserts
the WIRING, which is what silently regresses — a component regated behind
`!VIEWER`, or a store-only control mounted where there is no store.
"""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "frontend" / "src"


def _read(name: str) -> str:
    return (SRC / name).read_text("utf-8", errors="replace")


def test_the_viewer_mounts_the_agents_panel():
    """The one line the whole feature rests on."""
    app = _read("App.tsx")
    assert re.search(r"\{VIEWER && <AgentsPanel", app), (
        "App.tsx no longer mounts AgentsPanel for the viewer, so a bundle "
        "built around a failing step opens with its run invisible"
    )


def test_the_viewer_serves_the_run_as_carried_rather_than_as_a_store_row():
    """`step_ref` is the reason the bundle was sent.

    Served through `/api/traces` it is an ordinary store row: it renders, and
    it opens on step one. Served as the CARRIED run — the same path the app
    uses — the panel opens on the step it was built around, and the row says
    so.
    """
    viewer = _read("viewer.ts")
    assert '"/api/session/trace"' in viewer, "the viewer serves no carried run"
    assert "step_ref" in viewer

    # The store list is empty on purpose: this page has no store, and listing
    # the run there promised it was deletable, persistent, and searchable.
    store = re.search(r'if \(p === "/api/traces"\) \{(.*?)\n  \}', viewer, re.S)
    assert store, "the /api/traces handler moved"
    assert "return ok([]);" in store.group(1), (
        "the viewer lists the carried run in the trace store again, which "
        "costs the step_ref and promises three things that are not true of it"
    )


def test_the_carried_run_reaches_the_panel_through_the_session_state():
    """`AgentsPanel` learns a run exists from `session.trace`, so the viewer's
    own state has to publish that block in the same shape `runtime.session_info`
    does."""
    viewer = _read("viewer.ts")
    state = re.search(r"function state\(\) \{(.*?)\n\}", viewer, re.S)
    assert state, "viewer state() moved"
    body = state.group(1)
    for field in ("available", "step_ref", "n_steps", "n_steps_total", "truncated"):
        assert field in body, f"session state carries no {field} for the run"


def test_the_steps_are_normalised_the_way_the_server_normalises_them():
    """A `.mri` stores a reduced step. The panels read a missing key and a null
    one very differently: `undefined !== null` is true, which printed
    "undefined cache read", and an absent `seq` printed "step undefined"."""
    viewer = _read("viewer.ts")
    assert "function viewerSteps(" in viewer
    for field in (
        "tokens_cache_read",
        "tokens_cache_write",
        "tokens_reasoning",
        "adoptable",
        "seq",
    ):
        assert field in viewer, f"viewerSteps does not supply {field}"


def test_no_control_that_needs_a_store_a_disk_or_weights_is_mounted_there():
    """A control that can only ever refuse teaches a reader that the feature is
    broken, which is the rule `Playground` states three times.

    Checked by counting `!VIEWER` gates rather than by naming each block: the
    point is that the store-only region is gated at all, and the Playwright
    driver confirms none of the seven renders.
    """
    panel = _read("AgentsPanel.tsx")
    assert 'from "./viewer"' in panel, "AgentsPanel cannot see VIEWER"
    # The store block, the search box, the clear row, and the two refusals
    # that change wording in the viewer.
    assert panel.count("!VIEWER") >= 3, (
        "the store-only regions are no longer gated for the viewer"
    )
    assert "fromSession || VIEWER" in panel, (
        "the viewer no longer says why a step cannot be adopted, or offers to "
        "package the bundle it is already reading"
    )


def test_the_viewer_refuses_the_store_routes_with_a_reason():
    """A 404 reads as "the viewer is broken". Each of these is a thing a file
    cannot do, and the sentence says which."""
    viewer = _read("viewer.ts")
    for path in (
        "/api/traces/search",
        "/api/traces/import",
        "/bundle/preview",
        "/adopt",
        "/api/rubric/score",
    ):
        assert path in viewer, f"{path} has no viewer answer"
