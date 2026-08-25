"""A `.mri` section nobody can read is bytes nobody can see.

THREE TIMES this project has shipped a section with a writer and no reader,
and every one was found by a person opening a real file rather than by a test:

  the agent trace   `session.py` parsed it, `mcp_server` reported it, the web
                    UI had no `trace` field beside its siblings. Fixed with
                    `/api/session/trace` on 2026-08-19.
  the image run     A6 built the writer, the reader, the routes and the panel
                    on 2026-08-25 -- and mounted the panel inside App's
                    `!VIEWER` gate, so the one build it was written for never
                    rendered it.
  the robot finding `/api/vla/share` wrote a validated section from the day
                    the robot work landed. Nothing served it back, so the
                    recipient opened an empty text session: "1 tokens, 0
                    attention maps" over a measured occlusion map.

The pattern never varies: the FORMAT learns to carry something before the
surfaces learn to show it, and every layer looks correct on its own. So this
walks the predicates the parser actually exposes rather than a list somebody
remembers to update -- a hand-written list would have been written while the
gap was invisible and would have had the same hole.

Writing the fourth instance down is the point. `head_types` and `model_diff`
are in `NO_READER_YET` below because they are exactly that, found by this
test on the day it was written.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from modelmri import session
from modelmri.server import create_app

#: Sections served back by a route of their own. The route is checked to exist
#: and to answer, because a mapping to a route that is not there reads as
#: coverage while providing none.
SERVED_BY_ROUTE = {
    "has_vla": "/api/vla/replay",
    "has_image": "/api/image/replay",
    "has_trace": "/api/session/trace",
    # `/api/graph` answers for a live model and for an opened file alike --
    # it reads `runtime.replay` inside the same function rather than through a
    # second route.
    "has_graph": "/api/graph",
}

#: Sections read somewhere other than a route of their own. Just as valid a
#: reader; named so the mapping stays honest about HOW each one is read.
READ_ELSEWHERE = {
    # `runtime.session_info` puts these in `/api/session`, and the panels take
    # them as props (`sessionPatch`, `sessionGround`, `sessionPatchGraph`).
    "has_patch": "/api/session, as session.patch",
    "has_ground": "/api/session, as session.ground",
    "has_patch_graph": "/api/session, as session.patch_graph",
    # Not a screen but a real reader: `modelmri diff` compares two recordings'
    # head rankings, baselines and noise floors.
    "has_ranking": "modelmri/mri_diff.py, by `modelmri diff`",
}

#: SECTIONS WITH NO READER ANYWHERE. Written into every export by
#: `runtime.py` (`head_types=self._types_for_export()`,
#: `model_diff=self._model_diff_for_export()`), parsed and validated by
#: `session.py`, and read by nothing: no route, no `session_info`, no panel,
#: no CLI command, not `mri_diff`.
#:
#: Recorded rather than quietly tolerated, and rather than deleted: these are
#: measurements somebody paid forward passes for, and the fix is a reader, not
#: a smaller file. Listed here so the fourth instance of this pattern cannot
#: be discovered a fourth time by accident.
NO_READER_YET = {
    "has_head_types": (
        "written by runtime.py:_types_for_export, validated by "
        "session._head_types, read by nothing"
    ),
    "has_model_diff": (
        "written by runtime.py:_model_diff_for_export, validated by "
        "session._model_diff, read by nothing"
    ),
}


def _predicates() -> list[str]:
    """Asked of the class, never of a list in this file."""
    return sorted(
        name
        for name, _ in inspect.getmembers(session.Session, inspect.isfunction)
        if name.startswith("has_")
    )


def test_every_section_the_parser_exposes_is_accounted_for():
    """The one that would have caught the robot finding.

    A section with a `has_*` predicate is one the format carries and the
    parser validates. A NEW one fails here until somebody decides where it is
    read -- or writes down, in `NO_READER_YET`, that it is not."""
    known = set(SERVED_BY_ROUTE) | set(READ_ELSEWHERE) | set(NO_READER_YET)
    missing = [name for name in _predicates() if name not in known]
    assert not missing, (
        f"these sections are parsed and nothing is recorded as reading them: "
        f"{missing}. A writer does not imply a reader -- decide which surface "
        f"serves it and add it above, or record it in NO_READER_YET with what "
        f"writes it."
    )


def test_nothing_is_recorded_twice():
    """A section in two buckets means one of them is stale, and the stale one
    is the one somebody will read."""
    buckets = [set(SERVED_BY_ROUTE), set(READ_ELSEWHERE), set(NO_READER_YET)]
    for i, a in enumerate(buckets):
        for b in buckets[i + 1 :]:
            assert not (a & b), f"recorded in two places: {sorted(a & b)}"


@pytest.mark.parametrize("route", sorted(set(SERVED_BY_ROUTE.values())))
def test_each_named_route_exists_and_answers(route):
    client = TestClient(create_app())
    paths = {r.path for r in client.app.routes if hasattr(r, "path")}
    assert route in paths, f"{route} is recorded as a reader and is not a route"
    # With nothing open it must still ANSWER. `available: false` is a state,
    # and a 404 or a 500 here renders as "this measurement is broken" on every
    # session that simply carries no such section.
    assert client.get(route).status_code == 200


def test_the_viewer_shim_answers_them_too():
    """The recipient's build has no server behind it: `frontend/src/viewer.ts`
    re-implements these routes over the opened file. A route the shim does not
    handle falls through, and the panel renders an error or nothing at all --
    which is exactly how the image run and the robot finding stayed invisible.
    """
    shim = Path(__file__).resolve().parents[1] / "frontend" / "src" / "viewer.ts"
    text = shim.read_text(encoding="utf-8")
    unhandled = [
        r for r in sorted(set(SERVED_BY_ROUTE.values())) if f'"{r}"' not in text
    ]
    assert not unhandled, (
        f"{unhandled} are served by the app and not by the viewer shim, so a "
        f"`.mri` carrying that section opens with nothing on screen for the "
        f"person it was sent to."
    )


def test_the_sections_with_no_reader_are_still_being_written():
    """`NO_READER_YET` is a record of a gap, not a licence for one.

    If the writer goes away the entry is stale and must go with it; if a
    reader arrives the entry moves up. Either way this fails rather than
    letting a stale note sit here looking like a decision somebody made."""
    runtime = (
        Path(__file__).resolve().parents[1] / "modelmri" / "runtime.py"
    ).read_text(encoding="utf-8")
    for predicate in NO_READER_YET:
        field = predicate[len("has_") :]
        assert f"{field}=" in runtime, (
            f"{predicate} is recorded as written-but-unread, and nothing in "
            f"runtime.py writes it any more. Remove the entry, and the parser "
            f"for it if the format has genuinely dropped the section."
        )
