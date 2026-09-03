# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""No route that reads an untyped body may 500 on the shape of that body.

Twelve handlers take `await request.json()` rather than a Pydantic model. That
choice is right — their bodies are open-ended, and `occlusion` is a mapping
whose keys the caller picks, so modelling them would be modelling somebody
else's schema. Indexing the result without checking it is a different matter.

MEASURED before this existed, one route giving two answers to one question
depending only on whether the bad value happened to be a number or a string:

    POST /api/vla/sweep {"frame_stride": -5}      -> 422 "both strides must be at least 1"
    POST /api/vla/sweep {"episode_stride": "abc"} -> 500
    POST /api/vla/occlude [1, 2, 3]               -> 500

The test is written against the SOURCE's own list of coercions rather than a
list copied here, so a route added tomorrow with a new `_whole(body, ...)`
field is covered the day it lands rather than the day somebody remembers.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from modelmri.server import create_app

SERVER = Path(__file__).resolve().parents[1] / "modelmri" / "server.py"


def _untyped_routes() -> list[tuple[str, list[str]]]:
    """Every POST handler that reads a raw body, and the fields it coerces."""
    src = SERVER.read_text(encoding="utf-8")
    out: list[tuple[str, list[str]]] = []
    for m in re.finditer(r'@app\.post\("(/api/[^"]+)"\)', src):
        nxt = src.find("    @app.", m.end())
        seg = src[m.end() : nxt if nxt > 0 else len(src)]
        if "await request.json()" not in seg:
            continue
        fields = sorted(set(re.findall(r'_(?:whole|real)\(body, "([^"]+)"', seg)))
        out.append((m.group(1), fields))
    return out


ROUTES = _untyped_routes()


def test_the_scan_found_the_handlers():
    """A regex that matches nothing would make every test below vacuous."""
    assert len(ROUTES) >= 10, ROUTES


@pytest.mark.parametrize("path", [p for p, _ in ROUTES])
def test_a_json_array_as_the_whole_body_is_refused_not_crashed(path):
    """`body.get` on a list is an AttributeError, and it reached the 500."""
    client = TestClient(create_app(), raise_server_exceptions=False)
    r = client.post(path, json=[1, 2, 3])
    assert r.status_code != 500, r.text
    assert "JSON object" in r.text


@pytest.mark.parametrize(
    "path,field",
    [(p, f) for p, fields in ROUTES for f in fields],
)
def test_a_field_that_is_not_a_number_names_itself(path, field):
    """`int("abc")` raised `invalid literal for int() with base 10` into the
    generic 500 — a sentence about CPython rather than about the request."""
    client = TestClient(create_app(), raise_server_exceptions=False)
    r = client.post(path, json={field: "not-a-number"})
    assert r.status_code != 500, f"{path} {field}: {r.text}"
    # Either the coercion refused by name, or the route refused earlier for a
    # reason of its own (no model loaded, no dataset open). Both are answers;
    # neither is a crash.
    if "must be a" in r.text:
        assert field in r.text


@pytest.mark.parametrize(
    "path,field",
    [(p, f) for p, fields in ROUTES for f in fields],
)
def test_a_boolean_is_not_quietly_the_number_one(path, field):
    """`isinstance(True, int)` is True, so a JSON `true` would otherwise slide
    through as a stride of 1 — a measurement nobody asked for."""
    client = TestClient(create_app(), raise_server_exceptions=False)
    r = client.post(path, json={field: True})
    assert r.status_code != 500, f"{path} {field}: {r.text}"
