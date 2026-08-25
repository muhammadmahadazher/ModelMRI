"""Does a route send the keys `api.ts` says it sends?

Nothing checked this. `json<T>(r)` in `api.ts` is a bare type cast — it
renames the parsed body and verifies nothing — so a Python route and its
TypeScript declaration could drift apart in silence, and four of them had:

  GET /api/vla                   sent 5 fields the interface did not declare,
                                 and answered `n_layers: 0, n_heads: 0` beside
                                 `repo: null` at rest, so a resting panel read
                                 as a tower that exists and has no layers
  GET /api/sae                   sent `activation`, `threshold_span`,
                                 `release` — all three undeclared
  GET /api/telemetry             declared 17 fields as required that the
                                 no-generation state does not send at all
  GET /api/image/filmstrip/cost  declared 7 of the 16 keys it sends

The audit that found them noted that one test of this shape would have caught
all four at once, plus the missing `n_components` denominator on
`/api/attention/direct` and the demo's own drift. This is that test.

WHAT IT DOES NOT DO. It compares KEYS, not types: a `number` declared where a
string arrives goes unnoticed here. Keys are what actually drifted, they are
checkable without a TypeScript compiler, and a half-check that runs on every
commit beats a whole one that needs a toolchain the Python suite does not
have. `npm run build` type-checks the rest.

It also only exercises routes it can call AT REST — no model loaded, nothing
downloaded, no side effects. A route that refuses in that state is reported as
skipped rather than passed, so the coverage count is honest.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from modelmri.server import create_app

API_TS = Path(__file__).resolve().parents[1] / "frontend" / "src" / "api.ts"

#: Query strings for routes whose resting answer needs an argument. Values are
#: chosen so the route ANSWERS rather than refuses — the point is to see a
#: real payload, not to exercise validation.
ARGS = {
    "/api/image/filmstrip/cost": "?steps=20&every=4",
}

#: Routes this cannot reach at rest, with the reason. Listed rather than
#: silently skipped: a route that drops off the reachable set should be a
#: decision, not an absence nobody noticed.
UNREACHABLE = {
    "/api/attention/types": "needs a generation to classify heads from",
}


def _fields(name: str, text: str) -> tuple[set[str], set[str]] | None:
    """The top-level field names of an `api.ts` interface or union alias.

    Brace-depth aware, because several of these carry nested object literals
    (`SessionState.meta`, `ImageFilmstripPlan.selection`) and a flat regex
    reports their inner fields as the outer interface's own — which reads as
    eleven missing keys on a route that is perfectly correct.

    For `export type X = A | B` the members are resolved and merged: a union
    is satisfied when the payload matches EITHER shape, so the required set is
    the intersection and the optional set is everything else. `TelemetryReport`
    is exactly that — two genuinely different documents with `available` as
    the discriminator.
    """
    alias = re.search(r"export type " + re.escape(name) + r"\s*=\s*([^;]+);", text)
    if alias:
        members = [m.strip() for m in alias.group(1).split("|")]
        resolved = [_fields(m, text) for m in members]
        if not resolved or any(r is None for r in resolved):
            return None
        required = set.intersection(*[r[0] for r in resolved])
        every = set.union(*[r[0] | r[1] for r in resolved])
        return required, every - required

    head = re.search(
        r"export interface " + re.escape(name) + r"\s*(?:extends\s+([^{]+?)\s*)?\{",
        text,
    )
    if head is None:
        return None
    # `extends` is inheritance, and the payload carries the base's fields too.
    # `SessionTraceDoc extends TraceDoc` declares four of its own and inherits
    # nine; reading only the four reports the nine as undeclared.
    inherited_required: set[str] = set()
    inherited_optional: set[str] = set()
    for base in (head.group(1) or "").replace(",", " ").split():
        resolved = _fields(base, text)
        if resolved is None:
            return None
        inherited_required |= resolved[0]
        inherited_optional |= resolved[1]
    depth, start = 1, head.end()
    i = start
    while i < len(text) and depth:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    body = text[start : i - 1]
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    body = re.sub(r"//[^\n]*", "", body)

    required: set[str] = set(inherited_required)
    optional: set[str] = set(inherited_optional)
    depth = 0
    for line in body.splitlines():
        if depth == 0:
            match = re.match(r"\s*([a-zA-Z_][a-zA-Z0-9_]*)(\??)\s*:", line)
            if match:
                (optional if match.group(2) else required).add(match.group(1))
        depth += line.count("{") - line.count("}")
    return required, optional


def _routes(text: str) -> dict[str, str]:
    """Every `fetch("/api/…")` paired with the interface its result is cast to.

    Template literals are deliberately not matched: a path built from a
    variable carries an argument this cannot invent a value for, and guessing
    one would exercise a different route than the client calls.
    """
    pairs = re.findall(
        r'fetch\(\s*"(/api/[a-zA-Z0-9/_-]+)(\?[^"]*)?"\s*\)'
        r"(?:(?!fetch\().)*?json<([A-Za-z0-9_]+)>",
        text,
        re.S,
    )
    found: dict[str, str] = {}
    for path, query, interface in pairs:
        found.setdefault(path + (query or ""), interface)
    return found


@pytest.fixture(scope="module")
def api_text() -> str:
    return API_TS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def client() -> TestClient:
    # Loopback, because several of these carry the not-from-this-machine gate
    # and a refusal is not a payload to compare.
    return TestClient(create_app(), client=("127.0.0.1", 5000))


def test_every_get_route_sends_the_keys_its_interface_declares(api_text, client):
    """The whole point of the file. One assertion, every route it can reach."""
    drift: list[str] = []
    checked = 0
    for path, interface in sorted(_routes(api_text).items()):
        if path in UNREACHABLE:
            continue
        fields = _fields(interface, api_text)
        assert fields is not None, f"{interface} is not declared in api.ts"
        required, optional = fields

        response = client.get(path + ARGS.get(path, ""))
        if response.status_code != 200:
            continue
        body = response.json()
        if not isinstance(body, dict):
            continue

        checked += 1
        sent = set(body)
        missing = sorted(required - sent)
        extra = sorted(sent - required - optional)
        if missing:
            drift.append(f"{path} -> {interface}: declared, never sent {missing}")
        if extra:
            drift.append(f"{path} -> {interface}: sent, never declared {extra}")

    assert not drift, "the client and the server disagree:\n  " + "\n  ".join(drift)
    # A coverage floor, because a parser that silently stops matching would
    # otherwise turn this test green by checking nothing. 20 is comfortably
    # under the 24 reachable today and comfortably over an accident.
    assert checked >= 20, f"only {checked} routes were compared — did the parser break?"


def test_the_unreachable_list_has_no_stale_entries(api_text, client):
    """A route that starts answering at rest belongs back under the check. The
    list is a record of what this cannot exercise, not a place to park a
    failure."""
    known = _routes(api_text)
    for path, why in UNREACHABLE.items():
        assert path in known, f"{path} is no longer fetched from api.ts — drop it"
        response = client.get(path + ARGS.get(path, ""))
        assert response.status_code != 200, (
            f"{path} answers 200 at rest now ({why!r} is stale) — remove it from "
            f"UNREACHABLE so its keys are compared"
        )


def test_a_union_alias_is_satisfied_by_either_shape(api_text):
    """`TelemetryReport` is two genuinely different documents. Before the
    split it was one flat interface declaring 17 measurement fields as
    required, and the no-generation state sends exactly two keys — measured.

    This pins the parser as much as the type: an alias resolved to `None`
    would make the route above silently skip rather than fail."""
    fields = _fields("TelemetryReport", api_text)
    assert fields is not None, "TelemetryReport no longer resolves"
    required, optional = fields
    assert required == {"available"}, required
    assert {"reason", "tokens_per_s", "means"} <= optional


def test_a_nested_object_is_not_read_as_the_outer_interfaces_fields(api_text):
    """`SessionState.meta` and `ImageFilmstripPlan.selection` are object
    literals inline in the interface. A flat regex reported their inner field
    names as the outer interface's own, which showed up as eleven keys
    "declared but never sent" on a route that was entirely correct."""
    required, optional = _fields("SessionState", api_text)
    assert required == {"open"}, required
    for inner in ("model", "device", "dtype", "n_params"):
        assert inner not in required and inner not in optional, inner


# A route that cannot be reached AT REST is skipped by the test above — it
# needs a dataset opened, which CI has none of. That is honest and it is also
# a hole: `/api/vla/timeline` is a route with a declared interface and nothing
# in CI compares the two.
#
# The demo bundle closes it. `frontend/public/demo/vla.json` carries a REAL
# recorded answer from that route — baked by `scripts/bake_demo.py` off
# lerobot/pusht, not written by hand — so its keys are the route's keys, and
# checking them against `api.ts` is the same check by other means. It is also
# the payload the hosted demo actually serves, so drift here breaks a page
# people visit.
RECORDED = {
    "vla.json": [
        # bundle key path, api.ts interface
        (("timeline",), "EpisodeTimeline"),
        (("timeline", "tracks", 0), "TimelineTrack"),
        (("ood",), "EpisodeOod"),
        (("ood", "reference"), "OodReference"),
        (("ood", "reference", "distances"), "OodDistances"),
        (("ood", "frames", 0), "OodFrame"),
        (("ood_cost",), "EpisodeOodCost"),
    ],
}


def _dig(blob, path):
    for step in path:
        blob = blob[step]
    return blob


def test_a_recorded_payload_carries_the_keys_its_interface_declares(api_text):
    """For the routes CI has no dataset to reach live."""
    import json

    bundle_dir = API_TS.resolve().parents[2] / "frontend" / "public" / "demo"
    drift: list[str] = []
    checked = 0
    for name, entries in RECORDED.items():
        path = bundle_dir / name
        if not path.is_file():
            continue
        blob = json.loads(path.read_text(encoding="utf-8"))
        for keys, interface in entries:
            fields = _fields(interface, api_text)
            assert fields is not None, f"{interface} is not declared in api.ts"
            required, optional = fields
            try:
                body = _dig(blob, keys)
            except (KeyError, IndexError):
                drift.append(f"{name}:{'.'.join(map(str, keys))} is not in the bundle")
                continue
            checked += 1
            sent = set(body)
            missing = sorted(required - sent)
            extra = sorted(sent - required - optional)
            where = f"{name}:{'.'.join(map(str, keys))} -> {interface}"
            if missing:
                drift.append(f"{where}: declared, never sent {missing}")
            if extra:
                drift.append(f"{where}: sent, never declared {extra}")

    assert not drift, "the client and a recorded payload disagree:\n  " + "\n  ".join(
        drift
    )
    # Same floor argument as above: a bundle that stopped being written would
    # otherwise turn this green by comparing nothing.
    assert checked == sum(len(v) for v in RECORDED.values()), (
        f"only {checked} recorded payload(s) were compared"
    )
