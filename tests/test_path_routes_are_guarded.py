"""Every route that turns a request string into a path on this disk is guarded.

Reading a local corpus, a local document, a local checkpoint IS the feature
here — the alternative is uploading them, and a local-first tool that made you
move your files first would have no reason to exist. So these routes cannot be
removed, and what stands between them and a file-read primitive on the reader's
own disk is one of two boundaries:

  WHO    `_not_from_this_machine` — any file, but only the person at the
         keyboard may ask.
  WHERE  `custom.resolve_under_roots` — anyone on loopback may ask, but only
         under roots this server was already told about.

Both are real and the codebase uses both, which this file learned by being
written wrongly first: asserted against the WHO check alone, it failed on
`/api/gguf`, `/api/gguf/plan`, `/api/gguf/load` and `/api/quantdiff/behaviour`
— four routes that are guarded, by the other one. What must never happen is
NEITHER.

WHAT WOULD ACTUALLY GO WRONG. A page on any website can POST to localhost, and
the request arrives from 127.0.0.1 like every other. Without the `Origin` half
of the WHO check, a visited page could name any file the server's user can read
— an SSH key, a password store — and get the contents back as "prompts". That
is not hypothetical for a tool whose install instructions are "run it on your
own machine".

Nothing pinned any of this. Nine routes carry a boundary by hand, and a tenth
added without one would have been a real vulnerability that every existing test
still passed. This file is that pin, and it is deliberately written against the
SINKS — the readers that open a path — rather than against a list of route
names, because a list of names is what goes stale.

It is also the honest answer to CodeQL's `py/path-injection` on
`sweep.load_prompts`. That alert is correct that the path is user-provided and
wrong that it is unguarded: the sanitiser is an authorisation check three
frames up, which taint tracking cannot see. `/api/weights/scan` has the
identical shape and is not flagged, which is the tell. Rather than argue with
the scanner or narrow a feature to satisfy it, the guard it cannot see is
pinned here, where a regression fails loudly.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

SERVER = Path(__file__).resolve().parents[1] / "modelmri" / "server.py"

#: Functions that take a caller-supplied string and open what it names. Adding
#: a reader here without also guarding its routes is the failure this catches.
READERS = (
    "load_corpus",
    "load_prompts",
    "resolve_under_roots",
    "resolve_dir_under_roots",
    "weights_scan.scan",
    "weights_scan.scan_dir",
    "gguf_read.read",
)

#: Routes that reach a reader with a path this server ALREADY owns rather than
#: one the caller named — the trace store's own files, a cached snapshot this
#: process resolved itself. The guard is about strings crossing the wire, so a
#: path that never crossed it is not in scope. Each entry says which.
NOT_CALLER_NAMED: dict[str, str] = {}


def _routes() -> list[tuple[str, str, int]]:
    """(method+path, source of the handler, first line) for every route."""
    tree = ast.parse(SERVER.read_text(encoding="utf-8"))
    src = SERVER.read_text(encoding="utf-8").splitlines()
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            call = dec.func if isinstance(dec, ast.Call) else None
            if not isinstance(call, ast.Attribute):
                continue
            if not (isinstance(call.value, ast.Name) and call.value.id == "app"):
                continue
            if not dec.args or not isinstance(dec.args[0], ast.Constant):
                continue
            name = f"{call.attr.upper()} {dec.args[0].value}"
            body = "\n".join(src[node.lineno - 1 : node.end_lineno])
            found.append((name, body, node.lineno))
    return found


def test_every_route_that_opens_a_named_path_has_a_boundary_of_some_kind():
    """The one assertion, and it takes EITHER of two boundaries.

    Writing this against `_not_from_this_machine` alone was wrong, and the
    test said so: four routes failed it — `/api/gguf`, `/api/gguf/plan`,
    `/api/gguf/load`, `/api/quantdiff/behaviour` — and all four are guarded,
    by the other boundary. They pass their path through
    `custom.resolve_under_roots`, which refuses anything outside the server's
    own working directory and `MODELMRI_MODELS_DIR`.

    So there are two designs here and both are real:

      WHO   `_not_from_this_machine` — any file on the disk, but only the
            person at the keyboard may ask. What the corpus readers need,
            because "point it at my own text file" IS the feature and a
            local-first tool that made you move your files first would have
            no reason to exist.
      WHERE `resolve_under_roots` — anyone on loopback may ask, but only
            under roots this server was already told about. What the
            checkpoint readers need, because they are reached from a picker
            that already lists what is under those roots.

    What must never happen is NEITHER. A route with no boundary at all is a
    file-read primitive on the reader's own disk, and nothing before this
    file would have caught one being added.
    """
    unguarded = []
    checked = []
    for name, body, line in _routes():
        if not any(r + "(" in body or f"{r}," in body for r in READERS):
            continue
        if name in NOT_CALLER_NAMED:
            continue
        checked.append(name)
        who = "_not_from_this_machine" in body
        where = "resolve_under_roots" in body or "resolve_dir_under_roots" in body
        if not (who or where):
            unguarded.append(f"{name}  (server.py:{line})")

    assert not unguarded, (
        "these routes open a path named by the request and have neither "
        "boundary — not `_not_from_this_machine` (who asked) and not "
        "`resolve_under_roots` (where it may point):\n  " + "\n  ".join(unguarded)
    )
    # A floor, because a parser that stopped matching would pass by checking
    # nothing. Nine carried a boundary when this was written.
    assert len(checked) >= 9, (
        f"only {len(checked)} path-reading routes were found — did the reader "
        f"list or the route parser go stale? Found: {sorted(checked)}"
    )


def test_the_two_boundaries_are_both_actually_in_use():
    """Neither arm of the test above is dead. If every route ever ends up on
    one boundary, that is a design decision somebody should make on purpose
    rather than discover here."""
    who, where = [], []
    for name, body, _ in _routes():
        if not any(r + "(" in body or f"{r}," in body for r in READERS):
            continue
        if "_not_from_this_machine" in body:
            who.append(name)
        if "resolve_under_roots" in body or "resolve_dir_under_roots" in body:
            where.append(name)
    assert len(who) >= 4, f"the WHO boundary is barely used: {sorted(who)}"
    assert len(where) >= 3, f"the WHERE boundary is barely used: {sorted(where)}"


def test_the_guard_refuses_a_cross_site_origin_and_not_only_a_remote_address():
    """The half that matters against a visited page, and the half a reviewer
    is most likely to think is redundant.

    Loopback alone does not settle it: a page on any website can POST to
    localhost and the request arrives from 127.0.0.1. A JSON body already
    forces a preflight that fails without CORS headers — there are none — but
    that is a side effect being relied on rather than a decision, and side
    effects change with browser versions.
    """
    from fastapi.testclient import TestClient

    from modelmri.server import create_app

    client = TestClient(create_app(), client=("127.0.0.1", 5000))
    body = {"path": "/etc/passwd"}

    # From the page this tool serves: allowed past the guard, and refused later
    # for being the wrong kind of file — which is a different refusal.
    fine = client.post(
        "/api/weights/scan", json=body, headers={"Origin": "http://localhost:5900"}
    )
    assert "only possible from this machine" not in str(fine.json())

    # From anywhere else: refused before the path is looked at.
    for origin in ("https://evil.example", "http://127.0.0.1.evil.example"):
        r = client.post("/api/weights/scan", json=body, headers={"Origin": origin})
        assert r.status_code == 403, origin
        assert "came from another site" in r.json()["error"], origin


def test_a_request_from_another_machine_is_refused_even_with_no_origin():
    """A non-browser client sends no `Origin`, so the address check is the only
    one left — and `serve --host 0.0.0.0` makes every handler here reachable by
    whoever can route to the port."""
    from fastapi.testclient import TestClient

    from modelmri.server import create_app

    remote = TestClient(create_app(), client=("10.0.0.7", 5000))
    r = remote.post("/api/weights/scan", json={"path": "/etc/passwd"})
    assert r.status_code == 403
    assert "only possible from this machine" in r.json()["error"]


def test_the_guard_is_not_quietly_weakened_to_a_substring_check():
    """`127.0.0.1.evil.example` and `localhost.evil.example` both CONTAIN an
    allowed name. The guard parses the origin and compares the hostname, and a
    future edit to `in` rather than `==` would pass every other test here."""
    source = SERVER.read_text(encoding="utf-8")
    guard = source[source.index("def _not_from_this_machine") :][:2000]
    assert "urlparse" in guard, "the origin is parsed, not string-matched"
    assert re.search(r"host\s+not\s+in\s+\(", guard), (
        "the hostname is compared against a tuple of exact names"
    )
