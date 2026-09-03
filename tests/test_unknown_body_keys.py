# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""A request body that names a field no route has is a typo, not a default.

Pydantic ignores unknown keys unless told not to, and the consequence was
measured against the running server before this was written:

    POST /api/model/load {"model_id": "Qwen/Qwen3-1.7B"}
    -> 200 {"loaded": true, "hf_id": "Qwen/Qwen2.5-0.5B-Instruct", ...}

The field is `hf_id`. The caller named one model, was told it worked, and a
different one was loaded — after which every panel on the page measures a
model nobody asked for.

The load is the visible version. The dangerous one is a sweep or a probe with
a misspelled parameter: it runs, finishes, and reports numbers labelled as
though the parameter had been applied. That is a silently wrong measurement
presented as a right one, which is the failure this project exists to avoid,
manufactured from a single typo.

Two of this repo's OWN tests were sending `{"id": ...}` to the load route and
reading as though they loaded Qwen3-1.7B. They were loading the default. That
is how quiet this is.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from modelmri import server as srv
from modelmri.server import create_app


@pytest.fixture
def client():
    with TestClient(create_app()) as c:
        yield c


def _request_models():
    """Every pydantic model FastAPI will validate a request body with.

    Walked from the ROUTES rather than from the module namespace. The
    namespace version also collects `BaseModel` itself, which is imported
    there and is not a request model — and, worse, would miss a model defined
    elsewhere and bound here. What matters is what a POST body is checked
    against, which is exactly what the route table says.

    `server.py` uses `from __future__ import annotations`, so the annotations
    are STRINGS and have to be resolved against the module. Without that this
    finds nothing and passes for a set it never looked at, which is why the
    emptiness assertion below is not decoration.
    """
    app = create_app()
    out = {}
    for route in app.routes:
        if "POST" not in (getattr(route, "methods", None) or set()):
            continue
        for _param, ann in getattr(
            getattr(route, "endpoint", None), "__annotations__", {}
        ).items():
            name = ann if isinstance(ann, str) else getattr(ann, "__name__", "")
            cls = getattr(srv, name, None)
            if cls is not None and hasattr(cls, "model_fields"):
                out[name] = cls
    return out


def test_every_request_model_refuses_a_key_it_does_not_know():
    """Held on the whole set rather than on the one that was found.

    A model added later that forgets to inherit `Body` reopens exactly this
    hole for its own route, and nothing else here would notice.
    """
    models = _request_models()
    assert models, "no request models found — this test is not looking at anything"
    lax = [
        name
        for name, cls in models.items()
        if cls.model_config.get("extra") != "forbid"
    ]
    assert not lax, f"these accept unknown keys silently: {sorted(lax)}"


def test_the_load_route_refuses_the_misspelling_rather_than_loading_something_else(
    client,
):
    """The measured case, pinned by the exact body that produced it."""
    r = client.post("/api/model/load", json={"model_id": "Qwen/Qwen3-1.7B"})
    assert r.status_code == 422
    # The offending key is NAMED. "422" alone sends somebody to read the
    # source; "model_id — extra inputs are not permitted" does not.
    assert "model_id" in r.text


def test_the_right_key_is_untouched(client, monkeypatch):
    """The guard must not fire on the ordinary case.

    Patched at the loader rather than actually pulling weights: what is being
    checked is that validation LET IT THROUGH, which the refusal from the
    stub proves as well as a real load would and in a fraction of the time.
    """
    import transformers

    def explode(*_a, **_k):
        raise RuntimeError("reached the loader")

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", staticmethod(explode)
    )
    r = client.post("/api/model/load", json={"hf_id": "Qwen/Qwen3-1.7B"})
    assert r.status_code != 422, r.text


def test_an_empty_model_id_is_refused_by_name_rather_than_by_the_loader(client):
    """`ImageLoadRequest.repo` has carried `min_length` for months; this had none.

    An explicit `{"hf_id": ""}` went all the way to transformers and came back
    "Could not load '', and this is not one of the failures ModelMRI knows how
    to explain" — the sentence reserved for failures the tool genuinely does
    not understand, printed about the one input it understands perfectly.

    Two routes answering the same question about the same kind of input
    differently is the defect, not the wasted load.
    """
    r = client.post("/api/model/load", json={"hf_id": ""})
    assert r.status_code == 422
    assert "hf_id" in r.text
    assert "at least 1 character" in r.text
    # And the tool's own fallback sentence is NOT what a reader gets.
    assert "not one of the failures" not in r.text


def test_omitting_the_model_id_still_takes_the_default(client, monkeypatch):
    """The minimum applies to an empty string, not to an absent key.

    That distinction is the whole reason the field keeps its default: the
    client omits `hf_id` rather than sending an empty one (`api.ts` builds the
    body conditionally), so requiring it outright would break the ordinary
    call.
    """
    import transformers

    def explode(*_a, **_k):
        raise RuntimeError("reached the loader")

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", staticmethod(explode)
    )
    r = client.post("/api/model/load", json={"source": "hf"})
    assert r.status_code != 422, r.text


# The routes that take a raw body, and the reason each one is allowed to.
#
# `_request_models` above walks the route table and collects pydantic models.
# A route annotated `body: dict` or `request: Request` resolves to NOTHING, so
# it was not merely unchecked — it was INVISIBLE. The suite reported 33 clean
# models while thirteen routes had no key discipline at all, and the guard that
# was supposed to notice could not see them.
#
# Measured on one of them: `POST /api/rubric/score {"rulez": [...]}` — one
# transposed letter — answered 200 with "0 rule(s) against 111 recorded run(s).
# No run matched any rule.", where the correct spelling found 66 runs matching.
#
# Anything on this list is a deliberate exception with a stated reason. A NEW
# raw-bodied route fails the test below until it is either given a model or
# argued onto this list.
RAW_BODY_ALLOWED = {
    # OpenAI compatibility: clients send fields we do not model, and refusing
    # an unknown key here would break the thing the endpoint exists to be.
    "/v1/chat/completions": "OpenAI-compatible; unknown keys are expected",
    "/v1/completions": "OpenAI-compatible; unknown keys are expected",
    # These read the raw Request for something other than the body — the
    # loopback check on a `file` field, or a streamed upload — so a model
    # cannot replace the parameter. They validate their own fields explicitly.
    "/api/custom/ablate": "reads the raw Request; validates its own fields",
    "/api/diff/models": "loopback check on `file` needs the Request",
    "/api/features/evidence": "loopback check on `file` needs the Request",
    "/api/ground": "loopback check on `file` needs the Request",
    "/api/lens/tune": "loopback check on `file` needs the Request",
    "/api/otel/v1/traces": "OTLP wire format, not ours to constrain",
    "/api/patch/path": "reads the raw Request",
    "/api/patchscope": "reads the raw Request",
    "/api/probe": "reads the raw Request",
    "/api/session/open": "reads the raw Request",
    "/api/traces/import": "a whole trace document, shaped by the recorder",
    "/api/traces/import/inspect": "a file upload",
    "/api/vla/occlude": "reads the raw Request",
    "/api/vla/share": "reads the raw Request",
    "/api/vla/sweep": "reads the raw Request",
}


def test_no_new_route_takes_a_raw_body_without_saying_why():
    """The blind spot itself, made visible.

    A route annotated `dict` or `Request` cannot be reached by
    `_request_models`, so every check in this file silently skipped it. This
    one walks the same table and fails on any raw-bodied POST route that is not
    on the list above with a reason beside it.

    It does not claim the listed routes are safe — several validate their own
    fields by hand and one or two should still grow a model. It claims that
    adding a new one is a decision somebody made on purpose.
    """
    app = create_app()
    raw = []
    for route in app.routes:
        if "POST" not in (getattr(route, "methods", None) or set()):
            continue
        annotations = getattr(
            getattr(route, "endpoint", None), "__annotations__", {}
        ).items()
        # A ROUTE IS ONLY BLIND IF NOTHING VALIDATES ITS BODY.
        #
        # The first version of this flagged any parameter annotated `Request`,
        # and that is not the same question. `/api/weights/scan` is
        # `(req: ScanRequest, request: Request)` — its body IS checked against
        # a model, and the `Request` is there for `_not_from_this_machine`,
        # which reads the Origin header and the client address rather than the
        # body. Flagging it put a fully-validated route on an allowlist under
        # the reason "reads the raw Request", which it does not do, and the
        # entry then read as a decision somebody had made rather than as this
        # test's own imprecision. Caught when a new route with exactly that
        # correct shape was flagged too.
        validated = any(
            hasattr(
                getattr(
                    srv, (a if isinstance(a, str) else getattr(a, "__name__", "")), None
                ),
                "model_fields",
            )
            for param, a in annotations
            if param != "return"
        )
        if validated:
            continue
        for _param, ann in annotations:
            # `return` is in `__annotations__` too, and five routes declare
            # `-> dict` while taking no body whatsoever. Reading it as a raw
            # body flagged `/api/model/cancel` and four siblings that have
            # nothing to validate — the first version of this test did exactly
            # that, and the audit that prompted it counted them as defects.
            if _param == "return":
                continue
            name = ann if isinstance(ann, str) else getattr(ann, "__name__", "")
            if name in ("dict", "Request") and route.path not in RAW_BODY_ALLOWED:
                raw.append(f"{route.path} ({_param}: {name})")

    assert not raw, (
        "these POST routes take a raw body, so no unknown-key check applies to "
        "them and every test in this file skips them:\n  "
        + "\n  ".join(sorted(raw))
        + "\n\nGive each a Body-derived request model, or add it to "
        "RAW_BODY_ALLOWED with the reason it needs the raw Request."
    )


def test_the_allowlist_has_no_stale_entries():
    """An entry for a route that no longer takes a raw body is a licence
    nobody is using, and the next raw-bodied route at that path inherits it
    without anyone deciding."""
    app = create_app()
    raw_paths = set()
    for route in app.routes:
        if "POST" not in (getattr(route, "methods", None) or set()):
            continue
        annotations = getattr(
            getattr(route, "endpoint", None), "__annotations__", {}
        ).items()
        # The SAME question the test above asks, and it has to be asked the
        # same way or the two disagree. Five entries sat on this list reading
        # "reads the raw Request" for routes that take a request model AND a
        # `Request` — the second for `_not_from_this_machine`, which reads
        # headers rather than the body. Their bodies were validated the whole
        # time, so the licence was one nobody was using and the reason beside
        # it was not true.
        if any(
            hasattr(
                getattr(
                    srv,
                    (a if isinstance(a, str) else getattr(a, "__name__", "")),
                    None,
                ),
                "model_fields",
            )
            for param, a in annotations
            if param != "return"
        ):
            continue
        for _param, ann in annotations:
            if _param == "return":
                continue
            name = ann if isinstance(ann, str) else getattr(ann, "__name__", "")
            if name in ("dict", "Request"):
                raw_paths.add(route.path)

    stale = sorted(set(RAW_BODY_ALLOWED) - raw_paths)
    assert not stale, (
        "these are on the raw-body allowlist and no longer need to be — each "
        "either takes a request model now, or is gone:\n  " + "\n  ".join(stale)
    )
