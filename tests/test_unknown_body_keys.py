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
