"""The error contract: which "no" is which, and what never reaches the browser.

Three things are pinned here, and the third is the one that matters.

1. `Refusal` is still a `RuntimeError` and `BadRequest` is still a
   `ValueError`. Every `except RuntimeError` and `except ValueError` already
   in this codebase depends on that, and the whole migration is only safe to
   land module by module because of it. Deleting those base classes would look
   like tidying and would silently turn refusals into 500s in every module
   that had not been converted yet.

2. A `Refusal` reaches the client as 409 in its own words, because those words
   were written for the reader and are the answer.

3. Anything else is a 500 whose body does NOT contain the exception's text,
   and whose traceback IS in the log. That pairing is the point. Before this,
   a CUDA out-of-memory came back as `409 {"error": "CUDA out of memory. Tried
   to allocate ... C:\\Users\\...\\blobs\\..."}` — a crash reported as a
   conflict, in torch's words, naming directories on the machine running the
   server. Dropping the text without logging it would swap that for an
   erasure, which is worse. So both halves are asserted together, and a
   future `str(err)` on the 500 arm fails here.
"""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from modelmri.errors import BadRequest, Refusal
from modelmri.server import create_app

# Deliberately shaped like the exceptions this change is about: text nobody
# wrote for a user, carrying something from the machine underneath.
BROKE = (
    "CUDA out of memory. Tried to allocate 20.00 GiB "
    "(blobs/9f3c1a under the private cache)"
)


def app_with(method: str, raiser):
    """An app whose `runtime.<method>` raises whatever `raiser` raises."""
    app = create_app()
    setattr(app.state.runtime, method, raiser)
    return TestClient(app)


def boom(*_a, **_k):
    raise RuntimeError(BROKE)


# --------------------------------------------------------------- the base classes


def test_a_refusal_is_still_a_runtimeerror():
    """The property the migration rests on: an unconverted handler's
    `except RuntimeError` keeps catching a converted raise site."""
    # Asserted BEFORE the block: a `raise` as the last statement inside
    # `pytest.raises` reads as making everything after it unreachable,
    # because the analysis does not model the context manager swallowing
    # the exception. Same test, no dead-code warning.
    assert issubclass(Refusal, RuntimeError)
    with pytest.raises(RuntimeError):
        raise Refusal("no, and here is why")


def test_a_bad_request_is_still_a_valueerror():
    assert issubclass(BadRequest, ValueError)
    with pytest.raises(ValueError):
        raise BadRequest("layer must be in [0,12)")


def test_the_two_do_not_catch_each_other():
    """409 and 422 are different answers, so the types must not overlap."""
    assert not issubclass(Refusal, BadRequest)
    assert not issubclass(BadRequest, Refusal)


# ------------------------------------------------------------ through the server


def test_a_refusal_reaches_the_client_as_409_in_its_own_words():
    words = "This is a recording, and a `.mri` does not carry a model."

    def refuse(*_a, **_k):
        raise Refusal(words)

    r = app_with("features_summary", refuse).get("/api/features/summary")
    assert r.status_code == 409
    assert r.json()["error"] == words


def test_a_bad_request_reaches_the_client_as_422_in_its_own_words():
    words = "layer must be in [0,12)"

    def reject(*_a, **_k):
        raise BadRequest(words)

    r = app_with("features_summary", reject).get("/api/features/summary")
    assert r.status_code == 422
    assert r.json()["error"] == words


def test_an_unexpected_exception_is_a_500_that_does_not_quote_it():
    """THE assertion that stops someone putting `str(err)` back.

    A torch failure is not a refusal and not a conflict. The reader gets a
    sentence somebody wrote; torch's own text goes nowhere near the browser.
    """
    r = app_with("features_summary", boom).get("/api/features/summary")
    assert r.status_code == 500
    body = r.text
    assert BROKE not in body
    assert "GiB" not in body
    assert "blobs" not in body
    assert "RuntimeError" not in body
    assert "Traceback" not in body


def test_the_500_still_says_where_to_look():
    """Generic is not the same as useless. This is a local-first tool: the
    person reading the page started the process, so the terminal they already
    have is the honest place to send them."""
    r = app_with("features_summary", boom).get("/api/features/summary")
    assert "modelmri serve" in r.json()["error"]


def test_the_500_logs_the_traceback_rather_than_discarding_it(caplog):
    """The other half of not quoting the exception.

    If the text is removed from the response and not written anywhere, the
    failure has been erased rather than contained — and an erased failure is a
    worse bug than a leaked one.
    """
    with caplog.at_level(logging.ERROR, logger="modelmri"):
        app_with("features_summary", boom).get("/api/features/summary")

    records = [r for r in caplog.records if r.name == "modelmri"]
    assert records, "the 500 path recorded nothing"
    assert any(r.exc_info for r in records), "logged without the traceback"
    assert BROKE in caplog.text
    assert "/api/features/summary" in caplog.text


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("features_summary", "/api/features/summary"),
        ("feature_detail", "/api/features/3"),
        ("attention", "/api/attention?layer=0&head=0"),
        ("attention_diff", "/api/attention/diff"),
        ("ablate_heads", "/api/attention/ablate"),
        ("attribute_tokens", "/api/attention/attribute"),
        # The third ranking, added with the route rather than after it. It is
        # the one most likely to raise something that is not a Refusal — a CUDA
        # OOM in the middle of 518 forward passes — and its handler in
        # server.py has a `except Exception` arm that has to stay a logged 500
        # rather than drift back to a 409 carrying torch's own words.
        ("rank_features", "/api/features/ablate"),
        ("export_session", "/api/session/export"),
        ("set_steering", "/api/steer"),
    ],
)
def test_no_runtime_endpoint_leaks_a_broken_exception(method, path):
    """One case is an example; the rule is that none of them do it.

    Every route here is served entirely by runtime.py, so a RuntimeError
    arriving from one of them is torch or transformers rather than a decision
    ModelMRI made.

    This list used to carry a sentence excluding "routes that still pass
    through an unmigrated module", pointing at `server._unmigrated`. That
    helper is gone: it rested on a premise that was false for seven of the
    eight modules it named, and what it actually did was answer genuine
    breakage 409-with-raw-text on twelve routes while logging at a level
    nothing was listening to. The exclusion went with it, and the routes it
    was hiding are covered by the test below.
    """
    client = app_with(method, boom)
    r = client.post(path, json={}) if path == "/api/steer" else client.get(path)
    assert r.status_code == 500, f"{path} answered {r.status_code}"
    assert BROKE not in r.text
    assert "blobs" not in r.text


# --------------------------------------------- the routes _unmigrated used to hold
#
# Twelve arms caught a bare RuntimeError/ValueError and answered 409 or 422
# with the exception's own text. Measured before they came out, an internal
# `RuntimeError(BROKE)` reached the browser verbatim on every one. These are
# the same routes, driven the same way, through their real modules.


def app_where(patch, monkeypatch):
    """An app with one module-level function replaced by `boom`.

    `monkeypatch` is not optional decoration: two of these targets are module
    globals (`modelmri.hub.suggested`, `modelmri.ollama.pull`), and a plain
    setattr leaves them broken for every later test in the session. It did,
    and six unrelated tests failed before this took the fixture.

    The stub reader below is what makes this test say the same thing on every
    machine. `/api/vla/analyse` reads a frame before it calls the function
    under test, and `_reader()` needs both the `vla-lite` extra and a cached
    LeRobot dataset. On a developer box that has them the route reaches the
    patched function and answers 500, which is the thing being asserted; in CI
    it has neither, so an honest Refusal fires first and the route answers 409
    — passing locally and failing in CI, for a reason that has nothing to do
    with error handling. `_reader` short-circuits on `app.state.vla_reader`
    (server.py:629), so pre-seeding it puts every machine on the path the test
    is actually about.
    """
    import importlib

    app = create_app()

    class _StubReader:
        """Enough reader to get past the frame fetch. The frame is handed
        straight to the patched function, which raises, so its contents are
        never read."""

        def raw_frame(self, episode, t):
            return object()

    app.state.vla_reader = _StubReader()
    module, attr = patch
    if module == "state":
        target, attr = attr.split(".", 1)
        monkeypatch.setattr(getattr(app.state, target), attr, boom)
    elif module == "runtime":
        monkeypatch.setattr(app.state.runtime, attr, boom)
    else:
        monkeypatch.setattr(importlib.import_module(module), attr, boom)
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize(
    ("patch", "method", "path"),
    [
        (("modelmri.hub", "suggested"), "GET", "/api/hub/models"),
        (("modelmri.hub", "search"), "GET", "/api/hub/models?q=qwen"),
        (("modelmri.hub", "sign_in"), "POST", "/api/hub/signin"),
        (("modelmri.ollama", "pull"), "POST", "/api/ollama/pull"),
        (("runtime", "load_sae"), "POST", "/api/sae/load"),
        (("state", "vla.load"), "POST", "/api/vla/load"),
        (("state", "vla.attention"), "GET", "/api/vla/attention"),
        (("state", "vla.analyse"), "POST", "/api/vla/analyse"),
        (("state", "traces.import_trace"), "POST", "/api/traces/import"),
        (("runtime", "attention"), "GET", "/api/attention"),
        (("runtime", "open_session"), "POST", "/api/session/open"),
        (("runtime", "export_session"), "GET", "/api/session/export"),
    ],
)
def test_a_module_route_does_not_republish_a_broken_exception(
    patch, method, path, monkeypatch
):
    # `repo`/`hook` are here for /api/sae/load specifically. That route now
    # resolves an empty request against the registry — "the SAE for whatever
    # model is loaded" rather than one model's release answering for all of
    # them — so with nothing loaded it refuses BEFORE reaching `load_sae`,
    # and this test would then be asserting against a guard instead of the
    # thing it patched. Naming a repo skips the lookup and puts the patched
    # function back in the path. Every other route ignores the extra keys, as
    # they already do for the three above.
    body = {
        "token": "x",
        "name": "x",
        "prompt": "x",
        "repo": "someone/sae",
        "hook": "blocks.0.hook_resid_pre",
    }
    r = app_where(patch, monkeypatch).request(method, path, json=body)
    assert r.status_code == 500, f"{path} answered {r.status_code}"
    assert BROKE not in r.text
    assert "blobs" not in r.text


def test_those_routes_also_record_the_traceback(caplog, monkeypatch):
    """The half `_unmigrated` got wrong even where the status was arguable.

    It logged at `log.debug`, and this logger has no handler and an effective
    level of WARNING under what `modelmri serve` installs — so a torch failure
    on those routes was leaked to the browser AND erased from the terminal at
    once. Asserted at ERROR so a return to `debug` fails here.
    """
    with caplog.at_level(logging.ERROR, logger="modelmri"):
        app_where(("modelmri.hub", "suggested"), monkeypatch).get("/api/hub/models")
    records = [r for r in caplog.records if r.name == "modelmri"]
    assert records, "the 500 path recorded nothing"
    assert any(r.exc_info for r in records), "logged without the traceback"
    assert BROKE in caplog.text


def test_a_route_with_no_except_arm_still_answers_the_json_contract(monkeypatch):
    """28 of 56 routes have no `except` at all.

    Those answered Starlette's `text/plain` "Internal Server Error" — no
    `{"error": ...}`, no pointer at the terminal, for routes that talk to a
    network daemon or walk a filesystem and will realistically fail. One
    app-wide handler covers them rather than 28 more copies of the three arms.
    """
    import modelmri.ollama as _ollama

    app = create_app()
    monkeypatch.setattr(_ollama, "status", boom)
    r = TestClient(app, raise_server_exceptions=False).get("/api/ollama")
    assert r.status_code == 500
    assert r.headers["content-type"].startswith("application/json")
    assert "modelmri serve" in r.json()["error"]
    assert BROKE not in r.text


def test_the_taxonomy_answers_before_the_backstop_does():
    """The app-wide handler must not swallow a Refusal into a 500."""
    words = "This is a recording, and a `.mri` does not carry a model."

    def refuse(*_a, **_k):
        raise Refusal(words)

    app = create_app()
    app.state.runtime.features_summary = refuse  # per-instance, dies with the app
    r = TestClient(app, raise_server_exceptions=False).get("/api/features/summary")
    assert r.status_code == 409
    assert r.json()["error"] == words


# ------------------------------------------------ the two words, per module


def test_session_error_is_a_bad_request():
    """It was a plain ValueError, so the same sentence came back three ways:
    422 on /api/attention and /api/session/open through a transitional arm,
    and a generic 500 on /api/session/export, which never got one."""
    from modelmri.session import SessionError

    assert issubclass(SessionError, BadRequest)
    assert issubclass(SessionError, ValueError)


def test_an_unknown_baseline_is_a_bad_request_everywhere_it_is_checked():
    """errors.py names it as the type example of a BadRequest, and both
    checks used to answer 409 through a blanket AblationError/AttributionError
    wrap in runtime.py — 409 for the baseline, 422 for the layer index, on the
    same endpoint."""
    import torch

    from modelmri import ablate, attribute

    with pytest.raises(BadRequest):
        ablate.rank_heads(
            object(),
            lambda i: None,
            torch.zeros(1, 4, dtype=torch.long),
            position=0,
            layers=[0],
            n_heads=2,
            baseline="banana",
        )
    with pytest.raises(BadRequest):
        attribute.rank_tokens(
            object(), torch.zeros(1, 6, dtype=torch.long), position=4, baseline="banana"
        )
