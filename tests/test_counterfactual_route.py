# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""`POST /api/attention/counterfactual` — the route, not the search.

`tests/test_counterfactual.py` proves the measurement. This file proves the
three things a route around it can get wrong independently of whether the
search is any good:

  * answering 500 for a state somebody could have been told about. Every
    refusal below is a sentence naming what to do instead, and a 500 throws
    that sentence away and shows the reader "Internal Server Error";
  * inventing the missing half. The target has to be ONE token, and whether
    a word is one token is a property of the tokenizer, so it is measured and
    refused rather than silently truncated to the first piece;
  * losing the honesty fields between the module and the wire. `beats_controls`
    and an unmeasured control arm are the two things a reader acts on, and a
    route that drops them still answers 200.

The runtime method is replaced on the CLASS rather than on the app, because
each route closes over the `ModelRuntime` the app built for itself — patching
`app.state.runtime` would leave the closure pointing at the original and the
test would pass while testing nothing.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from modelmri.errors import BadRequest, Refusal
from modelmri.runtime import ModelRuntime
from modelmri.server import create_app


def _client() -> TestClient:
    return TestClient(create_app(), raise_server_exceptions=False)


PAYLOAD = {
    "position": 6,
    "found": True,
    "size": 1,
    "edit": [
        {
            "step": 1,
            "index": 3,
            "from_token_id": 12,
            "from_token": " Tower",
            "to_token_id": 9,
            "to_token": " Colosseum",
            "target_p_after": 0.81,
        }
    ],
    "edited_ids": [1, 4, 11, 9, 3, 5, 4],
    "beats_controls": True,
    "controls": {
        "same_positions": {
            "measured": True,
            "successes": 0,
            "samples": 24,
            "point": 0.0,
            "interval": [0.0, 0.13798],
            "confidence": 0.95,
        },
        "any_positions": {
            "measured": False,
            "successes": 0,
            "samples": 0,
            "point": None,
            "interval": None,
            "confidence": 0.95,
        },
        "measured": False,
        "not_measured_because": "every control draw was abandoned",
    },
}


# ------------------------------------------------------------ the happy path


def test_the_payload_reaches_the_wire_with_its_honesty_fields_intact(monkeypatch):
    monkeypatch.setattr(
        ModelRuntime, "token_counterfactual", lambda self, position, **kw: PAYLOAD
    )
    got = _client().post(
        "/api/attention/counterfactual", json={"position": 6, "target": "Rome"}
    )
    assert got.status_code == 200
    body = got.json()
    assert body["found"] is True
    assert body["beats_controls"] is True
    # The two fields a reader acts on. An arm with `measured: false` must keep
    # `point: null` across the wire — a JSON encoder that turned it into 0
    # would render as "we checked and it never happened", which is the
    # strongest possible support for a finding nobody measured.
    assert body["controls"]["any_positions"]["measured"] is False
    assert body["controls"]["any_positions"]["point"] is None
    assert body["controls"]["any_positions"]["interval"] is None
    assert body["controls"]["same_positions"]["successes"] == 0
    assert body["controls"]["same_positions"]["samples"] == 24


def test_the_request_forwards_every_knob_it_advertises(monkeypatch):
    seen = {}

    def fake(self, position, **kw):
        seen.update(kw)
        seen["position"] = position
        return PAYLOAD

    monkeypatch.setattr(ModelRuntime, "token_counterfactual", fake)
    _client().post(
        "/api/attention/counterfactual",
        json={
            "position": 4,
            "target": "Rome",
            "max_edits": 5,
            "n_proposals": 40,
            "n_controls": 12,
            "candidates": "pool",
            "seed": 7,
        },
    )
    assert seen["position"] == 4
    assert seen["target"] == "Rome"
    assert seen["max_edits"] == 5
    assert seen["n_proposals"] == 40
    assert seen["n_controls"] == 12
    assert seen["candidates"] == "pool"
    assert seen["seed"] == 7


# -------------------------------------------------------------- the refusals


def test_a_refusal_is_409_and_carries_its_sentence(monkeypatch):
    sentence = (
        "Generate something first, then ask what would make it say something else."
    )

    def fake(self, position, **kw):
        raise Refusal(sentence)

    monkeypatch.setattr(ModelRuntime, "token_counterfactual", fake)
    got = _client().post("/api/attention/counterfactual", json={"target": "Rome"})
    assert got.status_code == 409
    assert got.json()["error"] == sentence


def test_a_bad_request_is_422_and_carries_its_sentence(monkeypatch):
    sentence = "name the target once: either `target` as text or `target_token_id`."

    def fake(self, position, **kw):
        raise BadRequest(sentence)

    monkeypatch.setattr(ModelRuntime, "token_counterfactual", fake)
    got = _client().post("/api/attention/counterfactual", json={})
    assert got.status_code == 422
    assert got.json()["error"] == sentence


def test_an_unexpected_error_does_not_leak_and_does_not_500_silently(monkeypatch):
    def fake(self, position, **kw):
        raise ZeroDivisionError("this string must not reach the browser")

    monkeypatch.setattr(ModelRuntime, "token_counterfactual", fake)
    got = _client().post("/api/attention/counterfactual", json={"target": "Rome"})
    assert got.status_code == 500
    assert "must not reach the browser" not in got.text


# ----------------------------------------------------------------- the cost


def test_the_cost_route_prices_before_anything_is_spent(monkeypatch):
    priced = {"shortest": 115, "longest_search": 171, "most_expensive": 219}
    monkeypatch.setattr(
        ModelRuntime, "counterfactual_cost", lambda self, position, **kw: priced
    )
    got = _client().post(
        "/api/attention/counterfactual/cost", json={"position": 9, "max_edits": 3}
    )
    assert got.status_code == 200
    assert got.json()["most_expensive"] == 219


def test_the_cost_route_refuses_rather_than_pricing_an_impossible_run(monkeypatch):
    def fake(self, position, **kw):
        raise BadRequest("there is nothing editable before position 1.")

    monkeypatch.setattr(ModelRuntime, "counterfactual_cost", fake)
    got = _client().post("/api/attention/counterfactual/cost", json={"position": 1})
    assert got.status_code == 422
    assert "nothing editable" in got.json()["error"]


# ------------------------------------------------- naming the target token


class OneTokenizer:
    """A tokenizer where " Rome" is one piece and "Colosseum" is three."""

    PIECES = {
        " Rome": [21718],
        "Rome": [51, 638],
        " Colosseum": [3, 4, 5],
        "Colosseum": [3, 4, 5],
    }

    def __call__(self, text, add_special_tokens=True):
        ids = self.PIECES.get(text, [7])
        return type("Enc", (), {"input_ids": ids})()

    def decode(self, ids):
        return "|".join(str(i) for i in ids)


def _runtime_with(tokenizer):
    rt = ModelRuntime()
    rt.tokenizer = tokenizer
    return rt


def test_a_single_token_target_resolves_to_its_id():
    rt = _runtime_with(OneTokenizer())
    # The spaced form wins: mid-sentence the model predicts " Rome", and
    # `Rome` without the space is a different id it may never emit.
    assert rt.resolve_target_token("Rome") == 21718
    assert rt.resolve_target_token(" Rome") == 21718


def test_a_multi_token_target_is_refused_with_its_pieces_named():
    rt = _runtime_with(OneTokenizer())
    with pytest.raises(BadRequest) as caught:
        rt.resolve_target_token("Colosseum")
    said = caught.value.sentence
    assert "not a single token" in said
    # The refusal has to name the pieces, or the reader cannot act on it.
    assert "cuts into" in said
    assert "tokenizer, not" in said


def test_an_empty_target_is_refused_before_the_tokenizer_sees_it():
    rt = _runtime_with(OneTokenizer())
    for bad in ("", "   ", None, 7):
        with pytest.raises(BadRequest, match="name the token"):
            rt.resolve_target_token(bad)
