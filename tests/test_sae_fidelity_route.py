"""`/api/sae/fidelity` and its two prices — the surface, not the measurement.

`tests/test_sae_fidelity.py` proves `saes.ce_recovered` itself. This file
proves the things a route, a runtime method and a CLI verb around it can get
wrong independently of whether the measurement is any good:

  * putting the route under `/api/features/`. `/api/features/{feature_id}` has
    no path converter, so its regex swallows any single segment and a sibling
    declared after it answers 422 for a perfectly well formed request. Worse in
    the browser: `demo.ts` answers `/api/features/` as a PREFIX with the
    single-feature detail payload, at 200, so a fidelity card in a demo build
    would render a fabricated percentage as a measurement.
  * reintroducing a default floor. The module refuses to have one because the
    same reconstruction scores differently against the two floors, and a
    pydantic default, a query-string default or a pre-selected option would put
    the choice back without telling the reader it was made.
  * spending the corpus before anybody was told what it costs. `3n + 2` is
    exact and free to compute; a corpus file nobody counted can be twenty
    thousand lines.
  * turning a path in the body into a file read for whoever can reach the port.
  * losing "not measured" on the way to the wire. `n_floor_tokens` is None for
    the zero floor and must stay None: 0 would claim a mean taken over an empty
    corpus.

The runtime method is replaced on the CLASS rather than on the app, because
each route closes over the `ModelRuntime` the app built for itself — patching
`app.state.runtime` would leave the closure pointing at the original and the
test would pass while testing nothing.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
torch = pytest.importorskip("torch")

from fastapi.testclient import TestClient  # noqa: E402

from modelmri import (  # noqa: E402
    cli,
    paths,
    saes,
)
from modelmri import corpus_index as ci  # noqa: E402
from modelmri.errors import BadRequest, Refusal  # noqa: E402
from modelmri.runtime import ModelRuntime  # noqa: E402
from modelmri.server import create_app  # noqa: E402

#: The person at the keyboard. Handed in explicitly because `_is_loopback`
#: refuses a host that is not an IP address at all, and TestClient's default
#: peer is `testclient` — a name, not an address, so every corpus-path route
#: answers 403 unless the test says where it is calling from.
HERE = ("127.0.0.1", 5000)

#: A request that did not come from the person at the keyboard.
REMOTE = ("203.0.113.9", 51234)


def _client() -> TestClient:
    return TestClient(create_app(), client=HERE, raise_server_exceptions=False)


# Built through the real dataclasses rather than written out as a literal, so a
# field added to `CEFidelity` fails here at construction instead of quietly
# never being asserted on. `to_dict()` is what the route returns.
CALIBRATION = saes.SAECalibration(
    convention="centered+b_dec",
    center=True,
    subtract_b_dec=True,
    fvu=0.0731,
    l0=41.2,
    rel_err=0.2704,
    n_tokens=512,
    declared_b_dec=True,
    ranked=[("centered+b_dec", 0.0731), ("centered", 0.1902)],
    usable=True,
)

FIDELITY = saes.CEFidelity(
    repo="jbloom/GPT2-Small-SAEs-Reformatted",
    hook="blocks.8.hook_resid_pre",
    layer=8,
    point="resid_pre",
    floor=saes.FLOOR_ZERO,
    # The zero floor is not averaged from anything, so the token count under it
    # is None and never 0.
    floor_means="The floor is zero-ablation: at every token the stream was "
    "replaced by the zero vector.",
    n_floor_tokens=None,
    ce_clean=3.102,
    ce_recon=3.401,
    ce_ablate=6.884,
    numerator=3.483,
    denominator=3.782,
    ce_recovered=0.920941,
    corpus_label="notes.txt",
    corpus_sha256="a" * 64,
    n_sequences=10,
    n_sequences_given=10,
    truncated=False,
    n_tokens=812,
    n_tokens_seen=822,
    calibration=CALIBRATION,
    calibrated_here=True,
    replay_deviation_nats=0.0,
    splice_deviation_nats=1.2e-7,
    passes=32,
    elapsed_s=4.81,
)

PAYLOAD = {
    **FIDELITY.to_dict(),
    "receipt": {"op": "sae_fidelity", "request": {"floor": saes.FLOOR_ZERO}},
}


# ------------------------------------------------------------ the happy path


def test_the_payload_reaches_the_wire_with_every_field_it_carries(monkeypatch):
    monkeypatch.setattr(ModelRuntime, "sae_fidelity", lambda self, texts, **kw: PAYLOAD)
    got = _client().post(
        "/api/sae/fidelity",
        json={
            "texts": ["one line", "another"],
            "label": "pasted text",
            "floor": saes.FLOOR_ZERO,
        },
    )
    assert got.status_code == 200, got.text
    body = got.json()

    # Every published field, by name. The three losses are the point: a ratio
    # taken against another floor is a different number, and only these make it
    # undoable.
    for field in (
        "repo",
        "hook",
        "layer",
        "point",
        "floor",
        "floor_means",
        "n_floor_tokens",
        "ce_clean",
        "ce_recon",
        "ce_ablate",
        "numerator",
        "denominator",
        "ce_recovered",
        "corpus_label",
        "corpus_sha256",
        "n_sequences",
        "n_sequences_given",
        "truncated",
        "n_tokens",
        "n_tokens_seen",
        "calibration",
        "calibrated_here",
        "replay_deviation_nats",
        "splice_deviation_nats",
        "passes",
        "elapsed_s",
        "means",
        "receipt",
    ):
        assert field in body, f"{field} never reached the browser"

    assert body["floor"] == saes.FLOOR_ZERO
    # NOT 0. The zero floor is not a mean of anything, and 0 here would claim a
    # mean taken over an empty corpus.
    assert body["n_floor_tokens"] is None
    # The activation-space half travels whole, because `fvu` and `l0` beside
    # the percentage are what say whether the splice was worth reading.
    assert body["calibration"]["fvu"] == CALIBRATION.fvu
    assert body["calibration"]["l0"] == CALIBRATION.l0
    assert body["receipt"]["op"] == "sae_fidelity"


def test_the_request_forwards_every_knob_it_advertises(monkeypatch):
    seen: dict = {}

    def fake(self, texts, **kw):
        seen["texts"] = texts
        seen.update(kw)
        return PAYLOAD

    monkeypatch.setattr(ModelRuntime, "sae_fidelity", fake)
    _client().post(
        "/api/sae/fidelity",
        json={
            "texts": ["a b c", "d e f"],
            "label": "pasted text",
            "floor": saes.FLOOR_MEAN,
            "max_sequences": 1,
            "confirm": True,
        },
    )
    assert seen["texts"] == ["a b c", "d e f"]
    assert seen["floor"] == saes.FLOOR_MEAN
    assert seen["corpus_label"] == "pasted text"
    assert seen["max_sequences"] == 1
    assert seen["confirm"] is True


# ------------------------------------------------------------- the two floors


def test_the_floor_has_no_default_and_a_body_without_one_is_refused(monkeypatch):
    """The module's whole argument, arriving at the request model.

    'mean_ablate' and 'zero_ablate' give different percentages for the same
    reconstruction, so there is no house answer to a question the reader has to
    be told the answer to. A pydantic default would put one back silently.
    """
    called = False

    def fake(self, texts, **kw):
        nonlocal called
        called = True
        return PAYLOAD

    monkeypatch.setattr(ModelRuntime, "sae_fidelity", fake)
    got = _client().post("/api/sae/fidelity", json={"texts": ["a b"]})
    assert got.status_code == 422, got.text
    assert "floor" in got.text
    assert not called, "a missing floor reached the measurement"


def test_an_unknown_floor_is_the_modules_own_sentence(monkeypatch):
    def fake(self, texts, **kw):
        raise BadRequest(
            f"unknown floor {kw['floor']!r} — CE-recovered normalises against "
            f"one of {', '.join(saes.CE_FLOORS)}, and there is no default."
        )

    monkeypatch.setattr(ModelRuntime, "sae_fidelity", fake)
    got = _client().post(
        "/api/sae/fidelity",
        json={"texts": ["a b"], "label": "pasted text", "floor": "mean-ablate"},
    )
    assert got.status_code == 422
    assert "there is no default" in got.json()["error"]


# ----------------------------------------------------------------- the price


def test_the_estimate_is_three_passes_per_sequence_plus_two():
    """The cost function and the loop it prices must not drift apart.

    Asserted against `ce_recovered_passes` AND against the arithmetic, because
    a route that called the wrong estimator would still agree with itself.
    """
    got = _client().get("/api/sae/fidelity/estimate?sequences=10")
    assert got.status_code == 200, got.text
    body = got.json()
    assert body["passes"] == 32
    assert body["passes"] == saes.ce_recovered_passes(10)
    assert body["n_sequences"] == 10
    assert "32" in body["means"]


def test_the_estimate_costs_nothing_and_needs_no_model():
    """`3n + 2` is arithmetic. It answers on a server with nothing loaded,
    which is what lets a panel quote a price before anything runs."""
    client = _client()
    assert client.get("/api/sae").json()["loaded"] is False
    assert client.get("/api/sae/fidelity/estimate?sequences=1").json()["passes"] == 5


def test_the_estimate_refuses_a_corpus_of_nothing():
    got = _client().get("/api/sae/fidelity/estimate?sequences=0")
    assert got.status_code == 422
    assert "nothing to price" in got.json()["error"]


def test_the_estimate_says_when_the_run_will_need_confirming():
    """The gate is published by the same route that prices it.

    A panel that had to guess the threshold would be a second copy of it, and
    the copy is what drifts.
    """
    small = _client().get("/api/sae/fidelity/estimate?sequences=1").json()
    assert small["needs_confirmation"] is False
    assert small["confirm_above"] == saes.CE_CONFIRM_ABOVE_PASSES

    over = saes.CE_CONFIRM_ABOVE_PASSES  # 3n + 2 is above the gate for n = it
    big = _client().get(f"/api/sae/fidelity/estimate?sequences={over}").json()
    assert big["needs_confirmation"] is True


# ------------------------------------------------------------- the refusals


def test_a_refusal_is_409_and_carries_its_sentence(monkeypatch):
    sentence = (
        "No SAE loaded, so there is no reconstruction to score. Load one "
        "against this model first."
    )

    def fake(self, texts, **kw):
        raise Refusal(sentence)

    monkeypatch.setattr(ModelRuntime, "sae_fidelity", fake)
    got = _client().post(
        "/api/sae/fidelity",
        json={"texts": ["a b"], "label": "pasted text", "floor": saes.FLOOR_ZERO},
    )
    assert got.status_code == 409
    assert got.json()["error"] == sentence


def test_a_bad_request_is_422_and_carries_its_sentence(monkeypatch):
    sentence = "sequence 2 is 1 token(s) long."

    def fake(self, texts, **kw):
        raise BadRequest(sentence)

    monkeypatch.setattr(ModelRuntime, "sae_fidelity", fake)
    got = _client().post(
        "/api/sae/fidelity",
        json={"texts": ["a b"], "label": "pasted text", "floor": saes.FLOOR_ZERO},
    )
    assert got.status_code == 422
    assert got.json()["error"] == sentence


def test_an_unexpected_error_does_not_leak_and_does_not_500_silently(monkeypatch):
    def fake(self, texts, **kw):
        raise ZeroDivisionError("this string must not reach the browser")

    monkeypatch.setattr(ModelRuntime, "sae_fidelity", fake)
    got = _client().post(
        "/api/sae/fidelity",
        json={"texts": ["a b"], "label": "pasted text", "floor": saes.FLOOR_ZERO},
    )
    assert got.status_code == 500
    assert "must not reach the browser" not in got.text


def test_a_corpus_with_no_text_in_it_is_refused_before_the_model(monkeypatch):
    called = False

    def fake(self, texts, **kw):
        nonlocal called
        called = True
        return PAYLOAD

    monkeypatch.setattr(ModelRuntime, "sae_fidelity", fake)
    got = _client().post("/api/sae/fidelity", json={"floor": saes.FLOOR_ZERO})
    assert got.status_code == 422
    assert "Nothing is downloaded" in got.json()["error"]
    assert not called


def test_pasted_text_has_to_be_named_and_the_refusal_names_the_field(monkeypatch):
    """A loss is a loss ON something, and the something has to be sayable.

    `ce_recovered` refuses this too, but it names `corpus_label` — the Python
    parameter — and the field on the wire is `label`. A refusal that names a
    parameter the caller cannot see is one they cannot act on.
    """
    called = False

    def fake(self, texts, **kw):
        nonlocal called
        called = True
        return PAYLOAD

    monkeypatch.setattr(ModelRuntime, "sae_fidelity", fake)
    got = _client().post(
        "/api/sae/fidelity", json={"texts": ["a b"], "floor": saes.FLOOR_ZERO}
    )
    assert got.status_code == 422, got.text
    assert "`label`" in got.json()["error"]
    assert not called


def test_a_recording_refuses_because_a_mri_carries_no_model():
    app = create_app()
    app.state.runtime.replay = object()
    got = TestClient(app, client=HERE, raise_server_exceptions=False).post(
        "/api/sae/fidelity",
        json={"texts": ["a b"], "label": "pasted text", "floor": saes.FLOOR_ZERO},
    )
    assert got.status_code == 409, got.text
    assert "recording" in got.json()["error"]


def test_without_a_model_there_is_nothing_to_run_against():
    """A resting server refuses at the FIRST guard, which is the model.

    Named for the arm it actually exercises. `sae_fidelity` checks the model
    before the SAE, so a test that posted to a resting app under the SAE's name
    would be reporting on this arm instead — and would go on passing with the
    SAE guard deleted.
    """
    got = _client().post(
        "/api/sae/fidelity",
        json={"texts": ["a b"], "label": "pasted text", "floor": saes.FLOOR_ZERO},
    )
    assert got.status_code == 409, got.text
    assert "No model loaded" in got.json()["error"]


def test_without_an_sae_there_is_nothing_to_reconstruct():
    """With a model resident and no SAE, the refusal is the SAE's own.

    Asserted on wording only that sentence has. Without the guard this is not a
    refusal at all: `saes.ce_recovered` would be handed `sae=None`, and
    `getattr(None, "layer", 0)` answers 0 rather than raising, so the failure
    would surface as a 500 from inside the measurement.
    """
    app = create_app()
    app.state.runtime.backend = "hf"
    app.state.runtime.model = object()
    app.state.runtime.sae = None
    got = TestClient(app, client=HERE, raise_server_exceptions=False).post(
        "/api/sae/fidelity",
        json={"texts": ["a b"], "label": "pasted text", "floor": saes.FLOOR_ZERO},
    )
    assert got.status_code == 409, got.text
    assert "no reconstruction to score" in got.json()["error"]


def test_the_route_forbids_a_key_it_does_not_know():
    """`{"flor": "zero_ablate"}` must not run against something else.

    A body model that accepted the typo would report a percentage labelled
    with a floor the caller did not ask for.
    """
    got = _client().post(
        "/api/sae/fidelity",
        json={
            "texts": ["a b"],
            "label": "pasted text",
            "floor": saes.FLOOR_ZERO,
            "flor": "mean_ablate",
        },
    )
    assert got.status_code == 422


# ------------------------------------------------------------ the budget gate


class _Tok:
    """Enough tokenizer to turn a line into ids, and nothing else."""

    def __call__(self, text, return_tensors=None):
        # One id per word, and a blank line tokenizes to one — which is short
        # enough for `_sequence_for_ce` to refuse, which is the point.
        n = max(1, len(str(text).split()))
        return {"input_ids": torch.zeros(1, n, dtype=torch.long)}


class _Sae:
    """An SAE handle for the two fields the gate path reads."""

    layer = 0
    repo = "synthetic/gate"
    hook = "blocks.0.hook_resid_pre"


def _gated_runtime() -> ModelRuntime:
    """A runtime that gets as far as the price, and refuses recognisably after.

    `object()` publishes none of the decoder-block layouts `_block` walks, so
    everything past the gate ends in the "could not find this model's decoder
    blocks" refusal — a sentence nothing else here produces, which is what lets
    the tests below tell "the gate fired" from "something else did".
    """
    rt = ModelRuntime()
    rt.replay = None
    rt.backend = "hf"
    rt.model = object()
    rt.sae = _Sae()
    rt.tokenizer = _Tok()
    rt.device = "cpu"
    return rt


def _corpus_over_the_gate() -> list[str]:
    n = (saes.CE_CONFIRM_ABOVE_PASSES // 3) + 1
    return ["a b c"] * n


def test_a_corpus_nobody_priced_is_refused_rather_than_started():
    rt = _gated_runtime()
    with pytest.raises(Refusal) as caught:
        rt.sae_fidelity(
            _corpus_over_the_gate(), floor=saes.FLOOR_ZERO, corpus_label="big.txt"
        )
    said = caught.value.sentence
    # The count, the flag and the cheaper alternative. A refusal that names the
    # problem and no next step is a wall.
    assert "forward passes" in said
    assert "confirm" in said
    assert "max_sequences" in said


def test_confirming_gets_past_the_gate_and_not_past_anything_else():
    """The gate is the only thing `confirm` turns off.

    Asserted on the sentence of the refusal that comes AFTER it rather than on
    "some refusal was raised": a method that refuses everything would pass the
    test above, which is the trap this repo has walked into three times.
    """
    rt = _gated_runtime()
    with pytest.raises(Refusal) as caught:
        rt.sae_fidelity(
            _corpus_over_the_gate(),
            floor=saes.FLOOR_ZERO,
            corpus_label="big.txt",
            confirm=True,
        )
    assert "decoder blocks" in caught.value.sentence, caught.value.sentence


def test_capping_the_corpus_reprices_it_under_the_gate():
    """`max_sequences` is the alternative the refusal names, so it has to work.

    A cap that did not change the price would make the refusal's advice false.
    """
    rt = _gated_runtime()
    with pytest.raises(Refusal) as caught:
        rt.sae_fidelity(
            _corpus_over_the_gate(),
            floor=saes.FLOOR_ZERO,
            corpus_label="big.txt",
            max_sequences=4,
        )
    assert "decoder blocks" in caught.value.sentence, caught.value.sentence


def test_the_route_refuses_a_corpus_nobody_priced_and_confirm_is_the_way_past():
    """The gate, through HTTP, which is where the corpus nobody counted arrives.

    The three tests above hold `ModelRuntime.sae_fidelity` to it directly, and
    that leaves the route's own two lines — the `confirm` field's default and
    the argument it is forwarded as — asserted by nothing: flipping either to
    `True` auto-confirms every browser request while the runtime's own default
    stays innocent. This drives the real app, so both are covered, and the
    second half asserts on the refusal that comes AFTER the gate rather than on
    "some refusal", so a route that refused everything could not pass it.
    """
    app = create_app()
    rt = app.state.runtime
    rt.replay = None
    rt.backend = "hf"
    rt.model = object()
    rt.sae = _Sae()
    rt.tokenizer = _Tok()
    rt.device = "cpu"
    client = TestClient(app, client=HERE, raise_server_exceptions=False)
    body = {
        "texts": _corpus_over_the_gate(),
        "label": "big.txt",
        "floor": saes.FLOOR_ZERO,
    }

    got = client.post("/api/sae/fidelity", json=body)
    assert got.status_code == 409, got.text
    said = got.json()["error"]
    assert "forward passes" in said
    assert "max_sequences" in said

    got = client.post("/api/sae/fidelity", json={**body, "confirm": True})
    assert got.status_code == 409, got.text
    assert "decoder blocks" in got.json()["error"], got.text


def test_nothing_is_dropped_between_the_reader_and_the_refusal():
    """`_sequence_for_ce` refuses by POSITIONAL INDEX, so the list has to keep
    counting the lines the reader typed.

    `head_corpus` skips an empty sequence (`if ids.shape[-1] == 0: continue`),
    which is right for a sweep that reports a total and wrong here: every index
    after the skip would name a different line than the sentence claims.
    """
    rt = _gated_runtime()
    ids = rt._tokenized_for_ce(["a b c", "", "d e"])
    assert len(ids) == 3, "a line was dropped, so every index after it is a lie"
    with pytest.raises(BadRequest) as caught:
        saes._sequence_for_ce(ids[1], 1)
    assert "sequence 1" in caught.value.sentence


class _Blocks(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2)])


class _Model(torch.nn.Module):
    """A layout `decoder_blocks` recognises, and nothing else."""

    def __init__(self):
        super().__init__()
        self.model = _Blocks()


def test_the_orchestration_hands_over_a_reiterable_list_and_keeps_a_receipt(
    monkeypatch,
):
    """What `runtime.sae_fidelity` is FOR, with the measurement stubbed out.

    `sequences` is read twice inside `ce_recovered` — the mean floor needs a
    vector averaged over the whole corpus, which is not known until every
    sequence has been read once — so a generator would make the second sweep
    silently empty. And the SAE is as much a part of this measurement as the
    model is, so it is named in the receipt beside it.
    """
    captured: dict = {}

    def fake_ce(model, block, sequences, sae, **kw):
        # Read twice, deliberately: a generator would answer 2 then 0.
        captured["first"] = len(list(sequences))
        captured["second"] = len(list(sequences))
        captured.update(kw)
        return FIDELITY

    monkeypatch.setattr(saes, "ce_recovered", fake_ce)
    rt = _gated_runtime()
    rt.model = _Model()
    out = rt.sae_fidelity(
        ["a b", "c d"],
        floor=saes.FLOOR_ZERO,
        corpus_label="notes.txt",
        max_sequences=2,
    )
    assert captured["first"] == 2
    assert captured["second"] == 2, "the corpus could only be swept once"
    assert captured["floor"] == saes.FLOOR_ZERO
    assert captured["corpus_label"] == "notes.txt"
    assert captured["max_sequences"] == 2
    assert out["ce_recovered"] == FIDELITY.ce_recovered
    assert out["receipt"]["op"] == "sae_fidelity"


def test_the_cost_preflight_tokenizes_only_the_sequence_it_probes(monkeypatch):
    """A preflight that costs O(corpus) is not a preflight.

    `estimate_ce_recovered_cost` probes ONE sequence — representative here
    means the LENGTH — so tokenizing the rest would put the whole corpus on the
    accelerator to answer a question about one pass over one line of it, and
    could OOM ahead of the measurement it exists to price.
    """
    seen: dict = {}

    def fake_cost(model, block, ids, sae, **kw):
        seen.update(kw)
        seen["length"] = int(ids.shape[-1])
        return {"passes": kw["n_sequences"] * 3 + 2, "means": "priced"}

    monkeypatch.setattr(saes, "estimate_ce_recovered_cost", fake_cost)

    class _Counting(_Tok):
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, text, return_tensors=None):
            self.calls += 1
            return super().__call__(text, return_tensors=return_tensors)

    rt = _gated_runtime()
    rt.model = _Model()
    rt.tokenizer = _Counting()
    out = rt.sae_fidelity_cost(["one two three"] * 50)

    assert rt.tokenizer.calls == 1, (
        f"{rt.tokenizer.calls} sequences were tokenized onto the device to "
        f"probe one of them"
    )
    # And the count it prices is still the whole corpus, which is the number
    # the reader is about to spend.
    assert seen["n_sequences"] == 50
    assert out["passes"] == 152


# ------------------------------------------------- ids, and never a path


@pytest.fixture
def only_root(tmp_path, monkeypatch):
    """`tmp_path` as the one corpus root, and nothing else.

    The same fixture `tests/test_corpus_index.py` uses, for the same reason:
    the roots are the working directory, the home directory and temp, and a
    test that left them alone would be reading the developer's own disk.
    """
    import tempfile

    monkeypatch.setattr(paths, "_home", lambda: None)
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path / "nowhere"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MODELMRI_CORPUS_DIRS", raising=False)
    ci._INDEX.clear()
    yield tmp_path
    ci._INDEX.clear()


def test_a_corpus_id_from_the_listing_opens_the_file_it_stands_for(
    only_root, monkeypatch
):
    """The id is a dictionary key, never a path this route builds.

    `GET /api/corpus/available` mints the ids by walking the roots; this proves
    the fidelity route resolves one of those back to the same file, and that
    the corpus is named by the file rather than by whatever the caller called
    it.
    """
    (only_root / "notes.txt").write_text("one two\nthree four\n", encoding="utf-8")
    listing = ci.scan()
    chosen = [c for c in listing.corpora if c.relative.endswith("notes.txt")]
    assert chosen, listing.to_dict()

    seen: dict = {}

    def fake(self, texts, **kw):
        seen["texts"] = texts
        seen.update(kw)
        return PAYLOAD

    monkeypatch.setattr(ModelRuntime, "sae_fidelity", fake)
    got = TestClient(create_app(), client=HERE, raise_server_exceptions=False).post(
        "/api/sae/fidelity",
        json={
            "file": chosen[0].id,
            "label": "our word for it",
            "floor": saes.FLOOR_ZERO,
        },
    )
    assert got.status_code == 200, got.text
    assert seen["texts"] == ["one two", "three four"]
    # The FILE names the corpus. Passing the caller's label through would put
    # our word for it on somebody else's measurement.
    assert seen["corpus_label"] == "notes.txt"


def test_a_path_from_another_machine_is_refused_before_it_is_opened(only_root):
    """A path in the body names a file on the SERVER's disk.

    `serve` defaults to loopback but `--host` takes anything, so without this
    the route is an arbitrary file read for anyone who can reach the port.
    """
    got = TestClient(create_app(), client=REMOTE).post(
        "/api/sae/fidelity",
        json={"file": "/etc/passwd", "floor": saes.FLOOR_ZERO},
    )
    assert got.status_code == 403, got.text
    assert "only possible from this machine" in got.json()["error"]


def test_a_path_outside_the_roots_is_refused_by_name(only_root):
    outside = only_root.parent / "escaped.txt"
    outside.write_text("secret\n", encoding="utf-8")
    try:
        got = _client().post(
            "/api/sae/fidelity",
            json={"file": str(outside), "floor": saes.FLOOR_ZERO},
        )
        assert got.status_code == 422, got.text
        assert "outside the directories" in got.json()["error"]
    finally:
        outside.unlink(missing_ok=True)


# ------------------------------------------- the measured price of one pass


def test_the_measured_cost_route_prices_the_corpus_it_was_given(monkeypatch):
    """`ce_recovered_passes` transfers between machines; seconds do not.

    So the second estimator runs one real iteration here and projects from it,
    and it reports the length it probed, because a pass over 64 tokens does not
    price a pass over 512.
    """
    priced = {
        "estimate": {"passes": 32, "seconds": 3.2, "verdict": "ok"},
        "probe": {"seconds": 0.1},
        "passes": 32,
        "n_sequences": 10,
        "probed_sequence_length": 81,
        "means": "32 forward passes — three per sequence plus two.",
    }
    monkeypatch.setattr(
        ModelRuntime, "sae_fidelity_cost", lambda self, texts, **kw: priced
    )
    got = _client().post("/api/sae/fidelity/cost", json={"texts": ["a b", "c d"]})
    assert got.status_code == 200, got.text
    assert got.json()["passes"] == 32
    assert got.json()["probed_sequence_length"] == 81


def test_the_measured_cost_route_relays_a_refusal_rather_than_500(monkeypatch):
    def fake(self, texts, **kw):
        raise Refusal("No model loaded — pick one first.")

    monkeypatch.setattr(ModelRuntime, "sae_fidelity_cost", fake)
    got = _client().post("/api/sae/fidelity/cost", json={"texts": ["a b"]})
    assert got.status_code == 409
    assert got.json()["error"] == "No model loaded — pick one first."


def test_the_measured_cost_route_carries_the_same_path_boundary(only_root):
    got = TestClient(create_app(), client=REMOTE).post(
        "/api/sae/fidelity/cost", json={"file": "/etc/passwd"}
    )
    assert got.status_code == 403, got.text
    assert "only possible from this machine" in got.json()["error"]


# --------------------------------------------------------- the route's name


def test_the_route_is_not_under_api_features():
    """`/api/features/{feature_id}` has no path converter.

    A fidelity route declared under `/api/features/` would be swallowed by it
    and answer 422 for a well-formed request — and in a demo build `demo.ts`'s
    `/api/features/` prefix would answer 200 with a single feature's detail
    payload, which is a fabricated percentage rendered as a measurement.
    """
    app = create_app()
    declared = {getattr(r, "path", "") for r in app.routes}
    assert "/api/sae/fidelity" in declared
    assert "/api/sae/fidelity/cost" in declared
    assert "/api/sae/fidelity/estimate" in declared
    assert not [p for p in declared if p.startswith("/api/features/fidelity")]


def test_the_frontend_never_asks_for_fidelity_under_api_features():
    src = Path(__file__).resolve().parents[1] / "frontend" / "src"
    text = "".join(f.read_text("utf-8") for f in src.glob("*.ts*") if os.path.isfile(f))
    assert "/api/sae/fidelity" in text, "nothing in the page can reach the route"
    assert "/api/features/fidelity" not in text


# ------------------------------------------------------------------- the CLI


def _cli_corpus(tmp_path, n_lines: int) -> str:
    path = tmp_path / "corpus.txt"
    path.write_text("\n".join(["one two three"] * n_lines) + "\n", encoding="utf-8")
    return str(path)


def _over_the_gate() -> int:
    return (saes.CE_CONFIRM_ABOVE_PASSES // 3) + 1


def test_the_command_prints_the_price_before_it_loads_anything(
    tmp_path, capsys, monkeypatch
):
    """The projection comes out BEFORE the model does.

    `run_sweep` cannot manage that — `sweep.plan` needs a loaded runtime — but
    `ce_recovered_passes` is arithmetic, so there is no reason to make somebody
    wait through a multi-gigabyte load to be told the corpus is too big.
    """

    def never(self, *args, **kw):
        raise AssertionError("the model was loaded before the price was refused")

    monkeypatch.setattr(ModelRuntime, "load", never)
    code = cli.sae_fidelity(
        _cli_corpus(tmp_path, _over_the_gate()), model="tiny", floor=saes.FLOOR_ZERO
    )
    assert code == 2
    said = capsys.readouterr().err
    assert "forward passes" in said
    assert "--yes" in said


def test_yes_gets_the_command_past_the_gate(tmp_path, capsys, monkeypatch):
    """And past nothing else — `--yes` is the confirmation, not a second meaning."""
    seen: dict = {}

    monkeypatch.setattr(ModelRuntime, "load", lambda self, *a, **kw: None)
    monkeypatch.setattr(ModelRuntime, "unload", lambda self, *a, **kw: None)
    monkeypatch.setattr(
        ModelRuntime,
        "load_sae",
        lambda self, repo, hook, **kw: seen.update({"repo": repo, "hook": hook}),
    )

    def fake(self, texts, **kw):
        seen["n"] = len(texts)
        seen.update(kw)
        return PAYLOAD

    monkeypatch.setattr(ModelRuntime, "sae_fidelity", fake)
    n = _over_the_gate()
    code = cli.sae_fidelity(
        _cli_corpus(tmp_path, n),
        model="tiny",
        floor=saes.FLOOR_ZERO,
        repo="synthetic/x",
        hook="blocks.0.hook_resid_pre",
        yes=True,
    )
    said = capsys.readouterr()
    assert code == 0, said.err
    assert seen["n"] == n
    assert seen["floor"] == saes.FLOOR_ZERO
    assert seen["corpus_label"] == "corpus.txt"
    # The command already confirmed, in the same words, before the load.
    assert seen["confirm"] is True
    assert "CE-recovered" in said.out
    # An unknown is not a zero: the zero floor is averaged over nothing, and the
    # command says so rather than printing a count of none.
    assert "averaged over nothing" in said.out


def test_a_corpus_the_command_cannot_read_is_a_sentence_not_a_traceback(
    tmp_path, capsys
):
    code = cli.sae_fidelity(
        str(tmp_path / "nope.txt"), model="tiny", floor=saes.FLOOR_ZERO
    )
    assert code == 2
    assert "nope.txt" in capsys.readouterr().err


def test_a_cap_of_nothing_is_a_sentence_not_a_traceback(tmp_path, capsys):
    """`--max-sequences 0` names the flag rather than raising through main().

    `ce_recovered_passes` refuses a corpus of nothing with a `BadRequest`,
    which is a `ValueError` and so is not caught by the `Refusal` arm around
    the gate — it escaped as a traceback with this machine's paths in it. The
    corpus is not what is wrong here either: the file may hold plenty, and the
    cap is what emptied it, so the sentence names the flag the reader typed.
    """
    code = cli.sae_fidelity(
        _cli_corpus(tmp_path, 4),
        model="tiny",
        floor=saes.FLOOR_ZERO,
        max_sequences=0,
    )
    assert code == 2
    said = capsys.readouterr().err
    assert "--max-sequences" in said
    assert "at least one sequence" in said


def test_a_model_with_no_registered_sae_is_told_how_to_name_one():
    """A refusal with no next step is a wall.

    The server's version of this points at the logit lens, because the panel is
    right there; this one points at `--sae`, because somebody typing a command
    can name a repo directly.
    """
    with pytest.raises(Refusal) as caught:
        cli._sae_for("nobody/has-an-sae-for-this", "", "")
    assert "--sae" in caught.value.sentence


def test_the_parser_wires_sae_fidelity_and_requires_a_floor(monkeypatch, capsys):
    """`--floor` is required on this surface too, and has no default.

    A default in argparse would put the choice back exactly as silently as a
    default in the request model would.
    """
    seen: dict = {}
    monkeypatch.setattr(
        cli, "sae_fidelity", lambda corpus, **kw: (seen.update(kw), 0)[1]
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "modelmri",
            "sae",
            "fidelity",
            "--model",
            "m",
            "--corpus",
            "c.txt",
            "--floor",
            saes.FLOOR_MEAN,
        ],
    )
    with pytest.raises(SystemExit) as exited:
        cli.main()
    assert exited.value.code == 0
    assert seen["floor"] == saes.FLOOR_MEAN
    assert seen["model"] == "m"

    monkeypatch.setattr(
        sys,
        "argv",
        ["modelmri", "sae", "fidelity", "--model", "m", "--corpus", "c.txt"],
    )
    with pytest.raises(SystemExit) as exited:
        cli.main()
    assert exited.value.code == 2
    assert "--floor" in capsys.readouterr().err


# --------------------------------------------------------------- the panel

#: The card itself, read as text: there is no TypeScript runner here, which is
#: the same reason `test_the_frontend_never_asks_for_fidelity_under_api_
#: features` above reads source rather than importing anything.
CARD = Path(__file__).resolve().parents[1] / "frontend" / "src" / "SaeFidelity.tsx"


def test_the_panels_floor_select_opens_on_nothing():
    """The fourth surface the default can come back on.

    A pydantic default, a query-string default and an argparse `default=` are
    each refused by a test above. A PRE-SELECTED `<option>` is the same defect
    in the place it is least visible: the reader never sees a question, and the
    percentage is labelled with a floor they did not choose.
    """
    card = CARD.read_text("utf-8")
    assert 'const [floor, setFloor] = useState("");' in card, (
        "the floor state does not open empty — a pre-selected floor answers "
        "the question the reader has to be told the answer to"
    )
    at = card.find('id="ce-floor"')
    assert at > 0, "no ce-floor select in the card; this check has gone blind"
    offered = re.findall(r'<option value="([^"]*)"', card[at : at + 1500])
    assert offered[:1] == [""], f"the select opens on {offered[:1]}, not on nothing"
    assert saes.FLOOR_MEAN in offered and saes.FLOOR_ZERO in offered
    # And the control is not clickable until one has been chosen, because a
    # run with no floor is a percentage against nothing.
    assert "|| !floor}" in card


def test_the_panel_never_prices_a_corpus_it_will_not_run():
    """The button's number IS the confirmation, so it has to be this corpus's.

    `/api/sae/fidelity` reads `file` and discards `texts`, so a price taken
    from the pasted box while a file is chosen quotes a count the run will not
    spend — and, because the card sends `confirm` whenever it has a count to
    show, it would confirm past `CE_CONFIRM_ABOVE_PASSES` on a corpus nobody
    counted.
    """
    card = CARD.read_text("utf-8")
    bodies = {}
    for name in ("quote", "timeIt"):
        found = re.search(r"async function " + name + r"\(\) \{(.*?)\n  \}", card, re.S)
        assert found is not None, f"{name}() was renamed; this check is blind"
        bodies[name] = found.group(1)

    before_the_fetch = bodies["quote"].split("saeFidelityPrice")[0]
    assert 'file !== ""' in before_the_fetch, (
        "quote() prices the pasted box without checking whether a file is "
        "chosen, so the button can quote a count the run will not spend"
    )
    # And a price that lands after the corpus moved under it is dropped rather
    # than rendered. Both are round trips and neither the box nor the picker is
    # locked while one is in flight, so clearing the state is not enough on its
    # own — the reply would repopulate it from the corpus it replaced.
    for name, body in bodies.items():
        # The COMPARISON, not the read: capturing the generation and then not
        # checking it is the mutation this line exists to catch, and it leaves
        # `generation.current` in the body either way.
        assert "generation.current !== mine" in body, (
            f"{name}() renders a price that landed after the corpus changed"
        )


def test_sae_with_no_verb_says_what_the_verbs_are(capsys):
    import argparse

    assert cli.sae_command(argparse.Namespace(sae_command=None)) == 2
    assert "fidelity" in capsys.readouterr().err
