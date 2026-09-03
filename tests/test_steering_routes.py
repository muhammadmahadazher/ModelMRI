# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""The five routes that make the vector store reachable.

`tests/test_steer_vectors.py` proves the estimator and the store;
`tests/test_steering_directions.py` proves the runtime arm. This file proves
the things a route around them can get wrong on its own:

  * **an empty store answering like a failure.** Nothing has been fitted on a
    fresh machine, and that is the ordinary first state of the panel. A 409
    there teaches a reader the measurement is broken before they have done
    anything at all, so the catalogue answers `[]` with a model name beside it
    and no error;
  * **the route-ordering trap.** `/api/steer/directions/{name}` takes any
    single segment, so declared before its literal siblings it swallows both
    of them and answers a nonsense 422 for a request that was perfectly well
    formed. The 404 below is what a shadowed route cannot produce;
  * **losing the honest numbers between the module and the wire.** The
    per-layer null, the p-value and the estimate ARE the product of a fit,
    and a route that dropped them would still answer 200;
  * **breaking the arm that already existed.** `POST /api/steer` and
    `GET /api/steer` are what `FeaturesPanel` drives three times per A/B, and
    `demo.ts` mirrors their payload offline with nothing checking one against
    the other.

The app builds its own `ModelRuntime` and every route closes over it, so a
test that needs a model mutates `app.state.runtime`'s fields — the same
object those closures hold. Replacing the attribute would leave the closures
pointing at the original and the test would pass while testing nothing.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
torch = pytest.importorskip("torch")

from fastapi.testclient import TestClient  # noqa: E402

from modelmri import steer_vectors as sv  # noqa: E402
from modelmri.server import create_app  # noqa: E402

D_MODEL = 32
VOCAB = 64


@pytest.fixture
def client(tmp_path, monkeypatch) -> TestClient:
    """A fresh app whose vector store is this test's own directory."""
    monkeypatch.setenv("MODELMRI_HOME", str(tmp_path))
    return TestClient(create_app(), raise_server_exceptions=False)


def _vector(seed: int = 0, dims: int = D_MODEL):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(dims, generator=g)
    return v / v.norm()


def _save(
    name: str,
    *,
    dims: int = D_MODEL,
    layer: int = 1,
    model: str = "some/model",
    beats_null: bool = True,
):
    sv.save(
        name,
        _vector(dims=dims),
        {
            "model": model,
            "layer": layer,
            "hidden_size": dims,
            "method": "caa",
            "dtype": "float32",
            "beats_null": beats_null,
            "p_value": 0.111,
        },
    )


class Tok:
    def __call__(self, text: str, return_tensors: str = "pt"):
        ids = [(ord(c) % (VOCAB - 1)) + 1 for c in text[:24]] or [1]
        return {"input_ids": torch.tensor([ids])}

    def decode(self, ids):
        return "".join(f"<{i}>" for i in ids)


@pytest.fixture
def loaded(client):
    """Put a three-layer GPT-2 into the runtime the routes already hold."""
    import transformers

    cfg = transformers.GPT2Config(
        n_layer=3, n_head=2, n_embd=D_MODEL, vocab_size=VOCAB, n_positions=32
    )
    torch.manual_seed(0)
    runtime = client.app.state.runtime
    runtime.model = transformers.AutoModelForCausalLM.from_config(cfg).eval()
    runtime.tokenizer = Tok()
    runtime.hf_id = "tiny/gpt2-under-test"
    runtime.backend = "hf"
    runtime.device = "cpu"
    runtime.last_ids = torch.tensor([1, 5, 9, 13])
    runtime.last_ids_epoch = runtime.epoch
    return client


# ------------------------------------------------------------- the catalogue


def test_an_empty_store_is_an_empty_list_not_an_error(client):
    r = client.get("/api/steer/directions")
    assert r.status_code == 200
    body = r.json()
    assert body["directions"] == []
    assert body["model"] is None and body["hidden_size"] is None
    assert body["means"]


def test_with_nothing_loaded_compatibility_is_unknown_rather_than_false(client):
    """`False` is the positive claim "this cannot be applied here". With no
    model there is nothing to be compatible WITH, which is a different
    statement and must not render as a red cross on every card."""
    _save("politeness")
    row = client.get("/api/steer/directions").json()["directions"][0]
    assert row["compatible"] is None
    assert row["mismatch"] == ""


def test_a_row_that_fits_the_loaded_model_says_so(loaded):
    _save("politeness", model="tiny/gpt2-under-test")
    body = loaded.get("/api/steer/directions").json()
    assert body["model"] == "tiny/gpt2-under-test"
    assert body["hidden_size"] == D_MODEL
    row = body["directions"][0]
    assert row["compatible"] is True
    assert row["warnings"] == []
    assert "values" not in row


def test_a_row_of_the_wrong_width_carries_the_refusal_sentence(loaded):
    """Visible on the card, never hidden and never silently rescaled."""
    _save("from-a-wider-model", dims=D_MODEL * 2, model="Qwen/Qwen3-1.7B")
    row = loaded.get("/api/steer/directions").json()["directions"][0]
    assert row["compatible"] is False
    assert "Qwen/Qwen3-1.7B" in row["mismatch"]
    assert "tiny/gpt2-under-test" in row["mismatch"]


def test_a_row_from_another_checkpoint_warns_without_blocking(loaded):
    _save("borrowed", model="Qwen/Qwen3-1.7B")
    row = loaded.get("/api/steer/directions").json()["directions"][0]
    assert row["compatible"] is True
    assert any("equal size is not equal basis" in w for w in row["warnings"])


def test_a_direction_that_failed_its_null_says_so_in_the_catalogue(loaded):
    _save("nothing", model="tiny/gpt2-under-test", beats_null=False)
    row = loaded.get("/api/steer/directions").json()["directions"][0]
    assert row["compatible"] is True
    assert any("never evidence of anything" in w for w in row["warnings"])


def test_a_damaged_file_is_listed_as_damaged_not_dropped(client):
    from modelmri import paths

    store = paths.ensure(sv.store_dir())
    (store / "broken.json").write_text("{not json", encoding="utf-8")
    row = client.get("/api/steer/directions").json()["directions"][0]
    assert row["unreadable"] is True
    assert row["compatible"] is False
    assert row["mismatch"]


def test_asking_for_the_catalogue_does_not_create_the_store(client):
    """A read-only question that writes a directory into somebody's data
    folder is the rule `paths.py` states in its own docstring."""
    assert client.get("/api/steer/directions").status_code == 200
    assert not sv.store_dir().exists()


# ------------------------------------------------------------ apply and clear


def test_applying_with_no_model_refuses_in_words(client):
    _save("politeness")
    r = client.post(
        "/api/steer/direction", json={"name": "politeness", "strength": 2.0}
    )
    assert r.status_code == 409
    assert "No model loaded" in r.json()["error"]


def test_applying_a_direction_that_is_not_there_refuses_by_name(loaded):
    """BY NAME, so the name is asserted — the sibling delete test at the
    bottom of this file shows the form. A prefix match would go on passing
    against a sentence that had stopped saying which direction was missing,
    which is the whole content of this route's answer."""
    r = loaded.post("/api/steer/direction", json={"name": "never-fitted"})
    assert r.status_code == 409
    assert "no saved direction called 'never-fitted'" in r.json()["error"]


def test_apply_then_status_then_clear_round_trips(loaded):
    _save("politeness", model="tiny/gpt2-under-test")
    applied = loaded.post(
        "/api/steer/direction", json={"name": "politeness", "strength": 3.0}
    )
    assert applied.status_code == 200
    assert applied.json()["kind"] == "direction"

    status = loaded.get("/api/steer").json()
    assert status["active"] is True
    assert status["name"] == "politeness" and status["layer"] == 1
    assert status["strength"]["alpha"] == 3.0
    assert status["strength"]["relative"] is not None
    assert status["strength"]["measured"]

    cleared = loaded.delete("/api/steer")
    assert cleared.status_code == 200
    assert cleared.json() == {"active": False}
    assert loaded.get("/api/steer").json() == {"active": False}


def test_the_old_post_still_clears_a_direction(loaded):
    """`FeaturesPanel` promises it "always leaves the model clean"."""
    _save("politeness", model="tiny/gpt2-under-test")
    loaded.post("/api/steer/direction", json={"name": "politeness", "strength": 3.0})
    assert loaded.post("/api/steer", json={"feature_id": None}).json() == {
        "active": False
    }


def test_a_mismatched_direction_is_refused_at_the_apply_route(loaded):
    _save("from-a-wider-model", dims=D_MODEL * 2, model="Qwen/Qwen3-1.7B")
    r = loaded.post("/api/steer/direction", json={"name": "from-a-wider-model"})
    assert r.status_code == 409
    said = r.json()["error"]
    assert "Qwen/Qwen3-1.7B" in said and "tiny/gpt2-under-test" in said


def test_an_empty_name_is_rejected_by_the_request_model(loaded):
    assert loaded.post("/api/steer/direction", json={"name": ""}).status_code == 422


# ------------------------------------------------------------------ deleting


def test_deleting_a_direction_that_is_not_there_is_a_404_with_a_sentence(client):
    r = client.delete("/api/steer/directions/never-fitted")
    assert r.status_code == 404, "a shadowed route cannot answer this"
    assert "no saved direction called 'never-fitted'" in r.json()["error"]


def test_deleting_takes_it_out_of_the_catalogue(client):
    _save("politeness")
    assert len(client.get("/api/steer/directions").json()["directions"]) == 1
    r = client.delete("/api/steer/directions/politeness")
    assert r.status_code == 200 and r.json()["removed"] == "politeness"
    assert client.get("/api/steer/directions").json()["directions"] == []


def test_a_name_in_the_path_cannot_escape_the_store(client):
    """`_slug` strips separators before anything touches the filesystem, so
    the worst a crafted name can do is name a file that is not there.

    The status code alone cannot carry this: 404 and 422 are both "no file of
    that name", and a router that never reached the handler would answer the
    same way as one whose slug held. The planted file is the assertion — it
    sits one level above the store, which is where `..%2F` would land.
    """
    from modelmri import paths

    paths.ensure(sv.store_dir())
    outside = sv.store_dir().parent / "passwd.json"
    outside.write_text("not a direction, and not yours to delete", encoding="utf-8")

    r = client.delete("/api/steer/directions/..%2Fpasswd")
    assert r.status_code in (404, 422)
    assert outside.exists(), "the path segment escaped the store and unlinked a file"

    r = client.delete("/api/steer/directions/..%2F..%2Fetc%2Fpasswd")
    assert r.status_code in (404, 422)


# -------------------------------------------------------------------- fitting


def _pairs(n: int = 10):
    return (
        [f"pair {i} is a yes" for i in range(n)],
        [f"pair {i} is a no" for i in range(n)],
    )


def test_fitting_with_no_model_refuses_in_words(client):
    positive, negative = _pairs()
    r = client.post(
        "/api/steer/fit", json={"positive_texts": positive, "negative_texts": negative}
    )
    assert r.status_code == 409
    assert "No model loaded" in r.json()["error"]


def test_a_body_that_is_not_json_is_422(client):
    """Still 422, but through the shared body model rather than by hand.

    This route read the raw `Request` when it landed, which meant `Body`'s
    unknown-key refusal did not apply to it — `save_as` typed as `saveas`
    would have fitted a direction, published its p-value and saved nothing.
    `test_no_new_route_takes_a_raw_body_without_saying_why` caught that, so
    the hand-rolled sentences these two tests pinned are gone and FastAPI's
    own validation answers instead. The GUARANTEE is unchanged and is what is
    asserted: a body this route cannot read is a 422 that says so, never a
    500 and never a fit against defaults nobody asked for.
    """
    r = client.post(
        "/api/steer/fit",
        content=b"not json",
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 422, r.text
    assert "JSON" in r.text


def test_a_body_that_is_not_an_object_is_422(client):
    r = client.post("/api/steer/fit", json=[1, 2, 3])
    assert r.status_code == 422, r.text
    # Named, rather than a bare status: a caller who sent a list is told a
    # dictionary was expected instead of being sent to read the source.
    assert "dict" in r.text or "object" in r.text


def test_a_misspelled_field_is_refused_by_name_rather_than_fitted_around(client):
    """The reason the raw body was the defect, not a style preference.

    `save_as` is the field whose misspelling is invisible: the fit runs, the
    p-value is published, and nothing is written to the catalogue. Under the
    raw body that was a 200. It is now a 422 naming the key.
    """
    r = client.post(
        "/api/steer/fit",
        json={"positive_texts": ["a"], "negative_texts": ["b"], "saveas": "x"},
    )
    assert r.status_code == 422, r.text
    assert "saveas" in r.text


def test_layers_that_are_not_numbers_are_422(loaded):
    positive, negative = _pairs()
    r = loaded.post(
        "/api/steer/fit",
        json={
            "positive_texts": positive,
            "negative_texts": negative,
            "layers": ["seven"],
        },
    )
    assert r.status_code == 422
    assert "whole numbers" in r.json()["error"]


def test_too_few_pairs_is_refused_before_anything_runs(loaded):
    r = loaded.post(
        "/api/steer/fit",
        json={"positive_texts": ["a", "b"], "negative_texts": ["c", "d"]},
    )
    assert r.status_code == 409
    assert "at least 8" in r.json()["error"]


def test_an_unknown_method_is_refused_by_name(loaded):
    positive, negative = _pairs()
    r = loaded.post(
        "/api/steer/fit",
        json={
            "positive_texts": positive,
            "negative_texts": negative,
            "method": "magic",
        },
    )
    assert r.status_code == 422
    assert "unknown method" in r.json()["error"]


def test_the_estimate_comes_back_before_anything_is_spent(loaded):
    positive, negative = _pairs()
    body = loaded.post(
        "/api/steer/fit",
        json={
            "positive_texts": positive,
            "negative_texts": negative,
            "estimate_only": True,
        },
    ).json()
    assert body["ran"] is False
    assert body["layers"] == []
    assert body["estimate"]["passes"] == 20
    assert body["estimate"]["basis"] == "one probe pass on this machine"


def test_a_fit_publishes_the_whole_null_table(loaded):
    positive, negative = _pairs()
    body = loaded.post(
        "/api/steer/fit", json={"positive_texts": positive, "negative_texts": negative}
    ).json()
    assert body["ran"] is True
    assert len(body["layers"]) == 3
    for row in body["layers"]:
        for key in ("effect", "null_mean", "null_max", "beats_null", "p_value"):
            assert key in row, key
    assert body["means"]
    assert body["receipt"]["op"] == "fit_direction"


# ---------------------------------------------------------------- the A/B


@pytest.fixture
def generating(client):
    """A runtime that can actually decode: real tokenizer, random weights.

    `generate_stream` hands the tokenizer to a `TextIteratorStreamer` and
    reads `eos_token_id` and `chat_template` off it, which is more surface
    than the stub above is worth growing. The WEIGHTS are still built from a
    config and still random — nothing about this test depends on the model
    saying anything sensible, only on the two completions differing when a
    vector is being added to the stream and agreeing when it is not.

    BUILT HERE RATHER THAN FETCHED. This was `AutoTokenizer.from_pretrained
    ("gpt2", local_files_only=True)` behind two `pytest.skip`s, and it is the
    only test anywhere that drives the GENERATION path with a direction
    installed — every other assertion about the hook's arithmetic goes through
    `_logits`, which installs the handle itself and so cannot notice
    `generate_stream` failing to reach for one. A guard that disappears on any
    machine with a cold HuggingFace cache is not a guard, and the whole of
    what this needs from a tokenizer is a vocabulary and a decode.
    """
    import transformers
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {"<unk>": 0}
    for word in "The capital of France is a an and it was to be very".split():
        vocab.setdefault(word, len(vocab))
    while len(vocab) < VOCAB:
        vocab[f"tok{len(vocab)}"] = len(vocab)
    inner = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    inner.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=inner,
        unk_token="<unk>",
        eos_token="<unk>",
        pad_token="<unk>",
    )

    cfg = transformers.GPT2Config(
        n_layer=3,
        n_head=2,
        n_embd=D_MODEL,
        vocab_size=tokenizer.vocab_size,
        n_positions=64,
    )
    torch.manual_seed(0)
    runtime = client.app.state.runtime
    runtime.model = transformers.AutoModelForCausalLM.from_config(cfg).eval()
    runtime.tokenizer = tokenizer
    runtime.hf_id = "tiny/gpt2-under-test"
    runtime.backend = "hf"
    runtime.device = "cpu"
    return client


def _once(app, prompt: str = "The capital of France is") -> str:
    r = app.post(
        "/api/model/prompt",
        json={
            "prompt": prompt,
            "max_new_tokens": 6,
            "temperature": 0,
            # THE FOURTH ARGUMENT IS NOT OPTIONAL. An A/B that committed would
            # rebase every other panel's analysis target onto its own throwaway
            # completion — see `_Recording`'s docstring.
            "commit": False,
        },
    )
    assert r.status_code == 200, r.text
    return r.json()["generation"]


def test_the_ab_generates_under_direction_steering(generating):
    """The steered-vs-unsteered pair `FeaturesPanel` runs is two uncommitted
    completions with the hook installed for the second. It has to reach the
    direction arm too, or the panel's compare renders two identical columns
    and calls one of them steered."""
    _save("politeness", model="tiny/gpt2-under-test")
    base = _once(generating)

    applied = generating.post(
        "/api/steer/direction", json={"name": "politeness", "strength": 60.0}
    )
    assert applied.status_code == 200, applied.text
    assert _once(generating) != base

    generating.delete("/api/steer")
    assert _once(generating) == base, "clearing has to restore the baseline"
