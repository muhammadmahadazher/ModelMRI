"""Opening a `.mri` must drive the real panels, not a parallel read-only mode.

The point of routing replay through `runtime.attention()` is that every panel
already built keeps working with no changes. These tests assert that from the
outside: the same endpoints, the same shapes, on a machine with no model.
"""

from __future__ import annotations

import gzip
import json

from fastapi.testclient import TestClient

from modelmri import session
from modelmri.server import create_app


def client() -> TestClient:
    return TestClient(create_app())


def a_session(*, layers: int = 2, heads: int = 2, n: int = 4) -> bytes:
    matrices = {}
    for layer in range(layers):
        for head in range(heads):
            matrices[(layer, head)] = [
                [round(1.0 / (r + 1), 4) if c <= r else 0.0 for c in range(n)]
                for r in range(n)
            ]
    return session.build(
        model_id="gpt2",
        device="cuda:0",
        dtype="float16",
        n_params=124_000_000,
        tokens=[f"tok{i}" for i in range(n)],
        prompt="tok0tok1",
        generation="tok2tok3",
        attention=matrices,
        n_layers=layers,
        n_heads=heads,
        note="the induction head",
    )


def test_nothing_is_open_on_a_fresh_server():
    assert client().get("/api/session/state").json() == {"open": False}


def test_opening_a_session_makes_attention_work_without_a_model():
    c = client()
    assert c.get("/api/attention/meta").json()["available"] is False

    opened = c.post("/api/session/open", content=a_session()).json()
    assert opened["open"] is True
    assert opened["meta"]["model"] == "gpt2"
    assert opened["meta"]["note"] == "the induction head"
    assert opened["prompt"] == "tok0tok1"

    meta = c.get("/api/attention/meta").json()
    assert meta == {
        "available": True,
        "n_prompt": 0,
        "n_layers": 2,
        "n_heads": 2,
        "n_tokens": 4,
        "replay": True,
    }

    slice_ = c.get("/api/attention?layer=1&head=1").json()
    assert slice_["layer"] == 1 and slice_["head"] == 1
    assert slice_["tokens"] == ["tok0", "tok1", "tok2", "tok3"]
    assert len(slice_["matrix"]) == 4
    assert slice_["replay"] is True


def test_closing_a_session_puts_it_back_the_way_it_was():
    c = client()
    c.post("/api/session/open", content=a_session())
    assert c.post("/api/session/close").json() == {"open": False}
    assert c.get("/api/attention/meta").json()["available"] is False
    assert c.get("/api/attention?layer=0&head=0").status_code == 409


def test_a_slice_that_was_not_captured_is_a_422_with_a_reason():
    c = client()
    c.post("/api/session/open", content=a_session(layers=2, heads=2))
    r = c.get("/api/attention?layer=9&head=9")
    assert r.status_code == 422
    assert "does not contain layer 9 head 9" in r.json()["error"]


def test_garbage_is_refused_with_a_sentence_not_a_stack_trace():
    r = client().post("/api/session/open", content=b"this is not a session")
    assert r.status_code == 422
    message = r.json()["error"]
    assert "not a ModelMRI session" in message
    # No exception class names. They read as an internal fault when the real
    # situation is simply "wrong file".
    assert "Error" not in message
    assert "Share this view" in message


def test_an_empty_upload_is_refused():
    r = client().post("/api/session/open", content=b"")
    assert r.status_code == 422
    assert "empty" in r.json()["error"]


def test_a_future_version_asks_you_to_upgrade():
    doc = json.loads(gzip.decompress(a_session()))
    doc["format_version"] = 99
    r = client().post(
        "/api/session/open", content=gzip.compress(json.dumps(doc).encode())
    )
    assert r.status_code == 422
    assert "pip install -U modelmri" in r.json()["error"]


def test_exporting_with_nothing_loaded_explains_itself():
    r = client().get("/api/session/export")
    assert r.status_code == 409
    assert "error" in r.json()


def test_you_cannot_export_a_session_you_are_only_viewing():
    """Re-exporting a recording as if it were your own run is a lie."""
    c = client()
    c.post("/api/session/open", content=a_session())
    r = c.get("/api/session/export")
    assert r.status_code == 409
    assert "viewing a shared session" in r.json()["error"]


def test_loading_a_model_closes_the_recording(monkeypatch):
    """Asking for live weights must not leave you reading a recording.

    Driven through the real `load()`, via the Ollama branch — it is the one
    that reaches the same reset without downloading gigabytes. Setting
    `rt.replay = None` in the test instead would pass with the fix reverted.
    """
    from modelmri import ollama
    from modelmri import runtime as runtime_mod

    monkeypatch.setattr(
        ollama, "status", lambda *a, **k: {"up": True, "models": ["qwen3:0.6b"]}
    )
    rt = runtime_mod.ModelRuntime()
    rt.open_session(a_session())
    assert rt.attention_meta()["available"] is True

    rt.load("qwen3:0.6b", source="ollama")

    assert rt.session_info() == {"open": False}
    assert rt.attention_meta()["available"] is False


def test_a_committed_generation_closes_the_recording(monkeypatch):
    """Your own output above somebody else's heat map explains nothing."""
    import torch

    from modelmri import runtime as runtime_mod

    rt = runtime_mod.ModelRuntime()
    rt.open_session(a_session())

    # The narrowest possible stand-in for a loaded model: enough for
    # generate_stream to reach its commit block, nothing more.
    class FakeTokenizer:
        chat_template = None
        eos_token_id = 0

        def __call__(self, texts, return_tensors=None):
            class Batch(dict):
                def to(self, _device):
                    return self

            return Batch(input_ids=torch.tensor([[1, 2, 3]]))

        def decode(self, ids):
            return "x"

    class FakeModel:
        def generate(self, **kw):
            return torch.tensor([[1, 2, 3, 4]])

    monkeypatch.setattr(
        runtime_mod, "TextIteratorStreamer", lambda *a, **k: iter(["hi"])
    )
    rt.tokenizer, rt.model, rt.hf_id, rt.backend = (
        FakeTokenizer(),
        FakeModel(),
        "fake",
        "hf",
    )
    list(rt.generate_stream("hello", max_new_tokens=1))

    assert rt.session_info() == {"open": False}
    assert rt.last_prompt == "hello"


def test_an_open_session_survives_a_page_reload():
    """State lives on the server, so a refresh must not silently drop it."""
    c = client()
    c.post("/api/session/open", content=a_session())
    assert c.get("/api/session/state").json()["open"] is True
    assert c.get("/api/session").json()["model"]["loaded"] is False


# ------------------------------------------- the block count is a model fact


def test_the_layer_count_rides_with_the_loaded_model_not_with_a_run():
    """`/api/attention/meta` carries it too, but only AFTER a generation --
    attention is read off a real run and there is nothing to read before one.

    The probe and patchscope panels both need to offer a layer before there is
    anything to generate from, and reading the count from the attention route
    left their layer pickers EMPTY on a freshly loaded model: the one control
    neither panel works without was dead until the user generated something
    they did not need.
    """
    from modelmri.runtime import ModelStatus

    assert "n_layers" in ModelStatus(loaded=False).to_dict()


def test_an_unloaded_model_reports_no_layer_count_rather_than_zero():
    """Zero blocks is a claim about a model. No model is the absence of one,
    and a UI that renders 0 offers a dropdown with nothing in it."""
    from modelmri.runtime import ModelStatus

    assert ModelStatus(loaded=False).to_dict()["n_layers"] is None


def test_the_count_survives_a_model_with_no_config_at_all():
    """`self.model.config` is not a given.

    A TorchScript module, an adapter-loaded network and the stub the load
    tests use all have parameters and NO config, and reaching through it
    raised AttributeError from inside `status()` -- the one method every route
    calls, so a missing block count took the whole session endpoint down
    rather than reporting an unknown. Behaviour, not source text: the first
    version of this test asserted the exact getattr expression and went stale
    the moment the expression was fixed.
    """
    import torch

    from modelmri.runtime import ModelRuntime

    class _Param:
        dtype = torch.float32

        def numel(self):
            return 1

    class _NoConfig:
        """Parameters, no `.config`. Exactly what a TorchScript module is."""

        def parameters(self):
            return iter([_Param()])

    rt = ModelRuntime()
    rt.model = _NoConfig()
    rt.tokenizer = object()
    status = rt.status().to_dict()
    assert status["loaded"] is True, "the status must still be readable"
    assert status["n_layers"] is None, "unknown, not zero"


def test_a_config_without_a_block_count_reports_unknown_rather_than_zero():
    """0 is the positive claim "this model has no blocks", which the UI
    renders as an empty dropdown. None is the absence of a number, which it
    renders as a plain field the server will validate."""
    import torch

    from modelmri.runtime import ModelRuntime

    class _Param:
        dtype = torch.float32

        def numel(self):
            return 1

    class _Odd:
        config = type("C", (), {})()

        def parameters(self):
            return iter([_Param()])

    rt = ModelRuntime()
    rt.model = _Odd()
    rt.tokenizer = object()
    assert rt.status().to_dict()["n_layers"] is None
