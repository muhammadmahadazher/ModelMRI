# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""Smoke tests — no model download, just the app surface."""

import json
import time

import pytest
from fastapi.testclient import TestClient

from modelmri import __version__
from modelmri.server import create_app


def client() -> TestClient:
    return TestClient(create_app())


def test_version_present():
    assert __version__


def test_session_endpoint():
    r = client().get("/api/session")
    assert r.status_code == 200
    body = r.json()
    assert body["app"] == "modelmri"
    assert body["version"] == __version__
    assert body["model"]["loaded"] is False


def test_index_serves_playground():
    r = client().get("/")
    assert r.status_code == 200
    assert "ModelMRI" in r.text


def test_prompt_without_model_is_409():
    r = client().post("/api/model/prompt", json={"prompt": "hi"})
    assert r.status_code == 409


def test_attention_meta_unavailable_without_model():
    r = client().get("/api/attention/meta")
    assert r.status_code == 200
    assert r.json()["available"] is False


def test_attention_without_model_is_409():
    r = client().get("/api/attention?layer=0&head=0")
    assert r.status_code == 409


def test_attribute_without_a_generation_is_409():
    """A model IS loaded here, and nothing has been generated yet.

    This used a bare `client()` with nothing loaded, so it asserted "Generate
    something first" green for the state that wanted the opposite sentence —
    baking in the collapsed `not self.loaded or self.last_ids is None` guard
    that `_require_live_generation` used to carry. Following that instruction
    with nothing loaded gave a second refusal: POST /api/model/prompt -> 409
    "no model loaded". A next step the reader could not take.

    Both arms, so neither can drift back into the other.
    """
    app = create_app()
    # `loaded` is derived from `model`, not settable — see
    # test_attention_meta_says_which_kind_of_nothing.
    app.state.runtime.model = object()
    ungenerated = TestClient(app).get("/api/attention/attribute")
    assert ungenerated.status_code == 409
    assert "Generate something first" in ungenerated.json()["error"]

    empty = client().get("/api/attention/attribute")
    assert empty.status_code == 409
    assert "No model loaded" in empty.json()["error"]
    assert "Generate something first" not in empty.json()["error"]


def test_attribute_on_a_recording_is_409():
    """A `.mri` carries attention, not weights. Masking a token means running
    the model again, so there is nothing here to measure — and an empty
    ranking would read as "none of your words mattered"."""
    app = create_app()
    app.state.runtime.replay = object()

    r = TestClient(app).get("/api/attention/attribute")
    assert r.status_code == 409
    assert "recording" in r.json()["error"]


def test_the_logit_lens_refuses_a_recording():
    """The one replay-sensitive route whose guard is not in runtime.py.

    Every other one — attention_meta, attention_slice, compare, rank_heads,
    attribute_tokens, export_session — is a ModelRuntime method and opens with
    `if self.replay is not None`. The lens is computed in server.py from
    modelmri.lens, so it never passed a runtime guard, and `model is None`
    covered for it whenever a `.mri` was opened with nothing loaded.

    It stops covering the moment a recording is opened while the reader's own
    model is still resident: `model` and `last_ids` are both set, and the lens
    reports the LIVE model's layers under a replay pill that says "recorded,
    not live". A wrong answer wearing the right label is the failure this
    project exists not to ship.
    """
    app = create_app()
    runtime = app.state.runtime
    runtime.replay = object()
    # The state that used to hide it: a model IS loaded, so the `model is None`
    # branch below would not have fired.
    runtime.model = object()
    runtime.last_ids = object()

    r = TestClient(app).get("/api/lens")
    assert r.status_code == 409
    assert "recording" in r.json()["error"]


def test_attention_meta_says_which_kind_of_nothing():
    """ "No model loaded" and "no generation yet" are different instructions.

    Both used to arrive as a bare {"available": False}, so the panel could not
    tell the reader whether to pick a model or press the button in front of
    them — there was nothing in the payload to tell them apart with.
    """
    app = create_app()
    runtime = app.state.runtime

    # `loaded` is derived from `model`, not settable — which is the point: the
    # two states below differ only in whether a model is resident.
    assert runtime.model is None
    assert runtime.attention_meta() == {
        "available": False,
        "reason": "no model loaded",
    }

    runtime.model = object()  # loaded, but nothing generated yet
    runtime.last_ids = None
    meta = runtime.attention_meta()
    assert meta["available"] is False
    assert "generate" in meta["reason"].lower()
    assert meta["reason"] != "no model loaded"


class _OffsetTok:
    """Just enough tokenizer to exercise the offset mapping, no download."""

    is_fast = True

    def __init__(self, offsets, fast=True):
        self._offsets = offsets
        self.is_fast = fast

    def __call__(self, texts, **kw):
        return {"offset_mapping": [self._offsets]}


def test_user_span_leaves_out_an_added_token_at_index_zero():
    """The user's span starts at character 0, where the (0, 0) offset a fast
    tokenizer gives its own added tokens also starts. The overlap test is
    half-open, so index 0 stays out and the span is the five words."""
    from modelmri.runtime import _user_span

    prompt = "The capital of France is"
    offsets = [(0, 0), (0, 3), (3, 11), (11, 14), (14, 21), (21, 24)]
    assert _user_span(_OffsetTok(offsets), prompt, prompt) == (1, 6)


def test_user_span_refuses_when_the_prompt_is_ambiguous():
    """A chat template contains the words 'user', 'assistant' and 'model', so a
    prompt of exactly one of those matches the scaffolding too. Unknown is a
    state this field carries; a confident wrong span is not.

    The offsets here are the real ones for this text, so without the ambiguity
    guard `find` lands on the template's 'user' at index 1 and this returns
    (1, 2) — a span pointing at the scaffolding, labelled as the user's."""
    from modelmri.runtime import _user_span

    text = "<|im_start|>user\nuser<|im_end|>\n"
    offsets = [(0, 12), (12, 16), (16, 17), (17, 21), (21, 31), (31, 32)]
    assert text[12:16] == "user" and text[17:21] == "user"
    assert _user_span(_OffsetTok(offsets), "user", text) is None


def test_user_span_refuses_a_slow_tokenizer():
    """No offset mapping, nothing to map through."""
    from modelmri.runtime import _user_span

    tok = _OffsetTok([(0, 3)], fast=False)
    assert _user_span(tok, "The", "The") is None


def test_an_ollama_load_clears_the_user_span(monkeypatch):
    """It is one model's character offsets through one model's tokenizer.
    Carried across a load it would label the new model's tokens with the old
    one's arithmetic — and unlike a wrong number, a wrong group name looks
    like a fact about your prompt."""
    from modelmri import ollama
    from modelmri.runtime import ModelRuntime

    monkeypatch.setattr(
        ollama, "status", lambda host=None, timeout=None: {"up": True, "models": ["m"]}
    )
    monkeypatch.setattr(ollama, "is_instruct", lambda name, host=None: True)

    rt = ModelRuntime()
    assert rt.last_user_span is None, "the field has to exist before a generation"
    rt.last_user_span = (3, 8)
    rt.load("m", source="ollama")
    assert rt.last_user_span is None


def test_an_hf_load_clears_the_user_span(monkeypatch):
    """The same rule on the path that matters more — swapping one HuggingFace
    model for another, where the old span's indices are all still in range for
    the new tokenizer and so would be believed."""
    import torch

    from modelmri import runtime as runtime_mod
    from modelmri.runtime import ModelRuntime

    class _Param:
        dtype = torch.float32

        def numel(self):
            return 1

    class _Model:
        def to(self, *a, **k):
            return self

        def eval(self):
            return self

        def parameters(self):
            return iter([_Param()])

    monkeypatch.setattr(runtime_mod, "_require_causal_lm", lambda *_a: None)
    monkeypatch.setattr(runtime_mod, "_preflight", lambda *_a, **_k: None)
    monkeypatch.setattr(
        ModelRuntime, "_prefetch_weights", lambda self, hf_id: None, raising=True
    )
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", lambda *a, **k: object()
    )
    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained", lambda *a, **k: _Model()
    )

    rt = ModelRuntime()
    rt.last_user_span = (3, 8)
    assert rt.load("acme/other").loaded
    assert rt.last_user_span is None


def test_attribute_via_ollama_is_409():
    """Ollama hands back text. There is no forward pass to mask a token out
    of, and the refusal says which backend to load instead."""
    app = create_app()
    rt = app.state.runtime
    rt.backend, rt.hf_id = "ollama", "qwen3:0.6b"
    assert rt.loaded, "the ollama refusal has to be what rejects this, not 'no model'"

    r = TestClient(app).get("/api/attention/attribute")
    assert r.status_code == 409
    assert "Ollama" in r.json()["error"]


def test_feature_ablation_on_a_recording_is_409():
    """A `.mri` carries attention, not weights. Ranking features means
    subtracting one from the residual stream and running the model again, and
    there is no model here to run.

    The status code is also what pins the route's position in server.py.
    `/api/features/{feature_id}` is declared with no path converter, so its
    regex swallows any single segment; registered after it, `/api/features/
    ablate` would be parsed as feature_id="ablate" and answer 422 for a
    perfectly well-formed request. 409 can only come from the real handler.
    """
    app = create_app()
    app.state.runtime.replay = object()

    r = TestClient(app).get("/api/features/ablate")
    assert r.status_code == 409, r.text
    assert "recording" in r.json()["error"]


def test_feature_ablation_via_ollama_is_409():
    """Ollama hands back text. There is no residual stream to subtract a
    feature's decoder direction from, and the refusal says which backend to
    load instead."""
    app = create_app()
    rt = app.state.runtime
    rt.backend, rt.hf_id = "ollama", "qwen3:0.6b"
    assert rt.loaded, "the ollama refusal has to be what rejects this, not 'no model'"

    r = TestClient(app).get("/api/features/ablate")
    assert r.status_code == 409, r.text
    assert "Ollama" in r.json()["error"]


def test_feature_ablation_refuses_a_model_that_is_not_float32():
    """The refusal nothing inside the measurement can raise for itself.

    `feature_ablate` proves its floor by writing the captured stream back
    unchanged — bit-exact in every dtype, 0.0 in every dtype. So it cannot see
    that in bfloat16 a 1-ulp change to the stream is worth a real fraction of a
    nat by itself. On a real run a feature whose float32 effect is
    indistinguishable from zero moved the answer measurably in bfloat16 and
    outranked features with far more activation, while noise_floor_kl still
    read 0.0 beside it.

    Half-precision is the DEFAULT on a GPU — `devices.detect` picks bfloat16
    for any Ampere-or-newer NVIDIA card — so without this the ordinary path on
    an ordinary machine publishes a ranking of rounding error.
    """
    import torch

    class Half:
        """Just enough model to be asked its dtype."""

        def parameters(self):
            yield torch.zeros(1, dtype=torch.bfloat16)

    app = create_app()
    rt = app.state.runtime
    rt.backend = "hf"
    rt.model = Half()
    rt.sae = object()  # loaded, so the SAE refusal is not what answers
    rt.last_ids = torch.zeros(5, dtype=torch.long)
    rt.last_n_prompt_tokens = 5
    rt.epoch = rt.last_ids_epoch = 1

    with pytest.raises(RuntimeError, match="bfloat16"):
        rt.rank_features()

    r = TestClient(app).get("/api/features/ablate")
    assert r.status_code == 409, r.text
    body = r.json()["error"]
    assert "float32" in body, body
    # It has to say what to DO. A refusal naming only the problem leaves the
    # reader with a GPU they cannot turn off — and the remedy has to work on
    # the reader's machine, not just on NVIDIA. This used to name
    # CUDA_VISIBLE_DEVICES, which does nothing on Apple Silicon, an Intel GPU
    # or ROCm: the message told those readers to run a command that could not
    # help them.
    assert "MODELMRI_DEVICE" in body, body
    assert "CUDA_VISIBLE_DEVICES" not in body, (
        "a remedy that only works on one vendor's hardware is not a remedy"
    )

    # float32 gets past the gate: the next refusal is the fake SAE failing,
    # which reaches the 500 arm rather than the dtype one. The point is only
    # that the dtype check is not rejecting everything.
    class Full(Half):
        def parameters(self):
            yield torch.zeros(1, dtype=torch.float32)

    rt.model = Full()
    with pytest.raises(Exception) as caught:
        rt.rank_features()
    assert "float32" not in str(caught.value)


def test_sae_status_unloaded():
    r = client().get("/api/sae")
    assert r.status_code == 200
    assert r.json()["loaded"] is False


def test_sae_load_without_model_is_409():
    r = client().post("/api/sae/load", json={})
    assert r.status_code == 409


def test_features_without_sae_is_409():
    r = client().get("/api/features/summary")
    assert r.status_code == 409


def test_steer_without_sae_is_409():
    r = client().post("/api/steer", json={"feature_id": 7, "scale": 4.0})
    assert r.status_code == 409


def test_steer_clear_is_ok_without_sae():
    c = client()
    r = c.post("/api/steer", json={"feature_id": None})
    assert r.status_code == 200
    assert r.json()["active"] is False
    assert c.get("/api/steer").json()["active"] is False


def test_hub_auth_shape():
    r = client().get("/api/hub/auth")
    assert r.status_code == 200
    assert isinstance(r.json()["signed_in"], bool)


def test_hub_signin_rejects_a_bad_token(monkeypatch):
    from modelmri import hub

    monkeypatch.setattr(hub, "whoami", lambda tok=None: hub.HubAuth(signed_in=False))
    r = client().post("/api/hub/signin", json={"token": "hf_not_a_real_token"})
    assert r.status_code == 422
    assert "rejected" in r.json()["error"]


def test_a_token_is_not_access_to_a_gated_repo(monkeypatch):
    """Signing in does not grant a licence. We shipped `(not gated) or token`
    and it labelled every Gemma build usable for an account that had never
    accepted Google's terms — the picker promised what the loader refused."""
    from modelmri import hub

    monkeypatch.setattr(hub, "token", lambda: "hf_valid")
    monkeypatch.setattr(
        hub,
        "_api",
        lambda path, tok=None, timeout=10: [
            {"id": "google/gemma-3-270m-it", "gated": True},
            {"id": "meta-llama/Llama-3.2-1B", "gated": True},
            {"id": "Qwen/Qwen3-0.6B", "gated": False},
        ],
    )
    # access granted for Llama only
    monkeypatch.setattr(
        hub, "_has_access", lambda repo, tok: repo.startswith("meta-llama/")
    )
    by_id = {m["id"]: m for m in hub.search("x")}
    assert by_id["google/gemma-3-270m-it"]["usable"] is False
    assert by_id["meta-llama/Llama-3.2-1B"]["usable"] is True
    assert by_id["Qwen/Qwen3-0.6B"]["usable"] is True


def test_gated_access_check_is_not_fooled_by_a_missing_token():
    from modelmri import hub

    assert hub._has_access("google/gemma-3-270m-it", None) is False


def test_access_check_reads_the_status_not_the_body(monkeypatch):
    """auth-check answers 200 with an EMPTY body. Routing it through the JSON
    helper made json.load raise, so every repo — including ones the account
    HAD accepted — reported no access. It looked right only because the repos
    on hand were inaccessible anyway."""
    import urllib.error
    import urllib.request
    from contextlib import contextmanager

    from modelmri import hub

    @contextmanager
    def empty_200(_req, timeout=None):
        class R:
            status = 200

            def read(self):
                return b""  # exactly what the Hub sends

        yield R()

    monkeypatch.setattr(urllib.request, "urlopen", empty_200)
    assert hub._has_access("meta-llama/Llama-3.2-1B", "hf_tok") is True

    def forbidden(_req, timeout=None):
        raise urllib.error.HTTPError("u", 403, "Forbidden", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    assert hub._has_access("google/gemma-3-270m-it", "hf_tok") is False


def test_hub_signin_requires_a_token():
    assert client().post("/api/hub/signin", json={"token": ""}).status_code == 422


def test_hub_signin_never_writes_the_token_into_the_repo(tmp_path, monkeypatch):
    """The credential must live in the user's home dir, never in the project."""
    from modelmri import hub

    target = tmp_path / "hub.json"
    # The token location is resolved per-platform now, so patch the resolver
    # rather than a module constant that no longer exists.
    monkeypatch.setattr(hub, "_config_path", lambda: target)
    monkeypatch.setattr(
        hub, "whoami", lambda tok=None: hub.HubAuth(signed_in=True, user="tester")
    )
    auth = hub.sign_in("hf_fake")
    assert auth.user == "tester"
    assert json.loads(target.read_text())["token"] == "hf_fake"


def test_ollama_pull_when_daemon_is_down(monkeypatch):
    """A daemon that is not there is a 409 in ollama.py's own words.

    No stub for `pull`. It used to be monkeypatched with a plain
    `RuntimeError("ollama unreachable...")`, which was the shape of the
    transitional arm in server.py rather than the shape of anything ollama.py
    raises — the real module raises `Refusal` from `_unreachable`. When that
    arm came out the test failed, and it was the stub that was wrong. This
    points the client at a port nothing is listening on and lets the real code
    path produce the real exception.
    """
    import socket

    from modelmri import ollama

    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    dead = probe.getsockname()[1]
    probe.close()  # nothing is listening there now

    monkeypatch.setenv("OLLAMA_HOST", f"http://127.0.0.1:{dead}")
    monkeypatch.setattr(ollama, "manifest_size", lambda *_a, **_k: 0)
    r = client().post("/api/ollama/pull", json={"name": "qwen3:0.6b"})
    assert r.status_code == 409
    assert "unreachable" in r.json()["error"]


def test_load_progress_idle():
    r = client().get("/api/model/progress")
    assert r.status_code == 200
    body = r.json()
    assert body["active"] is False
    assert body["bytes_done"] == 0


def test_load_progress_reports_stages_and_bytes(tmp_path, monkeypatch):
    """A load must publish a legible stage before it finishes, not after."""
    from modelmri import progress

    blobs = tmp_path / "hub" / "models--acme--tiny" / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "w").write_bytes(b"x" * 4096)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setattr(
        progress, "_expected_files", lambda _id, *_: (frozenset(), 8192)
    )

    tracker = progress._Tracker()
    tracker.start("acme/tiny")
    try:
        for _ in range(40):  # the watcher thread polls; give it a beat
            if tracker.snapshot().bytes_total:
                break
            time.sleep(0.05)
        snap = tracker.snapshot()
        assert snap.active is True
        assert snap.stage == "resolving"
        assert snap.bytes_done == 4096
        assert snap.bytes_total == 8192
        tracker.stage("weights", "downloading")
        assert tracker.snapshot().stage == "weights"
    finally:
        tracker.finish()
    done = tracker.snapshot()
    assert done.active is False and done.stage == "ready" and done.error is None


def test_load_progress_flags_a_stalled_download(tmp_path, monkeypatch):
    """A dead download does not raise, it just stops moving. Observed in the
    wild: 128 MB of 3 GB, unchanged, forever."""
    from modelmri import progress

    blobs = tmp_path / "models--acme--big" / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "w").write_bytes(b"x" * 128)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(
        progress, "_expected_files", lambda _id, *_: (frozenset(), 3000)
    )
    monkeypatch.setattr(progress, "STALL_AFTER_S", 0.0)  # stall immediately

    tracker = progress._Tracker()
    tracker.start("acme/big")
    # A download stall is a claim about a download, so it is only made during
    # the download stage. Before this was scoped, a load wedged while moving
    # weights to the GPU was reported as a stalled download.
    tracker.stage("weights")
    try:
        for _ in range(60):
            if "stalled" in tracker.snapshot().detail:
                break
            time.sleep(0.05)
        assert "stalled" in tracker.snapshot().detail
        assert tracker.snapshot().bytes_done == 128
    finally:
        tracker.finish()


def test_load_progress_does_not_cry_stall_over_a_cached_model(tmp_path, monkeypatch):
    """Bytes never move when nothing is downloading. That is not a stall."""
    from modelmri import progress

    blobs = tmp_path / "models--acme--warm" / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "w").write_bytes(b"x" * 1000)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(
        progress, "_expected_files", lambda _id, *_: (frozenset(), 1000)
    )
    monkeypatch.setattr(progress, "STALL_AFTER_S", 0.0)

    tracker = progress._Tracker()
    tracker.start("acme/warm")
    try:
        time.sleep(0.9)
        detail = tracker.snapshot().detail
        assert "stalled" not in detail
        assert "local cache" in detail
    finally:
        tracker.finish()


def test_a_cache_that_turns_out_to_be_downloading_stops_saying_it_is_not(
    tmp_path, monkeypatch
):
    """ "No download needed" is decided from the directory's size at t=0, and
    a directory can be big for reasons that are not "we already have it".

    Seen for real: the cache held a legacy `pytorch_model.bin` beside the
    safetensors, so the tree measured well past the expected total and was
    declared complete. The loader then downloaded `rust_model.ot` for minutes
    behind a message reading "reading from local cache, no download needed",
    with the byte counter climbing past 100%. Every number on screen was wrong
    in the same direction, which is the only kind of wrong nobody catches.
    """
    from modelmri import progress

    blobs = tmp_path / "models--acme--stale" / "blobs"
    blobs.mkdir(parents=True)
    # Bigger than expected, exactly like a cache holding a second format.
    (blobs / "old").write_bytes(b"x" * 2000)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(
        progress, "_expected_files", lambda _id, *_: (frozenset(), 1000)
    )

    tracker = progress._Tracker()
    tracker.start("acme/stale")
    try:
        time.sleep(0.9)
        assert "local cache" in tracker.snapshot().detail  # the initial verdict

        # ...and now bytes arrive, which means it was not cached at all.
        (blobs / "new").write_bytes(b"y" * (64 * 1024 * 1024))
        deadline = time.time() + 5
        while time.time() < deadline and "local cache" in tracker.snapshot().detail:
            time.sleep(0.05)
        assert "local cache" not in tracker.snapshot().detail
        assert "download" in tracker.snapshot().detail
    finally:
        tracker.finish()


def test_a_subfolder_copy_of_the_weights_is_not_counted_twice(tmp_path, monkeypatch):
    """The numerator and the denominator have to count the same files.

    meta-llama/Llama-3.2-1B-Instruct ships `original/consolidated.00.pth`
    beside `model.safetensors`, both 2.472 GB — the same weights in Meta's
    own format, which `from_pretrained` never opens. The total came from the
    repo's top-level files and the on-disk figure walked the whole tree, so a
    fully cached model read 4.955 GB of 2.481 GB. Measured: 199.7%, displayed
    as "5.0 GB / 2.5 GB" over a full bar.
    """
    from modelmri import progress

    snap = tmp_path / "models--meta-llama--Llama-3.2-1B-Instruct" / "snapshots" / "r1"
    (snap / "original").mkdir(parents=True)
    (snap / "model.safetensors").write_bytes(b"x" * 2472)
    (snap / "original" / "consolidated.00.pth").write_bytes(b"x" * 2472)
    (snap / "config.json").write_bytes(b"x" * 9)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))

    hf_id = "meta-llama/Llama-3.2-1B-Instruct"
    wanted = frozenset({"model.safetensors", "config.json"})
    assert progress._bytes_on_disk(hf_id, wanted) == 2481
    # And without a file list — offline, or a repo that publishes no sizes —
    # the shape rule has to reach the same answer, because "top level only"
    # is what excludes the variant folders.
    assert progress._bytes_on_disk(hf_id) == 2481


def test_two_cached_revisions_are_not_added_together(tmp_path, monkeypatch):
    """A load reads one revision. Summing them reports a multiple of the truth."""
    from modelmri import progress

    root = tmp_path / "models--acme--two" / "snapshots"
    for rev, size in (("aaa", 500), ("bbb", 700)):
        (root / rev).mkdir(parents=True)
        (root / rev / "model.safetensors").write_bytes(b"x" * size)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    assert progress._bytes_on_disk("acme/two") == 700


def test_the_previous_loads_watcher_cannot_write_into_the_next_one(
    tmp_path, monkeypatch
):
    """Watchers share one snapshot, so they must be scoped to their own load.

    This is what put "5.0 GB / 2.5 GB" — Llama-3.2-1B's figures — on screen
    beside Qwen2.5-0.5B, a model with neither number, while the Qwen load was
    still queued behind a load that had stopped returning.
    """
    from modelmri import progress

    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(
        progress, "_expected_files", lambda _id, *_: (frozenset(), 4242)
    )

    tracker = progress._Tracker()
    tracker.start("acme/first")
    stale_gen = tracker._gen
    tracker.start("acme/second")
    try:
        # The first load's watcher, still running, tries to publish.
        assert tracker._publish(stale_gen, bytes_total=999_999) is False
        assert tracker.snapshot().bytes_total != 999_999
        assert tracker.snapshot().hf_id == "acme/second"
    finally:
        tracker.finish()


def test_a_quiet_load_is_diagnosed_by_stage(monkeypatch):
    """Two silences, two diagnoses. Saying "download stalled" while a model is
    being copied to the GPU sends people to look at their network."""
    from modelmri import progress

    note = progress._Tracker._quiet_note
    # A download runs in a child process, so this process burning no CPU is
    # normal; only bytes count.
    assert "stalled" in note("weights", False, progress.STALL_AFTER_S + 1, 0.0)
    assert note("weights", True, progress.STALL_AFTER_S + 1, 0.0) == ""
    assert note("weights", False, progress.STALL_AFTER_S - 1, 0.0) == ""
    # Every other stage is this process working, so no CPU means stopped.
    quiet = progress.WEDGED_AFTER_S + 1
    assert "Hub" in note("resolving", False, quiet, 0.0)
    assert "stopped rather than slowed" in note("device", False, quiet, 0.0)
    # Still burning CPU is not a wedge, however long it takes.
    assert note("device", False, quiet, progress.WEDGED_CPU_S + 1) == ""


def test_a_sign_of_life_restarts_both_halves_of_the_wedge_window():
    """The two numbers the wedge test compares have to describe the SAME
    window, or it weighs a duration against CPU spent outside it.

    They were two variables reset in different places: the clock restarted
    whenever bytes moved or the stage changed, the CPU reading only on a stage
    change. So once bytes had moved, `cpu_s` went on accumulating from the top
    of the stage — and a few seconds of ordinary work then held it above the
    threshold for good. The check disarmed itself within seconds of any stage
    beginning and could never fire again during the stretch it exists to
    watch.
    """
    from modelmri import progress

    clock = {"t": 0.0, "cpu": 0.0}
    window = progress._Window(lambda: clock["t"], lambda: clock["cpu"])

    clock["t"], clock["cpu"] = 100.0, 40.0
    assert window.quiet_s() == 100.0
    assert window.cpu_s() == 40.0

    window.mark()
    clock["t"], clock["cpu"] = 130.0, 40.5
    # Thirty seconds quiet, and half a CPU-second inside those thirty seconds.
    # The old pairing would have reported 40.5 here — the whole stage's CPU
    # against a thirty-second window.
    assert window.quiet_s() == 30.0
    assert window.cpu_s() == 0.5


def test_the_download_meter_does_not_outlive_the_download():
    """Reported from the browser: 21 minutes on "Moving to the accelerator"
    under a bar drawn FULL, reading "2.5 GB / 2.5 GB · 0 bytes left · ~0s
    left".

    Every number there was the finished download's, still being published
    into a snapshot the next stage shares. A byte count is progress for the
    work that produced it and for nothing else, so a stage that did not
    measure them does not inherit them. Zero is the field's way of saying
    unknown, and the UI draws an indeterminate bar for it.
    """
    from modelmri import progress

    tracker = progress._Tracker()
    tracker.start_external("acme/big", stage="weights")
    try:
        tracker.publish(bytes_done=2_500_000_000, bytes_total=2_500_000_000)
        # `_eta` withholds a figure until there is history to divide, so the
        # clock is wound back rather than slept through: the point is the
        # "~0s left" the reader saw, which needs an ETA to exist at all.
        tracker._t0 -= 5.0
        finished = tracker.snapshot()
        assert finished.bytes_done == finished.bytes_total == 2_500_000_000
        assert finished.eta_s == 0.0  # true of the download, and only of it

        tracker.stage("device", "moving to the GPU")
        moving = tracker.snapshot()
        assert moving.stage == "device"
        assert (moving.bytes_done, moving.bytes_total) == (0, 0)
        assert moving.eta_s is None
    finally:
        tracker.finish()


def test_two_moments_of_one_download_keep_their_bytes():
    """`resolving` and `weights` are one transfer seen twice, so moving
    between them must not drop the figures. Scoping the counters per STAGE
    rather than per phase would blank the bar at the instant the download it
    describes actually starts."""
    from modelmri import progress

    tracker = progress._Tracker()
    tracker.start_external("acme/big", stage="resolving")
    try:
        tracker.publish(bytes_done=10, bytes_total=100)
        tracker.stage("weights")
        snap = tracker.snapshot()
        assert (snap.bytes_done, snap.bytes_total) == (10, 100)
    finally:
        tracker.finish()


def test_the_stage_that_owns_the_meter_publishes_into_it():
    """Dropping the download's numbers leaves a gap, and the stage that took
    over fills it with its own. That is the whole point of scoping them —
    `runtime.move_to_device` does exactly this for the copy onto the GPU."""
    from modelmri import progress

    tracker = progress._Tracker()
    tracker.start_external("acme/big", stage="weights")
    try:
        tracker.publish(bytes_done=1000, bytes_total=1000)
        tracker.stage("device", "moving to the GPU")
        tracker.publish(bytes_done=400, bytes_total=1000)
        tracker._t0 -= 5.0
        snap = tracker.snapshot()
        assert (snap.bytes_done, snap.bytes_total) == (400, 1000)
        assert snap.eta_s is not None and snap.eta_s > 0
    finally:
        tracker.finish()


def test_the_watcher_stops_counting_the_cache_once_the_download_is_over(
    tmp_path, monkeypatch
):
    """The cache directory measures the DOWNLOAD.

    The watcher wrote it into the shared snapshot on every poll for the rest
    of the load, so whatever was last on disk was published as the device
    move's progress — and, the download being finished, that is a full bar.
    """
    from modelmri import progress

    blobs = tmp_path / "models--acme--big" / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "w").write_bytes(b"x" * 500)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(
        progress, "_expected_files", lambda _id, *_: (frozenset(), 1000)
    )

    tracker = progress._Tracker()
    tracker.start("acme/big")
    try:
        for _ in range(60):
            if tracker.snapshot().bytes_done == 500:
                break
            time.sleep(0.05)
        assert tracker.snapshot().bytes_done == 500

        tracker.stage("device", "moving to the GPU")
        # More bytes land in the cache. They are not this stage's progress,
        # and neither was the 500 already there.
        (blobs / "more").write_bytes(b"x" * 400)
        time.sleep(1.6)
        snap = tracker.snapshot()
        assert snap.stage == "device"
        assert (snap.bytes_done, snap.bytes_total) == (0, 0)
    finally:
        tracker.finish()


def test_a_device_move_reporting_progress_is_not_called_wedged(tmp_path, monkeypatch):
    """Liveness is whatever the READER is watching move, not the cache
    directory — which stands perfectly still throughout a device move.

    Both halves are asserted on one tracker, because "it never fires" is as
    easy to pass by accident as "it always does": the moving stage must stay
    quiet and the still one must speak.
    """
    from modelmri import progress

    blobs = tmp_path / "models--acme--big" / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "w").write_bytes(b"x" * 500)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(progress, "_expected_files", lambda _id, *_: (frozenset(), 500))
    monkeypatch.setattr(progress, "WEDGED_AFTER_S", 0.05)
    # No amount of CPU may excuse it, so the only thing under test is whether
    # the published counter counts as a sign of life.
    monkeypatch.setattr(progress, "WEDGED_CPU_S", 10.0**9)

    tracker = progress._Tracker()
    tracker.start("acme/big")
    try:
        tracker.stage("device", "moving to the GPU")
        for step in range(1, 26):  # ~2.5s of visible progress
            tracker.publish(bytes_done=step * 40, bytes_total=1000)
            time.sleep(0.1)
            assert "stopped rather than slowed" not in tracker.snapshot().detail

        # Now stop moving. Same stage, same cache directory, nothing else
        # changed — and it has to be noticed.
        for _ in range(40):
            if "stopped rather than slowed" in tracker.snapshot().detail:
                break
            time.sleep(0.1)
        assert "stopped rather than slowed" in tracker.snapshot().detail
    finally:
        tracker.finish()


def test_the_hub_is_not_called_when_the_hub_is_off_limits(monkeypatch):
    """HF_HUB_OFFLINE is the hub's own switch, and the meter has to honour it."""
    import huggingface_hub

    from modelmri import progress

    def explode(*a, **k):
        raise AssertionError("model_info called with HF_HUB_OFFLINE set")

    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", explode)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert progress._expected_files("acme/tiny") == (frozenset(), 0)
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    assert progress._expected_files("acme/tiny") == (frozenset(), 0)  # explode, caught


def test_a_second_load_is_refused_rather_than_queued_forever():
    """One model loads at a time. A load that stops returning used to hold the
    lock for the rest of the session, and every later request blocked in it
    with no timeout and no message."""
    import threading

    from modelmri import progress, runtime
    from modelmri.errors import Refusal

    rt = runtime.ModelRuntime.__new__(runtime.ModelRuntime)
    rt._lock = threading.Lock()
    rt._lock.acquire()  # stand in for a load that is not coming back
    try:
        progress.TRACKER.start("acme/wedged")
        try:
            with pytest.raises(Refusal) as err:
                with rt._load_slot("acme/other"):
                    pass
        finally:
            progress.TRACKER.finish()
    finally:
        rt._lock.release()
    assert "acme/other" in str(err.value)
    assert "acme/wedged" in str(err.value)  # names what is holding it, not just "busy"


def test_the_model_picker_and_the_load_meter_agree_on_size(tmp_path, monkeypatch):
    """Both sides read the same cache, so they must count it the same way.

    The picker had its own walk and reported Llama-3.2-1B at 4.96 GB — the
    weights plus their `original/` duplicate — against the 2.48 GB a load
    actually reads.
    """
    from modelmri import paths, progress, runtime

    hub = tmp_path / "hub"
    snap = hub / "models--acme--dup" / "snapshots" / "r1"
    (snap / "original").mkdir(parents=True)
    (snap / "model.safetensors").write_bytes(b"x" * 1_000_000)
    (snap / "original" / "consolidated.00.pth").write_bytes(b"x" * 1_000_000)
    monkeypatch.setattr(paths, "hf_hub_cache", lambda: hub)
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))

    listed = {m["id"]: m["size_gb"] for m in runtime.local_hf_models()}
    assert listed["acme/dup"] == round(progress._bytes_on_disk("acme/dup") / 1e9, 2)
    assert listed["acme/dup"] == 0.0  # 1 MB, not 2


def test_the_recorder_wheel_size_is_stated_identically_everywhere():
    """Four files quote the recorder wheel's size, and they drifted apart.

    A commit in this repo already corrected "7 KiB" once, with the note "a
    figure nobody rechecks is a figure that drifts" — and then it drifted
    again: docs/index.md, docs/guides/agents.md and pyproject.toml still said
    7 KiB while README.md said 9 KiB and the wheel was 8.94 KiB. Prose has no
    build step, so this is the build step.

    Checks the four against each other always, and against the real wheel
    whenever one has been built.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sites = {
        "docs/index.md": r"stdlib only — ([\d.]+) KiB",
        "docs/guides/agents.md": r"No torch, no SDK pins, ([\d.]+) KiB",
        "pyproject.toml": r"Stdlib-only and ([\d.]+) KiB",
        "README.md": r"an? ([\d.]+) KiB wheel",
    }
    found: dict[str, float] = {}
    for rel, pattern in sites.items():
        text = (root / rel).read_text("utf-8")
        m = re.search(pattern, text)
        assert m, f"{rel} no longer states the recorder wheel size"
        found[rel] = float(m.group(1))

    assert len(set(found.values())) == 1, f"the four disagree: {found}"

    # And against the artefact itself, when there is one to weigh.
    wheels = sorted((root / "packages" / "modelmri-record" / "dist").glob("*.whl"))
    if wheels:
        actual = wheels[-1].stat().st_size / 1024
        stated = next(iter(found.values()))
        assert abs(actual - stated) < 0.1, (
            f"{wheels[-1].name} is {actual:.2f} KiB, the docs say {stated} KiB"
        )


def test_load_progress_records_failure():
    from modelmri import progress

    tracker = progress._Tracker()
    tracker.start("acme/nope")
    tracker.finish(error="gated repo")
    snap = tracker.snapshot()
    assert snap.stage == "error" and snap.error == "gated repo"
    assert snap.active is False


def test_load_progress_never_raises_on_a_missing_cache(tmp_path, monkeypatch):
    """The meter must not be able to break the load it is measuring."""
    from modelmri import progress

    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "absent"))
    assert progress._bytes_on_disk("acme/tiny") == 0


def test_expected_bytes_counts_only_what_a_load_downloads(monkeypatch):
    """Some repos ship tflite/rust/h5/flax copies of the same weights.
    Counting them made a fully-cached model report a fraction of itself
    forever."""
    from types import SimpleNamespace

    import huggingface_hub

    from modelmri import progress

    files = [
        SimpleNamespace(rfilename="model.safetensors", size=100),
        SimpleNamespace(rfilename="pytorch_model.bin", size=100),
        SimpleNamespace(rfilename="tf_model.h5", size=100),
        SimpleNamespace(rfilename="rust_model.ot", size=100),
        SimpleNamespace(rfilename="64-8bits.tflite", size=100),
        SimpleNamespace(rfilename="config.json", size=3),
        SimpleNamespace(rfilename="merges.txt", size=2),
        SimpleNamespace(rfilename="onnx/model.onnx", size=999),
        SimpleNamespace(rfilename="README.md", size=50),
    ]
    monkeypatch.setattr(
        huggingface_hub.HfApi,
        "model_info",
        # `timeout` is not decoration: unbounded, this call ran on the watcher
        # thread before any figure was published — 1502 ms measured against a
        # model that was already complete on disk, and no ceiling at all on a
        # connection that never answers.
        lambda self, _id, files_metadata=False, timeout=None: SimpleNamespace(
            siblings=files
        ),
    )
    assert progress._expected_bytes("acme/tiny") == 105
    # The same list decides what counts on disk, which is the whole point:
    # one set of names, so the two sides cannot disagree.
    names, total = progress._expected_files("acme/tiny")
    assert names == {"model.safetensors", "config.json", "merges.txt"}
    assert total == 105


def test_bytes_on_disk_handles_every_cache_layout(tmp_path, monkeypatch):
    """blobs-only, snapshots-only and both-populated must all report the truth."""
    from modelmri import progress

    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))

    def layout(name: str, blob: int, snap: int) -> str:
        root = tmp_path / f"models--acme--{name}"
        (root / "blobs").mkdir(parents=True)
        (root / "snapshots" / "abc").mkdir(parents=True)
        if blob:
            (root / "blobs" / "w").write_bytes(b"x" * blob)
        if snap:
            (root / "snapshots" / "abc" / "w.safetensors").write_bytes(b"x" * snap)
        return f"acme/{name}"

    # blobs moved into snapshots (current hub): blobs empty, bytes are real
    assert progress._bytes_on_disk(layout("moved", 0, 900)) == 900
    # mid-download: only the partial blob exists
    assert progress._bytes_on_disk(layout("partial", 400, 0)) == 400
    # Windows copies / Unix symlinks: both sides look full, must not double
    assert progress._bytes_on_disk(layout("both", 900, 900)) == 900


def test_ws_without_a_model_is_an_error_not_a_silent_done():
    with client().websocket_connect("/ws/generate") as ws:
        ws.send_text(json.dumps({"prompt": "hi"}))
        assert ws.receive_json()["type"] == "error"


def test_ws_reports_a_mid_stream_crash_as_an_error(monkeypatch):
    """A generation that raises used to reach the browser as {"type":"done"} —
    an empty answer that read as "the model had nothing to say". CUDA OOM and
    unsupported architectures both land here.

    THIS ASSERTION CHANGED, AND THE ARGUMENT IT HELD DID NOT.

    It used to require the literal string "CUDA out of memory" in the message,
    on the reasoning that a stream which stops mid-sentence has to say why.
    That reasoning is right. Requiring torch's own words for it was not: the
    message was `f"{type(err).__name__}: {err}"`, so the busiest error path in
    the app published exactly what the module header of server.py forbids —
    measured, a `RuntimeError("CUDA out of memory ... <absolute path>")`
    arrived in the browser with that path in it.

    So the test now holds the argument directly. The stream says why it
    stopped, it points at the terminal, and torch's text is not in it.
    """
    from modelmri.server import create_app

    app = create_app()
    secret = r"C:\\Users\\somebody\\.cache\\huggingface\\blobs\\9f3c1a"

    def boom(*_a, **_k):
        yield "The"
        raise RuntimeError(f"CUDA out of memory. Tried to allocate 20 GiB at {secret}")

    monkeypatch.setattr(app.state.runtime, "generate_stream", boom)
    monkeypatch.setattr(type(app.state.runtime), "loaded", property(lambda self: True))

    with TestClient(app).websocket_connect("/ws/generate") as ws:
        ws.send_text(json.dumps({"prompt": "hi"}))
        assert ws.receive_json() == {"type": "token", "text": "The"}
        final = ws.receive_json()
    assert final["type"] == "error", f"crash surfaced as {final!r}"
    # It says why it stopped, and where the rest of the answer is.
    assert "failed mid-generation" in final["message"]
    assert "modelmri serve" in final["message"]
    # And it does not say it in torch's words.
    assert "CUDA out of memory" not in final["message"]
    assert secret not in final["message"]
    assert "RuntimeError" not in final["message"]


def test_ws_still_sends_a_refusal_in_its_own_words(monkeypatch):
    """The other side of the arm above: a deliberate no is not generic.

    Ollama quitting mid-session, a recording with no model behind it — those
    messages were written for the reader, and blanketing them into "something
    failed" would lose the one sentence that says what to do. Same split as
    the REST handlers, on the same socket.
    """
    from modelmri.errors import Refusal
    from modelmri.server import create_app

    app = create_app()
    words = "ollama unreachable at http://127.0.0.1:11434: Connection refused."

    def refuse(*_a, **_k):
        raise Refusal(words)
        yield  # pragma: no cover - generator signature

    monkeypatch.setattr(app.state.runtime, "generate_stream", refuse)
    monkeypatch.setattr(type(app.state.runtime), "loaded", property(lambda self: True))

    with TestClient(app).websocket_connect("/ws/generate") as ws:
        ws.send_text(json.dumps({"prompt": "hi"}))
        final = ws.receive_json()
    assert final == {"type": "error", "message": words}


def test_discovery_finds_all_three_shapes(tmp_path):
    """A cache entry, a plain from_pretrained folder, and a .gguf."""
    from modelmri.discover import scan

    cache = tmp_path / "hub" / "models--Qwen--Qwen3-0.6B" / "snapshots" / "abc"
    cache.mkdir(parents=True)
    (cache / "model.safetensors").write_bytes(b"x" * 2048)

    folder = tmp_path / "my-models" / "finetune-v3"
    folder.mkdir(parents=True)
    (folder / "config.json").write_text("{}")
    (folder / "model.safetensors").write_bytes(b"x" * 4096)

    (tmp_path / "my-models" / "phi.gguf").write_bytes(b"x" * 512)

    found, truncated = scan(tmp_path)
    assert truncated is False
    by_kind = {f.kind: f for f in found}
    assert by_kind["hf-cache"].id == "Qwen/Qwen3-0.6B"
    assert by_kind["folder"].name == "finetune-v3"
    assert by_kind["folder"].id == str(folder)  # a path transformers can load
    assert by_kind["gguf"].loadable is False
    assert "Ollama" in by_kind["gguf"].note


def test_discovery_does_not_descend_into_a_model(tmp_path):
    """A model dir full of shards must be one result, not one per shard."""
    from modelmri.discover import scan

    m = tmp_path / "big-model"
    (m / "extra").mkdir(parents=True)
    (m / "config.json").write_text("{}")
    for i in range(4):
        (m / f"model-0000{i}.safetensors").write_bytes(b"x" * 128)
    (m / "extra" / "config.json").write_text("{}")
    (m / "extra" / "model.safetensors").write_bytes(b"x" * 128)

    found, _ = scan(tmp_path)
    assert len(found) == 1
    assert found[0].name == "big-model"


def test_discovery_skips_the_expensive_useless_directories(tmp_path):
    from modelmri.discover import scan

    for junk in ("node_modules", ".git", ".venv", "site-packages"):
        d = tmp_path / junk / "pretend-model"
        d.mkdir(parents=True)
        (d / "config.json").write_text("{}")
        (d / "model.safetensors").write_bytes(b"x" * 64)
    found, _ = scan(tmp_path)
    assert found == []


def test_discovery_reports_a_truncated_scan_instead_of_lying(tmp_path, monkeypatch):
    """A cut-short walk that looks complete is how you conclude a model is
    missing when it is not."""
    from modelmri import discover as disc

    deep = tmp_path
    for i in range(4):
        deep = deep / f"level{i}"
    deep.mkdir(parents=True)
    (deep / "config.json").write_text("{}")
    (deep / "model.safetensors").write_bytes(b"x" * 64)

    _, truncated = disc.scan(tmp_path, budget_s=-1.0)  # budget already spent
    assert truncated is True


def test_discovery_endpoint_shape():
    r = client().get("/api/models/discovered")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["models"], list)
    assert isinstance(body["roots"], list) and body["roots"]
    assert isinstance(body["truncated"], bool)


def test_accelerator_endpoint():
    r = client().get("/api/accelerator")
    assert r.status_code == 200
    body = r.json()
    assert body["kind"] in {"cuda", "rocm", "xpu", "mps", "cpu"}
    assert body["dtype"] in {"float16", "bfloat16", "float32"}
    assert body["reason"]  # always explains itself


def test_device_detect_forced_cpu():
    from modelmri import devices

    d = devices.detect(prefer="cpu")
    assert d.kind == "cpu" and d.torch_device == "cpu" and d.dtype == "float32"


def test_device_detect_unavailable_backend_falls_back():
    """Asking for a backend this machine lacks must degrade, never raise."""
    from modelmri import devices

    d = devices.detect(prefer="definitely-not-a-backend")
    assert d.kind == "cpu"
    assert "not available" in d.reason


def test_device_detect_survives_a_broken_driver(monkeypatch):
    import torch

    from modelmri import devices

    def boom():
        raise RuntimeError("driver exploded")

    monkeypatch.setattr(torch.cuda, "is_available", boom)
    assert devices.detect().kind in {"xpu", "mps", "cpu"}


def test_vla_status_unloaded():
    r = client().get("/api/vla")
    assert r.status_code == 200
    body = r.json()
    assert body["loaded"] is False
    assert body["mode"] == "unavailable"


def test_vla_attention_meta_always_200():
    r = client().get("/api/vla/attention/meta")
    assert r.status_code == 200
    assert r.json()["available"] is False


def test_vla_attention_without_analysis_is_409():
    assert client().get("/api/vla/attention?layer=0").status_code == 409


def test_vla_load_missing_cache_is_409(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    r = client().post("/api/vla/load", json={"repo": "lerobot/does-not-exist"})
    assert r.status_code == 409
    assert "not cached" in r.json()["error"]


def test_vla_snapshot_path_requires_a_ref(tmp_path):
    """A Refusal now, not a FileNotFoundError, and that is the whole point.

    The /api/vla/* handlers caught FileNotFoundError and answered 409 with the
    exception's text, which could not tell this sentence from pyarrow or av
    failing to open a file — measured, a library's `FileNotFoundError(2, ...,
    <abs path>)` was published as a refusal on four routes. These messages are
    Refusals so they cannot be confused with a library's, and the arms above
    them narrowed to ImportError.
    """
    from modelmri.errors import Refusal
    from modelmri.vla_data import snapshot_path

    base = tmp_path / "lerobot" / "hub" / "datasets--lerobot--pusht"
    (base / "refs").mkdir(parents=True)
    with pytest.raises(Refusal, match="No snapshot ref") as err:
        snapshot_path(tmp_path)
    assert not isinstance(err.value, OSError)


def test_vla_snapshot_path_reads_non_main_ref(tmp_path):
    """PushT's ref is 'v3.0' — assuming 'main' would break discovery."""
    from modelmri.vla_data import snapshot_path

    base = tmp_path / "lerobot" / "hub" / "datasets--lerobot--pusht"
    (base / "refs").mkdir(parents=True)
    (base / "refs" / "v3.0").write_text("abc123")
    (base / "snapshots" / "abc123").mkdir(parents=True)
    assert snapshot_path(tmp_path).name == "abc123"


def test_local_models_endpoint(tmp_path, monkeypatch):
    hub = tmp_path / "hub" / "models--Qwen--Qwen3-1.7B"
    hub.mkdir(parents=True)
    (hub / "w.bin").write_bytes(b"x" * 1000)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    r = client().get("/api/models/local")
    assert r.status_code == 200
    ids = [m["id"] for m in r.json()]
    assert "Qwen/Qwen3-1.7B" in ids


def test_ollama_endpoint_down(monkeypatch):
    from modelmri import ollama

    monkeypatch.setattr(
        ollama, "status", lambda host=None, timeout=None: {"up": False, "models": []}
    )
    r = client().get("/api/ollama")
    assert r.status_code == 200
    assert r.json()["up"] is False


def test_load_rejects_bad_source():
    r = client().post("/api/model/load", json={"hf_id": "x", "source": "wat"})
    assert r.status_code == 422


def test_load_ollama_down_is_409(monkeypatch):
    from modelmri import ollama

    monkeypatch.setattr(
        ollama, "status", lambda host=None, timeout=None: {"up": False, "models": []}
    )
    r = client().post("/api/model/load", json={"hf_id": "llama3", "source": "ollama"})
    assert r.status_code == 409


def _trace_doc():
    return {
        "name": "t1",
        "started_at": "2026-08-07T00:00:00Z",
        "steps": [
            {"kind": "llm_call", "name": "plan", "started_ms": 0, "duration_ms": 100},
            {
                "kind": "tool_call",
                "name": "pytest",
                "started_ms": 120,
                "duration_ms": 400,
                "error": True,
            },
        ],
    }


def trace_client(tmp_path):
    from modelmri.server import create_app

    return TestClient(create_app(trace_db=str(tmp_path / "t.sqlite")))


def test_trace_import_and_fetch(tmp_path):
    c = trace_client(tmp_path)
    r = c.post("/api/traces/import", json=_trace_doc())
    assert r.status_code == 200
    tid = r.json()["id"]

    listing = c.get("/api/traces").json()
    assert listing[0]["id"] == tid
    assert listing[0]["n_steps"] == 2
    assert listing[0]["n_errors"] == 1

    doc = c.get(f"/api/traces/{tid}").json()
    assert doc["name"] == "t1"
    assert [s["kind"] for s in doc["steps"]] == ["llm_call", "tool_call"]
    assert doc["steps"][1]["error"] is True


def test_trace_import_rejects_bad_kind(tmp_path):
    c = trace_client(tmp_path)
    bad = _trace_doc()
    bad["steps"][0]["kind"] = "nonsense"
    assert c.post("/api/traces/import", json=bad).status_code == 422


def test_trace_404(tmp_path):
    assert trace_client(tmp_path).get("/api/traces/nope").status_code == 404


def test_record_module_offline(tmp_path, monkeypatch):
    import json as _json

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MODELMRI_TRACE_DIR", str(tmp_path / "parked"))
    from modelmri.record import step, trace

    with trace("offline-run", endpoint="http://127.0.0.1:1/nope"):
        step("llm_call", name="a", duration_ms=10)
        with step("subagent", name="child"):
            step("tool_call", name="b", duration_ms=5)

    files = list((tmp_path / "parked").glob("*.json"))
    assert len(files) == 1
    doc = _json.loads(files[0].read_text())
    kinds = [s["kind"] for s in doc["steps"]]
    assert kinds == ["llm_call", "subagent", "tool_call"]
    assert doc["steps"][2]["parent_id"] == doc["steps"][1]["id"]


def test_the_standalone_recorder_keeps_its_protections():
    """The one implementation must keep redaction and the shutdown flush.

    This replaced an anchor-based "have not drifted" check between two copies.
    That check passed while the copies had drifted badly — the in-tree one was
    missing redaction entirely — because it only asserted that a handful of
    shared strings appeared in both. A guard that cannot see the difference it
    exists to catch is worse than none, since it reads as coverage.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    standalone = (
        root / "packages" / "modelmri-record" / "modelmri_record" / "__init__.py"
    ).read_text(encoding="utf-8")

    assert "redact_document" in standalone
    assert "atexit.register" in standalone
    assert "recording must never crash the host app" in standalone

    intree = (root / "modelmri" / "record" / "__init__.py").read_text(encoding="utf-8")
    assert "from modelmri_record import" in intree, (
        "modelmri.record is a second implementation again — that is how the "
        "redaction gap happened"
    )


def test_a_model_swap_mid_generation_does_not_poison_the_attention_view():
    """A load that lands while tokens are still streaming used to leave the
    OLD model's token ids in last_ids. The next attention request then ran the
    NEW model's weights over them: no crash, just numbers about nothing."""
    import torch

    from modelmri.runtime import ModelRuntime

    rt = ModelRuntime()
    # Stand the runtime up far enough that the epoch check is the ONLY thing
    # that can reject the request; otherwise the test passes on an unrelated
    # guard and proves nothing.
    rt.backend = "hf"
    rt.model = object()
    rt.sae = object()
    rt.last_ids = torch.zeros(5, dtype=torch.long)

    rt.epoch = 7
    rt.last_ids_epoch = 7
    rt.epoch = 8  # a load landed after that generation

    assert rt.attention_meta()["available"] is False
    assert "model changed" in rt.attention_meta()["reason"]
    for call in (lambda: rt.attention(0, 0), rt._compute_features):
        with pytest.raises(RuntimeError, match="different model"):
            call()


@pytest.mark.parametrize("call", ("attribute_tokens", "ablate_heads", "rank_features"))
def test_a_load_that_lands_while_an_intervention_waits_for_the_lock_is_refused(call):
    """The epoch check used to be taken OUTSIDE `self._lock`, and `load` holds
    that same lock across the epoch bump and the model swap. A load landing in
    the window between the check and the acquisition therefore ran one model's
    token ids under another model's weights and returned a full ranking, while
    the identical call one moment later refused. Nothing downstream can catch
    that: the ids are the right length and the KLs are finite."""
    import threading

    import torch

    from modelmri.runtime import ModelRuntime

    rt = ModelRuntime()
    rt.backend = "hf"
    rt.model = object()
    rt.last_ids = torch.zeros(5, dtype=torch.long)
    rt.last_n_prompt_tokens = 5
    rt.epoch = rt.last_ids_epoch = 1

    held, release = threading.Event(), threading.Event()

    def hold_the_lock():
        with rt._lock:
            held.set()
            release.wait(10)

    holder = threading.Thread(target=hold_the_lock)
    holder.start()
    assert held.wait(10)

    raised: list[BaseException] = []

    def intervene():
        try:
            getattr(rt, call)()
        except BaseException as err:  # the refusal is the result
            raised.append(err)

    caller = threading.Thread(target=intervene)
    caller.start()
    # Long enough for the caller to be blocked on the lock in the interleaving
    # this test is about. If it has not got there yet the epoch is already
    # bumped when it does, and the refusal has to come either way — which is
    # the point, and is why this cannot go flaky.
    time.sleep(0.3)
    rt.epoch = 2  # the load lands
    release.set()
    caller.join(10)
    holder.join(10)

    assert raised and "different model" in str(raised[0]), (
        f"{call} returned instead of refusing: {raised}"
    )


def test_derived_state_is_served_when_the_epoch_still_matches(monkeypatch):
    """The guard must not block the normal case."""
    import torch

    from modelmri.runtime import ModelRuntime

    rt = ModelRuntime()
    rt.epoch = 3
    rt.last_ids_epoch = 3
    rt.last_ids = torch.zeros(5, dtype=torch.long)

    class Cfg:
        num_hidden_layers = 12
        num_attention_heads = 12

    class M:
        config = Cfg()

    rt.model = M()
    rt.backend = "hf"
    meta = rt.attention_meta()
    assert meta["available"] is True and meta["n_layers"] == 12


def test_a_failed_cpu_fallback_does_not_leave_the_progress_meter_running(monkeypatch):
    """float32 on CPU needs roughly double the VRAM figure that just failed, so
    a big model hits this path routinely. Uncaught, the exception escaped
    before TRACKER.finish() ran and the meter stayed 'active' for the rest of
    the session, with its watcher thread polling the disk forever."""
    import torch

    from modelmri import progress
    from modelmri.runtime import ModelRuntime

    rt = ModelRuntime()

    class Boom:
        def to(self, *a, **k):
            raise torch.cuda.OutOfMemoryError("no room")

        def eval(self):  # pragma: no cover - never reached
            return self

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", lambda *a, **k: object()
    )
    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained", lambda *a, **k: Boom()
    )

    # Patched at the DETECTOR, not by assigning `rt.accel.kind` after
    # construction. `load` re-resolves the device on every call now — it has
    # to, or one deliberate CPU load makes every later load CPU too — so a
    # field set on the instance beforehand is overwritten before the fallback
    # branch is ever reached.
    #
    # This also stops the test asking about the machine it runs on. Assigning
    # the field passed here (a GPU box) and failed on CI (no GPU), where
    # `detect("auto")` truthfully answered CPU and the fallback under test was
    # therefore never entered. `prefer="cpu"` still goes to the real detector,
    # because the fallback's own answer is part of what is being tested.
    from modelmri import devices as devices_mod

    real_detect = devices_mod.detect

    def as_if_cuda(prefer: str = "auto"):
        if prefer == "cpu":
            return real_detect(prefer="cpu")
        found = real_detect(prefer="cpu")
        found.kind = "cuda"
        found.name = "pretend CUDA card"
        found.vram_gb = 8.0
        return found

    monkeypatch.setattr(devices_mod, "detect", as_if_cuda)

    with pytest.raises(RuntimeError, match="does not fit"):
        rt.load("acme/enormous")

    snap = progress.TRACKER.snapshot()
    assert snap.active is False, "progress meter left running after a failed load"
    assert snap.stage == "error" and snap.error


def test_sae_rejects_a_hook_point_it_cannot_place():
    """The hook POINT used to be discarded, so a resid_post SAE was silently
    fed the stream entering the block instead of leaving it — plausible
    features describing activations it was never trained on."""
    from modelmri.saes import SAEHandle

    with pytest.raises(ValueError, match="Unsupported hook point"):
        SAEHandle.load("acme/sae", "blocks.4.hook_mlp_out")
    with pytest.raises(ValueError, match="Cannot parse layer"):
        SAEHandle.load("acme/sae", "nonsense")


def test_sae_hook_point_selects_the_side_of_the_block(monkeypatch):
    """resid_pre must hook the block's input, resid_post its output."""
    import torch

    from modelmri.runtime import ModelRuntime

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = []

        def forward(self, x):
            return (x + 100,)

    for point, expected in (("resid_pre", 1.0), ("resid_post", 101.0)):
        rt = ModelRuntime()
        block = Block()
        rt._block = lambda _layer, b=block: b
        rt.last_ids = torch.zeros(2, dtype=torch.long)
        rt.last_ids_epoch = rt.epoch

        class FakeSAE:
            layer, point_ = 0, point
            d_sae = 4

            def __init__(self):
                self.point = point

            def encode(self, resid):
                captured.append(float(resid.flatten()[0]))
                return torch.zeros(resid.shape[0], 4)

        captured: list[float] = []
        rt.sae = FakeSAE()
        rt.model = lambda ids: block(torch.ones(1, 2, 3))
        rt._compute_features()
        assert captured and captured[0] == expected, (
            f"{point}: hooked the wrong side (got {captured})"
        )


# ---------------------------------------------------------- custom models


def test_custom_status_is_empty_and_names_its_roots():
    r = client().get("/api/custom")
    assert r.status_code == 200
    body = r.json()
    assert body["loaded"] is False
    assert body["path"] is None
    assert body["roots"], "the panel needs to tell people where it may load from"


def test_custom_run_without_a_model_is_422():
    r = client().post("/api/custom/run", json={})
    assert r.status_code == 422
    assert "no custom model is loaded" in r.json()["error"]


def test_custom_load_outside_the_roots_is_refused(tmp_path, monkeypatch):
    monkeypatch.delenv("MODELMRI_MODELS_DIR", raising=False)
    outside = tmp_path / "sneaky.py"
    outside.write_text("def load(): pass", encoding="utf-8")
    r = client().post("/api/custom/load", json={"path": str(outside)})
    assert r.status_code == 422
    assert "outside" in r.json()["error"]


def test_custom_load_never_500s_on_a_users_broken_adapter(tmp_path, monkeypatch):
    """Their code raising is a 422 with the reason, not a stack trace."""
    monkeypatch.chdir(tmp_path)
    bad = tmp_path / "bad.py"
    bad.write_text("import nonexistent_module_xyz\n", encoding="utf-8")
    r = client().post("/api/custom/load", json={"path": str(bad)})
    assert r.status_code == 422
    assert "ModuleNotFoundError" in r.json()["error"]


def test_custom_round_trip_through_the_api(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.chdir(tmp_path)
    adapter = tmp_path / "net.py"
    adapter.write_text(
        "import torch\nfrom torch import nn\n"
        "def load():\n    return nn.Sequential(nn.Linear(6, 4), nn.ReLU())\n"
        "def example_input():\n    return torch.randn(2, 6)\n",
        encoding="utf-8",
    )
    c = client()
    r = c.post("/api/custom/load", json={"path": str(adapter)})
    assert r.status_code == 200, r.text
    assert r.json()["n_params"] == 28  # 6*4 + 4

    r = c.post("/api/custom/run", json={})
    assert r.status_code == 200, r.text
    layers = r.json()["layers"]
    assert [row["kind"] for row in layers] == ["Linear", "ReLU"]
    assert layers[0]["out_shape"] == [2, 4]

    assert c.post("/api/custom/unload").json()["loaded"] is False


def test_custom_candidates_does_not_import_what_it_finds(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MODELMRI_MODELS_DIR", raising=False)
    (tmp_path / "landmine.py").write_text(
        "def load(): ...\nraise SystemExit('discovery imported me')\n", encoding="utf-8"
    )
    r = client().get("/api/custom/candidates")
    assert r.status_code == 200
    assert "landmine.py" in [a["name"] for a in r.json()["adapters"]]


# ------------------------------------------------------------ version drift


def test_the_version_is_single_sourced():
    """pyproject must not carry its own copy of the version.

    Four hand-maintained copies is four chances to ship a wrong one, and the
    UI footer already shipped "MRI-0.3" for the whole 0.4 line. hatchling
    reads modelmri/__init__.py; nothing else should restate it.
    """
    try:
        import tomllib
    except ModuleNotFoundError:  # tomllib is 3.11+; the dev group backfills 3.10
        import tomli as tomllib
    from pathlib import Path

    pj = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text("utf-8")
    )
    assert "version" in pj["project"].get("dynamic", []), (
        "pyproject declares a literal version again — it will drift"
    )
    assert pj["tool"]["hatch"]["version"]["path"] == "modelmri/__init__.py"


def test_metadata_agrees_with_the_package_version():
    """CITATION.cff is what people cite; a stale one misattributes the work."""
    import re
    from pathlib import Path

    cff = (Path(__file__).resolve().parents[1] / "CITATION.cff").read_text("utf-8")
    cited = re.search(r"^version:\s*(\S+)", cff, re.M)
    assert cited, "CITATION.cff has no version"
    assert cited.group(1) == __version__, (
        f"CITATION.cff says {cited.group(1)}, package is {__version__}"
    )


def test_the_ui_never_hardcodes_a_version():
    """The footer read the literal "MRI-0.3" while the package was 0.4.0."""
    import re
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "frontend" / "src"
    offenders = []
    for f in src.glob("*.tsx"):
        # Strip comments first: the fix for this bug is documented in a
        # comment that quotes the offending literal, and a check that reads
        # prose about a bug as the bug is a check nobody keeps.
        code = re.sub(r"/\*.*?\*/", " ", f.read_text("utf-8"), flags=re.S)
        code = re.sub(r"^\s*//.*$", " ", code, flags=re.M)
        for m in re.finditer(r"MRI-\d+\.\d+", code):
            offenders.append(f"{f.name}: {m.group(0)}")
    assert offenders == [], (
        f"hardcoded version strings in the UI: {offenders}. "
        "Read it from /api/session instead."
    )


def test_the_documented_import_path_redacts(tmp_path, monkeypatch):
    """`from modelmri.record import trace` must scrub credentials.

    It did not. modelmri/record was a hand-maintained second copy that never
    got redaction, while the standalone package did — and the README documents
    *this* path, so the promise in SECURITY.md ("credentials are removed
    before anything leaves your process") was not kept for the people most
    likely to follow the docs. It is one re-export now.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MODELMRI_TRACE_DIR", str(tmp_path / "parked"))
    from modelmri.record import step, trace

    secret = "sk-ant-api03-" + "A" * 40
    # Unreachable endpoint on purpose: delivery falls back to a local file,
    # which is exactly where a leak would be visible.
    with trace("leak-check", endpoint="http://127.0.0.1:1/nope"):
        step("llm_call", name="call", input=f"Authorization: Bearer {secret}")

    written = list((tmp_path / "parked").glob("*.json"))
    assert written, "the recorder wrote nothing to fall back to"
    body = written[0].read_text(encoding="utf-8")
    assert secret not in body, "the documented import path leaked a credential"
    assert "REDACTED" in body.upper() or "***" in body


def test_record_is_one_implementation_now():
    """Two copies of a security-relevant module cannot drift if there is one."""
    import modelmri_record as standalone

    import modelmri.record as intree

    assert intree.trace is standalone.trace
    assert intree.step is standalone.step
    assert intree.__version__ == standalone.__version__


def test_the_logit_lens_agrees_with_the_model_it_is_reading():
    """The last hidden state is ALREADY normed; the lens normed it again.

    HuggingFace decoders apply the final norm and then record the hidden
    state, so `lm_head(hidden_states[-1])` reproduces `logits` exactly.
    Applying the norm a second time computes head(norm(norm(h))), and a norm
    with learned gamma/beta is not idempotent.

    On gpt2 completing "…located in the city of", the top row read ' the'
    while the model actually said ' Paris'. That row supplies `final`, which
    anchors settled_at and the whole agreement column — so one wrong row
    mislabels the table.
    """
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    from modelmri.lens import logit_lens

    tok = transformers.AutoTokenizer.from_pretrained("gpt2")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        "gpt2", attn_implementation="eager"
    )
    ids = tok(
        "The Eiffel Tower is located in the city of", return_tensors="pt"
    ).input_ids

    with torch.no_grad():
        truth = tok.decode([int(model(ids).logits[0, -1].argmax())])

    rows = logit_lens(model, tok, ids, top_k=3)["layers"]
    assert rows[-1]["tokens"][0] == truth, (
        f"the lens's final row says {rows[-1]['tokens'][0]!r} but the model "
        f"says {truth!r} — the lens is not reading the model it claims to"
    )


# ------------------------------------------------------------------- paths


def test_paths_follow_each_platform_convention(monkeypatch, tmp_path):
    """One dotfile directory on every OS was a Unix habit, not a decision.

    Forced per-platform rather than trusting the developer's machine — this
    project ships to Linux and macOS and has only ever been run on Windows.
    """
    from modelmri import paths

    for var in (
        "MODELMRI_HOME",
        "XDG_DATA_HOME",
        "XDG_CONFIG_HOME",
        "XDG_CACHE_HOME",
        "LOCALAPPDATA",
        "APPDATA",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(paths.Path, "home", classmethod(lambda cls: tmp_path))

    monkeypatch.setattr(paths.sys, "platform", "linux")
    assert paths.data_dir() == tmp_path / ".local" / "share" / "modelmri"
    assert paths.config_dir() == tmp_path / ".config" / "modelmri"
    assert paths.cache_dir() == tmp_path / ".cache" / "modelmri"

    monkeypatch.setattr(paths.sys, "platform", "darwin")
    assert paths.data_dir() == tmp_path / "Library" / "Application Support" / "ModelMRI"
    assert paths.cache_dir() == tmp_path / "Library" / "Caches" / "ModelMRI"

    monkeypatch.setattr(paths.sys, "platform", "win32")
    assert paths.data_dir().parts[-1] == "ModelMRI"
    assert "AppData" in str(paths.data_dir())


def test_xdg_variables_are_honoured(monkeypatch, tmp_path):
    from modelmri import paths

    monkeypatch.delenv("MODELMRI_HOME", raising=False)
    monkeypatch.setattr(paths.sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "d"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "c"))
    assert paths.data_dir() == tmp_path / "d" / "modelmri"
    assert paths.config_dir() == tmp_path / "c" / "modelmri"


def test_modelmri_home_overrides_everything(monkeypatch, tmp_path):
    from modelmri import paths

    monkeypatch.setenv("MODELMRI_HOME", str(tmp_path / "portable"))
    assert paths.data_dir() == tmp_path / "portable" / "data"
    assert paths.config_dir() == tmp_path / "portable" / "config"
    assert paths.cache_dir() == tmp_path / "portable" / "cache"


def test_hf_cache_honours_the_variable_huggingface_actually_uses(monkeypatch, tmp_path):
    """HF_HUB_CACHE was ignored by all six hand-rolled copies of this.

    huggingface_hub checks HF_HUB_CACHE before HF_HOME, so a machine that set
    it downloaded models to one directory while ModelMRI searched another.
    """
    from modelmri import paths

    for var in ("HF_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"):
        monkeypatch.delenv(var, raising=False)

    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    assert paths.hf_hub_cache() == tmp_path / "hf" / "hub"

    # HF_HUB_CACHE wins over HF_HOME, as it does in the library.
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "direct"))
    assert paths.hf_hub_cache() == tmp_path / "direct"


def test_asking_where_things_go_does_not_create_them(monkeypatch, tmp_path):
    """A read-only question must stay read-only."""
    from modelmri import paths

    monkeypatch.setenv("MODELMRI_HOME", str(tmp_path / "untouched"))
    paths.describe()
    paths.data_dir()
    paths.config_dir()
    assert not (tmp_path / "untouched").exists()


def test_the_paths_endpoint_reports_them():
    r = client().get("/api/paths")
    assert r.status_code == 200
    body = r.json()
    for key in ("data", "config", "cache", "hf_home", "hf_hub_cache", "cwd"):
        assert body.get(key), f"/api/paths did not report {key}"


# ------------------------------------- `.mri` ids minted by the `/v1` surface


def test_an_mri_id_that_was_never_issued_is_a_404():
    r = client().get("/v1/mri/mri-nothing-like-this")
    assert r.status_code == 404
    assert "never issued" in r.json()["error"]["message"]


def test_an_evicted_mri_id_is_a_410_and_not_a_404():
    """ "Ask again, sooner" and "you have the wrong id" have different fixes,
    so they get different codes. One 404 for both sends a client to debug the
    wrong one."""
    app = create_app()
    store = app.state.mri_store
    store.limit = 1
    gone = store.put(b"first")
    store.put(b"second")

    r = TestClient(app).get(f"/v1/mri/{gone}")
    assert r.status_code == 410
    assert "evicted" in r.json()["error"]["message"]


def test_a_held_mri_comes_back_as_the_bytes_that_went_in():
    app = create_app()
    mri_id = app.state.mri_store.put(b"gzipped-mri-bytes")

    r = TestClient(app).get(f"/v1/mri/{mri_id}")
    assert r.status_code == 200
    assert r.content == b"gzipped-mri-bytes"
    assert mri_id in r.headers["content-disposition"]


# ------------------------- a path in the body names a file on the SERVER's disk


REMOTE = {"client": ("203.0.113.9", 51234)}
PATH_ENDPOINTS = [
    "/api/features/evidence",
    "/api/diff/models",
    "/api/ground",
    "/api/lens/tune",
]


@pytest.mark.parametrize("route", PATH_ENDPOINTS)
def test_a_file_path_from_the_network_is_refused(route: str):
    """These turn a string from the request body into a path and read it.
    `custom.allowed_roots` says why that matters in one line — "a local tool
    that will import any path on the filesystem on request is a nastier
    primitive than it looks" — and reading a corpus, a document or a prompt
    set is the same primitive with the result coming back as text.

    `serve` defaults to loopback but `--host` takes anything, so on a server
    bound to 0.0.0.0 this was an arbitrary file read for anyone who could
    route to the port.
    """
    r = TestClient(create_app(), client=REMOTE["client"]).post(
        route, json={"file": "/etc/passwd", "question": "q", "prompts": ["a"]}
    )
    assert r.status_code == 403
    assert "only possible from this machine" in r.json()["error"]


@pytest.mark.parametrize("route", PATH_ENDPOINTS)
def test_a_file_path_from_another_site_is_refused(route: str):
    """Loopback alone does not settle it: a tab open on any website can POST
    to localhost and the request arrives from 127.0.0.1 like every other."""
    r = TestClient(create_app()).post(
        route,
        json={"file": "/etc/passwd", "question": "q", "prompts": ["a"]},
        headers={"Origin": "https://example.com"},
    )
    assert r.status_code == 403
    assert "came from another site" in r.json()["error"]


@pytest.mark.parametrize("route", PATH_ENDPOINTS)
def test_the_guard_only_covers_the_branch_that_names_a_path(route: str):
    """A request that sends its text inline names no file, so it must not be
    turned away — the guard is about the filesystem, not about the endpoint."""
    r = TestClient(create_app(), client=REMOTE["client"]).post(
        route, json={"texts": ["a"], "question": "q", "prompts": ["a"]}
    )
    assert r.status_code != 403, "a request naming no path was refused as if it did"


# ---------------------------------------------------------------------------
# Multimodal configs.
#
# MEASURED on google/gemma-4-E4B-it-qat-mobile-transformers: `Gemma4Config`
# has no `num_hidden_layers` at all. The shape lives in `text_config` (42
# layers, 8 heads) beside `vision_config` (16) and `audio_config` (12), and the
# decoder blocks are at `model.language_model.layers` rather than
# `model.layers`. Every Gemma 3 and Gemma 4 is shaped this way, so before this
# the newest models the tool can load reported `n_layers: None` and no
# introspection feature worked on them.
# ---------------------------------------------------------------------------


def test_a_multimodal_config_reports_the_text_towers_shape():
    from modelmri.runtime import text_config

    class Vision:
        num_hidden_layers = 16
        num_attention_heads = 12

    class Text:
        num_hidden_layers = 42
        num_attention_heads = 8

    class Multimodal:
        text_config = Text()
        vision_config = Vision()

    found = text_config(Multimodal())

    assert found.num_hidden_layers == 42, "took the vision tower, or nothing"
    assert found.num_attention_heads == 8


def test_a_plain_config_is_returned_unchanged():
    """Every caller uses the helper unconditionally, so it has to be a no-op on
    the single-tower models that were working before."""
    from modelmri.runtime import text_config

    class Plain:
        num_hidden_layers = 28
        num_attention_heads = 16

    cfg = Plain()
    assert text_config(cfg) is cfg


def test_a_config_with_neither_does_not_raise():
    """A shape that cannot be read is `None` downstream, not an exception on
    the load path."""
    from modelmri.runtime import text_config

    class Empty:
        pass

    cfg = Empty()
    assert text_config(cfg) is cfg
    assert getattr(text_config(cfg), "num_hidden_layers", None) is None


def test_the_language_tower_is_preferred_over_the_vision_tower():
    """A multimodal model has BOTH `model.language_model.layers` and
    `model.vision_tower.encoder.layers`. Picking the wrong one draws the image
    encoder's attention while the panel says it is the text model's."""
    from modelmri.runtime import decoder_blocks

    class Root:
        class model:
            class language_model:
                layers = ["text"] * 42

            class vision_tower:
                class encoder:
                    layers = ["vision"] * 16

    blocks = decoder_blocks(Root())

    assert blocks is not None and len(blocks) == 42
    assert blocks[0] == "text"


def test_an_unknown_layout_reports_none_rather_than_guessing():
    from modelmri.runtime import decoder_blocks

    class Exotic:
        pass

    assert decoder_blocks(Exotic()) is None


def test_ws_answers_a_frame_that_is_not_json_rather_than_dropping_the_socket():
    """`/ws/generate` had no arm for a malformed frame.

    Measured against this file: `hello` raised JSONDecodeError, `[1,2]` raised
    AttributeError on `.get`, and each escaped to uvicorn, which closes 1011
    with no `error` and no `done`. Starlette routes the app-level `Exception`
    handler exclusively through `ServerErrorMiddleware`, which returns early
    for non-http scopes, so a websocket gets no backstop from it — and
    `docs/reference/api.md` documents this endpoint as public API that answers
    with an error frame. The shipped playground registers no `onclose`, so its
    Generate button would stay disabled forever.
    """
    with client().websocket_connect("/ws/generate") as ws:
        ws.send_text("hello")
        answer = ws.receive_json()
        assert answer["type"] == "error"
        assert "not JSON" in answer["message"]

        # AND THE SOCKET SURVIVES. One client's bad frame is not a reason to
        # drop a connection the panel is still using.
        ws.send_text(json.dumps({"prompt": "hi"}))
        assert ws.receive_json()["type"] in ("error", "token", "done")


def test_ws_answers_a_json_frame_that_is_not_an_object():
    """`[1, 2]` parses fine and then raises AttributeError on `.get`."""
    with client().websocket_connect("/ws/generate") as ws:
        ws.send_text(json.dumps([1, 2]))
        answer = ws.receive_json()
        assert answer["type"] == "error"
        assert "list" in answer["message"]
        assert "`prompt`" in answer["message"]
