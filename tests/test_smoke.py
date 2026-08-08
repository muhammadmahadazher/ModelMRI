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
    monkeypatch.setattr(hub, "CONFIG", target)
    monkeypatch.setattr(
        hub, "whoami", lambda tok=None: hub.HubAuth(signed_in=True, user="tester")
    )
    auth = hub.sign_in("hf_fake")
    assert auth.user == "tester"
    assert json.loads(target.read_text())["token"] == "hf_fake"


def test_ollama_pull_when_daemon_is_down(monkeypatch):
    from modelmri import ollama

    def boom(name, host=ollama.DEFAULT_HOST):
        raise RuntimeError("ollama unreachable at 127.0.0.1:11434")
        yield  # pragma: no cover - generator signature

    monkeypatch.setattr(ollama, "pull", boom)
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
    monkeypatch.setattr(progress, "_expected_bytes", lambda _id: 8192)

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
    monkeypatch.setattr(progress, "_expected_bytes", lambda _id: 3000)
    monkeypatch.setattr(progress, "STALL_AFTER_S", 0.0)  # stall immediately

    tracker = progress._Tracker()
    tracker.start("acme/big")
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
    monkeypatch.setattr(progress, "_expected_bytes", lambda _id: 1000)
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
    """gpt2 ships tflite/rust/h5/flax copies of itself. Counting them made a
    fully-cached model report 26% forever."""
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
        lambda self, _id, files_metadata=False: SimpleNamespace(siblings=files),
    )
    assert progress._expected_bytes("acme/tiny") == 105


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
    unsupported architectures both land here."""
    from modelmri.server import create_app

    app = create_app()

    def boom(*_a, **_k):
        yield "The"
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(app.state.runtime, "generate_stream", boom)
    monkeypatch.setattr(type(app.state.runtime), "loaded", property(lambda self: True))

    with TestClient(app).websocket_connect("/ws/generate") as ws:
        ws.send_text(json.dumps({"prompt": "hi"}))
        assert ws.receive_json() == {"type": "token", "text": "The"}
        final = ws.receive_json()
    assert final["type"] == "error", f"crash surfaced as {final!r}"
    assert "CUDA out of memory" in final["message"]


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
    from modelmri.vla_data import snapshot_path

    base = tmp_path / "lerobot" / "hub" / "datasets--lerobot--pusht"
    (base / "refs").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="No snapshot ref"):
        snapshot_path(tmp_path)


def test_vla_snapshot_path_reads_non_main_ref(tmp_path):
    """PushT's ref is 'v3.0' — assuming 'main' would break discovery."""
    from modelmri.vla_data import snapshot_path

    base = tmp_path / "lerobot" / "hub" / "datasets--lerobot--pusht"
    (base / "refs").mkdir(parents=True)
    (base / "refs" / "v3.0").write_text("abc123")
    (base / "snapshots" / "abc123").mkdir(parents=True)
    assert snapshot_path(tmp_path).name == "abc123"


def test_local_models_endpoint(tmp_path, monkeypatch):
    hub = tmp_path / "hub" / "models--openai-community--gpt2"
    hub.mkdir(parents=True)
    (hub / "w.bin").write_bytes(b"x" * 1000)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    r = client().get("/api/models/local")
    assert r.status_code == 200
    ids = [m["id"] for m in r.json()]
    assert "openai-community/gpt2" in ids


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
    from modelmri.record import step, trace

    with trace("offline-run", endpoint="http://127.0.0.1:1/nope"):
        step("llm_call", name="a", duration_ms=10)
        with step("subagent", name="child"):
            step("tool_call", name="b", duration_ms=5)

    files = list((tmp_path / "modelmri-traces").glob("*.json"))
    assert len(files) == 1
    doc = _json.loads(files[0].read_text())
    kinds = [s["kind"] for s in doc["steps"]]
    assert kinds == ["llm_call", "subagent", "tool_call"]
    assert doc["steps"][2]["parent_id"] == doc["steps"][1]["id"]


def test_record_implementations_have_not_drifted():
    """modelmri/record and packages/modelmri-record are the same code until
    the standalone package is published and the in-tree copy becomes a
    re-export. Two copies of a security-relevant module silently diverging is
    exactly how a redaction fix lands in one and not the other."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    intree = (root / "modelmri" / "record" / "__init__.py").read_text(encoding="utf-8")
    standalone = (
        root / "packages" / "modelmri-record" / "modelmri_record" / "__init__.py"
    ).read_text(encoding="utf-8")

    # The standalone is the source of truth and is strictly ahead: it adds
    # redaction and the shutdown flush. What must not happen is the shared
    # core diverging.
    for anchor in (
        "def step(",
        "def instrument_anthropic(",
        "parent_id",
        "recording must never crash the host app",
    ):
        assert anchor in intree, f"in-tree copy lost {anchor!r}"
        assert anchor in standalone, f"standalone copy lost {anchor!r}"

    # And the standalone must keep the protections the in-tree one lacks.
    assert "redact_document" in standalone
    assert "atexit.register" in standalone


def test_a_model_swap_mid_generation_does_not_poison_the_attention_view():
    """A load that lands while tokens are still streaming used to leave the
    OLD model's token ids in last_ids. The next attention request then ran the
    NEW model's weights over them: no crash, just numbers about nothing."""
    from modelmri.runtime import ModelRuntime

    import torch

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


def test_derived_state_is_served_when_the_epoch_still_matches(monkeypatch):
    """The guard must not block the normal case."""
    from modelmri.runtime import ModelRuntime

    import torch

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
    rt.accel.kind = "cuda"  # so the CPU fallback branch is taken

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
