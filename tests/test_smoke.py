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
