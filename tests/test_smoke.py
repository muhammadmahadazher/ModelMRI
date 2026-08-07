"""Smoke tests — no model download, just the app surface."""

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
