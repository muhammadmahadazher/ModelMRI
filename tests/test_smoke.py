"""Smoke tests — no model download, just the app surface."""

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
