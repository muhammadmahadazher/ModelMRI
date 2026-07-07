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
