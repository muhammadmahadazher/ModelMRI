"""Unload, and pointing the scan somewhere else.

Both exist because of a gap a user found: the model that actually holds your
VRAM had no way to let go of it, and the scan could only look where the server
happened to be started.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from modelmri import custom
from modelmri.server import create_app


def _local_client() -> TestClient:
    """A client that looks like the person at the keyboard.

    TestClient reports its host as "testclient", which the scan route
    correctly refuses — widening where the tool may import from is loopback
    only. Spelling the address out keeps the tests honest about which side of
    that gate they are on.
    """
    return TestClient(create_app(), client=("127.0.0.1", 5900))


@pytest.fixture(autouse=True)
def _forget_roots():
    yield
    custom.clear_roots()


def test_unload_with_nothing_loaded_is_not_an_error():
    """Pressing it twice is not a mistake, and a 500 on the second press would
    read as one."""
    c = _local_client()
    r = c.post("/api/model/unload")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["unloaded"] is False
    assert body["was"] is None
    assert body["freed_bytes"] == 0
    assert body["status"]["loaded"] is False


def test_unload_clears_everything_a_load_set():
    """A half-unloaded runtime is worse than either state: the SAE would be
    bound to a model that is gone, and the retained attention would describe a
    sequence nothing can reproduce."""
    app = create_app()
    rt = app.state.runtime
    rt.model = object()
    rt.tokenizer = object()
    rt.hf_id = "acme/tiny"
    rt.backend = "hf"
    rt.last_ids = object()
    rt.last_user_span = (1, 2)
    rt.sae = object()
    rt._feats = object()
    rt._steer = (5, 1.0)
    rt.replay = object()

    before = rt.epoch
    out = rt.unload()

    assert out["unloaded"] is True and out["was"] == "acme/tiny"
    for field in (
        "model",
        "tokenizer",
        "hf_id",
        "backend",
        "last_ids",
        "last_user_span",
        "sae",
        "_feats",
        "_steer",
        "replay",
    ):
        assert getattr(rt, field) is None, f"{field} survived the unload"
    # The epoch has to move, or every panel keyed on it keeps showing the
    # analysis of a model that is no longer there.
    assert rt.epoch > before


def test_a_folder_can_be_added_to_the_scan(tmp_path):
    """The scan was limited to the launch directory, which is the wrong
    question to ask somebody whose model is on another drive."""
    c = _local_client()
    elsewhere = tmp_path / "over-here"
    elsewhere.mkdir()
    (elsewhere / "my_adapter.py").write_text(
        "def load():\n    return None\n", encoding="utf-8"
    )

    r = c.post("/api/custom/scan", json={"path": str(elsewhere)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert str(elsewhere.resolve()) in body["roots"]
    assert any(a["name"] == "my_adapter.py" for a in body["adapters"]), body["adapters"]


def test_adding_a_folder_widens_the_boundary_rather_than_removing_it(tmp_path):
    """`_resolve` still refuses anything outside the allowed roots. A local
    tool that will import any path handed to it is a nastier primitive than it
    looks, so the boundary moves — it does not disappear."""
    outside = Path(tempfile.mkdtemp()) / "not-added.py"
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_text("def load():\n    return None\n", encoding="utf-8")

    inside = tmp_path / "added"
    inside.mkdir()
    custom.add_root(str(inside))

    with pytest.raises(custom.AdapterError) as err:
        custom.resolve_under_roots(str(outside))
    assert "outside the directories" in str(err.value)


def test_a_file_is_refused_with_the_thing_to_do_instead(tmp_path):
    f = tmp_path / "model.pt"
    f.write_bytes(b"x")
    c = _local_client()
    r = c.post("/api/custom/scan", json={"path": str(f)})
    assert r.status_code == 422
    # Naming the problem without the remedy leaves the reader stuck.
    assert "folder" in r.json()["error"]


def test_a_missing_folder_says_which_one():
    c = _local_client()
    r = c.post("/api/custom/scan", json={"path": "/definitely/not/here"})
    assert r.status_code == 422
    assert "does not exist" in r.json()["error"]


def test_a_request_from_the_network_may_not_widen_the_boundary(tmp_path):
    """The reason this feature is not a security hole.

    Widening the roots turns "import from the directory I was started in" into
    "import from a directory somebody named", and the adapter loader IMPORTS
    what it loads — so on a server bound to 0.0.0.0 this would be remote code
    execution in two requests. The person at the keyboard may move the
    boundary; a stranger on the network may not.
    """
    elsewhere = tmp_path / "over-here"
    elsewhere.mkdir()

    remote = TestClient(create_app(), client=("10.0.0.7", 51234))
    r = remote.post("/api/custom/scan", json={"path": str(elsewhere)})
    assert r.status_code == 403, r.text
    assert "only possible from this machine" in r.json()["error"]
    # And it really did not take effect.
    assert elsewhere.resolve() not in custom.allowed_roots()


def test_the_root_of_a_filesystem_is_refused():
    """Adding one would make every .py on the machine importable, which is not
    what anybody means by "the folder my model is in"."""
    c = _local_client()
    root = Path(Path.cwd().anchor or "/")
    r = c.post("/api/custom/scan", json={"path": str(root)})
    assert r.status_code == 422, r.text
    assert "root of a filesystem" in r.json()["error"]
