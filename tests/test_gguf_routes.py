# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""The GGUF load path above `gguf_load` itself: the runtime and the routes.

`test_gguf_load.py` covers the arithmetic and the refusals. This covers the
half that had no tests at all, and the gap was not theoretical — deleting
`self.gguf = None` from the HF load path left every one of the 882 tests
green, and that line is the only thing stopping a full-precision model from
being captioned as quantised for the rest of the session.

Nothing here loads a model. Everything asserts on provenance bookkeeping,
route wiring and refusal text, all of which are decisions rather than
computations and none of which need weights to check.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from test_gguf_read import build  # the same synthetic-header builder

from modelmri import custom
from modelmri.runtime import ModelRuntime
from modelmri.server import create_app

F32 = 0


def _client() -> TestClient:
    """Loopback, like the person at the keyboard — the root gate checks it."""
    return TestClient(create_app(), client=("127.0.0.1", 5900))


@pytest.fixture(autouse=True)
def _forget_roots():
    yield
    custom.clear_roots()


def _gguf(tmp_path, *, arch="llama", elements=1_000, name="model.gguf"):
    path = build(
        tmp_path,
        metadata={"general.architecture": (0x8, arch)},
        tensors=[("token_embd.weight", F32, [elements], 0)],
    )
    return path.rename(path.parent / name)


# --------------------------------------------------------------- provenance


def test_a_fresh_runtime_has_no_gguf_provenance():
    """None, not {}. An empty dict reads as "from a GGUF, nothing to say"."""
    assert ModelRuntime().gguf is None


def test_status_carries_the_provenance_field_even_when_empty():
    """The UI branches on its presence, so it has to be in the shape always."""
    assert "gguf" in ModelRuntime().status().to_dict()


def test_unload_clears_the_provenance():
    """A caveat outliving the model it describes would caption the NEXT model
    as quantised."""
    r = ModelRuntime()
    r.gguf = {"plan": {"path": "x.gguf"}}
    r.unload()
    assert r.gguf is None


@pytest.mark.parametrize(
    "line",
    [
        # Every assignment of self.model must sit beside a self.gguf write.
        # Asserted against the source because the alternative is loading three
        # models in a unit test.
        "self.gguf = None",
    ],
)
def test_every_load_path_writes_the_provenance_field(line):
    """Regression for the untested line. `runtime.py` assigns `self.model` on
    four paths — HF, ollama, unload, gguf — and three of them must clear this
    while the fourth sets it. If a fifth appears without one, this fails."""
    import inspect

    from modelmri import runtime as runtime_mod

    src = inspect.getsource(runtime_mod)
    # Three clears (hf, ollama, unload) plus one set (`self.gguf = report`).
    assert src.count(line) >= 3, f"only {src.count(line)} clears of self.gguf"
    assert "self.gguf = report" in src


def test_the_status_dataclass_documents_what_none_means():
    from modelmri import runtime as runtime_mod

    doc = runtime_mod.ModelStatus.__doc__ or ""
    src = __import__("inspect").getsource(runtime_mod.ModelStatus)
    assert "quantised" in (doc + src).lower()


# ------------------------------------------------------------- plan_gguf


def test_plan_reports_whether_this_file_is_already_the_loaded_model(tmp_path):
    """Without it the panel tells you a model you are already running will not
    fit — true of a SECOND copy, and not the question being asked."""
    p = _gguf(tmp_path, elements=1_000)
    r = ModelRuntime()
    assert r.plan_gguf(str(p))["already_loaded"] is False
    r.gguf = {"plan": {"path": str(p)}}
    assert r.plan_gguf(str(p))["already_loaded"] is True


def test_already_loaded_compares_paths_not_strings(tmp_path):
    """`C:\\a\\b.gguf` and `C:/a/b.gguf` are the same file, and a string
    compare would call them different."""
    p = _gguf(tmp_path, elements=1_000)
    r = ModelRuntime()
    r.gguf = {"plan": {"path": str(p).replace("\\", "/")}}
    assert r.plan_gguf(str(p))["already_loaded"] is True


def test_a_different_gguf_is_not_already_loaded(tmp_path):
    a = _gguf(tmp_path, name="a.gguf")
    b = _gguf(tmp_path, name="b.gguf")
    r = ModelRuntime()
    r.gguf = {"plan": {"path": str(a)}}
    assert r.plan_gguf(str(b))["already_loaded"] is False


def test_the_plan_names_the_device_it_was_computed_for(tmp_path):
    """The verdict branches on it, so a plan without it is unreproducible."""
    p = _gguf(tmp_path, elements=1_000)
    assert ModelRuntime().plan_gguf(str(p))["device"]


# ---------------------------------------------------------------- routes


def test_the_plan_route_refuses_a_path_outside_the_roots(tmp_path):
    """Same boundary as every other file-reading route. A local tool that
    reads any path on request is a nastier primitive than it looks."""
    p = _gguf(tmp_path)
    r = _client().get("/api/gguf/plan", params={"path": str(p)})
    assert r.status_code == 409
    assert "outside the directories" in r.json()["error"]


def test_the_load_route_refuses_a_path_outside_the_roots(tmp_path):
    p = _gguf(tmp_path)
    r = _client().post("/api/gguf/load", json={"path": str(p)})
    assert r.status_code == 409
    assert "outside the directories" in r.json()["error"]


def test_the_plan_route_answers_for_a_file_under_a_root(tmp_path):
    p = _gguf(tmp_path, elements=1_000_000)
    custom.add_root(str(tmp_path))
    r = _client().get("/api/gguf/plan", params={"path": str(p)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["parameters"] == 1_000_000
    assert body["resident_bytes"] == 1_000_000 * (
        2 if body["dtype"] != "float32" else 4
    )
    assert body["verdict"] in {"fits", "tight", "will not fit", "unknown"}


def test_an_unknown_dtype_is_a_422_not_an_invented_number(tmp_path):
    """`bf16` is the spelling everyone writes. It used to score 4 bytes per
    parameter instead of 2 and the preflight reported double."""
    p = _gguf(tmp_path, elements=1_000)
    custom.add_root(str(tmp_path))
    r = _client().get("/api/gguf/plan", params={"path": str(p), "dtype": "bf16"})
    assert r.status_code == 422
    assert "unknown dtype" in r.json()["error"]


def test_the_load_route_rejects_an_unknown_dtype_too(tmp_path):
    """422 whether or not the optional `gguf` extra is installed.

    This originally asserted the same thing and passed locally while failing
    on all four CI cells, because `load()` called `_require_gguf()` before
    `plan()`: with the extra absent the answer was 409 "install
    modelmri[gguf]" — true, but not what was wrong with the request, and the
    caller fixes that and meets the real error second. An argument error must
    not depend on which extras are present.
    """
    p = _gguf(tmp_path, elements=1_000)
    custom.add_root(str(tmp_path))
    r = _client().post("/api/gguf/load", json={"path": str(p), "dtype": "float64"})
    assert r.status_code == 422, r.text
    assert "unknown dtype" in r.json()["error"]


def test_a_bad_argument_beats_a_missing_optional_dependency(tmp_path, monkeypatch):
    """Pinned directly, so it holds on a machine that HAS the extra too."""
    from modelmri import gguf_load
    from modelmri.errors import BadRequest, Refusal

    def no_gguf():
        raise Refusal("install modelmri[gguf]")

    monkeypatch.setattr(gguf_load, "_require_gguf", no_gguf)
    p = _gguf(tmp_path, elements=1_000)
    with pytest.raises(BadRequest, match="unknown dtype"):
        gguf_load.load(p, dtype="float64", device="cpu")


def test_a_companion_file_is_refused_by_the_route_with_a_sentence(tmp_path):
    p = _gguf(tmp_path, name="mmproj-model-f16.gguf")
    custom.add_root(str(tmp_path))
    r = _client().get("/api/gguf/plan", params={"path": str(p)})
    assert r.status_code == 409
    assert "projector" in r.json()["error"]


def test_an_unsupported_architecture_is_refused_by_the_route(tmp_path):
    p = _gguf(tmp_path, arch="rwkv7")
    custom.add_root(str(tmp_path))
    r = _client().get("/api/gguf/plan", params={"path": str(p)})
    assert r.status_code == 409
    assert "rwkv7" in r.json()["error"]


def test_the_reader_route_still_works_beside_the_new_ones(tmp_path):
    """`/api/gguf` is a literal path and must not be shadowed by, or shadow,
    `/api/gguf/plan`. FastAPI matches in definition order."""
    p = _gguf(tmp_path, elements=1_000)
    custom.add_root(str(tmp_path))
    c = _client()
    assert c.get("/api/gguf", params={"path": str(p)}).status_code == 200
    assert c.get("/api/gguf/plan", params={"path": str(p)}).status_code == 200


def test_all_three_gguf_routes_are_registered():
    paths = {r.path for r in create_app().routes if "gguf" in getattr(r, "path", "")}
    assert paths == {"/api/gguf", "/api/gguf/plan", "/api/gguf/load"}


def test_the_load_route_requires_a_path():
    """A 422 from the model, not a 500 from a None reaching Path()."""
    assert _client().post("/api/gguf/load", json={}).status_code == 422


# ---------------------------------------------------------------------------
# The quantisation-comparison route must not answer "does this path exist?"
# ---------------------------------------------------------------------------


def _behaviour(client, original, quantised):
    return client.post(
        "/api/quantdiff/behaviour",
        json={"quantised": quantised, "original": original, "prompt": "x"},
    )


def test_an_existing_and_a_missing_path_are_indistinguishable(tmp_path):
    """The route used to branch on `Path(original).exists()`, so an existing
    path outside the roots got "outside the directories" and a missing one
    fell through to the hub and got something else. That difference is an
    existence oracle for any file on the machine — small, but a primitive the
    server has no reason to hand out. CodeQL called it uncontrolled data in a
    path expression and was right.

    Both must now produce the same refusal.
    """
    p = _gguf(tmp_path)
    custom.add_root(str(tmp_path))
    c = _client()

    # OUTSIDE the roots, which is where the oracle would matter. A path under
    # a root is one the caller was already allowed to read, so telling them it
    # is missing discloses nothing.
    outside = tmp_path.parent / "outside-the-roots"
    outside.mkdir(exist_ok=True)
    real = _behaviour(c, str(outside.resolve()), str(p))
    missing = _behaviour(c, str(tmp_path.parent / "definitely-not-here"), str(p))

    assert real.status_code == missing.status_code == 409
    # Byte-identical but for the path echoed back: an existing directory and a
    # missing one are refused the same way, so neither answer says which.
    assert "outside the directories" in real.json()["error"]
    assert "outside the directories" in missing.json()["error"]
    assert (
        real.json()["error"].split("is outside")[1]
        == (missing.json()["error"].split("is outside")[1])
    )


@pytest.mark.parametrize(
    "original",
    ["/etc/passwd", "C:/Windows/System32/config/SAM", "../../secret", "~/.ssh/id_rsa"],
)
def test_a_path_shaped_original_never_skips_the_roots_gate(tmp_path, original):
    """Anything not hub-id-shaped goes through the gate unconditionally."""
    p = _gguf(tmp_path)
    custom.add_root(str(tmp_path))
    r = _behaviour(_client(), original, str(p))
    assert r.status_code == 409
    assert (
        "outside the directories" in r.json()["error"]
        or "does not exist" in (r.json()["error"])
    )


def test_a_hub_id_is_not_sent_through_the_filesystem_gate(tmp_path, monkeypatch):
    """A hub id is not a path and must not be refused for failing to be one —
    otherwise the feature only works against local checkpoints."""
    from modelmri import behavdiff

    called = {}

    def spy(*a, **kw):
        called["gate"] = True
        raise AssertionError("a hub id reached the filesystem gate")

    monkeypatch.setattr(custom, "resolve_dir_under_roots", spy)
    assert behavdiff.is_hub_id("Qwen/Qwen3-0.6B")
    assert not called


def test_an_empty_path_is_not_the_servers_own_directory():
    """`Path("").resolve()` is the process's working directory, and that is
    not a corner case here — it exists, it is inside an allowed root, and the
    resolver therefore accepted it.

    MEASURED before the bound: `POST /api/gguf/load {"path": ""}` answered 409
    "<the repo this server was started in> is not a file" — naming a directory
    the request never mentioned, with no next step. Worse on the sibling:
    `POST /api/quantdiff/behaviour {"quantised": "<real .gguf>", "original":
    ""}` was not refused at all, and handed the cwd to the loader as "the
    full-precision checkpoint" for a run documented as expensive.
    """
    c = _client()
    r = c.post("/api/gguf/load", json={"path": ""})
    assert r.status_code == 422
    assert "at least 1 character" in str(r.json()["detail"])

    for body in (
        {"quantised": "", "original": ""},
        {"quantised": "model.gguf", "original": ""},
        {"quantised": "", "original": "model"},
    ):
        r = c.post("/api/quantdiff/behaviour", json=body)
        assert r.status_code == 422, body
        assert "at least 1 character" in str(r.json()["detail"]), body
