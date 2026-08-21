"""Tests for modelmri-record. No network, no server, no model."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import modelmri_record as rec
from modelmri_record.redact import default_redactor, make_redactor


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    """Never touch the network; land traces in a temp dir.

    MODELMRI_TRACE_DIR is set explicitly because the undelivered-trace
    location is no longer the working directory: this package is imported by
    the user's agent, so the CWD is normally their git repo, and a trace
    holds full prompts and tool output. Pointing it at tmp_path keeps every
    assertion below unchanged while testing the supported override.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MODELMRI_TRACE_DIR", str(tmp_path / "modelmri-traces"))
    monkeypatch.setattr(
        rec.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(OSError())
    )
    return tmp_path


def test_a_trace_never_lands_in_the_working_directory(monkeypatch, tmp_path):
    """The default, with no override at all.

    Without this the whole suite could pass while the shipped default wrote
    into whatever repo the recorder was imported from — which is exactly the
    state this file was in before the location moved.
    """
    monkeypatch.delenv("MODELMRI_TRACE_DIR", raising=False)
    monkeypatch.setenv("MODELMRI_HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)

    with rec.trace("cwd-check", endpoint="http://127.0.0.1:1/nope"):
        rec.step("llm_call", name="a", duration_ms=1)

    assert not (tmp_path / "modelmri-traces").exists(), "wrote into the CWD"
    assert not list(tmp_path.glob("*.json"))
    parked = list((tmp_path / "home" / "data" / "undelivered").glob("*.json"))
    assert len(parked) == 1, "the trace was not kept anywhere"


def written(tmp_path) -> dict:
    files = list((tmp_path / "modelmri-traces").glob("*.json"))
    assert len(files) == 1, f"expected one trace file, got {files}"
    return json.loads(files[0].read_text())


# ---------------------------------------------------------------- structure


def test_nesting_records_parentage(offline):
    with rec.trace("run"):
        rec.step("llm_call", name="plan", duration_ms=10)
        with rec.step("subagent", name="child"):
            rec.step("tool_call", name="pytest", duration_ms=5)
    doc = written(offline)
    kinds = [s["kind"] for s in doc["steps"]]
    assert kinds == ["llm_call", "subagent", "tool_call"]
    assert doc["steps"][2]["parent_id"] == doc["steps"][1]["id"]
    assert doc["steps"][0]["parent_id"] is None


def test_an_exception_is_recorded_and_still_raised(offline):
    # The raising body is a function so that nothing follows a `raise` inside
    # a `with` — the analysis does not model pytest.raises swallowing it, and
    # reads the assertions below as dead code. Same test, no dead-code warning.
    def boom():
        with rec.trace("boom"):
            rec.step("tool_call", name="thing")
            raise ValueError("kaboom")

    with pytest.raises(ValueError):
        boom()
    doc = written(offline)
    assert doc["steps"][-1]["kind"] == "error"
    assert "kaboom" in doc["steps"][-1]["output"]


def test_step_outside_a_trace_is_a_noop():
    """Instrumentation left in library code must not explode for callers who
    never opted into tracing."""
    assert not rec.step("llm_call", name="orphan")


def test_with_step_outside_a_trace_does_not_raise():
    """0.1.0 returned a bare None here, so the documented `with` form was a
    TypeError for anyone not inside a trace."""
    with rec.step("subagent", name="orphan"):
        pass


def test_with_step_survives_a_worker_thread(offline):
    """contextvars do not cross thread boundaries, so a fan-out agent lands
    outside the trace on every worker -- which 0.1.0 turned into a crash."""
    from concurrent.futures import ThreadPoolExecutor

    def work():
        with rec.step("subagent", name="in-thread"):
            rec.step("tool_call", name="grep")
        return "ok"

    with rec.trace("threaded"):
        with ThreadPoolExecutor(max_workers=2) as pool:
            assert [f.result() for f in [pool.submit(work), pool.submit(work)]] == [
                "ok",
                "ok",
            ]


def test_concurrent_tasks_do_not_steal_each_others_parentage(offline):
    """A shared parent stack made task B's tool call a child of task A's open
    subagent, purely because A was inside a `with` at that moment. Parallel
    agents are the normal case, so the tree was wrong whenever it mattered."""
    import asyncio

    async def agent(tag: str) -> None:
        with rec.step("subagent", name=tag):
            await asyncio.sleep(0.01)
            rec.step("tool_call", name=f"{tag}-tool")

    async def both() -> None:
        await asyncio.gather(agent("A"), agent("B"))

    with rec.trace("parallel"):
        asyncio.run(both())

    steps = {s["id"]: s for s in written(offline)["steps"]}
    by_name = {s["name"]: s for s in steps.values()}
    for tag in ("A", "B"):
        sub, tool = by_name[tag], by_name[f"{tag}-tool"]
        assert sub["parent_id"] is None, f"{tag} subagent should be top level"
        assert tool["parent_id"] == sub["id"], f"{tag}-tool hung off the wrong parent"


def test_quick_successive_runs_do_not_overwrite_each_other(offline):
    """The offline filename was second-resolution, so three runs of the same
    agent inside one second left one file on disk."""
    for i in range(3):
        with rec.trace("agent-run"):
            rec.step("llm_call", name=f"call-{i}")
    files = list((offline / "modelmri-traces").glob("*.json"))
    assert len(files) == 3, f"expected 3 traces, found {len(files)}"


# ---------------------------------------------------------------- redaction


@pytest.mark.parametrize(
    "secret",
    [
        "sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAA",
        "sk-proj-BBBBBBBBBBBBBBBBBBBBBBBB",
        "hf_CCCCCCCCCCCCCCCCCCCCCCCC",
        "pypi-AgEIcHlwaS5vcmcCJDdkY2JkMmU5",
        "ghp_DDDDDDDDDDDDDDDDDDDDDDDDDD",
        "github_pat_EEEEEEEEEEEEEEEEEEEEEE",
        "xoxb-123456789012-abcdefghijkl",
        "AIzaSyDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD",
        "AKIAIOSFODNN7EXAMPLE",
        "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
    ],
)
def test_known_credential_shapes_never_reach_the_document(offline, secret):
    with rec.trace("leaky"):
        rec.step("llm_call", name="m", input=f"here is the key {secret} use it")
    raw = json.dumps(written(offline))
    assert secret not in raw, f"{secret[:12]}... survived redaction"
    assert "[redacted:" in raw


def test_a_private_key_block_is_removed_whole(offline):
    pem = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEowIBAAKCAQEAxyz\nmore\n"
        "-----END RSA PRIVATE KEY-----"
    )
    with rec.trace("pem"):
        rec.step("tool_call", name="cat", output=pem)
    raw = json.dumps(written(offline))
    assert "MIIEowIBAAKCAQEAxyz" not in raw
    assert "[redacted:private-key]" in raw


def test_redaction_keeps_the_trace_useful(offline):
    """A redactor that eats everything gets switched off, which protects
    nobody. Ordinary text, hashes and uuids must survive."""
    keep = (
        "commit 9f2a1c4de8b7 fixed the retry loop; "
        "run id 3f8a2b91-4c7d-4e11-9a3b-2f6c8d1e0a55 took 1200ms"
    )
    assert default_redactor(keep) == keep


def test_structural_fields_are_never_redacted(offline):
    with rec.trace("keep-structure"):
        rec.step(
            "llm_call",
            name="sk-model-name-lookalike",
            duration_ms=42,
            tokens_in=100,
            tokens_out=7,
        )
    s = written(offline)["steps"][0]
    assert s["duration_ms"] == 42 and s["tokens_in"] == 100 and s["tokens_out"] == 7
    assert s["name"] == "sk-model-name-lookalike"  # names are not payloads


def test_redaction_can_be_switched_off_deliberately(offline):
    with rec.trace("verbatim", redact=False):
        rec.step("llm_call", input="sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAA")
    assert "sk-ant-api03" in json.dumps(written(offline))


def test_custom_patterns_compose_with_the_defaults(offline):
    red = make_redactor([r"ACME-[0-9]{6}"])
    with rec.trace("custom", redact=red):
        rec.step("tool_call", input="ACME-123456 and hf_ZZZZZZZZZZZZZZZZZZZZZZZZ")
    raw = json.dumps(written(offline))
    assert "ACME-123456" not in raw
    assert "hf_ZZZZ" not in raw


# ---------------------------------------------------------------- delivery


def test_delivery_is_idempotent(offline):
    """The atexit flush and the normal exit path can both fire. Writing the
    trace twice would double every run in the viewer."""
    with rec.trace("once"):
        rec.step("llm_call", name="a")
    rec._flush_live()  # simulate the shutdown hook running afterwards
    assert len(list((offline / "modelmri-traces").glob("*.json"))) == 1


def test_a_trace_left_open_at_shutdown_is_still_flushed(offline):
    t = rec._Trace("orphaned", rec.DEFAULT_ENDPOINT, None)
    rec._live.append(t)
    token = rec._current.set(t)
    rec.step("llm_call", name="mid-flight")
    rec._current.reset(token)
    rec._flush_live()
    assert written(offline)["name"] == "orphaned"


def test_a_truncated_private_key_is_still_redacted(offline):
    """The recorder truncates payloads to 2-4k BEFORE redaction runs, so a
    real key arrives at the redactor with its -----END sentinel cut off. 0.1.0
    wrote the base64 body to disk in the clear."""
    body = "MIIJKQIBAAKCAgEA000SECRETKEYBYTES" + "A" * 3400
    pem = "-----BEGIN RSA PRIVATE KEY-----\n" + body + "\n-----END RSA PRIVATE KEY-----"
    with rec.trace("deploy"):
        try:
            with rec.step("tool_call", name="ssh-add"):
                raise ValueError("could not parse key:\n" + pem)
        except ValueError:
            pass
    raw = json.dumps(written(offline))
    assert "-----END RSA PRIVATE KEY-----" not in raw, "premise: tail was truncated"
    assert "MIIJKQIBAAKCAgEA000SECRETKEYBYTES" not in raw
    assert "[redacted:private-key]" in raw


def test_a_cyclic_payload_does_not_crash_the_host(offline):
    """Agent state graphs are cyclic by construction. json.dumps raises
    ValueError on those, and 0.1.0 let it escape into the caller."""
    node = {"name": "root"}
    node["parent"] = node
    with rec.trace("cycle"):
        rec.step("tool_call", name="dump", input=node)
    assert written(offline)["steps"][0]["input"]


def test_non_string_dict_keys_do_not_crash_the_host(offline):
    with rec.trace("keys"):
        rec.step("tool_call", name="grid", input={("r", 1): "v"})
    assert written(offline)["steps"][0]["input"]


def test_recording_never_raises_even_when_the_disk_refuses(offline, monkeypatch):
    """Recording must never take down the host app. That is the whole
    contract; if it can throw, nobody will leave it switched on."""
    monkeypatch.setattr(
        rec.Path, "mkdir", lambda *a, **k: (_ for _ in ()).throw(OSError("read-only"))
    )
    with rec.trace("doomed"):
        rec.step("llm_call", name="x")
    # reaching here without an exception IS the assertion


def test_import_costs_nothing_heavy():
    """The selling point is that instrumenting an agent does not cost a
    deep-learning install, so this has to be measured in a FRESH interpreter.
    Checking sys.modules in-process only tells you what the rest of the test
    session imported -- which is how this test passed alone and failed in the
    full run, proving nothing either way."""
    import subprocess

    code = (
        "import sys, modelmri_record;"
        "heavy={'torch','transformers','numpy','fastapi','pydantic','anthropic'};"
        "print(sorted(heavy & set(sys.modules)))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]", f"pulled heavy deps: {out.stdout.strip()}"


# ---------------------------------------------------------------------------
# deliver_otlp: opt-in, and it must never cost you the trace
# ---------------------------------------------------------------------------


def test_otlp_delivery_is_off_by_default():
    """This package is imported into other people's agents. A recorder that
    starts talking to the network because it can is one nobody should
    install."""
    import inspect

    sig = inspect.signature(rec.trace)
    assert sig.parameters["deliver_otlp"].default is None


def test_a_missing_modelmri_is_said_not_swallowed(monkeypatch, capsys):
    """The OTLP mapping lives in `modelmri` in one table, so emit and ingest
    cannot drift; copying it here would be the second copy that drifts. This
    package therefore keeps `dependencies = []` and the export is optional —
    but "you asked for an export and got nothing, silently" is the worst
    possible answer, so the absence is named."""
    import builtins

    real = builtins.__import__

    def no_modelmri(name, *a, **kw):
        if name == "modelmri" or name.startswith("modelmri."):
            raise ImportError("no modelmri")
        return real(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_modelmri)

    t = rec._Trace("x", "http://127.0.0.1:1/none", None, None, "http://127.0.0.1:1")
    rec._deliver_otlp(t, {"id": "t", "name": "x", "steps": []})

    said = capsys.readouterr()
    assert "not importable" in (said.err + said.out)
    assert "pip install modelmri" in (said.err + said.out)


def test_a_failing_export_does_not_raise(monkeypatch):
    """An export is a convenience. A collector being down must not cost
    somebody the trace itself, and must never crash the host app."""
    t = rec._Trace("x", "http://127.0.0.1:1/none", None, None, "http://127.0.0.1:1")
    # Unreachable on purpose; the point is that this returns rather than raises.
    rec._deliver_otlp(t, {"id": "t", "name": "x", "steps": []})


def test_no_endpoint_means_no_attempt(monkeypatch):
    """Off is off: nothing is imported and nothing is sent."""
    called = []
    monkeypatch.setattr(rec, "_complain", lambda m: called.append(m))
    t = rec._Trace("x", "http://127.0.0.1:1/none", None, None, None)
    rec._deliver_otlp(t, {"id": "t", "name": "x", "steps": []})
    assert called == []


def test_the_exported_document_is_the_redacted_one():
    """Exporting raw payloads to a third-party collector while the local copy
    is scrubbed would be the redactor working exactly backwards. `_deliver`
    redacts once and hands the SAME document to both paths."""
    import inspect

    src = inspect.getsource(rec._deliver)
    # The redaction happens before either delivery, and both are passed `doc`.
    assert src.index("redact_document") < src.index("_deliver_otlp(t, doc)")


def test_an_auto_instrumented_payload_says_how_much_was_cut():
    """The recorder's own cap is REPORTED, in the marker `traces._unclip` parses.

    Cut silently at 4,000 — well under the store's 20,000, so `_clip` never
    fired — a 50,000-character prompt reached the timeline looking whole,
    `truncated_in` read 0, and the only sentence the panel could draw named
    20,000: a cap that had not applied to that payload.
    """
    messages = [{"role": "user", "content": "x" * 50_000}]
    whole = json.dumps(messages, default=str)
    cut = len(whole) - rec.MAX_PREVIEW_CHARS

    preview = rec._msgs_preview({"messages": messages})

    assert preview == whole[: rec.MAX_PREVIEW_CHARS] + f"\u2026 [+{cut}]"
    assert preview.endswith(f"[+{cut}]")


def test_a_payload_under_the_cap_is_left_exactly_alone():
    """So the marker cannot become something every payload carries."""
    messages = [{"role": "user", "content": "hello"}]
    whole = json.dumps(messages, default=str)

    assert rec._msgs_preview({"messages": messages}) == whole
    assert "\u2026 [+" not in rec._msgs_preview({"messages": messages})


def test_the_recorders_marker_is_the_one_the_store_parses():
    """One shape for a cut payload, wherever the cut happened.

    `traces._CLIPPED` is what turns the marker back into `truncated_in` /
    `truncated_out`. A recorder that marked its cuts in its own dialect would
    leave the count at 0 and the panel silent — which is the bug this fixes,
    not a second version of it.
    """
    import re

    clipped = re.compile(r"\u2026 \[\+(\d+)\]$")
    marked = rec._cut("y" * 5_000, rec.MAX_PREVIEW_CHARS)

    found = clipped.search(marked)
    assert found is not None, "the store's regex has to match this"
    assert int(found.group(1)) == 5_000 - rec.MAX_PREVIEW_CHARS
