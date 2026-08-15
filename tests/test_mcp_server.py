"""MCP over stdio, driven as a real JSON-RPC session.

The invariant worth the most here: a refusal must reach the agent AS a
refusal. An agent that receives a fabricated number repeats it confidently to
a human, which is worse than one that receives an error — so "generate
something first" travels intact rather than becoming an empty list, a zero, or
a plausible-looking default.
"""

from __future__ import annotations

import io
import json

import pytest

from modelmri import mcp_server
from modelmri.errors import BadRequest, Refusal


def _session(lines, attach=""):
    """Run a list of JSON-RPC messages through `serve` and read the replies."""
    stdin = io.StringIO("\n".join(json.dumps(m) for m in lines) + "\n")
    stdout = io.StringIO()
    mcp_server.serve(attach=attach, stdin=stdin, stdout=stdout)
    return [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]


def _init(version=mcp_server.PROTOCOL_VERSION):
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"protocolVersion": version},
    }


# ------------------------------------------------------------- initialize


def test_it_initialises_and_names_itself():
    out = _session([_init()])
    assert len(out) == 1
    result = out[0]["result"]
    assert result["protocolVersion"] == mcp_server.PROTOCOL_VERSION
    assert result["serverInfo"]["name"] == "modelmri"
    assert result["serverInfo"]["version"], "the version must be real"


def test_an_unknown_protocol_version_is_refused_not_best_efforted():
    """A client expecting a later shape would otherwise receive answers in an
    older one with nothing saying so."""
    out = _session([_init("1999-01-01")])
    error = out[0]["error"]
    assert error["code"] == -32602
    assert mcp_server.PROTOCOL_VERSION in error["message"]
    assert "1999-01-01" in error["message"]
    assert "Refusing rather than" in error["message"]


def test_a_client_that_states_no_version_is_allowed():
    """Absent is not wrong — some clients negotiate later."""
    out = _session([{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}])
    assert "result" in out[0]


def test_the_initialized_notification_gets_no_reply():
    """A notification has no id; replying to one is a protocol error."""
    out = _session([_init(), {"jsonrpc": "2.0", "method": "notifications/initialized"}])
    assert len(out) == 1


# ------------------------------------------------------------ tools/list


def test_every_tool_is_listed_with_a_schema():
    out = _session([_init(), {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}])
    tools = out[1]["result"]["tools"]
    names = {t["name"] for t in tools}
    assert names == {
        "status",
        "list_models",
        "rank_attention_heads",
        "attribute_tokens",
        "logit_lens",
        "inspect_mri",
    }
    for tool in tools:
        assert tool["description"].strip()
        assert tool["inputSchema"]["type"] == "object"


def test_the_write_tools_are_deliberately_absent():
    """`load_model` must serialise on the runtime lock to be honest about
    being busy; `export_mri` writes a file on an agent's say-so. Both are real
    features and neither is a first cut."""
    out = _session([_init(), {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}])
    names = {t["name"] for t in out[1]["result"]["tools"]}
    assert "load_model" not in names
    assert "export_mri" not in names


def test_the_descriptions_say_what_the_number_is_not():
    """The agent reading these is the thing that will paraphrase the result to
    a human."""
    out = _session([_init(), {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}])
    by_name = {t["name"]: t for t in out[1]["result"]["tools"]}
    assert "noise floor" in by_name["rank_attention_heads"]["description"]
    assert "refuses" in by_name["rank_attention_heads"]["description"]
    assert "biased" in by_name["logit_lens"]["description"]


# --------------------------------------------- refusals travel as refusals


class _RefusingRuntime:
    """A runtime with nothing loaded, refusing the way the real one does."""

    def status(self):
        return {"loaded": False}

    def ablate_heads(self, layer=None, baseline="zero"):
        raise Refusal("Generate something first, then inspect attention.")

    def attribute_tokens(self, position=None):
        raise Refusal("Generate something first, then inspect attention.")

    def logit_lens(self, top_k=5, kind="plain"):
        raise Refusal("Generate something first, then inspect attention.")


def _call(name, args=None, runtime=None):
    server = mcp_server.Server(runtime=runtime or _RefusingRuntime())
    return server.handle(
        {
            "jsonrpc": "2.0",
            "id": 9,
            "method": "tools/call",
            "params": {"name": name, "arguments": args or {}},
        }
    )


@pytest.mark.parametrize(
    "tool", ["rank_attention_heads", "attribute_tokens", "logit_lens"]
)
def test_a_refusal_reaches_the_agent_as_an_error_with_its_sentence(tool):
    """An agent that receives a fabricated number repeats it confidently."""
    out = _call(tool)
    result = out["result"]
    assert result["isError"] is True
    assert "Generate something first" in result["content"][0]["text"]


def test_a_refusal_is_not_a_protocol_error():
    """The call was well formed; the answer is no. A -32xxx here would tell
    the client its request was malformed."""
    out = _call("logit_lens")
    assert "error" not in out
    assert out["result"]["isError"] is True


def test_an_unknown_tool_lists_what_there_is():
    out = _call("make_it_faster")
    text = out["result"]["content"][0]["text"]
    assert out["result"]["isError"] is True
    assert "no tool named" in text and "logit_lens" in text


def test_a_successful_call_is_not_flagged_as_an_error():
    out = _call("status")
    assert out["result"]["isError"] is False
    assert json.loads(out["result"]["content"][0]["text"]) == {"loaded": False}


# ----------------------------------------------- one implementation, not two


def test_each_tool_calls_the_runtime_method_the_http_route_calls():
    """Two implementations of "rank the attention heads" would drift, and the
    drift would be invisible: an agent gets one number and a human another."""
    seen = {}

    class Recording(_RefusingRuntime):
        def ablate_heads(self, layer=None, baseline="zero"):
            seen["ablate"] = (layer, baseline)
            return {"ok": True}

        def attribute_tokens(self, position=None):
            seen["attribute"] = position
            return {"ok": True}

        def logit_lens(self, top_k=5, kind="plain"):
            seen["lens"] = (top_k, kind)
            return {"ok": True}

    runtime = Recording()
    _call("rank_attention_heads", {"layer": 3, "baseline": "mean"}, runtime)
    _call("attribute_tokens", {"position": 7}, runtime)
    _call("logit_lens", {"top_k": 9, "kind": "tuned"}, runtime)
    assert seen == {"ablate": (3, "mean"), "attribute": 7, "lens": (9, "tuned")}


def test_omitted_arguments_reach_the_runtime_as_none_not_zero():
    """`layer=0` is layer zero; omitting it means every layer. Coercing the
    absence to 0 would silently restrict the sweep."""
    seen = {}

    class Recording(_RefusingRuntime):
        def ablate_heads(self, layer=None, baseline="zero"):
            seen["layer"] = layer
            return {}

        def attribute_tokens(self, position=None):
            seen["position"] = position
            return {}

    runtime = Recording()
    _call("rank_attention_heads", {}, runtime)
    _call("attribute_tokens", {}, runtime)
    assert seen == {"layer": None, "position": None}


def test_layer_zero_survives():
    seen = {}

    class Recording(_RefusingRuntime):
        def ablate_heads(self, layer=None, baseline="zero"):
            seen["layer"] = layer
            return {}

    _call("rank_attention_heads", {"layer": 0}, Recording())
    assert seen == {"layer": 0}


# ------------------------------------------------------------ inspect_mri


def test_inspect_mri_reads_a_real_file(tmp_path):
    from modelmri import session

    blob = session.build(
        model_id="gpt2",
        device="cpu",
        dtype="float32",
        n_params=124_000_000,
        tokens=["a", "b"],
        prompt="hello",
        generation="world",
        attention={(0, 0): [[1.0, 0.0], [0.5, 0.5]]},
        n_layers=1,
        n_heads=1,
        n_prompt=1,
    )
    path = tmp_path / "s.mri"
    path.write_bytes(blob)
    out = _call("inspect_mri", {"path": str(path)})
    doc = json.loads(out["result"]["content"][0]["text"])
    assert doc["model"] == "gpt2"
    assert doc["has"]["attention"] is True
    assert doc["has"]["vla"] is False


def test_inspect_mri_needs_a_path():
    out = _call("inspect_mri", {})
    assert out["result"]["isError"] is True
    assert "needs the path" in out["result"]["content"][0]["text"]


def test_inspect_mri_on_a_missing_file_says_so_without_the_path(tmp_path):
    out = _call("inspect_mri", {"path": str(tmp_path / "nope.mri")})
    text = out["result"]["content"][0]["text"]
    assert "no file at that path" in text
    assert str(tmp_path) not in text


# ------------------------------------------------------------- the attach


def test_attach_forwards_the_servers_own_refusal():
    """ "generate something first" is actionable; "HTTP 409" is not."""
    import urllib.error

    server = mcp_server.Server(attach="http://127.0.0.1:1")

    def boom(url, timeout=None):
        raise urllib.error.HTTPError(
            url,
            409,
            "Conflict",
            {},
            io.BytesIO(json.dumps({"error": "Generate something first."}).encode()),
        )

    server._get = lambda path: mcp_server.Server._get(server, path)
    import urllib.request

    real = urllib.request.urlopen
    urllib.request.urlopen = boom
    try:
        with pytest.raises(Refusal, match="Generate something first"):
            server.call("logit_lens", {})
    finally:
        urllib.request.urlopen = real


def test_attach_to_nothing_says_how_to_fix_it():
    server = mcp_server.Server(attach="http://127.0.0.1:1")
    with pytest.raises(BadRequest) as caught:
        server.call("status", {})
    message = str(caught.value)
    assert "modelmri serve" in message
    assert "drop --attach" in message


# ------------------------------------------------------------- robustness


def test_a_malformed_line_answers_rather_than_hanging():
    """Silence reads to a client as a hung server."""
    stdin = io.StringIO("{not json\n")
    stdout = io.StringIO()
    mcp_server.serve(stdin=stdin, stdout=stdout)
    out = json.loads(stdout.getvalue().strip())
    assert out["error"]["code"] == -32700


def test_a_non_object_message_is_answered():
    stdin = io.StringIO("[1, 2, 3]\n")
    stdout = io.StringIO()
    mcp_server.serve(stdin=stdin, stdout=stdout)
    assert json.loads(stdout.getvalue().strip())["error"]["code"] == -32600


def test_blank_lines_are_skipped():
    out = _session_with_raw("\n\n" + json.dumps(_init()) + "\n")
    assert len(out) == 1


def _session_with_raw(text):
    stdout = io.StringIO()
    mcp_server.serve(stdin=io.StringIO(text), stdout=stdout)
    return [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]


def test_an_unknown_method_is_a_protocol_error():
    out = _session([_init(), {"jsonrpc": "2.0", "id": 2, "method": "sing"}])
    assert out[1]["error"]["code"] == -32601


def test_ping_is_answered():
    out = _session([_init(), {"jsonrpc": "2.0", "id": 2, "method": "ping"}])
    assert out[1]["result"] == {}


def test_a_tool_that_explodes_does_not_kill_the_session():
    """One bad tool call must not take down an agent's whole connection."""

    class Exploding(_RefusingRuntime):
        def logit_lens(self, top_k=5, kind="plain"):
            raise RuntimeError("something deep")

    server = mcp_server.Server(runtime=Exploding())
    stdin = io.StringIO(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "logit_lens", "arguments": {}},
            }
        )
        + "\n"
    )
    stdout = io.StringIO()
    # Drive `serve`'s loop with this server's behaviour by monkeypatching the
    # class it builds.
    real = mcp_server.Server
    mcp_server.Server = lambda attach="": server
    try:
        mcp_server.serve(stdin=stdin, stdout=stdout)
    finally:
        mcp_server.Server = real
    out = json.loads(stdout.getvalue().strip())
    assert out["error"]["code"] == -32603
    assert "RuntimeError" in out["error"]["message"]
    # The exception's own text must not travel — only its type.
    assert "something deep" not in out["error"]["message"]


# ------------------------------------------- the two answers must be one


def test_status_is_the_same_document_in_process_and_over_attach():
    """The whole promise of --attach is that it changes WHERE, not WHAT.

    /api/session wraps the model status in an envelope carrying the server's
    own name and version. In-process returns the status itself. So `loaded`
    -- the key this tool's description names -- sat at the top level one way
    and under `model` the other, and nothing in the response said which.

    Real route, real runtime, no model loaded: the state every agent meets on
    its first call.
    """
    from fastapi.testclient import TestClient

    from modelmri.server import create_app

    client = TestClient(create_app())

    attached = mcp_server.Server(attach="http://testserver")
    attached._get = lambda path: client.get(path).json()

    assert attached.call("status", {}) == mcp_server.Server().call("status", {})


def test_an_attached_server_whose_session_shape_moved_is_refused():
    """Reshaping an envelope that is not there would invent a status."""
    server = mcp_server.Server(attach="http://testserver")
    server._get = lambda path: {"app": "modelmri", "version": "99.0.0"}

    with pytest.raises(Refusal) as caught:
        server.call("status", {})
    assert "/api/session" in str(caught.value)


def test_omitting_the_layer_ranks_every_layer_over_attach_too():
    """`layer` is documented "omit for all" and the route defaults to layer 0.

    Omitting it therefore swept the whole model in-process and ranked layer 0
    over --attach -- 450 forward passes against 14 on Qwen3-0.6B, and the
    cheap one came back labelled as if it were the model's.
    """
    asked: list[str] = []
    server = mcp_server.Server(attach="http://testserver")
    server._get = lambda path: asked.append(path) or {}

    server.call("rank_attention_heads", {})
    assert "scope=all" in asked[-1]

    # Naming a layer must NOT widen to the whole model.
    server.call("rank_attention_heads", {"layer": 3})
    assert "layer=3" in asked[-1]
    assert "scope=all" not in asked[-1]
