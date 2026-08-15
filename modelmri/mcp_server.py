"""Expose the measurements as MCP tools, so an agent can ask them directly.

MCP has become a client-side convention — LM Studio is a host with OAuth, Jan
and Open WebUI are hosts, llama.cpp's WebUI merged a browser-side client. Not
one tool in this category exposes model inspection *as* MCP tools.

JSON-RPC 2.0 over stdio, stdlib only, matching `modelmri-record`'s posture: a
client that wants to inspect a model should not have to install a web stack to
do it.

## One implementation per measurement

Every tool here is a THIN ADAPTER over an existing `ModelRuntime` method. Not
a reimplementation, not a simplified version — the same call the HTTP route
makes. Two implementations of "rank the attention heads" would drift, and the
drift would be invisible: an agent would get one number over MCP and a human
another over HTTP, and nothing would say which was stale.

## Refusals travel intact

An agent that receives a fabricated number will repeat it confidently, which
is worse than an agent that receives an error. So a refusal is returned AS a
tool error with the same authored sentence the browser gets — "generate
something first", "this model did not answer the rubric" — never as a zero, an
empty list, or a plausible-looking default.

## Read-only, on purpose

`load_model` and `export_mri` are deliberately absent. Loading has to
serialise on the runtime's lock and answer "busy loading X" to be honest about
it, and exporting writes a file on an agent's say-so. Both are real features;
neither is a first cut.

The honest current answer to "which agent wants to rank attention heads" is
"the maintainer, dogfooding" — so this ships small rather than speculating.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

from .errors import BadRequest, Refusal

# The MCP revision this speaks. PINNED, and an unknown one is refused rather
# than best-efforted: a client expecting a later shape would receive answers
# in an older one with nothing saying so.
PROTOCOL_VERSION = "2024-11-05"

SERVER_INFO = {"name": "modelmri", "version": ""}

# How long to wait on an attached server. An agent's tool call should fail
# with a sentence rather than hang a client's event loop.
ATTACH_TIMEOUT = 120


def _tools() -> list:
    """The tool list, as MCP describes them.

    Descriptions say what the number IS and what it is not, because the agent
    reading them is the thing that will paraphrase the result to a human.
    """
    return [
        {
            "name": "status",
            "description": (
                "What model is loaded on this machine, on what device and in "
                "what dtype, and whether a generation exists to measure. "
                "Returns loaded=false rather than an error when nothing is "
                "loaded."
            ),
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "list_models",
            "description": (
                "Models already on this machine — the local HuggingFace cache "
                "and any GGUF files found. Downloads nothing."
            ),
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "rank_attention_heads",
            "description": (
                "Ablate every attention head one at a time and rank them by "
                "how far the answer moves, in nats of KL. Carries the noise "
                "floor it was measured against: a score below that floor is "
                "arithmetic, not a finding. Requires a generation to already "
                "exist — this measures THAT generation, and refuses rather "
                "than inventing one."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "layer": {
                        "type": "integer",
                        "description": "restrict to one layer; omit for all",
                    },
                    "baseline": {
                        "type": "string",
                        "description": "zero, mean or resample — they disagree, "
                        "and which one produced a ranking is part of the answer",
                    },
                },
            },
        },
        {
            "name": "attribute_tokens",
            "description": (
                "How much each input token contributed to the answer, by "
                "masking it and measuring the KL. Carries its own noise floor "
                "for the same reason."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "position": {
                        "type": "integer",
                        "description": "which generated position to attribute; "
                        "omit for the last",
                    }
                },
            },
        },
        {
            "name": "logit_lens",
            "description": (
                "What the model's answer looked like at every layer, decoded "
                "through the unembedding. The plain lens is biased and this "
                "reports its held-out KL so the bias is visible rather than "
                "assumed away."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "top_k": {"type": "integer"},
                    "kind": {
                        "type": "string",
                        "description": "plain or tuned",
                    },
                },
            },
        },
        {
            "name": "inspect_mri",
            "description": (
                "Read a .mri someone shared: what is in it, what produced each "
                "number, and what it does NOT contain. Needs no model and "
                "downloads nothing."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    ]


class Server:
    """One MCP session over stdio.

    `attach` points at an already-running `modelmri serve`, so an agent does
    not load a second copy of a model that is already resident. Without it the
    tools run in this process against a runtime of their own.
    """

    def __init__(self, attach: str = "", runtime=None):
        self.attach = attach.rstrip("/") if attach else ""
        self._runtime = runtime
        self.initialised = False

    # ------------------------------------------------------------ plumbing

    def runtime(self):
        """The in-process runtime, built on first use.

        Imported lazily: `modelmri mcp --attach` never needs torch, and an
        agent that only calls `inspect_mri` should not pay for a deep-learning
        stack to read a file.
        """
        if self._runtime is None:
            from .runtime import ModelRuntime

            self._runtime = ModelRuntime()
        return self._runtime

    def _get(self, path: str) -> dict:
        url = f"{self.attach}{path}"
        try:
            with urllib.request.urlopen(url, timeout=ATTACH_TIMEOUT) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as err:
            # The server's own authored sentence, forwarded intact. An agent
            # that receives "generate something first" can act on it; one that
            # receives "HTTP 409" cannot.
            try:
                body = json.loads(err.read())
            except (ValueError, OSError):
                body = {}
            raise Refusal(
                str(body.get("error") or "the attached ModelMRI refused that.")
            ) from err
        except (urllib.error.URLError, OSError, ValueError) as err:
            raise BadRequest(
                f"could not reach a ModelMRI at {self.attach}. Start one with "
                f"`modelmri serve`, or drop --attach to run in this process."
            ) from err

    # --------------------------------------------------------------- tools

    def call(self, name: str, args: dict) -> dict:
        args = args or {}
        if name == "status":
            if self.attach:
                return self._get("/api/session")
            status = self.runtime().status()
            return status.to_dict() if hasattr(status, "to_dict") else dict(status)

        if name == "list_models":
            if self.attach:
                return self._get("/api/models/discovered")
            from . import discover

            return discover.discover()

        if name == "rank_attention_heads":
            layer = args.get("layer")
            baseline = str(args.get("baseline") or "zero")
            if self.attach:
                query = f"?baseline={baseline}" + (
                    f"&layer={int(layer)}" if layer is not None else ""
                )
                return self._get(f"/api/attention/ablate{query}")
            return self.runtime().ablate_heads(
                None if layer is None else int(layer), baseline
            )

        if name == "attribute_tokens":
            position = args.get("position")
            if self.attach:
                query = "" if position is None else f"?position={int(position)}"
                return self._get(f"/api/attention/attribute{query}")
            return self.runtime().attribute_tokens(
                None if position is None else int(position)
            )

        if name == "logit_lens":
            top_k = int(args.get("top_k") or 5)
            kind = str(args.get("kind") or "plain")
            if self.attach:
                return self._get(f"/api/lens?top_k={top_k}&kind={kind}")
            return self.runtime().logit_lens(top_k, kind)

        if name == "inspect_mri":
            # Always local: the file is on THIS machine, and an attached
            # server would be reading its own disk rather than the caller's.
            path = str(args.get("path") or "")
            if not path:
                raise BadRequest("inspect_mri needs the path to a .mri file.")
            from pathlib import Path

            from . import session as session_mod

            where = Path(path).expanduser()
            if not where.is_file():
                raise BadRequest("there is no file at that path.")
            parsed = session_mod.parse(where.read_bytes())
            # `meta`, not top-level fields: the setup that produced a file is
            # carried as one block so a reader gets all of it or knows it is
            # missing, rather than some of it defaulted.
            meta = parsed.meta if isinstance(parsed.meta, dict) else {}
            return {
                "model": meta.get("model"),
                "device": meta.get("device"),
                "dtype": meta.get("dtype"),
                "note": meta.get("note") or "",
                "scope": meta.get("scope") or "",
                "n_layers": parsed.n_layers,
                "n_heads": parsed.n_heads,
                "n_tokens": len(parsed.tokens),
                "prompt": parsed.prompt[:2000],
                "generation": parsed.generation[:2000],
                "has": {
                    "attention": bool(parsed.attention),
                    "lens": bool(parsed.lens),
                    "patch": parsed.has_patch(),
                    "ranking": parsed.has_ranking(),
                    "head_types": parsed.has_head_types(),
                    "ground": parsed.has_ground(),
                    "model_diff": parsed.has_model_diff(),
                    "vla": parsed.has_vla(),
                    "trace": parsed.has_trace(),
                    "graph": parsed.has_graph(),
                },
                "receipts": len(parsed.receipts),
            }

        raise BadRequest(
            f"no tool named {name!r}. This server offers "
            f"{', '.join(t['name'] for t in _tools())}."
        )

    # ------------------------------------------------------------ dispatch

    def handle(self, message: dict) -> dict | None:
        """One JSON-RPC message in, one response out (None for a notification)."""
        rpc_id = message.get("id")
        method = str(message.get("method") or "")
        params = message.get("params") or {}

        if method == "initialize":
            asked = str(params.get("protocolVersion") or "")
            if asked and asked != PROTOCOL_VERSION:
                # REFUSED, not best-efforted. A client expecting a later shape
                # would otherwise receive answers in an older one with nothing
                # saying so.
                return _error(
                    rpc_id,
                    -32602,
                    f"This server speaks MCP {PROTOCOL_VERSION} and the client "
                    f"asked for {asked}. Refusing rather than answering a "
                    f"protocol it does not implement.",
                )
            self.initialised = True
            from . import __version__

            return _ok(
                rpc_id,
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {**SERVER_INFO, "version": __version__},
                },
            )

        if method in ("notifications/initialized", "initialized"):
            return None

        if method == "tools/list":
            return _ok(rpc_id, {"tools": _tools()})

        if method == "tools/call":
            name = str(params.get("name") or "")
            try:
                result = self.call(name, params.get("arguments") or {})
            except (Refusal, BadRequest) as err:
                # A TOOL error, not a protocol error: the call was well formed
                # and the answer is "no". `isError` is what tells the agent it
                # did not receive a measurement.
                return _ok(
                    rpc_id,
                    {
                        "content": [{"type": "text", "text": str(err)}],
                        "isError": True,
                    },
                )
            return _ok(
                rpc_id,
                {
                    "content": [
                        {"type": "text", "text": json.dumps(result, default=str)}
                    ],
                    "isError": False,
                },
            )

        if method == "ping":
            return _ok(rpc_id, {})

        return _error(rpc_id, -32601, f"unknown method {method!r}")


def _ok(rpc_id, result) -> dict:
    return {"jsonrpc": "2.0", "id": rpc_id, "result": result}


def _error(rpc_id, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": code, "message": message}}


def serve(attach: str = "", stdin=None, stdout=None) -> int:
    """Read JSON-RPC from stdin, write responses to stdout. One line each."""
    source = stdin if stdin is not None else sys.stdin
    sink = stdout if stdout is not None else sys.stdout
    server = Server(attach=attach)

    for line in source:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except ValueError:
            # No id to answer against, so this is the one case that writes a
            # null-id error — the alternative is silence, which reads to a
            # client as a hung server.
            sink.write(json.dumps(_error(None, -32700, "not valid JSON")) + "\n")
            sink.flush()
            continue
        if not isinstance(message, dict):
            sink.write(json.dumps(_error(None, -32600, "not a JSON-RPC object")) + "\n")
            sink.flush()
            continue
        try:
            response = server.handle(message)
        except Exception as err:  # noqa: BLE001 - a tool must not kill the session
            response = _error(
                message.get("id"),
                -32603,
                f"{type(err).__name__} inside ModelMRI. The full error is in "
                f"the terminal running `modelmri mcp`.",
            )
        if response is not None:
            sink.write(json.dumps(response, default=str) + "\n")
            sink.flush()
    return 0
