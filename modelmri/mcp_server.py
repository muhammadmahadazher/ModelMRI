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
import urllib.parse
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

# How much prompt and generation text `inspect_mri` returns. A cap is right
# here — the result lands in a model's context and a 200 kB prompt would crowd
# out the answer it was fetched to support — and what was cut is reported
# beside it, because a truncated prompt an agent cannot tell from a whole one
# is a reasoning error waiting to happen.
MAX_MCP_TEXT = 2000

# The baselines and lens kinds these tools accept, restated here rather than
# imported from `ablate.py` — that module imports torch at the top, and
# `modelmri mcp --attach` is meant to run without a deep-learning stack. They
# are the `enum` in the schemas below AND the check the arguments are measured
# against, so a value the schema forbids cannot reach a URL.
BASELINES = ("zero", "mean", "resample")
# `both` included because `runtime.logit_lens` accepts it and `GET /api/lens`
# forwards it. This file's rule is that `--attach` changes WHERE a measurement
# runs, not WHAT can be asked — so an enum narrower than the runtime's would
# make the same request answerable over HTTP and refused over MCP. The tool
# description had advertised only two for long enough that validating against
# it would have been a silent narrowing rather than a fix.
LENS_KINDS = ("plain", "tuned", "both")


def _tools() -> list:
    """The tool list, as MCP describes them.

    Descriptions say what the number IS and what it is not, because the agent
    reading them is the thing that will paraphrase the result to a human.
    """
    return [
        {
            # This description claimed two things the payload does not answer.
            #
            # "What model is loaded on this machine" — MEASURED against a real
            # `/api/session` holding a 3.3 GB SDXL pipeline and a VLA: this
            # tool answered `loaded: false, hf_id: null`, because `call` lifts
            # `model` out of the envelope and the image and VLA handles beside
            # it go nowhere. NARROWED to the text model rather than widened to
            # carry them, and the reason is in `call` below: in-process this
            # Server holds a bare ModelRuntime with no image or VLA handle at
            # all, so those keys could only ever be null — unknown — over
            # --attach and absent in process. One tool answering two shapes
            # depending on a flag the calling agent cannot see is the exact
            # thing `test_status_is_the_same_document_in_process_and_over_attach`
            # exists to stop. An agent could not act on them either way: there
            # is no image or VLA tool on this surface.
            #
            # "and whether a generation exists to measure" — NO such field has
            # ever existed. `ModelStatus` is exactly loaded, hf_id, device,
            # dtype, n_params, instruct, gguf, n_layers, and `runtime.py` says
            # why in as many words beside `n_layers`: these are properties of
            # the LOADED MODEL, not of a run. So the clause is dropped rather
            # than the field added. The tools that need a generation already
            # say so in their own descriptions and refuse by name when there
            # is not one, which is where that answer is actionable.
            "name": "status",
            "description": (
                "The text model loaded on this machine, on what device and "
                "in what dtype. Returns loaded=false rather than an error "
                "when nothing is loaded. Text only — an image pipeline or a "
                "robot policy can be resident on this machine while this "
                "reads loaded=false, and neither has a tool here."
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
                        "enum": list(BASELINES),
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
                        "enum": list(LENS_KINDS),
                        "description": "plain, tuned or both",
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


def _arg_int(args: dict, name: str, default: int | None) -> int | None:
    """One integer argument, or a refusal naming the parameter and what it held.

    Every one of these was `int(args.get(name) or default)`, which agreed with
    the schema it advertises on none of the interesting inputs. MEASURED, with
    the HTTP layer stubbed: `{"layer": true}` became `layer=1` and ranked layer
    one — `runtime.py` refuses a bool layer in as many words, and this laundered
    it to an `int` before that refusal could ever fire. `{"top_k": 4.9}` became
    `top_k=4`, with `isError:false` on a number nobody asked for. `{"top_k": 0}`
    became 5, because `or` cannot tell a stated zero from an absent one — while
    `layer` and `position`, in the same function, already used `is None` and got
    it right. And `{"top_k": "five"}` escaped as JSON-RPC -32603, blaming
    ModelMRI for the caller's argument.

    Absent and an explicit `null` both take the default: that is what lets a
    client omit a parameter it is not setting. Nothing else is guessed at.
    """
    raw = args.get(name)
    if raw is None:
        return default
    # Bools FIRST: `isinstance(True, int)` is True in Python, so a JSON `true`
    # would otherwise pass straight through as the number 1.
    if isinstance(raw, bool):
        raise BadRequest(
            f"`{name}` must be a whole number, and this call sent "
            f"{str(raw).lower()}. Pass the number you meant."
        )
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        # `int(4.9)` truncates in silence. A caller who wrote 4.9 wanted
        # something this cannot deliver, and 4 is not it.
        if not raw.is_integer():
            raise BadRequest(
                f"`{name}` must be a whole number, and this call sent {raw!r}."
            )
        return int(raw)
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError) as err:
        raise BadRequest(
            f"`{name}` must be a whole number, and this call sent {raw!r}."
        ) from err


def _arg_choice(args: dict, name: str, choices: tuple[str, ...], default: str) -> str:
    """One string argument, checked against the `enum` its schema advertises.

    The check has to happen HERE and not only downstream, because the value was
    reaching the measurement through a URL that could edit itself. MEASURED:
    `{"baseline": "zero#"}` sent `?baseline=zero` — the `#` opened a fragment,
    urllib never transmitted the rest, and the attached server both missed the
    `scope=all` behind it and saw a baseline of `zero` that `ablate.py` was
    happy to accept. So its own "unknown baseline" refusal could not fire over
    --attach, and the payload named `baseline: "zero"` as though that were the
    question. `urlencode` below stops the value editing the URL; this stops it
    being a value the tool never offered.
    """
    raw = args.get(name)
    if raw is None:
        return default
    if not isinstance(raw, str):
        raise BadRequest(
            f"`{name}` must be one of {', '.join(choices)}, and this call sent {raw!r}."
        )
    if raw not in choices:
        # An empty string lands here rather than on the default: `or` treated
        # `""` as absent, so a client that stated a kind it did not have got a
        # plain lens back labelled as one it had chosen.
        raise BadRequest(f"unknown {name} {raw!r} — use one of {', '.join(choices)}.")
    return raw


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
        # `args or {}` alone let a non-mapping through to `args.get`, and what
        # happened next depended on the tool: `status` never reads them and
        # answered normally, `inspect_mri` raised AttributeError. `handle`
        # refuses that frame before it arrives here; this keeps the same answer
        # for a caller reaching `call` directly, so the two entrances cannot
        # disagree about one input.
        if args is None:
            args = {}
        if not isinstance(args, dict):
            raise BadRequest(
                f"tool arguments must be an object of named parameters, and "
                f"this call sent a {type(args).__name__}."
            )
        if name == "status":
            if self.attach:
                # /api/session wraps the model status in an envelope carrying
                # the server's own name and version; in-process returns the
                # status itself. This tool's description promises `loaded` at
                # the top level, so the envelope comes off here -- otherwise
                # the same tool answers `loaded` or `model.loaded` depending
                # on a flag the calling agent cannot see.
                envelope = self._get("/api/session")
                model = envelope.get("model")
                if not isinstance(model, dict):
                    raise Refusal(
                        f"the ModelMRI at {self.attach} answered /api/session "
                        f"without a `model` object. Refusing rather than "
                        f"reshaping a response whose shape has moved."
                    )
                return model
            status = self.runtime().status()
            return status.to_dict() if hasattr(status, "to_dict") else dict(status)

        if name == "list_models":
            if self.attach:
                return self._get("/api/models/discovered")
            from . import discover

            return discover.discover()

        if name == "rank_attention_heads":
            layer = _arg_int(args, "layer", None)
            baseline = _arg_choice(args, "baseline", BASELINES, "zero")
            if self.attach:
                # This tool's schema says of `layer`: "omit for all". The HTTP
                # route defaults the other way -- `scope=layer`, meaning layer
                # 0 -- so omitting it swept the whole model in-process and
                # ranked layer 0 over --attach. Not just a different cost
                # (450 passes against 14 on Qwen3-0.6B): a ranking of one
                # layer, returned as though it were the model's.
                #
                # Built with `urlencode` rather than an f-string, and that is
                # the reason `scope=all` now survives the trip: an f-string put
                # the agent's own text into the URL unescaped, so a `#` in a
                # value opened a fragment and everything after it -- including
                # this `scope=all` -- was never sent, while an `&` in a value
                # appended a parameter of the agent's choosing that overrode
                # one of these.
                fields = {"baseline": baseline}
                if layer is None:
                    fields["scope"] = "all"
                else:
                    fields["layer"] = layer
                query = urllib.parse.urlencode(fields)
                return self._get(f"/api/attention/ablate?{query}")
            return self.runtime().ablate_heads(layer, baseline)

        if name == "attribute_tokens":
            position = _arg_int(args, "position", None)
            if self.attach:
                query = (
                    ""
                    if position is None
                    else "?" + urllib.parse.urlencode({"position": position})
                )
                return self._get(f"/api/attention/attribute{query}")
            return self.runtime().attribute_tokens(position)

        if name == "logit_lens":
            top_k = _arg_int(args, "top_k", 5)
            kind = _arg_choice(args, "kind", LENS_KINDS, "plain")
            if self.attach:
                query = urllib.parse.urlencode({"top_k": top_k, "kind": kind})
                return self._get(f"/api/lens?{query}")
            return self.runtime().logit_lens(top_k, kind)

        if name == "inspect_mri":
            # Always local: the file is on THIS machine, and an attached
            # server would be reading its own disk rather than the caller's.
            raw_path = args.get("path")
            # `str(... or "")` turned a number into a path-shaped string, so a
            # mistyped argument was answered "there is no file at that path" --
            # a refusal naming the wrong cause, and one the caller cannot act
            # on. The type is the cause, so the type is what is said.
            if raw_path is not None and not isinstance(raw_path, str):
                raise BadRequest(
                    f"`path` must be the path to a .mri file as a string, and "
                    f"this call sent {raw_path!r}."
                )
            path = raw_path or ""
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
                # CLIPPED, AND SAID SO. These were sliced at 2000 with no
                # marker and no length, so an agent reading this tool got a
                # truncated prompt indistinguishable from a whole one — and
                # then reasoned about a `.mri` on the strength of two thirds
                # of the text that produced it. `modelmri inspect --json`, the
                # sibling for a human, returns both whole.
                #
                # The cap stays: an MCP result goes into a model's context and
                # a 200 kB prompt would crowd out the answer. What changes is
                # that the reader is told, in the same shape `traces.py` uses
                # — the text, plus how many characters are not in it.
                "prompt": parsed.prompt[:MAX_MCP_TEXT],
                "prompt_clipped": max(0, len(parsed.prompt) - MAX_MCP_TEXT),
                "generation": parsed.generation[:MAX_MCP_TEXT],
                "generation_clipped": max(0, len(parsed.generation) - MAX_MCP_TEXT),
                "has": {
                    "attention": bool(parsed.attention),
                    "lens": bool(parsed.lens),
                    "patch": parsed.has_patch(),
                    "ranking": parsed.has_ranking(),
                    "head_types": parsed.has_head_types(),
                    "ground": parsed.has_ground(),
                    "model_diff": parsed.has_model_diff(),
                    "vla": parsed.has_vla(),
                    "image": parsed.has_image(),
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
        params = message.get("params")
        if params is None:
            params = {}

        # INVALID PARAMS, not INTERNAL ERROR. `params` was used as a mapping
        # the moment it arrived, so a JSON-RPC frame carrying an array reached
        # `.get` and raised AttributeError -- reported to the client as -32603
        # "AttributeError inside ModelMRI", which says a well formed request
        # broke the server when a malformed one did not reach it.
        if not isinstance(params, dict):
            return _error(
                rpc_id,
                -32602,
                f"`params` must be a JSON object, and this request sent a "
                f"{type(params).__name__}. Send the method's parameters as an "
                f"object, or omit `params` entirely.",
            )

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
            arguments = params.get("arguments")
            if arguments is None:
                arguments = {}
            # On the same rule as `params` above, and the reason this needed
            # its own check: whether a non-object `arguments` crashed depended
            # on the TOOL. `{"name": "inspect_mri", "arguments": [1, 2]}` gave
            # -32603, while the identical frame naming `status` answered 200 --
            # that branch never reads the arguments. One malformed frame, two
            # outcomes, neither of them a message about the frame.
            if not isinstance(arguments, dict):
                return _error(
                    rpc_id,
                    -32602,
                    f"`arguments` must be a JSON object of the tool's "
                    f"parameters, and this request sent a "
                    f"{type(arguments).__name__}. See the tool's inputSchema "
                    f"in tools/list for the names it takes.",
                )
            try:
                result = self.call(name, arguments)
            except (Refusal, BadRequest) as err:
                # A TOOL error, not a protocol error: the call was well formed
                # and the answer is "no". `isError` is what tells the agent it
                # did not receive a measurement.
                return _tool_error(rpc_id, str(err))
            except (ValueError, TypeError) as err:
                # The backstop UNDER the argument checking above, not a
                # substitute for it: anything these tools can be handed should
                # already have met an authored sentence. What this stops is the
                # -32603 an unchecked one used to produce -- "ValueError inside
                # ModelMRI. The full error is in the terminal", pointing an
                # agent at a terminal `serve` writes nothing to, for an
                # argument the agent itself chose. The type travels and the
                # exception's own text does not, the rule `serve` already keeps.
                return _tool_error(
                    rpc_id,
                    f"{name or 'that tool'} could not use the arguments it was "
                    f"given ({type(err).__name__}). Check them against the "
                    f"tool's inputSchema in tools/list.",
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


def _tool_error(rpc_id, text: str) -> dict:
    """A refused tool call: a successful JSON-RPC response carrying `isError`.

    Not a -32xxx. That code says the REQUEST was malformed, and an agent told
    its request was malformed will rephrase the request rather than read the
    sentence explaining that the measurement cannot be made yet.
    """
    return _ok(rpc_id, {"content": [{"type": "text", "text": text}], "isError": True})


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
        except Exception as err:
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
