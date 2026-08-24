"""The sidecar: one process, one policy, one small contract on loopback.

Stdlib `http.server` on purpose. This runs inside a venv whose whole reason to
exist is lerobot's dependency pins, and adding a web framework to that venv is
adding another thing that can conflict with them. The contract is four routes
and a version number; it does not need a framework.

## What it refuses

**A request with the wrong contract, or none.** Not a best effort. An action
chunk is a claim about what a robot would DO, and one served across a version
boundary is a different policy's answer wearing this one's name. See
`contract.py`.

**A request before the policy has loaded.** `/act` with no policy is a 409
that says so, rather than a zero-filled chunk that looks like a decision.

**Anything that is not loopback.** The bind address is not configurable, and
the handler re-checks the peer: a bound socket is a promise about where it
listens, not about who reached it.

## What it always says

Every response carries `policy_repo`, `revision`, `device`, `dtype` and
`normalisation`. The last is not decoration — a policy's action space is
normalised against ITS training statistics and a dataset's against its own,
and ROADMAP #50 names overlaying two curves in different units as the
plausible-wrong output to refuse. Refusing needs both sides to state units.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .adapter import ShapeMoved
from .contract import CONTRACT, HOST, READY_PREFIX, ContractError, check
from .inputs import InputError

# lerobot and torch versions, read ONCE before the ready line is printed.
# Module state because it is a property of this process's environment that
# cannot change while it runs, and because the alternative -- importing inside
# a request handler -- put a twenty-second import in the path of the first
# status call.
_VERSIONS: tuple[str, str] = ("", "")

# The biggest request body accepted. Frames arrive as base64 PNG; a handful of
# camera views is well under this, and anything past it is not a frame.
MAX_BODY = 64 * 1024 * 1024


class Policy:
    """The loaded policy, or the reason there isn't one.

    Kept as a small object with a lock rather than module globals: `/act` and
    `/hidden` both run a forward pass, `ThreadingHTTPServer` will happily run
    them concurrently, and two forward passes interleaving through one torch
    module is how you get an activation from one request answering another.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.model = None
        self.loaded = None
        self.repo = ""
        self.revision = ""
        self.device = ""
        self.dtype = ""
        self.normalisation: dict = {}

    @property
    def ready(self) -> bool:
        return self.model is not None

    def describe(self) -> dict:
        # Everything the checkpoint said about itself, when one is loaded --
        # cameras, state width, action width, whether it samples. A caller
        # cannot assemble a valid request without those, and asking it to
        # guess is asking it to send something the policy will silently
        # reinterpret.
        if self.loaded is not None:
            return self.loaded.describe()
        # From the cache filled before the ready line, never imported here.
        # Importing lerobot inside a request is a twenty-second stall on the
        # FIRST /status, which is the one the parent makes immediately after
        # starting the process -- measured as `modelmri policy start`
        # reporting the sidecar it had just started as "did not answer in
        # time. It is running and busy, or wedged."
        lerobot_version, torch_version = _VERSIONS
        return {
            "policy_repo": self.repo,
            "revision": self.revision,
            "device": self.device,
            "dtype": self.dtype,
            "normalisation": dict(self.normalisation),
            "lerobot_version": lerobot_version,
            "torch_version": torch_version,
        }

    def load(self, repo: str, *, device: str = "") -> dict:
        """Bring the policy up. Imports lerobot HERE, not at module scope.

        A module-scope import would mean the process cannot start at all when
        the venv is half-built, and the parent would see a dead child with a
        traceback about an import rather than a sentence about an install.

        Everything lerobot-shaped is behind `adapter`, so what this method
        knows is "load, then describe" and what changes when lerobot moves is
        one file.
        """
        if not repo:
            raise RuntimeError(
                "no policy was named, and there is no default worth guessing: "
                "a checkpoint is what decides which cameras, which state width "
                "and which action space every later answer is about."
            )
        try:
            from . import adapter
        except ImportError as err:  # pragma: no cover - depends on the venv
            raise RuntimeError(
                f"this sidecar's environment is incomplete "
                f"({type(err).__name__}). Rebuild it with `modelmri policy "
                f"install --force`."
            ) from None

        loaded = adapter.load(repo, device=device)
        self.loaded = loaded
        self.model = loaded.policy
        self.repo = loaded.repo
        self.device = loaded.device
        self.dtype = loaded.dtype
        self.revision = loaded.revision
        # Read off the checkpoint rather than assumed. A policy that does not
        # publish its normalisation gets an EMPTY dict and the caller refuses
        # the overlay -- which is the right answer, and better than a default
        # that silently claims identity scaling.
        self.normalisation = loaded.normalisation
        return self.describe()


def _act(policy: Policy, body: dict) -> dict:
    """Validate the request against THIS policy, then run one forward pass.

    Validation first, and against the loaded checkpoint rather than against a
    schema. A schema can say "state is a list of numbers"; only the checkpoint
    knows that this policy's state is six wide and that a seven-wide vector
    would shift every joint by one and return a plausible chunk for a robot
    that does not exist.

    The FORWARD PASS is under the lock, not the whole call.
    `ThreadingHTTPServer` runs requests concurrently and two passes
    interleaving through one torch module is how an activation from one
    request ends up answering another — but decoding PNGs does not touch the
    model, and holding the lock through it would serialise work that has no
    reason to be serial.

    The `loaded` reference is captured once, before the lock. That is what
    makes the split safe: a concurrent `/load` builds a NEW `Loaded` and
    rebinds the attribute, so this request keeps validating and acting against
    one coherent checkpoint rather than half of each.
    """
    from . import adapter, inputs

    loaded = policy.loaded
    frames = inputs.read_frames(body, expected=list(loaded.cameras))
    state = inputs.read_state(body, width=loaded.state_width)
    instruction = inputs.read_instruction(body)
    seed = inputs.read_seed(body)

    with policy.lock:
        return adapter.act(
            loaded,
            frames=frames,
            state=state,
            instruction=instruction,
            seed=seed,
        )


def make_handler(policy: Policy):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        # A client that opens a connection and then stops sending would
        # otherwise hold a handler thread for the life of the process. This is
        # the only ceiling in the server, and it is generous: a request body
        # is a handful of PNGs over loopback.
        timeout = 120

        def log_message(self, *a):
            """Silence the default stderr access log.

            The parent reads this process's stdout for the ready line, and a
            request log interleaved with it is noise in the one channel that
            has to stay parseable.
            """

        # ------------------------------------------------------------ helpers
        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps({**payload, "contract": CONTRACT}).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-ModelMRI-Contract", str(CONTRACT))
            self.end_headers()
            self.wfile.write(body)

        def _send_bytes(self, code: int, raw: bytes, kind: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("X-ModelMRI-Contract", str(CONTRACT))
            self.end_headers()
            self.wfile.write(raw)

        def _local_only(self) -> bool:
            """A bound socket says where it listens, not who reached it."""
            peer = (self.client_address or ("",))[0]
            return peer in ("127.0.0.1", "::1", "::ffff:127.0.0.1")

        def _read(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BODY:
                # Close the connection rather than answer and keep it. The
                # body is still sitting unread in the socket, so a keep-alive
                # answer would leave the next request parser reading the tail
                # of this one as a request line -- which is not a security
                # boundary here (loopback only) but does turn one malformed
                # request into a stream of nonsense ones.
                self.close_connection = True
                raise ValueError(
                    f"a body of {length} bytes is outside what this accepts"
                )
            return json.loads(self.rfile.read(length).decode("utf-8"))

        # ------------------------------------------------------------- routes
        def do_GET(self):
            if not self._local_only():
                self._send(403, {"error": "this sidecar answers loopback only"})
                return
            if self.path == "/status":
                # A GET carries no body, so the client's version rides in a
                # header. Checked rather than skipped: `/status` is what the
                # panel polls, and a client one version out learning "ready:
                # true" from a sidecar it cannot actually talk to is the drift
                # this contract exists to make visible, arriving through the
                # one route that was not looking.
                #
                # ABSENT is allowed here and only here. `status` is also how a
                # human with curl asks a sidecar what it is, and refusing that
                # would make the diagnostic route need the thing it diagnoses.
                stated = self.headers.get("X-ModelMRI-Contract")
                if stated is not None:
                    try:
                        check(
                            int(stated) if stated.strip().isdigit() else stated,
                            side="ModelMRI client",
                        )
                    except ContractError as err:
                        self._send(409, {"error": str(err)})
                        return
                with policy.lock:
                    # Under the lock: `ready` and `describe()` are two reads of
                    # state that `load` rebinds, and taking them separately let
                    # a status arrive claiming ready with the previous
                    # policy's description attached.
                    answer = {"ready": policy.ready, **policy.describe()}
                self._send(200, answer)
                return
            self._send(404, {"error": f"no route {self.path}"})

        def do_POST(self):
            if not self._local_only():
                self._send(403, {"error": "this sidecar answers loopback only"})
                return
            try:
                body = self._read()
            except Exception as err:
                self._send(400, {"error": f"unreadable request body: {err}"})
                return

            # BEFORE anything is done with the payload. A request whose
            # contract does not match is not a request this understands, and
            # reading fields out of it would be guessing at their meaning.
            try:
                check(body.get("contract"), side="ModelMRI client")
            except ContractError as err:
                self._send(409, {"error": str(err)})
                return

            if self.path == "/load":
                try:
                    with policy.lock:
                        described = policy.load(
                            str(body.get("policy_repo") or ""),
                            device=str(body.get("device") or ""),
                        )
                except Exception as err:
                    self._send(409, {"error": str(err)})
                    return
                self._send(200, {"ready": True, **described})
                return

            if self.path == "/act":
                if not policy.ready:
                    self._send(
                        409,
                        {
                            "error": (
                                "no policy is loaded in this sidecar, so there "
                                "is nothing here that could act. POST /load "
                                "first."
                            )
                        },
                    )
                    return
                try:
                    answer = _act(policy, body)
                except (InputError, ShapeMoved) as err:
                    # A refusal, and the sentence is the whole point of it.
                    # 409 rather than 400: the request is well-formed JSON, it
                    # is the CONTENT that does not match what this policy
                    # consumes, and a caller retrying the same bytes will get
                    # the same answer.
                    self._send(409, {"error": str(err), **policy.describe()})
                    return
                except Exception as err:  # pragma: no cover - depends on lerobot
                    self._send(
                        500,
                        {
                            "error": (
                                f"the forward pass failed inside the policy "
                                f"({type(err).__name__}). This is the policy's "
                                f"own error, not a refusal from this sidecar."
                            )
                        },
                    )
                    return
                self._send(200, answer)
                return

            if self.path == "/hidden":
                if not policy.ready:
                    self._send(
                        409,
                        {
                            "error": (
                                "no policy is loaded in this sidecar, so there "
                                "are no activations to read. POST /load first."
                            )
                        },
                    )
                    return
                self._send(
                    501,
                    {
                        "error": (
                            "/hidden is not implemented in this build. /act "
                            "runs the real forward pass; capturing activations "
                            "out of it is the next piece."
                        ),
                        **policy.describe(),
                    },
                )
                return

            self._send(404, {"error": f"no route {self.path}"})

    return Handler


def serve(port: int = 0) -> None:
    """Listen, and print the port so the parent knows we are up.

    The ready line is printed AFTER the socket is listening AND after the
    heavy imports, so the parent waiting on it learns "ready to answer" rather
    than "the process exists". Those are not the same thing and the difference
    was measurable: with the import deferred, the first `/status` -- the one
    the parent makes the instant it sees the port -- spent twenty seconds
    inside `import lerobot`, and `modelmri policy start` reported the sidecar
    it had just started as wedged.

    Warming here rather than at module scope, for the reason `Policy.load`
    records: a module-scope import means the process cannot start at all when
    the venv is half-built, and the parent then sees a dead child with a
    traceback about an import instead of a sentence about an install. Here it
    is inside a running process that can still answer.
    """
    global _VERSIONS

    policy = Policy()
    httpd = ThreadingHTTPServer((HOST, port), make_handler(policy))
    chosen = httpd.server_address[1]

    try:
        from .adapter import versions

        _VERSIONS = versions()
    except Exception:
        # Both empty, which every reader already treats as "not reported".
        # A sidecar that cannot import its own adapter still serves: `/load`
        # will refuse with the sentence that names the rebuild command, and a
        # refusal from a live process beats no process at all.
        _VERSIONS = ("", "")

    print(f"{READY_PREFIX}{chosen}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        # Ctrl-C is how somebody stops a sidecar they started in a terminal,
        # so it is the NORMAL exit rather than a fault. Swallowed here so the
        # `finally` below closes the socket and the process leaves with 0 --
        # a traceback would make a deliberate stop look like a crash to the
        # parent reading this child's output.
        pass
    finally:
        httpd.server_close()


def _port(raw: str) -> int:
    """A port number, or argparse's own refusal naming what was sent.

    `int(argv[i + 1])` accepted anything `int()` did, and the socket was left
    to complain. MEASURED, both `--port 70000` and `--port -1` reached
    `ThreadingHTTPServer` and escaped as an unhandled
    `OverflowError: bind(): port must be 0-65535` -- a traceback out of a
    socket call, which reads as the sidecar being broken rather than the
    argument being wrong. Refused here so it is answered as what it is.
    """
    try:
        port = int(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"must be a whole number between 0 and 65535, and this call sent "
            f"{raw!r}. 0 asks the OS for a free port, which is what "
            f"`modelmri policy start` sends."
        ) from None
    if not 0 <= port <= 65535:
        raise argparse.ArgumentTypeError(
            f"must be between 0 and 65535, and this call sent {port}. 0 asks "
            f"the OS for a free port."
        )
    return port


def main(argv: list[str] | None = None) -> int:
    # argparse, not the hand-rolled scan this had. That scan looked for
    # `--port` and IGNORED every other argument, so `--help` fell through to
    # `serve(0)`: MEASURED, `python -m modelmri_policy --help` printed
    # `MODELMRI_POLICY_PORT=53649` and then served until it was killed at a
    # 10s timeout (rc=124). Somebody reading a sidecar's usage got a listening
    # socket instead, and in a terminal there is nothing to say which -- the
    # ready line is the only output either way.
    #
    # The silent half was as bad: a typo like `--prot 5000` was dropped on the
    # floor and the sidecar came up on an OS-chosen port, and `--port` with
    # nothing after it raised IndexError from inside argv indexing rather than
    # saying which flag was short of a value.
    #
    # stdlib, so the "no framework in this venv" rule at the top of this file
    # is untouched -- that rule is about lerobot's pins, and argparse has none.
    parser = argparse.ArgumentParser(
        prog="modelmri-policy",
        description=(
            "Hold one robot policy in this process and answer ModelMRI's "
            f"contract (v{CONTRACT}) on {HOST}. Started for you by `modelmri "
            "policy start`; run it by hand to debug a sidecar that will not "
            "come up."
        ),
        epilog=(
            f"Prints `{READY_PREFIX}<port>` on stdout once it is listening AND "
            f"its imports are warm, then serves until interrupted. That line "
            f"is the handshake: a parent that has seen it can call /status."
        ),
    )
    parser.add_argument(
        "--port",
        type=_port,
        default=0,
        metavar="N",
        help=(
            "port to listen on. 0 (the default) asks the OS for a free one "
            "and reports it on the ready line."
        ),
    )
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    serve(args.port)
    return 0
