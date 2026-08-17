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

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .contract import CONTRACT, HOST, READY_PREFIX, ContractError, check

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
        self.repo = ""
        self.revision = ""
        self.device = ""
        self.dtype = ""
        self.normalisation: dict = {}

    @property
    def ready(self) -> bool:
        return self.model is not None

    def describe(self) -> dict:
        return {
            "policy_repo": self.repo,
            "revision": self.revision,
            "device": self.device,
            "dtype": self.dtype,
            "normalisation": dict(self.normalisation),
        }

    def load(self, repo: str, *, device: str = "") -> dict:
        """Bring the policy up. Imports lerobot HERE, not at module scope.

        A module-scope import would mean the process cannot start at all when
        the venv is half-built, and the parent would see a dead child with a
        traceback about an import rather than a sentence about an install.
        """
        try:
            from lerobot.common.policies.factory import make_policy  # type: ignore
        except Exception as err:  # pragma: no cover - depends on the venv
            raise RuntimeError(
                f"this sidecar's environment cannot import lerobot "
                f"({type(err).__name__}: {err}). Rebuild it with "
                f"`modelmri policy install --force`."
            ) from None

        import torch  # type: ignore

        want = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = make_policy(repo)
        model.eval()
        model.to(want)

        self.model = model
        self.repo = repo
        self.device = str(want)
        self.dtype = str(next(model.parameters()).dtype).removeprefix("torch.")
        self.revision = _revision_of(repo)
        # Read off the policy rather than assumed. A policy that does not
        # publish its normalisation gets an EMPTY dict and the caller refuses
        # the overlay -- which is the right answer, and better than a default
        # that silently claims identity scaling.
        self.normalisation = _normalisation_of(model)
        return self.describe()


def _revision_of(repo: str) -> str:
    """The exact snapshot this policy came from, or "" when unknowable.

    Empty rather than "unknown" or "main": a caller comparing two runs needs
    to tell "these are the same weights" from "nobody recorded which weights",
    and a placeholder string collapses those into one.
    """
    try:
        from huggingface_hub import HfApi  # type: ignore

        return str(HfApi().model_info(repo).sha or "")
    except Exception:
        return ""


def _normalisation_of(model: object) -> dict:
    """The action-space statistics the policy normalises against.

    Empty when the policy does not publish them, and the caller must treat
    empty as "do not overlay", never as "identity". See ROADMAP #50.
    """
    stats = getattr(model, "normalize_targets", None) or getattr(
        model, "output_normalization", None
    )
    if stats is None:
        return {}
    out: dict = {}
    for key in ("mean", "std", "min", "max"):
        value = getattr(stats, key, None)
        if value is None:
            continue
        try:
            out[key] = [float(x) for x in value.flatten().tolist()]
        except Exception:
            continue
    return out


def make_handler(policy: Policy):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

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
                self._send(200, {"ready": policy.ready, **policy.describe()})
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

            if self.path in ("/act", "/hidden"):
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
                self._send(
                    501,
                    {
                        "error": (
                            f"{self.path} is not implemented in this build. The "
                            f"contract and the process boundary are in place; "
                            f"the forward pass is the next piece."
                        ),
                        **policy.describe(),
                    },
                )
                return

            self._send(404, {"error": f"no route {self.path}"})

    return Handler


def serve(port: int = 0) -> None:
    """Listen, and print the port so the parent knows we are up.

    The ready line is printed AFTER the socket is listening, so the parent
    waiting on it learns "ready to answer" rather than "the process exists".
    """
    policy = Policy()
    httpd = ThreadingHTTPServer((HOST, port), make_handler(policy))
    chosen = httpd.server_address[1]
    print(f"{READY_PREFIX}{chosen}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    port = 0
    if "--port" in argv:
        port = int(argv[argv.index("--port") + 1])
    serve(port)
    return 0
