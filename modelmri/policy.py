"""Ask a robot policy what it would DO, in a process that is not this one.

`vla.py` holds the vision tower and can say where a policy LOOKED. It cannot
say what the policy would do, because the action expert needs lerobot, and
lerobot pins torch and numpy hard enough that installing it into ModelMRI's
environment breaks ModelMRI. That is not a hypothetical: it is why
physical-AI-interpretability is stale on a modified fork.

So the action half lives behind a process boundary, in its own venv, speaking
one small HTTP contract on loopback. This module is the client.

## Three rules, and all three are refusals

**No sidecar is a refusal that names the command.** Not a crash, not an empty
action, not a zero-filled chunk. `vla.py` already refuses the action expert by
name; this replaces that refusal with one that says what to run.

**A contract mismatch is a refusal, never a best effort.** ROADMAP #49 is
explicit: "contract drift silently serving actions from a stale policy is the
worst failure available here." An action chunk is a claim about what a robot
would do. A stale one is a different policy's answer wearing this one's name,
and no panel downstream can tell. See `modelmri_policy.contract`, where the
version is declared independently on each side precisely so drift is visible.

**Two processes holding weights is a capacity refusal, not a hope.** The
sidecar loads a policy while this process may already hold a model. On the
8 GB card this project targets, that is the common case rather than the
corner: SmolVLA's tower plus a 1.7B LLM does not fit, and finding out by
OOM-ing halfway through a sweep is the outcome `capacity.guard` exists to
prevent. The check runs BEFORE the subprocess starts, because a refusal after
a two-minute venv build is a refusal that wasted two minutes.
"""

from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from .errors import Refusal

# What THIS side speaks. Declared here and again in
# `modelmri_policy.contract.CONTRACT`; the two are compared on every exchange.
# One shared constant would move both halves together and make drift
# undetectable, which is the opposite of the point.
CONTRACT = 1

# How long to wait for the sidecar to say it is serving. A policy load is
# weights off disk, so this is generous -- but bounded, because a child that
# never prints its ready line is a hang, and a hang with no message is the
# worst way to learn that lerobot failed to import.
START_TIMEOUT_S = 180.0

# Per-request ceiling. An action chunk is one forward pass; anything past this
# means the sidecar is wedged rather than busy.
REQUEST_TIMEOUT_S = 120.0

# What a policy costs to hold, when the sidecar has not said. Used only for
# the capacity check, and deliberately an OVER-estimate: under-quoting here
# produces an OOM instead of a refusal, and a refusal is the cheaper mistake.
# SmolVLA is ~2.2 GB in bf16; the headroom covers activations and the
# allocator's slack.
ASSUMED_POLICY_BYTES = 3_500_000_000

# What the sidecar's own environment costs on disk. Its own torch, its own
# CUDA wheels; this is the number that makes the venv a deliberate act rather
# than a detail.
VENV_DISK_BYTES = 6_000_000_000

INSTALL_HINT = (
    "Run `modelmri policy install` — it builds a separate virtual environment "
    "for the policy and its pinned lerobot, because installing lerobot beside "
    "ModelMRI breaks both."
)


class SidecarError(Refusal):
    """The sidecar cannot answer, and the message says what to do about it.

    A `Refusal` rather than a new root: the server already turns these into a
    409 with the sentence intact, and every message here is authored. See the
    policy comment at `server.py`'s import block.
    """


@dataclass
class PolicyStatus:
    """What the sidecar is, or why there isn't one."""

    running: bool = False
    contract: int | None = None
    policy_repo: str = ""
    revision: str = ""
    device: str = ""
    dtype: str = ""
    # The action-space statistics the policy was trained against. Carried so a
    # caller can refuse to overlay a policy's actions on a dataset's recorded
    # ones when the two are in different units -- see ROADMAP #50, where that
    # overlay is named as the plausible-wrong output to avoid.
    normalisation: dict = field(default_factory=dict)
    port: int = 0
    # Present when `running` is False: the sentence explaining why.
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "running": self.running,
            "contract": self.contract,
            "policy_repo": self.policy_repo,
            "revision": self.revision,
            "device": self.device,
            "dtype": self.dtype,
            "normalisation": dict(self.normalisation),
            "port": self.port,
            "reason": self.reason,
            "means": self.means(),
        }

    def means(self) -> str:
        if not self.running:
            return (
                f"No policy sidecar is running, so nothing here can say what "
                f"the robot would DO — only where it looked. {self.reason}"
            )
        return (
            f"Actions come from {self.policy_repo or 'a policy'} at revision "
            f"{self.revision or 'unknown'}, held in a separate process on "
            f"{self.device or 'an unnamed device'}. They are that policy's "
            f"output on the frame you gave it, not a demonstration and not "
            f"ground truth."
        )


def venv_dir() -> Path:
    """Where the policy's own environment lives.

    Under `paths.data_dir()` rather than beside the package: a venv inside
    site-packages is one `pip install --upgrade` away from being half-deleted,
    and this one holds a multi-gigabyte torch.
    """
    from . import paths

    return Path(paths.data_dir()) / "policy-venv"


def python_in(venv: Path) -> Path:
    """The interpreter inside a venv, on either platform layout."""
    win = venv / "Scripts" / "python.exe"
    return win if win.exists() or sys.platform == "win32" else venv / "bin" / "python"


def installed() -> bool:
    """Is there a usable sidecar environment on this machine?"""
    return python_in(venv_dir()).exists()


def require_installed() -> Path:
    """The sidecar interpreter, or a refusal naming the install command."""
    exe = python_in(venv_dir())
    if not exe.exists():
        raise SidecarError(
            f"The action expert is not installed on this machine, so there is "
            f"nothing here that can say what the robot would do. {INSTALL_HINT}"
        )
    return exe


def check_capacity(
    *,
    vram_gb: float | None,
    accel_name: str = "",
    already_held_bytes: int = 0,
    policy_bytes: int = 0,
    confirm: bool = False,
) -> None:
    """Refuse before starting a second process that also holds weights.

    Two rules, and they answer different questions.

    **Disk** goes through `capacity.free_space`, because the venv is a
    multi-gigabyte torch and a build that runs out of room halfway leaves a
    half-installed environment behind.

    **VRAM does NOT go through `capacity.guard`, and the reason matters.**
    That guard asks "could this model plausibly load at all", and it allows
    `VRAM_MULTIPLE` (4x) headroom because offloading and streaming rescue a
    single oversized model. Neither rescues THIS case: a policy pinned in the
    sidecar does not become smaller because this process would like some
    VRAM, and two resident processes cannot offload into each other's space.
    On an 8 GB card the guard's ceiling is 32 GB, so a 6 GB model beside a
    3.5 GB policy sails through a check that exists to catch exactly that.

    So residency is a straight sum against real VRAM. Having two capacity
    rules in one package would normally be the "one question, two answers"
    defect this project treats as a bug -- it is not, because these are two
    questions: *can it load* and *can both stay resident*.

    Runs BEFORE the subprocess starts. A refusal after a two-minute venv build
    is a refusal that wasted two minutes.
    """
    from . import capacity

    policy_need = int(policy_bytes or ASSUMED_POLICY_BYTES)

    # Disk, for the venv itself.
    volume, free = capacity.free_space(venv_dir())
    if free and VENV_DISK_BYTES > free:
        raise capacity.TooBig(
            f"the policy sidecar's environment needs about "
            f"{VENV_DISK_BYTES / 1e9:,.0f} GB for its own torch, and "
            f"{volume.drive or volume} has {free / 1e9:,.1f} GB free.",
            overridable=False,
        )

    # Residency. Unknown VRAM is NOT zero: an accelerator whose properties
    # could not be read is a different state from a machine with no GPU, and
    # refusing on no evidence would ban a configuration that may be fine.
    # `capacity.guard` records that same distinction one module over.
    if vram_gb is None or vram_gb <= 0 or confirm:
        return

    total = policy_need + max(0, int(already_held_bytes))
    if total <= vram_gb * 1e9:
        return

    held_gb = max(0, int(already_held_bytes)) / 1e9
    raise capacity.TooBig(
        f"the policy sidecar holds its own copy of the weights, so this "
        f"machine would need {total / 1e9:,.1f} GB resident at once — "
        f"{policy_need / 1e9:,.1f} GB for the policy on top of the "
        f"{held_gb:,.1f} GB already loaded here — and "
        f"{accel_name or 'this accelerator'} has {vram_gb:,.1f} GB. Two "
        f"processes cannot offload into each other's memory the way one "
        f"model can, so this is a hard fit rather than a slow one. Unload "
        f"the model in this process first, or run the sidecar on another "
        f"machine.",
        overridable=True,
    )


def check_contract(theirs: object, *, side: str = "policy sidecar") -> int:
    """Validate the sidecar's contract number, or refuse saying which side.

    Implemented HERE rather than imported from `modelmri_policy.contract`, and
    not because importing would be inconvenient -- it is impossible. The
    sidecar lives in its own venv precisely so lerobot's pins cannot reach
    this process, which means this process cannot reach its modules either.

    That constraint is the feature. Two independent declarations of the same
    number, compared on every exchange, is what makes drift visible; a shared
    import would move both halves together and there would be nothing left to
    check. `modelmri_policy/contract.py` says the same thing from the far
    side.
    """
    if not isinstance(theirs, int) or isinstance(theirs, bool):
        raise SidecarError(
            f"The {side} did not state a contract version (got "
            f"{type(theirs).__name__}). Every exchange carries one, so a "
            f"response without it is not from a sidecar this can trust — and "
            f"an action chunk is a claim about what a robot would do."
        )
    if theirs != CONTRACT:
        raise SidecarError(
            f"The {side} speaks contract {theirs}; this speaks {CONTRACT}. "
            f"These are different wire formats, and an action chunk read "
            f"across them would be a different policy's answer wearing this "
            f"one's name. Reinstall with `modelmri policy install --force` so "
            f"both halves match."
        )
    return theirs


def source_dir() -> Path | None:
    """The sidecar's source tree, when ModelMRI is running from a checkout.

    `None` from an installed wheel, where `packages/` was never shipped. The
    two cases install from different places and the difference is worth being
    explicit about: a contributor editing the sidecar wants THEIR copy in the
    venv, and a user who ran `pip install modelmri` wants the released one.
    """
    candidate = Path(__file__).resolve().parents[1] / "packages" / "modelmri-policy"
    return candidate if (candidate / "pyproject.toml").is_file() else None


def requirement() -> str:
    """What to hand pip: the local checkout if there is one, else the release."""
    local = source_dir()
    return f"{local}[policy]" if local is not None else "modelmri-policy[policy]"


def port_file() -> Path:
    """Where a running sidecar records how to reach it.

    A sidecar started by `modelmri policy start` in one terminal has to be
    findable from the server in another, and the port is chosen by the OS
    rather than fixed. The file is a claim, never a guarantee: every reader
    below asks `/status` before believing it, because a file outlives the
    process that wrote it and a stale port is worse than no port.
    """
    return venv_dir().parent / "policy-port.json"


def _drain(stream, sink: deque[str], echo) -> None:
    """Read a child's output to EOF, keeping the tail and passing it on.

    On a thread, and never optional. A child whose stdout is a pipe nobody
    reads blocks once the OS buffer fills -- roughly 64 KB, which pip clears
    in seconds. That exact deadlock is recorded in `runtime.py`'s prefetch
    comment; the difference here is that this side WANTS the output, so it
    drains rather than sending it to DEVNULL.
    """
    try:
        for raw in iter(stream.readline, ""):
            line = raw.rstrip("\r\n")
            sink.append(line)
            if echo is not None and line:
                echo(line)
    except (ValueError, OSError):
        # The pipe was closed under us -- the child exited. Not an error:
        # the exit code is what the caller reads, and it is already on its way.
        pass
    finally:
        try:
            stream.close()
        except (ValueError, OSError):
            pass


def _run(argv: list[str], *, echo=None, cancel=None, timeout: float) -> tuple[int, str]:
    """Run a child to completion, streaming its output, killable throughout.

    Returns `(returncode, tail)`. The tail is bounded at 40 lines because it
    exists to be quoted in a refusal: pip's failures put the reason at the
    END, and a wall of resolver output helps nobody.
    """
    import subprocess
    import time
    from collections import deque

    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        # Windows: its own process group, so terminating the install does not
        # also signal the server that spawned it.
        creationflags=(
            subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        ),
    )
    tail: deque[str] = deque(maxlen=40)
    reader = threading.Thread(
        target=_drain, args=(proc.stdout, tail, echo), daemon=True
    )
    reader.start()

    deadline = time.monotonic() + timeout
    stopped = ""
    while proc.poll() is None:
        if cancel is not None and cancel.wait(0.2):
            stopped = "cancelled"
            break
        if time.monotonic() > deadline:
            stopped = "timeout"
            break
        if cancel is None:
            time.sleep(0.1)

    if stopped:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
    reader.join(timeout=5)
    text = "\n".join(tail)
    if stopped == "cancelled":
        raise SidecarError(
            f"Stopped `{argv[0]} …` before it finished, as asked. The "
            f"half-built environment at {venv_dir()} is not usable; rerun "
            f"`modelmri policy install --force` to start it again."
        )
    if stopped == "timeout":
        raise SidecarError(
            f"`{Path(argv[0]).name} …` ran past {timeout / 60:,.0f} minutes "
            f"without finishing, so it was stopped. Last output:\n{text}"
        )
    return proc.returncode, text


# How long a full sidecar build may take. Generous because it downloads torch
# and CUDA wheels -- gigabytes, on whatever connection this machine has -- and
# a build killed at minute nine leaves the user with nothing to show for the
# nine minutes. Bounded because a pip resolver that has gone circular will
# otherwise sit there all day printing nothing.
INSTALL_TIMEOUT_S = 3600.0


def install(
    *,
    force: bool = False,
    echo=None,
    cancel=None,
    vram_gb: float | None = None,
) -> dict:
    """Build the sidecar's own environment, and verify it speaks this contract.

    Four steps, each of which can fail out loud: capacity, venv, pip, and a
    handshake against the thing just installed. The last one is the point --
    an install that finishes is not the same as an install that WORKS, and the
    failure this whole feature exists to prevent is a sidecar that answers
    with a contract nobody checked.

    `echo` receives each line of pip's output as it arrives, so a caller can
    show a live log rather than a spinner over a ten-minute silence.
    """
    import shutil

    venv = venv_dir()
    exe = python_in(venv)

    if exe.exists() and not force:
        # `probe`'s refusal is allowed straight through rather than re-wrapped.
        # It already names the environment, quotes what its interpreter said
        # and gives the remedy, and restating that here would either duplicate
        # it or replace the specifics with something vaguer. It also carries
        # the child's stderr, which is exactly the text this project refuses
        # to interpolate into a second sentence -- see test_no_exception_leaks.
        existing = probe(exe)
        return {
            "installed": True,
            "rebuilt": False,
            "venv": str(venv),
            "contract": existing,
            "requirement": requirement(),
            "means": (
                f"The action expert was already installed at {venv} and "
                f"speaks contract {existing}, which matches this side. "
                f"Nothing was rebuilt."
            ),
        }

    # Disk only. VRAM is not the question at install time -- nothing is
    # resident yet, and refusing to INSTALL because a model happens to be
    # loaded right now would be refusing a permanent thing over a temporary
    # one. `check_capacity` runs again, with residency, before the sidecar
    # actually starts.
    check_capacity(vram_gb=None)

    if venv.exists():
        if echo is not None:
            echo(f"removing the existing environment at {venv}")
        shutil.rmtree(venv, ignore_errors=True)
        if venv.exists():
            raise SidecarError(
                f"Could not remove the existing environment at {venv}. "
                f"Something is still holding a file inside it — a running "
                f"sidecar is the usual answer. Stop it and try again."
            )

    venv.parent.mkdir(parents=True, exist_ok=True)
    if echo is not None:
        echo(f"creating a virtual environment at {venv}")
    code, tail = _run(
        [sys.executable, "-m", "venv", str(venv)],
        echo=echo,
        cancel=cancel,
        timeout=300.0,
    )
    if code != 0 or not python_in(venv).exists():
        raise SidecarError(
            f"Could not create a virtual environment at {venv} "
            f"(exit {code}). On Debian and Ubuntu this usually means the "
            f"python3-venv package is missing. Output:\n{tail}"
        )

    exe = python_in(venv)
    want = requirement()
    if echo is not None:
        echo(f"installing {want} — this downloads its own torch, so it is slow")
    code, tail = _run(
        [
            str(exe),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            # Off on purpose. pip's bar redraws with carriage returns, and
            # every redraw becomes its own line once stdout is a pipe -- which
            # turns a download into thousands of near-identical log lines.
            "--progress-bar",
            "off",
            want,
        ],
        echo=echo,
        cancel=cancel,
        timeout=INSTALL_TIMEOUT_S,
    )
    if code != 0:
        local = source_dir()
        where = (
            f"from the checkout at {local}"
            if local is not None
            else "from PyPI, where modelmri-policy may not be published yet"
        )
        raise SidecarError(
            f"Installing {want} {where} failed (exit {code}). The environment "
            f"at {venv} is half-built and should be removed. pip said:\n{tail}"
        )

    # The handshake, against what was just installed rather than what was
    # asked for. A pip that resolved to a DIFFERENT release of the sidecar --
    # an old wheel cached locally, a version pin somewhere upstream -- is
    # exactly the drift the contract number exists to catch, and catching it
    # here is far cheaper than catching it mid-analysis.
    contract = probe(exe)
    return {
        "installed": True,
        "rebuilt": True,
        "venv": str(venv),
        "contract": contract,
        "requirement": want,
        "means": (
            f"The action expert now lives in its own environment at {venv} "
            f"and speaks contract {contract}, which matches this side. It "
            f"holds its own torch, so it does not disturb ModelMRI's."
        ),
    }


def probe(exe: Path) -> int:
    """Ask an installed sidecar what contract it speaks, without starting it.

    Runs in the sidecar's interpreter and prints one number. Deliberately
    imports only `modelmri_policy.contract`, which is stdlib-only -- so this
    answers even when lerobot itself is broken, and the two failures stay
    distinguishable instead of both arriving as "it does not work".
    """
    code, tail = _run(
        [
            str(exe),
            "-c",
            "import modelmri_policy.contract as c; print(c.CONTRACT)",
        ],
        timeout=120.0,
    )
    if code != 0:
        raise SidecarError(
            f"The environment at {exe.parent.parent} does not have a working "
            f"modelmri_policy in it, so nothing there could answer for a "
            f"policy. Rebuild it with `modelmri policy install --force`. Its "
            f"interpreter exited {code} saying:\n{tail}"
        )
    stated = tail.strip().splitlines()[-1].strip() if tail.strip() else ""
    return check_contract(
        int(stated) if stated.isdigit() else stated, side="installed sidecar"
    )


def _post(port: int, route: str, body: dict, *, timeout: float) -> tuple[dict, bytes]:
    """One request, with the contract on it and checked on the way back.

    Returns `(json_or_empty, raw_bytes)`; `/hidden` answers safetensors rather
    than JSON, and its contract rides in a header for the same reason.
    """
    payload = json.dumps({**body, "contract": CONTRACT}).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{route}",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            header_contract = resp.headers.get("X-ModelMRI-Contract")
            kind = (resp.headers.get("Content-Type") or "").split(";")[0].strip()
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", "replace")[:400]
        raise SidecarError(
            f"The policy sidecar refused {route} ({err.code}): {detail}"
        ) from None
    except urllib.error.URLError as err:
        raise SidecarError(
            f"The policy sidecar is not answering on port {port} "
            f"({err.reason}). It may have exited; `modelmri policy start` "
            f"brings it back."
        ) from None

    if kind == "application/json":
        try:
            data = json.loads(raw.decode("utf-8"))
        except ValueError:
            raise SidecarError(
                f"The policy sidecar answered {route} with something that is "
                f"not JSON, so there is nothing here to read."
            ) from None
        check_contract(data.get("contract"), side="policy sidecar")
        return data, raw

    check_contract(
        int(header_contract) if (header_contract or "").isdigit() else header_contract,
        side="policy sidecar",
    )
    return {}, raw


# ---------------------------------------------------------------- lifecycle

# The sidecar this process started, if it started one. Module state rather
# than a parameter because there is exactly one per machine by construction:
# the port file is a single path, and two sidecars would be two processes both
# claiming to be the policy.
_CHILD = None
_CHILD_LOCK = threading.Lock()

# The ready line, restated here for the same reason CONTRACT is. This side
# cannot import `modelmri_policy.contract` -- different venv -- so it declares
# what it expects to read, and a sidecar printing something else is a sidecar
# this cannot talk to, which is exactly what should be visible.
READY_PREFIX = "MODELMRI_POLICY_PORT="


def _get(port: int, route: str, *, timeout: float) -> dict:
    """A GET with the contract checked on the way back. See `_post`."""
    req = urllib.request.Request(f"http://127.0.0.1:{port}{route}", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", "replace")[:400]
        raise SidecarError(
            f"The policy sidecar refused {route} ({err.code}): {detail}"
        ) from None
    except urllib.error.URLError as err:
        raise SidecarError(
            f"The policy sidecar is not answering on port {port} "
            f"({err.reason})."
        ) from None
    except (TimeoutError, OSError) as err:
        # The type, not the text. A socket error's message carries whatever
        # the OS put in it, and this value is published to a browser.
        raise SidecarError(
            f"The policy sidecar did not answer on port {port} "
            f"({type(err).__name__})."
        ) from None
    except ValueError:
        raise SidecarError(
            f"The policy sidecar answered {route} with something that is not "
            f"JSON, so there is nothing here to read."
        ) from None
    check_contract(data.get("contract"), side="policy sidecar")
    return data


def _write_port(port: int, pid: int) -> None:
    path = port_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"port": port, "pid": pid, "contract": CONTRACT}),
        encoding="utf-8",
    )


def _read_port() -> int:
    """The recorded port, or 0.

    A missing file and a corrupt one both mean the same thing to every caller
    here -- there is no sidecar to attach to -- and `status` deciding by
    ASKING is what makes that safe. A port read out of a file is a claim.
    """
    try:
        data = json.loads(port_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    port = data.get("port")
    if not isinstance(port, int) or isinstance(port, bool):
        return 0
    return port if 0 < port < 65536 else 0


def _forget_port() -> None:
    try:
        port_file().unlink()
    except OSError:
        pass


def status(*, timeout: float = 5.0) -> PolicyStatus:
    """What the sidecar is right now, measured rather than remembered.

    Every branch ends in a sentence a panel can show. "running: false" is the
    least useful true thing this could return: "install it", "start it" and
    "it died" are three problems with three different fixes, and a boolean
    collapses them into one.
    """
    port = _read_port()
    if not port:
        if not installed():
            return PolicyStatus(
                reason=f"The action expert is not installed. {INSTALL_HINT}"
            )
        return PolicyStatus(
            reason=(
                "The action expert is installed but no sidecar is running. "
                "`modelmri policy start` brings one up."
            )
        )

    try:
        data = _get(port, "/status", timeout=timeout)
    except SidecarError:
        # The file said a port and nothing is there: that is a crash, not a
        # configuration. Say so, and clear the claim so the next caller is
        # told "not running" instead of being sent to the same dead port.
        #
        # The caught sentence is deliberately not repeated. It says "not
        # answering on port N", which is the thing this line already says
        # better, with the reason attached.
        _forget_port()
        return PolicyStatus(
            port=port,
            reason=(
                f"A sidecar was recorded on port {port} but is not answering, "
                f"so it has exited. `modelmri policy start` brings one back."
            ),
        )

    if not data.get("ready"):
        return PolicyStatus(
            contract=CONTRACT,
            port=port,
            reason=(
                "The sidecar is up but has no policy loaded, so there is "
                "nothing yet that could act. Load one to ask what it would do."
            ),
        )
    return PolicyStatus(
        running=True,
        contract=CONTRACT,
        policy_repo=str(data.get("policy_repo") or ""),
        revision=str(data.get("revision") or ""),
        device=str(data.get("device") or ""),
        dtype=str(data.get("dtype") or ""),
        normalisation=data.get("normalisation") or {},
        port=port,
    )


def start(
    *,
    policy_repo: str = "",
    device: str = "",
    echo=None,
    already_held_bytes: int = 0,
    vram_gb: float | None = None,
    accel_name: str = "",
    confirm: bool = False,
) -> PolicyStatus:
    """Bring a sidecar up, wait until it can answer, and record where it is.

    Attaches to a live one rather than starting a second. Two processes each
    holding a policy is the capacity failure this module exists to refuse, and
    starting one by accident would be this code causing it.

    The wait is for the READY line, not for the socket. A port that accepts
    connections before the server loop is running would let a caller send
    `/act` and get a refused connection that looks like a bug rather than a
    sequence.
    """
    import subprocess

    global _CHILD

    with _CHILD_LOCK:
        live = status()
        if live.port:
            return live

        exe = require_installed()
        # Residency, with what this process is already holding, BEFORE the
        # subprocess exists. A refusal that costs nothing is the point.
        check_capacity(
            vram_gb=vram_gb,
            accel_name=accel_name,
            already_held_bytes=already_held_bytes,
            confirm=confirm,
        )

        if echo is not None:
            echo(f"starting the policy sidecar from {exe}")
        proc = subprocess.Popen(
            [str(exe), "-m", "modelmri_policy", "--port", "0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=(
                subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
            ),
        )

        # One thread does both jobs: find the ready line, then keep draining.
        # Both halves are required. The line is what "started" means, and a
        # pipe left unread after it deadlocks the child the moment it writes
        # 64 KB of anything -- a traceback out of a request thread, say. That
        # exact deadlock is recorded in `runtime.py`'s prefetch comment.
        port = 0
        tail: deque[str] = deque(maxlen=40)
        ready = threading.Event()

        def _pump() -> None:
            nonlocal port
            try:
                for raw in iter(proc.stdout.readline, ""):
                    line = raw.rstrip("\r\n")
                    tail.append(line)
                    if echo is not None and line:
                        echo(line)
                    if port == 0 and line.startswith(READY_PREFIX):
                        stated = line[len(READY_PREFIX) :].strip()
                        if stated.isdigit():
                            port = int(stated)
                            ready.set()
            except (ValueError, OSError):
                pass
            finally:
                # EOF means the child exited. Set it either way, so a sidecar
                # that dies during import ends the wait immediately instead of
                # holding the caller for the full three minutes.
                ready.set()

        threading.Thread(target=_pump, daemon=True).start()
        ready.wait(START_TIMEOUT_S)

        if port == 0:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            said = "\n".join(tail).strip()
            raise SidecarError(
                "The policy sidecar did not report a port, so it was "
                "stopped. This is usually its environment failing to import "
                "lerobot; `modelmri policy install --force` rebuilds it."
                + (f" It said:\n{said}" if said else " It said nothing at all.")
            )

        _CHILD = proc
        _write_port(port, proc.pid)

    if policy_repo:
        load(policy_repo, device=device)
    return status()


def load(policy_repo: str, *, device: str = "") -> PolicyStatus:
    """Ask the running sidecar to hold a policy. Refuses if none is running."""
    port = _read_port()
    if not port:
        raise SidecarError(
            "No policy sidecar is running, so there is nothing to load a "
            "policy into. `modelmri policy start` brings one up."
        )
    _post(
        port,
        "/load",
        {"policy_repo": policy_repo, "device": device},
        timeout=START_TIMEOUT_S,
    )
    return status()


def stop(*, timeout: float = 15.0) -> bool:
    """Stop the sidecar THIS process started. Says whether one was running.

    Deliberately only a child of this process. A sidecar started in another
    terminal is that terminal's to stop: reading a pid out of a file and
    signalling it means signalling whatever inherited that pid after a crash,
    which is how a cleanup routine ends up killing something it never started.
    """
    import subprocess

    global _CHILD

    with _CHILD_LOCK:
        proc, _CHILD = _CHILD, None
        _forget_port()
        if proc is None or proc.poll() is not None:
            return False
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=timeout)
        return True
