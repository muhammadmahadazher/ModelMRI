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


class SidecarGone(SidecarError):
    """Nothing is listening on that port. The process is not there.

    The ONE failure that means "the sidecar died", separated from its five
    siblings because only this one justifies deleting the port file.

    An adversarial review of this module found the cost of not separating
    them. `status` caught the whole `SidecarError` class and reported every
    member of it as "it has exited" — so a live sidecar speaking the wrong
    contract, a wedged one answering 500, one answering slowly, and one
    answering non-JSON were all reported as a crash, AND their port record was
    deleted. The contract-drift refusal is the single thing this feature
    exists to make visible; it was the thing most reliably destroyed. Worse,
    with the record gone `start` then spawned a SECOND sidecar beside the live
    one — precisely the two-processes-holding-weights failure `check_capacity`
    exists to refuse.
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
    # Is a process ANSWERING on that port? Separate from `running`, which asks
    # whether a policy is loaded, and separate from `port`, which is only what
    # a file claimed.
    #
    # This field exists because `start` used to gate on `port` being non-zero.
    # A review found what that costs: after a crash, `status` returned the
    # dead port with `running=False`, `start` saw a truthy `port`, decided a
    # sidecar was already up, and returned without starting one. The first
    # `modelmri policy start` after any crash silently did nothing.
    reachable: bool = False
    # Present when `running` is False: the sentence explaining why.
    reason: str = ""

    # --- what the loaded checkpoint says it consumes ----------------------
    # Carried because a caller cannot assemble a valid request without them,
    # and the alternative to stating them is letting the caller guess. A VLA
    # given the wrong camera set or the wrong state width does not fail: it
    # answers a different question in the same shape.
    family: str = ""
    cameras: list = field(default_factory=list)
    state_width: int | None = None
    action_width: int | None = None
    chunk_size: int | None = None
    # Whether the action head samples. False means #50's instruction-swap test
    # has no reference to measure against -- its denominator is the policy's
    # own sampling spread -- and that is a refusal rather than a result of 0.
    samples: bool = False
    # The two versions in the OTHER environment. The whole point of the
    # separation is that these differ from this process's; printing them is
    # what makes the difference visible instead of merely true.
    lerobot_version: str = ""
    torch_version: str = ""

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
            "reachable": self.reachable,
            "reason": self.reason,
            "family": self.family,
            "cameras": list(self.cameras),
            "state_width": self.state_width,
            "action_width": self.action_width,
            "chunk_size": self.chunk_size,
            "samples": self.samples,
            "lerobot_version": self.lerobot_version,
            "torch_version": self.torch_version,
            "accelerated": self.accelerated(),
            "means": self.means(),
        }

    def accelerated(self) -> bool | None:
        """Is the sidecar's torch a GPU build? `None` when nothing has said.

        A tri-state on purpose. "running on the processor" and "nobody has
        reported a torch build" lead to different actions — reinstall versus
        start the sidecar — and a bare False would tell somebody to fix a
        machine that is fine.
        """
        if not self.torch_version:
            return None
        return "+cpu" not in self.torch_version

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


def _status_phrase(code: object) -> str:
    """ " Not Found" for 404, or "" for a code the stdlib does not know.

    Looked up from the integer rather than read off the response, so the words
    in a refusal are always this machine's own. `otel.py` needed the identical
    thing for the identical reason and has its own copy: these two modules
    share no import today, and a `utils` created to hold six lines is a
    dependency edge bought for nothing.
    """
    import http

    try:
        return f" {http.HTTPStatus(int(code)).phrase}"
    except (ValueError, TypeError):
        return ""


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
    check_disk: bool = True,
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

    # Disk, for the venv itself -- and only when one is about to be BUILT.
    # `start` passes `check_disk=False`, because refusing to launch a sidecar
    # that is already installed, over room needed to install it, is refusing a
    # thing that has already happened. The bytes are on the disk; that is what
    # "installed" means.
    volume, free = capacity.free_space(venv_dir()) if check_disk else (None, 0)
    if check_disk and free and VENV_DISK_BYTES > free:
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


# The wheel index that carries CUDA builds, matched to the one ModelMRI's own
# pyproject pins so both processes end up on the same CUDA minor. Two torches
# built against different CUDA runtimes in one machine's memory is a class of
# problem nobody wants to debug from a robot panel.
CUDA_WHEEL_INDEX = "https://download.pytorch.org/whl/cu128"


def cuda_index() -> str:
    """The CUDA wheel index, or "" when this machine would not use it.

    Asked of the HARDWARE, not of ModelMRI's torch. `devices.detect()` reaches
    "cuda" only through `torch.cuda.is_available()`, which is a fact about the
    torch build in THIS environment — and this function is choosing the torch
    build for a DIFFERENT one. A machine with a real NVIDIA card and a CPU
    ModelMRI would have been given a CPU sidecar too, permanently, because the
    thing being fixed was used as the test for whether it needed fixing.

    `devices._nvidia_present()` is the right question and already exists: it
    asks nvidia-smi, which answers about the driver rather than about a wheel.
    Either signal is enough — a working CUDA torch here proves a card, and a
    driver proves one torch cannot see.

    Windows only. A Linux machine gets CUDA from PyPI's default wheel already,
    so pointing at the index there would download the same thing twice.
    """
    if sys.platform != "win32":
        return ""
    try:
        from . import devices

        if devices._nvidia_present():
            return CUDA_WHEEL_INDEX
        return CUDA_WHEEL_INDEX if devices.detect().kind == "cuda" else ""
    except Exception:
        # Measuring the machine is best-effort. Failing to measure means the
        # default wheel, which is slow rather than wrong — and `status` keeps
        # reporting the build that actually landed either way.
        return ""


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
            # Closing a pipe the child already closed. There is nothing left
            # to do about it and nothing to report: the caller is about to
            # read the exit code, which is the answer it actually wants, and
            # a failure to close a stream that is already gone would only
            # obscure it.
            pass


# The floor lerobot itself declares (`requires-python >=3.12` on every 0.6
# release). ModelMRI supports 3.10, so on two of the four Pythons this project
# is tested against, a venv built from `sys.executable` CANNOT hold the policy
# stack at all -- pip would refuse it with a resolver message about a version
# that is not the user's fault and does not name the fix.
#
# Stated rather than fetched: reading it from PyPI at install time would put a
# network call in front of a check, and pip remains the authority anyway. If
# lerobot raises its floor this becomes an over-strict pre-check whose refusal
# is still true in direction, and pip's error still arrives underneath it.
POLICY_PYTHON_MIN = (3, 12)


def _version_of(exe: Path) -> tuple[int, int] | None:
    """Ask an interpreter its version. `None` when it will not answer.

    Run, not parsed from the filename. `python3.12` on PATH is a name somebody
    chose, and a symlink pointing somewhere else is exactly the situation that
    ends with a venv one minor too old and a confusing pip error.
    """
    try:
        code, said = _run(
            [str(exe), "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
            timeout=30.0,
        )
    except SidecarError:
        return None
    if code != 0:
        return None
    line = said.strip().splitlines()[-1].strip() if said.strip() else ""
    parts = line.split(".")
    if len(parts) != 2 or not all(p.isdigit() for p in parts):
        return None
    return (int(parts[0]), int(parts[1]))


def interpreter_for_venv() -> Path:
    """A Python new enough to hold the policy stack, or a refusal saying so.

    This process's own interpreter first, and any other only when that one is
    too old. Borrowing a different Python is a real change in what gets built
    -- a second CUDA wheel set compiled for a different ABI -- so it is a
    fallback rather than a preference.
    """
    import shutil

    mine = Path(sys.executable)
    if sys.version_info >= POLICY_PYTHON_MIN:
        return mine

    want = f"{POLICY_PYTHON_MIN[0]}.{POLICY_PYTHON_MIN[1]}"
    tried: list[str] = []
    names = [f"python3.{minor}" for minor in range(POLICY_PYTHON_MIN[1], 20)]
    names += [f"python{POLICY_PYTHON_MIN[0]}.{POLICY_PYTHON_MIN[1]}"]
    for name in names:
        found = shutil.which(name)
        if not found:
            continue
        tried.append(found)
        got = _version_of(Path(found))
        if got is not None and got >= POLICY_PYTHON_MIN:
            return Path(found)

    if sys.platform == "win32":
        # The py launcher is how a Windows machine usually has several. Asking
        # it for a specific minor is the only reliable way to find one, since
        # Windows does not put `python3.12` on PATH.
        launcher = shutil.which("py")
        if launcher:
            for minor in range(POLICY_PYTHON_MIN[1], 20):
                code, said = _run(
                    [
                        launcher,
                        f"-{POLICY_PYTHON_MIN[0]}.{minor}",
                        "-c",
                        "import sys; print(sys.executable)",
                    ],
                    timeout=30.0,
                )
                path = said.strip().splitlines()[-1].strip() if said.strip() else ""
                if code == 0 and path and Path(path).exists():
                    tried.append(path)
                    got = _version_of(Path(path))
                    if got is not None and got >= POLICY_PYTHON_MIN:
                        return Path(path)

    looked = f" Looked at: {', '.join(tried)}." if tried else ""
    raise SidecarError(
        f"The action expert needs Python {want} or newer — that is lerobot's "
        f"own floor, not this project's — and ModelMRI is running on "
        f"{sys.version_info[0]}.{sys.version_info[1]}. Nothing newer was found "
        f"on this machine.{looked} Install a Python {want}+ and it will be "
        f"used for the sidecar's environment; ModelMRI itself can stay where "
        f"it is, which is the entire point of the two being separate."
    )


def _run(argv: list[str], *, echo=None, cancel=None, timeout: float) -> tuple[int, str]:
    """Run a child to completion, streaming its output, killable throughout.

    Returns `(returncode, tail)`. The tail is bounded at 40 lines because it
    exists to be quoted in a refusal: pip's failures put the reason at the
    END, and a wall of resolver output helps nobody.
    """
    import subprocess
    import time
    from collections import deque

    try:
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            # Windows: its own process group, so terminating the install does
            # not also signal the server that spawned it.
            creationflags=(
                subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
            ),
        )
    except OSError as err:
        # A program that will not start is a sentence naming the program, not
        # a WinError somebody has to look up. Reachable in practice: every
        # `exists()` check upstream is a moment before this line, and a venv
        # being rebuilt in another terminal can win that race.
        raise SidecarError(
            f"Could not run `{argv[0]}` ({type(err).__name__}). It is not "
            f"there, or it is not something this machine can execute."
        ) from None
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

    # Which Python, before which package. A venv one minor too old fails at
    # the pip step with a resolver message about `requires-python` that reads
    # as a broken install rather than as a fact about lerobot.
    base = interpreter_for_venv()
    if echo is not None and base != Path(sys.executable):
        echo(
            f"ModelMRI runs on Python {sys.version_info[0]}.{sys.version_info[1]}, "
            f"which lerobot does not support — building the sidecar from {base}"
        )

    venv.parent.mkdir(parents=True, exist_ok=True)
    if echo is not None:
        echo(f"creating a virtual environment at {venv}")
    code, tail = _run(
        [str(base), "-m", "venv", str(venv)],
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
    pip = [
        str(exe),
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        # Off on purpose. pip's bar redraws with carriage returns, and every
        # redraw becomes its own line once stdout is a pipe -- which turns a
        # download into thousands of near-identical log lines.
        "--progress-bar",
        "off",
    ]

    # CUDA torch FIRST, when this machine has a CUDA card, and this is not an
    # optimisation. PyPI's default torch wheel on Windows is CPU-only, so a
    # plain `pip install lerobot` gives the sidecar a CPU torch while ModelMRI
    # itself runs on the GPU -- measured on this machine as `2.11.0+cpu`, with
    # lerobot printing "No accelerated backend detected. Using default cpu,
    # this will be slow." One process on the card and one on the processor is
    # a silent asymmetry: the numbers are still right, the wait is forty times
    # longer, and nothing on the page says why.
    #
    # A separate call rather than `--extra-index-url` because pip picks
    # freely between indexes and the choice would not be reproducible.
    index = cuda_index()
    if index:
        if echo is not None:
            echo(f"installing CUDA torch from {index} — this machine has a CUDA card")
        code, tail = _run(
            [*pip, "--index-url", index, "torch", "torchvision"],
            echo=echo,
            cancel=cancel,
            timeout=INSTALL_TIMEOUT_S,
        )
        if code != 0:
            # NOT fatal. A CPU sidecar is slow, not wrong, and refusing the
            # whole install because the CUDA index was unreachable would trade
            # a working slow thing for nothing at all. It is reported instead,
            # and `status` keeps saying which build is there.
            if echo is not None:
                echo(
                    "the CUDA index did not answer; carrying on with the "
                    "default torch, which will run the policy on the processor"
                )

    want = requirement()
    if echo is not None:
        echo(f"installing {want} — this downloads its own torch, so it is slow")
    code, tail = _run(
        [*pip, want],
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
        # The CODE and the standard phrase for it, never the response BODY.
        #
        # The body is written by whatever is listening on that port. Usually
        # that is our own sidecar and the sentence is authored — but the
        # contract has not been checked at this point (a 4xx never reaches
        # `check_contract`), so "usually ours" is the whole problem. CodeQL
        # traced it: body -> SidecarError -> `status().reason` -> `means()` ->
        # five route bodies -> a browser.
        #
        # Nothing actionable is lost. 409 and 500 send you to different
        # places, and the contract-drift sentence — the one worth reading —
        # comes from `check_contract`, which this module writes itself.
        raise SidecarError(
            f"The policy sidecar refused {route} with "
            f"{err.code}{_status_phrase(err.code)}."
        ) from None
    except urllib.error.URLError as err:
        raise SidecarGone(
            f"The policy sidecar is not answering on port {port} "
            f"({type(err.reason).__name__ if err.reason else type(err).__name__})."
            f" It may have exited; `modelmri policy start` brings it back."
        ) from None
    except (TimeoutError, OSError) as err:
        # Present here for the same reason it is in `_get`: without it a bare
        # socket error escaped `_post` as an OSError while the identical
        # condition arrived from `_get` as an authored refusal. One question,
        # two answers, and only one of them reached a panel as a sentence.
        raise SidecarError(
            f"The policy sidecar accepted the connection on port {port} but "
            f"did not answer {route} in time ({type(err).__name__}). A "
            f"forward pass on a large policy can outlast this."
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
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{route}",
        # A GET has no body to put the version in, so it rides in a header.
        # The sidecar checks it: `/status` is what the panel polls, and a
        # client one version out being told "ready: true" by a sidecar it
        # cannot talk to is the drift this contract exists to catch, arriving
        # through the one route that was not looking.
        headers={"X-ModelMRI-Contract": str(CONTRACT)},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        # The CODE and the standard phrase for it, never the response BODY.
        #
        # The body is written by whatever is listening on that port. Usually
        # that is our own sidecar and the sentence is authored — but the
        # contract has not been checked at this point (a 4xx never reaches
        # `check_contract`), so "usually ours" is the whole problem. CodeQL
        # traced it: body -> SidecarError -> `status().reason` -> `means()` ->
        # five route bodies -> a browser.
        #
        # Nothing actionable is lost. 409 and 500 send you to different
        # places, and the contract-drift sentence — the one worth reading —
        # comes from `check_contract`, which this module writes itself.
        raise SidecarError(
            f"The policy sidecar refused {route} with "
            f"{err.code}{_status_phrase(err.code)}."
        ) from None
    except urllib.error.URLError as err:
        # `SidecarGone`, and only here. Nothing accepted the connection, which
        # is the one condition that actually means the process is not there.
        # The TYPE of the reason, not its text. `URLError.reason` is the
        # underlying OSError, whose message is whatever the operating system
        # put in it -- and this sentence reaches a browser through
        # `status().reason`, `means()` and the /api/policy body. CodeQL found
        # that path (py/stack-trace-exposure) on five routes at once; the
        # project's own leak test missed it because its regex looks for
        # `{err}` and this was `{err.reason}`, which is the same leak wearing
        # an attribute.
        raise SidecarGone(
            f"The policy sidecar is not answering on port {port} "
            f"({type(err.reason).__name__ if err.reason else type(err).__name__})."
        ) from None
    except (TimeoutError, OSError) as err:
        # A timeout is NOT a death. Something accepted the connection and then
        # took too long, which is a sidecar that is busy or wedged — and
        # treating it as gone would delete the record of a process that is
        # still holding a policy in memory.
        #
        # The type, not the text. A socket error's message carries whatever
        # the OS put in it, and this value is published to a browser.
        raise SidecarError(
            f"The policy sidecar accepted the connection on port {port} but "
            f"did not answer in time ({type(err).__name__}). It is running and "
            f"busy, or wedged."
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
        # Already gone, or held by something else. Either way the CLAIM is
        # what matters, not the file: every reader calls `status`, which asks
        # the recorded port whether it answers and treats silence as "not
        # running". A file that outlives this call is at worst read once more
        # and rejected the same way.
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
    except SidecarGone:
        # Nothing accepted the connection: the process really is not there.
        # This is the ONLY branch that deletes the record, because it is the
        # only one where the record is false.
        _forget_port()
        return PolicyStatus(
            reason=(
                f"A sidecar was recorded on port {port} but nothing is "
                f"listening there, so it has exited. `modelmri policy start` "
                f"brings one back."
            ),
        )
    except SidecarError as err:
        # Something IS there and it answered wrongly — a contract mismatch, a
        # 500 from a wedged process, a timeout, a non-JSON body. The port
        # record stays, because a live process holding a policy in memory is
        # exactly what it records, and the authored sentence is carried
        # through instead of being replaced by a guess about a crash.
        #
        # This is the branch an adversarial review found missing. Collapsing
        # it into the one above destroyed the contract-drift refusal — the
        # single failure this whole feature exists to surface — and then
        # deleted the port record, after which `start` would spawn a second
        # sidecar beside the live one.
        return PolicyStatus(
            port=port,
            reachable=True,
            reason=str(err),
        )

    if not data.get("ready"):
        return PolicyStatus(
            contract=CONTRACT,
            port=port,
            reachable=True,
            # The environment's versions travel even with nothing loaded: an
            # empty sidecar still has a torch build, and "it will run on the
            # processor" is worth knowing BEFORE waiting for a policy to load.
            lerobot_version=str(data.get("lerobot_version") or ""),
            torch_version=str(data.get("torch_version") or ""),
            reason=(
                "The sidecar is up but has no policy loaded, so there is "
                "nothing yet that could act. Load one to ask what it would do."
            ),
        )
    return PolicyStatus(
        running=True,
        reachable=True,
        contract=CONTRACT,
        policy_repo=str(data.get("policy_repo") or ""),
        revision=str(data.get("revision") or ""),
        device=str(data.get("device") or ""),
        dtype=str(data.get("dtype") or ""),
        normalisation=data.get("normalisation") or {},
        port=port,
        family=str(data.get("family") or ""),
        cameras=list(data.get("cameras") or []),
        state_width=_whole(data.get("state_width")),
        action_width=_whole(data.get("action_width")),
        chunk_size=_whole(data.get("chunk_size")),
        samples=bool(data.get("samples")),
        lerobot_version=str(data.get("lerobot_version") or ""),
        torch_version=str(data.get("torch_version") or ""),
    )


def _whole(value: object) -> int | None:
    """An int off the wire, or None. `None` is a real answer here.

    "this policy consumes no state" and "the state is zero wide" are different
    facts, and a caller building a request has to be able to tell them apart.
    A bool is rejected because `isinstance(True, int)` is True and a width of
    `True` would silently become 1.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value > 0 else None


def _kill(proc, tail: deque[str], pump=None) -> str:
    """Stop a child and return what it said, without racing its reader.

    The join matters. `tail` is a deque the pump thread is still appending to,
    and joining an already-dead process's reader is instant — but reading the
    deque while another thread appends to it is how a message arrives
    half-written, which in this module is a message somebody is meant to act
    on.
    """
    import subprocess

    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            # Reaped, not merely signalled. `kill()` without a `wait()` leaves
            # a zombie on POSIX for as long as this process lives.
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # A process that ignored SIGKILL is stuck in the kernel --
                # uninterruptible I/O, almost always a GPU driver call. There
                # is no stronger signal to send and nothing here can free it,
                # so waiting longer would only hold the caller hostage to a
                # thing neither of them can fix. The caller's own message
                # already names the sidecar and the port.
                pass
    if pump is not None:
        pump.join(timeout=5)
    return "\n".join(tail).strip()


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
        # `reachable`, not `port`. A port is what a FILE said; reachable is
        # what a process answered. Gating on the port meant the first
        # `modelmri policy start` after any crash returned the dead port and
        # started nothing — silently, because "attached to the running one" and
        # "did nothing" print the same way.
        if live.reachable:
            return live

        exe = require_installed()
        # Residency, with what this process is already holding, BEFORE the
        # subprocess exists. A refusal that costs nothing is the point.
        check_capacity(
            vram_gb=vram_gb,
            accel_name=accel_name,
            already_held_bytes=already_held_bytes,
            confirm=confirm,
            check_disk=False,
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
                # The child's stdout went away mid-read, which means the child
                # went away. Not handled here on purpose: `ready.set()` in the
                # finally below ends the wait immediately, and the caller then
                # finds `port == 0` and raises the refusal that names the
                # likely cause and the command that fixes it. Raising from
                # this thread would lose that sentence.
                pass
            finally:
                # EOF means the child exited. Set it either way, so a sidecar
                # that dies during import ends the wait immediately instead of
                # holding the caller for the full three minutes.
                ready.set()

        pump = threading.Thread(target=_pump, daemon=True)
        pump.start()

        # Everything from here to `_CHILD = proc` runs under a guard, because
        # between those two lines this process owns a child that nothing else
        # can see. An exception in the middle — a KeyboardInterrupt during the
        # three-minute wait is the realistic one — would leave a sidecar
        # holding a policy in memory with no handle, no port record and no way
        # to stop it short of the task manager.
        try:
            ready.wait(START_TIMEOUT_S)

            if port == 0:
                said = _kill(proc, tail, pump)
                raise SidecarError(
                    "The policy sidecar did not report a port, so it was "
                    "stopped. This is usually its environment failing to "
                    "import lerobot; `modelmri policy install --force` "
                    "rebuilds it."
                    + (f" It said:\n{said}" if said else " It said nothing at all.")
                )
        except BaseException:
            # BaseException, not Exception: KeyboardInterrupt and SystemExit
            # are exactly the two that arrive during a long wait, and they are
            # the two a bare `except Exception` would let orphan the child.
            _kill(proc, tail, pump)
            raise

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
        # Ownership FIRST, and the port record only if we own the child.
        # Forgetting unconditionally meant a process that started nothing
        # still unlinked the one machine-wide port file — erasing the record
        # of somebody else's live sidecar, which is the same class of harm as
        # signalling a pid you did not spawn, and the exact thing this
        # function's docstring says it refuses to do.
        if proc is None or proc.poll() is not None:
            return False
        _forget_port()
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=timeout)
        return True


# ------------------------------------------------------------ asking it to act


def _raw_base64(payload):
    """A `data:` URL reduced to the base64 the sidecar's contract asks for.

    `vla_data.encode_png` returns a DATA URL, and that is correct — the frame
    server hands the same string to an `<img src>` and the browser needs the
    prefix. The sidecar does not: its `decode_frame` calls
    `base64.b64decode(payload, validate=True)`, which rejects
    `data:image/png;base64,` as a non-base64 digit.

    MEASURED: every call to /act carrying a frame from `reader.frame(...)`
    failed on that, so `/api/vla/actions/{compare,swap,knockout}` could not
    succeed on any dataset or any frame. The sidecar's refusal named the
    camera and said the payload was not a base64 image, which was true and
    unhelpful, because nothing upstream was listening.

    Stripped HERE rather than at the four call sites, because this function is
    the process boundary and a boundary is where a contract is owed. Anything
    that is not a `data:` URL passes through untouched, so a caller already
    sending raw base64 is unaffected.
    """
    if isinstance(payload, str) and payload.startswith("data:"):
        head, _, body = payload.partition(",")
        # Only for a base64 payload. A `data:` URL can also be percent-encoded
        # text, and handing THAT to a base64 decoder would swap one wrong
        # answer for another.
        if "base64" in head:
            return body
    return payload


def act(
    *,
    frames: dict,
    state: list | None = None,
    instruction: str = "",
    seed: int | None = None,
    timeout: float = REQUEST_TIMEOUT_S,
) -> dict:
    """What would this policy DO on this frame.

    Returns the sidecar's answer with the provenance it carries: the chunk
    itself, plus which policy, which revision, which device, which seed, and
    the normalisation the actions are expressed in.

    None of that is decoration. An action chunk without its policy is a number
    with no owner, and one without its normalisation cannot be drawn against a
    dataset's recorded actions -- the two are in different units and nothing
    in the numbers says so. ROADMAP #50 names that overlay as the
    plausible-wrong output to refuse.

    `seed=None` means "do not touch the RNG", and it is NOT the same as
    `seed=0`. Collapsing them would make every unseeded call silently
    reproducible and hide the fact that most of these policies sample at all.
    """
    port = _read_port()
    if not port:
        raise SidecarError(
            "No policy sidecar is running, so nothing here can say what the "
            "robot would do — only where it looked. `modelmri policy start` "
            "brings one up."
        )
    # The sidecar wants raw base64; the browser wants a data URL. Both are
    # served, and the conversion happens once, here.
    body: dict = {
        "frames": {cam: _raw_base64(v) for cam, v in (frames or {}).items()},
        "instruction": instruction,
    }
    if state is not None:
        body["state"] = list(state)
    # Present only when asked for, so the sidecar can tell "seed 0" from
    # "nobody asked", which are different requests.
    if seed is not None:
        body["seed"] = int(seed)
    data, _ = _post(port, "/act", body, timeout=timeout)
    return data


def hidden(
    *,
    frames: dict,
    layers: list[int],
    state: list | None = None,
    instruction: str = "",
    timeout: float = REQUEST_TIMEOUT_S,
) -> tuple[dict, bytes]:
    """The activations behind an action, as safetensors bytes.

    Bytes rather than JSON because these are tensors: a single layer of a
    vision tower is a few hundred thousand floats, and JSON would spend more
    time formatting them than the forward pass took to produce them.

    Returns `(headers_as_dict, raw_bytes)`. The contract rides in a header for
    the same reason it rides in every JSON body -- a sidecar can be restarted
    underneath a live client, and activations read across a version boundary
    are a different policy's internals wearing this one's name.
    """
    port = _read_port()
    if not port:
        raise SidecarError(
            "No policy sidecar is running, so there are no activations to "
            "read. `modelmri policy start` brings one up."
        )
    body: dict = {
        "frames": frames,
        "instruction": instruction,
        "layers": [int(n) for n in layers],
    }
    if state is not None:
        body["state"] = list(state)
    return _post(port, "/hidden", body, timeout=timeout)
