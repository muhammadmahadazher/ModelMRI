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
import urllib.error
import urllib.request
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
