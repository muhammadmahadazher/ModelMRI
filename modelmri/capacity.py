"""Will this download fit, and could it ever load? One answer, one place.

This module exists because of a specific incident: a click on
`zai-org/GLM-5.2` in the model picker began fetching 1506.7 GB onto a laptop
with an 8.6 GB GPU and 88 GB of free disk. Nothing warned, nothing asked, and
the only way to stop it was to kill the server process.

Every download path goes through `guard()` — HuggingFace and Ollama both —
so there is one rule rather than one rule per source that drift apart. It is
enforced on the server, not in the browser: a check the client performs is a
check the client can skip.

Two thresholds, deliberately different in kind:

* **Disk.** If it does not fit, it is refused and there is no override. There
  is no version of filling someone's disk that ends well, and "I know what
  I'm doing" does not create free space.
* **Accelerator.** If it fits on disk but dwarfs the GPU, it is refused with
  an override, because CPU offload and quantised loads are legitimate. The
  ceiling is generous on purpose: a guard that fires during ordinary work is
  a guard people learn to click through.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

# Below this, an oversized model is nobody's emergency: even a wrong answer
# costs a few minutes. The accelerator rule only starts mattering above it.
MIN_INTERESTING_GB = 20.0

# How much bigger than VRAM a download may be before we ask. Four is roughly
# the point past which no amount of offloading rescues the load.
VRAM_MULTIPLE = 4.0


log = logging.getLogger(__name__)


class TooBig(ValueError):
    """Refused before anything was downloaded, with both numbers named."""

    def __init__(self, message: str, *, overridable: bool) -> None:
        super().__init__(message)
        self.overridable = overridable
        # The same contract `Refusal` and `BadRequest` carry. This is not one
        # of them — it subclasses plain `ValueError`, which is exactly why the
        # image route needed its own `except` arm — but it is published the
        # same way, so it has to be published through the same field. A
        # handler that catches both and reads one differently is the seam
        # where one of the two stops being checked.
        self.sentence = str(message)

    def __reduce__(self):
        """Survive `copy`, `deepcopy` and `pickle`.

        `BaseException.__reduce__` rebuilds an exception by calling the class
        with `self.args` — positionally. `overridable` is keyword-ONLY, so it
        is not in `args`, and every one of the three raised:

            TypeError: TooBig.__init__() missing 1 required keyword-only
            argument: 'overridable'

        A refusal that cannot be copied is a refusal that dies on any path
        which moves it between contexts, and the failure arrives as a
        confusing TypeError about the exception rather than the sentence the
        exception was carrying.
        """
        return (
            _rebuild_too_big,
            (self.args[0] if self.args else "", self.overridable),
        )


def _rebuild_too_big(message: str, overridable: bool) -> TooBig:
    """Module level so `pickle` can find it by name — a closure or a lambda
    could not be pickled either, which would move the problem rather than fix
    it."""
    return TooBig(message, overridable=overridable)


def free_space(target: Path) -> tuple[Path, int]:
    """(the volume we would write into, its free bytes). 0 when unknowable.

    Walks up to the nearest directory that exists — on a fresh machine the
    cache has not been created yet, and `disk_usage` on a missing path raises
    rather than answering about its volume.
    """
    probe = Path(target)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return probe, shutil.disk_usage(probe).free
    except OSError:
        return probe, 0


def ollama_models_dir() -> Path:
    """Where Ollama stores blobs, honouring OLLAMA_MODELS like Ollama does.

    Not the same volume as the HuggingFace cache in general, so the disk
    check has to be told which one it is talking about.
    """
    from . import paths

    if raw := (os.environ.get("OLLAMA_MODELS") or "").strip():
        # paths._expand, not a local try/except: this caught (OSError,
        # ValueError) and missed the RuntimeError that `~` with no home
        # raises on Windows. See its docstring.
        if resolved := paths._expand(raw):
            return resolved

    home = paths._home()
    return (home / ".ollama" / "models") if home else Path(".ollama") / "models"


def _human(gb: float) -> str:
    return f"{gb / 1000:,.1f} TB" if gb >= 1000 else f"{gb:,.1f} GB"


def guard(
    need_bytes: int,
    target: Path,
    *,
    label: str,
    vram_gb: float | None,
    accel_name: str = "",
    confirm: bool = False,
    free_override: int | None = None,
) -> None:
    """Raise TooBig unless this download can fit and could plausibly load.

    `need_bytes` of 0 means "the source published nothing to go on" — GGUF
    and pickle repos, mostly. That is treated as unknown and allowed through,
    never as small: refusing on no evidence would ban legitimate models, and
    pretending unknown means zero is how a guard lets past the one download
    it existed to stop.
    """
    if need_bytes <= 0:
        return

    need_gb = need_bytes / 1e9
    volume, measured = free_space(target)
    # `free_override` lets a caller state the disk situation — used by tests,
    # so a verdict does not depend on how full the developer's drive is.
    free = measured if free_override is None else free_override

    if not free:
        # `free_space` returns 0 for a volume it could not measure, and the
        # refusal below is correctly SKIPPED — refusing on no evidence would
        # ban a legitimate download, the same argument `need_bytes <= 0` makes
        # above. What was missing is that nobody was told the check did not
        # happen, so a download proceeded looking exactly like one that had
        # been cleared. The terminal is where this project already puts what a
        # reader needs and a response should not carry.
        # `%r`, not `%s`. `label` is a repo id and `target` a path, and both
        # arrive from a request — a newline in either forges a log line, which
        # is what CodeQL's log-injection rule is about and it is right. `repr`
        # escapes them, and it also makes a trailing space or a zero-width
        # character visible rather than invisible in a log somebody is reading
        # to work out what happened.
        log.warning(
            "disk space could not be measured for %r (%r), so %r was NOT "
            "checked against free space before downloading",
            str(volume),
            str(target),
            str(label),
        )

    if free and need_bytes > free:
        where = volume.drive or volume
        raise TooBig(
            f"{label} needs {_human(need_gb)} and {where} has "
            f"{_human(free / 1e9)} free. Pick a smaller model, or point the "
            f"cache at a drive with room.",
            overridable=False,
        )

    # `vram_gb or 0` collapsed "we could not read it" into "there is none",
    # so an accelerator whose properties are unreadable got the same ceiling as
    # a machine with no GPU. They are different states and the message below
    # says different things about them.
    vram = vram_gb if vram_gb is not None else 0.0
    ceiling = max(VRAM_MULTIPLE * vram, MIN_INTERESTING_GB)
    if need_gb > ceiling and not confirm:
        # THREE states, not two. The comment above says these are different
        # and the code still collapsed them: `vram` is 0.0 for an unreadable
        # accelerator AND for a machine with no GPU, so branching on its
        # truthiness printed "this machine has no GPU" at somebody whose GPU
        # is sitting right there and merely did not answer -- sending them to
        # buy hardware they own, over a driver or permissions problem.
        #
        # `vram_gb is None` alone does NOT separate them, which is the part
        # worth being careful about: `devices._cpu()` returns None for a real
        # CPU-only machine, and an Intel XPU whose properties could not be
        # read returns None too, deliberately, with its name intact ("an Intel
        # GPU we cannot describe is still an Intel GPU"). The NAME is what
        # tells them apart.
        named_accelerator = bool(accel_name) and accel_name.strip().lower() not in (
            "cpu",
            "none",
        )
        if vram:
            machine = f"{accel_name or 'your GPU'} has {vram:,.1f} GB"
        elif vram_gb is None and named_accelerator:
            machine = (
                f"the amount of memory on {accel_name} could not be read, so "
                f"this is being judged as if there were none"
            )
        else:
            machine = "this machine has no GPU"
        raise TooBig(
            f"{label} is {_human(need_gb)} to download, and {machine}. It "
            f"would take a very long time and then fail to load. Load it "
            f"anyway only if you know what you are doing.",
            overridable=True,
        )
