"""Accelerator detection — use the GPU when there is one, fall back cleanly.

Vendor coverage, in the order we prefer them:

  cuda  NVIDIA (and AMD ROCm, which reports itself through torch.cuda with
        torch.version.hip set — same API surface, different backend)
  xpu   Intel Arc / Data Center GPU Max, via torch.xpu
  mps   Apple Silicon
  cpu   always available

Everything is probed defensively: a broken driver, a CPU-only wheel, or a
GPU too small for the model must degrade to CPU rather than crash. Nothing
here imports anything heavier than torch.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass


@dataclass
class Device:
    kind: str  # cuda | rocm | xpu | mps | cpu
    torch_device: str  # what to pass to .to()
    name: str
    vram_gb: float | None
    dtype: str  # float16 | bfloat16 | float32
    reason: str  # why this one was chosen (shown in the UI)

    def to_dict(self) -> dict:
        return asdict(self)


def _cuda_like(index: int | None = None) -> Device | None:
    """The CUDA/ROCm device, optionally a NAMED one.

    `index` exists because asking for `cuda:1` used to probe whichever card
    was current and then overwrite only the device STRING -- so the panel
    showed card 0's name and VRAM while the model loaded onto card 1. That is
    the same failure the comment below describes for the default path, and
    the fix landed there and not here.
    """
    import torch

    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            return None
        if index is not None and not 0 <= index < torch.cuda.device_count():
            return None
        # torch's own current device, not device 0. On a multi-GPU box the two
        # differ the moment anything sets CUDA_VISIBLE_DEVICES or calls
        # `set_device` — and this tool then reported card 0's name and VRAM
        # while loading onto a different card. Every number on screen would
        # describe hardware the model was not running on, which is worse than
        # no number at all.
        index = torch.cuda.current_device() if index is None else index
        props = torch.cuda.get_device_properties(index)
        vram = round(props.total_memory / 1e9, 1)
        is_rocm = bool(getattr(torch.version, "hip", None))
        # bf16 needs Ampere (SM80) or newer on NVIDIA; ROCm reports it directly
        try:
            bf16 = torch.cuda.is_bf16_supported()
        except Exception:
            bf16 = False
        return Device(
            kind="rocm" if is_rocm else "cuda",
            torch_device=f"cuda:{index}",
            name=props.name,
            vram_gb=vram,
            dtype="bfloat16" if bf16 else "float16",
            reason=("AMD ROCm GPU detected" if is_rocm else "NVIDIA GPU detected")
            + f" ({vram} GB)"
            + (f", device {index}" if index else ""),
        )
    except Exception:
        return None


def _xpu() -> Device | None:
    import torch

    try:
        xpu = getattr(torch, "xpu", None)
        if xpu is None or not xpu.is_available():
            return None
        name = "Intel GPU"
        vram = None
        # Defaulted before the try, because the except below falls through to
        # a Device() that needs it: an Intel GPU we cannot describe is still an
        # Intel GPU, and losing it to a NameError would be worse than a
        # cosmetic label.
        index = 0
        try:
            index = xpu.current_device()
            props = xpu.get_device_properties(index)
            name = getattr(props, "name", name)
            total = getattr(props, "total_memory", None)
            vram = round(total / 1e9, 1) if total else None
        except (AttributeError, AssertionError, RuntimeError):
            # Asking a driver about a device it may not really have. Measured
            # on this machine (torch 2.11.0+cu128, CUDA build, no Intel GPU):
            # `get_device_properties(0)` raises AssertionError, "Torch not
            # compiled with XPU enabled". That path is normally unreachable
            # because `is_available()` above returns False first (verified: it
            # returns False without raising, device_count() == 0). What
            # actually arrives here is a machine that HAS an Intel GPU where
            # `torch.xpu._lazy_init()` fails on a driver or level-zero problem
            # (RuntimeError), or an older torch whose `torch.xpu` has no
            # `get_device_properties` (AttributeError). Those two are read
            # from torch's source, not observed — there is no Intel GPU here
            # to observe them on, which is the reason all three types stay.
            #
            # Continuing is the point: an Intel GPU we cannot describe is
            # still an Intel GPU, and the caller gets it with name="Intel GPU"
            # and vram_gb=None rather than being dropped to CPU over a
            # cosmetic label. Err generous with this tuple for the same
            # reason — anything it misses hits the outer `except Exception`,
            # which returns None and loses the GPU entirely.
            pass
        return Device(
            kind="xpu",
            torch_device=f"xpu:{index}",
            name=name,
            vram_gb=vram,
            dtype="float16",
            reason="Intel GPU detected",
        )
    except Exception:
        return None


def _system_ram_gb() -> float | None:
    """System RAM in GB, or None. Shared with the capability report rather
    than reimplemented: one reader, three platforms, no extra dependency."""
    from .doctor import _ram_bytes

    b = _ram_bytes()
    return round(b / 1e9, 1) if b else None


def _mps() -> Device | None:
    import torch

    try:
        backend = getattr(torch.backends, "mps", None)
        if backend is None or not backend.is_available():
            return None
        # Unified memory: on Apple Silicon the GPU addresses system RAM, so
        # there is no separate VRAM figure — but reporting None meant every
        # consumer downstream treated the machine as having no memory budget
        # at all. `capacity.guard` collapsed it to 0 GB and `doctor` printed no
        # size estimate on exactly the machines that most need one. The honest
        # number is the system's, labelled as what it is.
        unified = _system_ram_gb()
        return Device(
            kind="mps",
            torch_device="mps",
            name="Apple Silicon GPU",
            vram_gb=unified,
            dtype="float16",
            reason=(
                f"Apple Silicon detected, {unified:.1f} GB unified memory"
                if unified
                else "Apple Silicon detected"
            ),
        )
    except Exception:
        return None


def _cpu(reason: str = "no GPU detected") -> Device:
    return Device(
        kind="cpu",
        torch_device="cpu",
        name="CPU",
        vram_gb=None,
        dtype="float32",  # fp16 on CPU is slower than fp32 almost everywhere
        reason=reason,
    )


def detect(prefer: str = "auto") -> Device:
    """Pick the best available device.

    prefer="auto" (default) walks cuda/rocm -> xpu -> mps -> cpu.
    prefer="cpu" forces CPU. Any other value is treated as an explicit
    torch device string and used verbatim if that backend is available.

    `MODELMRI_DEVICE` overrides the default, and exists because the remedies
    printed elsewhere needed something true to point at. The float32-only
    refusal used to say `CUDA_VISIBLE_DEVICES=`, which is NVIDIA's variable
    and does nothing on Apple Silicon, an Intel GPU or ROCm — the reader on
    any of those was told to run a command that could not work. One variable,
    every backend.
    """
    if prefer in ("auto", "", None):
        prefer = os.environ.get("MODELMRI_DEVICE", "auto").strip() or "auto"

    if prefer == "cpu":
        return _cpu(
            "forced to CPU by MODELMRI_DEVICE"
            if os.environ.get("MODELMRI_DEVICE", "").strip() == "cpu"
            else "forced by the caller"
        )

    if prefer not in ("auto", "", None):
        # An explicit index is READ, not pasted on. `cuda:1` used to probe
        # whichever card was current, keep that card's name and VRAM, and
        # overwrite the device string alone -- so every number on screen, and
        # every capacity check that reads `vram_gb`, described a different
        # card from the one about to be loaded. `_cuda_like` records the same
        # failure for the default path: "worse than no number at all".
        head, _, tail = prefer.partition(":")
        wanted = int(tail) if tail.isdigit() else None
        for probe in (_cuda_like, _xpu, _mps):
            found = _cuda_like(wanted) if probe is _cuda_like else probe()
            if found and found.torch_device.split(":")[0] == head:
                # Only when the probe could not honour the index itself.
                # `_cuda_like` returns the card it actually read, so trust it.
                if found.torch_device.split(":")[0] != prefer.split(":")[0]:
                    continue
                if probe is not _cuda_like:
                    found.torch_device = prefer
                found.reason = f"requested explicitly ({found.torch_device})"
                return found
        if head in ("cuda", "rocm") and wanted is not None:
            return _cpu(
                f"{prefer} was requested and there is no CUDA device at index {wanted}"
            )
        return _cpu(f"{prefer} was requested but is not available")

    for probe in (_cuda_like, _xpu, _mps):
        found = probe()
        if found is not None:
            return found

    # A GPU may be physically present while torch was installed CPU-only —
    # that is the most common "why isn't it using my GPU" case, so say so.
    return _cpu(_cpu_reason())


def _nvidia_present() -> bool:
    """Is there an NVIDIA GPU here that torch simply cannot talk to?"""
    import shutil
    import subprocess

    if not shutil.which("nvidia-smi"):
        return False
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=4,
        )
        return out.returncode == 0 and bool(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return False


def _cpu_reason() -> str:
    """Why we are on the CPU — and, when it is fixable, the exact command.

    "reinstall it with GPU support (see the README)" is not an instruction.
    The owner of an RTX 4060 sat on CPU-only torch reading that message. If a
    GPU is physically present we now name it and print the line to paste.
    """
    import sys

    import torch

    if getattr(torch.version, "cuda", None) or getattr(torch.version, "hip", None):
        return "no usable GPU found (driver or device unavailable)"

    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    if _nvidia_present():
        return (
            "an NVIDIA GPU is present but torch is a CPU-only build "
            f"({torch.__version__}). Install a CUDA build for this Python "
            f"({tag}):  pip install --index-url "
            "https://download.pytorch.org/whl/cu128 --force-reinstall torch"
        )
    return (
        f"torch is a CPU-only build ({torch.__version__}) and no NVIDIA GPU "
        "was detected. If you have one, check its driver with nvidia-smi; "
        "otherwise this is expected and everything still works, slower."
    )


def torch_dtype(device: Device):
    import torch

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[device.dtype]
