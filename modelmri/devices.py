# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

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
import sys
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


def _ram_posix() -> int | None:
    """Linux and most BSDs, through sysconf."""
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        return None
    return pages * page_size if pages > 0 and page_size > 0 else None


def _ram_windows() -> int | None:
    """GlobalMemoryStatusEx, through ctypes. No third-party dependency."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes

        class _MemStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemStatus()
        status.dwLength = ctypes.sizeof(_MemStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys)
    except Exception:
        return None
    return None


def _ram_macos() -> int | None:
    """hw.memsize, through sysctl."""
    if sys.platform != "darwin":
        return None
    try:
        import subprocess

        out = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return None
    return (
        int(out.stdout.strip())
        if out.returncode == 0 and out.stdout.strip().isdigit()
        else None
    )


def _ram_bytes() -> int | None:
    """Physical RAM, or None. Three platforms, no third-party dependency.

    Returning None is a real answer here. A machine whose RAM cannot be read is
    not a machine with 8 GB, and every sentence downstream is written to cope
    with not knowing.

    Split into three probes rather than one function with fall-through
    `except: pass` arms: the fall-through WAS the handling, but a reader — and
    a static analyser — cannot tell that from a swallowed error, and it read as
    three empty excepts in a row.
    """
    for probe in (_ram_posix, _ram_windows, _ram_macos):
        got = probe()
        if got:
            return got
    return None


def _system_ram_gb() -> float | None:
    """System RAM in GB, or None. Shared with the capability report rather
    than reimplemented: one reader, three platforms, no extra dependency."""
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


def _free_total(device: Device) -> tuple[int | None, int | None]:
    """Bytes free and total on `device`, right now, or `(None, None)`.

    Separate from `Device` on purpose. Everything on that dataclass is a
    PROPERTY of the hardware — its name, its total memory, whether it does
    bf16 — and stays true for as long as the machine is on. Free memory is a
    MEASUREMENT, true at the instant it was read and false as soon as another
    process allocates. Storing it on `Device` would let a cached instance
    carry a number that was right an hour ago and is quoted as if it were now.

    `None` is UNKNOWN and never 0. Apple's unified memory and Intel's XPU have
    no equivalent of `mem_get_info`, and reporting 0 free on a Mac would say
    the machine is out of memory when nobody asked it.
    """
    try:
        import torch
    except Exception:
        return None, None

    if device.kind in ("cuda", "rocm"):
        try:
            index = int(device.torch_device.partition(":")[2] or 0)
            free, total = torch.cuda.mem_get_info(index)
            return int(free), int(total)
        except Exception:
            # A driver that answers `get_device_properties` and refuses
            # `mem_get_info` is a real configuration, not an impossible one.
            # Total is still known from the properties read; free is not.
            total = int(device.vram_gb * 1e9) if device.vram_gb else None
            return None, total

    if device.kind == "cpu":
        total = _ram_bytes() or None
        # Free system RAM needs a dependency this project does not take, and
        # "available" on an OS with a page cache is a judgement rather than a
        # reading. Total is honest; free stays unknown.
        return None, total

    total = int(device.vram_gb * 1e9) if device.vram_gb else None
    return None, total


@dataclass
class DeviceOption:
    """One device the user could send a model to, and what it costs.

    `is_default` marks what `detect("auto")` would pick if nobody chose — so a
    picker can show the current behaviour as the default rather than
    re-deriving the preference order and getting a different answer.
    """

    device: Device
    free_bytes: int | None
    total_bytes: int | None
    is_default: bool

    def to_dict(self) -> dict:
        d = self.device.to_dict()
        d["id"] = self.device.torch_device
        d["free_bytes"] = self.free_bytes
        d["total_bytes"] = self.total_bytes
        d["is_default"] = self.is_default
        return d


def available() -> list[dict]:
    """EVERY device on this machine, not just the one that would be chosen.

    `detect()` answers "where should this go", which is the right question
    right up until somebody wants to answer it themselves — a second GPU they
    keep free for something else, a card another process is already filling,
    or a deliberate run on CPU to compare against. None of those are reachable
    when the only thing the tool will say is which device it picked.

    Every CUDA index is probed individually rather than reporting "cuda" once.
    A machine with two cards has two different names, two different VRAM
    figures and two different amounts free, and collapsing them into one row
    hides the entire reason somebody opened this list.

    Default behaviour is unchanged: the row `detect("auto")` would have picked
    is flagged `is_default`, and choosing nothing still goes there.
    """
    found: list[Device] = []

    try:
        import torch

        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        # No torch, or a torch that cannot answer. CPU is still a real answer
        # and the list must never come back empty — an empty picker reads as
        # "this machine has no devices", which is never true.
        count = 0

    for index in range(count):
        card = _cuda_like(index)
        if card is not None:
            found.append(card)

    for probe in (_xpu, _mps):
        try:
            other = probe()
        except Exception:
            other = None
        if other is not None:
            found.append(other)

    found.append(_cpu("always available"))

    try:
        default = detect("auto").torch_device
    except Exception:
        default = "cpu"

    rows: list[dict] = []
    for device in found:
        free, total = _free_total(device)
        rows.append(
            DeviceOption(
                device=device,
                free_bytes=free,
                total_bytes=total,
                is_default=device.torch_device == default,
            ).to_dict()
        )

    # If nothing matched the default — a card that vanished between the two
    # probes, or a `MODELMRI_DEVICE` naming something absent — say so by
    # marking CPU rather than leaving every row unflagged. A list where
    # nothing is the default is a list that cannot explain what happens when
    # you choose nothing.
    if not any(r["is_default"] for r in rows):
        for row in rows:
            if row["kind"] == "cpu":
                row["is_default"] = True
                break

    return rows
