"""What can this machine actually do?

Written because "it should not finish installing on machines that cannot run
it" is a reasonable thing to want and a hard thing to do honestly.

WHY THIS IS NOT AN INSTALL HOOK. A wheel is an archive. pip does not execute
code from it at install time -- that is the difference between a wheel and an
sdist, and it is deliberate. So there is no supported place to put a check that
runs during `pip install modelmri`. Anything claiming otherwise either ships an
sdist-only `setup.py` (skipped for the wheel, which is what almost everyone
gets) or abuses an entry point.

The honest version is a check that runs the first time you ask the tool to do
something -- `modelmri serve` prints it, and `modelmri doctor` prints it on
demand. That is also the better place for it: the same machine may be perfectly
able to open a `.mri` recording and unable to load a 7B model, and those are
different answers to different questions.

WHAT IT REFUSES TO GUESS. Every figure here is read off this machine at the
moment you ask. Where a number cannot be determined it says so rather than
substituting a default -- an invented RAM figure is worse than none, because
the sentence built on it reads exactly as confidently.
"""

from __future__ import annotations

import os
import platform
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Roughly what a checkpoint of N parameters occupies, per parameter, at the
# precision the runtime would pick. Not a guess: bfloat16 and float16 are two
# bytes by definition, and `devices.torch_dtype` chooses one of them for every
# accelerator it recognises. CPU falls back to float32, which is four.
_BYTES_PER_PARAM = {"float32": 4, "bfloat16": 2, "float16": 2}

# The headroom a load needs beyond the weights themselves: activations, the KV
# cache, and the allocator's own fragmentation. Measured against the sizes this
# tool actually loads rather than derived -- see the capacity guard, which uses
# the same shape of reasoning for disk.
_OVERHEAD = 1.35


@dataclass
class Report:
    """Everything measured, plus what it implies. Nothing here is a default."""

    os_name: str = ""
    os_version: str = ""
    arch: str = ""
    python: str = ""
    cpu_count: int | None = None
    ram_gb: float | None = None
    disk_free_gb: float | None = None
    disk_path: str = ""
    torch_version: str | None = None
    torch_build: str | None = None
    accelerator: str = ""
    accelerator_kind: str = ""
    vram_gb: float | None = None
    dtype: str = ""
    accelerator_reason: str = ""
    can_view_recordings: bool = True
    can_load_models: bool = False
    largest_model_b: float | None = None
    notes: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)


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


def _largest_model_b(budget_gb: float | None, dtype: str) -> float | None:
    """Billions of parameters that fit in `budget_gb`, at this dtype."""
    if not budget_gb or budget_gb <= 0:
        return None
    per_param = _BYTES_PER_PARAM.get(dtype, 4)
    return round(budget_gb * 1e9 / (per_param * _OVERHEAD) / 1e9, 1)


def check() -> Report:
    """Measure this machine. Never raises: a report is better than a traceback."""
    r = Report(
        os_name=platform.system() or "unknown",
        os_version=platform.release() or "",
        arch=platform.machine() or "unknown",
        python=platform.python_version(),
        cpu_count=os.cpu_count(),
    )

    ram = _ram_bytes()
    r.ram_gb = round(ram / 1e9, 1) if ram else None

    try:
        from . import paths

        target = paths.hf_hub_cache()
        probe = target
        while not probe.exists() and probe.parent != probe:
            probe = probe.parent
        r.disk_path = str(target)
        r.disk_free_gb = round(shutil.disk_usage(probe).free / 1e9, 1)
    except Exception as err:  # a report, not a load
        r.notes.append(f"could not measure free disk space ({type(err).__name__})")

    # torch is the dividing line. Without it the viewer still works and nothing
    # else does, which is exactly the distinction this report exists to draw.
    try:
        import torch

        r.torch_version = torch.__version__
        cuda = getattr(getattr(torch, "version", None), "cuda", None)
        hip = getattr(getattr(torch, "version", None), "hip", None)
        r.torch_build = (
            f"cuda {cuda}" if cuda else f"rocm {hip}" if hip else "cpu-only wheel"
        )
    except ImportError:
        r.blockers.append(
            "PyTorch is not installed, so no model can be loaded. `pip install "
            "modelmri` pulls it in; if you installed with --no-deps, that is why."
        )
        r.notes.append(
            "Opening a shared .mri recording still works without PyTorch — that "
            "is deliberate, and it is why this is a warning rather than a "
            "refusal to install."
        )
        return r

    try:
        from . import devices

        dev = devices.detect()
        r.accelerator = dev.name
        r.accelerator_kind = dev.kind
        r.vram_gb = dev.vram_gb
        r.dtype = dev.dtype
        r.accelerator_reason = dev.reason
    except Exception as err:  # same
        r.notes.append(f"could not detect an accelerator ({type(err).__name__})")
        return r

    r.can_load_models = True
    # On an accelerator the ceiling is VRAM; on CPU it is RAM, and the runtime
    # uses float32 there, so the same machine fits half as much.
    # An accelerator that reports no VRAM figure is not an accelerator with no
    # memory. Apple Silicon has unified memory and Intel XPU properties are not
    # always readable, and both used to produce no size estimate at all — on
    # the machines least able to guess for themselves.
    budget = r.ram_gb if r.accelerator_kind == "cpu" else (r.vram_gb or r.ram_gb)
    r.largest_model_b = _largest_model_b(budget, r.dtype or "float32")

    if r.accelerator_kind == "cpu":
        r.notes.append(
            "No accelerator was found, so models run on the CPU. That works and "
            "it is slow — expect seconds per token rather than tokens per "
            "second, and prefer the smallest models."
        )
    if r.disk_free_gb is not None and r.disk_free_gb < 5:
        r.blockers.append(
            f"Only {r.disk_free_gb} GB free where models are cached "
            f"({r.disk_path}). Most models will not finish downloading."
        )
    if budget is not None and budget < 2:
        r.blockers.append(
            f"Only {budget} GB available to hold a model. The smallest models "
            f"this tool ships defaults for need about 1 GB, so this machine is "
            f"at or below the floor."
        )
    return r


def render(r: Report) -> str:
    """The report as a human reads it. Same text for the CLI and the server."""
    n = lambda v, unit="": f"{v}{unit}" if v is not None else "could not measure"  # noqa: E731

    lines = [
        "ModelMRI — what this machine can do",
        "",
        f"  os          {r.os_name} {r.os_version} ({r.arch})",
        f"  python      {r.python}",
        f"  cpu         {n(r.cpu_count)} logical cores",
        f"  ram         {n(r.ram_gb, ' GB')}",
        f"  disk        {n(r.disk_free_gb, ' GB')} free at {r.disk_path or 'unknown'}",
    ]
    if r.torch_version:
        lines.append(f"  torch       {r.torch_version} ({r.torch_build})")
        lines.append(f"  accelerator {r.accelerator or 'none'} ({r.accelerator_kind})")
        if r.vram_gb is not None:
            lines.append(f"  vram        {r.vram_gb} GB")
        if r.dtype:
            lines.append(f"  precision   {r.dtype}")
        if r.accelerator_reason:
            lines.append(f"              {r.accelerator_reason}")
    else:
        lines.append("  torch       not installed")

    lines.append("")
    if r.can_load_models and r.largest_model_b:
        lines.append(
            f"  Models up to roughly {r.largest_model_b}B parameters should fit. "
            f"That is arithmetic on the numbers above, not a promise: a model "
            f"with a long context or an unusual architecture can need more."
        )
    elif r.can_load_models:
        lines.append(
            "  A model can be loaded, but the memory ceiling could not be "
            "measured, so there is no size estimate to give you."
        )
    lines.append(
        "  Opening a shared .mri recording works on any machine, with or "
        "without PyTorch and with or without a GPU."
    )

    for note in r.notes:
        lines.append(f"\n  note     {note}")
    for blocker in r.blockers:
        lines.append(f"\n  PROBLEM  {blocker}")
    if not r.blockers:
        lines.append("\n  No blockers found.")
    return "\n".join(lines)


def one_line(r: Report) -> str:
    """The startup banner's version: what matters, in a sentence."""
    if not r.torch_version:
        return "no PyTorch — .mri recordings will open, models will not load"
    where = r.accelerator or "cpu"
    size = f", fits ~{r.largest_model_b}B" if r.largest_model_b else ""
    warn = (
        f"  [{len(r.blockers)} problem(s), run `modelmri doctor`]" if r.blockers else ""
    )
    return f"{where} · {r.dtype or 'unknown precision'}{size}{warn}"


def write_to(stream=None) -> int:
    """Print the report. Exit code 1 when something would stop a load."""
    r = check()
    print(render(r), file=stream or sys.stdout)
    return 1 if r.blockers else 0


def cache_marker() -> Path | None:
    """Where the once-per-install notice records that it has been shown."""
    try:
        from . import paths

        return paths.data_dir() / "capability-notice"
    except Exception:  # the notice is best effort
        return None
