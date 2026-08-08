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


def _cuda_like() -> Device | None:
    import torch

    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            return None
        props = torch.cuda.get_device_properties(0)
        vram = round(props.total_memory / 1e9, 1)
        is_rocm = bool(getattr(torch.version, "hip", None))
        # bf16 needs Ampere (SM80) or newer on NVIDIA; ROCm reports it directly
        try:
            bf16 = torch.cuda.is_bf16_supported()
        except Exception:
            bf16 = False
        return Device(
            kind="rocm" if is_rocm else "cuda",
            torch_device="cuda:0",
            name=props.name,
            vram_gb=vram,
            dtype="bfloat16" if bf16 else "float16",
            reason=("AMD ROCm GPU detected" if is_rocm else "NVIDIA GPU detected")
            + f" ({vram} GB)",
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
        try:
            props = xpu.get_device_properties(0)
            name = getattr(props, "name", name)
            total = getattr(props, "total_memory", None)
            vram = round(total / 1e9, 1) if total else None
        except Exception:
            pass
        return Device(
            kind="xpu",
            torch_device="xpu:0",
            name=name,
            vram_gb=vram,
            dtype="float16",
            reason="Intel GPU detected",
        )
    except Exception:
        return None


def _mps() -> Device | None:
    import torch

    try:
        backend = getattr(torch.backends, "mps", None)
        if backend is None or not backend.is_available():
            return None
        return Device(
            kind="mps",
            torch_device="mps",
            name="Apple Silicon GPU",
            vram_gb=None,
            dtype="float16",
            reason="Apple Silicon detected",
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
    """
    if prefer == "cpu":
        return _cpu("forced by the caller")

    if prefer not in ("auto", "", None):
        for probe in (_cuda_like, _xpu, _mps):
            found = probe()
            if found and found.torch_device.split(":")[0] == prefer.split(":")[0]:
                found.torch_device = prefer
                found.reason = f"requested explicitly ({prefer})"
                return found
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
