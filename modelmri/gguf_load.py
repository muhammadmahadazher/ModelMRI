# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""Run a GGUF through the introspection stack — and say what that costs.

`gguf_read.py` reads the header. This loads the weights, which is a different
job with a different price, and the price is the reason this module exists at
all rather than being three lines inside `runtime.py`.

The belief worth destroying: that a 4-bit GGUF loads as a 4-bit model. It does
not. Transformers has no 4-bit kernels for these types — it dequantises every
tensor back to full precision on the way in and hands PyTorch ordinary dense
weights. So the file on disk stops being a useful guide to what the load will
need, in both directions:

    Qwen3-0.6B-Q4_K_M.gguf         file 0.397 GB -> resident 1.192 GB (3.00x)
    SmolLM2-135M-Q4_K_M.gguf       file 0.105 GB -> resident 0.269 GB (2.55x)

Both produced by `scripts/measure_docs.py --gguf FILE`, RTX 4060 / bfloat16,
transformers 5.13. Run it and you get these back; that is the point.

The resident figure is `parameters x dtype bytes` and it is EXACT — predicted
from the header against the built module weighed afterwards, error 0.000000 on
both files, because element counts come from the tensor shapes and do not
depend on the quantisation at all.

The peak is a different kind of number and this module used to blur them. The
dequantiser materialises the whole checkpoint in float32 before anything is
cast, so a `dtype=bfloat16` load still transits through `parameters x 4` — but
that is a PREDICTION, and the process RSS that results is a MEASUREMENT, and
they are not the same to the digit:

    SmolLM2-135M    predicted 0.538 GB    sampled RSS delta 0.585 GB   +8.6%
    Qwen3-0.6B      predicted 2.384 GB    sampled RSS delta 2.30  GB   -3.5%

Opposite signs, so there is no correction factor to apply — RSS also carries
the tokeniser and the allocator's own timing, which land differently at 135M
than at 596M. `Loaded` therefore reports both figures and their signed error
rather than picking one. For months the docstring here quoted the 2.30 as
though it were the prediction, and nothing in the repo could reproduce it.

The preflight is still the point: `parameters x 4` is computable from the
first few hundred kilobytes of the file, so the honest answer — "this will not
fit" — is available before the download rather than after it. It is a
projection accurate to about ten percent, which is what a refusal needs.

What a loaded GGUF is NOT is the original model. It is the quantised weights,
dequantised: every number measured on it is a number about the quantised
model, and `quantdiff.py` exists to say how far apart those two are. Nothing
here silently substitutes one for the other, and `notes` says so on every
result.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from .errors import BadRequest, Refusal

# The only dtypes transformers will build a GGUF into, and what each costs per
# parameter. A dict `.get(dtype, 4)` sat here first, which is this project's
# named bug class in its literal form: `dtype="bf16"` -- the spelling everyone
# actually writes -- scored 4 bytes instead of 2, so the preflight reported
# double, could refuse a model that fits, and then died inside transformers
# with an AttributeError this module blamed on transformers. `float64` was
# worse: a valid torch attribute that under-predicted by 2x and waved the load
# through. A guard whose whole purpose is to refuse before a twenty-minute OOM
# must not invent the number it refuses on.
DTYPE_BYTES = {"float32": 4, "bfloat16": 2, "float16": 2}

# Measured, not assumed: see the module docstring. Transformers' GGUF reader
# builds the entire dequantised checkpoint as float32 numpy arrays and only
# then casts to the requested dtype, so the transient high-water mark is set
# by float32 regardless of what you ask for.
DEQUANT_TRANSIT_BYTES_PER_PARAM = 4

# How full a device may be projected to get before this refuses without an
# explicit confirm. Above this the allocator is fighting fragmentation rather
# than running out cleanly, and the failure arrives minutes in.
REFUSE_ABOVE_FRACTION = 0.9

# Files that are part of a model but are not a causal language model. Loading
# one produces a stack trace from deep inside transformers about a missing
# `block_count`; refusing by name produces a sentence.
NOT_A_LANGUAGE_MODEL = {
    "mmproj": "a multimodal projector — the vision tower's adapter, not the "
    "language model. Load the main file instead.",
    "mtp": "a multi-token-prediction head — a speculative-decoding extra, not "
    "the language model. Load the main file instead.",
}


class Unsupported(Refusal):
    """This file cannot become a torch module, and here is which part of it."""


@dataclass
class Plan:
    """What loading this file would cost, computed before loading it."""

    path: str
    architecture: str | None
    parameters: int
    file_bytes: int
    dtype: str
    # parameters x dtype bytes. Exact — element counts come from the tensor
    # shapes and are as known for an unrecognised quantisation as for F32.
    resident_bytes: int
    # parameters x 4. The float32 transit through the dequantiser, which is the
    # figure that decides whether the load survives, not the resident one.
    peak_host_bytes: int
    device: str
    # None when the platform does not report it. Never 0 — "we could not ask"
    # and "there is none left" are different answers and only one of them is a
    # reason to refuse.
    device_free_bytes: int | None
    host_free_bytes: int | None
    # Total, not just free. When the transit figure is above TOTAL host RAM the
    # answer is "no amount of closing things helps", which is a different piece
    # of advice from "you have 3 GB free" — and the second one, alone, sends
    # people off to quit Chrome for twenty minutes.
    host_total_bytes: int | None
    verdict: str  # "fits" | "tight" | "will not fit" | "unknown"
    why: str
    notes: list[str] = field(default_factory=list)

    @property
    def expansion(self) -> float | None:
        """How much bigger the loaded model is than the file. The headline."""
        if not self.file_bytes:
            return None
        return self.resident_bytes / self.file_bytes

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "architecture": self.architecture,
            "parameters": self.parameters,
            "file_bytes": self.file_bytes,
            "dtype": self.dtype,
            "resident_bytes": self.resident_bytes,
            "peak_host_bytes": self.peak_host_bytes,
            "expansion": round(self.expansion, 2) if self.expansion else None,
            "device": self.device,
            "device_free_bytes": self.device_free_bytes,
            "host_free_bytes": self.host_free_bytes,
            "host_total_bytes": self.host_total_bytes,
            "verdict": self.verdict,
            "why": self.why,
            "notes": self.notes,
            "means": (
                "resident_bytes is parameters x dtype bytes and is exact. "
                "peak_host_bytes is parameters x 4: transformers dequantises "
                "the whole checkpoint to float32 in host RAM before casting, "
                "so a bfloat16 load still transits through the float32 figure. "
                "Neither is the file size, and the file size is what people "
                "budget against."
            ),
        }


def _gb(n: int | float | None) -> str:
    return "unknown" if n is None else f"{n / 1e9:.2f} GB"


# ------------------------------------------------------------------ the file


def find_file(target: str | Path) -> Path:
    """The one .gguf to load, or a refusal naming what was there instead.

    Accepts the file itself or a directory holding it. A directory with more
    than one candidate is refused rather than guessed at: repos routinely ship
    Q4_K_M beside Q8_0 beside BF16, and picking for the user means picking
    which quantisation their measurements describe.
    """
    p = Path(target).expanduser()
    if p.is_file():
        found = p
    elif p.is_dir():
        # Sorted so the refusal below lists them in a stable order; a set here
        # made the error message reshuffle between runs.
        candidates = sorted(q for q in p.glob("*.gguf") if q.is_file())
        if not candidates:
            raise BadRequest(f"no .gguf file in {p}")
        if len(candidates) > 1:
            names = ", ".join(q.name for q in candidates[:6])
            more = "" if len(candidates) <= 6 else f", and {len(candidates) - 6} more"
            raise BadRequest(
                f"{p} holds {len(candidates)} GGUF files ({names}{more}). Name "
                "the one you mean — they are different quantisations, and "
                "which one you load is which one your measurements describe."
            )
        found = candidates[0]
    else:
        raise BadRequest(f"no such file or directory: {p}")

    if found.suffix.lower() != ".gguf":
        raise BadRequest(f"{found.name} is not a .gguf file")

    stem = found.stem.lower()
    # The marker must be a WHOLE token, with `-`, `_` or `.` on both sides.
    #
    # Two directions, and the old rule got both wrong. It refused
    # `llama-mtpx-7b` (marker as a mere substring), and it let
    # `mmproj_model_f16.gguf` and `Qwen2.5-VL-7B.mmproj-f16.gguf` through
    # because it only knew `-` as a separator -- both are real spellings, and
    # both then produced the deep transformers stack trace this table exists
    # to replace with a sentence.
    #
    # Note `mtp-tuned-7b.gguf` IS refused, and that is correct rather than a
    # false positive: a leading `mtp` token is precisely how llama.cpp names a
    # multi-token-prediction head. The old comment cited that filename as an
    # example of something that must be allowed, which was wrong about real
    # naming, and the test then dodged the claim by testing `mtpc-7b` instead.
    # What genuinely must pass is a marker that is a PREFIX of a longer token
    # -- `mtpc`, `mtpx` -- and that is what the token split protects.
    parts = re.split(r"[-_.]", stem)
    for marker, why in NOT_A_LANGUAGE_MODEL.items():
        if marker in parts:
            raise Unsupported(f"{found.name} is {why}")

    # Anchored to digits. `gguf-split` always emits `-NNNNN-of-NNNNN`, so the
    # bare substring bought nothing and refused any model whose name happens to
    # contain `-of-` — mixture-of-experts and chain-of-thought finetunes are
    # named that way — while telling them to run `gguf-split --merge` on a file
    # that is not split.
    if re.search(r"-\d+-of-\d+$", stem):
        raise Unsupported(
            f"{found.name} is one shard of a split GGUF. Transformers loads "
            "single-file GGUFs only — merge the shards with llama.cpp's "
            "`gguf-split --merge` first, or download a build that is not split."
        )
    return found


def supported_architectures() -> list[str]:
    """Architectures transformers can turn into a module, asked at runtime.

    Read from the installed transformers rather than copied into a constant
    here: the list grows every release, and a hardcoded copy would refuse
    models that the library in front of it supports perfectly well.
    """
    try:
        from transformers.integrations import ggml
    except Exception:  # pragma: no cover - transformers is a hard dependency
        return []
    mapping = getattr(ggml, "GGUF_CONFIG_MAPPING", {})
    # `general` and `tokenizer` are metadata sections in that table, not model
    # architectures, and listing them in a refusal would send people looking
    # for a "general" model.
    return sorted(k for k in mapping if k not in {"general", "tokenizer"})


def _require_gguf() -> None:
    try:
        import gguf  # noqa: F401
    except Exception as err:
        raise Refusal(
            "Loading a GGUF's weights needs the `gguf` package, which reads "
            "the quantised blocks. Reading the header does not — that is "
            "`modelmri.gguf_read` and it uses the stdlib. Install with:\n"
            "    pip install 'modelmri[gguf]'"
        ) from err


# ------------------------------------------------------------- the preflight


def plan(
    target: str | Path,
    *,
    dtype: str = "bfloat16",
    device_kind: str = "cpu",
    device_free_bytes: int | None = None,
    host_free_bytes: int | None = None,
) -> Plan:
    """Both memory figures and a verdict, without opening the weights.

    `device_free_bytes` and `host_free_bytes` are measured here when not
    supplied. They stay None when the platform will not answer, and a None
    never becomes a 0: a guard that refuses because it could not measure is a
    guard that locks out everyone it cannot see.
    """
    from . import gguf_read

    path = find_file(target)
    g = gguf_read.read(path)
    s = g.summary()
    arch = s["architecture"]
    params = int(s["parameters"])
    if dtype not in DTYPE_BYTES:
        raise BadRequest(
            f"unknown dtype {dtype!r}. This loads GGUFs as "
            f"{', '.join(sorted(DTYPE_BYTES))} -- those are what transformers "
            "will build one into, and the choice is half of the memory figure, "
            "so it is not something to guess at."
        )
    dtype_bytes = DTYPE_BYTES[dtype]

    notes: list[str] = []
    supported = supported_architectures()
    if arch and supported and arch not in supported:
        raise Unsupported(
            f"transformers {_transformers_version()} cannot build a module for "
            f"GGUF architecture '{arch}'. It knows: {', '.join(supported)}.\n"
            "The header still reads — architecture, tensor table, bits per "
            "weight and quantisation damage all work on this file. Only the "
            "forward pass needs an architecture the library implements."
        )
    if not arch:
        notes.append(
            "the file declares no general.architecture, so transformers will "
            "have to guess at the config; if it cannot, the load fails there"
        )

    resident = params * dtype_bytes
    peak = params * DEQUANT_TRANSIT_BYTES_PER_PARAM

    if device_free_bytes is None and device_kind != "cpu":
        from . import budget

        device_free_bytes = budget.free_memory(device_kind).free_bytes
    host_total_bytes = _host_total()
    if host_free_bytes is None:
        host_free_bytes = _host_free()

    # The float32 transit always happens in host RAM, whatever the target
    # device is, because the dequantiser runs before anything moves.
    verdict, why = _verdict(
        resident=resident,
        peak=peak,
        device_kind=device_kind,
        device_free=device_free_bytes,
        host_free=host_free_bytes,
        host_total=host_total_bytes,
    )

    notes.append(
        f"{_gb(g_file_bytes := path.stat().st_size)} on disk becomes "
        f"{_gb(resident)} of {dtype} tensors — {resident / g_file_bytes:.2f}x — "
        f"because transformers has no kernel for these quantised types and "
        f"dequantises every tensor on the way in."
    )
    notes.append(
        "every number measured on this model is a number about the QUANTISED "
        "model. Use quantdiff against the original safetensors to see how far "
        "apart they are."
    )
    if s["why_unmeasured"]:
        notes.append(
            "the parameter count is still exact — it comes from tensor shapes, "
            "not from the quantisation type — but this file's byte totals are "
            f"withheld: {s['why_unmeasured']}"
        )

    return Plan(
        path=str(path),
        architecture=arch,
        parameters=params,
        file_bytes=g_file_bytes,
        dtype=dtype,
        resident_bytes=resident,
        peak_host_bytes=peak,
        device=device_kind,
        device_free_bytes=device_free_bytes,
        host_free_bytes=host_free_bytes,
        host_total_bytes=host_total_bytes,
        verdict=verdict,
        why=why,
        notes=notes,
    )


def _verdict(
    *,
    resident: int,
    peak: int,
    device_kind: str,
    device_free: int | None,
    host_free: int | None,
    host_total: int | None = None,
) -> tuple[str, str]:
    """ "fits" / "tight" / "will not fit" / "unknown", and the sentence for it.

    Two independent gates, because they fail at different moments and a single
    combined number would hide which one is about to break: the float32 transit
    happens in host RAM during the load, and the resident tensors then have to
    sit on the target device afterwards.
    """
    if host_total is not None and peak > host_total:
        # Checked before the free-memory arm, because the advice differs. This
        # machine cannot load this file with every other program closed, and
        # saying "3 GB free" first would send someone off to quit things for
        # twenty minutes to no effect.
        return (
            "will not fit",
            f"dequantising needs about {_gb(peak)} of host RAM (the float32 "
            f"transit) and this machine has {_gb(host_total)} in total. "
            "Closing other programs cannot change that.",
        )
    if host_free is not None and peak > host_free:
        return (
            "will not fit",
            f"dequantising needs about {_gb(peak)} of host RAM (the float32 "
            f"transit) and {_gb(host_free)} is free of "
            f"{_gb(host_total)} total. This fails during the load, before the "
            "model reaches any device.",
        )

    # The float32 transit is host RAM whatever the target device is, so the
    # band applies to it on every path -- not only the CPU one. Without this a
    # CUDA load needing 15.9 GB of transit against 16 GB free reported "fits"
    # while the identical numbers on CPU reported "tight ... may thrash". Same
    # allocation, same risk, two answers.
    if host_free is not None and peak > host_free * REFUSE_ABOVE_FRACTION:
        return (
            "tight",
            f"dequantising needs about {_gb(peak)} of host RAM against "
            f"{_gb(host_free)} free — over {REFUSE_ABOVE_FRACTION:.0%} of what "
            "is left. It may load and it may thrash. The weights themselves "
            f"are only {_gb(resident)}; the transit is the larger figure.",
        )

    if device_kind == "cpu":
        if host_free is None:
            return "unknown", "this platform does not report free host memory."
        # On CPU the transit and the residency share one pool, and the transit
        # is the larger of the two, so it is the whole test.
        if peak > host_free * REFUSE_ABOVE_FRACTION:
            return (
                "tight",
                f"{_gb(peak)} needed against {_gb(host_free)} free — over "
                f"{REFUSE_ABOVE_FRACTION:.0%} of what is left. It may load and "
                "it may thrash.",
            )
        return (
            "fits",
            f"{_gb(peak)} peak against {_gb(host_free)} free host RAM, "
            f"settling to {_gb(resident)}.",
        )

    if device_free is None:
        return (
            "unknown",
            f"{_gb(peak)} of host RAM for the load is available, but this "
            f"{device_kind} device does not report free memory, so whether "
            f"{_gb(resident)} of weights will sit on it is untested.",
        )
    if resident > device_free:
        return (
            "will not fit",
            f"the weights are {_gb(resident)} at this dtype and the "
            f"{device_kind} device has {_gb(device_free)} free. The file being "
            "smaller than that is not the relevant number.",
        )
    if resident > device_free * REFUSE_ABOVE_FRACTION:
        return (
            "tight",
            f"{_gb(resident)} of weights against {_gb(device_free)} free — "
            f"over {REFUSE_ABOVE_FRACTION:.0%} of the device, leaving almost "
            "nothing for activations or attention maps.",
        )
    return (
        "fits",
        f"{_gb(peak)} peak in host RAM during the load, then {_gb(resident)} "
        f"of weights on {_gb(device_free)} of free {device_kind} memory.",
    )


def _host_total() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except Exception:
        return None


def _host_free() -> int | None:
    try:
        import psutil
    except Exception:
        return None
    try:
        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def _transformers_version() -> str:
    try:
        import transformers

        return transformers.__version__
    except Exception:
        return "(not installed)"


# ------------------------------------------------------------------ the load


@dataclass
class Loaded:
    model: object
    tokenizer: object
    plan: Plan
    # What the module actually weighed once built, against what `plan`
    # predicted. Reported rather than trusted: the prediction is arithmetic on
    # a header, and a header can describe a file that does not load the way it
    # says.
    measured_resident_bytes: int
    load_seconds: float
    # Process RSS delta across the load, sampled. None when psutil is absent
    # -- never 0, which would read as "the dequantiser allocated nothing".
    #
    # This exists because the module docstring quoted a measured peak that
    # nothing in the repo could reproduce: `plan` predicts `parameters x 4`
    # and the script printed the prediction, so the measured figure survived
    # only as prose. A number nobody can re-derive is the exact failure this
    # project is about.
    measured_peak_host_bytes: int | None = None

    @property
    def prediction_error(self) -> float | None:
        if not self.plan.resident_bytes:
            return None
        return (
            self.measured_resident_bytes - self.plan.resident_bytes
        ) / self.plan.resident_bytes

    @property
    def peak_error(self) -> float | None:
        """How far `parameters x 4` was from the RSS actually observed.

        Signed, and NOT systematically one sign: +8.6% on SmolLM2-135M
        against -3.5% on Qwen3-0.6B. RSS carries the tokeniser and the
        allocator's own release timing as well as the float32 arrays, and
        those land differently at different model sizes, so there is no
        correction factor to fold into the prediction. Reported rather than
        hidden -- a prediction that is never checked is a guess with a
        decimal point.
        """
        if self.measured_peak_host_bytes is None or not self.plan.peak_host_bytes:
            return None
        return (
            self.measured_peak_host_bytes - self.plan.peak_host_bytes
        ) / self.plan.peak_host_bytes

    def to_dict(self) -> dict:
        err = self.prediction_error
        return {
            "plan": self.plan.to_dict(),
            "measured_resident_bytes": self.measured_resident_bytes,
            "prediction_error": round(err, 4) if err is not None else None,
            "measured_peak_host_bytes": self.measured_peak_host_bytes,
            "peak_error": (
                round(self.peak_error, 4) if self.peak_error is not None else None
            ),
            "load_seconds": round(self.load_seconds, 2),
        }


def load(
    target: str | Path,
    *,
    dtype: str = "bfloat16",
    device: str = "cpu",
    device_kind: str | None = None,
    confirm: bool = False,
    on_stage=None,
) -> Loaded:
    """Load a GGUF as a real torch module, refusing first if it will not fit.

    `confirm=True` overrides a "tight" verdict and nothing else. "will not fit"
    is arithmetic, not a preference: the host RAM needed for the float32
    transit is larger than the host RAM that exists, and confirming does not
    create any.
    """
    import time

    import torch

    # `plan` FIRST, `_require_gguf` after. The other order put a check on an
    # optional dependency in front of plain argument validation, so
    # `dtype="float64"` on a machine without `modelmri[gguf]` reported "install
    # the gguf extra" instead of "unknown dtype" -- true, but not the thing
    # that was wrong, and the user fixes it and hits the real error second.
    #
    # It also made the answer depend on which extras happened to be installed:
    # the same request returned 422 here and 409 in CI, which is how this was
    # found. `plan` needs nothing beyond the stdlib header reader.
    kind = device_kind or (device.split(":")[0] if device else "cpu")
    p = plan(target, dtype=dtype, device_kind=kind)
    _require_gguf()

    if p.verdict == "will not fit":
        raise Refusal(
            f"{Path(p.path).name} will not load here: {p.why}\n"
            f"The file is {_gb(p.file_bytes)}; the loaded model is "
            f"{_gb(p.resident_bytes)}. Transformers dequantises GGUF weights "
            "— it has no kernel for these types — so the disk figure is not "
            "the memory figure.\n"
            "For chat only, the ollama backend runs this file at its real bit "
            "width. Full introspection needs the dequantised weights to fit."
        )
    if p.verdict == "tight" and not confirm:
        raise Refusal(
            f"{Path(p.path).name} is a tight fit: {p.why}\n"
            "Load it anyway with confirm=True, or pick a smaller build."
        )

    if on_stage:
        on_stage("dequantise", f"{p.parameters:,} parameters from {p.dtype} GGUF")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = Path(p.path)
    directory, filename = str(path.parent), path.name

    # Sample RSS across the load so the transit prediction can be checked
    # rather than asserted. A background sampler rather than a before/after
    # pair, because the float32 arrays are freed as the cast proceeds -- the
    # high-water mark is in the middle and both endpoints miss it.
    sampler = _RssSampler()
    sampler.start()
    t0 = time.perf_counter()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            directory,
            gguf_file=filename,
            dtype=getattr(torch, dtype),
            # Not optional here. sdpa returns attentions=None without saying
            # so, and half of this tool is attention maps — a GGUF loaded with
            # the default kernel would silently have no attention to show.
            attn_implementation="eager",
        )
    except Exception as err:
        raise Refusal(
            f"transformers could not build a model from {filename}: "
            f"{type(err).__name__}. The header reads fine — architecture, "
            "tensor table and quantisation damage all still work on this file."
        ) from err
    finally:
        # In the `finally` so a failed load still stops the thread. A sampler
        # left running polls forever and holds a reference to the process.
        sampler.stop()
    load_seconds = time.perf_counter() - t0

    if on_stage:
        on_stage("device", f"moving to {device}")
    try:
        # Metered, like the HuggingFace path. Imported here rather than at the
        # top because `runtime` imports this module; a function-local import is
        # how the rest of this package breaks that cycle.
        #
        # The reason is weaker here than there and the meter is still worth it:
        # a dequantised model is already in RAM, so this is a plain host copy
        # at bus speed rather than the mmap being pulled off a disk. It is
        # still gigabytes, and a reader watching one stage report bytes and
        # the other report nothing would reasonably read that as a hang.
        from .device_move import move_to_device

        move_to_device(model, device)
    except Exception as err:
        # Was bare, so a CUDA OOM here escaped as a 500 "check the terminal"
        # -- and this is precisely where the preflight is weakest, because a
        # "fits" verdict is about free memory rather than a fragmented heap.
        # The HF path has said this properly for months; so does this now.
        raise Refusal(
            f"{filename} dequantised but would not move onto {device}: "
            f"{type(err).__name__}. The weights are {_gb(p.resident_bytes)} at "
            f"{dtype} against {_gb(p.device_free_bytes)} free — either "
            "something else took the memory in between, or the device is too "
            "fragmented to hand out one block that size."
        ) from err
    model.eval()

    measured = sum(q.numel() * q.element_size() for q in model.parameters())

    try:
        tokenizer = AutoTokenizer.from_pretrained(directory, gguf_file=filename)
    except Exception as err:
        # After the expensive part, so failing here silently wasted the whole
        # dequantise. Named rather than escaping as a 500.
        raise Refusal(
            f"{filename} loaded, but transformers could not build its "
            f"tokeniser: {type(err).__name__}. The file's tokeniser section "
            "may use a scheme this transformers does not implement — the "
            "header still reads."
        ) from err
    return Loaded(
        model=model,
        tokenizer=tokenizer,
        plan=p,
        measured_resident_bytes=measured,
        load_seconds=load_seconds,
        measured_peak_host_bytes=sampler.delta(),
    )


class _RssSampler:
    """Process RSS high-water mark over a span, or None if psutil is absent.

    None rather than 0 on every failure path. A 0 here would read as "the
    dequantiser allocated nothing", which is the opposite of what this module
    exists to say.
    """

    INTERVAL_S = 0.02

    def __init__(self) -> None:
        self._proc = None
        self._base = 0
        self._peak = 0
        self._stop = None
        self._thread = None
        try:
            import psutil

            self._proc = psutil.Process()
        except Exception:  # noqa: S110 - absence is reported as None, not raised
            pass

    def start(self) -> None:
        if self._proc is None:
            return
        import threading

        self._base = self._peak = self._proc.memory_info().rss
        self._stop = threading.Event()

        def watch() -> None:
            while not self._stop.wait(self.INTERVAL_S):
                try:
                    self._peak = max(self._peak, self._proc.memory_info().rss)
                except Exception:
                    return

        self._thread = threading.Thread(target=watch, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._stop is not None:
            self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def delta(self) -> int | None:
        """Growth over the span, not absolute RSS — the interpreter, torch and
        whatever else was already resident are not this load's cost."""
        if self._proc is None:
            return None
        return max(0, self._peak - self._base)


def blocks_of(model):
    """The transformer block list, or a refusal naming what was looked for.

    Same three paths the rest of the codebase walks. Duplicated rather than
    imported so that a GGUF-shaped model with an unusual layout fails here,
    where the message can say "this GGUF", rather than inside the ablation
    loop where it cannot.
    """
    for attribute_path in ("model.layers", "transformer.h", "gpt_neox.layers"):
        node = model
        for part in attribute_path.split("."):
            node = getattr(node, part, None)
            if node is None:
                break
        if node is not None:
            return node
    raise Refusal(
        "this model's transformer blocks are not at model.layers, "
        "transformer.h or gpt_neox.layers, so per-layer analysis cannot "
        "reach them. Logit lens and generation still work."
    )


def default_cache_dir() -> Path:
    """Where a downloaded GGUF lands — HuggingFace's own cache, unmodified."""
    from . import paths

    return paths.hf_hub_cache()


def download(repo_id: str, filename: str, *, revision: str | None = None) -> Path:
    """Fetch one GGUF from the hub into HuggingFace's cache.

    One file, named. GGUF repos routinely ship a dozen quantisations of the
    same model and a whole-repo snapshot would pull every one of them —
    measured on ggml-org/gemma-4-E2B-it-GGUF, that is 2.8 GB asked for against
    19 GB delivered.
    """
    from huggingface_hub import hf_hub_download

    try:
        got = hf_hub_download(repo_id, filename, revision=revision)
    except Exception as err:
        raise BadRequest(
            f"could not download {filename} from {repo_id}: {type(err).__name__}"
        ) from err
    return Path(os.fspath(got))
