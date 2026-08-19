"""Look inside a checkpoint before loading it, and refuse when it bites.

This closes a hole in ModelMRI's own posture rather than adding a product
feature, and the distinction matters: the tool downloads arbitrary weights
from the Hub onto somebody's laptop, checks how BIG they are with
`capacity.guard`, and then hands them to `from_pretrained` without ever asking
what is inside. It also accepts a user-supplied `adapter.py` and TorchScript
as a documented feature.

It already knew this risk class existed. `circuit.py` reads a `.pt` through a
restricted unpickler precisely because a pickle is a program, and its module
docstring says so. That defence was applied in exactly one place.

## Why a pickle is the whole problem

A `.bin`, `.pt`, `.pth` or `.ckpt` is a pickle, and unpickling is not parsing:
`GLOBAL` names any importable object and `REDUCE` calls it. `os.system` is an
importable object. The payload does not have to be exotic — it runs during
load, before a single tensor is read, and nothing about the file's size or
name says so.

So this walks the opcode stream with `pickletools.genops` and never executes
it. Reading the instructions is not running them.

## What it decides, and what it refuses to decide

Three verdicts, and the third is the one that keeps this honest:

  `safe`       nothing executable found, in a format this can fully read
  `dangerous`  a specific finding, named, with the opcode and the target
  `unscanned`  this could not read it — the format, or a stream that will
               not parse. Reported as unscanned, never as clean.

`dangerous` means something executable was positively identified. Anything
this merely could not read is `unscanned`, including a pickle whose opcode
stream will not walk: "I could not tell" and "I found something" are different
answers, and only one of them should stop a load.

`unscanned` exists for the same reason `gguf_read` reports `ggml type 37
(unknown)` rather than guessing: a scanner that answers "clean" for a file it
could not open is worse than no scanner, because the answer is load-bearing
and wrong.

## Where it runs

At the moment of risk — on the download path and the custom-model path, next
to the disk and VRAM refusals — rather than as a separate command somebody has
to remember. promptfoo's ModelAudit scans a file you point it at; this refuses
to load one.

`safetensors` is the format that makes all of this unnecessary: it is a JSON
header and a block of floats, with no mechanism to execute anything. When a
repository offers both, saying so is more useful than any finding here.
"""

from __future__ import annotations

import json
import os
import pickletools
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from .errors import Refusal

SAFE = "safe"
DANGEROUS = "dangerous"
UNSCANNED = "unscanned"

# Modules that have no business being named by a tensor file. A pickle that
# reaches for any of these during load is not describing weights.
#
# Deny-list rather than allow-list, deliberately, and the reason is worth
# stating because the opposite is usually right: an allow-list here would have
# to enumerate every module every legitimate checkpoint format names —
# torch's rebuild machinery, numpy's dtypes, collections, every model library
# that ever pickled a config — and a false positive means refusing to load a
# model that is fine. `circuit.py` CAN use an allow-list because it reads one
# known format; this reads whatever the Hub served.
#
# The consequence is stated rather than hidden: this catches the known-bad,
# not everything bad. `safetensors` is the answer to everything bad.
DANGEROUS_MODULES = frozenset(
    {
        "os",
        "posix",
        "nt",
        "subprocess",
        "sys",
        "shutil",
        "socket",
        "http",
        "urllib",
        "urllib.request",
        "requests",
        "ftplib",
        "telnetlib",
        "smtplib",
        "pty",
        "platform",
        "ctypes",
        "importlib",
        "imp",
        "runpy",
        "code",
        "codeop",
        "pickle",
        "shelve",
        "dill",
        "multiprocessing",
        "asyncio",
        "webbrowser",
        "tempfile",
        "glob",
        "pathlib",
    }
)

# Callables that execute a string, wherever they are found. `builtins.eval` is
# the obvious one; `torch.jit.annotations` and friends are not here because
# they do not execute caller-controlled text.
DANGEROUS_NAMES = frozenset(
    {
        "eval",
        "exec",
        "execfile",
        "compile",
        "open",
        "__import__",
        "system",
        "popen",
        "spawn",
        "spawnl",
        "spawnv",
        "fork",
        "execv",
        "execve",
        "call",
        "check_call",
        "check_output",
        "run",
        "Popen",
        "getattr",
        "setattr",
        "apply",
        "load",
        "loads",
        "breakpoint",
    }
)

# A pickle inside a pickle: the payload is a base64 or hex blob that a second
# `pickle.loads` unwraps at load time, so the outer opcode stream looks dull.
DECODE_CHAIN = frozenset({"b64decode", "b64encode", "a85decode", "decodebytes"})

# The largest pickle stream this will walk. A 200 MB opcode stream is not a
# config, and walking one costs real time on a machine that is about to need
# its memory. Past this the file is `unscanned` WITH the reason, which is a
# refusal to guess rather than a pass.
MAX_PICKLE_BYTES = 64 * 1024 * 1024

# Ratio past which a zip entry is treated as a decompression bomb. 100:1 is
# ordinary for float data; 1000:1 is not a tensor.
BOMB_RATIO = 1000

# Executable magic numbers, for a payload hidden in a data blob.
EXECUTABLE_MAGIC = {
    b"MZ": "a Windows executable (PE)",
    b"\x7fELF": "a Linux executable (ELF)",
    b"\xcf\xfa\xed\xfe": "a macOS executable (Mach-O)",
    b"\xfe\xed\xfa\xce": "a macOS executable (Mach-O)",
    b"\xca\xfe\xba\xbe": "a macOS universal binary",
}

# Extensions that are pickles, and therefore programs.
PICKLE_SUFFIXES = frozenset({".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle"})

# Extensions that cannot execute anything by construction.
INERT_SUFFIXES = frozenset({".safetensors", ".gguf", ".json", ".txt", ".md", ".model"})

# Weight formats this CANNOT read. Listed so a directory walk surfaces them as
# `unscanned` instead of stepping over them, which is the difference between
# "3 files, all fine" and "3 files read, 1 not looked at".
#
# The first directory scan filtered on formats it understood, so an `.h5`
# beside a poisoned `.pt` simply did not appear in the output — a silent
# truncation in the one tool whose entire job is to not be quietly wrong.
# Several of these DO carry executable payloads (Keras Lambda layers live in
# `.h5` and `.keras`; ONNX and TensorRT can embed custom ops), which is why
# omitting them was worse than merely incomplete.
UNREADABLE_MODEL_SUFFIXES = frozenset(
    {
        ".h5",
        ".hdf5",
        ".keras",
        ".pb",
        ".tflite",
        ".onnx",
        ".engine",
        ".plan",
        ".mlmodel",
        ".mlpackage",
        ".caffemodel",
        ".params",
        ".npz",
        ".joblib",
        ".msgpack",
        ".ot",
        ".tar",
    }
)


class Unsafe(Refusal):
    """This file executes something on load, and the message says what.

    A `Refusal` so the server turns it into a 409 with the sentence intact,
    like every other authored refusal in the package.
    """


@dataclass
class Finding:
    """One reason a file is dangerous, specific enough to act on."""

    kind: str
    detail: str
    where: str = ""

    def to_dict(self) -> dict:
        return {"kind": self.kind, "detail": self.detail, "where": self.where}


@dataclass
class Report:
    """What was found, and — as importantly — what could not be looked at."""

    path: str = ""
    verdict: str = UNSCANNED
    format: str = ""
    findings: list[Finding] = field(default_factory=list)
    # Why a file could not be read, when the verdict is `unscanned`. Never
    # empty in that case: "unscanned" with no reason is indistinguishable
    # from a scan that found nothing.
    reason: str = ""
    scanned_bytes: int = 0

    @property
    def dangerous(self) -> bool:
        return self.verdict == DANGEROUS

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "verdict": self.verdict,
            "format": self.format,
            "findings": [f.to_dict() for f in self.findings],
            "reason": self.reason,
            "scanned_bytes": self.scanned_bytes,
            "means": self.means(),
        }

    def means(self) -> str:
        if self.verdict == DANGEROUS:
            first = self.findings[0] if self.findings else None
            rest = (
                f" and {len(self.findings) - 1} more finding(s)"
                if len(self.findings) > 1
                else ""
            )
            return (
                f"{Path(self.path).name} executes something when it is "
                f"loaded: {first.detail if first else 'see findings'}{rest}. "
                f"Unpickling is not parsing — the payload runs before a single "
                f"tensor is read. If this repository also publishes "
                f"safetensors, use that instead; it is a JSON header and a "
                f"block of floats with no way to execute anything."
            )
        if self.verdict == UNSCANNED:
            return (
                f"{Path(self.path).name} was NOT scanned: {self.reason} This "
                f"is not a clean bill of health — nothing here looked inside "
                f"it, and saying 'safe' for a file this could not read would "
                f"be the answer people act on being wrong."
            )
        return (
            f"{Path(self.path).name} ({self.format}) contains nothing "
            f"executable that this recognises. It is a deny-list, so it "
            f"catches the known-bad rather than everything bad — a "
            f"safetensors file needs no scan at all because there is no "
            f"mechanism in the format to execute anything."
        )


def scan(path: str | Path) -> Report:
    """Look inside one file. Never imports it, never executes it."""
    p = Path(path)
    report = Report(path=str(p))
    if not p.is_file():
        report.reason = "there is no file at that path."
        return report

    suffix = p.suffix.lower()
    try:
        report.scanned_bytes = p.stat().st_size
    except OSError:
        report.scanned_bytes = 0

    if suffix in INERT_SUFFIXES:
        report.verdict = SAFE
        report.format = suffix.lstrip(".") or "unknown"
        return report

    if suffix in PICKLE_SUFFIXES:
        return _scan_pickle_container(p, report)

    if suffix in (".py",):
        return _scan_python(p, report)

    if suffix in UNREADABLE_MODEL_SUFFIXES:
        report.format = suffix.lstrip(".")
        report.reason = (
            f"'{suffix}' is a weight format this cannot read. Several of them "
            f"can carry executable payloads — Keras Lambda layers, ONNX and "
            f"TensorRT custom ops — so this is a gap in coverage rather than "
            f"a reason for comfort."
        )
        return report

    report.reason = (
        f"'{suffix or 'no extension'}' is not a format this knows how to read."
    )
    return report


def _scan_pickle_container(p: Path, report: Report) -> Report:
    """A torch checkpoint is usually a zip of pickles; sometimes it is one."""
    if zipfile.is_zipfile(p):
        report.format = "torch zip archive"
        return _scan_zip(p, report)
    report.format = "raw pickle"
    try:
        data = p.read_bytes()
    except OSError:
        report.reason = "the file could not be read."
        return report
    if len(data) > MAX_PICKLE_BYTES:
        report.reason = (
            f"the pickle stream is {len(data) / 1e6:,.0f} MB, past the "
            f"{MAX_PICKLE_BYTES / 1e6:,.0f} MB this will walk."
        )
        return report
    findings, unreadable = _walk_pickle(data, where=p.name)
    if unreadable:
        report.reason = f"{unreadable} It may not be a pickle at all."
        return report
    return _finish(report, findings)


def _scan_zip(p: Path, report: Report) -> Report:
    findings: list[Finding] = []
    try:
        with zipfile.ZipFile(p) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                # A decompression bomb, checked from the HEADER rather than by
                # extracting. Extracting to find out whether extracting is
                # safe is the trap this avoids.
                if info.compress_size and info.file_size / info.compress_size > (
                    BOMB_RATIO
                ):
                    findings.append(
                        Finding(
                            "zip bomb",
                            f"{info.filename} expands "
                            f"{info.file_size / max(1, info.compress_size):,.0f}x, "
                            f"to {info.file_size / 1e9:,.1f} GB",
                            info.filename,
                        )
                    )
                    continue
                if not info.filename.endswith(("data.pkl", ".pkl", ".pickle")):
                    continue
                if info.file_size > MAX_PICKLE_BYTES:
                    report.reason = (
                        f"{info.filename} is {info.file_size / 1e6:,.0f} MB, "
                        f"past what this will walk."
                    )
                    return report
                with zf.open(info) as fh:
                    found, unreadable = _walk_pickle(fh.read(), where=info.filename)
                    if unreadable:
                        # One unreadable member makes the WHOLE archive
                        # unscanned. Reporting the other entries as safe would
                        # be a clean bill of health for a container with a
                        # part nobody could look at.
                        report.reason = f"{info.filename}: {unreadable}"
                        return report
                    findings.extend(found)
    except (zipfile.BadZipFile, OSError, RuntimeError) as err:
        report.reason = f"the archive could not be read ({type(err).__name__})."
        return report
    return _finish(report, findings)


def _walk_pickle(data: bytes, *, where: str) -> tuple[list[Finding], str]:
    """Read the opcode stream. Reading instructions is not running them.

    Returns `(findings, unreadable_reason)`. The second is `""` when the
    stream parsed — a value rather than an exception, because the two answers
    "I found nothing" and "I could not look" are both ordinary outcomes here
    and neither is exceptional.
    """
    findings: list[Finding] = []
    for magic, what in EXECUTABLE_MAGIC.items():
        if data.startswith(magic):
            findings.append(Finding("embedded executable", what, where))
            return findings, ""

    try:
        ops = list(pickletools.genops(data))
    except Exception as err:
        # UNSCANNED, not dangerous — and the distinction is the whole
        # discipline of this module. `dangerous` means "something executable
        # was positively identified"; a stream that will not parse means "I
        # could not tell", which is the definition of unscanned.
        #
        # Calling it dangerous was tried and was wrong in practice. `guard`
        # raises on dangerous, so a merely CORRUPT file -- or a `.pt` that is
        # not a pickle at all, which is common -- stopped the custom-model
        # loader before it could produce its actual diagnosis, replacing a
        # useful "this is a plain checkpoint, not TorchScript, here is what to
        # do" with an accusation. The suite caught it.
        return [], (
            f"the opcode stream could not be walked ({type(err).__name__}), "
            f"so nothing here can say what it does on load."
        )

    # A string operand stack, tracked exactly rather than pattern-matched.
    #
    # Protocol 2 emits `GLOBAL` with "module name" as one argument. Protocol 4
    # and 5 -- what `pickle.dumps` writes by default, and therefore what a real
    # attack ships -- emit `STACK_GLOBAL`, which POPS the module and name from
    # the stack as two separate string pushes. The first version of this only
    # read opcode arguments, so it saw the dull half of every modern payload
    # and passed `os.system` as clean. Its own tests, built with real
    # `pickle.dumps`, caught that.
    #
    # Only string-producing opcodes are tracked, and only the last few: this
    # is not an interpreter and must never become one.
    strings: list[str] = []
    STRING_OPS = {
        "SHORT_BINUNICODE",
        "BINUNICODE",
        "BINUNICODE8",
        "UNICODE",
        "STRING",
        "BINSTRING",
        "SHORT_BINSTRING",
    }

    def flag(module: str, name: str) -> None:
        root = module.split(".")[0]
        target = f"{module}.{name}".strip(".")
        if root in DANGEROUS_MODULES or module in DANGEROUS_MODULES:
            findings.append(
                Finding(
                    "executes on load",
                    f"names `{target}` — the `{root}` module has no role in "
                    f"describing weights",
                    where,
                )
            )
        elif name in DECODE_CHAIN:
            findings.append(
                Finding(
                    "decode-then-execute chain",
                    f"names `{target}`, the usual first half of a payload that "
                    f"a second unpickle unwraps at load time",
                    where,
                )
            )
        elif name in DANGEROUS_NAMES:
            findings.append(
                Finding(
                    "executes on load",
                    f"names `{target}`, which runs whatever it is given",
                    where,
                )
            )

    for op, arg, _pos in ops:
        if op.name in STRING_OPS and isinstance(arg, str):
            strings.append(arg)
            # Bounded. A pickle with a million strings must not become a
            # million-entry list in a function whose job is to be cheap.
            if len(strings) > 4:
                del strings[:-4]
            continue
        if op.name == "STACK_GLOBAL":
            # The two most recent pushes, in order: module then name.
            if len(strings) >= 2:
                flag(strings[-2], strings[-1])
                del strings[-2:]
            continue
        if op.name in ("GLOBAL", "INST", "OBJ") and isinstance(arg, str):
            module, _, name = arg.partition(" ")
            flag(module, name)

    return findings, ""


def _scan_python(p: Path, report: Report) -> Report:
    """A user-supplied adapter. It is MEANT to be code; say so plainly.

    Not a finding-by-finding scan, because every line of it is by definition
    executable and a list of dangerous imports would imply the rest is safe.
    """
    report.format = "python source"
    report.reason = (
        "this is source code, which runs in full when imported. Nothing here "
        "can make that safe, and a list of suspicious lines would imply the "
        "unlisted ones are fine. Read it before pointing ModelMRI at it."
    )
    return report


def _finish(report: Report, findings: list[Finding]) -> Report:
    report.findings = findings
    report.verdict = DANGEROUS if findings else SAFE
    return report


class ScanTree(list):
    """The reports, plus what the walk could not tell you.

    A `list` subclass so every existing caller keeps working unchanged — they
    index it, iterate it and take its length — while a caller that wants to be
    honest about the walk has somewhere to read it from.

    `readable=False` is NOT "no weight files here". A directory that could not
    be opened has unknown contents, and a scanner whose job is to say whether
    a file is safe to load is the last place an unearned "all clear" belongs.
    """

    def __init__(self, reports=(), *, n_total: int = 0, readable: bool = True):
        super().__init__(reports)
        #: How many weight-shaped files the walk SAW, including those past the
        #: limit. `len(self)` is how many were scanned.
        self.n_total = n_total
        #: Whether the directory could be walked at all.
        self.readable = readable

    @property
    def truncated(self) -> bool:
        return self.n_total > len(self)

    def __eq__(self, other: object) -> bool:
        """Rows AND `n_total`, `readable`, against another ScanTree.

        Inheriting `list.__eq__` made two of these compare equal while
        disagreeing about `n_total`, `readable` — which is the entire reason this
        class exists rather than a plain list. CodeQL flags the shape;
        the bug it describes is real here.

        Against anything that is NOT one of these, the rows decide. A
        plain list carries no claim about the walk, so there is nothing
        for it to disagree with, and `scan_dir(absent) == []` stays
        true — as it should.
        """
        if isinstance(other, ScanTree):
            return (
                list(self) == list(other)
                and self.n_total == other.n_total
                and self.readable == other.readable
            )
        return list(self) == other

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    #: Unhashable, like the `list` it extends. Spelled out because
    #: defining `__eq__` sets this to None anyway and a reader should
    #: not have to remember that rule.
    __hash__ = None  # type: ignore[assignment]


def scan_dir(root: str | Path, *, limit: int = 200) -> ScanTree:
    """Every weight-shaped file under a directory, worst first.

    `limit` bounds how many are OPENED; the walk keeps counting past it, so
    the result can say "3 of 84" rather than "3". Counting is free — the
    expensive step is `scan`, which reads the file and walks pickle opcodes,
    and `sorted(base.rglob("*"))` has already materialised the tree.
    """
    base = Path(root)
    if not base.is_dir():
        return ScanTree()

    # PROBED, not assumed. `is_dir()` is true for a directory this account
    # cannot open, and `rglob` then yields nothing — so an unreadable folder
    # walked to an empty list and was reported as "nothing weight-shaped was
    # found at that path", which is a claim about contents nobody saw.
    try:
        with os.scandir(base) as it:
            next(it, None)
    except OSError:
        return ScanTree(readable=False)

    seen: list[Report] = []
    n_total = 0
    interesting = PICKLE_SUFFIXES | INERT_SUFFIXES | UNREADABLE_MODEL_SUFFIXES | {".py"}
    try:
        entries = sorted(base.rglob("*"))
    except OSError:
        # The walk started and could not finish. Whatever was reached is real,
        # but the tree is not fully known — say so rather than imply it is.
        return ScanTree(readable=False)
    for path in entries:
        try:
            if not (path.is_file() and path.suffix.lower() in interesting):
                continue
        except OSError:
            continue
        n_total += 1
        if len(seen) < limit:
            seen.append(scan(path))
    order = {DANGEROUS: 0, UNSCANNED: 1, SAFE: 2}
    seen.sort(key=lambda r: (order.get(r.verdict, 3), r.path))
    return ScanTree(seen, n_total=n_total)


def guard(path: str | Path, *, confirm: bool = False) -> Report:
    """Scan, and raise rather than let a dangerous file be loaded.

    `confirm` lets somebody proceed anyway — their machine, their decision —
    but it has to be said out loud, which is the whole difference between a
    refusal and a warning nobody reads.

    An `unscanned` verdict does NOT raise. Refusing every format this cannot
    read would make the scanner a gate on its own coverage, and most of what
    it cannot read is harmless.
    """
    report = scan(path)
    if report.dangerous and not confirm:
        raise Unsafe(report.means())
    return report


def prefer_safetensors(names: list[str]) -> str:
    """The sentence to say when a repo offers both. `""` when it does not.

    More useful than any finding this module can produce: the whole class of
    problem disappears if the other file is taken instead.
    """
    has_safe = any(str(n).lower().endswith(".safetensors") for n in names)
    risky = [str(n) for n in names if Path(str(n)).suffix.lower() in PICKLE_SUFFIXES]
    if has_safe and risky:
        return (
            f"This repository publishes safetensors as well as "
            f"{len(risky)} pickle file(s). Prefer the safetensors: it is a "
            f"JSON header and a block of floats, with no mechanism to execute "
            f"anything on load."
        )
    return ""


def _json_or_none(text: str):
    """Kept beside the readers that need it; `None` rather than a raise."""
    try:
        return json.loads(text)
    except ValueError:
        return None
