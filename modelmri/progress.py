"""Live progress for the current model load.

A cold load is minutes long: the weights download, then a slow disk reads
them back. With no feedback the UI just says "loading" and people
reasonably conclude it has hung. So the load publishes what stage it is in
and, while bytes are moving, how far along the download is.

Byte counting reads the HuggingFace cache directory rather than hooking
huggingface_hub's internals. That works for every download path it might
take (hf_transfer, xet, a resumed partial file) and cannot break when the
library reorganises itself.

It takes the *larger* of blobs/ and snapshots/ rather than their sum,
because which one holds the bytes depends on the machine and we measured
all three cases: on Unix the blobs are real and snapshots are symlinks to
them; on Windows without developer mode snapshots are full copies; and with
a current hub the finished blob is *moved* into snapshots, leaving blobs
empty. Summing double-counts two of those, and counting blobs alone reports
zero forever on the third.

The numerator and the denominator have to agree on which files count, and
for a long time they did not: the total came from the repo's top-level
files while the on-disk figure walked the whole tree. Any repo that ships a
second copy of its weights in a subfolder then read as more than 100%.
meta-llama/Llama-3.2-1B-Instruct is one -- it carries
`original/consolidated.00.pth` beside `model.safetensors`, both 2.472 GB --
and it displayed as "5.0 GB / 2.5 GB", measured 199.7%. So the Hub's file
list now decides one set of names, and both sides count exactly that set.

Everything here is best-effort. A load must never fail because its
progress meter did.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# What from_pretrained actually fetches: one weight format plus the small
# config/tokenizer files. A whitelist, not a blacklist -- popular repos ship
# tflite, rust, h5, flax and onnx variants nobody asked for, and gpt2 alone
# carries four of them. Blacklisting them made a fully-cached gpt2 report 26%.
_CONFIG = (".json", ".txt", ".model")

# How long a download may sit at the same byte count before we call it stalled.
#
# This was 45 s, chosen before hf_xet existed. huggingface_hub 1.x installs
# hf_xet by default and it does not stream into the blob: it reconstructs
# from its own chunk cache and writes the file in large, infrequent jumps.
# Watching a healthy 324 MB download of EleutherAI/pythia-160m, the blob sat
# unchanged at 2.1 MB from 4.1 s to 75.7 s -- a 71.6 s gap -- and again from
# 85.4 s to 144.4 s. At 45 s that download would have been called stalled
# twice while it was working perfectly. 180 s clears the longest gap
# measured with room to spare.
STALL_AFTER_S = 180.0

# How long *any* stage may go with no bytes, no stage change and no CPU
# before we say the load is wedged rather than slow. Separate from the
# download threshold because it is a different claim about a different
# failure: a download that stopped receiving is still a live process, while
# this is a load that has stopped executing.
WEDGED_AFTER_S = 120.0

# CPU seconds the whole process must burn during a wedge window to count as
# "still doing something". A load that is genuinely working spends far more
# than this; the polling in here spends far less. Measured on a wedged load:
# 0.3 CPU-seconds and 0 bytes read over 12 s while `.to(cuda)` never
# returned, against 295 MB/s available from the same file in another
# process and 1266 MB/s host-to-device on the same GPU.
WEDGED_CPU_S = 2.0

# Seconds to wait on the Hub for a file listing. Unbounded, this ran on the
# watcher thread before the first byte figure was published: measured at
# 1502 ms on a fully cached model that needed no network at all, and with
# no ceiling at all on a bad connection.
HUB_TIMEOUT_S = 8.0

# How much new data has to land before "already cached" is treated as having
# been wrong. Large enough that a lock file or a rewritten config cannot trip
# it, small enough to catch a real download in its first seconds.
_CACHE_WRONG_AFTER = 32 * 1024 * 1024


def _weight_files(names: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """The single weight format a load will download, preferring safetensors."""
    for ext in (".safetensors", ".bin", ".pth"):
        if picked := [(n, s) for n, s in names if n.endswith(ext)]:
            return picked
    return []


def hub_cache() -> Path:
    """Where huggingface_hub keeps blobs, honouring the usual overrides.

    A one-line delegation, kept only because callers import this name. It
    used to hand-roll the resolution and got it subtly wrong: no `~`
    expansion, and no HUGGINGFACE_HUB_CACHE, so the download meter watched a
    directory nothing was being written to and reported 0 bytes forever.
    """
    from . import paths

    return paths.hf_hub_cache()


def _model_dir(hf_id: str) -> Path:
    return hub_cache() / f"models--{hf_id.replace('/', '--')}"


def _size(f: Path) -> int:
    """One file's size, or 0. Per file, deliberately.

    The whole walk used to sit under a single try/except, so one file
    disappearing -- and the hub moves blobs into snapshots mid-load, so they
    do -- zeroed the entire count and dropped the bar to nothing.
    """
    try:
        return f.stat().st_size
    except OSError:
        return 0


def _tree_bytes(root: Path, keep=None) -> int:
    """Bytes under `root`, optionally only the files `keep` accepts.

    `keep` receives the path relative to `root`, in posix form, so it can
    match the names the Hub publishes.
    """
    try:
        files = list(root.rglob("*"))
    except OSError:
        return 0
    total = 0
    for f in files:
        try:
            if not f.is_file():
                continue
        except OSError:
            continue
        if keep is not None and not keep(f.relative_to(root).as_posix()):
            continue
        total += _size(f)
    return total


def _default_keep(name: str) -> bool:
    """The files a load pulls, when the Hub could not be asked.

    Top level only: every subfolder in a model repo is a variant of the same
    weights in another runtime's format (onnx/, gguf/, coreml/, and Meta's
    original/), and `from_pretrained` reads none of them.
    """
    if "/" in name:
        return False
    return name.endswith((".safetensors", ".bin", ".pth", *_CONFIG))


def _revision_bytes(snapshots: Path, keep) -> int:
    """The largest single revision, never the sum of several.

    A cache can hold more than one revision of the same repo. Adding them
    reports a model at a multiple of its real size, and a load reads exactly
    one of them.
    """
    try:
        revs = [d for d in snapshots.iterdir() if d.is_dir()]
    except OSError:
        return 0
    return max((_tree_bytes(rev, keep) for rev in revs), default=0)


def _bytes_on_disk(hf_id: str, wanted: frozenset[str] | None = None) -> int:
    """Bytes of this model already on disk. See the module docstring for why
    this is a max and not a sum.

    `wanted` is the exact set of repo files this load will read, from the
    Hub's own listing. Without it -- offline, or a repo that publishes no
    file sizes -- we fall back to a shape rule, which is less precise but
    still excludes the subfolder duplicates that caused the 199.7% reading.
    """
    keep = (lambda n: n in wanted) if wanted else _default_keep
    model = _model_dir(hf_id)
    # blobs/ is content-addressed and flat: the names are hashes, so there is
    # nothing there to match against a file list. Everything in it counts,
    # including the `.incomplete` partials that are the only visible sign of
    # a download in flight.
    return max(_tree_bytes(model / "blobs"), _revision_bytes(model / "snapshots", keep))


def _offline() -> bool:
    """Whether the Hub is off limits, by the hub's own environment variable."""
    return os.environ.get("HF_HUB_OFFLINE", "").strip().lower() not in (
        "",
        "0",
        "false",
    )


def _expected_files(hf_id: str) -> tuple[frozenset[str], int]:
    """The exact files a load pulls, and their total size.

    An empty set with a 0 total means "could not be determined" -- offline,
    a rate limit, a private repo, or a repo that publishes no sizes. The
    caller shows an indeterminate bar rather than inventing a denominator.
    """
    if _offline():
        return frozenset(), 0
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(hf_id, files_metadata=True, timeout=HUB_TIMEOUT_S)
        files = info.siblings or []
        # Variants live in subfolders (onnx/, gguf/, coreml/, original/) we
        # never touch. This is also what keeps the numerator honest: the same
        # names go on to select what counts on disk.
        sized = [(f.rfilename, f.size or 0) for f in files if "/" not in f.rfilename]
        keep = _weight_files(sized) + [(n, s) for n, s in sized if n.endswith(_CONFIG)]
        return frozenset(n for n, _ in keep), sum(s for _, s in keep)
    except Exception:
        return frozenset(), 0


def _expected_bytes(hf_id: str) -> int:
    """Total download size for the files a load actually pulls. 0 if unknown."""
    return _expected_files(hf_id)[1]


@dataclass
class Snapshot:
    active: bool = False
    hf_id: str | None = None
    stage: str = ""  # resolving | weights | device | ready | error | cancelled
    detail: str = ""
    bytes_done: int = 0
    bytes_total: int = 0  # 0 means "unknown", the UI shows an indeterminate bar
    elapsed_s: float = 0.0
    # Seconds left at the average rate so far. None means "not enough signal
    # to say", which is a real answer and not a zero.
    eta_s: float | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _eta(done: int, total: int, elapsed: float) -> float | None:
    """Seconds remaining at the average rate so far, or None.

    The AVERAGE rate, deliberately, not an instantaneous one. hf_xet writes
    blobs in large infrequent jumps — a 71.6 second gap was measured during a
    perfectly healthy download — so an instantaneous estimate swings between
    "12 seconds" and "four hours" on the same transfer, and a number that
    jumps like that teaches the reader to ignore it.

    None until there is something to divide: no total (the UI shows an
    indeterminate bar), nothing transferred, or under two seconds of history.
    A countdown that starts wrong is worse than one that starts late.
    """
    if not total or done <= 0 or elapsed < 2.0:
        return None
    if done >= total:
        return 0.0
    rate = done / elapsed
    if rate <= 0:
        return None
    return round((total - done) / rate, 1)


class _Tracker:
    """Single in-flight load. Loads are serialised by the runtime lock."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snap = Snapshot()
        self._t0 = 0.0
        self._stop: threading.Event | None = None
        # Which load is current. A watcher writes into the shared snapshot,
        # so without this the previous load's watcher can publish its byte
        # counts under the next load's name -- and did: a hung load of
        # Llama-3.2-1B showed "5.0 GB / 2.5 GB" against a queued
        # Qwen2.5-0.5B, which has neither number.
        self._gen = 0
        # Set by the user asking to stop. The loader polls it; see
        # runtime._prefetch_weights for why a flag is enough to actually
        # halt a download that has already started.
        self.cancelled = threading.Event()

    def request_cancel(self) -> bool:
        """Ask the running load to stop. False if there is nothing running."""
        with self._lock:
            if not self._snap.active:
                return False
            self._snap.detail = "stopping…"
        self.cancelled.set()
        return True

    def snapshot(self) -> Snapshot:
        with self._lock:
            snap = Snapshot(**asdict(self._snap))
        if snap.active:
            snap.elapsed_s = round(time.monotonic() - self._t0, 1)
        snap.eta_s = _eta(snap.bytes_done, snap.bytes_total, snap.elapsed_s)
        return snap

    def start_external(
        self, name: str, *, stage: str = "weights", detail: str = ""
    ) -> None:
        """Track a long job that reports its OWN byte counts.

        `start` spawns a watcher thread that polls the HuggingFace cache on
        disk, because that is the only way to see what hf_hub is doing. An
        Ollama pull streams its own `completed`/`total` per chunk, so there is
        nothing to watch — and a watcher pointed at the HF cache during an
        Ollama pull would publish an unrelated directory's size as this job's
        progress, which is the "5.0 GB / 2.5 GB" failure again in a new place.
        """
        self._t0 = time.monotonic()
        self.cancelled.clear()
        with self._lock:
            self._gen += 1
            self._snap = Snapshot(active=True, hf_id=name, stage=stage, detail=detail)
        self._stop = None

    def publish(
        self,
        *,
        bytes_done: int | None = None,
        bytes_total: int | None = None,
        stage: str | None = None,
        detail: str | None = None,
    ) -> None:
        """Write a job's own numbers into the snapshot.

        Ignored once the job is no longer the active one, on the same rule as
        `_publish`: a finished job's last chunk must not land on top of the
        next one's.
        """
        with self._lock:
            if not self._snap.active:
                return
            if bytes_done is not None:
                self._snap.bytes_done = max(0, int(bytes_done))
            if bytes_total is not None:
                self._snap.bytes_total = max(0, int(bytes_total))
            if stage:
                self._snap.stage = stage
            if detail:
                self._snap.detail = detail

    def start(self, hf_id: str) -> None:
        self._t0 = time.monotonic()
        self.cancelled.clear()
        with self._lock:
            self._gen += 1
            gen = self._gen
            self._snap = Snapshot(
                active=True, hf_id=hf_id, stage="resolving", detail="contacting the Hub"
            )
        self._stop = threading.Event()
        threading.Thread(
            target=self._watch, args=(hf_id, self._stop, gen), daemon=True
        ).start()

    def stage(self, stage: str, detail: str = "") -> None:
        with self._lock:
            if self._snap.active:
                self._snap.stage = stage
                if detail:
                    self._snap.detail = detail

    def finish(self, error: str | None = None) -> None:
        if self._stop is not None:
            self._stop.set()
        with self._lock:
            self._snap.active = False
            self._snap.stage = "error" if error else "ready"
            self._snap.error = error
            self._snap.elapsed_s = round(time.monotonic() - self._t0, 1)
            if not error:
                self._snap.detail = "ready"

    def _publish(self, gen: int, **fields) -> bool:
        """Write into the snapshot, but only while this load is still the one
        running. False means this watcher is obsolete and should stop."""
        with self._lock:
            if gen != self._gen or not self._snap.active:
                return False
            for key, value in fields.items():
                setattr(self._snap, key, value)
            return True

    def _watch(self, hf_id: str, stop: threading.Event, gen: int) -> None:
        """Poll the cache directory until the load ends."""
        # Disk first, Hub second. The listing is a network call -- 1502 ms
        # measured on a model that was already complete on disk -- and until
        # it returned the UI had no numbers at all.
        start_bytes = _bytes_on_disk(hf_id)
        if not self._publish(gen, bytes_done=start_bytes):
            return
        wanted, total = _expected_files(hf_id)
        if wanted:
            # Recount against the real file list: the shape rule keeps
            # sibling weight formats a load will not read.
            start_bytes = _bytes_on_disk(hf_id, wanted)
        cached = bool(total) and start_bytes >= total * 0.98
        detail = ""
        if cached:
            detail = "reading from local cache, no download needed"
        elif total:
            detail = f"downloading {total / 1e9:.1f} GB"
        if not self._publish(
            gen,
            bytes_total=total,
            bytes_done=min(start_bytes, total) if total else start_bytes,
            **({"detail": detail} if detail else {}),
        ):
            return

        last_change = time.monotonic()
        last_bytes = start_bytes
        # Process-wide CPU, not this thread's: the question a wedge asks is
        # whether *anything* in here is still executing. It is a plain
        # counter read, so unlike asking CUDA how much memory it has handed
        # out it cannot itself block on the thing that is stuck.
        last_cpu = time.process_time()
        last_stage = ""
        warning: str | None = None  # our own text, if we have overwritten detail
        said = ""  # what the load was saying before we did
        while not stop.wait(0.7):
            done = _bytes_on_disk(hf_id, wanted)
            now = time.monotonic()
            if done != last_bytes:
                last_bytes, last_change = done, now
            with self._lock:
                if gen != self._gen or not self._snap.active:
                    return
                if self._snap.stage != last_stage:
                    last_stage, last_change = self._snap.stage, now
                    last_cpu = time.process_time()
                self._snap.bytes_done = min(done, total) if total else done
                if done > start_bytes and self._snap.stage == "resolving":
                    self._snap.stage = "weights"
                # "Already cached" was decided from the tree's size before
                # anything started, and a tree can be large for reasons other
                # than holding what we need. Real case: a gpt2 cache with a
                # legacy pytorch_model.bin beside the safetensors measured
                # 1045 MB against an expected 551 MB, so the load announced
                # "no download needed" and then downloaded for 275 seconds
                # under that message. Bytes arriving is proof it was wrong.
                if cached and done > start_bytes + _CACHE_WRONG_AFTER:
                    cached = False
                    last_change = now
                    self._snap.detail = (
                        f"downloading {total / 1e9:.1f} GB" if total else "downloading"
                    )
                stalled_s = now - last_change
                cpu_s = time.process_time() - last_cpu
                note = self._quiet_note(self._snap.stage, cached, stalled_s, cpu_s)
                # Restore whatever the load itself was saying once it moves
                # again, so a warning cannot outlive the condition it
                # described. Only our own text is replaced.
                if note:
                    if warning is None:
                        said = self._snap.detail
                    warning = note
                    self._snap.detail = note
                elif warning is not None:
                    if self._snap.detail == warning:
                        self._snap.detail = said
                    warning = None

    @staticmethod
    def _quiet_note(stage: str, cached: bool, quiet_s: float, cpu_s: float) -> str:
        """What to say about a load that has gone quiet, or "" if it is fine.

        Two different silences with two different diagnoses, so they get two
        different sentences and two different thresholds.
        """
        if stage == "weights":
            # A download that dies mid-flight does not raise; it simply stops
            # moving, and the load then hangs for as long as anyone is willing
            # to wait. We watched one sit at 128 MB of 3 GB indefinitely.
            #
            # CPU says nothing here: the download runs in a child process
            # (see runtime._prefetch_weights), so this one is idle by design
            # while gigabytes arrive. Bytes on disk are the only evidence.
            if cached or quiet_s <= STALL_AFTER_S:
                return ""
            return (
                f"no new data for {int(quiet_s)}s - the download may have "
                f"stalled; cancel and retry, or pick a smaller model"
            )
        if quiet_s <= WEDGED_AFTER_S or cpu_s >= WEDGED_CPU_S:
            return ""
        if stage == "resolving":
            return (
                f"no reply from the Hub for {int(quiet_s)}s - check the "
                f"connection, or Stop and try again"
            )
        # Every other stage is this process doing its own work, so no CPU for
        # this long means it has stopped, not that it is slow. Measured on a
        # real one: `.to(cuda)` never returned, 0.3 CPU-seconds and 0 bytes
        # read over 12s, while the same file read at 295 MB/s and the same
        # GPU took host-to-device copies at 1266 MB/s from another process.
        return (
            f"no progress for {int(quiet_s)}s, and {cpu_s:.1f}s of CPU used in "
            f"that time - this load has stopped rather than slowed; Stop it, "
            f"and restart `modelmri serve` if that does not clear it"
        )


TRACKER = _Tracker()

# A SECOND SLOT, because a pull and a load are different jobs that genuinely
# overlap: you can download one model while another is loaded, and the picker
# that starts the download is a sheet over the page that starts the load.
#
# Sharing one tracker was tried and is exactly the bug this module already
# documents fixing once. Measured: an Ollama pull of gemma3:1b was in flight,
# a page load loaded gpt2, and `/api/model/progress` answered
#
#     {"hf_id": "gpt2", "bytes_done": 394812192, "bytes_total": 815310432,
#      "stage": "ready", "active": false}
#
# — one job's name against another job's byte counts, with the pull still
# running and its updates silently dropped because the generation had moved
# on. Same shape as the "5.0 GB / 2.5 GB" report that started all of this.
PULLS = _Tracker()
