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
STALL_AFTER_S = 45.0


def _weight_files(names: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """The single weight format a load will download, preferring safetensors."""
    for ext in (".safetensors", ".bin", ".pth"):
        if picked := [(n, s) for n, s in names if n.endswith(ext)]:
            return picked
    return []


def hub_cache() -> Path:
    """Where huggingface_hub keeps blobs, honouring the usual overrides."""
    if hub := os.environ.get("HF_HUB_CACHE"):
        return Path(hub)
    home = os.environ.get("HF_HOME")
    base = Path(home) if home else Path.home() / ".cache" / "huggingface"
    return base / "hub"


def _model_dir(hf_id: str) -> Path:
    return hub_cache() / f"models--{hf_id.replace('/', '--')}"


def _tree_bytes(root: Path) -> int:
    try:
        return sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    except OSError:
        return 0


def _bytes_on_disk(hf_id: str) -> int:
    """Bytes of this model already on disk. See the module docstring for why
    this is a max and not a sum."""
    model = _model_dir(hf_id)
    return max(_tree_bytes(model / "blobs"), _tree_bytes(model / "snapshots"))


def _expected_bytes(hf_id: str) -> int:
    """Total download size for the files a load actually pulls. 0 if unknown."""
    try:
        from huggingface_hub import HfApi

        files = HfApi().model_info(hf_id, files_metadata=True).siblings or []
        # Variants live in subfolders (onnx/, gguf/, coreml/) we never touch.
        sized = [(f.rfilename, f.size or 0) for f in files if "/" not in f.rfilename]
        keep = _weight_files(sized) + [(n, s) for n, s in sized if n.endswith(_CONFIG)]
        return sum(s for _, s in keep)
    except Exception:
        return 0


@dataclass
class Snapshot:
    active: bool = False
    hf_id: str | None = None
    stage: str = ""  # resolving | weights | device | ready | error
    detail: str = ""
    bytes_done: int = 0
    bytes_total: int = 0  # 0 means "unknown", the UI shows an indeterminate bar
    elapsed_s: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class _Tracker:
    """Single in-flight load. Loads are serialised by the runtime lock."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snap = Snapshot()
        self._t0 = 0.0
        self._stop: threading.Event | None = None

    def snapshot(self) -> Snapshot:
        with self._lock:
            snap = Snapshot(**asdict(self._snap))
        if snap.active:
            snap.elapsed_s = round(time.monotonic() - self._t0, 1)
        return snap

    def start(self, hf_id: str) -> None:
        self._t0 = time.monotonic()
        with self._lock:
            self._snap = Snapshot(
                active=True, hf_id=hf_id, stage="resolving", detail="contacting the Hub"
            )
        self._stop = threading.Event()
        threading.Thread(
            target=self._watch, args=(hf_id, self._stop), daemon=True
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

    def _watch(self, hf_id: str, stop: threading.Event) -> None:
        """Poll the cache directory until the load ends."""
        start_bytes = _bytes_on_disk(hf_id)
        total = _expected_bytes(hf_id)
        cached = bool(total) and start_bytes >= total * 0.98
        with self._lock:
            self._snap.bytes_total = total
            self._snap.bytes_done = start_bytes
            if cached:
                self._snap.detail = "reading from local cache, no download needed"
            elif total:
                self._snap.detail = f"downloading {total / 1e9:.1f} GB"

        last_change = time.monotonic()
        last_bytes = start_bytes
        while not stop.wait(0.7):
            done = _bytes_on_disk(hf_id)
            now = time.monotonic()
            if done != last_bytes:
                last_bytes, last_change = done, now
            with self._lock:
                if not self._snap.active:
                    return
                self._snap.bytes_done = done
                if done > start_bytes and self._snap.stage == "resolving":
                    self._snap.stage = "weights"
                # A download that dies mid-flight does not raise; it simply
                # stops moving, and the load then hangs for as long as anyone
                # is willing to wait. We watched one sit at 128 MB of 3 GB
                # indefinitely. Say so rather than spin a bar over nothing.
                stalled_s = now - last_change
                if not cached and stalled_s > STALL_AFTER_S and done < total:
                    self._snap.detail = (
                        f"no new data for {int(stalled_s)}s - the download may have "
                        f"stalled; cancel and retry, or pick a smaller model"
                    )


TRACKER = _Tracker()
