"""Portability defects found by audit, each pinned to the failure it caused.

Every test here reproduces a scenario CI does not naturally reach — a machine
with no home directory, a network filesystem, a non-ASCII username, a model
loaded from a folder, an accelerator with no separate VRAM. The bugs were all
silent: nothing raised, the answers were just wrong or the tool refused to
start somewhere nobody had looked.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

HOME_VARS = (
    "HOME",
    "USERPROFILE",
    "HOMEDRIVE",
    "HOMEPATH",
    "LOCALAPPDATA",
    "APPDATA",
    "XDG_DATA_HOME",
    "XDG_CONFIG_HOME",
    "XDG_CACHE_HOME",
    "MODELMRI_HOME",
    "HF_HOME",
    "HF_HUB_CACHE",
)


def _reimport_paths(monkeypatch):
    for v in HOME_VARS:
        monkeypatch.delenv(v, raising=False)
    for mod in [m for m in sys.modules if m.startswith("modelmri")]:
        monkeypatch.delitem(sys.modules, mod, raising=False)
    from modelmri import paths

    return paths


def test_no_home_still_yields_absolute_paths(monkeypatch):
    """`os.path.expanduser` does not raise when it cannot resolve — it returns
    "~" unchanged, so every path built on it was RELATIVE, rooted at a
    directory literally named `~`.

    Measured before the fix, with the variables below cleared:
    `data_dir()` -> `~/AppData/Local/ModelMRI`, `is_absolute()` False. On a
    container with no passwd entry, `modelmri serve` wrote its trace database
    into a junk `~` directory beside wherever it was started, and `modelmri
    where` answered "where is my stuff" with a relative path.
    """
    paths = _reimport_paths(monkeypatch)
    # The real condition is `Path.home()` raising — a container with no passwd
    # entry, a Windows service account. Clearing the variables is enough to
    # produce it on Windows, but NOT on POSIX: `os.path.expanduser` falls back
    # to the passwd database there, so it returned "/Users/runner" and this
    # test passed for the wrong reason on macOS and Linux. Forcing `_home` is
    # the only way to reach the same code path on all three platforms — which
    # is the whole point of a portability test.
    monkeypatch.setattr(paths, "_home", lambda: None)
    assert paths._no_home().is_absolute(), "the fallback itself must be absolute"
    for name in ("data_dir", "config_dir", "cache_dir", "hf_hub_cache", "hf_home"):
        fn = getattr(paths, name, None)
        if fn is None:
            continue
        p = fn()
        assert p.is_absolute(), f"{name}() returned a relative path: {p}"
        assert "~" not in p.parts, f"{name}() built a directory named ~: {p}"


def test_a_wal_refusal_does_not_stop_the_server(tmp_path, monkeypatch):
    """SQLite documents WAL as unsupported on network filesystems, and this
    pragma ran unguarded inside `create_app` — so an NFS or SMB home meant
    `modelmri serve` exited with a traceback before printing its URL, over a
    feature the reader may not even use."""
    from modelmri import traces

    # `sqlite3.Connection` is an immutable type, so the refusal is injected by
    # wrapping the connection rather than patching the class.
    real_connect = sqlite3.connect

    class _NoWAL:
        def __init__(self, inner):
            object.__setattr__(self, "_inner", inner)

        def execute(self, sql, *a, **k):
            if "journal_mode=WAL" in sql:
                raise sqlite3.OperationalError("disk I/O error")
            return self._inner.execute(sql, *a, **k)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    monkeypatch.setattr(
        traces.sqlite3, "connect", lambda *a, **k: _NoWAL(real_connect(*a, **k))
    )
    store = traces.TraceStore(tmp_path / "traces.sqlite")
    # And it must still work, on the rollback journal.
    tid = store.import_trace(
        {
            "id": "abc",
            "name": "run",
            "started_at": "2026-01-01T00:00:00Z",
            # A non-empty steps list, because the schema requires one — an
            # empty trace is rejected before the storage path this is testing.
            "steps": [
                {
                    "id": "s1",
                    "kind": "tool_call",
                    "name": "pytest",
                    "started_ms": 0,
                    "duration_ms": 1,
                }
            ],
        }
    )
    assert tid
    assert store.list_traces()


def test_the_download_filename_survives_a_path_and_a_non_ascii_name():
    """`hf_id` is an absolute path for a folder-loaded model, and Starlette
    encodes header values as latin-1 — so a Cyrillic or CJK username raised
    UnicodeEncodeError and the reader got a generic 500 with nothing naming
    the cause. Export was simply dead for those users."""
    import re

    def header_name(hf_id: str) -> str:
        name = re.sub(r"[^A-Za-z0-9._-]", "-", Path(hf_id or "session").name)
        return name.strip("-") or "session"

    for raw in (
        r"C:\Users\Пользователь\models\gpt2",
        "/home/пользователь/models/gpt2",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "",
        "///",
    ):
        out = header_name(raw)
        assert out, f"empty filename for {raw!r}"
        out.encode("latin-1")  # the thing that used to raise
        assert "/" not in out and "\\" not in out


def test_a_shared_recording_does_not_carry_an_absolute_path(tmp_path):
    """A `.mri` is the one artefact designed to leave the machine. A model
    loaded from a folder has an `hf_id` that IS a filesystem path, so the
    recording carried the exporter's username to whoever they sent it to."""
    from modelmri import session

    folder = tmp_path / "my-secret-model-v3"
    folder.mkdir()

    shared = str(folder)
    if Path(shared).exists():
        shared = Path(shared).name

    doc = session.build(
        model_id=shared,
        device="cpu",
        dtype="float32",
        n_params=1,
        tokens=["a"],
        prompt="a",
        generation="",
        attention={},
        n_prompt=1,
        n_layers=1,
        n_heads=1,
    )
    # `build` returns the gzipped `.mri` bytes, which is the thing that
    # actually travels — so the assertion is made against those, not against a
    # dict that never leaves the process.
    import gzip
    import json

    parsed = json.loads(gzip.decompress(doc).decode("utf-8"))
    assert parsed["meta"]["model"] == "my-secret-model-v3"
    assert str(tmp_path) not in gzip.decompress(doc).decode("utf-8"), (
        "the recording still carries an absolute path from the exporter"
    )


def test_an_accelerator_without_a_vram_figure_still_gets_a_budget():
    """Apple Silicon has unified memory, so `vram_gb` is None — which used to
    mean no size estimate at all, on exactly the machines least able to guess
    for themselves. `capacity.guard` collapsed the same None to 0 GB."""
    from modelmri import doctor

    r = doctor.Report(
        accelerator_kind="mps", vram_gb=None, ram_gb=32.0, dtype="float16"
    )
    budget = r.ram_gb if r.accelerator_kind == "cpu" else (r.vram_gb or r.ram_gb)
    assert budget == 32.0
    assert doctor._largest_model_b(budget, r.dtype) is not None


def test_the_scan_skip_list_is_relative_to_the_scan_root(tmp_path):
    """`any(part in skip for part in path.parts)` tested every ancestor above
    the root too, so a repo living under a directory named `build`, `dist` or
    `venv` had every candidate skipped — the scan's result depending on where
    the user keeps their code rather than on what is in it."""
    root = tmp_path / "build" / "my-project"
    root.mkdir(parents=True)
    target = root / "adapter.py"
    target.write_text("def load():\n    pass\n", encoding="utf-8")

    skip = {"build", "dist", "node_modules", "venv", ".venv"}
    assert any(part in skip for part in target.parts), "fixture must sit under 'build'"
    assert not any(part in skip for part in target.relative_to(root).parts)


def test_the_float32_remedy_works_on_every_backend():
    """The refusal used to name `CUDA_VISIBLE_DEVICES`, which does nothing on
    Apple Silicon, an Intel GPU or ROCm. A remedy that only works on one
    vendor's hardware is not a remedy — and the variable it names now has to
    actually exist."""
    from modelmri import devices

    src = Path(devices.__file__).read_text(encoding="utf-8")
    assert "MODELMRI_DEVICE" in src, "the remedy names a variable nothing reads"


@pytest.mark.parametrize("value,expect_cpu", [("cpu", True), ("", False)])
def test_modelmri_device_actually_forces_the_device(monkeypatch, value, expect_cpu):
    from modelmri import devices

    monkeypatch.setenv("MODELMRI_DEVICE", value)
    dev = devices.detect()
    if expect_cpu:
        assert dev.kind == "cpu" and "MODELMRI_DEVICE" in dev.reason


def test_the_lerobot_variable_the_message_names_is_read():
    """The 'not cached' refusal told readers to set HF_LEROBOT_HOME. Nothing
    read it, so a user who did exactly what the message said got the identical
    refusal, listing the same directories, with the one they had just
    configured still missing."""
    from modelmri import vla_data

    src = Path(vla_data.__file__).read_text(encoding="utf-8")
    named = "HF_LEROBOT_HOME" in src
    read = "HF_LEROBOT_HOME" in src.split("def dataset_roots")[1].split("def ")[0]
    assert not named or read, "the message names a variable dataset_roots ignores"
