"""Paths must resolve on machines that are not this one.

Every failure here was found by an audit of code I had already tested and
shipped. They share a shape: a location computed correctly in one module and
approximately somewhere else, so the tool downloads to a directory it does
not search, or lists a dataset it then refuses to open. The tests assert
agreement between the modules, not just plausibility within one.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from modelmri import custom, discover, ollama, paths, progress


@pytest.fixture
def clean_env(monkeypatch):
    for var in (
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "HF_HOME",
        "XDG_CACHE_HOME",
        "MODELMRI_HOME",
        "MODELMRI_MODELS_DIR",
        "HF_LEROBOT_HOME",
        "OLLAMA_HOST",
    ):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


# ------------------------------------------------- importable without a home


def test_import_survives_an_unresolvable_home():
    """Containers run as a UID with no passwd entry. Import must not die.

    A module-level `Path.home()` fires before MODELMRI_HOME can be read, so
    the documented escape hatch for exactly this situation never gets a turn.

    The HuggingFace variables are stripped as well, and that is the whole
    difference between this test finding the bug and not. With HF_HOME set,
    `hf_home()` returns at its first branch and never reaches
    `_hub_constant` — the function that actually raised. A developer with
    HF_HOME exported saw this pass; windows-latest, which has none, saw it
    fail. The test was measuring the machine's environment, not the code.

    Windows is where it bites: `os.path.expanduser` does not raise, it hands
    back "~/.cache" unexpanded, and `Path.expanduser()` then raises
    RuntimeError with no passwd database to fall back on.
    """
    env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in (
            "HOME",
            "USERPROFILE",
            "HOMEPATH",
            "HOMEDRIVE",
            "HF_HOME",
            "HF_HUB_CACHE",
            "HUGGINGFACE_HUB_CACHE",
            "XDG_CACHE_HOME",
        )
    }
    env["MODELMRI_HOME"] = str(Path(__file__).parent)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            "import modelmri.paths as p; print(p.describe()['data'])",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr


# ------------------------------------------------- one cache, one answer


def test_hf_hub_cache_honours_the_legacy_variable(clean_env, tmp_path):
    """huggingface_hub still reads HUGGINGFACE_HUB_CACHE; so must we."""
    clean_env.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "legacy"))
    assert paths.hf_hub_cache() == tmp_path / "legacy"


def test_hf_hub_cache_prefers_the_current_variable(clean_env, tmp_path):
    clean_env.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "legacy"))
    clean_env.setenv("HF_HUB_CACHE", str(tmp_path / "current"))
    assert paths.hf_hub_cache() == tmp_path / "current"


def test_every_module_resolves_the_cache_the_same_way(clean_env, tmp_path):
    """The bug class this whole module exists to prevent."""
    clean_env.setenv("HF_HUB_CACHE", str(tmp_path / "elsewhere"))
    from modelmri import vla

    expected = tmp_path / "elsewhere"
    assert paths.hf_hub_cache() == expected
    assert progress.hub_cache() == expected
    assert vla.hub_root() == expected


def test_a_blank_cache_variable_does_not_mean_the_working_directory(clean_env):
    clean_env.setenv("HF_HUB_CACHE", "")
    assert paths.hf_hub_cache() != Path("")
    assert progress.hub_cache() != Path("")


def test_a_tilde_cache_is_expanded_not_taken_literally(clean_env):
    clean_env.setenv("HF_HUB_CACHE", "~/hf-cache")
    assert paths.hf_hub_cache() == Path.home() / "hf-cache"
    assert progress.hub_cache() == Path.home() / "hf-cache"


def test_the_scan_root_is_the_cache_itself_not_its_parent(clean_env, tmp_path):
    """`HF_HUB_CACHE=D:\\hf` made the scan root `D:\\` — the whole drive."""
    cache = tmp_path / "hf"
    (cache / "models--x--y").mkdir(parents=True)
    clean_env.setenv("HF_HUB_CACHE", str(cache))
    clean_env.setenv("HF_HOME", str(tmp_path / "unrelated-home"))
    resolved = [r.resolve() for r in discover.roots()]
    assert cache.resolve() in resolved
    assert tmp_path.resolve() not in resolved


# ------------------------------------------------- MODELMRI_MODELS_DIR


def test_models_dir_expands_a_tilde(clean_env):
    """Unexpanded, `~/models` resolves to `<cwd>/~/models` and matches nothing."""
    clean_env.setenv("MODELMRI_MODELS_DIR", "~/models")
    assert paths.models_dirs() == [Path.home() / "models"]


def test_models_dir_expands_environment_variables(clean_env, tmp_path):
    clean_env.setenv("SOME_ROOT", str(tmp_path))
    var = "%SOME_ROOT%" if sys.platform == "win32" else "$SOME_ROOT"
    clean_env.setenv("MODELMRI_MODELS_DIR", var)
    assert paths.models_dirs() == [tmp_path]


def test_models_dir_splits_on_the_platform_separator(clean_env, tmp_path):
    clean_env.setenv(
        "MODELMRI_MODELS_DIR",
        os.pathsep.join([str(tmp_path / "a"), str(tmp_path / "b")]),
    )
    assert paths.models_dirs() == [tmp_path / "a", tmp_path / "b"]


def test_models_dir_ignores_empty_entries(clean_env, tmp_path):
    clean_env.setenv("MODELMRI_MODELS_DIR", f"{tmp_path}{os.pathsep}{os.pathsep}  ")
    assert paths.models_dirs() == [tmp_path]


def test_the_adapter_loader_and_the_scanner_agree_on_the_roots(clean_env, tmp_path):
    """One expanded `~`, the other did not; adapters were refused from a
    directory the picker had just listed."""
    home_dir = Path.home() / ".modelmri-portability-probe"
    clean_env.setenv("MODELMRI_MODELS_DIR", "~/.modelmri-portability-probe")
    assert home_dir.resolve() in [r.resolve() for r in custom.allowed_roots()]
    assert home_dir in paths.models_dirs()


# ------------------------------------------------- ollama


def test_ollama_host_is_read_from_the_environment(clean_env):
    """Read at call time — an import-time constant cannot see a later setenv."""
    clean_env.setenv("OLLAMA_HOST", "http://gpu-box:11434")
    assert ollama.default_host() == "http://gpu-box:11434"


def test_a_bare_host_and_port_becomes_a_url(clean_env):
    clean_env.setenv("OLLAMA_HOST", "gpu-box:11434")
    assert ollama.default_host() == "http://gpu-box:11434"


def test_ollama_falls_back_to_localhost(clean_env):
    assert ollama.default_host() == "http://127.0.0.1:11434"


# ------------------------------------------------- `modelmri where` is true


def test_describe_names_the_files_not_only_the_directories(clean_env, tmp_path):
    """It reported a directory while the caller read a different file."""
    clean_env.setenv("MODELMRI_HOME", str(tmp_path))
    info = paths.describe()
    assert info["trace_db"] == str(paths.trace_db_path())
    assert info["hub_token"] == str(paths.token_path())
    assert info["trace_db"].endswith(".sqlite")


def test_describe_reports_the_configured_model_dirs(clean_env, tmp_path):
    clean_env.setenv("MODELMRI_MODELS_DIR", str(tmp_path))
    assert paths.describe()["models_dirs"] == [str(tmp_path)]


def test_describe_survives_no_home(clean_env, tmp_path, monkeypatch):
    clean_env.setenv("MODELMRI_HOME", str(tmp_path))

    def boom():
        raise RuntimeError("Could not determine home directory.")

    monkeypatch.setattr(Path, "home", staticmethod(boom))
    info = paths.describe()
    assert info["legacy"] is None
    assert info["data"] == str(tmp_path / "data")


# ------------------------------------------------- the token file


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_the_token_is_never_world_readable_even_for_an_instant(tmp_path, monkeypatch):
    """It was created at 0644 and narrowed afterwards. On a shared host that
    window is enough to read someone's HuggingFace token."""
    from modelmri import hub

    target = tmp_path / "hub.json"
    seen: list[int] = []
    real_open = os.open

    def watched(path, flags, mode=0o777, *a, **kw):
        fd = real_open(path, flags, mode, *a, **kw)
        # The write is atomic, so the descriptor that matters belongs to the
        # temp file (`hub.json.<pid>.tmp`) that is then renamed into place —
        # matching the target name exactly saw nothing at all, and the test
        # failed for the code being correct.
        if str(path).startswith(str(target)):
            seen.append(os.fstat(fd).st_mode & 0o777)
        return fd

    monkeypatch.setattr(os, "open", watched)
    old = os.umask(0o022)
    try:
        hub._write_private(target, json.dumps({"token": "hf_secret"}))
    finally:
        os.umask(old)

    assert target.read_text(encoding="utf-8")
    assert seen, "the token file was not created through os.open with a mode"
    assert all(m == 0o600 for m in seen), f"created world-readable: {seen}"
    assert target.stat().st_mode & 0o077 == 0


def test_a_failed_token_write_leaves_the_old_one_intact(tmp_path, monkeypatch):
    """Non-atomic writes left truncated JSON that the reader silently ate,
    signing the user out with no message."""
    from modelmri import hub

    target = tmp_path / "hub.json"
    hub._write_private(target, json.dumps({"token": "first"}))

    real_replace = os.replace
    monkeypatch.setattr(
        os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full"))
    )
    with pytest.raises(OSError):
        hub._write_private(target, json.dumps({"token": "second"}))
    monkeypatch.setattr(os, "replace", real_replace)

    assert json.loads(target.read_text(encoding="utf-8"))["token"] == "first"
    assert not list(tmp_path.glob("*.tmp*")), "left a temp file behind"


def test_the_docstring_does_not_promise_a_path_we_stopped_using():
    """SECURITY.md and this docstring both claimed ~/.modelmri/hub.json."""
    from modelmri import hub

    root = Path(__file__).resolve().parents[1]
    texts = {
        "hub.py docstring": hub.__doc__ or "",
        "SECURITY.md": (root / "SECURITY.md").read_text(encoding="utf-8"),
        "getting-started": (root / "docs" / "getting-started.md").read_text(
            encoding="utf-8"
        ),
    }
    for where, text in texts.items():
        assert "~/.modelmri/hub.json" not in text, (
            f"{where} still points at the pre-0.6 location"
        )


def test_an_undelivered_trace_never_lands_in_the_working_directory(
    tmp_path, monkeypatch
):
    """The recorder is imported by the user's agent, so the CWD is their repo.

    A trace holds full prompts and tool output. Dropping one there put an
    untracked transcript of their conversations a `git add -A` away from a
    public push.
    """
    monkeypatch.delenv("MODELMRI_TRACE_DIR", raising=False)
    monkeypatch.setenv("MODELMRI_HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)

    from modelmri.record import step, trace

    with trace("cwd-check", endpoint="http://127.0.0.1:1/nope"):
        step("llm_call", name="a", duration_ms=1)

    assert not list(tmp_path.glob("modelmri-traces")), "wrote into the CWD"
    assert not list(tmp_path.glob("*.json"))
    parked = list((tmp_path / "home" / "data" / "undelivered").glob("*.json"))
    assert len(parked) == 1, "the trace was not kept anywhere"
    assert parked[0].parent == paths.undelivered_traces()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows has no POSIX modes")
def test_windows_does_not_claim_a_protection_it_cannot_apply():
    """`chmod(0600)` on Windows toggles the read-only bit and nothing else."""
    from modelmri import hub

    root = Path(__file__).resolve().parents[1]
    security = (root / "SECURITY.md").read_text(encoding="utf-8").lower()
    assert "0600" in security
    assert "windows" in security, "the POSIX-only caveat is not stated anywhere"
    assert "posix" in (hub.__doc__ or "").lower()
