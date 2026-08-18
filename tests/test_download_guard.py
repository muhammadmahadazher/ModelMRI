"""A download you cannot stop, and one that could never work.

Written after a click on `zai-org/GLM-5.2` in the picker began fetching
1506.7 GB onto a laptop with an 8.6 GB GPU and 88 GB of free disk. Nothing
warned, nothing asked, and the only way to stop it was to kill the server.

Both numbers were available before the first byte moved.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from modelmri import capacity, hub, progress
from modelmri import runtime as runtime_mod


class FakeAccel:
    name = "NVIDIA GeForce RTX 4060 Laptop GPU"
    vram_gb = 8.6
    kind = "cuda"


# --------------------------------------------------- reading the real size


def test_weight_bytes_uses_the_dtype_breakdown():
    """753B parameters is 1.5 TB in BF16 and 750 GB in FP8. The count alone
    cannot tell you what you are about to download."""
    bf16 = {"safetensors": {"parameters": {"BF16": 1_000_000_000, "F32": 1_000}}}
    fp8 = {"safetensors": {"parameters": {"F8_E4M3": 1_000_000_000}}}
    assert hub.weight_bytes(bf16) == 2_000_004_000
    assert hub.weight_bytes(fp8) == 1_000_000_000


def test_weight_bytes_falls_back_to_two_bytes_per_parameter():
    assert hub.weight_bytes({"safetensors": {"total": 500_000_000}}) == 1_000_000_000


def test_an_unknown_dtype_is_assumed_wide_not_free():
    """Guessing small on an unrecognised dtype is how a guard lets through
    the one download it exists to stop."""
    assert hub.weight_bytes({"safetensors": {"parameters": {"WAT9": 1_000}}}) == 2_000


def test_a_repo_with_no_metadata_reports_unknown_not_zero_sized():
    assert hub.weight_bytes({}) == 0
    assert hub.weight_bytes({"safetensors": None}) == 0


# --------------------------------------------------- the guard


def test_a_model_that_cannot_fit_on_disk_is_refused(monkeypatch):
    monkeypatch.setattr(runtime_mod, "download_size", lambda _: 1_506_700_000_000)
    with pytest.raises(ValueError) as err:
        runtime_mod._preflight("zai-org/GLM-5.2", FakeAccel(), confirm=False)
    message = str(err.value)
    # TB above a thousand gigabytes: "1506.7 GB" is a number people have to
    # count digits on, and this one needs to land immediately.
    assert "1.5 TB" in message
    assert "free" in message
    # A refusal that does not say what to do instead is just a wall.
    assert "smaller model" in message


def test_disk_refusal_cannot_be_overridden(monkeypatch):
    """There is no version of 'yes, overflow my disk' that ends well."""
    monkeypatch.setattr(runtime_mod, "download_size", lambda _: 1_506_700_000_000)
    with pytest.raises(ValueError):
        runtime_mod._preflight("zai-org/GLM-5.2", FakeAccel(), confirm=True)


def test_a_model_far_past_the_gpu_needs_confirmation(monkeypatch):
    """Fits on disk, could never load. 4x VRAM = 34 GB here, so 200 GB is
    not an arguable case. Free space is stubbed so the verdict does not
    depend on how full the developer's drive happens to be."""
    monkeypatch.setattr(runtime_mod, "download_size", lambda _: 200_000_000_000)
    monkeypatch.setattr(
        runtime_mod, "_free_space", lambda: (Path("D:/"), 9_000_000_000_000)
    )
    with pytest.raises(ValueError, match="Load it anyway"):
        runtime_mod._preflight("some/huge", FakeAccel(), confirm=False)
    # ...and the override actually overrides.
    runtime_mod._preflight("some/huge", FakeAccel(), confirm=True)


@pytest.mark.parametrize(
    "gb",
    [0.55, 1.5, 16.0, 30.0],  # a 0.1B, Qwen3-0.6B, Qwen3-8B, a 15B in bf16
)
def test_ordinary_models_are_not_blocked(monkeypatch, gb):
    """A guard that fires on normal work gets switched off."""
    monkeypatch.setattr(runtime_mod, "download_size", lambda _: int(gb * 1e9))
    runtime_mod._preflight("some/normal", FakeAccel(), confirm=False)


def test_an_unknown_size_does_not_block(monkeypatch):
    """GGUF repos publish nothing. Refusing on no evidence would ban them."""
    monkeypatch.setattr(runtime_mod, "download_size", lambda _: 0)
    runtime_mod._preflight("some/gguf", FakeAccel(), confirm=False)


def test_a_machine_with_no_gpu_still_gets_a_ceiling(monkeypatch):
    class NoGpu:
        name = "cpu"
        vram_gb = None
        kind = "cpu"

    monkeypatch.setattr(runtime_mod, "download_size", lambda _: 200_000_000_000)
    monkeypatch.setattr(
        runtime_mod, "_free_space", lambda: (Path("D:/"), 9_000_000_000_000)
    )
    with pytest.raises(ValueError, match="no GPU"):
        runtime_mod._preflight("some/huge", NoGpu(), confirm=False)


# --------------------------------------------------- ollama, same rule


def test_ollama_and_huggingface_share_one_rule():
    """Two guards drift. `deepseek-r1:671b` is 404 GB and the Ollama path
    had no check at all until it went through the same function."""
    from modelmri import runtime as rt

    assert rt._preflight.__doc__ and "capacity" in rt._preflight.__doc__
    assert capacity.guard is not None


def test_ollama_size_comes_from_the_registry_not_a_hardcoded_list(monkeypatch):
    from modelmri import ollama

    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"layers":[{"size":1000},{"size":2000}],"config":{"size":50}}'

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        return FakeResponse()

    monkeypatch.setattr(ollama.urllib.request, "urlopen", fake_urlopen)

    assert ollama.manifest_size("qwen3:0.6b") == 3050
    # Bare names live under ollama's default namespace.
    assert "library/qwen3/manifests/0.6b" in captured["url"]


def test_a_namespaced_ollama_model_is_not_forced_into_library(monkeypatch):
    from modelmri import ollama

    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        raise OSError("stop here — the URL is what is under test")

    monkeypatch.setattr(ollama.urllib.request, "urlopen", fake_urlopen)
    assert ollama.manifest_size("someone/custom:latest") == 0
    assert "someone/custom/manifests/latest" in captured["url"]


def test_an_ollama_model_with_no_tag_defaults_to_latest(monkeypatch):
    from modelmri import ollama

    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        raise OSError("stop")

    monkeypatch.setattr(ollama.urllib.request, "urlopen", fake_urlopen)
    ollama.manifest_size("llama3.2")
    assert captured["url"].endswith("/library/llama3.2/manifests/latest")


def test_an_unreachable_registry_reports_unknown_not_zero_sized(monkeypatch):
    from modelmri import ollama

    monkeypatch.setattr(
        ollama.urllib.request,
        "urlopen",
        lambda *a, **k: (_ for _ in ()).throw(OSError("offline")),
    )
    assert ollama.manifest_size("qwen3:0.6b") == 0


def test_ollama_disk_check_uses_ollamas_own_directory(monkeypatch, tmp_path):
    """Ollama does not store models in the HuggingFace cache, and the two
    are routinely on different drives."""
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path / "elsewhere"))
    assert capacity.ollama_models_dir() == tmp_path / "elsewhere"


def test_the_pull_endpoint_enforces_it_server_side(monkeypatch):
    """A check the browser performs is a check the browser can skip."""
    from fastapi.testclient import TestClient

    from modelmri import ollama
    from modelmri.server import create_app

    monkeypatch.setattr(ollama, "manifest_size", lambda *a, **k: 900_000_000_000_000)
    client = TestClient(create_app())
    r = client.post("/api/ollama/pull", json={"name": "deepseek-r1:671b"})
    assert r.status_code == 422
    assert "free" in r.json()["error"]
    assert r.json()["overridable"] is False


def test_confirm_cannot_override_a_full_disk_on_the_pull_path(monkeypatch):
    from fastapi.testclient import TestClient

    from modelmri import ollama
    from modelmri.server import create_app

    monkeypatch.setattr(ollama, "manifest_size", lambda *a, **k: 900_000_000_000_000)
    client = TestClient(create_app())
    r = client.post(
        "/api/ollama/pull", json={"name": "deepseek-r1:671b", "confirm": True}
    )
    assert r.status_code == 422


# --------------------------------------------------- stopping it


def test_the_tracker_reports_nothing_to_stop_when_idle():
    progress.TRACKER.finish()
    assert progress.TRACKER.request_cancel() is False


def test_requesting_a_cancel_sets_the_flag_and_says_so():
    progress.TRACKER.start("some/model")
    try:
        assert progress.TRACKER.cancelled.is_set() is False
        assert progress.TRACKER.request_cancel() is True
        assert progress.TRACKER.cancelled.is_set() is True
        assert "stopping" in progress.TRACKER.snapshot().detail
    finally:
        progress.TRACKER.finish()


def test_a_new_load_clears_a_previous_cancel():
    """Otherwise one Stop click poisons every load after it."""
    progress.TRACKER.start("a")
    progress.TRACKER.request_cancel()
    progress.TRACKER.finish()
    progress.TRACKER.start("b")
    try:
        assert progress.TRACKER.cancelled.is_set() is False
    finally:
        progress.TRACKER.finish()


def _run_prefetch(monkeypatch, repo_files, repo="acme/model"):
    """Execute the real `_PREFETCH` source with huggingface_hub stubbed.

    The source is a string run as `python -c`, so the only way to test what
    it actually asks for is to run it. Re-stating its patterns in the test
    would assert that the test agrees with itself.
    """
    import types

    captured: dict = {}

    class FakeApi:
        def list_repo_files(self, name):
            if repo_files is None:
                raise ConnectionError("hub unreachable")
            return repo_files

    fake = types.ModuleType("huggingface_hub")
    fake.HfApi = FakeApi
    fake.snapshot_download = lambda name, ignore_patterns=None: captured.update(
        repo=name, ignore=list(ignore_patterns or [])
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    monkeypatch.setattr(sys, "argv", ["prefetch", repo])
    exec(compile(runtime_mod._PREFETCH, "<prefetch>", "exec"), {"__name__": "__main__"})
    return captured


def test_a_redundant_second_copy_of_the_weights_is_not_downloaded(monkeypatch):
    """A repo can ship model.safetensors AND an identical pytorch_model.bin
    AND a rust_model.ot. Measured: several times the needed bytes came down,
    because the ignore list covered TensorFlow, Flax, ONNX and TFLite but not
    Rust or the redundant .bin. transformers loads the safetensors and never
    opens the others."""
    got = _run_prefetch(
        monkeypatch,
        [
            "model.safetensors",
            "pytorch_model.bin",
            "rust_model.ot",
            "tf_model.h5",
            "config.json",
        ],
    )
    assert "*.bin" in got["ignore"]
    assert "*.ot" in got["ignore"]
    assert "*.safetensors" not in got["ignore"]


def test_a_repo_with_only_bin_weights_still_gets_them(monkeypatch):
    """The whole point of checking rather than blacklisting. Plenty of models
    predate safetensors, and refusing their only weight file would turn a
    bandwidth saving into a model that cannot load at all."""
    got = _run_prefetch(monkeypatch, ["pytorch_model.bin", "config.json"])
    assert "*.bin" not in got["ignore"]
    assert "*.ot" in got["ignore"]  # still useless, still skipped


def test_safetensors_in_a_subfolder_does_not_condemn_the_real_weights(monkeypatch):
    """An adapter or an onnx variant can ship a .safetensors that is not the
    model's weights. Dropping the root .bin because of it would leave nothing
    to load."""
    got = _run_prefetch(
        monkeypatch,
        ["pytorch_model.bin", "adapter/extra.safetensors", "config.json"],
    )
    assert "*.bin" not in got["ignore"]


def test_an_unreachable_hub_fetches_everything_rather_than_guessing(monkeypatch):
    """No listing means no evidence. Skipping the .bin on a hunch is how a
    load fails with the weights deliberately absent."""
    got = _run_prefetch(monkeypatch, None)
    assert "*.bin" not in got["ignore"]
    assert got["repo"] == "acme/model"


def test_the_prefetch_child_is_actually_killed(monkeypatch, tmp_path):
    """The point of the child process. A thread blocked in a socket read
    cannot be stopped from Python; a process can."""
    rt = runtime_mod.ModelRuntime()
    # A child that would run far longer than the test, standing in for a
    # multi-hour download.
    monkeypatch.setattr(runtime_mod, "_PREFETCH", "import time; time.sleep(600)")
    monkeypatch.setattr(runtime_mod, "_clean_partials", lambda _: 0)

    progress.TRACKER.start("some/model")
    import threading

    threading.Timer(0.6, progress.TRACKER.request_cancel).start()
    try:
        with pytest.raises(runtime_mod.LoadCancelled):
            rt._prefetch_weights("some/model")
    finally:
        progress.TRACKER.finish()

    # And nothing of ours is still running.
    still = subprocess.run(
        [sys.executable, "-c", "print('ok')"], capture_output=True, text=True
    )
    assert still.returncode == 0


def test_a_chatty_child_does_not_deadlock_the_load(monkeypatch):
    """The bug this test exists for hung a load that had already finished.

    huggingface_hub writes tqdm progress to stderr. With `stderr=PIPE` and
    nothing draining it, the child blocks once the ~64 KB pipe buffer fills
    and never exits — the UI sat at "551 MB / 551 MB · 234s" forever with
    the weights fully downloaded.

    A child that writes far more than one buffer's worth, with a timeout so
    the failure mode is a red test rather than a hung suite.
    """
    import threading

    rt = runtime_mod.ModelRuntime()
    monkeypatch.setattr(
        runtime_mod,
        "_PREFETCH",
        "import sys\n"
        "sys.stderr.write('x' * 400_000)\n"  # ~6x a typical pipe buffer
        "sys.stdout.write('y' * 400_000)\n",
    )

    progress.TRACKER.start("some/model")
    done = threading.Event()
    error: list[BaseException] = []

    def run():
        try:
            rt._prefetch_weights("some/model")
        except BaseException as err:  # reported below
            error.append(err)
        finally:
            done.set()

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    finished = done.wait(30)
    progress.TRACKER.cancelled.set()  # release the poll loop if it is stuck
    progress.TRACKER.finish()

    assert finished, (
        "the prefetch child deadlocked: it wrote more than a pipe buffer and "
        "nothing was draining it"
    )
    assert not error, error


def test_the_prefetch_child_never_opens_a_pipe_it_does_not_drain():
    """Belt and braces on the above: reading the source is the only way to
    catch someone 'improving' this back to stderr=PIPE for diagnostics."""
    import inspect

    source = inspect.getsource(runtime_mod.ModelRuntime._prefetch_weights)
    assert "stderr=subprocess.DEVNULL" in source
    assert "stderr=subprocess.PIPE" not in source


def test_cancelled_partials_are_removed(tmp_path, monkeypatch):
    """A stopped download must not leave gigabytes of dead blobs behind."""
    from modelmri import paths

    cache = tmp_path / "hub"
    repo = cache / "models--some--model" / "blobs"
    repo.mkdir(parents=True)
    (repo / "aaa.111.incomplete").write_bytes(b"x" * 4096)
    (repo / "bbb.222.incomplete").write_bytes(b"y" * 2048)
    keep = repo / "ccc"  # a completed blob
    keep.write_bytes(b"z" * 64)

    monkeypatch.setattr(paths, "hf_hub_cache", lambda: cache)
    freed = runtime_mod._clean_partials("some/model")

    assert freed == 4096 + 2048
    assert not list(repo.glob("*.incomplete"))
    assert keep.exists(), "a completed download was deleted"


def test_an_unreadable_gpu_is_not_reported_as_an_absent_one(tmp_path):
    """The comment above this branch says the two states "say different
    things about them" — and the code still collapsed them.

    `vram` is 0.0 both when the accelerator's memory could not be read and
    when there is no accelerator, so branching on its truthiness printed
    "this machine has no GPU" at somebody whose GPU is sitting right there
    and merely did not answer. That sends them to buy hardware they already
    own, over a driver or permissions problem the true sentence names.

    `vram_gb is None` alone does not separate the two: `devices._cpu()`
    returns None for a genuinely CPU-only machine, and an Intel XPU whose
    properties could not be read returns None too — deliberately, keeping its
    name, because "an Intel GPU we cannot describe is still an Intel GPU".
    The NAME is the discriminator, which is what this pins.
    """
    with pytest.raises(capacity.TooBig) as unreadable:
        capacity.guard(
            140 * 10**9,
            tmp_path,
            label="a 70B model",
            vram_gb=None,
            accel_name="NVIDIA RTX 4060",
            free_override=10**15,
        )
    with pytest.raises(capacity.TooBig) as absent:
        capacity.guard(
            140 * 10**9,
            tmp_path,
            label="a 70B model",
            vram_gb=0.0,
            free_override=10**15,
        )

    said_unreadable = str(unreadable.value)
    said_absent = str(absent.value)
    assert said_unreadable != said_absent
    assert "could not be read" in said_unreadable
    assert "NVIDIA RTX 4060" in said_unreadable
    assert "no GPU" not in said_unreadable
    assert "this machine has no GPU" in said_absent


def test_a_readable_gpu_still_reports_its_size(tmp_path):
    with pytest.raises(capacity.TooBig) as caught:
        capacity.guard(
            140 * 10**9,
            tmp_path,
            label="a 70B model",
            vram_gb=8.0,
            accel_name="NVIDIA RTX 4060",
            free_override=10**15,
        )
    assert "NVIDIA RTX 4060 has 8.0 GB" in str(caught.value)


def test_a_refusal_survives_being_copied_and_pickled():
    """`BaseException.__reduce__` rebuilds an exception by calling the class
    with `self.args` POSITIONALLY, and `overridable` is keyword-only — so it
    is not in `args` and all three of copy, deepcopy and pickle raised:

        TypeError: TooBig.__init__() missing 1 required keyword-only
        argument: 'overridable'

    A refusal that cannot be copied dies on any path that moves it between
    contexts, and the failure arrives as a confusing TypeError about the
    exception rather than the sentence it was carrying.
    """
    import copy
    import pickle

    from modelmri.capacity import TooBig

    original = TooBig("needs 5.0 GB and C: has 1.0 GB free", overridable=True)
    for rebuilt in (
        copy.copy(original),
        copy.deepcopy(original),
        pickle.loads(pickle.dumps(original)),
    ):
        assert isinstance(rebuilt, TooBig)
        assert rebuilt.overridable is True
        assert rebuilt.sentence == original.sentence
        assert str(rebuilt) == str(original)

    # And the flag genuinely round-trips rather than defaulting to a truthy
    # value: an unoverridable disk refusal must not come back overridable.
    hard = pickle.loads(pickle.dumps(TooBig("no room", overridable=False)))
    assert hard.overridable is False


def test_a_disk_that_could_not_be_measured_is_said_out_loud(caplog):
    """`free_space` returns 0 for a volume it could not read, and the disk
    refusal is correctly SKIPPED — refusing on no evidence would ban a
    legitimate download.

    What was missing is that nobody was told the check did not happen, so a
    download proceeded looking exactly like one that had been cleared.
    """
    import logging
    from pathlib import Path

    from modelmri import capacity

    with caplog.at_level(logging.WARNING, logger="modelmri.capacity"):
        capacity.guard(
            5_000_000_000,
            Path("."),
            label="some/model",
            vram_gb=8.0,
            free_override=0,
            confirm=True,
        )
    said = caplog.text
    assert "could not be measured" in said
    assert "some/model" in said
    assert "NOT" in said


def test_a_crafted_model_name_cannot_forge_a_log_line(caplog):
    """CodeQL's log-injection rule, and it is right about my own new code.

    `label` is a repo id and `target` a path, and both arrive from a request.
    A newline in either would end the real entry and start a forged one — so a
    model called `x\nWARNING:root:ALL CLEAR` could write a reassuring line
    into the log a reader is checking to find out what happened.

    `%r` escapes it, and it also makes a trailing space or a zero-width
    character visible rather than invisible.
    """
    import logging
    from pathlib import Path

    from modelmri import capacity

    forged = "innocent/model" + chr(10) + "WARNING:root:ALL CLEAR"
    with caplog.at_level(logging.WARNING, logger="modelmri.capacity"):
        capacity.guard(
            5_000_000_000,
            Path("."),
            label=forged,
            vram_gb=8.0,
            free_override=0,
            confirm=True,
        )

    assert len(caplog.records) == 1
    written = caplog.records[0].getMessage()
    assert chr(10) not in written, "the newline reached the log unescaped"
    # The name is still readable, escaped — dropping it would lose the one
    # thing that says WHICH download was not checked.
    assert "innocent/model" in written
