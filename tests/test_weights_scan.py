"""Looking inside a checkpoint before loading it.

The payloads here are REAL. Every dangerous pickle in this file is built with
`pickle.dumps` on an object whose `__reduce__` names something executable —
the same bytes an attacker would ship — and the test asserts the scanner
refuses them. A fixture of hand-written opcodes would test the fixture.

None of them is ever unpickled. `pickletools.genops` walks the stream, and
reading instructions is not running them; if that distinction ever stopped
holding, this test file would be the exploit.

The third verdict is the one worth reading the tests for. `unscanned` is not
`safe`, and the difference is the whole reason this module is trustworthy: a
scanner that answers "clean" for a file it could not open is worse than no
scanner, because the answer is load-bearing and wrong.
"""

from __future__ import annotations

import pickle
import zipfile

import pytest

from modelmri import weights_scan as ws

# ------------------------------------------------------ real payloads


class _RunsAShellCommand:
    """The canonical malicious checkpoint. `os.system` runs on load."""

    def __reduce__(self):
        import os

        return (os.system, ("echo pwned",))


class _RunsPython:
    def __reduce__(self):
        return (eval, ("__import__('os').listdir('.')",))


class _UnwrapsASecondPickle:
    """The payload is inert-looking base64 that a second unpickle expands."""

    def __reduce__(self):
        import base64

        return (base64.b64decode, ("Y29zCnN5c3RlbQo=",))


class _Harmless:
    """A plain object. Pickles to tuples and strings, executes nothing."""

    def __init__(self):
        self.layers = 12
        self.name = "a config"


def _write(tmp_path, name, obj):
    p = tmp_path / name
    p.write_bytes(pickle.dumps(obj))
    return p


# --------------------------------------------------------- the refusals


def test_a_pickle_that_runs_a_shell_command_is_refused(tmp_path):
    """The canonical malicious checkpoint, built with real `pickle.dumps` and
    never unpickled here."""
    report = ws.scan(_write(tmp_path, "pytorch_model.bin", _RunsAShellCommand()))
    assert report.verdict == ws.DANGEROUS
    assert report.findings

    # `nt.system` on Windows, `posix.system` on Linux and macOS — `os.system`
    # IS one of those two, and the pickle records whichever machine wrote it.
    # This assertion was `"os" in detail` and failed on Windows, which is the
    # useful kind of failure: it proves the scanner reads the module the file
    # actually names rather than the one the author typed, and it is why all
    # three are in `DANGEROUS_MODULES`. A checkpoint pickled on Linux and
    # loaded on Windows still says `posix`.
    detail = report.findings[0].detail
    assert any(m in detail for m in ("nt.system", "posix.system", "os.system")), detail
    assert "before a single tensor is read" in report.means()


def test_a_pickle_that_evals_a_string_is_refused(tmp_path):
    report = ws.scan(_write(tmp_path, "model.pt", _RunsPython()))
    assert report.verdict == ws.DANGEROUS
    assert any("eval" in f.detail for f in report.findings)


def test_a_decode_then_execute_chain_is_caught_before_the_second_stage(tmp_path):
    """The outer stream looks dull on purpose — the dangerous names live
    inside a base64 blob a second `pickle.loads` unwraps at load time."""
    report = ws.scan(_write(tmp_path, "weights.ckpt", _UnwrapsASecondPickle()))
    assert report.verdict == ws.DANGEROUS
    assert any("second unpickle" in f.detail for f in report.findings)


def test_an_ordinary_pickle_is_not_flagged(tmp_path):
    """A deny-list that fires on a plain config is a deny-list nobody keeps
    switched on."""
    report = ws.scan(_write(tmp_path, "config.pkl", _Harmless()))
    assert report.verdict == ws.SAFE
    assert report.findings == []


def test_the_dangerous_payload_is_never_executed(tmp_path):
    """The load-bearing claim of the whole module. If scanning ever ran the
    payload, this file would BE the exploit."""
    marker = tmp_path / "pwned.txt"

    class _WritesAFile:
        def __reduce__(self):
            return (open, (str(marker), "w"))

    ws.scan(_write(tmp_path, "evil.bin", _WritesAFile()))
    assert not marker.exists(), "scanning executed the payload"


# ------------------------------------------- torch's real container shape


def test_a_torch_zip_archive_is_walked_entry_by_entry(tmp_path):
    """`torch.save` writes a zip with `data.pkl` inside, which is the shape
    almost every real checkpoint has."""
    p = tmp_path / "pytorch_model.bin"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("archive/data.pkl", pickle.dumps(_RunsAShellCommand()))
        zf.writestr("archive/data/0", b"\x00" * 256)
    report = ws.scan(p)
    assert report.verdict == ws.DANGEROUS
    assert report.format == "torch zip archive"
    assert "data.pkl" in report.findings[0].where


def test_a_clean_torch_zip_archive_passes(tmp_path):
    p = tmp_path / "pytorch_model.bin"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("archive/data.pkl", pickle.dumps(_Harmless()))
        zf.writestr("archive/data/0", b"\x00" * 256)
    assert ws.scan(p).verdict == ws.SAFE


def test_a_zip_bomb_is_caught_from_the_header_without_extracting(tmp_path):
    """Extracting a file to find out whether extracting it is safe is the trap
    this avoids — the ratio comes from the zip's own directory entry."""
    p = tmp_path / "model.ckpt"
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("archive/huge.bin", b"\x00" * (8 * 1024 * 1024))
    report = ws.scan(p)
    assert report.verdict == ws.DANGEROUS
    assert any(f.kind == "zip bomb" for f in report.findings)


def test_an_embedded_executable_is_named(tmp_path):
    p = tmp_path / "weights.pt"
    p.write_bytes(b"MZ\x90\x00" + b"\x00" * 64)
    report = ws.scan(p)
    assert report.verdict == ws.DANGEROUS
    assert "Windows executable" in report.findings[0].detail


def test_a_corrupt_pickle_is_unscanned_rather_than_accused(tmp_path):
    """ "I could not tell" and "I found something" are different answers, and
    only one of them should stop a load.

    This was DANGEROUS at first, which made `guard` raise on it — and a `.pt`
    that is not a pickle at all is common, so the custom-model loader stopped
    before it could produce its real diagnosis ("this is a plain checkpoint,
    not TorchScript, here is what to do") and produced an accusation instead.
    The suite caught it. `unscanned` still refuses to call it clean."""
    p = tmp_path / "broken.pt"
    p.write_bytes(b"\x80\x04\x95\xff\xff\xff\xff\xff\xff\xff\xffnot a pickle")
    report = ws.scan(p)
    assert report.verdict == ws.UNSCANNED
    assert "could not be walked" in report.reason
    assert "may not be a pickle at all" in report.reason
    assert "not a clean bill of health" in report.means()
    # And it does not block a load, which is the point of the change.
    ws.guard(p)


# ------------------------------------------- the third verdict does the work


def test_safetensors_needs_no_scan_and_says_why(tmp_path):
    p = tmp_path / "model.safetensors"
    p.write_bytes(b'{"__metadata__":{}}')
    report = ws.scan(p)
    assert report.verdict == ws.SAFE
    assert "no mechanism in the format" in report.means()


def test_an_unknown_format_is_unscanned_and_never_safe(tmp_path):
    """The rule `gguf_read` already applies to unknown ggml types. A scanner
    that answers "clean" for a file it could not open is worse than no
    scanner, because the answer is the one people act on."""
    p = tmp_path / "weights.h5"
    p.write_bytes(b"\x89HDF\r\n\x1a\n")
    report = ws.scan(p)
    assert report.verdict == ws.UNSCANNED
    assert report.reason
    assert "NOT scanned" in report.means()
    assert "not a clean bill of health" in report.means()


def test_a_missing_file_is_unscanned_with_a_reason(tmp_path):
    report = ws.scan(tmp_path / "nothing-here.bin")
    assert report.verdict == ws.UNSCANNED
    assert "no file at that path" in report.reason


def test_unscanned_always_carries_a_reason(tmp_path):
    """ "unscanned" with an empty reason is indistinguishable from a scan that
    found nothing, which is exactly the confusion this verdict exists to
    prevent."""
    for name, data in (("a.h5", b"\x89HDF"), ("b.onnx", b"\x08\x01"), ("c", b"x")):
        p = tmp_path / name
        p.write_bytes(data)
        report = ws.scan(p)
        if report.verdict == ws.UNSCANNED:
            assert report.reason, f"{name} was unscanned with no reason"


def test_python_source_is_never_called_safe(tmp_path):
    """Every line of an adapter is executable by definition. Listing the
    suspicious ones would imply the rest are fine."""
    p = tmp_path / "adapter.py"
    p.write_text("import torch\n\ndef load(path):\n    return torch.load(path)\n")
    report = ws.scan(p)
    assert report.verdict == ws.UNSCANNED
    assert "runs in full when imported" in report.reason


# ------------------------------------------------------------- the gate


def test_guard_raises_on_a_dangerous_file(tmp_path):
    p = _write(tmp_path, "model.bin", _RunsAShellCommand())
    with pytest.raises(ws.Unsafe) as caught:
        ws.guard(p)
    assert "executes something when it is loaded" in str(caught.value)


def test_guard_can_be_overridden_but_only_out_loud(tmp_path):
    """Their machine, their decision — but it has to be said, which is the
    difference between a refusal and a warning nobody reads."""
    p = _write(tmp_path, "model.bin", _RunsAShellCommand())
    report = ws.guard(p, confirm=True)
    assert report.dangerous, "confirming must not change the verdict"


def test_guard_does_not_raise_on_unscanned(tmp_path):
    """Refusing every format this cannot read would make the scanner a gate on
    its own coverage, and most of what it cannot read is harmless."""
    p = tmp_path / "weights.h5"
    p.write_bytes(b"\x89HDF")
    assert ws.guard(p).verdict == ws.UNSCANNED


def test_guard_passes_a_clean_file_through(tmp_path):
    p = _write(tmp_path, "config.pkl", _Harmless())
    assert ws.guard(p).verdict == ws.SAFE


# ------------------------------------------------------ the better answer


def test_a_repo_offering_both_is_told_to_take_the_safetensors():
    """More useful than any finding this module can produce: the whole class
    of problem disappears if the other file is taken instead."""
    said = ws.prefer_safetensors(["model.safetensors", "pytorch_model.bin"])
    assert "Prefer the safetensors" in said
    assert "no mechanism to execute anything" in said


def test_nothing_is_said_when_there_is_no_safer_choice():
    assert ws.prefer_safetensors(["pytorch_model.bin"]) == ""
    assert ws.prefer_safetensors(["model.safetensors"]) == ""


# ------------------------------------------------------------ directories


def test_a_directory_scan_puts_the_dangerous_files_first(tmp_path):
    _write(tmp_path, "ok.pkl", _Harmless())
    _write(tmp_path, "evil.bin", _RunsAShellCommand())
    (tmp_path / "model.safetensors").write_bytes(b"{}")
    reports = ws.scan_dir(tmp_path)
    assert reports[0].verdict == ws.DANGEROUS
    assert {r.verdict for r in reports} <= {ws.SAFE, ws.DANGEROUS, ws.UNSCANNED}


def test_a_directory_scan_is_bounded(tmp_path):
    for i in range(12):
        (tmp_path / f"w{i}.safetensors").write_bytes(b"{}")
    assert len(ws.scan_dir(tmp_path, limit=5)) == 5


def test_scanning_a_path_that_is_not_a_directory_returns_nothing(tmp_path):
    assert ws.scan_dir(tmp_path / "absent") == []


def test_a_weight_format_it_cannot_read_still_appears_in_a_directory_scan(tmp_path):
    """The first version filtered the walk to formats it understood, so an
    `.h5` beside a poisoned `.pt` simply did not appear — "3 files, all fine"
    when one had never been looked at. Several of these DO carry executable
    payloads (Keras Lambda layers, ONNX custom ops), which makes omitting them
    worse than merely incomplete."""
    _write(tmp_path, "evil.bin", _RunsAShellCommand())
    (tmp_path / "keras_model.h5").write_bytes(b"\x89HDF\r\n\x1a\n")
    (tmp_path / "graph.onnx").write_bytes(b"\x08\x01\x12\x00")

    reports = ws.scan_dir(tmp_path)
    names = {__import__("pathlib").Path(r.path).name for r in reports}
    assert "keras_model.h5" in names, "a weight format was silently stepped over"
    assert "graph.onnx" in names

    h5 = next(r for r in reports if r.path.endswith(".h5"))
    assert h5.verdict == ws.UNSCANNED
    assert "cannot read" in h5.reason
    assert "rather than a reason for comfort" in h5.reason
