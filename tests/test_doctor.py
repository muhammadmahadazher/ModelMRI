"""The capability report. It runs on machines nobody here has, so it is
written to degrade rather than guess, and these tests are about the degrading.
"""

from __future__ import annotations

import sys
from pathlib import Path

from modelmri import doctor


def test_check_never_raises_even_when_everything_underneath_fails(monkeypatch):
    """A report is more useful than a traceback, and this runs at startup: a
    failure here would stop `modelmri serve` on a machine whose only sin was
    being unusual."""

    def explode(*a, **k):
        raise OSError("no")

    monkeypatch.setattr(doctor.shutil, "disk_usage", explode)
    monkeypatch.setattr(doctor, "_ram_bytes", lambda: None)
    r = doctor.check()
    assert isinstance(r, doctor.Report)
    assert doctor.render(r)
    assert doctor.one_line(r)


def test_an_unmeasurable_number_says_so_rather_than_defaulting(monkeypatch):
    """An invented RAM figure is worse than none, because the sentence built on
    it reads exactly as confidently as a real one."""
    monkeypatch.setattr(doctor, "_ram_bytes", lambda: None)
    r = doctor.check()
    assert r.ram_gb is None
    assert "could not measure" in doctor.render(r)
    # And it must not silently become a size estimate.
    assert "roughly None" not in doctor.render(r)


def test_the_size_estimate_is_arithmetic_on_the_dtype():
    """float32 fits half of what bfloat16 does, because a parameter is four
    bytes rather than two. The CPU path uses float32, so the same machine
    genuinely fits less."""
    big = doctor._largest_model_b(8.0, "bfloat16")
    small = doctor._largest_model_b(8.0, "float32")
    assert big is not None and small is not None
    assert abs(big / small - 2.0) < 0.05, "float32 must halve the estimate"
    # No budget is not a budget of zero.
    assert doctor._largest_model_b(None, "bfloat16") is None
    assert doctor._largest_model_b(0, "bfloat16") is None


def test_a_machine_without_torch_is_told_the_viewer_still_works(monkeypatch):
    """The reason this is a warning and not a refusal to install: `modelmri
    open` deliberately needs no torch and no GPU, so a machine that cannot load
    a model can still read a recording somebody sent."""
    real_import = (
        __builtins__["__import__"]
        if isinstance(__builtins__, dict)
        else __builtins__.__import__
    )

    def no_torch(name, *a, **k):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("No module named torch")
        return real_import(name, *a, **k)

    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr("builtins.__import__", no_torch)
    for mod in [m for m in sys.modules if m.startswith("torch")]:
        monkeypatch.delitem(sys.modules, mod, raising=False)

    r = doctor.check()
    assert r.can_load_models is False
    assert r.can_view_recordings is True
    assert any("PyTorch is not installed" in b for b in r.blockers)
    assert any(".mri" in n for n in r.notes)
    text = doctor.render(r)
    assert "PROBLEM" in text
    assert ".mri recording works on any machine" in text


def test_the_startup_line_stays_one_line():
    """It is printed under the serving banner. A paragraph there would bury the
    URL somebody is trying to click."""
    r = doctor.check()
    line = doctor.one_line(r)
    assert "\n" not in line and len(line) < 160


def test_blockers_decide_the_exit_code(monkeypatch, capsys):
    """`modelmri doctor` is scriptable, so the exit code has to mean something."""

    def blocked():
        return doctor.Report(blockers=["nope"])

    def clean():
        return doctor.Report()

    monkeypatch.setattr(doctor, "check", blocked)
    assert doctor.write_to() == 1
    monkeypatch.setattr(doctor, "check", clean)
    assert doctor.write_to() == 0
    assert "No blockers found" in capsys.readouterr().out


def test_ram_is_read_without_a_third_party_dependency():
    """psutil is not a dependency and must not become one: this package is
    already a 2.5 GB torch install, and the recorder beside it is stdlib-only
    on purpose."""
    from modelmri import devices

    # The probes live in `devices` now, with the other hardware reads. They
    # moved because `doctor` imports `devices` to ask what accelerator is
    # present, and `devices` was importing `_ram_bytes` back — a cycle CodeQL
    # flagged. Reading them here rather than in `doctor` keeps this test about
    # what it was always about: that all three platforms are covered, and that
    # none of it arrived via a third-party package.
    for module in (doctor, devices):
        source = (module.__file__ or "").replace(".pyc", ".py")
        assert "psutil" not in Path(source).read_text(encoding="utf-8")

    text = Path((devices.__file__ or "").replace(".pyc", ".py")).read_text(
        encoding="utf-8"
    )
    # All three branches have to be present, or one platform silently reports
    # "could not measure" forever.
    assert "SC_PHYS_PAGES" in text  # posix
    assert "GlobalMemoryStatusEx" in text  # windows
    assert "hw.memsize" in text  # macos
    # And `doctor` must still be able to answer, wherever the reads live.
    assert doctor._ram_bytes is devices._ram_bytes
