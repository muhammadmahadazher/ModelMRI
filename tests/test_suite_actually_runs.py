"""Every `importorskip` in this suite must name something the dev env installs.

`pytest.importorskip` is the right tool for a dependency a USER may not have.
It is the wrong outcome for a dependency the PROJECT tests against, because a
skip is reported as a pass: the file does not run, nothing is red, and the
feature looks covered.

Three suites were dark when this was written, and none of them announced it:

  * `test_grammar.py`      — `lmformatenforcer`, 17 tests. Constrained decoding
                             shipped with its entire suite switched off.
  * `test_hdf5_data.py`    — `h5py`, the ALOHA/robomimic reader.
  * `test_vla_routing.py`  — `pyarrow`, camera routing.

54 tests in total, passing by not existing. The individual `importorskip`
lines are still correct and still belong: they are what lets somebody run the
suite on a base install. What was missing is anything checking that OUR
environment is not the base install.

This is the same rule the rest of the project holds to. A green result from a
check that did not run is worse than a red one, because nobody goes looking.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent

# `importorskip("x")` and `importorskip("x.y")`, first argument only.
_CALL = re.compile(r"importorskip\(\s*[\"']([A-Za-z0-9_.]+)[\"']")


def _guarded() -> dict[str, list[str]]:
    """module name -> the test files that skip on it."""
    out: dict[str, list[str]] = {}
    for path in sorted(TESTS.glob("test_*.py")):
        if path.name == Path(__file__).name:
            continue
        for name in _CALL.findall(path.read_text(encoding="utf-8")):
            out.setdefault(name, []).append(path.name)
    return out


def test_the_sweep_finds_the_guards_at_all():
    """A regex that matched nothing would make every assertion below vacuous —
    which is the failure mode this whole file is about."""
    found = _guarded()
    assert found, "no importorskip calls found; the pattern has gone stale"
    assert "torch" in found, "torch is guarded in several files and was not seen"


@pytest.mark.parametrize("module", sorted(_guarded()))
def test_every_guarded_dependency_is_installed_for_development(module):
    """A skip here means a whole file is not running.

    Fix it by adding the package to the `dev` dependency group in
    pyproject.toml — NOT by removing the `importorskip`, which is what lets a
    user run this suite without the optional extras.
    """
    try:
        importlib.import_module(module)
    except ImportError:  # pragma: no cover - the failure IS the message
        where = ", ".join(_guarded()[module])
        pytest.fail(
            f"{module!r} is not installed, so {where} skips entirely and its "
            f"tests are reported as passing. Add it to the `dev` dependency "
            f"group in pyproject.toml."
        )
