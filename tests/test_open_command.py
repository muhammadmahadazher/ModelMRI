"""`modelmri open` must stay cheap, and must stay a file reader.

The command exists so that someone sent a 54 KB recording can read it. It
used to start the full application, which imports torch and transformers —
26 seconds on a cold cache — for an analysis that needs neither. The first
person to run it pressed ctrl-c partway through, reasonably concluding it
had hung.

The fix was to serve the bundled viewer from the standard library. These
tests keep it that way: the expensive imports are the kind of thing that
creeps back in through one convenient top-level import.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "modelmri" / "static" / "viewer"


def run(code: str) -> subprocess.CompletedProcess:
    import os

    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )


def test_the_open_path_does_not_import_torch():
    """The whole point. 26 seconds to read a 54 KB file is not a wait, it is
    a reason to give up — and somebody did."""
    done = run(
        "import sys\n"
        "from modelmri import cli, session\n"
        "heavy = sorted(m for m in ('torch', 'transformers', 'fastapi', "
        "'uvicorn', 'numpy') if m in sys.modules)\n"
        "print(','.join(heavy))"
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "", (
        f"`modelmri open` now imports {done.stdout.strip()} — the command "
        "exists to avoid exactly that"
    )


def test_parsing_a_session_needs_nothing_heavy():
    done = run(
        "import sys\n"
        "from modelmri import session\n"
        "assert 'torch' not in sys.modules\n"
        "print('ok')"
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ok"


@pytest.mark.skipif(
    not (BUNDLE / "index.html").is_file(),
    reason="no bundled viewer (run scripts/build_frontend.py --viewer)",
)
def test_the_viewer_bundle_is_self_contained():
    """It is served by a stdlib http.server with no API behind it, so a
    reference to an external host would simply fail in the page."""
    index = (BUNDLE / "index.html").read_text(encoding="utf-8")
    assert "assets/" in index
    for bad in ("http://", "https://"):
        assert bad not in index, f"index.html reaches out to {bad}"
    assert list(BUNDLE.glob("assets/*.js")), "no bundle emitted"


@pytest.mark.skipif(not (BUNDLE / "index.html").is_file(), reason="no bundled viewer")
def test_the_viewer_serves_and_hands_over_the_file(tmp_path):
    """End to end through the real command, with a real socket."""
    import http.client
    import threading
    import time

    from modelmri import cli, session

    mri = tmp_path / "probe.mri"
    mri.write_bytes(
        session.build(
            model_id="t/t",
            device="cpu",
            dtype="float32",
            n_params=1,
            tokens=["a", "b"],
            prompt="a",
            generation="b",
            attention={(0, 0): [[1.0, 0.0], [0.5, 0.5]]},
            n_layers=1,
            n_heads=1,
        )
    )

    port = 5934
    thread = threading.Thread(
        target=cli.serve_viewer,
        args=(mri,),
        kwargs={"host": "127.0.0.1", "port": port, "browser": False},
        daemon=True,
    )
    thread.start()

    deadline = time.monotonic() + 30
    conn = None
    while time.monotonic() < deadline:
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("GET", "/")
            if conn.getresponse().status == 200:
                break
        except OSError:
            time.sleep(0.1)
    else:
        pytest.fail("the viewer never came up")

    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", "/session.mri")
    resp = conn.getresponse()
    assert resp.status == 200
    served = resp.read()
    assert served == mri.read_bytes(), "the served file is not the one named"
    # And it round-trips, so what the page will parse is a real session.
    assert session.parse(served).tokens == ["a", "b"]


def test_a_file_that_is_not_a_session_never_starts_a_server(tmp_path):
    """One sentence and exit 2, before anything binds a port."""
    junk = tmp_path / "notes.mri"
    junk.write_text("just some notes", encoding="utf-8")
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            "from modelmri.cli import main; main()",
            "open",
            str(junk),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT)},
    )
    assert done.returncode == 2
    assert "not a ModelMRI session" in done.stderr
    assert "Traceback" not in done.stderr


def test_a_missing_file_says_so(tmp_path):
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            "from modelmri.cli import main; main()",
            "open",
            str(tmp_path / "nope.mri"),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT)},
    )
    assert done.returncode == 2
    assert "no such file" in done.stderr


def test_the_viewer_resolves_urls_rather_than_pattern_matching_them():
    """`?f=` names a file the local server is serving.

    The first version tried to spot absolute URLs by pattern — reject
    `scheme:` or a leading `//`. A backslash walked straight through both
    (`?f=\\\\evil.com/x` resolves protocol-relative), so a link was enough to
    make someone's browser fetch a host of the sender's choosing, including
    LAN and localhost addresses the sender cannot reach.

    The rule is now the only one that cannot be spelled around: resolve the
    URL and compare its origin. This test asserts the *approach*, because a
    future 'simplification' back to a regex would silently reopen it.
    tests/viewer_check.py proves the behaviour in a real browser.
    """
    source = (ROOT / "frontend" / "src" / "viewer.ts").read_text(encoding="utf-8")
    assert "autoOpenPath" in source
    assert "resolved.origin !== location.origin" in source, (
        "the origin comparison is gone — a pattern match is not a substitute"
    )
    assert "new URL(raw, location.href)" in source
    # And the pattern-matching form must not come back.
    assert 'startsWith("//")' not in source
    assert "/^[a-z][a-z0-9+.-]*:/i" not in source


def test_the_viewer_has_exactly_one_build_output():
    """`modelmri open` serves it from the package and the Pages deploy copies
    it to /viewer/. Two build outputs would be two things that can disagree
    about what the viewer is — so vite emits one, into the package."""
    config = (ROOT / "frontend" / "vite.config.ts").read_text(encoding="utf-8")
    assert '"../modelmri/static/viewer"' in config
    assert "viewer-dist" not in config, "a second viewer output has come back"

    pages = (ROOT / ".github" / "workflows" / "pages.yml").read_text(encoding="utf-8")
    assert "modelmri/static/viewer" in pages
    assert "viewer-dist" not in pages


def test_the_wheel_is_told_to_carry_the_viewer():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "modelmri/static/viewer/index.html" in pyproject
    assert "modelmri/static/viewer/assets/**" in pyproject


def test_the_release_workflow_actually_builds_the_viewer():
    """The bundle is not committed, so if the release stops building it the
    wheel goes out with a `modelmri open` that can only apologise — and
    nothing else would notice. It shipped that way once, saved only by the
    bundle having been committed by accident."""
    release = (ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )
    assert "npm run build:viewer" in release, (
        "the release builds the app but not the viewer"
    )
    assert "static/viewer/" in release, "the wheel check does not look for the viewer"


def test_the_built_viewer_is_not_committed():
    """Vite emits content-hashed filenames, so committing them adds a fresh
    dead copy of the whole bundle on every build. Same rule as static/app."""
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "modelmri/static/viewer/*" in ignore

    tracked = subprocess.run(
        ["git", "ls-files", "modelmri/static/viewer"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=60,
    ).stdout.split()
    stray = [f for f in tracked if not f.endswith(".gitkeep")]
    assert not stray, f"built viewer files are committed: {stray[:4]}"


def test_the_session_module_has_no_heavy_imports_at_module_level():
    """torch is used for the tensor fast path, but only inside the function
    that needs it — importing the format must not cost seconds."""
    source = (ROOT / "modelmri" / "session.py").read_text(encoding="utf-8")
    head = source.split("def ")[0]
    for heavy in ("import torch", "import numpy", "from torch"):
        assert heavy not in head, f"{heavy} at module level in session.py"


def test_json_only_dependencies_for_the_reader():
    """A sanity check that the format is stdlib-parseable, which is what
    makes a zero-dependency reader possible at all."""
    from modelmri import session

    raw = session.build(
        model_id="m",
        device="cpu",
        dtype="f32",
        n_params=1,
        tokens=["x"],
        prompt="",
        generation="",
        attention={(0, 0): [[1.0]]},
        n_layers=1,
        n_heads=1,
    )
    import gzip

    doc = json.loads(gzip.decompress(raw))
    assert doc["format"] == "modelmri-session"
