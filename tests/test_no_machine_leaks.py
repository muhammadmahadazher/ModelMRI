"""Nothing may be baked in. The tool must behave the same on any machine.

Two guarantees, both enforced rather than asserted in a comment:

1. **No absolute path literal exists in shipped code.** A grep, so a new one
   cannot be added quietly.
2. **Nothing from the developer's machine reaches the API.** The whole app is
   run inside a synthetic HOME / cache / working directory, every endpoint
   that can carry a path is called, and every absolute path in every response
   must point inside that sandbox. If any real location leaked -- a
   hardcoded `J:\\...`, a `Path.home()` that ignored the override, a cache
   resolved from the wrong variable -- it shows up here as a path outside
   the sandbox.

The second one is the test that would have caught most of the seventeen
portability bugs found by audit, before they shipped.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SHIPPED = [
    *(ROOT / "modelmri").rglob("*.py"),
    *(ROOT / "frontend" / "src").rglob("*.ts"),
    *(ROOT / "frontend" / "src").rglob("*.tsx"),
    *(ROOT / "packages").rglob("*.py"),
]

# A drive letter, or a POSIX home, inside a string literal.
_ABSOLUTE = re.compile(r"""['"](?:[A-Za-z]:[\\/]|/home/|/Users/|/root/)""")


def test_no_source_file_contains_an_absolute_path():
    offenders = []
    for path in SHIPPED:
        if "__pycache__" in path.parts or "static" in path.parts:
            continue
        for n, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            stripped = line.strip()
            if stripped.startswith(("#", "*", "//")):
                continue  # prose about a path is not a path
            if _ABSOLUTE.search(line):
                offenders.append(f"{path.relative_to(ROOT)}:{n}: {stripped[:90]}")
    assert not offenders, "absolute paths in shipped code:\n" + "\n".join(offenders)


def test_no_source_file_names_this_developers_machine():
    """A username or drive that made it into a string would follow the
    package to every user who installed it."""
    # As a path component, not anywhere: the project's own GitHub URL
    # legitimately contains the author's name.
    needles = ("My Drive", "Claude_Experiments", r"\mahad", "/mahad/")
    offenders = []
    for path in SHIPPED:
        if "__pycache__" in path.parts or "static" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for needle in needles:
            if needle in text:
                offenders.append(f"{path.relative_to(ROOT)} contains {needle!r}")
    assert not offenders, "\n".join(offenders)


# ------------------------------------------------------- the sandbox test


def _paths_in(value, out: list[str]) -> list[str]:
    """Every string in a JSON response that looks like an absolute path."""
    if isinstance(value, str):
        if re.match(r"^(?:[A-Za-z]:[\\/]|/)", value) and not value.startswith(
            ("/api/", "/ws/", "http")
        ):
            out.append(value)
    elif isinstance(value, dict):
        for v in value.values():
            _paths_in(v, out)
    elif isinstance(value, list):
        for v in value:
            _paths_in(v, out)
    return out


@pytest.fixture
def sandboxed(tmp_path, monkeypatch):
    """Run the whole app as if on a machine that is not this one."""
    home = tmp_path / "home"
    cache = tmp_path / "hf" / "hub"
    work = tmp_path / "project"
    nets = tmp_path / "my-nets"
    for d in (home, cache, work, nets):
        d.mkdir(parents=True)

    monkeypatch.setenv("MODELMRI_HOME", str(home))
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("MODELMRI_MODELS_DIR", str(nets))
    monkeypatch.setenv("MODELMRI_TRACE_DIR", str(tmp_path / "traces"))
    for stale in ("HUGGINGFACE_HUB_CACHE", "XDG_CACHE_HOME", "HF_LEROBOT_HOME"):
        monkeypatch.delenv(stale, raising=False)
    monkeypatch.chdir(work)
    return tmp_path


# Every endpoint that can put a filesystem path in front of a user.
PATH_BEARING = [
    "/api/paths",
    "/api/models/discovered",
    "/api/models/local",
    "/api/custom",
    "/api/vla",
    "/api/vla/datasets",
    "/api/session",
]


def test_no_endpoint_leaks_a_path_from_outside_the_sandbox(sandboxed):
    from fastapi.testclient import TestClient

    from modelmri.server import create_app

    client = TestClient(create_app())
    sandbox = str(sandboxed).lower()

    leaked: list[str] = []
    for endpoint in PATH_BEARING:
        response = client.get(endpoint)
        assert response.status_code == 200, f"{endpoint} -> {response.status_code}"
        for found in _paths_in(response.json(), []):
            if not found.lower().startswith(sandbox):
                leaked.append(f"{endpoint}: {found}")

    assert not leaked, (
        "these paths came from the real machine rather than the sandbox, "
        "so they are hardcoded or resolved from the wrong place:\n  "
        + "\n  ".join(leaked)
    )


def test_the_picker_reports_the_directory_it_was_launched_in(sandboxed):
    """The 'Looked in:' line, which is the one people actually notice."""
    from fastapi.testclient import TestClient

    from modelmri.server import create_app

    client = TestClient(create_app())
    roots = [r.lower() for r in client.get("/api/models/discovered").json()["roots"]]
    assert any((sandboxed / "project").name in r for r in roots), roots
    assert any("hf" in r for r in roots), roots


# ------------------------------------------------------- OS conventions


@pytest.mark.parametrize(
    "platform,marker",
    [
        ("win32", "AppData"),
        ("darwin", "Application Support"),
        ("linux", ".local"),
    ],
)
def test_each_platform_gets_its_own_convention(platform, marker, monkeypatch, tmp_path):
    """Same code, three answers -- not one machine's layout for everyone."""
    from modelmri import paths

    for var in ("MODELMRI_HOME", "XDG_DATA_HOME", "LOCALAPPDATA", "APPDATA"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))

    assert marker in str(paths.data_dir())


def test_one_environment_variable_relocates_everything(monkeypatch, tmp_path):
    """The container escape hatch, on every platform."""
    from modelmri import paths

    monkeypatch.setenv("MODELMRI_HOME", str(tmp_path / "all-of-it"))
    for kind in (paths.data_dir(), paths.config_dir(), paths.cache_dir()):
        assert str(kind).startswith(str(tmp_path / "all-of-it"))


def test_paths_are_split_with_the_platform_separator(monkeypatch, tmp_path):
    """`;` on Windows, `:` elsewhere. Getting this wrong turns two
    directories into one nonexistent one."""
    from modelmri import paths

    a, b = tmp_path / "a", tmp_path / "b"
    monkeypatch.setenv("MODELMRI_MODELS_DIR", f"{a}{os.pathsep}{b}")
    assert paths.models_dirs() == [a, b]


# ---------------------------------------------------------------------------
# What the routes are allowed to publish
# ---------------------------------------------------------------------------
#
# CodeQL flags every `JSONResponse({"error": str(err)})` in server.py as
# "information exposure through an exception", seven times. It is right to
# ask: `str(err)` on an arbitrary exception is a stack-trace leak, and this
# repo has already shipped six of those (see the 0.10 changelog).
#
# It is wrong here, but only because of an invariant no comment can hold:
# every exception those handlers catch carries an AUTHORED sentence. The
# proof has to be executable, or the next `raise Refusal(str(some_torch_err))`
# turns the false positive into a true one silently.

# Exception types the browser-facing handlers in server.py catch and stringify.
_PUBLISHED = {
    "Refusal",
    "BadRequest",
    "AdapterError",
    "TooBig",
    "TooCostly",
    "Unsupported",
    "SessionError",
}

# Names a caught exception is bound to in this codebase.
_CAUGHT = {"err", "e", "exc"}


def _raises_embedding_a_caught_exception():
    """Every raise of a published type whose message embeds `str(err)`.

    Yields (path, lineno, source, marked) — `marked` is whether a `leak-ok`
    note sits within a few lines, which is this repo's existing convention for
    a deliberate one.
    """
    import ast

    for path in sorted((ROOT / "modelmri").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        for node in ast.walk(ast.parse(text)):
            if not isinstance(node, ast.Raise) or not isinstance(node.exc, ast.Call):
                continue
            fn = node.exc.func
            name = getattr(fn, "id", None) or getattr(fn, "attr", None)
            if name not in _PUBLISHED:
                continue
            embeds = False
            for sub in ast.walk(node):
                if (
                    isinstance(sub, ast.Call)
                    and getattr(sub.func, "id", "") == "str"
                    and sub.args
                    and getattr(sub.args[0], "id", "") in _CAUGHT
                ):
                    embeds = True
                if isinstance(sub, ast.FormattedValue) and isinstance(
                    sub.value, ast.Name
                ):
                    if sub.value.id in _CAUGHT:
                        embeds = True
            if embeds:
                # The WHOLE statement plus a little either side. A `raise`
                # spanning four lines carries its marker wherever the
                # interpolation is, not necessarily on the first line —
                # custom.py puts it on line 3 of 4, and a window anchored to
                # `lineno` alone reported that deliberate site as unmarked.
                end = getattr(node, "end_lineno", None) or node.lineno
                window = "\n".join(lines[max(0, node.lineno - 5) : end + 1])
                yield path, node.lineno, ast.unparse(node), "leak-ok" in window


def test_every_published_exception_that_embeds_another_is_marked():
    """A foreign exception's text may reach the browser only deliberately.

    Measured at the time of writing: 230 raise sites of published types, 10 of
    which embed a caught exception's text. Four are custom.py publishing the
    READER'S OWN adapter code, which is the one case where the text is theirs
    to see. Six are runtime.py re-raising AblationError / AttributionError /
    FeatureAblationError / PatchError, all of which are this project's own
    types carrying authored sentences — see the test below.

    If this fails, either mark the new site `leak-ok` with a reason or stop
    embedding the exception.
    """
    unmarked = [
        f"{p.relative_to(ROOT)}:{n}  {src[:90]}"
        for p, n, src, marked in _raises_embedding_a_caught_exception()
        if not marked
    ]
    assert not unmarked, (
        "unmarked exception text reaching a published error:\n" + "\n".join(unmarked)
    )


def test_the_internal_error_types_never_embed_a_foreign_exception():
    """The invariant that makes the six `leak-ok` sites in runtime.py safe.

    `raise Refusal(str(err))` publishes whatever `AblationError` was built
    with. That is fine exactly as long as those four types are always
    constructed from authored text — so this walks every one of their raise
    sites and checks. Break it and the CodeQL alert stops being a false
    positive.
    """
    import ast

    internal = {
        "AblationError",
        "AttributionError",
        "FeatureAblationError",
        "PatchError",
    }
    offenders, total = [], 0
    for path in sorted((ROOT / "modelmri").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Raise) or not isinstance(node.exc, ast.Call):
                continue
            fn = node.exc.func
            name = getattr(fn, "id", None) or getattr(fn, "attr", None)
            if name not in internal:
                continue
            total += 1
            for sub in ast.walk(node):
                if (
                    isinstance(sub, ast.Call)
                    and getattr(sub.func, "id", "") == "str"
                    and sub.args
                    and getattr(sub.args[0], "id", "") in _CAUGHT
                ) or (
                    isinstance(sub, ast.FormattedValue)
                    and isinstance(sub.value, ast.Name)
                    and sub.value.id in _CAUGHT
                ):
                    offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert total >= 15, f"only found {total} raise sites — did the types move?"
    assert not offenders, (
        "these carry a foreign exception's text and are re-published verbatim "
        "by runtime.py's `raise Refusal(str(err))`:\n" + "\n".join(offenders)
    )
