# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""The license boundary is architecture, and this file is where it is checked.

LICENSING.md draws one line through the repository: the application is
AGPL-3.0-only, and the pieces meant to live inside other people's software —
`modelmri-record`, `modelmri-policy`, the `npx modelmri` shim, the `.mri`
codec — are Apache-2.0. A line that is only written down drifts the first time
somebody adds an import, so it is checked here, by path and by `ast`, on every
pull request.

This file imports nothing from `modelmri` on purpose. It reads source text and
walks the tree, so it runs anywhere the repository is checked out — including
on a machine with no torch — and it can never be broken by the code it judges.

What it enforces:

- every first-party source file carries the SPDX identifier its path requires;
- nothing under an Apache-2.0 directory is marked AGPL;
- the Apache-2.0 packages and the codec files import nothing from the
  application — the dependency direction is application → packages, never the
  other way — beyond the crossings listed in KNOWN_CROSSINGS;
- the five codec files import only the standard library and each other, again
  beyond the listed crossings, so that a `.mri` file can always be read
  without torch;
- the crossings list is a ratchet: every entry must still exist, so fixing one
  means deleting its line here, and nothing can be added to it quietly;
- `modelmri/__init__.py` imports nothing, because every codec file executes it;
- the npm shim imports nothing from the application;
- the TypeScript codec package does not exist yet, and the check that will
  govern it is exercised today against synthetic packages, so it is not a skip;
- every direct dependency of the three Python projects and of the frontend is
  listed in THIRD_PARTY_NOTICES.md.

Why a ratchet rather than a clean assertion: when this file was written the
tree had six crossings the licensing brief's audit had not seen (they are the
entries below, each with where it goes). The `.mri` codec's extraction into
its own package is the change that removes them; until then this file makes
sure the number only goes down.

Mutation-checked when written: a flipped header, a new `import torch` in
`fmt.py`, an AGPL header inside a package, a new application import inside a
package, a crossing removed from the list while the code still has it, an
import added to `modelmri/__init__.py`, a `require` in the shim, and a
dependency removed from the notices each turn exactly the test that names
them red.
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

AGPL = "AGPL-3.0-only"
APACHE = "Apache-2.0"

CODEC_FILES = (
    "modelmri/session.py",
    "modelmri/receipts.py",
    "modelmri/errors.py",
    "modelmri/fmt.py",
    "modelmri/paths.py",
)
CODEC_MODULES = frozenset("modelmri." + Path(p).stem for p in CODEC_FILES)
APACHE_PACKAGE_DIRS = (
    "packages/modelmri-record",
    "packages/modelmri-policy",
    "npm-stub",
)
APACHE_DIRS = APACHE_PACKAGE_DIRS + ("examples",)
AGPL_DIRS = ("modelmri", "frontend", "tests", "scripts")
SOURCE_SUFFIXES = {".py", ".ts", ".tsx", ".js", ".mjs", ".cjs", ".css"}
PRUNED_DIRS = {"node_modules", "static", "dist", ".venv", "site", "__pycache__", ".git"}
# REUSE-IgnoreStart — the tag as a string, which `reuse lint` would otherwise
# read as this file's own (empty) license expression.
TAG = "SPDX-License-Identifier:"
# REUSE-IgnoreEnd
TS_CODEC_PACKAGE = ROOT / "packages" / "mri-ts"
FIRST_PARTY = {"modelmri", "modelmri-record", "modelmri-policy"}

# The crossings that exist today, by file, with where each one goes. An entry
# here is a debt with a name, not a permission: the tests below fail if a file
# imports anything not listed, AND fail if a listed import is gone (delete the
# line — the list only shrinks). The codec-package extraction (the brief's
# PR4) is where these are paid.
KNOWN_CROSSINGS: dict[str, dict[str, str]] = {
    "modelmri/session.py": {
        "torch": "`_quantise`'s tensor fast path; the codec keeps the list path, the app converts first",
        "modelmri.bundle": "`bundle.prepare` redacts at write time; the writer takes a prepared document instead",
    },
    "modelmri/paths.py": {
        "huggingface_hub": "HF cache constants and repo-id validation; not codec material, stays in the app",
        "huggingface_hub.utils": "same",
    },
    "packages/modelmri-record/modelmri_record/__init__.py": {
        "modelmri.paths": "optional: the app's data dir for undelivered traces; the location moves into the recorder",
        "modelmri.otel": "optional: OTLP delivery through the app's bridge; the bridge moves into the recorder",
    },
    "packages/modelmri-record/modelmri_record/pytest_plugin.py": {
        "modelmri.patterns": "optional: structural assertions need the app's pattern counts; they move into the recorder",
    },
}


# --------------------------------------------------------------------- helpers


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def source_files() -> list[Path]:
    found: list[Path] = []
    for top in APACHE_DIRS + AGPL_DIRS:
        base = ROOT / top
        if not base.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d not in PRUNED_DIRS)
            for name in filenames:
                if Path(name).suffix in SOURCE_SUFFIXES:
                    found.append(Path(dirpath) / name)
    return sorted(found)


def expected_license(path: Path) -> str:
    r = rel(path)
    if r in CODEC_FILES or r.startswith(tuple(d + "/" for d in APACHE_DIRS)):
        return APACHE
    if r.startswith(tuple(d + "/" for d in AGPL_DIRS)):
        return AGPL
    raise AssertionError(f"{r} is not placed by the path matrix in LICENSING.md")


def declared_license(path: Path) -> str | None:
    with open(path, encoding="utf-8") as f:
        head = [next(f, "") for _ in range(8)]
    for line in head:
        if TAG in line:
            value = line.split(TAG, 1)[1].strip()
            for close in (" */", "*/", "-->"):
                if value.endswith(close):
                    value = value[: -len(close)].strip()
            return value
    return None


def package_root(path: Path) -> Path:
    """The directory above the outermost package this file belongs to."""
    d = path.parent
    while (d / "__init__.py").exists() and d != ROOT:
        d = d.parent
    return d


def module_name(path: Path) -> str:
    parts = path.relative_to(package_root(path)).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def is_first_party_module(dotted: str, root: Path) -> bool:
    """Does `dotted` name a module or package on disk?

    Absolute imports resolve from every first-party import root: the
    repository root (where `modelmri` lives) and the importing file's own
    package root (where `modelmri_record` or `modelmri_policy` lives).
    """
    for base in (ROOT, root):
        p = base.joinpath(*dotted.split("."))
        if p.with_suffix(".py").exists() or (p / "__init__.py").exists():
            return True
    return False


def imported_modules(path: Path) -> list[tuple[str, int]]:
    """Every module this file imports, resolved to an absolute dotted name.

    `from X import a` names the submodule `X.a` when one exists on disk, and
    `X` itself otherwise (then `a` is an attribute, like `__version__`).
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    own = module_name(path)
    root = package_root(path)
    is_package = path.name == "__init__.py"
    out: list[tuple[str, int]] = []

    def from_names(base: str, names: list[ast.alias], lineno: int) -> None:
        for alias in names:
            candidate = f"{base}.{alias.name}" if base else alias.name
            if base and is_first_party_module(candidate, root):
                out.append((candidate, lineno))
            else:
                out.append((base or alias.name, lineno))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append((alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                from_names(node.module or "", node.names, node.lineno)
                continue
            base_parts = own.split(".") if own else []
            if not is_package:
                base_parts = base_parts[:-1]
            drop = node.level - 1
            base_parts = base_parts[: len(base_parts) - drop] if drop else base_parts
            base = ".".join(base_parts)
            if node.module:
                from_names(
                    f"{base}.{node.module}" if base else node.module,
                    node.names,
                    node.lineno,
                )
            else:
                from_names(base, node.names, node.lineno)
    return out


def top_level(name: str) -> str:
    return name.split(".", 1)[0]


def is_stdlib(name: str) -> bool:
    return top_level(name) in sys.stdlib_module_names


def python_files_under(*tops: str) -> list[Path]:
    files: list[Path] = []
    for top in tops:
        base = ROOT / top
        if not base.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d not in PRUNED_DIRS)
            files.extend(Path(dirpath) / n for n in filenames if n.endswith(".py"))
    return sorted(files)


def known(path: Path) -> dict[str, str]:
    return KNOWN_CROSSINGS.get(rel(path), {})


# ---------------------------------------------------------------- the headers


@pytest.mark.parametrize("path", source_files(), ids=rel)
def test_every_first_party_source_file_carries_the_header_its_path_requires(
    path: Path,
) -> None:
    want = expected_license(path)
    got = declared_license(path)
    assert got is not None, (
        f"{rel(path)} has no SPDX-License-Identifier in its first lines. Every first-party "
        f"source file states its license; this one is {want} by the matrix in LICENSING.md."
    )
    assert got == want, (
        f"{rel(path)} says {got}; the path matrix in LICENSING.md puts it under {want}. "
        "A file does not change license by editing its header — change LICENSING.md, or the path."
    )


def test_no_agpl_marked_file_lives_inside_an_apache_directory() -> None:
    offenders = [
        rel(p)
        for top in APACHE_DIRS
        for p in source_files()
        if rel(p).startswith(top + "/") and declared_license(p) == AGPL
    ]
    assert not offenders, (
        "AGPL-3.0-only headers inside Apache-2.0 directories — a copyleft file inside a permissive "
        f"package makes the package copyleft for everyone who installs it: {offenders}"
    )


# ------------------------------------------------------------- the boundary


def apache_python_files() -> list[Path]:
    return python_files_under(*APACHE_PACKAGE_DIRS) + [ROOT / p for p in CODEC_FILES]


@pytest.mark.parametrize("path", apache_python_files(), ids=rel)
def test_apache_code_imports_nothing_new_from_the_application(path: Path) -> None:
    """The dependency direction is application → packages, never the other way."""
    r = rel(path)
    is_codec = r in CODEC_FILES
    allowed = known(path)
    bad = []
    for name, lineno in imported_modules(path):
        if top_level(name) != "modelmri":
            continue
        if name == "modelmri":
            # The package init, which defines `__version__` and nothing else —
            # see test_the_package_init_imports_nothing. Every module inside
            # `modelmri` executes it anyway; the recorder never imports it.
            if is_codec:
                continue
        if is_codec and name in CODEC_MODULES:
            continue
        if name in allowed:
            continue
        bad.append(f"line {lineno}: {name}")
    assert not bad, (
        f"{r} is Apache-2.0 and imports the AGPL-3.0-only application: {bad}. "
        "The packages and the .mri codec must stay importable without the application, "
        "or their license stops meaning what LICENSING.md says it means. Move the shared "
        "thing down into the package, or call it from the application side. (The crossings "
        "that already existed when this check was written are listed in KNOWN_CROSSINGS with "
        "where each one goes; this is not one of them.)"
    )


@pytest.mark.parametrize("path", [ROOT / p for p in CODEC_FILES], ids=lambda p: rel(p))
def test_the_codec_closure_admits_nothing_new(path: Path) -> None:
    """The five codec files import only the standard library and each other."""
    allowed = known(path)
    bad = []
    for name, lineno in imported_modules(path):
        if (
            is_stdlib(name)
            or name in CODEC_MODULES
            or name == "modelmri"
            or name in allowed
        ):
            continue
        bad.append(f"line {lineno}: {name}")
    assert not bad, (
        f"{rel(path)} imports {bad}. The .mri codec is the part of the application that reads "
        "and writes the format; it is Apache-2.0 and must stay importable with nothing but the "
        "standard library, so that anyone can open a .mri file without torch, without a GPU, "
        "and without ModelMRI. If the format needs this import, the codec is the wrong place "
        "for the code that needs it. (KNOWN_CROSSINGS lists the debts that predate this check; "
        "this is not one of them.)"
    )


@pytest.mark.parametrize(
    "entry",
    sorted((f, m) for f, ms in KNOWN_CROSSINGS.items() for m in ms),
    ids=lambda e: f"{e[0]}::{e[1]}",
)
def test_every_listed_crossing_still_exists_so_the_list_only_shrinks(
    entry: tuple[str, str],
) -> None:
    file, module = entry
    present = {name for name, _ in imported_modules(ROOT / file)}
    assert module in present, (
        f"{file} no longer imports {module} — good. Delete its line from KNOWN_CROSSINGS in "
        "tests/test_licensing.py so the debt is recorded as paid; the list only shrinks."
    )


def test_the_package_init_imports_nothing() -> None:
    """`import modelmri` must cost nothing: every codec file, and `modelmri open`, runs it."""
    tree = ast.parse((ROOT / "modelmri" / "__init__.py").read_text(encoding="utf-8"))
    imports = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert not imports, (
        f"modelmri/__init__.py imports something (lines {imports}). It must stay a version "
        "constant and a docstring: it executes before any codec file loads, and an import "
        "there is an import the .mri reader pays for — torch included, one hop away."
    )


NODE_BUILTINS = {
    "assert",
    "buffer",
    "child_process",
    "console",
    "crypto",
    "events",
    "fs",
    "http",
    "https",
    "net",
    "os",
    "path",
    "process",
    "readline",
    "stream",
    "url",
    "util",
    "zlib",
}


def test_the_npm_shim_imports_nothing_from_the_application() -> None:
    src = (ROOT / "npm-stub" / "cli.js").read_text(encoding="utf-8")
    targets = re.findall(
        r"""require\(\s*["']([^"']+)["']\s*\)|^\s*import\b[^"']*["']([^"']+)["']""",
        src,
        re.M,
    )
    names = [a or b for a, b in targets]
    foreign = [n for n in names if n.removeprefix("node:") not in NODE_BUILTINS]
    assert not foreign, (
        f"npm-stub/cli.js imports {foreign}. The shim is Apache-2.0 because it contains no "
        "application code; an import of anything but a Node builtin changes that."
    )


# ------------------------------------------------- the TypeScript codec package


TS_IMPORT = re.compile(r"""(?:from|import)\s+["']([^"']+)["']""")


def typescript_package_violations(package_dir: Path) -> list[str]:
    """Apache header on every .ts/.tsx, and no import that reaches the application."""
    problems: list[str] = []
    for path in sorted(package_dir.rglob("*.ts")) + sorted(package_dir.rglob("*.tsx")):
        if "node_modules" in path.parts or "dist" in path.parts:
            continue
        if declared_license(path) != APACHE:
            problems.append(
                f"{path.relative_to(package_dir).as_posix()}: not marked {APACHE}"
            )
        for target in TS_IMPORT.findall(path.read_text(encoding="utf-8")):
            if target.startswith("."):
                resolved = (path.parent / target).resolve()
                if package_dir.resolve() not in resolved.parents:
                    problems.append(
                        f"{path.relative_to(package_dir).as_posix()}: imports outside the package: {target}"
                    )
            elif target.startswith("modelmri") or "/frontend/" in target:
                problems.append(
                    f"{path.relative_to(package_dir).as_posix()}: imports the application: {target}"
                )
    return problems


# REUSE-IgnoreStart — synthetic packages with headers of their own, which
# `reuse lint` would otherwise read as this file's license expressions.
def synthetic_packages(tmp_path: Path) -> tuple[Path, Path]:
    clean = tmp_path / "clean"
    (clean / "src").mkdir(parents=True)
    (clean / "src" / "index.ts").write_text(
        "// SPDX-License-Identifier: Apache-2.0\n"
        "// SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher\n"
        'import { read } from "./reader";\nexport { read };\n',
        encoding="utf-8",
    )
    (clean / "src" / "reader.ts").write_text(
        "// SPDX-License-Identifier: Apache-2.0\n"
        "// SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher\n"
        "export function read() {}\n",
        encoding="utf-8",
    )
    dirty = tmp_path / "dirty"
    (dirty / "src").mkdir(parents=True)
    (dirty / "src" / "index.ts").write_text(
        "// SPDX-License-Identifier: AGPL-3.0-only\n"
        'import { api } from "../../../frontend/src/api";\nexport { api };\n',
        encoding="utf-8",
    )
    return clean, dirty


# REUSE-IgnoreEnd


def test_the_typescript_codec_package_check_works_and_the_package_is_still_absent(
    tmp_path: Path,
) -> None:
    clean, dirty = synthetic_packages(tmp_path)
    assert typescript_package_violations(clean) == []
    found = typescript_package_violations(dirty)
    assert any("not marked Apache-2.0" in p for p in found), found
    assert any("outside the package" in p for p in found), found

    # The real package does not exist yet. The day it does, this assertion is
    # replaced by `assert typescript_package_violations(TS_CODEC_PACKAGE) == []`,
    # and the two synthetic cases above stay as the proof that the check bites.
    assert not TS_CODEC_PACKAGE.exists(), (
        f"{rel(TS_CODEC_PACKAGE)} now exists: switch this test to enforcing "
        "typescript_package_violations() on it, and keep the synthetic cases."
    )


# ------------------------------------------------------- third-party notices


NAME = re.compile(r"^\s*[\"']([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)")


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def direct_python_dependencies(pyproject: Path) -> set[str]:
    """Names from `dependencies = [...]` and every `[project.optional-dependencies]` list.

    A small reader rather than a TOML parser, because the test must run on
    Python 3.10 where `tomllib` does not exist and nothing else may be imported.
    An empty list is a real answer — `modelmri-record` has no dependencies on
    purpose — so the reader's sanity check is that the key exists, not that it
    has entries.
    """
    text = pyproject.read_text(encoding="utf-8")
    assert re.search(r"^dependencies\s*=", text, re.M), (
        f"{pyproject.name} declares no `dependencies` key at all — the reader would find nothing"
    )
    names: set[str] = set()
    in_list = False
    section = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped
            in_list = False
            continue
        if not in_list:
            if section == "[project]" and re.match(r"^dependencies\s*=\s*\[", stripped):
                in_list = "]" not in stripped.split("[", 1)[1]
            elif section == "[project.optional-dependencies]" and re.match(
                r"^[A-Za-z0-9_-]+\s*=\s*\[", stripped
            ):
                in_list = "]" not in stripped.split("[", 1)[1]
            else:
                continue
            if not in_list:
                continue
        if stripped.startswith("]"):
            in_list = False
            continue
        m = NAME.match(stripped)
        if m:
            names.add(normalize(m.group(1)))
    return names - FIRST_PARTY


def direct_npm_dependencies(package_json: Path) -> set[str]:
    data = json.loads(package_json.read_text(encoding="utf-8"))
    assert "dependencies" in data, f"{package_json.name} declares no `dependencies` key"
    return {
        normalize(n)
        for key in ("dependencies", "devDependencies")
        for n in data.get(key, {})
    }


def listed_in_notices() -> set[str]:
    text = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    start = text.find("<!-- generated-section: dependencies")
    end = text.find("<!-- /generated-section -->")
    assert start != -1 and end != -1, (
        "THIRD_PARTY_NOTICES.md has lost its generated-section markers"
    )
    return {
        normalize(m) for m in re.findall(r"^\| `([^`]+)` \|", text[start:end], re.M)
    }


@pytest.mark.parametrize(
    "manifest",
    [
        "pyproject.toml",
        "packages/modelmri-record/pyproject.toml",
        "packages/modelmri-policy/pyproject.toml",
        "frontend/package.json",
    ],
)
def test_every_direct_dependency_is_listed_in_third_party_notices(
    manifest: str,
) -> None:
    path = ROOT / manifest
    wanted = (
        direct_npm_dependencies(path)
        if path.suffix == ".json"
        else direct_python_dependencies(path)
    )
    missing = sorted(wanted - listed_in_notices())
    assert not missing, (
        f"{manifest} depends on {missing}, which THIRD_PARTY_NOTICES.md does not list. "
        "Run `uv run python scripts/third_party_notices.py` and commit the result: a dependency "
        "does not land without its notice."
    )
