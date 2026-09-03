# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""Write the dependency section of THIRD_PARTY_NOTICES.md from the lockfiles.

    uv run python scripts/third_party_notices.py          # rewrite the section
    uv run python scripts/third_party_notices.py --check  # exit 1 if it would change

Python packages come from `uv.lock` (name, version, where from) and their
license and homepage from the metadata of the environment this runs in —
which is the project's own environment under `uv run`, so the numbers are
read from installed packages rather than typed. A locked package that is not
installed here (an extra that was not synced) is listed with its license
marked `not installed here`, never guessed.

npm packages come from `frontend/package-lock.json`, which records each
package's license, so no second tool and no `node_modules` are needed. Only
the runtime tree — what `dependencies` reaches — is bundled into the built
application; the build-time tree (`devDependencies` and everything under
them) is listed by count with its direct entries named, because it is not
shipped.

The section is the text between the two markers in THIRD_PARTY_NOTICES.md;
everything outside them is hand-maintained and untouched. A test
(`tests/test_licensing.py`) refuses any direct dependency that has no row
here, which is what makes running this a condition of adding a dependency.
"""

from __future__ import annotations

import json
import re
import sys
from importlib import metadata
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parents[1]
NOTICES = ROOT / "THIRD_PARTY_NOTICES.md"
START = "<!-- generated-section: dependencies"
END = "<!-- /generated-section -->"
PROJECTS = {
    "modelmri": ROOT / "pyproject.toml",
    "modelmri-record": ROOT / "packages" / "modelmri-record" / "pyproject.toml",
    "modelmri-policy": ROOT / "packages" / "modelmri-policy" / "pyproject.toml",
}


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


# ------------------------------------------------------------------- python


def direct_python(pyproject: Path) -> set[str]:
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project = data.get("project", {})
    specs = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        specs.extend(extra)
    return {
        normalize(
            re.match(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?", s.strip()).group(0)
        )
        for s in specs
    }


def _license_from(get, get_all) -> str:
    """One license string from metadata fields, in the order PEP 639 prefers."""
    lic = get("License-Expression") or ""
    if not lic:
        classifiers = [
            c for c in (get_all("Classifier") or []) if c.startswith("License ::")
        ]
        if classifiers:
            lic = "; ".join(c.split("::")[-1].strip() for c in classifiers)
    if not lic:
        raw = (get("License") or "").strip()
        lic = raw.splitlines()[0][:60] if raw else ""
    return lic


def _homepage_from(get, get_all) -> str:
    home = get("Home-page") or ""
    if not home:
        for url in get_all("Project-URL") or []:
            label, _, target = url.partition(",")
            if label.strip().lower() in {
                "homepage",
                "home",
                "repository",
                "source",
                "source code",
            }:
                home = target.strip()
                break
    return home


def pypi_license(name: str, version: str) -> tuple[str, str]:
    """The same fields from PyPI's JSON API, for a locked package that is not
    installed in this environment. Network; marked as such in the table."""
    import urllib.request

    url = (
        f"https://pypi.org/pypi/{name}/{version}/json"
        if version
        else f"https://pypi.org/pypi/{name}/json"
    )
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            info = json.load(resp)["info"]
    except Exception as exc:
        return f"not installed here; PyPI lookup failed ({type(exc).__name__})", ""
    classifiers = info.get("classifiers") or []
    lic = _license_from(
        lambda k: info.get(
            {"License-Expression": "license_expression", "License": "license"}[k]
        ),
        lambda k: classifiers if k == "Classifier" else [],
    )
    urls = info.get("project_urls") or {}
    home = info.get("home_page") or ""
    if not home:
        for label, target in urls.items():
            if label.strip().lower() in {
                "homepage",
                "home",
                "repository",
                "source",
                "source code",
            }:
                home = target
                break
    return (lic + " (PyPI metadata)") if lic else "no license metadata (PyPI)", home


def installed_license(name: str, version: str = "") -> tuple[str, str]:
    """(license, homepage) from the installed distribution, else from PyPI, else honest text."""
    try:
        meta = metadata.distribution(name).metadata
    except metadata.PackageNotFoundError:
        lic, home = pypi_license(name, version)
    else:
        lic = _license_from(meta.get, meta.get_all) or "no license metadata"
        home = _homepage_from(meta.get, meta.get_all)
    return lic, home or f"https://pypi.org/project/{name}/"


def runtime_closure(lock: dict, own: set[str]) -> set[str]:
    """Every locked package reachable from the three projects' runtime and
    optional dependencies. Everything else in the lock is there for the dev
    group only — ruff, pytest, reuse and what they pull in — and is never
    installed with a wheel, which the table has to say."""
    packages = {normalize(p["name"]): p for p in lock.get("package", [])}
    stack: list[tuple[str, tuple[str, ...]]] = []
    for name in own:
        p = packages.get(name)
        if not p:
            continue
        for d in p.get("dependencies", []):
            stack.append((d["name"], tuple(d.get("extra", []))))
        for lst in p.get("optional-dependencies", {}).values():
            for d in lst:
                stack.append((d["name"], tuple(d.get("extra", []))))
    seen: set[tuple[str, tuple[str, ...]]] = set()
    reached: set[str] = set()
    while stack:
        name, extras = stack.pop()
        key = (normalize(name), extras)
        if key in seen:
            continue
        seen.add(key)
        reached.add(normalize(name))
        p = packages.get(normalize(name))
        if not p:
            continue
        deps = list(p.get("dependencies", []))
        for extra in extras:
            deps.extend(p.get("optional-dependencies", {}).get(extra, []))
        for d in deps:
            stack.append((d["name"], tuple(d.get("extra", []))))
    return reached - own


def python_rows() -> list[str]:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    direct = {name: direct_python(p) for name, p in PROJECTS.items()}
    own = {normalize(n) for n in PROJECTS}
    runtime = runtime_closure(lock, own)
    rows = []
    locked: set[str] = set()
    # A name can be locked more than once — torch is, from PyPI and from the
    # CUDA index the dev environment uses — and a table that lists it twice
    # with nothing to tell the rows apart reads as a mistake. Say where each
    # one comes from.
    repeated = {
        n
        for n in (normalize(p["name"]) for p in lock.get("package", []))
        if sum(normalize(p["name"]) == n for p in lock.get("package", [])) > 1
    }
    for pkg in sorted(lock.get("package", []), key=lambda p: normalize(p["name"])):
        name = normalize(pkg["name"])
        locked.add(name)
        if name in own:
            continue
        version = pkg.get("version", "")
        if name in repeated:
            source = pkg.get("source", {})
            where = (
                source.get("registry")
                or source.get("git")
                or source.get("editable")
                or source.get("path")
                or "?"
            )
            version = f"{version} (from {where})"
        lic, home = installed_license(pkg["name"], pkg.get("version", ""))
        direct_for = [proj for proj, names in direct.items() if name in names]
        if direct_for:
            relation = "direct: " + ", ".join(direct_for)
        elif name in runtime:
            relation = "transitive"
        else:
            relation = "dev group only"
        shipped = (
            "installed with the wheel, not inside it"
            if name in runtime
            else "never installed with a wheel (development tooling)"
        )
        rows.append(
            f"| `{pkg['name']}` | {version} | {lic} | {home} | {relation} | {shipped} |"
        )
    # A declared dependency the lockfile does not pin — an extra that resolves in
    # its own environment (`modelmri-policy`'s `policy` extra, whose lerobot needs
    # Python 3.12, is installed into the sidecar's own venv). Listed from PyPI
    # rather than dropped: the notice covers what the packages declare, not only
    # what this repository's lock happens to hold.
    for proj, names in direct.items():
        for name in sorted(names - locked - own):
            lic, home = installed_license(name, "")
            rows.append(
                f"| `{name}` | not pinned by uv.lock | {lic} | {home} | direct: {proj} (an extra resolved in its own environment) | installed where that extra is installed |"
            )
    return rows


# ---------------------------------------------------------------------- npm


def npm_rows() -> tuple[list[str], list[str], int]:
    lock = json.loads(
        (ROOT / "frontend" / "package-lock.json").read_text(encoding="utf-8")
    )
    packages = lock.get("packages", {})
    root = packages.get("", {})
    runtime_direct = set(root.get("dependencies", {}))
    dev_direct = set(root.get("devDependencies", {}))

    def entry(name: str) -> dict:
        return packages.get(f"node_modules/{name}", {})

    # Everything reachable from the runtime dependencies is what the bundle can contain.
    runtime: set[str] = set()
    stack = list(runtime_direct)
    while stack:
        name = stack.pop()
        if name in runtime:
            continue
        runtime.add(name)
        stack.extend(entry(name).get("dependencies", {}))
    rows = []
    for name in sorted(runtime):
        e = entry(name)
        relation = "direct" if name in runtime_direct else "transitive"
        rows.append(
            f"| `{name}` | {e.get('version', '')} | {e.get('license', 'no license field')} | https://www.npmjs.com/package/{name} | {relation} | bundled into the built app if imported |"
        )
    dev_rows = [
        f"| `{name}` | {entry(name).get('version', '')} | {entry(name).get('license', 'no license field')} | https://www.npmjs.com/package/{name} | direct (dev) | build-time only, not shipped |"
        for name in sorted(dev_direct)
    ]
    total = len([k for k in packages if k.startswith("node_modules/")])
    return rows, dev_rows, total


# ------------------------------------------------------------------ section


def render() -> str:
    py = python_rows()
    npm_runtime, npm_dev, npm_total = npm_rows()
    header = "| package | version | license | homepage | relation | shipped? |\n|---|---|---|---|---|---|"
    lines = [
        START + "\n     Written by scripts/third_party_notices.py from uv.lock and",
        "     frontend/package-lock.json. Do not edit by hand; run the script. -->",
        "",
        "### Python",
        "",
        f"{len(py)} packages in `uv.lock`, every extra included. Licenses are read from the",
        "installed metadata of the environment the script ran in; a package that was not",
        "installed there says so rather than guessing.",
        "",
        header,
        *py,
        "",
        "### npm — runtime tree (what the built application can bundle)",
        "",
        header,
        *npm_runtime,
        "",
        f"### npm — build-time tools ({npm_total} packages in the lockfile in total; direct entries below, not shipped)",
        "",
        header,
        *npm_dev,
        "",
        END,
    ]
    return "\n".join(lines)


def main() -> int:
    text = NOTICES.read_text(encoding="utf-8")
    start = text.find(START)
    end = text.find(END)
    if start == -1 or end == -1:
        print(
            "THIRD_PARTY_NOTICES.md has lost its generated-section markers",
            file=sys.stderr,
        )
        return 2
    new = text[:start] + render() + text[end + len(END) :]
    if "--check" in sys.argv[1:]:
        if new != text:
            print(
                "THIRD_PARTY_NOTICES.md is out of date: run scripts/third_party_notices.py",
                file=sys.stderr,
            )
            return 1
        print("THIRD_PARTY_NOTICES.md matches the lockfiles")
        return 0
    if new != text:
        NOTICES.write_text(new, encoding="utf-8", newline="\n")
        print("THIRD_PARTY_NOTICES.md: dependency section rewritten")
    else:
        print("THIRD_PARTY_NOTICES.md: already current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
