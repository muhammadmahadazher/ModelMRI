"""Build the frontend, optionally in a working directory off the repo.

Normally `npm run build` inside frontend/ is all you need, and that is what
CI does. But some of us keep the repo on a synced virtual drive (Google
Drive, OneDrive, a network share). Those filesystems evict cold files and
reject reparse points, and a node_modules tree of forty thousand tiny files
is exactly what they handle worst -- a package.json quietly becomes zero
bytes and the build dies with an unrelated-looking error.

`--work DIR` copies the sources to a real local disk, installs and builds
there, then copies dist/ back into the package. Same output, same lockfile,
no node_modules on the synced drive at all.

That flag existed and still had to be remembered, which meant it was reached
for only *after* a build had already failed with a confusing error. So the
synced-drive case is now detected and handled by default: on a checkout under
a known sync root the work directory is chosen automatically, and the reason
is printed. `--in-place` overrides it, `--work` still names the directory.

  python scripts/build_frontend.py                       # auto: local if synced
  python scripts/build_frontend.py --work C:/build/mri   # out of tree, named
  python scripts/build_frontend.py --in-place            # never relocate (CI)

DIR mirrors the repo layout (DIR/frontend, DIR/modelmri/static/app) because
vite.config.ts writes its output as a path relative to frontend/.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
DIST = ROOT / "modelmri" / "static" / "app"

# Everything the build reads. node_modules and dist are deliberately absent:
# they are outputs, and copying them is what we are trying to avoid.
SOURCES = (
    "src",
    "public",
    "index.html",
    "package.json",
    "package-lock.json",
    "tsconfig.json",
    "tsconfig.node.json",
    "vite.config.ts",
)


# Directory names that mean "this path is a sync client's mount". Matched
# case-insensitively against every component of the repo's absolute path.
# Deliberately a name list rather than a filesystem-type probe: on Windows a
# Google Drive letter reports as a perfectly ordinary NTFS/ReFS volume, so
# there is nothing in the API to ask.
SYNC_MARKERS = (
    "my drive",  # Google Drive for Desktop
    "shared drives",
    "google drive",
    "onedrive",
    "dropbox",
    "icloud drive",
    "creative cloud files",
)


def synced_under(path: Path) -> str | None:
    """The sync-root component of `path`, or None. Name only, no side effects."""
    for part in path.resolve().parts:
        if part.strip().lower() in SYNC_MARKERS:
            return part
    return None


def default_work_dir() -> Path:
    """Somewhere local and predictable to build when the repo is on a mount.

    The system temp directory, not a hardcoded C:/build: temp is writable
    without asking, is already excluded from every sync client, and is the
    one path that exists on all three platforms. Stable across runs on
    purpose -- node_modules survives, so the second build skips `npm ci`
    entirely, which is most of the wall clock.
    """
    import tempfile

    return Path(tempfile.gettempdir()) / "modelmri-frontend-build"


def run(cmd: list[str], cwd: Path) -> None:
    print(f"$ {' '.join(cmd)}  (in {cwd})", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True, shell=sys.platform == "win32")


def sync_sources(work: Path) -> None:
    work.mkdir(parents=True, exist_ok=True)
    for name in SOURCES:
        src = FRONTEND / name
        if not src.exists():
            continue
        dst = work / name
        if src.is_dir():
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--work",
        metavar="DIR",
        help="build in DIR on a local disk instead of in frontend/",
    )
    ap.add_argument(
        "--in-place",
        action="store_true",
        help="build inside frontend/ even on a synced drive (what CI does)",
    )
    ap.add_argument("--demo", action="store_true", help="build the hosted demo bundle")
    ap.add_argument(
        "--viewer",
        action="store_true",
        help="build the .mri viewer (zero-install, reads a dropped file)",
    )
    args = ap.parse_args()

    work_root = args.work
    if not work_root and not args.in_place and (marker := synced_under(ROOT)):
        # Detected rather than configured, and SAID rather than done quietly:
        # a build that silently relocates is one nobody can debug when the
        # output turns up somewhere unexpected.
        work_root = str(default_work_dir())
        print(
            f"repo is under {marker!r}, which is a sync client's mount -- "
            "node_modules is 40,000 small files and those filesystems evict "
            "and corrupt them.\n"
            f"  building in {work_root} instead (--in-place to override)",
            flush=True,
        )

    if work_root:
        base = Path(work_root).resolve()
        work = base / "frontend"
        sync_sources(work)
        print(f"sources synced to {work}", flush=True)
    else:
        base, work = ROOT, FRONTEND

    if not (work / "node_modules" / "vite").is_dir():
        run(["npm", "ci", "--no-audit", "--no-fund"], work)

    target = "build:viewer" if args.viewer else ("build:demo" if args.demo else "build")
    run(["npm", "run", target], work)

    if work_root:
        # vite.config.ts writes its outDir relative to frontend/, so the
        # out-of-tree build lands in the mirrored layout under base.
        out = (
            "modelmri/static/viewer"
            if args.viewer
            else "demo-dist"
            if args.demo
            else "modelmri/static/app"
        )
        built, dest = base / out, ROOT / out
        if not built.is_dir():
            print(f"build produced nothing at {built}", file=sys.stderr)
            return 1
        # rmtree can fail on a synced drive that still holds a handle, and
        # `ignore_errors` means it fails *silently* -- after which copytree
        # raised FileExistsError on a directory the previous line was
        # supposed to have removed. dirs_exist_ok makes the copy authoritative
        # either way.
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(built, dest, dirs_exist_ok=True)
        print(f"copied {built} -> {dest}", flush=True)
        if args.demo or args.viewer:
            return 0 if (dest / "index.html").is_file() else 1
        final = dest
    else:
        final = DIST

    files = sorted(p.name for p in final.rglob("*") if p.is_file())
    print(f"{len(files)} files in {final}", flush=True)
    return 0 if any(f == "index.html" for f in files) else 1


if __name__ == "__main__":
    raise SystemExit(main())
