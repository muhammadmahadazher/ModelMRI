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

  python scripts/build_frontend.py                       # in place (CI)
  python scripts/build_frontend.py --work C:/build/mri   # out of tree

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
    ap.add_argument("--demo", action="store_true", help="build the hosted demo bundle")
    args = ap.parse_args()

    if args.work:
        base = Path(args.work).resolve()
        work = base / "frontend"
        sync_sources(work)
        print(f"sources synced to {work}", flush=True)
    else:
        base, work = ROOT, FRONTEND

    if not (work / "node_modules" / "vite").is_dir():
        run(["npm", "ci", "--no-audit", "--no-fund"], work)

    run(["npm", "run", "build:demo" if args.demo else "build"], work)

    if args.work:
        # vite.config.ts writes its outDir relative to frontend/, so the
        # out-of-tree build lands in the mirrored layout under base.
        built = base / ("demo-dist" if args.demo else "modelmri/static/app")
        dest = ROOT / ("demo-dist" if args.demo else "modelmri/static/app")
        if not built.is_dir():
            print(f"build produced nothing at {built}", file=sys.stderr)
            return 1
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(built, dest)
        print(f"copied {built} -> {dest}", flush=True)
        if args.demo:
            return 0 if (dest / "index.html").is_file() else 1

    files = sorted(p.name for p in DIST.rglob("*") if p.is_file())
    print(f"{len(files)} files in {DIST}", flush=True)
    return 0 if any(f == "index.html" for f in files) else 1


if __name__ == "__main__":
    raise SystemExit(main())
