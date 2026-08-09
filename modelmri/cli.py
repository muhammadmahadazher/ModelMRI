"""Command-line interface: `modelmri serve`."""

from __future__ import annotations

import argparse
import os
import sys

from . import __version__


def main() -> None:
    # Windows consoles hand Python a cp1252 stdout, which cannot encode a path
    # containing (say) a Cyrillic or CJK username. Printing where things live
    # would then die with a UnicodeEncodeError -- the command that exists to
    # answer "where is my stuff?" failing precisely for the users whose stuff
    # is hardest to find. backslashreplace degrades instead of raising.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="backslashreplace")
            except (ValueError, OSError):
                pass

    parser = argparse.ArgumentParser(
        prog="modelmri",
        description="ModelMRI — Chrome DevTools for AI models and agents.",
    )
    parser.add_argument(
        "--version", action="version", version=f"modelmri {__version__}"
    )
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="Start the ModelMRI server")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=5900)

    sub.add_parser("where", help="Print every directory ModelMRI reads or writes")

    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn

        print(f"ModelMRI {__version__} serving on http://{args.host}:{args.port}")
        uvicorn.run(
            "modelmri.server:create_app",
            factory=True,
            host=args.host,
            port=args.port,
        )
    elif args.command == "where":
        from . import paths

        info = paths.describe()
        platform = info.pop("platform")
        width = max(len(k) for k in info)
        print(f"ModelMRI {__version__} on {platform}")
        print()
        for key, value in info.items():
            if value is None or value == []:
                continue
            if isinstance(value, list):
                value = os.pathsep.join(value)
            print(f"  {key:<{width}}  {value}")
        print()
        print("  Override any of it:")
        print("    MODELMRI_HOME        all of the above under one directory")
        print("    MODELMRI_MODELS_DIR  extra places to look for your models")
        print("    MODELMRI_TRACE_DIR   where undelivered traces are written")
        print("    HF_HOME/HF_HUB_CACHE where models download (HuggingFace's)")
    else:
        parser.print_help()
