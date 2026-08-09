"""Command-line interface: `modelmri serve`, `modelmri open`, `modelmri where`."""

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

    opener = sub.add_parser(
        "open", help="Open a shared analysis (.mri) — no model needed"
    )
    opener.add_argument("file", help="the .mri someone sent you")
    opener.add_argument("--host", default="127.0.0.1")
    opener.add_argument("--port", type=int, default=5900)
    opener.add_argument(
        "--no-browser", action="store_true", help="just serve it, don't open a tab"
    )

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
    elif args.command == "open":
        from pathlib import Path

        from . import session

        target = Path(args.file).expanduser()
        if not target.is_file():
            print(f"modelmri: no such file: {target}", file=sys.stderr)
            raise SystemExit(2)

        # Parse before starting anything. Someone who was sent the wrong file
        # should get one sentence, not a server they then have to shut down.
        try:
            parsed = session.parse(target.read_bytes())
        except session.SessionError as err:
            print(f"modelmri: {err}", file=sys.stderr)
            raise SystemExit(2) from err

        note = (parsed.meta.get("note") or "").strip()
        print(f"ModelMRI {__version__} — opening {target.name}")
        print(f"  model     {parsed.meta.get('model') or 'unknown'}")
        if note:
            print(f"  note      {note}")
        print(f"  contains  {len(parsed.tokens)} tokens, "
              f"{len(parsed.attention)} attention maps")
        print("  no model will be loaded — this is a recording\n")

        # The server picks this up in create_app. An environment variable
        # rather than a module global because uvicorn may import the factory
        # in a fresh interpreter.
        os.environ["MODELMRI_OPEN"] = str(target.resolve())

        url = f"http://{args.host}:{args.port}"
        if not args.no_browser:
            import threading
            import webbrowser

            # After the server is listening, not before -- a tab that opens
            # onto a connection-refused page reads as a broken install.
            threading.Timer(1.5, lambda: webbrowser.open(url)).start()

        import uvicorn

        print(f"serving on {url}  (ctrl-c to stop)")
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
