"""Command-line interface: `modelmri serve`."""

from __future__ import annotations

import argparse

from . import __version__


def main() -> None:
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
    else:
        parser.print_help()
