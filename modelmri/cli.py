"""Command-line interface: `modelmri serve`, `modelmri open`, `modelmri where`."""

from __future__ import annotations

import argparse
import os
import sys

from . import __version__


def serve_viewer(target, *, host: str, port: int, browser: bool) -> None:
    """Serve the bundled `.mri` viewer, using only the standard library.

    `modelmri open` used to start the full application, which imports torch
    and transformers — measured at 26 seconds — to display a 54 KB recording
    that needs neither. The first person to run it pressed ctrl-c partway
    through, reasonably concluding it had hung.

    Nothing here imports anything heavy. It serves the same viewer bundle
    that is published to GitHub Pages, plus the one file, and the page opens
    it from the `?f=` link rather than making you find and drop a file you
    just named on the command line.

    Two things this deliberately does NOT do: bind anything but the loopback
    interface, and serve any directory other than the viewer's own. A local
    file reader has no business being reachable from the network or exposing
    the tree it happens to be started in.
    """
    import functools
    import http.server
    import socketserver
    import threading
    import webbrowser
    from importlib.resources import files
    from pathlib import Path

    bundle = Path(str(files("modelmri") / "static" / "viewer"))
    if not (bundle / "index.html").is_file():
        print(
            "modelmri: this build has no bundled viewer.\n"
            "  Use `modelmri serve` and open the file from the page, or read\n"
            "  it at https://muhammadmahadazher.github.io/ModelMRI/viewer/",
            file=sys.stderr,
        )
        raise SystemExit(1)

    payload = Path(target).read_bytes()
    name = "session.mri"

    # Said once, plainly, rather than assumed. The docstring above promises
    # loopback; --host takes anything, and a recording is somebody's prompts.
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(
            f"  NOTE: serving on {host}, not loopback — anyone who can reach "
            f"this\n  machine on port {port} can read this recording.",
            file=sys.stderr,
        )

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(bundle), **kw)

        def _is_ours(self) -> bool:
            """Reject a request that reached us under someone else's name.

            A page on any website can point a name it controls at 127.0.0.1
            and then read whatever answers — DNS rebinding. The recording is
            somebody's prompts and generations, so checking the Host is the
            difference between "served to me" and "served to a tab I happened
            to have open".
            """
            sent = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
            if sent in ("127.0.0.1", "localhost", "::1", "", host):
                return True
            self.send_error(
                421,
                "Misdirected Request",
                f"This viewer only answers to localhost, not {sent!r}.",
            )
            return False

        def _payload_headers(self) -> bool:
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(payload)))
            # It is one local file for one local page; nothing should cache
            # it, and nothing else should be allowed to frame or embed it.
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            return True

        def do_GET(self):  # noqa: N802 - stdlib's spelling
            if not self._is_ours():
                return
            if self.path.split("?")[0] == f"/{name}":
                self._payload_headers()
                self.wfile.write(payload)
                return
            super().do_GET()

        def do_HEAD(self):  # noqa: N802
            # Overriding do_GET alone made HEAD /session.mri answer 404 while
            # GET answered 200 — the two disagreeing about whether a file
            # exists is the kind of thing that breaks a client for no visible
            # reason.
            if not self._is_ours():
                return
            if self.path.split("?")[0] == f"/{name}":
                self._payload_headers()
                return
            super().do_HEAD()

        def log_message(self, *a):  # a request log is noise here
            pass

        def handle_one_request(self):
            # A browser that closes a keep-alive socket, or a scanner sending
            # a malformed path, otherwise prints a full traceback into the
            # terminal of someone who only wanted to look at a file.
            try:
                super().handle_one_request()
            except (ConnectionError, TimeoutError):
                self.close_connection = True

    class Server(socketserver.ThreadingTCPServer):
        # ThreadingTCPServer, not TCPServer. `daemon_threads` on a
        # single-threaded server does nothing at all: one browser holding a
        # keep-alive socket open stalled every later request, and the page
        # simply never finished loading.
        allow_reuse_address = True
        daemon_threads = True

    try:
        httpd = Server((host, port), functools.partial(Handler))
    except OSError as err:
        print(f"modelmri: cannot listen on {host}:{port} — {err}", file=sys.stderr)
        print("  another ModelMRI may be running; try --port 5901", file=sys.stderr)
        raise SystemExit(1) from err

    url = f"http://{host}:{port}/?f={name}"
    if browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    print(f"  reading it at {url}\n  (ctrl-c to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        httpd.server_close()


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
        try:
            uvicorn.run(
                "modelmri.server:create_app",
                factory=True,
                host=args.host,
                port=args.port,
            )
        except KeyboardInterrupt:
            print("\nstopped.")
            return
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
        print(
            f"  contains  {len(parsed.tokens)} tokens, "
            f"{len(parsed.attention)} attention maps"
        )
        print("  no model will be loaded — this is a recording\n")

        serve_viewer(
            target, host=args.host, port=args.port, browser=not args.no_browser
        )
        return
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
