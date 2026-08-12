"""Command-line interface: `modelmri serve`, `modelmri open`, `modelmri where`."""

from __future__ import annotations

import argparse
import os
import sys

from . import __version__


def inspect_session(path, *, as_json: bool = False) -> int:
    """Describe a `.mri` on the terminal. Returns the exit code.

    Same discipline as `open`: no torch, no transformers, no server. A `.mri`
    is gzipped JSON and everything below comes from the standard library plus
    session.py, so this stays instant on a cold cache — the reason `open` was
    rewritten in the first place was that 26 seconds of imports to read a
    54 KB file reads as a hang, and somebody pressed ctrl-c.
    """
    import json
    from pathlib import Path

    from . import session

    target = Path(path).expanduser()
    if not target.is_file():
        print(f"modelmri: no such file: {target}", file=sys.stderr)
        return 2
    try:
        parsed = session.parse(target.read_bytes())
    except session.SessionError as err:
        print(f"modelmri: {err}", file=sys.stderr)
        return 2

    meta = parsed.meta
    slices = sorted(
        (int(k.split(":")[0]), int(k.split(":")[1]))
        for k in parsed.attention
        if k.count(":") == 1 and all(part.isdigit() for part in k.split(":"))
    )
    summary = {
        "file": target.name,
        "bytes": target.stat().st_size,
        "model": meta.get("model") or "unknown",
        "device": meta.get("device"),
        "dtype": meta.get("dtype"),
        "n_params": meta.get("n_params"),
        "created_at": meta.get("created_at"),
        "modelmri": meta.get("modelmri"),
        "note": (meta.get("note") or "").strip(),
        "scope": meta.get("scope") or "",
        "precision": meta.get("precision") or "",
        "n_tokens": len(parsed.tokens),
        "n_prompt": parsed.n_prompt,
        "n_layers": parsed.n_layers,
        "n_heads": parsed.n_heads,
        "attention_maps": len(parsed.attention),
        "layers_present": sorted({li for li, _ in slices}),
        "heads_present": sorted({hi for _, hi in slices}),
        "lens_rows": len(parsed.lens),
        "patch": {
            "present": parsed.has_patch(),
            "components": sorted(parsed.patch.get("grids", {})),
            "clean": parsed.patch.get("clean", ""),
            "corrupt": parsed.patch.get("corrupt", ""),
        },
        "prompt": parsed.prompt,
        "generation": parsed.generation,
    }
    if as_json:
        print(json.dumps(summary, indent=2))
        return 0

    def line(k: str, v) -> None:
        print(f"  {k:<14}{v}")

    print(f"{target.name} — {summary['bytes'] / 1024:.1f} KB")
    line("model", summary["model"])
    if summary["n_params"]:
        line("size", f"{summary['n_params'] / 1e6:,.0f}M parameters")
    if summary["device"] or summary["dtype"]:
        line("ran on", f"{summary['device'] or '?'} · {summary['dtype'] or '?'}")
    line("recorded", f"{summary['created_at']} by ModelMRI {summary['modelmri']}")
    if summary["note"]:
        line("note", summary["note"])
    print()
    line("tokens", f"{summary['n_tokens']} ({summary['n_prompt']} prompt)")
    line("shape", f"{summary['n_layers']} layers x {summary['n_heads']} heads")
    line("attention", f"{summary['attention_maps']} maps")
    if summary["scope"]:
        line("", summary["scope"])
    if summary["lens_rows"]:
        line("logit lens", f"{summary['lens_rows']} rows")
    if summary["patch"]["present"]:
        line("patching", ", ".join(summary["patch"]["components"]))
        line("  clean", summary["patch"]["clean"])
        line("  corrupt", summary["patch"]["corrupt"])
    print()
    # Truncated on purpose: `inspect` is triage, and a 4,000-token prompt
    # scrolling past is the opposite of it. `--json` gives the whole thing.
    line("prompt", _clip(summary["prompt"]))
    line("answer", _clip(summary["generation"]))
    if summary["precision"]:
        print()
        line("precision", summary["precision"])
    return 0


def _clip(text: str, width: int = 62) -> str:
    flat = " ".join((text or "").split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


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

        def do_GET(self):  # stdlib's spelling
            if not self._is_ours():
                return
            if self.path.split("?")[0] == f"/{name}":
                self._payload_headers()
                self.wfile.write(payload)
                return
            super().do_GET()

        def do_HEAD(self):
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


def _tree_bytes(root) -> int:
    """Bytes this tree occupies on disk.

    `lstat`, not `stat`, and symlinks are not followed. The HuggingFace cache
    is built out of `snapshots/` symlinks pointing at `blobs/`, so following
    them counts every weight file two or three times: an 8 GB cache was
    reported at 20 GB, in the confirmation prompt for deleting it.
    """
    total = 0
    try:
        for f in root.rglob("*"):
            try:
                st = f.lstat()
            except OSError:
                # A file that vanished between the walk and the lstat, or one
                # this account cannot stat. Skipping just this entry is what
                # keeps the walk going: without it the outer handler catches
                # the same OSError and returns whatever had accumulated so
                # far, so one unreadable file near the start would report a
                # cache of gigabytes as almost empty. Either way the number
                # can only be an undercount, and this is the smaller one.
                continue
            # S_ISREG on the entry itself: a symlink contributes only its own
            # tiny inode, and the blob it points at is counted once, where it
            # actually lives.
            if not f.is_symlink() and st.st_mode & 0o170000 == 0o100000:
                total += st.st_size
    except OSError:
        return total
    return total


def uninstall(*, yes: bool = False, models: bool = False) -> int:
    """Remove everything ModelMRI has written, after showing what that is.

    Leaving a trail nobody can find is the same discourtesy as a bad install.
    Every location is resolved by `paths`, per-platform and per-account, so
    this deletes *your* directories on *your* OS — there is no list of
    guessed paths here to go stale.

    Two things it deliberately will not do without being asked:

    * The HuggingFace cache is shared. `transformers`, `datasets` and every
      other tool on the machine read the same directory, so deleting it as
      part of removing one app would take other people's downloads with it.
      `--models` opts in, and the size is shown either way.
    * It cannot remove the installed package while running from inside it, so
      it prints the `pip uninstall` line rather than pretending.
    """
    from pathlib import Path

    from . import paths

    print(f"ModelMRI {__version__} — what is on this machine\n")

    targets: list[tuple[str, Path]] = []
    kept: list[Path] = []
    for label, path in (
        ("data", paths.data_dir()),
        ("config", paths.config_dir()),
        ("cache", paths.cache_dir()),
        ("legacy", paths.legacy_root()),
    ):
        if path is None:
            continue
        resolved = Path(path)
        if not resolved.exists():
            continue
        # On Windows cache_dir() is data_dir()/Cache — nested, not equal — so
        # an equality check saw two distinct paths, listed both, deleted the
        # parent, and then reported the child as a failure it had itself
        # caused. Containment is the test that matches the comment.
        if any(resolved == k or k in resolved.parents for k in kept):
            continue
        kept.append(resolved)
        targets.append((label, resolved))

    if not targets:
        print("  nothing to remove — ModelMRI has not written anything here.")
    for label, path in targets:
        print(f"  {label:<8} {path}  ({_tree_bytes(path) / 1e6:.1f} MB)")

    # Existence, not size, decides whether this is disclosed and whether
    # `--models` acts on it. Gating on bytes meant an empty-but-present cache
    # silently did nothing under `--models`, and the SHARED warning — the
    # whole reason this is opt-in — disappeared with it.
    hub = paths.hf_hub_cache()
    if hub.exists():
        hub_bytes = _tree_bytes(hub)
        print(f"\n  models   {hub} ({hub_bytes / 1e9:.2f} GB)")
        print(
            "           SHARED with transformers, datasets and anything else "
            "using\n           the HuggingFace cache."
            + (" Deleting it, as asked." if models else " Left alone.")
        )
        if models:
            targets.append(("models", hub))

    if not targets:
        return 0

    if not yes:
        print()
        try:
            reply = input("Delete the above? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            reply = ""
        if reply not in ("y", "yes"):
            print("nothing deleted.")
            return 1

    import shutil

    freed = 0
    failures = 0
    for label, path in targets:
        before = _tree_bytes(path)
        errors: list[str] = []
        # rmtree stops at the FIRST entry it cannot remove, having already
        # deleted everything it walked before that. Reporting "could not
        # remove <the whole directory>" then tells the user it was left alone
        # when most of it is gone. Collect every failure and re-measure.
        # `onexc` is 3.12+; `onerror` is what 3.10 and 3.11 have, and it is
        # only deprecated, not removed. This package supports >=3.10, so it
        # has to speak both.
        if sys.version_info >= (3, 12):
            shutil.rmtree(path, onexc=lambda _f, p, e: errors.append(f"{p}: {e}"))
        else:
            shutil.rmtree(
                path, onerror=lambda _f, p, info: errors.append(f"{p}: {info[1]}")
            )
        after = _tree_bytes(path) if path.exists() else 0
        freed += before - after
        if errors:
            failures += 1
            print(
                f"  PARTLY removed {label:<8} {path}\n"
                f"    {len(errors)} item(s) could not be deleted; "
                f"{after / 1e6:.1f} MB remains:",
                file=sys.stderr,
            )
            for line in errors[:5]:
                print(f"      {line}", file=sys.stderr)
        else:
            print(f"  removed {label:<8} {path}")

    print(f"\nfreed {freed / 1e6:.1f} MB")
    print("\nThe package itself is still installed. To remove it:")
    print("  pip uninstall modelmri modelmri-record")
    # A partial delete is not success. Exit non-zero so a script that chains
    # off this does not assume the machine is clean.
    return 2 if failures else 0


def main() -> None:
    # BEFORE anything else. huggingface_hub computes its cache constants at
    # import time, so this has to win that race -- every reader inside
    # ModelMRI re-reads the environment at call time and will follow.
    from . import paths as _paths

    _paths.adopt_models_home()

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
                # This pair is exact, not defensive, and was checked by
                # provoking each case on CPython 3.13.12: a closed underlying
                # buffer gives ValueError("I/O operation on closed file."), a
                # detached one ValueError("underlying buffer has been
                # detached"), and a stream already read from gives
                # io.UnsupportedOperation — which subclasses BOTH OSError and
                # ValueError, so the tuple already had it. A bad codec would
                # be LookupError, but the encoding here is the literal
                # "utf-8", so there is no way to reach it. Streams that are
                # not TextIOWrapper (io.StringIO under pytest's capture) have
                # no `.reconfigure` at all and never get here — the getattr
                # above stops them.
                #
                # Carrying on is right because this is the fallback, not the
                # feature: a stream we cannot reconfigure is one nobody is
                # reading, and refusing to start `modelmri` over the encoding
                # of a closed stdout would be the actual failure.
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

    # `open` starts a viewer; this one prints and exits. Someone triaging an
    # issue with six attached `.mri` files wants to know which is which
    # without opening six browser tabs, and a `.mri` is JSON under a gzip
    # header, so answering that needs no browser, no model and no torch.
    reader = sub.add_parser(
        "inspect", help="Print what a .mri contains, without opening anything"
    )
    reader.add_argument("file", help="the .mri to describe")
    reader.add_argument(
        "--json",
        action="store_true",
        help="emit the summary as JSON instead of text",
    )

    sub.add_parser("where", help="Print every directory ModelMRI reads or writes")

    # `pip install` cannot run this for you: a wheel is an archive and pip does
    # not execute code from it, which is the whole difference between a wheel
    # and an sdist. So the capability check lives here and on the serve banner,
    # where it can also give a different answer to "can I open a recording"
    # than to "can I load a 7B model".
    sub.add_parser(
        "doctor",
        help="Report what this machine can and cannot run, and why",
    )

    remove = sub.add_parser(
        "uninstall", help="Remove everything ModelMRI has written to this machine"
    )
    remove.add_argument(
        "--yes", action="store_true", help="skip the confirmation prompt"
    )
    remove.add_argument(
        "--models",
        action="store_true",
        help="also delete the HuggingFace model cache (shared with other tools)",
    )

    args = parser.parse_args()
    if args.command == "doctor":
        from . import doctor as _doctor

        # SystemExit, not a bare return of an int: `main` is annotated -> None
        # and every other exit code in this file is raised, not returned. A
        # function that returns None on most paths and 1 on one is a function
        # whose caller has to know which.
        raise SystemExit(_doctor.write_to())

    if args.command == "serve":
        import uvicorn

        # Measured on THIS machine at startup, every time. A user who lands on
        # a page saying "no model loaded" should already know whether that is
        # a choice or a limit.
        from . import doctor as _doctor

        report = _doctor.check()
        print(f"ModelMRI {__version__} serving on http://{args.host}:{args.port}")
        print(f"  {_doctor.one_line(report)}")
        for blocker in report.blockers:
            print(f"  PROBLEM  {blocker}", file=sys.stderr)
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
    elif args.command == "inspect":
        raise SystemExit(inspect_session(args.file, as_json=args.json))
    elif args.command == "uninstall":
        raise SystemExit(uninstall(yes=args.yes, models=args.models))
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
        print("    MODELMRI_MODELS_HOME where downloaded models go")
        print("    MODELMRI_MODELS_DIR  extra places to look for your models")
        print("    MODELMRI_TRACE_DIR   where undelivered traces are written")
        print("    HF_HOME/HF_HUB_CACHE where models download (HuggingFace's)")
    else:
        parser.print_help()
