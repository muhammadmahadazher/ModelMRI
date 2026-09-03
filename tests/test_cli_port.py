# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""`--port` is checked when it is parsed, not when the socket is bound.

`type=int` accepted anything an int can hold, so `--port -1` reached the
socket: `doctor.check()` 5.1s, `import modelmri.server` 15.3s, `create_app()`
0.4s — and only then `OverflowError: bind(): port must be 0-65535`, delivered
as a traceback with a chained CancelledError and no clean shutdown.

Seventeen seconds to reject a number argparse rejects in a millisecond, and
`--port abc` on the same flag already refused cleanly in under half a second —
so one kind of mistake got two answers, seventeen seconds and two exit codes
apart.

Nothing here binds a socket or imports the server; the point is that the
refusal arrives before any of that work is done.
"""

from __future__ import annotations

import sys
import time

import pytest

from modelmri import cli


def _serve(port: str) -> tuple[int, float]:
    """Run `modelmri serve --port <port>`, returning (exit code, seconds)."""
    argv = sys.argv
    started = time.monotonic()
    try:
        sys.argv = ["modelmri", "serve", "--port", port]
        with pytest.raises(SystemExit) as caught:
            cli.main()
        return int(caught.value.code or 0), time.monotonic() - started
    finally:
        sys.argv = argv


@pytest.mark.parametrize("port", ["-1", "70000", "1000000000000", "65536"])
def test_a_port_outside_the_range_is_refused_before_any_work(port, capsys):
    code, took = _serve(port)

    assert code == 2, "argparse's own code for a bad argument, as `abc` gets"
    assert took < 5, (
        f"took {took:.1f}s — the refusal happened after the doctor check and "
        f"the server import rather than at parse time"
    )
    said = capsys.readouterr().err
    assert "not a usable port" in said
    assert "0 to 65535" in said, "name the range that does exist"
    assert "5900" in said, "and the default, so there is a next step"


def test_a_port_that_is_not_a_number_keeps_its_own_sentence(capsys):
    """This arm already worked; the fix must not replace argparse's message
    with a worse one."""
    code, _ = _serve("abc")

    assert code == 2
    assert "not a port number" in capsys.readouterr().err


@pytest.mark.parametrize("port", ["0", "1", "5900", "65535"])
def test_a_usable_port_parses(port):
    """So the guard cannot become "refuse everything". 0 is legitimate — it
    asks the OS to pick a free port."""
    assert cli._port(port) == int(port)
