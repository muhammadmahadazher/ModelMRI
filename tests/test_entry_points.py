# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""`python -m ...` has to either run the thing or say it did not.

Two entry points did neither, in opposite directions, and both failures are
the same shape: the process exits, or does not, in a way that is
indistinguishable from having worked.

`python -m modelmri.cli <anything>` returned rc=0 with ZERO bytes on stdout
and stderr — MEASURED on `doctor --help`, on `totally-bogus --nonsense`, and
on `check /nonexistent.json --no-errors`. `-m` imports a module and runs
whatever the module runs, and `cli.py` defines `main` without ever calling it.
The third of those is the one that matters: `modelmri check` is a CI gate, and
a gate that exits 0 having done nothing is a green tick nobody earned. There
was no `__main__.py` either, so the conventional guess `python -m modelmri`
could not run at all.

What is pinned below is `python -m modelmri`, which now routes to the same
`cli.main` the console script does. The `if __name__ == "__main__"` guard that
would also make the `modelmri.cli` spelling work belongs at the foot of
`cli.py`; these tests do not assert it, because asserting the current rc=0
would be pinning the defect.

`python -m modelmri_policy --help` did the mirror image: its argv scan looked
for `--port` and ignored everything else, so `--help` fell through to
`server.main()`. MEASURED: it printed `MODELMRI_POLICY_PORT=53649` and served
until it was killed at a 10s timeout (rc=124). Somebody asking a sidecar for
its usage got a listening socket, and the ready line is the only output either
way, so there is nothing on screen to say which happened.

Nothing here binds a socket, loads a model or downloads anything: every
process below is expected to exit on its own, and each is bounded anyway.
"""

from __future__ import annotations

import re
import subprocess
import sys

import pytest
from modelmri_policy.contract import READY_PREFIX

# Generous, because it covers a cold interpreter start plus importing the CLI
# on a Windows runner with a virus scanner in the path -- not because anything
# here is expected to take seconds. A hang is the failure being tested for, so
# `timeout` expiring has to be a test failure and not a wait.
BUDGET = 120


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *args],
        capture_output=True,
        text=True,
        timeout=BUDGET,
    )


def served(stdout: str) -> bool:
    """Did this process print the ready line — i.e. actually bind a socket?

    A whole line of `MODELMRI_POLICY_PORT=<digits>`, not the substring: the
    `--help` epilog quotes the prefix while documenting it, and a substring
    test reads that sentence as a running server.
    """
    return bool(re.search(rf"^{re.escape(READY_PREFIX)}\d+$", stdout, re.M))


# --------------------------------------------------- python -m modelmri


def test_a_command_that_does_not_exist_is_refused_rather_than_ignored():
    """rc=2 and a sentence, where the module path gave rc=0 and no bytes."""
    done = run("-m", "modelmri", "totally-bogus", "--nonsense")

    assert done.returncode == 2, (
        f"argparse's own code for a bad command; got {done.returncode} with "
        f"stdout={done.stdout[:200]!r} stderr={done.stderr[:200]!r}"
    )
    assert "totally-bogus" in done.stderr, "the refusal has to name what was sent"


def test_a_gate_that_cannot_run_exits_non_zero(tmp_path):
    """The CI case, and the reason this file exists.

    `modelmri check` is meant to fail a pull request. Run as
    `python -m modelmri.cli` it exited 0 without reading anything, which a CI
    log renders identically to a check that ran and passed.
    """
    missing = tmp_path / "nonexistent.json"
    done = run("-m", "modelmri", "check", str(missing), "--no-errors")

    assert done.returncode != 0, (
        "a check that could not read its input reported success: "
        f"stdout={done.stdout[:200]!r}"
    )
    assert "nonexistent.json" in (done.stdout + done.stderr), (
        "the refusal has to name the path it could not read"
    )


def test_the_package_entry_point_answers_at_all():
    """`python -m modelmri` used to be "'modelmri' is a package and cannot be
    directly executed" — loud, at least, but the guess it refused is the one
    people make."""
    done = run("-m", "modelmri", "--help")

    assert done.returncode == 0
    assert "usage: modelmri" in done.stdout
    # The same parser the console script drives, not a second one that could
    # drift from it.
    assert "serve" in done.stdout and "check" in done.stdout


# -------------------------------------------- python -m modelmri_policy


def test_asking_the_sidecar_for_its_usage_does_not_start_it():
    """A `--help` that serves forever is the worst answer available: the
    terminal shows the ready line, which is what a successful START looks
    like."""
    done = run("-m", "modelmri_policy", "--help")

    assert done.returncode == 0
    assert not served(done.stdout), "it bound a socket instead of answering"
    assert "usage: modelmri-policy" in done.stdout
    assert "--port" in done.stdout


@pytest.mark.parametrize("port", ["70000", "-1", "abc"])
def test_a_port_the_socket_would_reject_is_refused_by_the_parser(port):
    """MEASURED before this: `--port 70000` and `--port -1` both reached
    `ThreadingHTTPServer` and escaped as an unhandled `OverflowError: bind():
    port must be 0-65535` — a traceback out of a socket call, which reads as
    the sidecar being broken rather than the argument being wrong. The same
    claim `test_cli_port.py` makes one process up."""
    done = run("-m", "modelmri_policy", "--port", port)

    assert done.returncode == 2, f"got {done.returncode}: {done.stderr[:300]!r}"
    assert "Traceback" not in done.stderr
    assert port in done.stderr, "the refusal has to name the value that was sent"


def test_an_argument_it_does_not_understand_is_named_not_dropped():
    """The silent half of the same defect: the scan looked only for `--port`,
    so `--prot 5000` was discarded and the sidecar came up on an OS-chosen
    port — a request nobody made, answered as though it had been."""
    done = run("-m", "modelmri_policy", "--prot", "5000")

    assert done.returncode == 2
    assert not served(done.stdout)
    assert "--prot" in done.stderr
