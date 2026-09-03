# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""`python -m modelmri` — so the conventional guess reaches the real CLI.

There was no `if __name__ == "__main__"` anywhere under `modelmri/` and no
`__main__.py`, so the guess one directory down did the worst thing a command
can do: MEASURED, `python -m modelmri.cli doctor --help`,
`python -m modelmri.cli totally-bogus --nonsense` and
`python -m modelmri.cli check /nonexistent.json --no-errors` each returned
rc=0 with zero bytes on stdout AND stderr. `-m` imports a module and runs
whatever the module runs; that one defines `main` and never calls it.

The third of those is why this is worth a file. `modelmri check` is a CI gate,
and a gate that exits 0 without running is indistinguishable from a gate that
ran and passed — the same shape `modelmri_record.__main__` was written
against ("a trace that never existed looks exactly like one that was never
started"), one command line over. A green tick nobody earned is worse than a
red one, because nobody goes looking.

`python -m modelmri` itself already failed loudly — "'modelmri' is a package
and cannot be directly executed" — and the supported entry point, the
`modelmri` console script, was healthy throughout: `main()` raises SystemExit
with argparse's own code for `--help` and for a bad command. This routes the
guess to that same `main`, so there is one entry point with one behaviour
rather than a documented one that answers and an undocumented one that lies.
"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    # `main()` returns None; the interesting exit codes come from argparse
    # raising SystemExit on its way through (0 for `--help`, 2 for a command
    # that does not exist) and travel untouched. SystemExit(None) is 0, which
    # is the right answer for a command that ran and finished.
    raise SystemExit(main())
