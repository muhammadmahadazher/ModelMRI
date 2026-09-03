# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""Isolation the whole suite depends on but nothing was enforcing.

`custom._SESSION_ROOTS` is module-level: the folders a person has asked this
run to also look in for adapters. That is right for the product — one server,
one process, one allow-list, and the docstring's promise that it "does not
survive a restart" holds. It is wrong for a test suite, where one process runs
hundreds of tests and a root added by one of them stays visible to every test
after it.

`custom.clear_roots()` was written for exactly this and its docstring says
"Used by the tests" — it just was not wired to run between them. So a test
that called `add_root(tmp_path)` left its sandbox in the allow-list, and
`test_no_machine_leaks` later read `/api/custom`, saw a directory belonging to
a different test, and correctly reported it as a path from outside its own
sandbox.

It surfaced only once `h5py`, `pyarrow` and `lm-format-enforcer` went into the
dev group: the test that adds the root had been skipping. The isolation gap
was always there, waiting for the suite to be turned all the way on.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _forget_session_roots():
    """No test inherits another's adapter search path.

    Cleared on the way in as well as out: a test that fails partway through
    should not poison the rest of the file, and the ordering under xdist is
    not something any single test can reason about.
    """
    from modelmri import custom

    custom.clear_roots()
    yield
    custom.clear_roots()
