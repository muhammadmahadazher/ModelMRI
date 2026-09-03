# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""An `mri_trace` fixture, so a trace produced inside a test can gate a merge.

Opik ships a PyTest integration, Laminar runs evals in CI, Braintrust has a
`bt` CLI. All three need their platform reachable from the runner: an API key
in the build, and a vendor in the critical path of every merge.

This needs neither. Delivery is redirected into an in-memory document, so a
test asserts against the steps directly, with no server, no SQLite file, no
network and no account. It runs in a GitHub Actions container with the
network off.

## What this cannot do for you

It requires YOUR AGENT TO BE CALLABLE FROM A TEST. If the agent only runs as a
CLI that shells out, reads argv and exits, no fixture can wrap it — you would
need to expose a function first. Saying otherwise would be selling a feature
that quietly does not apply to a large share of the people reading it.

## Usage

    def test_the_planner_does_not_thrash(mri_trace):
        with mri_trace("planning") as t:
            my_agent.run("book me a flight")
        t.assert_no_errors()
        t.assert_max_steps(20)
        t.assert_no_loops()

Every assertion is structural. The failure message names the steps, so a red
CI log points at something rather than saying `assert False`.
"""

from __future__ import annotations

import pytest

import modelmri_record


class TraceAssertions:
    """The recorded steps, and structural assertions over them.

    Wraps the document rather than the store: nothing here touches disk, so
    the same test runs identically on a laptop and in a container.
    """

    def __init__(self, name: str):
        self.name = name
        self.doc: dict = {"id": "", "name": name, "steps": []}

    # ---------------------------------------------------------- the steps

    @property
    def steps(self) -> list:
        return self.doc.get("steps") or []

    def of_kind(self, kind: str) -> list:
        return [s for s in self.steps if s.get("kind") == kind]

    def named(self, name: str) -> list:
        return [s for s in self.steps if s.get("name") == name]

    # ----------------------------------------------------- the assertions

    def assert_no_errors(self) -> None:
        bad = [s for s in self.steps if s.get("error")]
        if bad:
            names = ", ".join(str(s.get("name") or s.get("id")) for s in bad[:5])
            raise AssertionError(
                f"{len(bad)} step(s) recorded an error: {names}"
                + (" …" if len(bad) > 5 else "")
            )

    def assert_max_steps(self, limit: int) -> None:
        if len(self.steps) > limit:
            raise AssertionError(f"{len(self.steps)} steps, limit {limit}")

    def assert_no_retry_storms(self, window_ms: int | None = None) -> None:
        found = self._analyse(window_ms)
        if found.retry_storms:
            worst = found.retry_storms[0]
            raise AssertionError(
                f"{worst.label} failed {worst.count} times in a row "
                f"(steps {', '.join(worst.step_ids[:4])})"
            )

    def assert_no_loops(self) -> None:
        found = self._analyse(None)
        if not found.cycles_scanned:
            # A green assertion from a scan that did not run is worse than a
            # red one, and the same rule holds in `modelmri check`.
            raise AssertionError(
                f"{found.n_steps} steps is over the limit the cycle scan runs "
                f"on, so this cannot assert there are no loops"
            )
        if found.cycles:
            worst = found.cycles[0]
            raise AssertionError(
                f"a sequence of {worst.cycle_length} steps repeated "
                f"{worst.count} times back to back: {worst.label}"
            )

    def assert_max_repeat(self, limit: int) -> None:
        found = self._analyse(None)
        for repeat in found.repeats:
            if repeat.count > limit:
                raise AssertionError(
                    f"{repeat.label} ran {repeat.count} times with the same "
                    f"input, limit {limit}"
                )

    def _analyse(self, window_ms):
        # Imported here, not at module scope: this plugin loads in every
        # pytest run in the environment, including ones that have nothing to
        # do with ModelMRI, and `modelmri` is not a dependency of the
        # recorder. A test that never calls an assertion never pays for it,
        # and a test that does gets a clear ImportError naming the package.
        try:
            from modelmri import patterns
        except ImportError as err:  # pragma: no cover - environment-specific
            raise AssertionError(
                "the structural assertions need the `modelmri` package "
                "installed beside `modelmri-record`; `assert_no_errors` and "
                "`assert_max_steps` work without it."
            ) from err
        if window_ms is None:
            return patterns.analyse(self.steps)
        return patterns.analyse(self.steps, window_ms=window_ms)


@pytest.fixture
def mri_trace():
    """Record a trace inside a test, delivered to memory rather than a store.

    Yields a factory used as a context manager, so one test can record more
    than one run and assert on each separately.
    """
    made: list = []

    class _Factory:
        def __call__(self, name: str = "test"):
            return _Recording(name, made)

        @property
        def all(self) -> list:
            return list(made)

    yield _Factory()


class _Recording:
    """`with mri_trace("name") as t:` — the trace block, captured in memory."""

    def __init__(self, name: str, registry: list):
        self.name = name
        self.registry = registry
        self.result = TraceAssertions(name)
        self._ctx = None
        self._restore = None

    def __enter__(self):
        # Redirect delivery. `_deliver` is what ships a finished trace, so
        # replacing it captures the run without a server, a file or a port.
        #
        # It takes the `_Trace`, not a document — `t.document()` is what
        # builds the dict. Assuming a document here (the first version of
        # this) silently captured nothing and every assertion passed on an
        # empty step list, which is the "green tick that verified nothing"
        # failure this whole feature exists to prevent.
        self._restore = getattr(modelmri_record, "_deliver", None)
        if self._restore is None:  # pragma: no cover - shape guard
            raise RuntimeError(
                "this modelmri-record has no _deliver to redirect, so a trace "
                "recorded here would go to the real endpoint. Refusing rather "
                "than letting a test post to a server."
            )

        def capture(t, *args, **kwargs):
            t.delivered = True
            self.result.doc = t.document()
            return None

        modelmri_record._deliver = capture
        self._ctx = modelmri_record.trace(self.name)
        self._ctx.__enter__()
        return self.result

    def __exit__(self, *exc):
        try:
            if self._ctx is not None:
                self._ctx.__exit__(*exc)
        finally:
            if self._restore is not None:
                modelmri_record._deliver = self._restore
            self.registry.append(self.result)
        return False
