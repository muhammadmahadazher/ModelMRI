# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""The `mri_trace` fixture, exercised by running pytest inside pytest.

`pytester` runs a real pytest session in a temp directory, so these are not
assertions about the plugin's internals — they are the plugin doing its job
for a test file that looks like one somebody would write.

The failure this guards hardest: a fixture that captures NOTHING makes every
structural assertion pass on an empty step list. That is a green test suite
that verified nothing, which is worse than no plugin at all. The first draft
did exactly that by assuming `_deliver` took a document rather than a trace.
"""

from __future__ import annotations

import pytest

pytest_plugins = ["pytester"]


@pytest.fixture
def plugin_args(pytestconfig):
    """`-p modelmri_record.pytest_plugin`, but ONLY when it is not already on.

    The plugin ships a `[project.entry-points.pytest11]` entry, so wherever the
    package is INSTALLED pytest auto-loads it as `modelmri_record`. Naming the
    module again then registers the same object under a second name:

      ValueError: Plugin already registered under a different name:
      modelmri_record=<module 'modelmri_record.pytest_plugin' ...>

    Passing it unconditionally worked locally, where this repo runs on
    PYTHONPATH and no entry point exists to auto-load — and broke all four
    cross-platform CI jobs, where it does. `hasplugin` is the public way to
    ask, so this works in both.
    """
    if pytestconfig.pluginmanager.hasplugin("modelmri_record"):
        return ()
    return ("-p", "modelmri_record.pytest_plugin")


@pytest.fixture(autouse=True)
def _plugin_on_path(pytester, monkeypatch):
    """Make the recorder and modelmri importable inside the inner session."""
    import modelmri_record

    import modelmri

    roots = [
        str(__import__("pathlib").Path(modelmri.__file__).parent.parent),
        str(__import__("pathlib").Path(modelmri_record.__file__).parent.parent),
    ]
    monkeypatch.setenv("PYTHONPATH", __import__("os").pathsep.join(roots))
    return pytester


def test_the_fixture_actually_captures_the_steps(pytester, plugin_args):
    """If it captured nothing, every assertion below would pass vacuously —
    so this counts the steps before asserting anything about them."""
    pytester.makepyfile(
        """
        import modelmri_record as rec

        def test_it(mri_trace):
            with mri_trace("run") as t:
                with rec.step("llm_call", name="plan", input="hello"):
                    pass
                with rec.step("tool_call", name="search", input="q"):
                    pass
            assert len(t.steps) == 2, t.steps
            assert [s["kind"] for s in t.steps] == ["llm_call", "tool_call"]
            assert t.of_kind("llm_call")[0]["name"] == "plan"
        """
    )
    pytester.runpytest(*plugin_args).assert_outcomes(passed=1)


def test_assert_no_errors_fails_on_a_failing_step(pytester, plugin_args):
    pytester.makepyfile(
        """
        import modelmri_record as rec

        def test_it(mri_trace):
            with mri_trace("run") as t:
                rec.step("tool_call", name="search", input="q", error=True)
            t.assert_no_errors()
        """
    )
    result = pytester.runpytest(*plugin_args)
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*1 step(s) recorded an error: search*"])


def test_assert_no_loops_fails_on_a_repeating_sequence(pytester, plugin_args):
    pytester.makepyfile(
        """
        import modelmri_record as rec

        def test_it(mri_trace):
            with mri_trace("run") as t:
                for _ in range(4):
                    rec.step("llm_call", name="think", input="next?")
                    rec.step("tool_call", name="act", input="go")
            t.assert_no_loops()
        """
    )
    result = pytester.runpytest(*plugin_args)
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*sequence of 2 steps repeated 4 times*"])


def test_assert_max_steps_names_the_limit(pytester, plugin_args):
    pytester.makepyfile(
        """
        import modelmri_record as rec

        def test_it(mri_trace):
            with mri_trace("run") as t:
                for i in range(5):
                    rec.step("tool_call", name=f"s{i}", input=str(i))
            t.assert_max_steps(3)
        """
    )
    result = pytester.runpytest(*plugin_args)
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*5 steps, limit 3*"])


def test_a_clean_run_passes_every_assertion(pytester, plugin_args):
    pytester.makepyfile(
        """
        import modelmri_record as rec

        def test_it(mri_trace):
            with mri_trace("run") as t:
                rec.step("llm_call", name="plan", input="a")
                rec.step("tool_call", name="fetch", input="b")
            t.assert_no_errors()
            t.assert_max_steps(10)
            t.assert_no_loops()
            t.assert_no_retry_storms()
            t.assert_max_repeat(1)
        """
    )
    pytester.runpytest(*plugin_args).assert_outcomes(passed=1)


def test_one_test_can_record_more_than_one_run(pytester, plugin_args):
    pytester.makepyfile(
        """
        import modelmri_record as rec

        def test_it(mri_trace):
            with mri_trace("first") as a:
                rec.step("llm_call", name="plan", input="a")
            with mri_trace("second") as b:
                rec.step("tool_call", name="fetch", input="b")
                rec.step("tool_call", name="fetch", input="b")
            assert len(a.steps) == 1
            assert len(b.steps) == 2
            assert len(mri_trace.all) == 2
        """
    )
    pytester.runpytest(*plugin_args).assert_outcomes(passed=1)


def test_nothing_is_delivered_to_a_real_endpoint(pytester, plugin_args):
    """The whole point is that this runs with the network off. If delivery
    were not redirected, this would attempt a real request."""
    pytester.makepyfile(
        """
        import modelmri_record as rec
        import urllib.request

        def test_it(mri_trace, monkeypatch):
            def boom(*a, **k):
                raise AssertionError("the plugin tried to reach the network")
            monkeypatch.setattr(urllib.request, "urlopen", boom)
            with mri_trace("run") as t:
                rec.step("llm_call", name="plan", input="a")
            assert len(t.steps) == 1
        """
    )
    pytester.runpytest(*plugin_args).assert_outcomes(passed=1)


def test_delivery_is_restored_after_the_block(pytester, plugin_args):
    """A test that leaves `_deliver` replaced would silently swallow every
    trace the rest of the suite records."""
    pytester.makepyfile(
        """
        import modelmri_record as rec

        def test_it(mri_trace):
            before = rec._deliver
            with mri_trace("run") as t:
                rec.step("llm_call", name="plan", input="a")
            assert rec._deliver is before
        """
    )
    pytester.runpytest(*plugin_args).assert_outcomes(passed=1)


def test_an_exception_inside_the_block_still_restores_delivery(pytester, plugin_args):
    pytester.makepyfile(
        """
        import pytest
        import modelmri_record as rec

        def test_it(mri_trace):
            before = rec._deliver
            with pytest.raises(ValueError):
                with mri_trace("run") as t:
                    rec.step("llm_call", name="plan", input="a")
                    raise ValueError("boom")
            assert rec._deliver is before
        """
    )
    pytester.runpytest(*plugin_args).assert_outcomes(passed=1)
