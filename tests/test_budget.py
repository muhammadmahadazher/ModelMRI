# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""The preflight has to be wrong in the safe direction, or not answer.

Two failure modes are worth more than all the others here, and they are
mirror images. Reading "could not measure" as zero COST approves an analysis
that will run the card out of memory; reading it as zero FREE refuses every
analysis on a machine that simply has no accelerator to ask. Both are the
`.get(name, 0.0)` shape that made 206 robot episodes show one video, so most
of this file is about `None` staying `None`.

The third is arithmetic: time multiplies across sequential passes and peak
memory does not. Multiplying the peak by 132 would refuse every analysis this
tool offers, which is why `retained_bytes` is a parameter the caller states
rather than something this module infers.
"""

from __future__ import annotations

import pytest

from modelmri import budget


def _mem(**kw) -> budget.Memory:
    return budget.Memory(**kw)


def _probe(seconds=0.1, **kw) -> budget.Probe:
    return budget.Probe(seconds=seconds, memory=_mem(**kw), device_kind="cuda")


# ------------------------------------------------------------------ arithmetic


def test_time_multiplies_across_passes():
    est = budget.project(_probe(seconds=0.25), 132)
    assert est.seconds == pytest.approx(33.0)
    assert est.passes == 132


def test_peak_memory_does_not_multiply():
    """The whole point. 132 sequential passes hold one pass's peak."""
    est = budget.project(
        _probe(peak_bytes=1_400_000_000, free_bytes=8_600_000_000), 132
    )
    assert est.peak_bytes == 1_400_000_000


def test_retained_bytes_are_added_once_not_per_pass():
    est = budget.project(
        _probe(peak_bytes=1_000_000_000, free_bytes=8_000_000_000),
        100,
        retained_bytes=500_000_000,
    )
    assert est.peak_bytes == 1_500_000_000
    assert any("across passes" in n for n in est.notes)


def test_zero_passes_is_a_programming_error():
    with pytest.raises(ValueError):
        budget.project(_probe(), 0)


# ------------------------------------------------- unknown is never a number


def test_unmeasured_cost_does_not_become_zero():
    """No peak reading => no verdict, and nothing that reads as 'free'."""
    est = budget.project(_probe(free_bytes=8_000_000_000, reason="no allocator"), 132)
    assert est.peak_bytes is None
    assert est.fraction_of_free is None
    assert est.verdict == "unknown"
    assert est.unmeasured == "no allocator"


def test_unmeasured_free_memory_does_not_become_zero():
    """Knowing the cost but not the budget must not refuse everything."""
    est = budget.project(_probe(peak_bytes=4_000_000_000), 132)
    assert est.peak_bytes == 4_000_000_000
    assert est.free_bytes is None
    assert est.fraction_of_free is None
    assert est.verdict == "unknown"
    assert any("not checked against a budget" in n for n in est.notes)


def test_unknown_verdict_never_raises():
    est = budget.project(_probe(reason="cpu"), 10_000)
    budget.check(est, label="a huge sweep")  # must not raise


def test_a_pass_that_frees_more_than_it_takes_reads_as_no_cost_not_negative():
    mem = _mem(peak_bytes=0, free_bytes=8_000_000_000)
    est = budget.project(budget.Probe(0.1, mem, "cuda"), 10)
    assert est.peak_bytes == 0
    assert est.fraction_of_free == 0.0


# ---------------------------------------------------------------- the verdict


@pytest.mark.parametrize(
    "peak,free,expected",
    [
        (1_000_000_000, 8_000_000_000, "ok"),  # 12%
        (5_600_000_000, 8_000_000_000, "tight"),  # 70%
        (7_600_000_000, 8_000_000_000, "refuse"),  # 95%
    ],
)
def test_verdict_tracks_the_fraction_of_free(peak, free, expected):
    est = budget.project(_probe(peak_bytes=peak, free_bytes=free), 50)
    assert est.verdict == expected


def test_check_refuses_and_names_both_numbers():
    est = budget.project(
        _probe(peak_bytes=7_600_000_000, free_bytes=8_000_000_000), 132
    )
    with pytest.raises(budget.TooCostly) as caught:
        budget.check(est, label="Ranking every head")

    message = str(caught.value)
    assert "7.6 GB" in message and "8.0 GB" in message
    assert "Ranking every head" in message
    assert caught.value.overridable is True


def test_check_is_overridable():
    est = budget.project(
        _probe(peak_bytes=7_600_000_000, free_bytes=8_000_000_000), 132
    )
    assert budget.check(est, label="x", confirm=True) is est


def test_too_costly_is_a_refusal_so_the_server_answers_409():
    """errors.py reserves 409 for a deliberate no with a sentence."""
    from modelmri.errors import Refusal

    assert issubclass(budget.TooCostly, Refusal)


# ------------------------------------------------------- honesty of the label


def test_every_estimate_says_what_it_was_built_from():
    est = budget.project(_probe(peak_bytes=1, free_bytes=2), 5)
    assert "one probe pass" in est.basis


def test_to_dict_is_json_safe():
    import json

    est = budget.project(_probe(peak_bytes=1_000, free_bytes=8_000), 7)
    json.dumps(est.to_dict())  # must not raise
    assert est.to_dict()["verdict"] == "ok"


# ------------------------------------------------------- the machine we're on


def test_cpu_reports_no_budget_rather_than_zero():
    mem = budget.free_memory("cpu")
    assert mem.free_bytes is None
    assert mem.total_bytes is None
    assert "no accelerator" in mem.reason


def test_mps_does_not_pass_off_a_ceiling_as_free_memory():
    mem = budget.free_memory("mps")
    assert mem.free_bytes is None
    assert "one pool" in mem.reason


def test_reset_peak_on_cpu_says_it_did_not_take():
    assert budget.reset_peak("cpu") is False


def test_probe_still_times_a_pass_with_no_accelerator():
    """The memory half being unavailable must not cost us the time half."""
    calls = []

    def run():
        calls.append(1)
        sum(range(50_000))

    probe = budget.probe_pass(run, "cpu")
    assert calls == [1]
    assert probe.seconds > 0
    assert probe.memory.peak_bytes is None
    assert probe.memory.reason


def test_probe_runs_the_callable_exactly_once():
    calls = []
    budget.probe_pass(lambda: calls.append(1), "cpu")
    assert len(calls) == 1
