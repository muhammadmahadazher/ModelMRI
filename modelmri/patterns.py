"""Structural facts about a run, computed exactly, offline, with no model asked.

Laminar's Signals asks an LLM to extract a described behaviour. LangSmith
Engine clusters issues with a model. Braintrust's Loop proposes metrics from
patterns. Three model judgements dressed as findings, none of which runs
without a cloud LLM and a key.

A loop is a structural fact about a graph. It is countable exactly, in
milliseconds, on a laptop with the network off, and the count is not a matter
of opinion.

## Counts, never verdicts

This is the whole discipline of the module and it is easy to lose.

Paginating an API 14 times and thrashing against a failing tool 14 times are
STRUCTURALLY IDENTICAL. Nothing here can tell them apart, and nothing here
tries. Every finding says what happened and how many times; none says whether
that was good. The moment a finding is worded as a verdict — "excessive",
"redundant", "should be" — it becomes exactly the model judgement this exists
to replace, except without the model, which is worse.

So: no severity, no score, no threshold above which something becomes a
problem. `means()` reports a count and the reader decides.

## What it cannot see

Repeats are detected by hashing `(kind, name, input)`, so a prompt carrying a
timestamp, a UUID or a cursor is a NEAR-repeat that will not group. That is a
real limit and it belongs in front of the reader, not buried here — `Findings`
carries `near_repeats_not_detected` so the panel can say it.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field

# Two error steps of the same name count as a storm when they land inside this
# many milliseconds of each other. Not a verdict about retries — it is the
# window that decides whether consecutive failures are ONE episode or several,
# and it is reported beside every finding so it can be argued with.
DEFAULT_RETRY_WINDOW_MS = 4_000

# A cycle is a contiguous run of steps that repeats back to back. Bounds, both
# reported rather than silent:
MIN_CYCLE_LEN = 2
MAX_CYCLE_LEN = 12
# Above this many steps the cycle scan is skipped entirely rather than run
# partially — a half-scanned trace reporting "no cycles" is a wrong answer,
# where "not scanned" is a true one.
MAX_STEPS_FOR_CYCLES = 4_000

# How much of a step's input feeds the repeat hash. Hashing megabytes to
# compare two steps is waste; the prefix that decides equality for a
# pathological repeat is short.
INPUT_HASH_CHARS = 4_000


def _signature(step: dict) -> str:
    """The identity a repeat is counted against: (kind, name, input)."""
    payload = "\x00".join(
        (
            str(step.get("kind") or ""),
            str(step.get("name") or ""),
            str(step.get("input") or "")[:INPUT_HASH_CHARS],
        )
    )
    return hashlib.sha256(payload.encode("utf-8", "replace")).hexdigest()[:16]


@dataclass
class Finding:
    """One structural fact, and the steps it is made of.

    `count` is the number of occurrences. There is deliberately no severity
    and no threshold: see the module docstring.
    """

    kind: str  # "repeat" | "retry_storm" | "cycle"
    label: str  # what repeated, in the trace's own words
    count: int
    step_ids: list = field(default_factory=list)
    # Only set where it means something; None elsewhere, never 0.
    span_ms: int | None = None
    cycle_length: int | None = None
    signature: str = ""

    def to_dict(self) -> dict:
        out = asdict(self)
        out["means"] = self.means()
        return out

    def means(self) -> str:
        """A count and its steps. Never a judgement about them."""
        if self.kind == "repeat":
            return (
                f"{self.label} ran {self.count} times with the same input. "
                f"Whether that is a loop worth fixing or a page-by-page walk "
                f"of an API is not something this can tell you."
            )
        if self.kind == "retry_storm":
            span = "" if self.span_ms is None else f" within {self.span_ms} ms"
            return (
                f"{self.label} failed {self.count} times in a row{span}. "
                f"This counts consecutive failures of the same name; it does "
                f"not know whether they were retries."
            )
        if self.kind == "cycle":
            return (
                f"a sequence of {self.cycle_length} steps repeated "
                f"{self.count} times back to back: {self.label}."
            )
        return f"{self.label}: {self.count}"


@dataclass
class Findings:
    """Everything structural about one run."""

    repeats: list = field(default_factory=list)
    retry_storms: list = field(default_factory=list)
    cycles: list = field(default_factory=list)
    n_steps: int = 0
    retry_window_ms: int = DEFAULT_RETRY_WINDOW_MS
    # True when the trace was too long to scan for cycles. Reported, because
    # "no cycles found" and "not looked for" are different answers.
    cycles_scanned: bool = True
    # Hashing the input misses a repeat whose prompt carries a timestamp or a
    # cursor. Carried so the panel can say so rather than implying the count
    # is exhaustive.
    near_repeats_not_detected: bool = True

    @property
    def all(self) -> list:
        return [*self.repeats, *self.retry_storms, *self.cycles]

    def to_dict(self) -> dict:
        return {
            "repeats": [f.to_dict() for f in self.repeats],
            "retry_storms": [f.to_dict() for f in self.retry_storms],
            "cycles": [f.to_dict() for f in self.cycles],
            "n_steps": self.n_steps,
            "retry_window_ms": self.retry_window_ms,
            "cycles_scanned": self.cycles_scanned,
            "near_repeats_not_detected": self.near_repeats_not_detected,
            "means": self.means(),
        }

    def means(self) -> str:
        parts = [
            f"{self.n_steps} steps, analysed structurally — no model was asked "
            f"and nothing here is a judgement."
        ]
        if not self.all:
            parts.append(
                "No step ran twice with the same input, no name failed twice "
                "in a row, and no sequence repeated back to back."
            )
        else:
            bits = []
            if self.repeats:
                bits.append(f"{len(self.repeats)} repeated step(s)")
            if self.retry_storms:
                bits.append(f"{len(self.retry_storms)} run(s) of consecutive failures")
            if self.cycles:
                bits.append(f"{len(self.cycles)} repeating sequence(s)")
            parts.append("Found " + ", ".join(bits) + ".")
        if not self.cycles_scanned:
            parts.append(
                f"CYCLES WERE NOT SCANNED: over {MAX_STEPS_FOR_CYCLES} steps the "
                f"scan is skipped rather than run partially, because a "
                f"half-scanned trace reporting 'no cycles' is a wrong answer."
            )
        parts.append(
            "Repeats are matched on the exact input, so a prompt carrying a "
            "timestamp, a cursor or a UUID is a near-repeat that does not "
            "group here and is not counted."
        )
        return " ".join(parts)


def _ordered(steps) -> list:
    """Steps in recorded order, which is `seq` where present.

    A dict from the store already arrives ordered, but a hand-written document
    need not, and comparing consecutive steps in the wrong order invents
    adjacency that did not happen.
    """
    return sorted(
        [s for s in steps if isinstance(s, dict)],
        key=lambda s: (
            s.get("seq") if isinstance(s.get("seq"), int) else 0,
            s.get("started_ms") if isinstance(s.get("started_ms"), int) else 0,
        ),
    )


def find_repeats(steps, *, min_count: int = 2) -> list:
    """Steps sharing an exact (kind, name, input), counted."""
    groups: dict = {}
    for step in steps:
        groups.setdefault(_signature(step), []).append(step)
    out = []
    for sig, members in groups.items():
        if len(members) < min_count:
            continue
        first = members[0]
        out.append(
            Finding(
                kind="repeat",
                label=f"{first.get('kind') or 'step'} {first.get('name') or ''}".strip(),
                count=len(members),
                step_ids=[str(m.get("id") or "") for m in members],
                signature=sig,
            )
        )
    out.sort(key=lambda f: (-f.count, f.label))
    return out


def find_retry_storms(steps, *, window_ms: int = DEFAULT_RETRY_WINDOW_MS) -> list:
    """Consecutive error steps of the same name, close together in time.

    CONSECUTIVE, not merely nearby: two failures with a success between them
    are two failures, and calling that a storm would be the module inventing a
    narrative. A step with no `started_ms` cannot be placed in the window, so
    it ends the run rather than being assumed adjacent.
    """
    out = []
    run: list = []

    def flush():
        if len(run) >= 2:
            times = [s.get("started_ms") for s in run]
            span = (
                max(times) - min(times)
                if all(isinstance(t, int) for t in times)
                else None
            )
            out.append(
                Finding(
                    kind="retry_storm",
                    label=str(run[0].get("name") or run[0].get("kind") or "a step"),
                    count=len(run),
                    step_ids=[str(s.get("id") or "") for s in run],
                    span_ms=span,
                )
            )
        run.clear()

    for step in steps:
        if not step.get("error"):
            flush()
            continue
        name = str(step.get("name") or "")
        started = step.get("started_ms")
        if run:
            prev = run[-1]
            same = str(prev.get("name") or "") == name
            near = (
                isinstance(started, int)
                and isinstance(prev.get("started_ms"), int)
                and (started - prev["started_ms"]) <= window_ms
            )
            if not (same and near):
                flush()
        run.append(step)
    flush()
    out.sort(key=lambda f: -f.count)
    return out


def find_cycles(steps) -> tuple:
    """Contiguous sequences that repeat back to back.

    Returns (findings, scanned). A trace longer than `MAX_STEPS_FOR_CYCLES` is
    not scanned at all, and says so — a partial scan that reports "no cycles"
    is a wrong answer where "not looked for" is a true one.

    The FUNDAMENTAL PERIOD wins. `think act think act think act think act` is
    reported as a 2-step block four times, not a 4-step block twice — both are
    true of the same eight steps, but the shortest period is the one that says
    what is actually repeating. Scanning longest-first (the first version of
    this) reported the 4-step reading and buried the answer.

    Only MAXIMAL repeats are reported: a block claims its steps once, so one
    fact is never listed as three.
    """
    seq = [f"{s.get('kind') or ''}:{s.get('name') or ''}" for s in steps]
    n = len(seq)
    if n > MAX_STEPS_FOR_CYCLES:
        return [], False

    out = []
    covered = [False] * n
    # SHORTEST first, so the fundamental period claims the span before any
    # multiple of it can. A 2-step block repeating consumes all eight steps,
    # leaving nothing for the 4-step reading of the same eight.
    for length in range(MIN_CYCLE_LEN, min(MAX_CYCLE_LEN, n // 2) + 1):
        i = 0
        while i + 2 * length <= n:
            if any(covered[i : i + 2 * length]):
                i += 1
                continue
            block = seq[i : i + length]
            reps = 1
            j = i + length
            while j + length <= n and seq[j : j + length] == block and not any(
                covered[j : j + length]
            ):
                reps += 1
                j += length
            if reps >= 2:
                span = j - i
                for k in range(i, j):
                    covered[k] = True
                out.append(
                    Finding(
                        kind="cycle",
                        label=" → ".join(block),
                        count=reps,
                        cycle_length=length,
                        step_ids=[str(steps[k].get("id") or "") for k in range(i, j)],
                    )
                )
                i += span
            else:
                i += 1
    out.sort(key=lambda f: (-(f.count * (f.cycle_length or 1)), -f.count))
    return out, True


def analyse(steps, *, window_ms: int = DEFAULT_RETRY_WINDOW_MS) -> Findings:
    """Every structural finding for one run."""
    ordered = _ordered(steps)
    cycles, scanned = find_cycles(ordered)
    return Findings(
        repeats=find_repeats(ordered),
        retry_storms=find_retry_storms(ordered, window_ms=window_ms),
        cycles=cycles,
        n_steps=len(ordered),
        retry_window_ms=window_ms,
        cycles_scanned=scanned,
    )


# ------------------------------------------------------------- across runs


@dataclass
class Recurring:
    """One finding, and how many of the recorded runs contain it."""

    kind: str
    label: str
    n_runs: int
    of_runs: int
    total_count: int
    signature: str = ""
    trace_ids: list = field(default_factory=list)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["means"] = self.means()
        return out

    def means(self) -> str:
        return (
            f"{self.label} — in {self.n_runs} of {self.of_runs} recorded run(s), "
            f"{self.total_count} time(s) in total. A pattern in one run is an "
            f"anecdote; this counts the runs it appears in, and stops there."
        )


def across_runs(traces, *, window_ms: int = DEFAULT_RETRY_WINDOW_MS) -> list:
    """The same structural finding, counted over many runs.

    `traces` is an iterable of trace documents. Grouping is by the finding's
    own identity — the input hash for a repeat, the name for a storm, the
    sequence for a cycle — so "this happens on most runs" is answerable.
    """
    seen: dict = {}
    total = 0
    for doc in traces:
        if not isinstance(doc, dict):
            continue
        total += 1
        trace_id = str(doc.get("id") or "")
        found = analyse(doc.get("steps") or [], window_ms=window_ms)
        # A finding occurring twice in ONE run still counts as one run.
        for finding in found.all:
            key = (finding.kind, finding.signature or finding.label)
            entry = seen.get(key)
            if entry is None:
                entry = Recurring(
                    kind=finding.kind,
                    label=finding.label,
                    n_runs=0,
                    of_runs=0,
                    total_count=0,
                    signature=finding.signature,
                )
                seen[key] = entry
            if trace_id not in entry.trace_ids:
                entry.trace_ids.append(trace_id)
                entry.n_runs += 1
            entry.total_count += finding.count
    out = list(seen.values())
    for entry in out:
        entry.of_runs = total
    out.sort(key=lambda e: (-e.n_runs, -e.total_count, e.label))
    return out
