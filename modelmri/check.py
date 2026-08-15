"""Gate a merge on structural facts about a recorded run.

Opik ships a PyTest integration, Laminar runs evals in CI, Braintrust has a
`bt` CLI. All three need their platform reachable from the runner, which means
an API key in the build and a vendor in the critical path of every merge.

This is a stdlib-only recorder and a SQLite file. It runs inside a GitHub
Actions container with no network and no account, and it answers in
milliseconds.

## Only what a change can actually break

The default assertions are STRUCTURAL — error steps, step counts, retry
storms, loops — because those are properties of the agent's behaviour that a
prompt or tool-definition change is responsible for.

Timing and cost gates exist and are OPT-IN, and the help text says they are
flaky. A shared CI runner is slow for reasons that have nothing to do with the
diff, and a wall-clock gate that goes red on a noisy neighbour teaches people
to ignore the check — which costs more than the gate was ever worth.

## What it does not assert

`patterns.py` reports counts, never verdicts, and that discipline survives
here: `--no-loops` fails on a repeat because YOU asked it to, not because a
repeat is bad. Paginating an API 14 times and thrashing 14 times are
structurally identical, and this file cannot tell them apart either.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .errors import BadRequest
from . import patterns as patterns_mod

# Exit codes. Distinct, because a script has to tell them apart:
#
#   0 — every assertion held.
#   1 — an assertion FAILED. The actionable one; this is what gates a merge.
#   2 — NOTHING WAS CHECKED: the trace could not be read, or no assertion was
#       chosen. Never 0. A green tick from a run that verified nothing is the
#       same defect as `--no-loops` passing on a trace too long to scan, and
#       it is how a typo'd flag turns into a build that always succeeds. It is
#       also not 1: a missing file is a broken pipeline, not a broken agent,
#       and folding those together sends people to debug the wrong thing.
PASS, FAILED, NOTHING_CHECKED = 0, 1, 2


@dataclass
class Assertion:
    """One gate, and whether it held."""

    name: str
    ok: bool
    detail: str
    # The steps that made it fail, so CI output points at something.
    step_ids: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def line(self) -> str:
        return f"{'PASS' if self.ok else 'FAIL'}  {self.name}: {self.detail}"


@dataclass
class Result:
    trace_id: str = ""
    name: str = ""
    n_steps: int = 0
    assertions: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(a.ok for a in self.assertions)

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "n_steps": self.n_steps,
            "ok": self.ok,
            "assertions": [a.to_dict() for a in self.assertions],
        }

    def report(self) -> str:
        head = f"modelmri check — {self.name or self.trace_id} ({self.n_steps} steps)"
        if not self.assertions:
            return (
                f"{head}\n\nNo assertions were chosen, so nothing was checked. "
                f"Pass at least one of --no-errors, --max-steps, "
                f"--no-retry-storms, --no-loops."
            )
        lines = [head, ""] + [f"  {a.line()}" for a in self.assertions]
        failed = [a for a in self.assertions if not a.ok]
        lines.append("")
        lines.append(
            "All assertions held."
            if not failed
            else f"{len(failed)} of {len(self.assertions)} assertions failed."
        )
        return "\n".join(lines)


def load(target: str) -> dict:
    """A trace document, from a JSON file or the local store.

    A path that exists wins over a trace id, because a file is unambiguous and
    an id that happens to look like a filename is not.
    """
    if not target:
        raise BadRequest("nothing to check: pass a trace id or a path to a .json")

    where = Path(target)
    if where.is_file():
        try:
            doc = json.loads(where.read_text("utf-8"))
        except (OSError, ValueError) as err:
            raise BadRequest(
                f"that file could not be read as a trace document: "
                f"{type(err).__name__}."
            ) from err
        if not isinstance(doc, dict) or not isinstance(doc.get("steps"), list):
            raise BadRequest(
                "a trace document is an object with a 'steps' list, and that "
                "file has neither."
            )
        return doc

    from . import paths, traces as traces_mod

    # The store the server actually opens. `TraceStore(None)` does not mean
    # "the default" — it raises inside pathlib.
    store = traces_mod.TraceStore(paths.trace_db_path())
    try:
        doc = store.get_trace(target)
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()
    if doc is None:
        raise BadRequest(
            f"no trace {target!r} in this machine's store, and no file at that "
            f"path either. `modelmri traces` lists what is recorded."
        )
    return doc


def run(
    doc: dict,
    *,
    no_errors: bool = False,
    max_steps: int | None = None,
    no_retry_storms: bool = False,
    no_loops: bool = False,
    max_repeat: int | None = None,
    max_ms: int | None = None,
) -> Result:
    """Apply the chosen assertions. Nothing is checked unless asked for."""
    steps = doc.get("steps") or []
    out = Result(
        trace_id=str(doc.get("id") or ""),
        name=str(doc.get("name") or ""),
        n_steps=len(steps),
    )

    if no_errors:
        bad = [s for s in steps if isinstance(s, dict) and s.get("error")]
        out.assertions.append(
            Assertion(
                name="no-errors",
                ok=not bad,
                detail=(
                    "no step recorded an error"
                    if not bad
                    else f"{len(bad)} step(s) recorded an error: "
                    + ", ".join(str(s.get("name") or s.get("id")) for s in bad[:5])
                    + (" …" if len(bad) > 5 else "")
                ),
                step_ids=[str(s.get("id") or "") for s in bad],
            )
        )

    if max_steps is not None:
        out.assertions.append(
            Assertion(
                name="max-steps",
                ok=len(steps) <= max_steps,
                detail=f"{len(steps)} steps, limit {max_steps}",
            )
        )

    if no_retry_storms or no_loops or max_repeat is not None:
        found = patterns_mod.analyse(steps)

        if no_retry_storms:
            storms = found.retry_storms
            out.assertions.append(
                Assertion(
                    name="no-retry-storms",
                    ok=not storms,
                    detail=(
                        "no name failed twice in a row inside "
                        f"{found.retry_window_ms} ms"
                        if not storms
                        else "; ".join(
                            f"{s.label} failed {s.count}x in a row" for s in storms[:3]
                        )
                    ),
                    step_ids=[sid for s in storms for sid in s.step_ids],
                )
            )

        if no_loops:
            cycles = found.cycles
            # A cap that was not scanned cannot assert "no loops". Passing
            # here would be the check reporting a clean bill of health from a
            # scan that never ran.
            if not found.cycles_scanned:
                out.assertions.append(
                    Assertion(
                        name="no-loops",
                        ok=False,
                        detail=(
                            f"NOT CHECKED — {found.n_steps} steps is over the "
                            f"{patterns_mod.MAX_STEPS_FOR_CYCLES} the cycle scan "
                            f"runs on, so this cannot say there are no loops. "
                            f"Failing rather than passing, because a green "
                            f"check from a scan that did not run is worse than "
                            f"a red one."
                        ),
                    )
                )
            else:
                out.assertions.append(
                    Assertion(
                        name="no-loops",
                        ok=not cycles,
                        detail=(
                            "no sequence repeated back to back"
                            if not cycles
                            else "; ".join(
                                f"{c.cycle_length} steps repeated {c.count}x "
                                f"({c.label})"
                                for c in cycles[:3]
                            )
                        ),
                        step_ids=[sid for c in cycles for sid in c.step_ids],
                    )
                )

        if max_repeat is not None:
            worst = found.repeats[0] if found.repeats else None
            count = worst.count if worst else 0
            out.assertions.append(
                Assertion(
                    name="max-repeat",
                    ok=count <= max_repeat,
                    detail=(
                        f"no step ran more than {max_repeat} times with the "
                        f"same input"
                        if count <= max_repeat
                        else f"{worst.label} ran {count} times with the same "
                        f"input, limit {max_repeat}"
                    ),
                    step_ids=list(worst.step_ids) if worst and count > max_repeat else [],
                )
            )

    if max_ms is not None:
        # OPT-IN and flaky by nature; the detail says so on every run, not
        # only in --help, because the person reading a red CI log is not the
        # person who added the flag.
        timed = [
            s.get("duration_ms")
            for s in steps
            if isinstance(s, dict) and isinstance(s.get("duration_ms"), int)
        ]
        total = sum(timed)
        untimed = len(steps) - len(timed)
        out.assertions.append(
            Assertion(
                name="max-ms",
                ok=total <= max_ms,
                detail=(
                    f"{total} ms across {len(timed)} timed step(s), limit "
                    f"{max_ms}"
                    + (
                        f" ({untimed} step(s) recorded no duration and are not "
                        f"in this total)"
                        if untimed
                        else ""
                    )
                    + ". WALL CLOCK — a shared runner is slow for reasons that "
                    "have nothing to do with your diff."
                ),
            )
        )

    return out


def check(target: str, **kwargs) -> tuple:
    """(Result, exit code, error). Never raises for an unreadable trace."""
    try:
        doc = load(target)
    except BadRequest as err:
        return None, NOTHING_CHECKED, str(err)
    result = run(doc, **kwargs)
    if not result.assertions:
        # Not 0. Nothing was verified, and a passing build that verified
        # nothing is worse than a failing one.
        return result, NOTHING_CHECKED, ""
    return result, (PASS if result.ok else FAILED), ""
