"""Score recorded runs against exact predicates. No model, so no judgement.

Every competitor's "evaluation" of an agent run asks a language model whether
it went well. That answer needs a calibration gate nobody ships, and it costs
a key and a round trip per run.

A predicate does not. "This trace has an error step", "this tool was called
with an argument matching this regex", "this run made more than 12 LLM calls"
— all exact, all answerable in SQL against a table that already indexes
`(trace_id, seq)`, all reproducible next Tuesday on a laptop with the network
off.

## Nothing here is a verdict

A predicate says a trace MATCHED. It does not say the trace was good, bad,
correct or wasteful. `patterns.py` holds the same line for structural findings
and this holds it for scored ones: the moment a rubric prints "failed" for
something it merely counted, it is a judgement with no judge behind it.

The user names the predicate. If they call it "too many retries", that phrase
is theirs and appears as theirs.

## Duration outliers over three traces are not statistics

This is the module's sharpest rule, and it is the roadmap's own caveat.

A "slowest 10%" over four runs is one run, and calling it an outlier is
arithmetic dressed as evidence. So every distribution-shaped predicate carries
a minimum n, REFUSES below it, and prints the n it did have. A threshold that
quietly reports nothing when the sample is too small is indistinguishable from
one that found nothing.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field

from .errors import BadRequest
from .step_kinds import VALID_KINDS

# Distribution predicates need at least this many traces before they will flag
# anything. Stated, refused against, and printed — never silent.
MIN_TRACES_FOR_OUTLIERS = 8

# A regex from a user is run against every stored payload. Bounded so a
# pathological pattern cannot take the server down on a big store.
MAX_PATTERN_CHARS = 500
MAX_RULES = 64

# What a rule may test. Each is exact; none is a judgement.
KINDS = (
    "has_error",  # any step recorded an error
    "kind_count",  # how many steps of one kind
    "step_count",  # how many steps in total
    "tool_input_matches",  # regex over the input of tool steps
    "any_input_matches",  # regex over the input of every step
    "output_matches",  # regex over the output of every step
    "duration_over",  # total wall clock above a stated number
    "slowest_percent",  # distribution — needs MIN_TRACES_FOR_OUTLIERS
)

OPERATORS = ("gt", "gte", "lt", "lte", "eq")

# The kinds that compare a measurement against `value`. Derived from the list
# above rather than written out twice: a new counting kind that forgot to
# appear here would go back to silently thresholding at zero.
NUMERIC_KINDS = tuple(k for k in KINDS if k.endswith(("_count", "_over", "_percent")))


class RubricError(BadRequest):
    """This rule cannot be evaluated honestly, and the message says why."""


def _compare(value, op: str, target) -> bool:
    if op == "gt":
        return value > target
    if op == "gte":
        return value >= target
    if op == "lt":
        return value < target
    if op == "lte":
        return value <= target
    return value == target


@dataclass
class Rule:
    """One exact predicate, named by the person who wrote it."""

    name: str
    kind: str
    pattern: str = ""
    step_kind: str = ""
    op: str = "gt"
    value: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def means(self) -> str:
        if self.kind == "has_error":
            return f"{self.name}: the run has at least one step marked as an error."
        if self.kind == "kind_count":
            return (
                f"{self.name}: the run has {self.op} {self.value:g} steps of "
                f"kind {self.step_kind}."
            )
        if self.kind == "step_count":
            return f"{self.name}: the run has {self.op} {self.value:g} steps."
        if self.kind == "tool_input_matches":
            return f"{self.name}: some tool call's input matches /{self.pattern}/."
        if self.kind == "any_input_matches":
            return f"{self.name}: some step's input matches /{self.pattern}/."
        if self.kind == "output_matches":
            return f"{self.name}: some step's output matches /{self.pattern}/."
        if self.kind == "duration_over":
            return (
                f"{self.name}: the run's recorded steps span more than "
                f"{self.value:g} ms of wall clock."
            )
        if self.kind == "slowest_percent":
            return (
                f"{self.name}: the run is in the slowest {self.value:g}% of "
                f"the set, measured over at least {MIN_TRACES_FOR_OUTLIERS} runs."
            )
        return self.name


def parse_rule(raw: dict) -> Rule:
    """One rule from an untrusted document, or a refusal that names the field."""
    if not isinstance(raw, dict):
        raise RubricError("a rule must be an object with a name and a kind.")
    name = str(raw.get("name") or "").strip()
    if not name:
        raise RubricError(
            "every rule needs a name. The name is YOURS — it is what appears "
            "beside a matching run, and nothing here invents one."
        )
    kind = str(raw.get("kind") or "")
    if kind not in KINDS:
        raise RubricError(
            f"{name!r} has kind {kind!r}, and this evaluates {', '.join(KINDS)}."
        )

    op = str(raw.get("op") or "gt")
    if op not in OPERATORS:
        raise RubricError(
            f"{name!r} uses operator {op!r}; use one of {', '.join(OPERATORS)}."
        )

    # Four kinds compare against `value`; the rest read `pattern` or nothing,
    # and must stay writable without one.
    value = raw.get("value", 0)
    if kind in NUMERIC_KINDS:
        # This used to coerce anything unusable to 0.0, and that is the one
        # failure this parser cannot afford. `{"kind": "duration_over", "op":
        # "gt", "value": "500"}` -- a string, the ordinary JSON slip -- became
        # `duration_over gt 0`, which every run on earth satisfies. The rubric
        # then reported a full-marks match against a threshold nobody wrote.
        # Every other malformed field here refuses and names itself; so does
        # this one now.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RubricError(
                f"{name!r} is a {kind} rule, so `value` has to be a number. "
                # `default=repr` because this parser is also called with plain
                # Python dicts, and a value json cannot serialise would raise
                # TypeError from inside the error path -- losing the refusal
                # that was being written.
                f"It was given {json.dumps(value, default=repr)}, and a "
                f"threshold this "
                f"cannot read is not one it will guess at."
            )
        if not math.isfinite(value):
            # NaN loses every comparison and infinity wins every one, in both
            # cases with nothing on screen saying the rule never really ran.
            raise RubricError(
                f"{name!r} has a `value` of {value}, which no measurement can "
                f"be above or below in any useful sense."
            )
    elif isinstance(value, bool) or not isinstance(value, (int, float)):
        # Unused by this kind: carried as 0 rather than refused, so a rule
        # that legitimately has no threshold is not asked to invent one.
        value = 0.0

    pattern = str(raw.get("pattern") or "")
    if kind.endswith("_matches"):
        if not pattern:
            raise RubricError(f"{name!r} matches a pattern and none was given.")
        if len(pattern) > MAX_PATTERN_CHARS:
            raise RubricError(
                f"{name!r}'s pattern is {len(pattern)} characters and the cap "
                f"is {MAX_PATTERN_CHARS}."
            )
        try:
            re.compile(pattern)
        except re.error as err:
            # The POSITION, which is an int, not `str(err)`. Interpolating a
            # caught exception's own text is the thing `test_no_exception_leaks`
            # forbids, and the exemption is not worth taking for a message this
            # can author itself. `from err` keeps the engine's wording on the
            # traceback for whoever is debugging, which is where it belongs.
            where = getattr(err, "pos", None)
            at = f" at character {where}" if isinstance(where, int) else ""
            raise RubricError(
                f"{name!r}'s pattern is not a valid regular expression{at}. "
                f"Check the brackets, parentheses and escapes there."
            ) from err

    step_kind = str(raw.get("step_kind") or "")
    if kind == "kind_count":
        if step_kind not in VALID_KINDS:
            raise RubricError(
                f"{name!r} counts steps of kind {step_kind!r}, and a step may "
                f"be one of {', '.join(sorted(VALID_KINDS))}."
            )
    if kind == "slowest_percent" and not 0 < value < 100:
        raise RubricError(
            f"{name!r} asks for the slowest {value:g}%, which has to be "
            f"between 0 and 100."
        )

    return Rule(
        name=name,
        kind=kind,
        pattern=pattern,
        step_kind=step_kind,
        op=op,
        value=float(value),
    )


def parse(raw) -> list:
    """A whole rubric from an untrusted document."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError as err:
            raise RubricError("that rubric is not readable JSON.") from err
    if isinstance(raw, dict):
        # A DICT WITHOUT `rules` IS A MISTAKE, not an empty rubric. This read
        # `raw.get("rules", [])`, so any object lacking the key became `[]` —
        # which is a legal rubric that matches nothing and reports 200.
        #
        # Paired with the route's old `parse(body.get("rules", body))`, one
        # transposed letter did the whole thing silently: `{"rulez": [...]}`
        # answered "0 rule(s) against 111 recorded run(s). No run matched any
        # rule." while the correct spelling found 66 matching runs.
        #
        # An empty rubric is still expressible, and still answered rather than
        # refused — as `[]`, or as `{"rules": []}`. What is refused is a
        # document that never mentioned rules at all.
        if "rules" not in raw:
            raise RubricError(
                "a rubric is a list of rules, or an object with a `rules` "
                "key — this object has neither, so there is nothing to score "
                'against. An empty rubric is `{"rules": []}`.'
            )
        raw = raw["rules"]
    if not isinstance(raw, list):
        raise RubricError("a rubric is a list of rules.")
    if len(raw) > MAX_RULES:
        raise RubricError(
            f"that rubric has {len(raw)} rules and the cap is {MAX_RULES}. "
            f"Cut it rather than having it cut for you, so you know which "
            f"rules the answer is about."
        )
    rules = [parse_rule(r) for r in raw]

    # NAMES MUST BE UNIQUE, because the name is the key. `slow_cut`,
    # `report.skipped` and `counts()` are all dicts keyed by `rule.name`, so
    # two rules sharing one collapse into a single entry — measured, a rubric
    # with two rules called "same" reported `counts() == {"same": 1}` for one
    # rule that matched and one that did not, and a skipped distribution rule
    # would suppress an unrelated rule of the same name entirely.
    seen: dict = {}
    for rule in rules:
        if rule.name in seen:
            raise RubricError(
                f"two rules are both called {rule.name!r}. The name is what "
                f"appears beside a matching run and what the counts are keyed "
                f"by, so two rules sharing one would report as a single "
                f"result. Rename one."
            )
        seen[rule.name] = rule
    return rules


@dataclass
class Hit:
    """One rule against one run."""

    rule: str
    matched: bool
    detail: str = ""
    # Steps that made it match, so a result points at something.
    step_ids: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Scored:
    """One run, every rule's answer for it, and enough to tell it apart.

    The name alone does not identify a run. A hundred rows reading
    "generation", "generation", "generation" are a hundred rows a reader cannot
    order, date, or choose between — the list becomes a wall rather than a
    result, and the rule that matched is attached to something anonymous.

    So the summary's own identifying fields ride along. Not derived here and
    not recomputed: `traces.list_traces` already answers all of them, and a
    second implementation of "how long did this take" is a second number to
    disagree with the first.

    `source` and `demo` matter most of the three. The store draws those
    distinctions deliberately — a playground generation is not a run of your
    own agent code, and scripted sample data is not either — and a rubric row
    that dropped them would put all three kinds in one undifferentiated list,
    which is exactly the distinction the store keeps them for.
    """

    trace_id: str
    name: str = ""
    hits: list = field(default_factory=list)
    # ISO 8601 from the store, passed through verbatim. Formatting is the
    # reader's timezone's business and this has no idea what that is.
    started_at: str = ""
    # `None`, not 0, when the store has no duration for this run. A run whose
    # length was never recorded did not take no time.
    total_ms: int | None = None
    n_steps: int = 0
    n_errors: int = 0
    # "app" for a generation made in the playground, "" for a trace written
    # before the key existed. Never guessed from the name.
    source: str = ""
    demo: bool = False

    @property
    def matched(self) -> list:
        return [h.rule for h in self.hits if h.matched]

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "hits": [h.to_dict() for h in self.hits],
            "matched": self.matched,
            "started_at": self.started_at,
            "total_ms": self.total_ms,
            "n_steps": self.n_steps,
            "n_errors": self.n_errors,
            "source": self.source,
            "demo": self.demo,
        }


@dataclass
class Report:
    """A whole rubric over a whole store."""

    rows: list = field(default_factory=list)
    n_traces: int = 0
    rules: list = field(default_factory=list)
    # Rules that could not be answered, and why. Distribution predicates land
    # here below the minimum n rather than reporting "no matches".
    skipped: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "rows": [r.to_dict() for r in self.rows],
            "n_traces": self.n_traces,
            "rules": [r.to_dict() | {"means": r.means()} for r in self.rules],
            "skipped": dict(self.skipped),
            "counts": self.counts(),
            "means": self.means(),
        }

    def counts(self) -> dict:
        out: dict = {}
        for row in self.rows:
            for hit in row.hits:
                if hit.matched:
                    out[hit.rule] = out.get(hit.rule, 0) + 1
        return out

    def means(self) -> str:
        parts = [
            f"{len(self.rules)} rule(s) against {self.n_traces} recorded run(s). "
            f"Every rule here is exact — no model was asked, and a match is a "
            f"match rather than a verdict."
        ]
        counts = self.counts()
        if counts:
            named = ", ".join(
                f"{name} on {n} run(s)" for name, n in sorted(counts.items())
            )
            parts.append(f"Matched: {named}.")
        else:
            parts.append("No run matched any rule.")
        if self.skipped:
            named = "; ".join(f"{k} — {v}" for k, v in sorted(self.skipped.items()))
            parts.append(f"NOT EVALUATED: {named}")
        return " ".join(parts)


def _total_ms(steps) -> int | None:
    """Wall clock the recorded steps span, or `None` when nobody timed it.

    Steps with no duration contribute their start but not a length, because
    `None` means nobody wrote one down — treating it as 0 would shorten a run
    to make it look faster than it was measured to be.

    AND THAT IS EXACTLY WHAT IT DID when NO step carried one. The span from
    first start to last end collapses to 0, and 0 went to `_compare` as a
    measurement: a run nobody timed matched "under 500 ms" with the detail "0
    ms of recorded wall clock", and sorted to the fast end of every
    `slowest_percent` cut. An unknown duration is not a fast one.

    `None` is that answer. A run with SOME durations still returns a span,
    which is a floor and the honest best available — the rules that read it
    are about the whole run, not about a per-step total.

    Floats count. `isinstance(length, int)` dropped a step recorded as
    `1500.0`, which is what `(t1 - t0) * 1000` produces in any recorder that
    does not round; `check.py` had the same defect in its own gate.
    """
    if not steps:
        return None
    starts, ends = [], []
    timed = 0
    for step in steps:
        start = step.get("started_ms")
        if not isinstance(start, int) or isinstance(start, bool):
            continue
        length = step.get("duration_ms")
        usable = (
            float(length)
            if isinstance(length, (int, float))
            and not isinstance(length, bool)
            and math.isfinite(length)
            else None
        )
        if usable is not None:
            timed += 1
        starts.append(start)
        ends.append(start + (usable or 0))
    if not ends or not timed:
        return None
    # LAST END MINUS FIRST START, not just the last end. An imported trace
    # whose offsets begin at a wall-clock epoch rather than 0 would otherwise
    # report a span of about 1.7 trillion ms and win every duration rule.
    return int(round(max(ends) - min(starts)))


def score(traces_and_steps, rules) -> Report:
    """Apply every rule to every run.

    `traces_and_steps` is an iterable of (summary, steps). The caller does the
    fetching, so this module never opens a database and can be tested without
    one.
    """
    runs = [
        (dict(summary), list(steps))
        for summary, steps in traces_and_steps
        if isinstance(summary, dict)
    ]
    report = Report(n_traces=len(runs), rules=list(rules))

    # Distribution predicates need the whole set before any single run can be
    # answered, so their inputs are gathered first.
    durations = [_total_ms(steps) for _, steps in runs]
    slow_cut: dict = {}
    for rule in rules:
        if rule.kind != "slowest_percent":
            continue
        if len(runs) < MIN_TRACES_FOR_OUTLIERS:
            report.skipped[rule.name] = (
                f"{len(runs)} run(s) recorded and this needs at least "
                f"{MIN_TRACES_FOR_OUTLIERS}. The slowest 10% of four runs is "
                f"one run, and calling that an outlier is arithmetic dressed "
                f"as evidence."
            )
            continue
        # Untimed runs are not in the distribution. Including them as 0
        # dragged the cut down and put every one of them at the fast end of a
        # ranking they were never measured for.
        ordered = sorted(d for d in durations if d is not None)
        if not ordered:
            report.skipped[rule.name] = (
                "no run in this set recorded a duration, so there is no "
                "distribution to take a percentile of."
            )
            continue
        # The cut is the (100 - value)th percentile: "slowest 10%" means above
        # the 90th.
        index = min(
            len(ordered) - 1,
            max(0, int(round((1 - rule.value / 100.0) * (len(ordered) - 1)))),
        )
        slow_cut[rule.name] = ordered[index]

    for (summary, steps), total in zip(runs, durations, strict=True):
        # `total_ms` comes from the summary rather than from `total`: the
        # latter is this scorer's own duration, already clamped to >= 0 for
        # the percentile maths, and a row that showed 0 for a run the store
        # has no duration for would be reporting a measurement it does not
        # have. `is None` rather than `or None` — 0 is a real duration for a
        # run that finished inside a millisecond.
        recorded = summary.get("total_ms")
        row = Scored(
            trace_id=str(summary.get("id") or ""),
            name=str(summary.get("name") or ""),
            started_at=str(summary.get("started_at") or ""),
            total_ms=None if recorded is None else int(recorded),
            n_steps=int(summary.get("n_steps") or 0),
            n_errors=int(summary.get("n_errors") or 0),
            source=str(summary.get("source") or ""),
            demo=bool(summary.get("demo")),
        )
        for rule in rules:
            if rule.name in report.skipped:
                continue
            row.hits.append(_apply(rule, steps, total, slow_cut))
        report.rows.append(row)
    return report


def _apply(rule: Rule, steps, total_ms: int, slow_cut: dict) -> Hit:
    if rule.kind == "has_error":
        bad = [s for s in steps if s.get("error")]
        return Hit(
            rule=rule.name,
            matched=bool(bad),
            detail=f"{len(bad)} step(s) recorded an error",
            step_ids=[str(s.get("id") or "") for s in bad],
        )

    if rule.kind == "step_count":
        return Hit(
            rule=rule.name,
            matched=_compare(len(steps), rule.op, rule.value),
            detail=f"{len(steps)} steps",
        )

    if rule.kind == "kind_count":
        of = [s for s in steps if s.get("kind") == rule.step_kind]
        return Hit(
            rule=rule.name,
            matched=_compare(len(of), rule.op, rule.value),
            detail=f"{len(of)} {rule.step_kind} step(s)",
            step_ids=[str(s.get("id") or "") for s in of],
        )

    if rule.kind.endswith("_matches"):
        pattern = re.compile(rule.pattern)
        field_name = "output" if rule.kind == "output_matches" else "input"
        wanted = steps
        if rule.kind == "tool_input_matches":
            wanted = [s for s in steps if s.get("kind") == "tool_call"]
        hit = [
            s
            for s in wanted
            if isinstance(s.get(field_name), str) and pattern.search(s[field_name])
        ]
        return Hit(
            rule=rule.name,
            matched=bool(hit),
            detail=f"{len(hit)} step(s) matched",
            step_ids=[str(s.get("id") or "") for s in hit],
        )

    if rule.kind == "duration_over":
        # `_compare`, not a hardcoded `>`. `parse_rule` VALIDATES `op` against
        # OPERATORS, so a rubric written with `op: "lt"` parses without
        # complaint and then silently ran as `>` — measured, a 100 ms run
        # against `lt 500` did not match. Validating a field and then ignoring
        # it is worse than not offering it.
        if total_ms is None:
            return Hit(
                rule=rule.name,
                matched=False,
                detail=(
                    "no step in this run recorded a duration, so its wall "
                    "clock is unknown rather than zero"
                ),
            )
        return Hit(
            rule=rule.name,
            matched=_compare(total_ms, rule.op, rule.value),
            detail=f"{total_ms} ms of recorded wall clock",
        )

    if rule.kind == "slowest_percent":
        cut = slow_cut.get(rule.name)
        if cut is None:  # pragma: no cover - skipped rules never reach here
            return Hit(rule=rule.name, matched=False, detail="not evaluated")
        if total_ms is None:
            return Hit(
                rule=rule.name,
                matched=False,
                detail=(
                    "no step in this run recorded a duration, so it is not in "
                    "the distribution this cut was taken from"
                ),
            )
        return Hit(
            rule=rule.name,
            matched=total_ms >= cut,
            detail=f"{total_ms} ms against a {cut} ms cut",
        )

    return Hit(rule=rule.name, matched=False, detail="unknown rule kind")
