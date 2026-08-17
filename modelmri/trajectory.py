"""Where a run went, beside where it was supposed to go — with no judge.

`patterns.py` counts loops and retry storms; `rubric.py` counts kinds and
regexes tool arguments. Neither can answer the question people actually ask
about an agent: "it was supposed to search, then read, then write, and this
run skipped read". That is the most common agent regression there is, and it
is invisible to both of them — the loop counter sees no loop and every
predicate the rubric can express still passes.

Every competitor scores it with a language model, which costs a key, a round
trip and a different answer next Tuesday. It does not need one. Two sequences
of `(kind, name)` pairs have an edit distance, a longest common subsequence is
exact, and the whole comparison is arithmetic over strings: offline,
reproducible, and fast enough that nobody has to decide whether to run it.
Measured on this machine, a 500-step plan against a 500-step run — the largest
comparison this will do — aligned in 24 ms.

## The alignment is the answer. There is no score.

This is the rule the module exists to hold and the easy one to lose, because a
single number is what every dashboard wants and a ratio is one division away.

"2 steps missing, 1 extra, 3 with changed arguments" is a description of two
sequences. "Plan Adherence 0.71" is a claim that the run was 29% wrong, and
nothing here can support it: A SHORTER PATH IS NOT A WORSE PATH. An agent that
reached the answer in three steps where the plan named five either found a
shortcut or missed the point, and a sequence alignment cannot tell those apart
because it does not know what any step was FOR. So the counts ship, the ratio
never does, and that sentence travels in `means()` rather than in this
docstring, because the person who needs it is looking at the panel.

`patterns.py` holds the same line for structural findings and states the
reason in the same words: the moment a count is worded as a verdict it becomes
the model judgement this exists to replace, except without the model.

## Reordered is a third answer, not two wrong ones

An edit distance alone reports a moved step as one deletion plus one
insertion — "read is missing, and there is an unexpected read" — which is two
findings about one event and reads as worse than what happened. So the
leftovers on each side are paired by identity afterwards, in order, and a pair
becomes `reordered`.

It says the step happened somewhere else, NOT that the order was wrong. A plan
is frequently a set of things to do rather than a sequence, and this module
cannot tell which kind it was handed.

## Repeats are where a naive alignment lies

Match by set membership and `search, search, search` against `search, search`
reports everything matched. Match by first occurrence and the third `search`
pairs with the second. A longest common subsequence gets it right — two
matched, one missing — which is the whole reason for the DP table.

What it cannot get right is WHICH of the three is the missing one, because
there are three alignments of exactly the same length. `matched` is the same
number in every one of them (the length of a longest common subsequence is
unique even when the subsequence is not); the row it is attributed to is not.
So repeated identities are reported as a fact, rather than the reader being
left to assume the named occurrence is the one that went wrong.

## Arguments are compared before they are redacted, and never the other way

Argument values reach a panel, an `.mri` and eventually a GitHub issue, so
everything echoed here — argument values and the step names printed beside
them — goes through the recorder's credential patterns, taken from
`modelmri.record.redact` rather than copied, the way `bundle.py` takes them.
The in-tree copy of that recorder drifted once and what it had lost was
precisely the redaction.

The ORDER is load-bearing and is the defect worth naming: redact first and two
different API keys both become `[redacted:api-key]`, compare equal, and the
step reports "arguments unchanged". A false all-clear on the one field most
likely to hold the thing that broke. So equality is decided on the raw value
and only the display is scrubbed — same rule for the truncation cap, because
two payloads that differ after the first 500 characters are not one payload.

## Bounded, and honest about which bound did what

The DP table is `len(reference) x len(candidate)` cells, so a 10,000-step run
against a 10,000-step plan is 100 million of them. Three bounds, and they
behave differently on purpose:

  the table   REFUSED past `MAX_ALIGN_CELLS`, never trimmed. Comparing a
              prefix would report every reference step past the cut as
              `missing`, which is precisely the wrong conclusion this module
              exists to produce correctly. `plan_comparison()` prices it first
              so a caller need not discover the refusal by hitting it.
  each side   REFUSED past `MAX_STEPS_PER_SIDE`, because the cell budget alone
              does not stop a one-step plan against a quarter-million-step
              run — that product fits, and the row objects do not.
  the listing CAPPED and reported. The counts are computed over the whole
              alignment before any row is dropped, and matched rows are
              dropped before differing ones, so a capped listing still holds
              every difference and still counts what it is not showing.

Nothing here does I/O or holds a model, which is what lets the arithmetic be
tested without a GPU, a network or an agent.
"""

from __future__ import annotations

import json
from array import array
from dataclasses import dataclass, field

from .errors import BadRequest
from .step_kinds import VALID_KINDS

# What one row of an alignment can be. `reordered` is derived after the edit
# distance rather than by it -- see the module docstring.
MATCHED = "matched"
MISSING = "missing"
EXTRA = "extra"
REORDERED = "reordered"

# What one argument difference can be. `only_in_reference` is not "removed"
# and `only_in_candidate` is not "added": a plan that names three arguments
# and a call that passes five is the ordinary case, and calling the other two
# additions would be a judgement about a plan's completeness.
CHANGED = "changed"
ONLY_IN_REFERENCE = "only_in_reference"
ONLY_IN_CANDIDATE = "only_in_candidate"

# The DP table is one cell per (reference step, candidate step) pair. 250,000
# is 500 x 500, measured at 24 ms on this machine with the table held as
# `array("i")` rows; the same table as Python lists is about 10x the memory
# for the same answer. Past this the comparison is REFUSED rather than run on
# a prefix, because a prefix reports the rest of the plan as missing.
MAX_ALIGN_CELLS = 250_000

# And a bound on each side alone, because the cell budget does not constrain a
# 1-step plan against a 250,000-step run: that fits, and would build a quarter
# of a million Row objects to hold an answer nobody can read. 5,000 is the
# order `bundle.py` already refuses a trace section at, and for the same
# reason — past it, what you want is a subtree rather than a transcript.
MAX_STEPS_PER_SIDE = 5_000

# The most rows a single alignment will list. The COUNTS are computed before
# this applies, so a capped listing never changes the answer -- it changes how
# much of it is enumerated, and says how much it left out.
MAX_ROWS_LISTED = 2_000

# Per matched pair: how many argument keys are compared, and how much of a
# value is shown. Both cut, both counted, neither silent. The comparison
# itself always runs on the FULL value.
MAX_ARG_KEYS = 40
MAX_ARG_CHARS = 500

# How many repeated identities are named. Bounded because a 1-step plan
# against a 250,000-step run is inside the cell budget and could otherwise
# produce a list longer than the alignment.
MAX_REPEATS_LISTED = 50


class TrajectoryError(BadRequest):
    """This comparison cannot be drawn honestly, and the message says why.

    A `BadRequest` rather than a `Refusal` for the same reason `BundleError`
    is: every message here names something in the CALL to change -- a plan
    with no steps, a step with no name, two sequences too long to align --
    rather than a state the caller has to go and fix elsewhere.
    """


def _seq(step) -> int:
    """A step's recorded position, or 0 when it did not record one.

    `isinstance(True, int)` is True, so a hand-written or ingested document
    carrying `"seq": true` would otherwise sort as position 1 and silently
    move a step. Absent and unreadable both mean "this document does not order
    itself", and the stable sort below then leaves list order alone.
    """
    if not isinstance(step, dict):
        return 0
    value = step.get("seq")
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


def _ordered(steps) -> list:
    """Steps in recorded order, which is `seq` where a document supplies one.

    A written plan has no `seq` and must keep the order it was written in; a
    trace from the store arrives ordered already; a hand-written document need
    not. `sorted` is stable, so all-zero keys preserve list order and the
    three cases need no flag to tell them apart.
    """
    return sorted([s for s in steps if s is not None], key=_seq)


@dataclass
class _Step:
    """One step of either side, normalised to what an alignment needs."""

    kind: str
    name: str
    step_id: str = ""
    # `None` means nothing here could read arguments for this step, which is a
    # different answer from `{}` -- "this step took no arguments". Collapsing
    # the first into the second makes two unreadable steps compare equal and
    # report "arguments identical", which is a false all-clear.
    arguments: dict | None = None
    # Where the arguments came from, so a row can say what it compared.
    arguments_source: str = ""
    # The name as it is SHOWN, which is the name after the recorder's
    # credential patterns have run over it. Alignment uses `name`, never this
    # one -- see `_diff_arguments` for why the order matters.
    display_name: str = ""


def _label(kind: str, name: str) -> str:
    """How a step is named to a reader. Never an empty string."""
    return f"{kind} {name}".strip() or "(an unnamed step)"


def _arguments_of(raw: dict) -> tuple[dict | None, str]:
    """A step's arguments and where they were read from.

    Four sources, in the order a real document answers them. `None` rather
    than `{}` when none of them can, because "no arguments" and "no arguments
    this could read" are different facts and only one of them makes an
    argument diff meaningful.
    """
    for field_name in ("args", "arguments"):
        value = raw.get(field_name)
        if isinstance(value, dict):
            return dict(value), field_name

    payload = raw.get("input")
    if isinstance(payload, dict):
        return dict(payload), "input object"
    if isinstance(payload, str) and payload:
        try:
            parsed = json.loads(payload)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed, "input json"
        # Free text. Compared as ONE argument under the recorder's own field
        # name rather than being dropped: a changed prompt is a real
        # difference, and the source travels so a reader knows the diff is a
        # whole payload rather than a named parameter.
        return {"input": payload}, "input text"
    return None, ""


def _normalise(raw, *, where: str) -> _Step:
    """One step of a plan or a run, from whatever shape the caller has.

    A bare string is a name with NO kind, and the kind stays empty rather than
    defaulting to `tool_call`. A default would be a guess that silently
    changes the answer: every plan step would fail to match the same call
    recorded as an `mcp_call`, and the run would read as having skipped all of
    them. `align` sees the empty kind and drops kind from the key for the
    whole comparison instead, which it then says out loud.
    """
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            raise TrajectoryError(
                f"a step of {where} is an empty string. A step with no name "
                f"cannot be aligned against anything, so this refuses rather "
                f"than matching it to whatever happens to sit opposite it."
            )
        kind, sep, name = text.partition(":")
        if sep:
            # It reads as `kind:name`. If the prefix is not a kind a recorder
            # writes, the string is probably a tool whose own name contains a
            # colon -- but "probably" is not a basis for choosing between two
            # readings that align differently, so it refuses and says how to
            # write the one it cannot guess.
            if kind in VALID_KINDS:
                return _Step(kind=kind, name=name.strip() or kind)
            raise TrajectoryError(
                f"{text!r} in {where} reads as `kind:name`, and {kind!r} is "
                f"not a step kind — a step is one of "
                f"{', '.join(sorted(VALID_KINDS))}. If that colon belongs to "
                f'the tool\'s name, write the step as {{"name": {text!r}}} so '
                f"nothing here has to guess which half is which."
            )
        return _Step(kind="", name=text)

    if not isinstance(raw, dict):
        raise TrajectoryError(
            f"a step of {where} is neither an object nor a name. Give each "
            f"step as a string like 'search', or as an object with a 'name' "
            f"and optionally a 'kind'."
        )

    name = str(raw.get("name") or "").strip()
    if not name:
        raise TrajectoryError(
            f"a step of {where} has no name. The name is what an alignment "
            f"matches on, so a nameless step would match every other nameless "
            f"step and nothing else."
        )
    kind = str(raw.get("kind") or "").strip()
    if kind and kind not in VALID_KINDS:
        raise TrajectoryError(
            f"{name!r} in {where} has kind {kind!r}, and a recorded step is "
            f"one of {', '.join(sorted(VALID_KINDS))}. A kind nothing writes "
            f"matches nothing, so every step naming it would report as "
            f"missing — which looks like a finding about the run."
        )
    arguments, source = _arguments_of(raw)
    return _Step(
        kind=kind,
        name=name,
        step_id=str(raw.get("id") or ""),
        arguments=arguments,
        arguments_source=source,
    )


def _redact(text: str, tally: dict) -> tuple[str, bool]:
    """Scrub credential shapes out of a value, counting what fired.

    The patterns come from `modelmri.record.redact` — the recorder's, not a
    copy — for the reason `bundle.py` gives: an in-tree copy of that module
    once drifted and what it had lost was precisely the credential redaction.

    Applied SEQUENTIALLY and counted as they fire, again as `bundle.py` does:
    `bearer` and `api-key` overlap on an `Authorization:` header, and counting
    each against the untouched text would report two secrets where there was
    one.
    """
    from .record import redact

    fired = False
    for name, pattern in redact._PATTERNS:
        text, n = pattern.subn(f"[redacted:{name}]", text)
        if n:
            tally[name] = tally.get(name, 0) + n
            fired = True
    return text, fired


def _render(value) -> str:
    """An argument value as one comparable string.

    `sort_keys=True` so that a nested object written in a different key order
    is not reported as a changed argument — JSON has no order and a diff that
    thinks it does produces a difference nobody made. `default=repr` because
    this is also called with plain Python objects a caller passed in, and a
    TypeError from inside the renderer would lose the whole comparison.
    """
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True, default=repr)
    except (TypeError, ValueError):  # pragma: no cover - default=repr covers it
        return repr(value)


@dataclass
class ArgDiff:
    """One argument that differs at a position where both sides have a step."""

    key: str
    status: str
    # `None` means the side did not carry this key at all, which is different
    # from carrying it as an empty string.
    reference: str | None = None
    candidate: str | None = None
    # Whether the DISPLAYED values were scrubbed. When true, two values that
    # look identical here were still compared as different — the comparison
    # ran before the scrubbing did.
    redacted: bool = False
    truncated: bool = False

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "status": self.status,
            "reference": self.reference,
            "candidate": self.candidate,
            "redacted": self.redacted,
            "truncated": self.truncated,
        }


def _show(text: str, tally: dict, counters: dict) -> tuple[str, bool, bool]:
    """A raw value prepared for display: truncated, then redacted.

    Truncation before redaction rather than after, because a value cut at
    `MAX_ARG_CHARS` may end mid-token and the recorder's own patterns already
    carry the rule for a credential whose tail was cut off — that headless
    PEM block was found in the wild against 0.1.0.
    """
    truncated = len(text) > MAX_ARG_CHARS
    if truncated:
        cut = len(text) - MAX_ARG_CHARS
        counters["truncated"] = counters.get("truncated", 0) + 1
        text = text[:MAX_ARG_CHARS] + f"… [{cut:,} characters not shown]"
    shown, redacted = _redact(text, tally)
    return shown, redacted, truncated


def _diff_arguments(
    ref: _Step, cand: _Step, tally: dict, counters: dict
) -> tuple[list, str]:
    """Argument differences at a position where both sides have a step.

    Returns `(diffs, note)`. The note is why nothing was compared, and is
    empty exactly when the comparison ran — because "no differences" and "not
    compared" are the two answers a reader must never confuse, and an empty
    list is what both of them look like.
    """
    if ref.arguments is None and cand.arguments is None:
        return [], (
            "neither side records arguments for this step, so nothing here "
            "compared them. That is not agreement."
        )
    if ref.arguments is None:
        return [], (
            f"the plan names no arguments for this step, so the "
            f"{len(cand.arguments)} it was called with are not reported as "
            f"additions. A plan that names a step and not its arguments is "
            f"the ordinary case, not an omission."
        )
    if cand.arguments is None:
        return [], (
            "the run records no arguments for this step, so the plan's are "
            "not reported as removals. Nothing here can tell an argument that "
            "was dropped from one that was never written down."
        )

    keys = sorted(set(ref.arguments) | set(cand.arguments))
    if len(keys) > MAX_ARG_KEYS:
        counters["keys_dropped"] = counters.get("keys_dropped", 0) + (
            len(keys) - MAX_ARG_KEYS
        )
        keys = keys[:MAX_ARG_KEYS]

    diffs = []
    for arg in keys:
        in_ref = arg in ref.arguments
        in_cand = arg in cand.arguments
        # RAW values decide equality. Redacting first would make two different
        # API keys both `[redacted:api-key]`, compare equal, and report the
        # step as unchanged -- a false all-clear on the field most likely to
        # be the thing that broke.
        raw_ref = _render(ref.arguments[arg]) if in_ref else None
        raw_cand = _render(cand.arguments[arg]) if in_cand else None
        if in_ref and in_cand and raw_ref == raw_cand:
            continue

        shown_ref, red_a, cut_a = (
            _show(raw_ref, tally, counters)
            if raw_ref is not None
            else (None, False, False)
        )
        shown_cand, red_b, cut_b = (
            _show(raw_cand, tally, counters)
            if raw_cand is not None
            else (None, False, False)
        )
        diffs.append(
            ArgDiff(
                key=arg,
                status=(
                    CHANGED
                    if in_ref and in_cand
                    else (ONLY_IN_REFERENCE if in_ref else ONLY_IN_CANDIDATE)
                ),
                reference=shown_ref,
                candidate=shown_cand,
                redacted=bool(red_a or red_b),
                truncated=bool(cut_a or cut_b),
            )
        )
    return diffs, ""


@dataclass
class Row:
    """One position of the alignment, in the order the run happened."""

    status: str
    kind: str
    name: str
    # `None` on the side that has no step here. Never -1 and never 0: position
    # 0 is a real position.
    reference_index: int | None = None
    candidate_index: int | None = None
    step_id: str = ""
    argument_diffs: list = field(default_factory=list)
    # False when the two sides could not be compared on arguments at all,
    # with `arguments_note` saying why. An empty diff list under
    # `compared=False` is not agreement.
    arguments_compared: bool = False
    arguments_note: str = ""

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "kind": self.kind,
            "name": self.name,
            "reference_index": self.reference_index,
            "candidate_index": self.candidate_index,
            "step_id": self.step_id,
            "argument_diffs": [d.to_dict() for d in self.argument_diffs],
            "arguments_compared": self.arguments_compared,
            "arguments_note": self.arguments_note,
            "means": self.means(),
        }

    def means(self) -> str:
        """What this position is. Never whether it should have been."""
        who = _label(self.kind, self.name)
        if self.status == MISSING:
            return (
                f"the plan names {who} at position {self.reference_index} and "
                f"no step in the run aligned with it. Whether that is a "
                f"skipped step or a shorter path is not something a sequence "
                f"alignment can tell you."
            )
        if self.status == EXTRA:
            return (
                f"{who} ran at position {self.candidate_index} of the run and "
                f"the plan does not name it. Unplanned is not the same as "
                f"wrong — a plan is rarely a complete transcript."
            )
        if self.status == REORDERED:
            return (
                f"{who} ran at position {self.candidate_index} of the run "
                f"where the plan has it at {self.reference_index}. It "
                f"happened somewhere else, which is not the same claim as the "
                f"order being wrong."
            )
        changed = len(self.argument_diffs)
        if not self.arguments_compared:
            return f"{who} ran where the plan names it; {self.arguments_note}"
        if not changed:
            return (
                f"{who} ran where the plan names it, with every compared "
                f"argument identical."
            )
        named = ", ".join(d.key for d in self.argument_diffs[:4])
        rest = f" and {changed - 4} more" if changed > 4 else ""
        return (
            f"{who} ran where the plan names it, with {changed} argument(s) "
            f"different: {named}{rest}."
        )


@dataclass
class Alignment:
    """Two trajectories, positioned against each other. Not scored."""

    rows: list = field(default_factory=list)
    # "kind and name" or "name alone" — derived from whether every step on
    # both sides states a kind, and reported because it changes the answer.
    matched_on: str = "kind and name"
    n_reference: int = 0
    n_candidate: int = 0
    # Counted over the WHOLE alignment, before `rows` is capped. Computing
    # these from a trimmed list is the silent-truncation defect this module
    # spends a constant avoiding.
    n_matched: int = 0
    n_missing: int = 0
    n_extra: int = 0
    n_reordered: int = 0
    n_changed_arguments: int = 0
    # Paired positions where arguments could not be compared at all. Reported
    # beside the count above, because "0 changed arguments" over 5 pairs that
    # were never compared is not the same finding as over 5 that were.
    n_arguments_not_compared: int = 0
    cells: int = 0
    rows_not_listed: int = 0
    matched_rows_not_listed: int = 0
    arg_keys_dropped: int = 0
    arg_values_truncated: int = 0
    # [{"label": "api-key", "count": 2}] — what the recorder's patterns
    # replaced in the values echoed here.
    redactions: list = field(default_factory=list)
    # [{"identity": "tool_call search", "in_reference": 3, "in_candidate": 2}]
    repeated_identities: list = field(default_factory=list)
    repeats_not_listed: int = 0

    def to_dict(self) -> dict:
        return {
            "rows": [r.to_dict() for r in self.rows],
            "matched_on": self.matched_on,
            "n_reference": self.n_reference,
            "n_candidate": self.n_candidate,
            "n_matched": self.n_matched,
            "n_missing": self.n_missing,
            "n_extra": self.n_extra,
            "n_reordered": self.n_reordered,
            "n_changed_arguments": self.n_changed_arguments,
            "n_arguments_not_compared": self.n_arguments_not_compared,
            "cells": self.cells,
            "rows_not_listed": self.rows_not_listed,
            "matched_rows_not_listed": self.matched_rows_not_listed,
            "arg_keys_dropped": self.arg_keys_dropped,
            "arg_values_truncated": self.arg_values_truncated,
            "redactions": [dict(r) for r in self.redactions],
            "repeated_identities": [dict(r) for r in self.repeated_identities],
            "repeats_not_listed": self.repeats_not_listed,
            "means": self.means(),
        }

    def means(self) -> str:
        parts = [
            f"{self.n_reference} planned step(s) against {self.n_candidate} "
            f"recorded step(s), aligned on {self.matched_on}. "
            f"{self.n_matched} matched in place, {self.n_missing} missing, "
            f"{self.n_extra} extra, {self.n_reordered} in a different place, "
            f"and {self.n_changed_arguments} step(s) present on both sides ran "
            f"with different arguments."
        ]
        parts.append(
            "THIS IS AN ALIGNMENT, NOT A SCORE. There is no adherence number "
            "here and there will not be one: A SHORTER PATH IS NOT A WORSE "
            "PATH, and a single figure says that it is. An agent that skipped "
            "a step the plan names may have found a shortcut or missed the "
            "point, and nothing in a sequence alignment separates those — it "
            "does not know what any step was FOR. The counts say what "
            "differs; whether that is a regression is yours."
        )

        if self.n_candidate == 0:
            parts.append(
                "The run has no recorded steps at all, so every planned step "
                "is missing by construction. A run that did nothing and a run "
                "that was never recorded are the same document here."
            )
        elif self.n_matched == 0 and self.n_reordered == 0:
            parts.append(
                "NOTHING ALIGNED. The two trajectories have no step name in "
                "common, so this is two lists side by side rather than a "
                "comparison — check that the plan names the tools the way the "
                "recorder does before reading anything into the counts."
            )

        if self.matched_on != "kind and name":
            parts.append(
                "Kinds were dropped from the comparison because at least one "
                "step states none, so a tool called through MCP and the same "
                "name called directly are one step here."
            )
        if self.n_reordered:
            parts.append(
                "REORDERED MEANS SOMEWHERE ELSE, NOT OUT OF ORDER. A plan is "
                "often a set of things to do rather than a sequence, and "
                "nothing here knows which kind it was handed."
            )
        if self.repeated_identities:
            first = self.repeated_identities[0]
            more = (
                f" ({len(self.repeated_identities) - 1} other identit(y/ies) "
                f"repeat too"
                + (
                    f", and {self.repeats_not_listed} more are not listed)"
                    if self.repeats_not_listed
                    else ")"
                )
                if len(self.repeated_identities) > 1
                else ""
            )
            parts.append(
                f"`{first['identity']}` appears {first['in_reference']} time(s) "
                f"in the plan and {first['in_candidate']} time(s) in the "
                f"run{more}. Several alignments of exactly this length exist "
                f"when an identity repeats: `matched` is the same number in "
                f"every one of them, but WHICH occurrence is called missing or "
                f"reordered is not."
            )
        if self.n_arguments_not_compared:
            parts.append(
                f"{self.n_arguments_not_compared} paired step(s) had no "
                f"arguments to compare on one side or the other, so they are "
                f"not part of the {self.n_changed_arguments} above and are not "
                f"evidence that their arguments agreed."
            )
        if self.redactions:
            kinds = ", ".join(f"{r['count']}x {r['label']}" for r in self.redactions)
            parts.append(
                f"{sum(r['count'] for r in self.redactions)} credential-shaped "
                f"value(s) were replaced in what is shown here: {kinds}. "
                f"Equality was decided BEFORE that ran, so two values that "
                f"read identically above were still compared as different."
            )
        if self.arg_values_truncated:
            parts.append(
                f"{self.arg_values_truncated} argument value(s) are shown cut "
                f"to {MAX_ARG_CHARS} characters, with the number omitted marked "
                f"in each. They were compared whole."
            )
        if self.arg_keys_dropped:
            parts.append(
                f"{self.arg_keys_dropped} argument key(s) past {MAX_ARG_KEYS} "
                f"per step were not compared and are neither reported as "
                f"changed nor as unchanged."
            )
        if self.rows_not_listed:
            parts.append(
                f"{self.rows_not_listed} row(s) are not listed "
                f"({self.matched_rows_not_listed} of them matching steps, "
                f"dropped first). Every count above is over the whole "
                f"alignment rather than over what is listed."
            )
        return " ".join(parts)


def plan_comparison(n_reference, n_candidate) -> dict:
    """What an alignment of these two lengths would cost, before it is run.

    A caller can price the work and shorten the span itself rather than
    discovering the cap by hitting it, which is the same courtesy
    `budget.py` extends before a forward pass.
    """
    for value, which in ((n_reference, "n_reference"), (n_candidate, "n_candidate")):
        # `isinstance(True, int)` is True, and `plan_comparison(True, True)`
        # would otherwise cheerfully report a 1-cell comparison.
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TrajectoryError(
                f"{which} has to be a step count — a whole number of steps, "
                f"zero or more."
            )
    cells = n_reference * n_candidate
    fits = cells <= MAX_ALIGN_CELLS
    return {
        "cells": cells,
        "cap": MAX_ALIGN_CELLS,
        "fits": fits,
        "means": (
            f"{n_reference} against {n_candidate} steps is {cells:,} cells of "
            f"table, "
            + (
                f"inside the {MAX_ALIGN_CELLS:,} this will build."
                if fits
                else (
                    f"past the {MAX_ALIGN_CELLS:,} this will build. It refuses "
                    f"rather than aligning a prefix, because a prefix reports "
                    f"every step past the cut as missing — which is the exact "
                    f"wrong conclusion this measurement exists to get right. "
                    f"Compare a subtree, or a plan rather than a transcript."
                )
            )
        ),
    }


def _lcs(left: list, right: list) -> list:
    """Index pairs of one longest common subsequence, in order.

    The table is kept for the traceback, as `array("i")` rows: 4 bytes a cell
    against a Python int's 8 plus a pointer into a list, which at the cap is
    1 MB rather than roughly 10. Bounded by the caller, never here — a
    function that quietly shrank its own input would be the truncation this
    module refuses.
    """
    n, m = len(left), len(right)
    zeros = array("i", [0]) * (m + 1)
    table = [zeros]
    for i in range(1, n + 1):
        previous = table[i - 1]
        current = array("i", [0]) * (m + 1)
        value = left[i - 1]
        for j in range(1, m + 1):
            if value == right[j - 1]:
                current[j] = previous[j - 1] + 1
            else:
                up, back = previous[j], current[j - 1]
                current[j] = up if up >= back else back
        table.append(current)

    pairs = []
    i, j = n, m
    while i > 0 and j > 0:
        if left[i - 1] == right[j - 1]:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif table[i - 1][j] >= table[i][j - 1]:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def _identities(keys) -> list:
    """Every distinct alignment key, in first-seen order, as ints.

    Interned so the DP loop compares small integers rather than tuples of
    strings. The table is the hot part of this module and it is the only part
    that is.
    """
    table: dict = {}
    out = []
    for k in keys:
        got = table.get(k)
        if got is None:
            got = table[k] = len(table)
        out.append(got)
    return out


def align(*, reference, candidate) -> Alignment:
    """Two trajectories, positioned against each other.

    `reference` is what was supposed to happen — a written plan, or a run kept
    as the one that was right. `candidate` is what did. Each is a list of
    recorded-step dicts, or of plain names, or a mixture; the caller does the
    fetching, so this module never opens a store and can be tested without one.

    Every count it returns is a count. There is no score here on purpose, and
    `means()` says why where the reader will see it.
    """
    ref = [_normalise(s, where="the plan") for s in _ordered(reference)]
    cand = [_normalise(s, where="the run") for s in _ordered(candidate)]

    if not ref:
        raise TrajectoryError(
            "there is no reference trajectory to compare against. An empty "
            "plan makes every recorded step 'extra', which reads as a finding "
            "about the run and is a fact about the plan. Name the steps you "
            "expected — bare tool names are enough."
        )

    for steps, which in ((ref, "the plan"), (cand, "the run")):
        if len(steps) > MAX_STEPS_PER_SIDE:
            raise TrajectoryError(
                f"{which} has {len(steps):,} steps and this aligns up to "
                f"{MAX_STEPS_PER_SIDE:,} a side. Aligning a prefix instead "
                f"would report every step past the cut as missing, which is "
                f"the exact wrong conclusion this measurement exists to get "
                f"right — so compare a subtree, or a plan rather than a "
                f"transcript."
            )

    cells = len(ref) * len(cand)
    if cells > MAX_ALIGN_CELLS:
        priced = plan_comparison(len(ref), len(cand))
        raise TrajectoryError(priced["means"])

    # If ANY step on either side states no kind, kind cannot be part of the
    # key for ANY of them: a plan written as bare tool names would otherwise
    # match nothing in a run that records `tool_call`, and every step would
    # report as both missing and extra. Dropping it costs the distinction
    # between an MCP call and a direct one, which is said out loud rather than
    # absorbed.
    on_name_only = any(not s.kind for s in ref) or any(not s.kind for s in cand)
    matched_on = "name alone" if on_name_only else "kind and name"

    def key_of(step: _Step):
        return step.name if on_name_only else (step.kind, step.name)

    ref_keys = [key_of(s) for s in ref]
    cand_keys = [key_of(s) for s in cand]
    # Interned together, so the same identity gets the same integer on both
    # sides and the DP loop compares ints rather than tuples of strings.
    interned = _identities(ref_keys + cand_keys)
    pairs = _lcs(interned[: len(ref_keys)], interned[len(ref_keys) :])
    matched_ref = {i for i, _ in pairs}
    matched_cand = {j for _, j in pairs}

    # A moved step is one deletion plus one insertion to an edit distance, and
    # reporting it as both is two findings about one event. Leftovers of the
    # same identity are paired in order -- FIFO, so repeated identities pair
    # predictably rather than by whichever the set iterated first.
    waiting: dict = {}
    for j, k in enumerate(cand_keys):
        if j not in matched_cand:
            waiting.setdefault(k, []).append(j)
    reorder_ref: dict = {}
    reorder_cand: dict = {}
    for i, k in enumerate(ref_keys):
        if i in matched_ref:
            continue
        queue = waiting.get(k)
        if queue:
            j = queue.pop(0)
            reorder_ref[i] = j
            reorder_cand[j] = i

    tally: dict = {}
    counters: dict = {}
    # Names are scrubbed ONCE per step rather than once per row, and after the
    # keys above are built from the raw ones. A tool name is an identifier
    # rather than model output, so this almost never fires -- but "almost
    # never" is not a reason for the one field this module prints on every row
    # to be the one field it never looked at.
    for step in (*ref, *cand):
        step.display_name, _ = _redact(step.name, tally)

    out = Alignment(
        matched_on=matched_on,
        n_reference=len(ref),
        n_candidate=len(cand),
        cells=cells,
    )

    def paired_row(i: int, j: int, status: str) -> Row:
        diffs, note = _diff_arguments(ref[i], cand[j], tally, counters)
        row = Row(
            status=status,
            kind=cand[j].kind or ref[i].kind,
            name=cand[j].display_name,
            reference_index=i,
            candidate_index=j,
            step_id=cand[j].step_id,
            argument_diffs=diffs,
            arguments_compared=not note,
            arguments_note=note,
        )
        if note:
            out.n_arguments_not_compared += 1
        elif diffs:
            out.n_changed_arguments += 1
        return row

    # Walked as a merge over the matched anchors, so the listing reads in the
    # order the RUN happened with the plan's unmet steps slotted where it
    # expected them. A reordered pair is emitted once, at the position it
    # actually ran, carrying both indices.
    rows: list[Row] = []
    i = j = 0

    def drain_reference(until: int) -> None:
        nonlocal i
        while i < until:
            if i not in reorder_ref:
                rows.append(
                    Row(
                        status=MISSING,
                        kind=ref[i].kind,
                        name=ref[i].display_name,
                        reference_index=i,
                    )
                )
                out.n_missing += 1
            i += 1

    def drain_candidate(until: int) -> None:
        nonlocal j
        while j < until:
            origin = reorder_cand.get(j)
            if origin is None:
                rows.append(
                    Row(
                        status=EXTRA,
                        kind=cand[j].kind,
                        name=cand[j].display_name,
                        candidate_index=j,
                        step_id=cand[j].step_id,
                    )
                )
                out.n_extra += 1
            else:
                rows.append(paired_row(origin, j, REORDERED))
                out.n_reordered += 1
            j += 1

    for ri, cj in pairs:
        drain_reference(ri)
        drain_candidate(cj)
        rows.append(paired_row(ri, cj, MATCHED))
        out.n_matched += 1
        i, j = ri + 1, cj + 1
    drain_reference(len(ref))
    drain_candidate(len(cand))

    out.arg_keys_dropped = counters.get("keys_dropped", 0)
    out.arg_values_truncated = counters.get("truncated", 0)
    out.redactions = [
        {"label": label, "count": count} for label, count in sorted(tally.items())
    ]
    out.repeated_identities, out.repeats_not_listed = _repeats(ref_keys, cand_keys)
    out.rows = _cap_rows(rows, out)
    return out


def _repeats(ref_keys: list, cand_keys: list) -> tuple[list, int]:
    """Identities that occur more than once and exist on both sides.

    Both sides, because that is exactly where the ambiguity lives: an identity
    the plan never names produces `extra` rows however it repeats, and there is
    no question of which planned step went unmatched.
    """
    shared = set(ref_keys) & set(cand_keys)
    found = []
    for k in sorted(shared, key=str):
        in_ref = ref_keys.count(k)
        in_cand = cand_keys.count(k)
        if in_ref < 2 and in_cand < 2:
            continue
        found.append(
            {
                "identity": k if isinstance(k, str) else _label(*k),
                "in_reference": in_ref,
                "in_candidate": in_cand,
            }
        )
    return found[:MAX_REPEATS_LISTED], max(0, len(found) - MAX_REPEATS_LISTED)


def _cap_rows(rows: list, out: Alignment) -> list:
    """The listing, bounded — with the differences kept and the cut counted.

    Matched rows go first, because a reader opening this is looking for what
    differs and a matched row is the one whose absence from the listing costs
    least. The counts on `out` are already computed over every row, so nothing
    dropped here changes an answer.
    """
    if len(rows) <= MAX_ROWS_LISTED:
        return rows
    kept = [r for r in rows if r.status != MATCHED]
    out.matched_rows_not_listed = len(rows) - len(kept)
    if len(kept) > MAX_ROWS_LISTED:
        # Even the differences do not fit. Cut from the tail so the listing
        # stays in run order, and count what went.
        out.rows_not_listed = out.matched_rows_not_listed + (
            len(kept) - MAX_ROWS_LISTED
        )
        return kept[:MAX_ROWS_LISTED]
    out.rows_not_listed = out.matched_rows_not_listed
    return kept


def read_plan(raw) -> list:
    """A reference trajectory from an untrusted document.

    Accepts a JSON string, a list, or an object with a `steps` or `plan` key,
    and refuses by naming the field — the same contract `rubric.parse` holds,
    because both are fed documents somebody hand-wrote at two in the morning.

    Returns the raw step entries rather than parsed ones: `align` normalises
    both sides through one function, and a second normaliser here would be the
    drift `modelmri.record` already had to be rescued from.
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError as err:
            raise TrajectoryError(
                "that plan is not readable JSON. A plan is a list of step "
                "names, or of objects with a 'name' and optionally a 'kind'."
            ) from err
    if isinstance(raw, dict):
        for field_name in ("steps", "plan", "trajectory"):
            if field_name in raw:
                raw = raw[field_name]
                break
        else:
            raise TrajectoryError(
                "that plan is an object with no 'steps', 'plan' or "
                "'trajectory' list in it, so there is nothing here to align "
                "against."
            )
    if not isinstance(raw, list):
        raise TrajectoryError(
            "a plan is a list of steps, in the order they were expected to "
            "happen."
        )
    # Normalised once here purely to raise the per-step refusals eagerly, so a
    # malformed plan is rejected when it is READ rather than halfway through a
    # comparison somebody is already looking at.
    for entry in raw:
        _normalise(entry, where="the plan")
    return list(raw)
