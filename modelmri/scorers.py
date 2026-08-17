"""Named metrics that need no model, each one stating when it is wrong.

The deterministic four-fifths of any evaluation — did it say the right word, is
that JSON, is the number within tolerance, did it use a phrase it was told not
to — needs no judge, no key and no forward pass. This project already ships the
two hard primitives: `judge.py` reads a model's probability mass on a rubric
somebody wrote, and `rubric.py` runs exact predicates over recorded traces.
Neither is a catalogue. So the easy part gets hand-rolled at every call site,
and a hand-rolled exact match is exactly where an unannounced `.strip().lower()`
goes in and never comes out.

DeepEval ships fifty of these; Braintrust ships autoevals. Both give each entry
a docstring.

WHAT IS DIFFERENT HERE IS THAT EVERY ENTRY STATES ITS OWN FAILURE MODE, and
that sentence travels in the result rather than in the documentation. The person
who needs to read "exact match is whitespace-sensitive and your model ended its
answer with a newline" is looking at a red row, not at a manual. A catalogue
whose entries say when they are wrong is a different product from one whose
entries are merely documented.

## Unmeasurable is not zero

A metric that cannot be computed returns no score and a sentence saying why. It
does not return 0.0, and the difference is arithmetic rather than manners: a
zero averages into a suite total as a failure the model may not have earned,
while a refusal is excluded from the denominator and counted separately, in
public. `weights_scan.py` draws the same line between `dangerous` and
`unscanned` — "I found something" and "I could not tell" are different answers
and only one of them should count against anybody.

Malformed JSON is the case that matters most, because it can be on either side.
If the FIXTURE does not parse, the model's answer was never compared to
anything, and scoring that row zero would be recording a fault against the
model for a broken test file.

## Normalisation is a parameter, never a default

Every text metric takes a `Normalisation` and applies nothing at all unless one
is passed. `AS_IS` really is as-is: a trailing newline fails an exact match, and
that is the honest answer to "are these the same string".

The presets say what they cost. Case folding is not a relabelling — `ß` folds
to `ss`, so the string got longer and an edit distance measured afterwards is a
distance between two strings that neither side produced. `NFKC` rewrites
characters rather than merely composing accents: `①` becomes `1` and full-width
letters become ASCII. Every result carries the sentence describing what was
done, because a metric that quietly collapsed whitespace and one that did not
are two different metrics wearing the same name.

## Nothing here is a verdict on a model

A metric that passes says the string matched. It does not say the answer was
right, and `rubric.py` holds the same line for its predicates. These metrics are
deterministic — the same two strings give the same score next Tuesday on a
laptop with the network off — so all of the variance lives in the outputs fed
in, and a pass rate measured over one generation is a sample of the model, not
a property of it.

## Cost is stated before it is spent

Stdlib only, no I/O, no model, no accelerator. The two operations that are not
free are bounded by measured numbers rather than round ones, and past a bound
the answer is unmeasurable-with-the-lengths-named rather than a number computed
on a silently truncated pair.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass, field

from .errors import BadRequest

# The longest pair `edit_similarity` will align. The dynamic programme is
# O(len(a) * len(b)) cells and there is no way around that; the two-row form
# below keeps memory at O(min(len(a), len(b))) integers, so the cap is on TIME.
#
# MEASURED here, CPython 3.13 on this machine, two-row DP over random text:
#
#       1,000 x 1,000     1,000,000 cells     0.249 s
#       2,000 x 2,000     4,000,000 cells     0.990 s
#       4,000 x 4,000    16,000,000 cells     6.900 s
#       8,000 x 8,000    64,000,000 cells    35.775 s
#
# Superlinear in cells past roughly four million, which is the working row
# falling out of cache. One second per row is a suite that finishes; seven
# seconds per row over a thousand rows is two hours. Past the cap this reports
# unmeasurable WITH both lengths, never a distance computed on a truncated
# pair — truncating changes the number and nothing in the output would say so.
MAX_EDIT_CHARS = 2_000

# The most leaf paths a JSON structural diff will walk before it refuses. A
# document with more scalars than this is a data dump rather than an answer,
# and the diff would be a wall of paths nobody reads. Refused with the count
# rather than diffed to the cap: a partial diff reports "12 differences" for a
# document that has four hundred.
MAX_JSON_LEAVES = 5_000

# How many individual differences travel in a result. The TOTAL is always
# reported beside this, because a list that stops at fifty and does not say so
# reads as "fifty differences".
MAX_DIFFERENCES_SHOWN = 50

# How much of a differing value is quoted back. A model that returned a
# 40 KB string should not put 40 KB into every row of a report.
MAX_VALUE_CHARS = 200

# Bounds on a caller-supplied regular expression. `rubric.py` sets the same
# pattern bound for the same reason.
#
# Neither is a defence against catastrophic backtracking: `(a+)+b` is
# exponential in the length of the subject and no length cap makes that safe.
# The patterns here come from the person running the suite rather than from the
# model, which is the actual reason this is acceptable, and it is stated rather
# than implied.
MAX_PATTERN_CHARS = 500
MAX_REGEX_INPUT_CHARS = 100_000

# How many substrings a contains-check will take. Refused past this rather than
# truncated, because a silently shortened needle list turns a failing check into
# a passing one.
MAX_NEEDLES = 200

# Row dictionaries returned by `score_rows` by default. The totals are computed
# over every row; this bounds only what is handed back for display, and the
# response says how many were shown against how many there were.
DEFAULT_ROW_SAMPLE = 200


# ------------------------------------------------------------- normalisation


@dataclass(frozen=True)
class Normalisation:
    """What is done to text before it is compared, named and stated.

    Applied in one fixed order — Unicode form, then case, then whitespace —
    because the order changes the answer. NFKC turns a full-width `Ａ` into
    `A`; folding case first would leave the full-width form alone and the two
    sides would still differ.
    """

    name: str
    # "", "NFC", "NFD", "NFKC" or "NFKD". Empty means the text is compared in
    # whatever form it arrived in, which is a real choice: two strings that
    # render identically can be different sequences of code points.
    unicode_form: str = ""
    # `str.casefold`, not `str.lower`. Lowercasing is a display transform and
    # gets caseless matching wrong for exactly the scripts nobody tests on;
    # casefold is the operation the Unicode standard defines for this.
    fold_case: bool = False
    collapse_whitespace: bool = False
    strip: bool = False

    def apply(self, text: str) -> str:
        out = text
        if self.unicode_form:
            out = unicodedata.normalize(self.unicode_form, out)
        if self.fold_case:
            out = out.casefold()
        if self.collapse_whitespace:
            out = " ".join(out.split())
        if self.strip:
            out = out.strip()
        return out

    def describe(self) -> str:
        """The sentence that travels in every result this touched."""
        if not (
            self.unicode_form
            or self.fold_case
            or self.collapse_whitespace
            or self.strip
        ):
            return (
                "Nothing was changed before comparing — this is the raw text on "
                "both sides, so a trailing newline or a capital letter is a "
                "difference."
            )
        did = []
        if self.unicode_form:
            did.append(f"normalised to Unicode {self.unicode_form}")
        if self.fold_case:
            did.append("case-folded")
        if self.collapse_whitespace:
            did.append("had every run of whitespace collapsed to one space")
        if self.strip:
            did.append("stripped of leading and trailing whitespace")
        costs = []
        if self.fold_case:
            costs.append(
                "case folding is not a relabelling — `ß` folds to `ss`, so the "
                "compared string can be longer than the one produced"
            )
        if self.unicode_form.startswith("NFK"):
            costs.append(
                f"{self.unicode_form} rewrites characters rather than only "
                f"composing accents — `①` becomes `1` and full-width letters "
                f"become ASCII"
            )
        if self.collapse_whitespace:
            costs.append(
                "indentation and line breaks are gone, which matters when the "
                "answer was code or a table"
            )
        tail = f" Under `{self.name}` {'; '.join(costs)}." if costs else ""
        return f"Before comparing, both sides were {', '.join(did)}.{tail}"


AS_IS = Normalisation("as_is")
TRIMMED = Normalisation("trimmed", strip=True)
COLLAPSED = Normalisation("collapsed", collapse_whitespace=True, strip=True)
LENIENT = Normalisation(
    "lenient", fold_case=True, collapse_whitespace=True, strip=True
)
UNICODE_LENIENT = Normalisation(
    "unicode_lenient",
    unicode_form="NFKC",
    fold_case=True,
    collapse_whitespace=True,
    strip=True,
)

NORMALISATIONS = {
    n.name: n
    for n in (AS_IS, TRIMMED, COLLAPSED, LENIENT, UNICODE_LENIENT)
}


def resolve_normalisation(value) -> Normalisation:
    """A `Normalisation`, from one or from the name of a preset.

    Names are accepted so a suite can be driven from a configuration file
    without the file being able to invent a policy — an unknown name is a
    refusal that lists the ones that exist, rather than a quiet fall back to
    doing nothing, which would silently make every comparison stricter.
    """
    if isinstance(value, Normalisation):
        return value
    if isinstance(value, str):
        found = NORMALISATIONS.get(value)
        if found is not None:
            return found
        raise BadRequest(
            f"`{value}` is not a normalisation this knows. The policies are: "
            f"{', '.join(sorted(NORMALISATIONS))}. Falling back to doing "
            f"nothing would silently make every comparison stricter than the "
            f"one that was asked for."
        )
    raise BadRequest(
        "a normalisation must be a Normalisation or the name of one of: "
        f"{', '.join(sorted(NORMALISATIONS))}."
    )


# -------------------------------------------------------------- the result


@dataclass
class Result:
    """One metric's answer about one row: a score, a verdict, or neither.

    `score` is `None` exactly when the metric could not be computed, and in
    that case `reason` says why in a sentence. `passed` is `None` both then and
    when a graded metric was given no threshold — a score with no stated bar is
    a measurement, and calling it a pass would be inventing the bar.

    The invariants are enforced in `__post_init__` rather than trusted, because
    the one bug this type exists to prevent is a metric returning 0.0 for
    something it never measured.
    """

    metric: str
    score: float | None = None
    passed: bool | None = None
    # Why there is no score. Never empty when `score` is None.
    reason: str = ""
    # When this metric is wrong. Travels in the body, per the house rule that
    # the caveat belongs where the reader is looking.
    failure_mode: str = ""
    # What was done to the text first, as a sentence. Empty for metrics that
    # compare no text.
    normalisation: str = ""
    # One observation about THIS row, where there is one worth making — "these
    # differ only in case and whitespace" is the actionable half of a failure.
    note: str = ""
    detail: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.score is None and not self.reason:
            raise ValueError(
                "a Result with no score must carry a reason. An unmeasurable "
                "row with no explanation is indistinguishable from a bug, and "
                "the whole point of this type is that the two never blur."
            )
        if self.score is None and self.passed is not None:
            raise ValueError(
                "a Result with no score cannot carry a verdict: nothing was "
                "measured, so nothing passed or failed."
            )
        if self.score is not None and not 0.0 <= self.score <= 1.0:
            raise ValueError(
                f"a score must be between 0 and 1; this one is {self.score}. "
                "Every metric here is aggregated with every other, so a score "
                "on another scale would move a suite total without anything "
                "saying it had."
            )

    @property
    def measurable(self) -> bool:
        return self.score is not None

    def means(self) -> str:
        if not self.measurable:
            return (
                f"`{self.metric}` COULD NOT BE MEASURED on this row: "
                f"{self.reason} This is not a score of zero. An unmeasurable "
                f"row is left out of the denominator when these are "
                f"aggregated, because a zero would average in as a failure "
                f"that nothing here established."
            )
        if self.passed is True:
            head = f"`{self.metric}` PASSED, scoring {self.score:.4g} out of 1."
        elif self.passed is False:
            head = f"`{self.metric}` FAILED, scoring {self.score:.4g} out of 1."
        else:
            head = (
                f"`{self.metric}` scored {self.score:.4g} out of 1 and returns "
                f"NO verdict, because no threshold was stated. A score without "
                f"a stated bar is a measurement; calling it a pass would be "
                f"inventing the bar."
            )
        parts = [head, self.note, self.normalisation, self.failure_mode]
        return " ".join(p for p in parts if p)

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "measurable": self.measurable,
            "score": self.score,
            "passed": self.passed,
            "reason": self.reason,
            "failure_mode": self.failure_mode,
            "normalisation": self.normalisation,
            "note": self.note,
            "detail": dict(self.detail),
            "means": self.means(),
        }


# ------------------------------------------------------------- the catalogue


@dataclass(frozen=True)
class Metric:
    """One entry in the catalogue, with the sentence that makes it useful."""

    name: str
    summary: str
    # THE differentiator. Not "what this does" — when this is wrong.
    failure_mode: str
    # What the second argument is, in words, or "" for a metric that needs
    # nothing to compare against.
    reference: str
    # Whether this can score between 0 and 1, or only ever 0 and 1. A mean over
    # binary metrics is a pass rate; a mean over graded ones is not, and a
    # suite mixing them should know which it has.
    graded: bool
    options: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "summary": self.summary,
            "failure_mode": self.failure_mode,
            "reference": self.reference,
            "graded": self.graded,
            "options": [{"name": n, "means": m} for n, m in self.options],
        }


_NORMALISATION_OPTION = (
    "normalisation",
    "a Normalisation, or the name of a preset: "
    + ", ".join(sorted(NORMALISATIONS))
    + ". Nothing is applied unless one is given.",
)

CATALOGUE: dict[str, Metric] = {
    "exact_match": Metric(
        name="exact_match",
        summary="The output is the expected string, character for character.",
        failure_mode=(
            "Exact match is case- and whitespace-sensitive by default: an "
            "answer that is right but ends in a newline, or writes `Paris.` "
            "where the fixture says `Paris`, scores zero. A normalisation "
            "changes that AND changes what was compared, which is why the one "
            "used is named in every result instead of assumed."
        ),
        reference="the expected text",
        graded=False,
        options=(_NORMALISATION_OPTION,),
    ),
    "numeric_close": Metric(
        name="numeric_close",
        summary="The output parses as a number within a stated tolerance.",
        failure_mode=(
            "A tolerance is a window somebody chose, and absolute and relative "
            "windows disagree at scale: `abs_tol=0.01` is strict on 0.5 and "
            "meaningless on 1e9, and `rel_tol=0.01` is the other way round. "
            "When both are given, satisfying EITHER passes — that is "
            "math.isclose's rule, stated because assuming both must hold is "
            "just as reasonable. The parse is strict: `42` is a number and "
            "`the answer is 42` is not, because a metric that hunts for the "
            "first number in a sentence picks the wrong one as soon as there "
            "are two."
        ),
        reference="the expected number",
        graded=False,
        options=(
            (
                "abs_tol",
                "a fixed window: the answer passes if it is within this much "
                "of the expected value, whatever the size of the value.",
            ),
            (
                "rel_tol",
                "a proportional window: 0.01 means within 1% of the expected "
                "value. Undefined when the expected value is zero.",
            ),
        ),
    ),
    "json_valid": Metric(
        name="json_valid",
        summary="The output is text that parses as JSON.",
        failure_mode=(
            "Valid is not correct — `{}` parses. Python's parser is also more "
            "permissive than the format, so two things are handled explicitly: "
            "`NaN`, `Infinity` and `-Infinity` are accepted by json.loads and "
            "are not JSON, so they FAIL here; and duplicate keys are legal to "
            "write and unpredictable to read, so they are reported rather than "
            "failed, with the last value silently winning inside every parser "
            "including this one."
        ),
        reference="",
        graded=False,
    ),
    "json_diff": Metric(
        name="json_diff",
        summary=(
            "Both sides parse as JSON and agree at every path, scored by the "
            "fraction of paths that agree."
        ),
        failure_mode=(
            "A structural diff scores paths, not meaning: one wrong value in "
            "fifty scores 0.98, and if that value was the one that mattered "
            "the score is a comfortable number for a wrong answer. Numbers "
            "compare by value, so `1` and `1.0` agree; booleans are not "
            "numbers here, so `true` and `1` do not. List order is "
            "significant — the same items reordered is every path changed. "
            "With ignore_extra_keys, a document that returns every key with "
            "every value cannot fail on extras."
        ),
        reference="the expected JSON document, as text or already decoded",
        graded=True,
        options=(
            (
                "ignore_extra_keys",
                "when true, keys present in the output and absent from the "
                "expected document are neither scored nor reported as "
                "differences.",
            ),
        ),
    ),
    "edit_similarity": Metric(
        name="edit_similarity",
        summary=(
            "One minus the Levenshtein distance over the longer string: 1.0 "
            "for identical text, 0.0 for text with nothing in common."
        ),
        failure_mode=(
            "It counts characters, not meaning: a correct answer phrased "
            "differently scores low, and a wrong answer one character from the "
            "right one scores 0.99. It counts CODE POINTS, so an emoji written "
            "with a skin-tone modifier is two edits and one glyph, and a "
            "decomposed accent is two edits where a composed one is one — "
            "which is what the unicode_lenient normalisation is for. It "
            "returns no verdict unless min_similarity is stated, because the "
            "bar depends on the task and every default would be somebody "
            "else's."
        ),
        reference="the expected text",
        graded=True,
        options=(
            _NORMALISATION_OPTION,
            (
                "min_similarity",
                "the similarity at or above which this counts as a pass. "
                "Without it the result carries a score and no verdict.",
            ),
        ),
    ),
    "regex_match": Metric(
        name="regex_match",
        summary="The output matches a regular expression.",
        failure_mode=(
            "`search` finds the pattern anywhere in the output, so only `^` "
            "and `$` anchor it — and `$` also matches before a trailing "
            "newline. `.` does not cross a newline unless dotall is set, which "
            "is how a pattern that works on one-line answers fails on a "
            "paragraph. It tests the surface string: `\\bno\\b` does not find "
            "`Nope`. A pattern that can match the empty string is refused in "
            "search mode rather than quietly passing everything."
        ),
        reference="a regular expression",
        graded=False,
        options=(
            (
                "mode",
                "search (anywhere), match (from the start) or fullmatch (the "
                "whole string). Default search.",
            ),
            ("ignore_case", "compile the pattern with re.IGNORECASE."),
            ("dotall", "let `.` match a newline."),
            ("multiline", "let `^` and `$` match at every line break."),
        ),
    ),
    "schema_conformance": Metric(
        name="schema_conformance",
        summary=(
            "The output parses as JSON and satisfies a JSON Schema, restricted "
            "to the keywords implemented here."
        ),
        failure_mode=(
            "Only part of JSON Schema is implemented, and a schema using "
            "anything else is REFUSED rather than half-checked — a quietly "
            "ignored `pattern` passes every document that violates it. "
            "Conformance is shape, not truth: a perfectly conforming object "
            "can be entirely wrong. Following the specification, a float with "
            "no fractional part satisfies `integer`, so `1.0` conforms where "
            "`1` was meant; booleans never satisfy `number` or `integer`."
        ),
        reference="a JSON Schema using only the supported keywords",
        graded=False,
    ),
    "contains_all": Metric(
        name="contains_all",
        summary=(
            "Every required substring appears in the output, scored by the "
            "fraction present."
        ),
        failure_mode=(
            "Substrings, not words: `kill` is found inside `skill` and `no` "
            "inside `nothing`. The score is the fraction of required strings "
            "present, so an output containing four of five scores 0.8 while "
            "missing the only one that mattered. Normalisation applies to the "
            "needles as well as to the text, so a needle with a trailing space "
            "is changed too."
        ),
        reference="the substrings that must all appear",
        graded=True,
        options=(_NORMALISATION_OPTION,),
    ),
    "contains_none": Metric(
        name="contains_none",
        summary=(
            "No forbidden substring appears in the output, scored by the "
            "fraction absent."
        ),
        failure_mode=(
            "The same substring trap as contains_all — `kill` is found inside "
            "`skill` — and a sharper one on top: finding nothing means nothing "
            "was found IN ONE SAMPLE OF SURFACE STRINGS. A paraphrase walks "
            "straight through a banned-phrase list, so a pass here is evidence "
            "about this one string and never a property of the model."
        ),
        reference="the substrings that must not appear",
        graded=True,
        options=(_NORMALISATION_OPTION,),
    ),
}


def catalogue() -> list[dict]:
    """Every metric, with its stated failure mode. The point of the module."""
    return [CATALOGUE[name].to_dict() for name in sorted(CATALOGUE)]


def describe(name: str) -> dict:
    """One metric, or a refusal that lists the ones that exist."""
    return _metric(name).to_dict()


def _metric(name: str) -> Metric:
    found = CATALOGUE.get(name)
    if found is None:
        raise BadRequest(
            f"`{name}` is not a metric in this catalogue. The ones that exist "
            f"are: {', '.join(sorted(CATALOGUE))}."
        )
    return found


# ------------------------------------------------------------------ helpers


def _unmeasurable(metric: str, reason: str, *, normalisation: str = "") -> Result:
    return Result(
        metric=metric,
        score=None,
        passed=None,
        reason=reason,
        failure_mode=CATALOGUE[metric].failure_mode,
        normalisation=normalisation,
    )


def _measured(
    metric: str,
    *,
    score: float,
    passed: bool | None,
    normalisation: str = "",
    note: str = "",
    detail: dict | None = None,
) -> Result:
    return Result(
        metric=metric,
        score=score,
        passed=passed,
        failure_mode=CATALOGUE[metric].failure_mode,
        normalisation=normalisation,
        note=note,
        detail=detail or {},
    )


def _as_text(value, *, metric: str) -> tuple[str, str]:
    """The string to compare, or a reason there is not one. Never a coercion.

    A missing output is the case this exists for: the field was absent or the
    model returned nothing, which is not a wrong answer and must not be scored
    as one.
    """
    if value is None:
        return "", (
            "there is no output for this row — the field was absent or the "
            "model returned nothing. A missing answer is not a wrong answer, "
            "so this is unmeasurable rather than a failure."
        )
    if isinstance(value, str):
        return value, ""
    return "", (
        f"the output is a {type(value).__name__} rather than text, and "
        f"`{metric}` compares text. Converting it here would make the integer "
        f"1 and the string \"1\" the same answer; convert it deliberately at "
        f"the call site instead."
    )


def _reference_text(value, *, metric: str) -> str:
    """The expected string, or a `BadRequest` — the fixture is the caller's.

    Deliberately asymmetric with `_as_text`. A model that returned nothing is a
    measurement outcome; a fixture that contains nothing is a broken call, and
    reporting it as an unmeasurable row would hide a mistake in the suite
    behind a per-row footnote.
    """
    if value is None:
        raise BadRequest(
            f"`{metric}` was given nothing to compare against. Every row needs "
            f"an expected value; a row without one is a gap in the suite "
            f"rather than a result."
        )
    if isinstance(value, str):
        return value
    raise BadRequest(
        f"`{metric}` needs expected text and was given a "
        f"{type(value).__name__}. Convert the fixture at the call site — doing "
        f"it here would make the integer 1 and the string \"1\" the same "
        f"expected answer."
    )


# Accepted number syntax. `float()` alone is too generous for this job: it
# takes `nan`, `inf`, `1_000` and `  5  `, and every one of those would make a
# metric answer a question nobody asked. Written out so the grammar is the
# stated one rather than whatever the C library accepts this year.
_NUMBER_RE = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?\Z")


def _as_number(value) -> tuple[float, str]:
    """A finite float, or a reason this is not one."""
    if value is None:
        return 0.0, (
            "there is no output for this row — the field was absent or the "
            "model returned nothing, which is not a wrong number."
        )
    # isinstance(True, int) is True, so booleans reach the numeric branch and
    # would score as 1 and 0. `True` is not the answer 1 to any question this
    # library is asked.
    if isinstance(value, bool):
        return 0.0, (
            "the output is the boolean "
            f"{value}, not a number. Python treats True as 1, which is exactly "
            "how a boolean silently becomes a correct numeric answer."
        )
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            return 0.0, (
                "the output is not a finite number — it is a NaN or an "
                "infinity — so no distance from the expected value exists. "
                "NaN in particular compares unequal to everything including "
                "itself, so a tolerance check on it would report a failure "
                "that is really an absence."
            )
        return number, ""
    if isinstance(value, str):
        text = value.strip()
        if not _NUMBER_RE.match(text):
            return 0.0, (
                f"{_brief(value)} does not parse as a number on its own. This "
                f"metric parses the whole output rather than hunting for a "
                f"number inside it, because a hunt picks the wrong one as soon "
                f"as there are two — and it rejects thousands separators, "
                f"because a comma is a decimal point in half the world."
            )
        return float(text), ""
    return 0.0, (
        f"the output is a {type(value).__name__}, which is not a number and "
        f"not text that could be one."
    )


def _brief(value) -> str:
    """A value, quoted back short enough to sit in a row of a report."""
    try:
        text = value if isinstance(value, str) else json.dumps(value, default=repr)
    except (TypeError, ValueError):
        text = repr(value)
    if len(text) <= MAX_VALUE_CHARS:
        return text
    return f"{text[:MAX_VALUE_CHARS]}… ({len(text):,} characters in total)"


# ----------------------------------------------------------------- the metrics


def exact_match(output, expected, *, normalisation=AS_IS) -> Result:
    """The output is the expected string, character for character.

    Nothing is normalised unless a normalisation is passed, and whichever was
    used travels in the result.
    """
    norm = resolve_normalisation(normalisation)
    target = _reference_text(expected, metric="exact_match")
    text, why = _as_text(output, metric="exact_match")
    if why:
        return _unmeasurable("exact_match", why, normalisation=norm.describe())

    left, right = norm.apply(text), norm.apply(target)
    same = left == right
    note = ""
    if not same and UNICODE_LENIENT.apply(text) == UNICODE_LENIENT.apply(target):
        # The actionable half of a failure. A row that differs only in case,
        # whitespace or Unicode form is a normalisation decision nobody made,
        # not a model that got the answer wrong, and saying so is the whole
        # difference between a red row and a fixed suite.
        note = (
            "These differ ONLY in case, whitespace or Unicode form: they are "
            "equal under the `unicode_lenient` normalisation. That is a "
            "decision about this suite rather than a wrong answer."
        )
    return _measured(
        "exact_match",
        score=1.0 if same else 0.0,
        passed=same,
        normalisation=norm.describe(),
        note=note,
        detail={
            "compared_output": _brief(left),
            "compared_expected": _brief(right),
            "normalisation": norm.name,
            "changed_by_normalisation": bool(left != text or right != target),
        },
    )


def numeric_close(output, expected, *, abs_tol=None, rel_tol=None) -> Result:
    """The output parses as a number inside a tolerance that was STATED.

    One of `abs_tol` and `rel_tol` must be given. There is no default, because
    the only honest default is exact float equality and almost no evaluation
    means that — while a metric that quietly picked 1e-9 would be answering a
    question the caller never asked.
    """
    if abs_tol is None and rel_tol is None:
        raise BadRequest(
            "numeric_close needs a stated tolerance. Pass abs_tol= for a fixed "
            "window (abs_tol=0.01 means within one hundredth, whatever the "
            "size of the number) or rel_tol= for a proportional one "
            "(rel_tol=0.01 means within 1% of the expected value). Passing "
            "neither would compare two floats for exact equality, which is "
            "almost never what an evaluation means and would never say so."
        )
    abs_tol = _tolerance(abs_tol, "abs_tol")
    rel_tol = _tolerance(rel_tol, "rel_tol")

    want, why = _as_number(expected)
    if why:
        raise BadRequest(
            f"numeric_close was given an expected value it cannot read as a "
            f"number: {why}"
        )

    got, why = _as_number(output)
    if why:
        return _unmeasurable("numeric_close", why)

    difference = got - want
    error = abs(difference)
    # None rather than 0.0, and this is the sharpest instance of the rule in
    # the module: a relative error against an expected value of zero does not
    # exist, and reporting 0.0 would read as a perfect answer.
    relative = error / abs(want) if want != 0.0 else None

    if want == 0.0 and rel_tol is not None and abs_tol is None:
        return _unmeasurable(
            "numeric_close",
            "a relative tolerance is a proportion of the expected value, and "
            "the expected value is zero, so there is no proportion to take. "
            "Give abs_tol as well — an absolute window is the only kind that "
            "means anything around zero.",
        )

    within_abs = error <= abs_tol if abs_tol is not None else None
    # None in two different situations, and neither of them is False: the
    # relative arm was not asked for, or it was asked for and does not exist
    # because the expected value is zero. Reporting False for "undefined" would
    # read as an answer that failed the check.
    within_rel = None if rel_tol is None or relative is None else relative <= rel_tol
    note = (
        (
            "The expected value is zero, so the relative tolerance does not "
            "apply to this row — a proportion of zero is not a quantity — and "
            "only the absolute window was used."
        )
        if rel_tol is not None and relative is None
        else ""
    )
    passed = within_abs is True or within_rel is True
    return _measured(
        "numeric_close",
        score=1.0 if passed else 0.0,
        passed=passed,
        note=note,
        detail={
            "output_value": got,
            "expected_value": want,
            "difference": difference,
            "absolute_error": error,
            "relative_error": relative,
            "abs_tol": abs_tol,
            "rel_tol": rel_tol,
            "within_abs": within_abs,
            "within_rel": within_rel,
        },
    )


def _tolerance(value, name: str) -> float | None:
    """A tolerance, or a refusal naming the parameter that is wrong."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BadRequest(
            f"{name} must be a number; it was given a "
            f"{type(value).__name__}. A boolean in particular is not a "
            f"tolerance — Python would read True as a window of 1."
        )
    number = float(value)
    if not math.isfinite(number):
        raise BadRequest(
            f"{name} must be finite. An infinite tolerance passes every "
            f"answer, which is a metric that cannot fail."
        )
    if number < 0:
        raise BadRequest(
            f"{name} must not be negative; a negative window fails every "
            f"answer including an exactly correct one."
        )
    return number


def _reject_constant(name: str):
    """`json.loads` accepts NaN and Infinity. JSON does not."""
    raise ValueError(name)


def _parse_json(text: str) -> tuple[object, str, list[str], tuple]:
    """Parse JSON strictly. Returns (value, reason, duplicate_keys, position).

    `reason` is `""` when it parsed — a value rather than an exception, because
    "this is not JSON" is an ordinary measurement outcome here and not an
    exceptional one.

    The decoder's own message is deliberately NOT quoted back. `errors.py` sets
    the rule for published text: never interpolate a caught exception's string.
    The line and column are numbers this module chose to report, so they travel.
    """
    duplicates: list[str] = []

    def pairs(items):
        seen = {}
        for key, value in items:
            if key in seen:
                duplicates.append(key)
            seen[key] = value
        return seen

    try:
        value = json.loads(
            text, parse_constant=_reject_constant, object_pairs_hook=pairs
        )
    except RecursionError:
        # A document nested deeper than the interpreter's stack. Not a syntax
        # error and not a pass: the parser gave up, which is unmeasurable.
        return None, (
            "the document is nested too deeply for the parser to walk, so "
            "nothing here read it."
        ), duplicates, ()
    except json.JSONDecodeError as err:
        return (
            None,
            (
                f"it did not parse as JSON — the decoder stopped at line "
                f"{err.lineno}, column {err.colno}."
            ),
            duplicates,
            (err.lineno, err.colno),
        )
    except ValueError:
        # Raised by `_reject_constant`. `json.loads` accepts NaN, Infinity and
        # -Infinity by default and none of them is JSON; a document containing
        # one parses in Python and fails in every strict reader downstream.
        return (
            None,
            (
                "it contains NaN, Infinity or -Infinity. Python's json module "
                "accepts those and the JSON specification does not, so this "
                "document parses here and fails in a strict reader elsewhere."
            ),
            duplicates,
            (),
        )
    return value, "", duplicates, ()


_JSON_TYPE_NAMES = {
    type(None): "null",
    bool: "boolean",
    int: "integer",
    float: "number",
    str: "string",
    list: "array",
    dict: "object",
}


def _json_type(value) -> str:
    # bool before int, always: isinstance(True, int) is True, and a boolean
    # reported as an integer is how `true` starts satisfying `minimum: 0`.
    if isinstance(value, bool):
        return "boolean"
    return _JSON_TYPE_NAMES.get(type(value), "unknown")


def json_valid(output) -> Result:
    """The output is text that parses as JSON, strictly."""
    if output is None:
        return _unmeasurable(
            "json_valid",
            "there is no output for this row — the field was absent or the "
            "model returned nothing. Nothing was parsed, and an absent answer "
            "is not invalid JSON.",
        )
    if not isinstance(output, str):
        return _unmeasurable(
            "json_valid",
            f"the output is already a {type(output).__name__} rather than "
            f"text, so there is no document to parse. This metric asks whether "
            f"the model's TEXT is JSON, which is a question about a string.",
        )

    value, reason, duplicates, position = _parse_json(output)
    ok = not reason
    note = ""
    if duplicates:
        # Reported, not failed. Duplicate keys are legal to write, and what a
        # reader does with them is unspecified — Python keeps the last. Failing
        # would be a judgement; staying silent would hide a value that vanished.
        note = (
            f"{len(duplicates)} duplicate key(s) were present "
            f"({', '.join(sorted(set(duplicates))[:5])}). The document still "
            f"parses; the LAST value for each key won and the earlier ones are "
            f"gone, here and in every other parser."
        )
    return _measured(
        "json_valid",
        score=1.0 if ok else 0.0,
        passed=ok,
        note=note,
        detail={
            "parsed": ok,
            "top_level": _json_type(value) if ok else None,
            "line": position[0] if position else None,
            "column": position[1] if position else None,
            "duplicate_keys": sorted(set(duplicates)),
            "reason": reason,
        },
    )


def _json_value(value, *, side: str) -> tuple[object, str]:
    """A decoded document, or a reason there is not one.

    A string is treated as JSON TEXT and parsed; anything else is taken as an
    already-decoded document. Stated because the rule has to be one or the
    other: a bare string cannot be both a JSON document and the text of one.
    """
    if value is None:
        return None, (
            f"there is no {side} for this row. Nothing was compared, and a "
            f"missing document is not a mismatched one."
        )
    if isinstance(value, str):
        decoded, reason, _dupes, _pos = _parse_json(value)
        if reason:
            return None, f"the {side} did not parse: {reason}"
        return decoded, ""
    return value, ""


def _leaves(value, *, cap: int) -> tuple[dict, bool]:
    """Every scalar in a document, keyed by the path that reaches it.

    Iterative rather than recursive on purpose: a document nested a few hundred
    deep would exhaust the interpreter's stack inside a metric whose entire job
    is to not fall over on the model's output.

    An empty object or array is its own leaf, so `{}` against `{"a": 1}` is a
    difference rather than an empty comparison. Paths are for display — a key
    containing a dot renders ambiguously and nothing keys off that string.
    """
    out: dict = {}
    stack = [("$", value)]
    while stack:
        path, node = stack.pop()
        if isinstance(node, dict) and node:
            for key in node:
                stack.append((f"{path}.{key}", node[key]))
        elif isinstance(node, list) and node:
            for index, item in enumerate(node):
                stack.append((f"{path}[{index}]", item))
        else:
            out[path] = node
        if len(out) + len(stack) > cap:
            return out, True
    return out, False


def _same_scalar(a, b) -> bool:
    """JSON equality for two leaves, with booleans held apart from numbers."""
    # isinstance(True, int) is True, so `true == 1` in Python. In JSON they are
    # different types and a model that returned one for the other was wrong.
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    if not isinstance(a, bool) and isinstance(a, (int, float)):
        if not isinstance(b, bool) and isinstance(b, (int, float)):
            # 1 and 1.0 are the same JSON number written two ways.
            return a == b
        return False
    if type(a) is not type(b):
        return False
    return a == b


def json_diff(output, expected, *, ignore_extra_keys: bool = False) -> Result:
    """Both sides parse as JSON, and every path agrees.

    Unmeasurable when either side does not parse, and the reason names WHICH
    side — a fixture that does not parse means the model's answer was never
    compared to anything, and scoring that row zero would record a fault
    against the model for a broken test file.
    """
    if not isinstance(ignore_extra_keys, bool):
        raise BadRequest(
            "ignore_extra_keys must be true or false; it decides whether extra "
            "keys are scored at all."
        )

    want, reason = _json_value(expected, side="expected document")
    if reason:
        return _unmeasurable("json_diff", reason)
    got, reason = _json_value(output, side="output")
    if reason:
        return _unmeasurable("json_diff", reason)

    got_leaves, got_over = _leaves(got, cap=MAX_JSON_LEAVES)
    want_leaves, want_over = _leaves(want, cap=MAX_JSON_LEAVES)
    if got_over or want_over:
        which = "output" if got_over else "expected document"
        return _unmeasurable(
            "json_diff",
            f"the {which} has more than {MAX_JSON_LEAVES:,} leaf values, past "
            f"what this will walk. A partial diff would report a handful of "
            f"differences for a document that has hundreds, which is a wrong "
            f"number rather than an incomplete one.",
        )

    paths = set(want_leaves)
    if not ignore_extra_keys:
        paths |= set(got_leaves)

    differences = []
    matched = 0
    for path in sorted(paths):
        in_got, in_want = path in got_leaves, path in want_leaves
        if in_got and in_want:
            if _same_scalar(got_leaves[path], want_leaves[path]):
                matched += 1
                continue
            kind = (
                "type_changed"
                if _json_type(got_leaves[path]) != _json_type(want_leaves[path])
                else "changed"
            )
            differences.append(
                {
                    "path": path,
                    "kind": kind,
                    "expected": _brief(want_leaves[path]),
                    "found": _brief(got_leaves[path]),
                }
            )
        elif in_want:
            differences.append(
                {
                    "path": path,
                    "kind": "missing",
                    "expected": _brief(want_leaves[path]),
                    "found": None,
                }
            )
        else:
            differences.append(
                {
                    "path": path,
                    "kind": "unexpected",
                    "expected": None,
                    "found": _brief(got_leaves[path]),
                }
            )

    score = matched / len(paths) if paths else 1.0
    note = ""
    if ignore_extra_keys:
        extra = len(set(got_leaves) - set(want_leaves))
        note = (
            f"Extra keys were not scored: {extra} path(s) present in the "
            f"output and absent from the expected document were ignored, so "
            f"an output that returned every possible key could not fail here."
        )
    return _measured(
        "json_diff",
        score=score,
        passed=not differences,
        note=note,
        detail={
            "differences": differences[:MAX_DIFFERENCES_SHOWN],
            "differences_shown": min(len(differences), MAX_DIFFERENCES_SHOWN),
            "differences_total": len(differences),
            "paths_compared": len(paths),
            "paths_agreeing": matched,
            "ignore_extra_keys": ignore_extra_keys,
        },
    )


def levenshtein(a: str, b: str) -> int:
    """Edit distance in code points, over two rows rather than a matrix.

    O(min(len(a), len(b))) integers of memory, whatever the length of the other
    side — which is what makes the cap in this module a cap on time alone.
    """
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (ca != cb),
                )
            )
        previous = current
    return previous[-1]


def edit_similarity(
    output, expected, *, normalisation=AS_IS, min_similarity=None
) -> Result:
    """One minus the normalised edit distance, and a verdict only if asked.

    Without `min_similarity` this reports a score and `passed=None`. That is
    not a gap: a similarity of 0.87 is a measurement, and whether it is a pass
    depends on the task. Every default here would be somebody else's threshold
    quietly applied to this one.
    """
    norm = resolve_normalisation(normalisation)
    if min_similarity is not None:
        if isinstance(min_similarity, bool) or not isinstance(
            min_similarity, (int, float)
        ):
            raise BadRequest(
                "min_similarity must be a number between 0 and 1, or omitted "
                "for a score with no verdict."
            )
        if not 0.0 <= float(min_similarity) <= 1.0:
            raise BadRequest(
                f"min_similarity must be between 0 and 1; it was "
                f"{min_similarity}. The score it is compared against never "
                f"leaves that range, so a bar outside it either passes "
                f"everything or fails everything."
            )

    target = _reference_text(expected, metric="edit_similarity")
    text, why = _as_text(output, metric="edit_similarity")
    if why:
        return _unmeasurable("edit_similarity", why, normalisation=norm.describe())

    left, right = norm.apply(text), norm.apply(target)
    if len(left) > MAX_EDIT_CHARS or len(right) > MAX_EDIT_CHARS:
        return _unmeasurable(
            "edit_similarity",
            (
                f"the strings are {len(left):,} and {len(right):,} characters "
                f"after normalisation, past the {MAX_EDIT_CHARS:,} this will "
                f"align. The alignment is quadratic — measured at 0.99 s for a "
                f"2,000-character pair and 6.9 s for a 4,000-character one — "
                f"and truncating to fit would change the distance without "
                f"anything saying so."
            ),
            normalisation=norm.describe(),
        )

    distance = levenshtein(left, right)
    longest = max(len(left), len(right))
    # Two empty strings are identical, which is a similarity of 1.0. Stated
    # rather than left to a division: 0/0 is where a metric invents an answer.
    similarity = 1.0 if longest == 0 else 1.0 - distance / longest
    passed = None if min_similarity is None else similarity >= float(min_similarity)
    note = (
        ""
        if min_similarity is not None
        else (
            "No min_similarity was given, so this row carries a score and no "
            "verdict, and it is counted in the mean score but not in any pass "
            "rate."
        )
    )
    return _measured(
        "edit_similarity",
        score=similarity,
        passed=passed,
        normalisation=norm.describe(),
        note=note,
        detail={
            "distance": distance,
            "longest": longest,
            "min_similarity": (
                None if min_similarity is None else float(min_similarity)
            ),
            "output_length": len(left),
            "expected_length": len(right),
        },
    )


_REGEX_MODES = ("search", "match", "fullmatch")


def regex_match(
    output,
    pattern,
    *,
    mode: str = "search",
    ignore_case: bool = False,
    dotall: bool = False,
    multiline: bool = False,
) -> Result:
    """The output matches a regular expression, in a stated mode."""
    if mode not in _REGEX_MODES:
        raise BadRequest(
            f"`{mode}` is not a match mode. The modes are: "
            f"{', '.join(_REGEX_MODES)} — search finds the pattern anywhere, "
            f"match requires it at the start, fullmatch requires it to consume "
            f"the whole string."
        )
    if not isinstance(pattern, str) or not pattern:
        raise BadRequest(
            "regex_match needs a non-empty pattern. An empty pattern matches "
            "every string including an empty one, so a check built on it "
            "cannot fail."
        )
    if len(pattern) > MAX_PATTERN_CHARS:
        raise BadRequest(
            f"the pattern is {len(pattern):,} characters, past the "
            f"{MAX_PATTERN_CHARS:,} this will compile."
        )

    flags = 0
    if ignore_case:
        flags |= re.IGNORECASE
    if dotall:
        flags |= re.DOTALL
    if multiline:
        flags |= re.MULTILINE
    try:
        compiled = re.compile(pattern, flags)
    except re.error as err:
        # The position is a number this module chose to report; `re`'s own
        # message is not quoted, for the reason `errors.py` gives.
        where = f" The parser stopped at character {err.pos}." if err.pos else ""
        raise BadRequest(
            f"this is not a valid regular expression.{where} The pattern was: "
            f"{_brief(pattern)}"
        ) from None

    if mode == "search" and compiled.search("") is not None:
        # A pattern like `a*` matches the empty string, and `search` finds it
        # inside every output there is. Refused rather than passed: a check
        # that cannot fail is worse than no check, because it reads as one.
        raise BadRequest(
            "this pattern matches the empty string, so in search mode it "
            "matches every output there is, including an empty one. Use "
            "fullmatch if the whole answer must match, or tighten the pattern "
            "— a check that cannot fail reads exactly like one that passed."
        )

    text, why = _as_text(output, metric="regex_match")
    if why:
        return _unmeasurable("regex_match", why)
    if len(text) > MAX_REGEX_INPUT_CHARS:
        return _unmeasurable(
            "regex_match",
            f"the output is {len(text):,} characters, past the "
            f"{MAX_REGEX_INPUT_CHARS:,} this will run a pattern over. Matching "
            f"a slice of it would answer a question about the slice.",
        )

    found = getattr(compiled, mode)(text)
    return _measured(
        "regex_match",
        score=1.0 if found else 0.0,
        passed=found is not None,
        detail={
            "pattern": pattern,
            "mode": mode,
            "flags": {
                "ignore_case": ignore_case,
                "dotall": dotall,
                "multiline": multiline,
            },
            "matched_text": _brief(found.group(0)) if found else None,
            "span": list(found.span()) if found else None,
            "groups": (
                {k: _brief(v) for k, v in found.groupdict().items() if v is not None}
                if found
                else {}
            ),
        },
    )


# The JSON Schema keywords this implements. Anything else is refused rather
# than ignored — a schema whose `pattern` was silently dropped passes every
# document that violates it, and the caller has no way to notice.
SUPPORTED_KEYWORDS = frozenset(
    {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "enum",
        "const",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "uniqueItems",
        "pattern",
    }
)

# Keywords that describe rather than constrain. Accepted and ignored, which is
# what the specification says they mean — unlike the ones above, ignoring these
# cannot make a violating document pass.
ANNOTATION_KEYWORDS = frozenset(
    {
        "$schema",
        "$id",
        "$comment",
        "title",
        "description",
        "default",
        "examples",
        "deprecated",
        "readOnly",
        "writeOnly",
    }
)

_SCHEMA_TYPES = frozenset(
    {"object", "array", "string", "number", "integer", "boolean", "null"}
)


def check_schema(schema) -> None:
    """Refuse a schema this cannot fully enforce, naming the keyword.

    The alternative is what every hand-rolled validator does: ignore what it
    does not know and report conformance anyway. That answer is load-bearing
    and wrong, which is the same failure `weights_scan.py` refuses to make when
    it reports `unscanned` for a file it could not read.
    """
    stack = [(schema, "$")]
    while stack:
        node, path = stack.pop()
        if not isinstance(node, dict):
            raise BadRequest(
                f"the schema at {path} is a {type(node).__name__}; every "
                f"schema and sub-schema here must be an object."
            )
        for key, value in node.items():
            if key in ANNOTATION_KEYWORDS:
                continue
            if key not in SUPPORTED_KEYWORDS:
                raise BadRequest(
                    f"`{key}` (at {path}) is not implemented by "
                    f"schema_conformance. The keywords it enforces are: "
                    f"{', '.join(sorted(SUPPORTED_KEYWORDS))}. A schema whose "
                    f"`{key}` was quietly ignored would pass every document "
                    f"that violates it, so this refuses instead. Use the "
                    f"`jsonschema` package for the full specification, or "
                    f"remove the keyword."
                )
            if key == "properties":
                if not isinstance(value, dict):
                    raise BadRequest(
                        f"`properties` at {path} must be an object mapping "
                        f"names to schemas."
                    )
                for name, sub in value.items():
                    stack.append((sub, f"{path}.properties.{name}"))
            elif key == "items":
                stack.append((value, f"{path}.items"))
            elif key == "additionalProperties" and not isinstance(value, bool):
                raise BadRequest(
                    f"`additionalProperties` at {path} must be true or false "
                    f"here. A schema in that position is a constraint this "
                    f"does not enforce, and enforcing part of it would report "
                    f"conformance for a document that violates the rest."
                )
            elif key == "type":
                names = value if isinstance(value, list) else [value]
                unknown = [str(t) for t in names if t not in _SCHEMA_TYPES]
                if unknown:
                    raise BadRequest(
                        f"`type` at {path} names {', '.join(unknown)}, which "
                        f"is not a JSON type. The types are: "
                        f"{', '.join(sorted(_SCHEMA_TYPES))}."
                    )


def _type_matches(value, want: str) -> bool:
    actual = _json_type(value)
    if want == "number":
        # A boolean is never a number here, which `_json_type` already
        # guarantees by testing bool first.
        return actual in ("integer", "number")
    if want == "integer":
        # The specification says a float with a zero fractional part is an
        # integer, so `1.0` conforms. Stated in the failure mode, because a
        # reader who wanted `1` will get `1.0` and a green row.
        return actual == "integer" or (actual == "number" and float(value).is_integer())
    return actual == want


def _violations(schema: dict, instance, path: str) -> list[dict]:
    """Every way one document fails one schema. Iterative, for deep documents."""
    found: list[dict] = []
    stack = [(schema, instance, path)]
    while stack:
        node, value, where = stack.pop()

        if "type" in node:
            wanted = node["type"]
            names = wanted if isinstance(wanted, list) else [wanted]
            if not any(_type_matches(value, str(t)) for t in names):
                found.append(
                    {
                        "path": where,
                        "keyword": "type",
                        "detail": (
                            f"expected {' or '.join(str(t) for t in names)}, "
                            f"found {_json_type(value)}"
                        ),
                    }
                )
                # A wrong type makes every other constraint at this path
                # meaningless, so nothing below it is checked.
                continue

        if "const" in node and not _same_scalar(value, node["const"]):
            found.append(
                {
                    "path": where,
                    "keyword": "const",
                    "detail": f"expected {_brief(node['const'])}, found {_brief(value)}",
                }
            )
        if "enum" in node and not any(_same_scalar(value, o) for o in node["enum"]):
            found.append(
                {
                    "path": where,
                    "keyword": "enum",
                    "detail": f"{_brief(value)} is not one of the allowed values",
                }
            )

        if isinstance(value, str):
            found.extend(_string_violations(node, value, where))
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            found.extend(_number_violations(node, value, where))
        if isinstance(value, list):
            found.extend(_array_violations(node, value, where))
            if "items" in node:
                for index, item in enumerate(value):
                    stack.append((node["items"], item, f"{where}[{index}]"))
        if isinstance(value, dict):
            found.extend(_object_violations(node, value, where))
            for name, sub in (node.get("properties") or {}).items():
                if name in value:
                    stack.append((sub, value[name], f"{where}.{name}"))
    return found


def _string_violations(node: dict, value: str, where: str) -> list[dict]:
    out = []
    if "minLength" in node and len(value) < node["minLength"]:
        out.append(
            {
                "path": where,
                "keyword": "minLength",
                "detail": f"{len(value)} characters, minimum {node['minLength']}",
            }
        )
    if "maxLength" in node and len(value) > node["maxLength"]:
        out.append(
            {
                "path": where,
                "keyword": "maxLength",
                "detail": f"{len(value)} characters, maximum {node['maxLength']}",
            }
        )
    if "pattern" in node:
        try:
            hit = re.search(str(node["pattern"]), value)
        except re.error:
            raise BadRequest(
                f"the `pattern` at {where} is not a valid regular expression, "
                f"so nothing here can enforce it."
            ) from None
        if hit is None:
            out.append(
                {
                    "path": where,
                    "keyword": "pattern",
                    "detail": f"{_brief(value)} does not match /{node['pattern']}/",
                }
            )
    return out


def _number_violations(node: dict, value, where: str) -> list[dict]:
    checks = (
        ("minimum", value < node.get("minimum", value), "below the minimum"),
        ("maximum", value > node.get("maximum", value), "above the maximum"),
        (
            "exclusiveMinimum",
            value <= node.get("exclusiveMinimum", value - 1),
            "not above the exclusive minimum",
        ),
        (
            "exclusiveMaximum",
            value >= node.get("exclusiveMaximum", value + 1),
            "not below the exclusive maximum",
        ),
    )
    return [
        {
            "path": where,
            "keyword": keyword,
            "detail": f"{value} is {why} {node[keyword]}",
        }
        for keyword, failed, why in checks
        if keyword in node and failed
    ]


def _array_violations(node: dict, value: list, where: str) -> list[dict]:
    out = []
    if "minItems" in node and len(value) < node["minItems"]:
        out.append(
            {
                "path": where,
                "keyword": "minItems",
                "detail": f"{len(value)} items, minimum {node['minItems']}",
            }
        )
    if "maxItems" in node and len(value) > node["maxItems"]:
        out.append(
            {
                "path": where,
                "keyword": "maxItems",
                "detail": f"{len(value)} items, maximum {node['maxItems']}",
            }
        )
    if node.get("uniqueItems"):
        # Compared by their JSON text rather than by hashing, because a list of
        # dicts is unhashable and `[{"a": 1}, {"a": 1}]` is a real duplicate.
        seen, duplicates = set(), 0
        for item in value:
            key = json.dumps(item, sort_keys=True, default=repr)
            if key in seen:
                duplicates += 1
            seen.add(key)
        if duplicates:
            out.append(
                {
                    "path": where,
                    "keyword": "uniqueItems",
                    "detail": f"{duplicates} repeated item(s)",
                }
            )
    return out


def _object_violations(node: dict, value: dict, where: str) -> list[dict]:
    out = [
        {
            "path": where,
            "keyword": "required",
            "detail": f"`{name}` is required and absent",
        }
        for name in (node.get("required") or [])
        if name not in value
    ]
    if node.get("additionalProperties") is False:
        allowed = set(node.get("properties") or {})
        extra = sorted(set(value) - allowed)
        if extra:
            out.append(
                {
                    "path": where,
                    "keyword": "additionalProperties",
                    "detail": f"not allowed: {', '.join(extra)}",
                }
            )
    return out


def schema_conformance(output, schema) -> Result:
    """The output parses as JSON and satisfies a schema this fully enforces."""
    if not isinstance(schema, dict):
        raise BadRequest(
            f"a schema must be an object; this one is a "
            f"{type(schema).__name__}."
        )
    check_schema(schema)

    got, reason = _json_value(output, side="output")
    if reason:
        return _unmeasurable("schema_conformance", reason)

    found = _violations(schema, got, "$")
    return _measured(
        "schema_conformance",
        # Binary on purpose. A fraction of satisfied constraints would need a
        # denominator nobody agreed on — a schema with one required field and
        # one with forty are not on the same scale.
        score=0.0 if found else 1.0,
        passed=not found,
        detail={
            "violations": found[:MAX_DIFFERENCES_SHOWN],
            "violations_shown": min(len(found), MAX_DIFFERENCES_SHOWN),
            "violations_total": len(found),
            "keywords_enforced": sorted(SUPPORTED_KEYWORDS),
        },
    )


def _needles(values, *, metric: str) -> list[str]:
    """The substring list, or a refusal naming what is wrong with it."""
    if isinstance(values, str):
        raise BadRequest(
            f"`{metric}` takes a list of substrings, and a bare string would "
            f"be read as a list of its characters. Pass [\"{values[:40]}\"] if "
            f"one substring is what you meant."
        )
    try:
        items = list(values)
    except TypeError:
        raise BadRequest(
            f"`{metric}` takes a list of substrings; it was given a "
            f"{type(values).__name__}."
        ) from None
    if not items:
        raise BadRequest(
            f"`{metric}` was given no substrings. An empty list has nothing to "
            f"check, and scoring it 1.0 would put a passing row in the suite "
            f"for a check that never ran."
        )
    if len(items) > MAX_NEEDLES:
        raise BadRequest(
            f"`{metric}` was given {len(items):,} substrings, past the "
            f"{MAX_NEEDLES:,} it will take. Shortening the list here would "
            f"turn a failing check into a passing one."
        )
    for item in items:
        if not isinstance(item, str) or not item:
            raise BadRequest(
                f"every substring must be non-empty text; this list contains a "
                f"{type(item).__name__}"
                + ("" if not isinstance(item, str) else " that is empty")
                + ". An empty substring is inside every string there is, so it "
                "would pass — or fail — everything."
            )
    return [str(i) for i in items]


def contains_all(output, needles, *, normalisation=AS_IS) -> Result:
    """Every required substring appears; scored by the fraction present."""
    norm = resolve_normalisation(normalisation)
    wanted = _needles(needles, metric="contains_all")
    text, why = _as_text(output, metric="contains_all")
    if why:
        return _unmeasurable("contains_all", why, normalisation=norm.describe())

    haystack = norm.apply(text)
    # The needles are normalised too. Not doing so is the bug that makes a
    # case-insensitive check fail on a capitalised needle, and it fails
    # silently, which is worse.
    present = [n for n in wanted if norm.apply(n) in haystack]
    missing = [n for n in wanted if norm.apply(n) not in haystack]
    return _measured(
        "contains_all",
        score=len(present) / len(wanted),
        passed=not missing,
        normalisation=norm.describe(),
        detail={
            "required": wanted,
            "present": present,
            "missing": missing,
            "found": len(present),
            "of": len(wanted),
        },
    )


def contains_none(output, needles, *, normalisation=AS_IS) -> Result:
    """No forbidden substring appears; scored by the fraction absent."""
    norm = resolve_normalisation(normalisation)
    banned = _needles(needles, metric="contains_none")
    text, why = _as_text(output, metric="contains_none")
    if why:
        return _unmeasurable("contains_none", why, normalisation=norm.describe())

    haystack = norm.apply(text)
    hits = [n for n in banned if norm.apply(n) in haystack]
    return _measured(
        "contains_none",
        score=(len(banned) - len(hits)) / len(banned),
        passed=not hits,
        normalisation=norm.describe(),
        detail={
            "forbidden": banned,
            "found": hits,
            "absent": len(banned) - len(hits),
            "of": len(banned),
        },
    )


# ------------------------------------------------------------------ dispatch


def run(name: str, output=None, reference=None, **options) -> Result:
    """One named metric, by name, so a suite can live in a configuration file.

    Every argument is checked against the catalogue entry before the metric
    runs, so an option this metric does not have is a refusal that names the
    ones it does — rather than a TypeError from somewhere inside, or worse, a
    silently ignored keyword that made the check weaker than it was written.
    """
    metric = _metric(name)
    unknown = sorted(set(options) - {n for n, _ in metric.options})
    if unknown:
        allowed = ", ".join(n for n, _ in metric.options) or "none"
        raise BadRequest(
            f"`{name}` has no option(s) named {', '.join(unknown)}. Its "
            f"options are: {allowed}. An ignored option would leave this check "
            f"weaker than the one that was written down."
        )
    if metric.reference and reference is None:
        raise BadRequest(
            f"`{name}` needs {metric.reference}; it was given nothing to "
            f"compare against."
        )
    if not metric.reference and reference is not None:
        raise BadRequest(
            f"`{name}` compares the output against nothing — it asks a "
            f"question about the output alone — so a reference here would be "
            f"silently discarded."
        )

    if name == "exact_match":
        return exact_match(output, reference, **options)
    if name == "numeric_close":
        return numeric_close(output, reference, **options)
    if name == "json_valid":
        return json_valid(output)
    if name == "json_diff":
        return json_diff(output, reference, **options)
    if name == "edit_similarity":
        return edit_similarity(output, reference, **options)
    if name == "regex_match":
        return regex_match(output, reference, **options)
    if name == "schema_conformance":
        return schema_conformance(output, reference)
    if name == "contains_all":
        return contains_all(output, reference, **options)
    return contains_none(output, reference, **options)


# --------------------------------------------------------------- aggregation


@dataclass
class _Tally:
    """Counters over a stream of results. Holds no rows, so a suite of a
    million rows costs the same as a suite of ten."""

    total: int = 0
    measured: int = 0
    judged: int = 0
    passes: int = 0
    score_sum: float = 0.0
    lowest: float | None = None
    highest: float | None = None
    reasons: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)

    def add(self, result: Result) -> None:
        self.total += 1
        self.metrics[result.metric] = self.metrics.get(result.metric, 0) + 1
        if not result.measurable:
            self.reasons[result.reason] = self.reasons.get(result.reason, 0) + 1
            return
        self.measured += 1
        self.score_sum += result.score
        self.lowest = (
            result.score if self.lowest is None else min(self.lowest, result.score)
        )
        self.highest = (
            result.score if self.highest is None else max(self.highest, result.score)
        )
        if result.passed is not None:
            self.judged += 1
            self.passes += int(result.passed)


def aggregate(results, *, name: str = "") -> dict:
    """Totals over many rows, with the unmeasurable ones OUT of the denominator.

    Three denominators, reported separately, because they answer three
    questions and collapsing them is the arithmetic this module exists to
    prevent:

      `rows`        every row submitted
      `measured`    rows a score exists for — the denominator of `mean_score`
      `judged`      rows a verdict exists for — the denominator of `pass_rate`

    `judged` is smaller than `measured` whenever a graded metric ran without a
    stated threshold, and both are smaller than `rows` whenever something could
    not be measured. Every one of those gaps is reported rather than closed.
    """
    tally = _Tally()
    for result in results:
        if not isinstance(result, Result):
            raise BadRequest(
                f"aggregate takes Result objects; it was given a "
                f"{type(result).__name__}."
            )
        tally.add(result)
    return _summarise(tally, name=name)


def _summarise(tally: _Tally, *, name: str) -> dict:
    if not tally.total:
        raise BadRequest(
            "no rows were scored, so there is nothing to aggregate. An empty "
            "suite is not a passing suite — it is a run that did not happen."
        )
    unmeasurable = tally.total - tally.measured
    mean = tally.score_sum / tally.measured if tally.measured else None
    rate = tally.passes / tally.judged if tally.judged else None
    reasons = sorted(tally.reasons.items(), key=lambda kv: (-kv[1], kv[0]))
    return {
        "name": name,
        "rows": tally.total,
        "measured": tally.measured,
        "unmeasurable": unmeasurable,
        "judged": tally.judged,
        "passed": tally.passes,
        "failed": tally.judged - tally.passes,
        # None, never 0.0, when nothing was measured. A suite where every row
        # refused has no score, and 0.0 would read as a total failure.
        "mean_score": mean,
        "lowest_score": tally.lowest,
        "highest_score": tally.highest,
        "pass_rate": rate,
        "unmeasurable_reasons": [
            {"reason": reason, "rows": count} for reason, count in reasons
        ],
        "metrics": dict(sorted(tally.metrics.items())),
        "means": _aggregate_means(
            tally=tally,
            name=name,
            unmeasurable=unmeasurable,
            mean=mean,
            rate=rate,
            reasons=reasons,
        ),
    }


def _aggregate_means(*, tally, name, unmeasurable, mean, rate, reasons) -> str:
    subject = name or "This suite"
    if not tally.measured:
        return (
            f"{subject}: NOTHING WAS MEASURED. All {tally.total} row(s) came "
            f"back unmeasurable, so there is no score and no pass rate here — "
            f"reporting 0% would be publishing a result nobody obtained. The "
            f"commonest reason was: {reasons[0][0]}"
        )

    if rate is None:
        head = (
            f"{subject}: {tally.measured} of {tally.total} row(s) were "
            f"measured, and NONE of them carries a verdict — every metric that "
            f"ran was graded without a stated threshold, so there is a mean "
            f"score of {mean:.4g} and no pass rate."
        )
    else:
        head = (
            f"{subject}: {tally.passes} of {tally.judged} judged row(s) "
            f"passed, a pass rate of {rate:.1%}. That denominator is the rows "
            f"that could be JUDGED, not the {tally.total} submitted."
        )

    excluded = (
        (
            f" {unmeasurable} row(s) could not be measured and are excluded "
            f"from every denominator above: an unmeasurable row is not a "
            f"failure, and counting it as one would move this number without "
            f"anything having been measured. The commonest reason "
            f"({reasons[0][1]} row(s)) was: {reasons[0][0]}"
        )
        if unmeasurable
        else " Every row was measurable."
    )
    ungraded = (
        (
            f" {tally.measured - tally.judged} measured row(s) carry a score "
            f"and no verdict, because a graded metric ran without a stated "
            f"threshold; they are in the mean and not in the pass rate."
        )
        if rate is not None and tally.judged < tally.measured
        else ""
    )
    spread = (
        f" Mean score {mean:.4g} over {tally.measured} measured row(s), "
        f"ranging {tally.lowest:.4g} to {tally.highest:.4g}."
    )
    mixed = (
        (
            f" These rows mix {len(tally.metrics)} different metrics "
            f"({', '.join(sorted(tally.metrics))}), and the mean above averages "
            f"across them — a graded similarity and a binary match are not on "
            f"the same scale, so read the per-metric counts rather than the one "
            f"number."
        )
        if len(tally.metrics) > 1
        else ""
    )
    return (
        f"{head}{excluded}{ungraded}{spread}{mixed} These metrics are "
        f"deterministic — the same strings score the same next Tuesday — so "
        f"all of the variance lives in the outputs that were fed in. A rate "
        f"over one generation is a sample of the model, not a property of it."
    )


def score_rows(
    name: str,
    outputs,
    *,
    references=None,
    reference=None,
    sample: int = DEFAULT_ROW_SAMPLE,
    **options,
) -> dict:
    """One named metric over many rows, aggregated, streaming.

    `reference` is one comparison shared by every row — a pattern, a schema, a
    banned-phrase list. `references` is one per row, and its length must match,
    because a suite quietly zipped to the shorter of the two would score a
    subset of its rows and report the total.

    Rows are consumed one at a time and only the counters are kept, so this
    costs the same on a million rows as on ten. At most `sample` row
    dictionaries are returned for display, and how many were shown against how
    many there were is in the response.
    """
    _metric(name)
    if references is not None and reference is not None:
        raise BadRequest(
            "pass `reference` for one comparison shared by every row, or "
            "`references` for one per row — not both, because nothing here can "
            "tell which of the two you meant to apply."
        )
    if isinstance(sample, bool) or not isinstance(sample, int) or sample < 0:
        raise BadRequest("sample must be a whole number of rows, or 0 for none.")

    per_row = list(references) if references is not None else None
    if per_row is not None and hasattr(outputs, "__len__"):
        if len(outputs) != len(per_row):
            raise BadRequest(
                f"there are {len(outputs):,} outputs and {len(per_row):,} "
                f"references. Zipping them would score the shorter list and "
                f"report the total, so this refuses instead."
            )

    tally = _Tally()
    shown: list[dict] = []
    index = 0
    for index, output in enumerate(outputs):
        if per_row is not None and index >= len(per_row):
            raise BadRequest(
                f"there are more outputs than references — the {index + 1:,}th "
                f"output has nothing to compare against."
            )
        this = per_row[index] if per_row is not None else reference
        result = run(name, output, this, **options)
        tally.add(result)
        if len(shown) < sample:
            shown.append(result.to_dict())
    if per_row is not None and tally.total != len(per_row):
        raise BadRequest(
            f"there are {tally.total:,} outputs and {len(per_row):,} "
            f"references, so {len(per_row) - tally.total:,} reference(s) were "
            f"never used."
        )

    summary = _summarise(tally, name=name)
    summary["rows_shown"] = len(shown)
    summary["rows_total"] = tally.total
    summary["row_results"] = shown
    if len(shown) < tally.total:
        summary["means"] += (
            f" {len(shown):,} of {tally.total:,} row results are included "
            f"below; the totals above are over all {tally.total:,}."
        )
    return summary
