"""The set of cases you care about, and two runs of it side by side.

`sweep` runs one measurement over many prompts and reports the distribution.
`mri_diff` compares two `.mri` of the SAME prompt. Neither answers the question
somebody asks after every edit — **did this help on the forty cases I care
about** — because answering it needs the set to be a stored thing with an
identity, and each run of it to be a stored thing that can be joined back to
that identity. A dataset and an experiment.

Every observability platform in this category has both. This project had
neither, and so had no way to say that a change helped 31 cases, hurt 4, and
could not be measured on 5.

## What a row holds, and why that is the whole point

Every competitor's experiment row is an output and a score. A regression there
reads `faithfulness 0.71 -> 0.63`, and the next question — WHY — starts a fresh
investigation from nothing, usually by re-running the case by hand.

A row here carries the output, the scores, the RECEIPT that says what produced
them, and the INTERNALS the measurement saw: which heads ranked top, which
patching sites carried signal and with what sign. So a regression row can say
that the top-5 head ranking changed and that `L6.resid` flipped sign, which is
a lead rather than a number to be unhappy about.

Internals are compared STRUCTURALLY, never through a table of known key names.
A list of names is a ranking and is compared as one; a mapping of names to
signed numbers is a set of sites and is compared for sign flips and for keys
that appeared or vanished. `imaging.py` was written this way for the same
reason: a name map stops working the week somebody adds a panel, and it fails
by quietly comparing nothing.

## The denominator

`sweep` enforces the rule this file inherits: a case that could not be measured
is a ROW carrying the sentence, never a gap. The join has to keep it. So the
denominator of a comparison is the UNION of the case ids on both sides — a case
one run never wrote a row for is `unmeasurable`, with a sentence naming the run
that lacked it, and the four counts always sum to the union. An intersection
would report "38 unchanged, 0 worse" about a run that died after case 38 of 40,
and that report is indistinguishable from a clean one.

## No verdict

Counts and deltas. There is no aggregate score in this module and there should
never be one: a single number over forty heterogeneous cases is precisely what
makes a regression invisible, because two cases collapsing and three improving
slightly average out to fine.

The deltas are summarised as median, IQR and range with their own `n`,
following `sweep`'s rule that there is never a mean without a spread. That `n`
is the rows measured on BOTH sides, which is not the denominator, and it is
labelled so wherever it appears.

## Which direction is better

Stated by the caller, never inferred. KL divergence is better lower and
faithfulness is better higher; a module that guessed would silently invert
every conclusion in half of all comparisons, and the output would look right.
There is no default and there is no name map of metrics either.

## The floor

Not invented here. `mri_diff` takes its floor from the quantisation scale the
files themselves carry; an experiment file carries whatever floor the person
who ran it stated for each metric, and a comparison uses the COARSER of the
two, because two files cannot be compared more finely than the coarser of them
can represent. When neither states one and the caller gives none, the
comparison is exact and SAYS it is exact — a difference in the last binary
digit counting as "better" is a fact about float arithmetic rather than about
the model, and the reader has to be told which they are looking at.

## Reference outputs are optional, and their absence is visible

Plenty of useful sets have no expected output: you are comparing two runs
against each other, not against an answer key. So `reference` is `None` when
absent and never `""`, the two are different datasets by fingerprint, and the
comparison reports how many rows had one. With no dataset supplied to resolve
them that count is `None` rather than 0 — "this comparison did not look" and
"the dataset has none" are different answers, and only one of them means
something about the data.

## Plain files, no server

JSONL on disk with a header line carrying the schema version, so a dataset is a
file you commit next to the code it tests and read with `head`. No account, no
hosted store, no network. The version is checked on read, and a file from a
newer ModelMRI is refused by name rather than parsed hopefully.

Torch-free, and the only I/O is two readers and two writers, so the arithmetic
here is checkable in CI in milliseconds on a machine with no accelerator.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import receipts
from .errors import BadRequest, Refusal

# Bumped when a field CHANGES MEANING, not when one is added. A reader of
# version 1 skips fields it does not know, which is safe for additions and
# catastrophic for redefinitions -- so the version is the promise that
# `scores` still means what it meant, and adding `internals` did not need it.
SCHEMA_VERSION = 1

DATASET = "dataset"
EXPERIMENT = "experiment"

# The four outcomes a case can have in a comparison. `unmeasurable` is a
# first-class member rather than an error state: it is what a crashed run, a
# refused measurement and a metric nobody recorded all collapse to, and each
# of those carries its own sentence saying which it was.
BETTER = "better"
WORSE = "worse"
UNCHANGED = "unchanged"
UNMEASURABLE = "unmeasurable"

# Internals findings borrow `mri_diff`'s three words on purpose: the same three
# answers, about the same kind of question, should read the same way.
SAME = "same"
CHANGED = "changed"
NOT_COMPARABLE = "not comparable"

# How deep into a ranking counts as "the top". Five because that is what the
# panels show and what a reader means by "the heads that matter"; it travels in
# every finding so a different reader can disagree with it.
TOP_K = 5

# How many names to print before "and N more". A sentence listing forty head
# ids is not read by anybody.
NAMED = 6

# `meta["source"]` for an experiment converted out of somebody else's eval log.
INSPECT = "inspect"

# Inspect's own correct/incorrect markers, and the ONLY two string score values
# this module has a number for.
#
# The problem they solve: Inspect's canonical score value is the string "C" or
# "I", and `_score_of` refuses a string -- correctly, because there is no way
# to order one against another. An experiment written straight out of an eval
# log therefore compared as 100% `unmeasurable`, and it did not even refuse
# early, because the metric-present gate checks score KEYS: `match` looked
# present and every row then silently degraded.
#
# So the number arrives under its OWN name, `<scorer>_correct`, beside the
# marker the log actually wrote. Rewriting `match` from "C" to 1.0 in place
# would make the file claim a number nobody recorded, which is the fabrication
# this project refuses everywhere else; adding a separately named column, with
# this mapping stated in the file's own `meta`, is a transcription of a
# marker Inspect defines and `inspect_io._failed` already reads in production.
#
# P (partial) and N (no answer) are deliberately absent. Deciding what partial
# credit is worth is a judgement, and one invented here would be
# indistinguishable from one somebody made.
SCORE_MARKERS = {"C": 1.0, "I": 0.0}

# The suffix that names a marker's number. A scorer in the log that already
# owns the name keeps it -- see `_inspect_scores`.
DERIVED_SUFFIX = "_correct"


class DifferentDatasets(Refusal):
    """These two runs did not cover the same cases — or cannot be shown to.

    Its own class because the caller's correct response is unlike any other
    refusal here: not "measure something else" but "you are holding the wrong
    pair of files". Comparing row 12 of one set against row 12 of another
    produces a table of real numbers about nothing.
    """


# --------------------------------------------------------------------- cases


@dataclass
class Case:
    """One input, with the identity that joins it to a result in every run."""

    case_id: str
    input_text: str
    # `None` means no expected output was recorded, which is the ordinary state
    # for a set you run to compare two versions against EACH OTHER. `""` would
    # mean the expected output is the empty string, which is a different and
    # much stranger claim, and the fingerprint below tells them apart.
    reference: str | None = None
    tags: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Dataset:
    """A stored set of inputs, with an identity that survives being written."""

    name: str
    cases: list[Case] = field(default_factory=list)
    description: str = ""
    created_at: str = ""
    # Set by the reader when the file's own header disagrees with what was in
    # it. Empty for a file this module wrote and finished writing.
    truncated: str = ""
    edited: str = ""

    @property
    def n_references(self) -> int:
        """How many cases carry an expected output. Never inferred from 0."""
        return sum(1 for c in self.cases if c.reference is not None)

    def fingerprint(self) -> str:
        """A content hash of what this set actually asks, order-independent.

        This is the whole mechanism behind refusing to compare two experiments
        over different data, so what goes into it is a decision rather than a
        detail:

        - The case ids, because that is what a result joins on.
        - The inputs, because a set whose prompt was edited is a different set
          even under the same ids.
        - The references, because a score computed against one answer key is
          not comparable to a score computed against another.

        Sorted by id, so reordering the file does not change which dataset it
        is. `None` and `""` serialise differently, which is exactly the
        distinction this project keeps insisting on.
        """
        material = json.dumps(
            [
                [c.case_id, c.input_text, c.reference]
                for c in sorted(self.cases, key=lambda c: c.case_id)
            ],
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        return receipts.digest(material)

    def validated(self) -> Dataset:
        """Refuse a set that cannot be joined, before anything is run on it."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise BadRequest(
                "this dataset has no name. The name is what an experiment "
                "records having run on, so a nameless set produces results "
                "nothing can be matched back to."
            )
        if not self.cases:
            raise BadRequest(
                "this dataset has no cases in it. An empty set is not a run "
                "where everything passed — there is nothing here to run."
            )
        seen: dict[str, int] = {}
        for i, case in enumerate(self.cases):
            if not isinstance(case.case_id, str) or not case.case_id.strip():
                raise BadRequest(
                    f"case {i} has no id. The id is what joins this input to "
                    f"its result in every experiment, so a case without one "
                    f"cannot be compared across runs."
                )
            if not isinstance(case.input_text, str):
                raise BadRequest(
                    f"case {case.case_id} has an input that is not text "
                    f"({type(case.input_text).__name__}). This module stores "
                    f"inputs as JSON strings."
                )
            if case.reference is not None and not isinstance(case.reference, str):
                raise BadRequest(
                    f"case {case.case_id} has a reference output that is not "
                    f"text ({type(case.reference).__name__}). Use `None` for "
                    f"no reference; `None` and a value are the two states this "
                    f"reads."
                )
            if case.case_id in seen:
                raise BadRequest(
                    f"case id {case.case_id!r} appears at positions "
                    f"{seen[case.case_id]} and {i}. A result joins on the id, "
                    f"so two cases sharing one cannot be told apart: the "
                    f"denominator would count them twice and the comparison "
                    f"once. Give them distinct ids."
                )
            seen[case.case_id] = i
        return self

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "fingerprint": self.fingerprint(),
            "n_cases": len(self.cases),
            "n_references": self.n_references,
            "cases": [c.to_dict() for c in self.cases],
            "truncated": self.truncated,
            "edited": self.edited,
            "means": self.means(),
        }

    def means(self) -> str:
        missing = len(self.cases) - self.n_references
        refs = (
            f"{self.n_references} of {len(self.cases)} cases carry an expected "
            f"output; {missing} do not, and those {missing} can only be "
            f"compared between runs rather than against an answer."
            if self.n_references
            else (
                "No case here carries an expected output, which is an "
                "ordinary state: this set compares two runs against each "
                "other rather than against an answer key."
            )
        )
        trouble = " ".join(s for s in (self.truncated, self.edited) if s)
        return (
            f"{self.name}: {len(self.cases)} cases, fingerprint "
            f"{self.fingerprint()}. {refs} Two experiments are only compared "
            f"when they record this same fingerprint, so editing an input or a "
            f"reference here makes every earlier run incomparable — which is "
            f"the point, because a score against a changed answer key is a "
            f"different measurement." + (f" {trouble}" if trouble else "")
        )


def from_inputs(
    name: str,
    inputs: list[str],
    *,
    references: list[str | None] | None = None,
    description: str = "",
) -> Dataset:
    """A dataset from plain strings, with content-addressed ids.

    The id is a hash of the input, so the same input in two datasets is the
    same case and a set can be rebuilt from the same file and still join.

    A repeated input is REFUSED naming both positions rather than deduplicated
    or suffixed. Two identical inputs hash to one id, so keeping both would put
    two cases in the denominator and one in the comparison; dropping one
    silently would change the size of the set somebody wrote. Somebody who
    genuinely wants the same input twice — to look at sampling variance, say —
    is asking for two distinct cases and should give them distinct ids.
    """
    if references is not None and len(references) != len(inputs):
        raise BadRequest(
            f"{len(inputs)} inputs were given and {len(references)} reference "
            f"outputs. They are paired by position, so a mismatch means at "
            f"least one case would be scored against another case's answer. "
            f"Pass `None` in the list for a case that has no reference."
        )
    seen: dict[str, int] = {}
    cases: list[Case] = []
    for i, text in enumerate(inputs):
        if not isinstance(text, str):
            raise BadRequest(
                f"input {i} is a {type(text).__name__}, not text. This module "
                f"stores inputs as JSON strings."
            )
        case_id = receipts.digest(text)
        if case_id in seen:
            raise BadRequest(
                f"inputs {seen[case_id]} and {i} are identical, so both hash "
                f"to case id {case_id}. One id cannot carry two rows: the "
                f"denominator would count both and the comparison one. Remove "
                f"the repeat, or build the cases yourself with distinct ids."
            )
        seen[case_id] = i
        cases.append(
            Case(
                case_id=case_id,
                input_text=text,
                reference=references[i] if references is not None else None,
            )
        )
    return Dataset(name=name, cases=cases, description=description).validated()


# ------------------------------------------------------------ from a recording


# Step kinds whose `input` is a prompt somebody would want to re-run. A tool
# call's input is arguments and a user turn's is a message; neither is the
# thing a dataset case holds.
PROMPT_KINDS = ("llm_call",)


def from_traces(
    traces,
    *,
    name: str,
    only_errors: bool = False,
    description: str = "",
) -> tuple:
    """A dataset built from runs that were recorded, and what was left out.

    The loop this closes: a recorded failure currently leaves the tool. You
    watch an agent go wrong, and there is no way to turn that into a case you
    re-run after changing something. Curation needs no model at all, which is
    why it can happen offline here.

    Returns `(dataset, report)`. The report is not optional decoration —
    nothing is dropped silently, and three things genuinely can be:

      * a trace with no prompt-bearing step, which cannot become a case
      * a trace whose prompt is identical to an earlier one, because
        `from_inputs` hashes the input for the case id and one id cannot
        carry two rows
      * every non-error trace, when `only_errors` is set

    ## What it will NOT do

    It does not name the failure mode, and it does not write an expected
    answer. The row is EVIDENCE — the input that produced a run somebody
    thought was wrong. Deciding what the right answer was is a judgement, and
    a judgement invented here would be indistinguishable from one somebody
    made, which is the fabrication this project refuses everywhere else.
    Every case comes back with `reference: None` and it is for a human to
    fill in.

    `traces` are already-fetched documents. This module never opens a store —
    the same rule `trajectory.align` follows, and what lets both be tested
    without one.
    """
    if not isinstance(name, str) or not name.strip():
        raise BadRequest(
            "a dataset needs a name. It is what an experiment records having "
            "run against, so an unnamed one cannot be compared to anything."
        )

    inputs: list[str] = []
    kept: list[str] = []
    seen: dict[str, str] = {}
    skipped: list[dict] = []

    for doc in traces or []:
        if not isinstance(doc, dict):
            raise BadRequest(
                f"a trace has to be a recorded document, not a {type(doc).__name__}."
            )
        trace_id = str(doc.get("id") or "")
        steps = doc.get("steps") or []
        failed = any(s.get("error") for s in steps if isinstance(s, dict))

        if only_errors and not failed:
            skipped.append(
                {
                    "trace_id": trace_id,
                    "why": "no step in it recorded an error, and only failures "
                    "were asked for",
                }
            )
            continue

        text = _prompt_of(steps)
        if text is None:
            skipped.append(
                {
                    "trace_id": trace_id,
                    "why": (
                        f"no step of a kind that carries a prompt "
                        f"({', '.join(PROMPT_KINDS)}) — a tool call's input is "
                        f"arguments, not a case"
                    ),
                }
            )
            continue

        digest = receipts.digest(text)
        if digest in seen:
            # Reported with BOTH ids rather than deduplicated quietly: two
            # runs of the same prompt is a real thing somebody may have meant,
            # and they need to know which recording is now standing for both.
            skipped.append(
                {
                    "trace_id": trace_id,
                    "why": (
                        f"its prompt is identical to trace {seen[digest]}, "
                        f"which is already case {digest}. One case id cannot "
                        f"carry two rows"
                    ),
                }
            )
            continue

        seen[digest] = trace_id
        inputs.append(text)
        kept.append(trace_id)

    if not inputs:
        # `Dataset.validated()` refuses an empty set, and rightly — but its
        # sentence is "there is nothing here to run", which throws away the
        # one thing the caller needs. Nothing surviving is the MOST
        # informative outcome of this function, and the reasons are the
        # answer: every recording was a duplicate, or none carried a prompt,
        # or the error filter excluded them all.
        reasons = "; ".join(
            f"{s['trace_id'] or 'an unnamed run'}: {s['why']}" for s in skipped
        )
        raise BadRequest(
            f"none of the {len(traces or [])} recorded run(s) could become a "
            f"case, so there is no dataset to build. "
            + (reasons or "nothing was selected to build one from.")
        )

    dataset = from_inputs(name.strip(), inputs, description=description)
    return dataset, {
        "kept": kept,
        "skipped": skipped,
        "n_seen": len(traces or []),
        "means": _from_traces_means(len(traces or []), kept, skipped, only_errors),
    }


def _prompt_of(steps) -> str | None:
    """The first prompt-bearing step carrying TEXT, or `None` if there is none.

    First rather than longest or last: a run's first `llm_call` is the one the
    rest of it followed from, and picking by any other rule would be choosing
    which prompt the case is about by accident.

    "Carrying text" is the part the name leaves out, and it is a real
    difference: a step whose `input` is absent, blank or not a string is
    skipped and the search continues, so on a recording whose first `llm_call`
    logged nothing the case is about the SECOND one. That is the right
    behaviour — an empty prompt is not a case, and stopping at it would turn
    every such recording into a skip — but it is not what "the first" says on
    its own, and a reader trusting the shorter sentence would expect `None`.
    """
    for step in steps:
        if not isinstance(step, dict):
            continue
        if step.get("kind") not in PROMPT_KINDS:
            continue
        text = step.get("input")
        if isinstance(text, str) and text.strip():
            return text
    return None


def _from_traces_means(
    n_seen: int, kept: list, skipped: list, only_errors: bool
) -> str:
    which = "failed runs" if only_errors else "runs"
    head = (
        f"{len(kept)} case(s) from {n_seen} recorded {which}, "
        f"{len(skipped)} left out and each one says why."
    )
    return (
        f"{head} Every case carries the INPUT and no expected answer: the row "
        f"is evidence that a run happened, and deciding what the right answer "
        f"was is a judgement. One invented here would be indistinguishable "
        f"from one you made."
    )


# --------------------------------------------------------------- experiments


@dataclass
class Result:
    """One case's outcome in one run. Written whether or not it worked."""

    case_id: str
    # `None` is "this run produced no output", which is not "the output was
    # the empty string". A refused row has None here and a sentence below.
    output: str | None = None
    # Named metrics, plural, because a single number per row is the shape that
    # hides a regression. Nothing here blends them.
    scores: dict = field(default_factory=dict)
    # The sentence saying why this row has no scores. Empty when it does. NOT a
    # bool: "it failed" is not actionable and this always is. Same field name
    # and same rule as `sweep.Row`.
    could_not_measure: str = ""
    # What produced the numbers -- `receipts.Receipt.to_dict()`.
    receipt: dict = field(default_factory=dict)
    # What the measurement SAW: rankings, signed sites. This is the field the
    # rest of the category does not have, and `compare_internals` reads it
    # structurally rather than by key name.
    internals: dict = field(default_factory=dict)

    @property
    def measured(self) -> bool:
        return not self.could_not_measure

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Experiment:
    """One run of one dataset: one row per case, refusals included."""

    name: str
    dataset_name: str
    # The dataset's content hash at the moment this ran. Empty means the run
    # did not record it, and a comparison refuses rather than assuming.
    dataset_fingerprint: str = ""
    results: list[Result] = field(default_factory=list)
    # What was under test — a model id, a commit, a prompt version. Free text,
    # because this module cannot know what somebody changed and inventing a
    # schema for it would only be guessed at.
    label: str = ""
    started_at: str = ""
    # Smallest meaningful difference per metric, stated by whoever ran it. Not
    # derivable from the file: a score is a float somebody computed and the
    # file cannot know its precision.
    metric_floors: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    truncated: str = ""

    @property
    def n_measured(self) -> int:
        return sum(1 for r in self.results if r.measured)

    def validated(self) -> Experiment:
        if not isinstance(self.name, str) or not self.name.strip():
            raise BadRequest(
                "this experiment has no name. Two runs are told apart by name "
                "in every sentence a comparison writes."
            )
        seen: dict[str, int] = {}
        for i, row in enumerate(self.results):
            if not isinstance(row.case_id, str) or not row.case_id.strip():
                raise BadRequest(
                    f"result {i} names no case, so nothing can join it back to "
                    f"an input."
                )
            if row.case_id in seen:
                raise BadRequest(
                    f"case {row.case_id!r} has results at positions "
                    f"{seen[row.case_id]} and {i}. One run produces one row "
                    f"per case; two means the comparison would silently pick "
                    f"whichever it read last."
                )
            seen[row.case_id] = i
        return self

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "dataset_name": self.dataset_name,
            "dataset_fingerprint": self.dataset_fingerprint,
            "started_at": self.started_at,
            "n_results": len(self.results),
            "n_measured": self.n_measured,
            "n_could_not_measure": len(self.results) - self.n_measured,
            "metric_floors": dict(self.metric_floors),
            "meta": dict(self.meta),
            "results": [r.to_dict() for r in self.results],
            "truncated": self.truncated,
        }


# ------------------------------------------------------------------ the files


def _dump(obj: dict) -> str:
    """One JSON line, with NaN refused rather than written as `NaN`.

    `json.dumps` writes bare `NaN` by default, which is not JSON and which
    every strict reader rejects — so a file written that way is unreadable by
    anything but Python, and the failure surfaces days later on somebody
    else's machine.
    """
    return json.dumps(obj, ensure_ascii=False, allow_nan=False)


def write_dataset(dataset: Dataset, path: str | Path) -> Path:
    """Header line, then one case per line. The header carries the version.

    A header line rather than a version stamped on every row: one file has one
    schema, and repeating it per row invites a file that carries two.

    `created_at` is written EXACTLY as the dataset states it, empty included.
    It used to fall back to this machine's clock, which meant a set converted
    out of somebody else's eval log -- the one caller that has a real date and
    can fail to have one -- came back dated to the afternoon it was imported,
    with nothing in the file saying the date was invented here. A writer is
    not a producer: it does not know when the thing it is writing happened,
    and an unknown that reads as a recorded value is worse than a blank.
    """
    from . import __version__

    dataset = dataset.validated()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "modelmri_schema": SCHEMA_VERSION,
        "kind": DATASET,
        "name": dataset.name,
        "description": dataset.description,
        "created_at": dataset.created_at,
        "n_cases": len(dataset.cases),
        "n_references": dataset.n_references,
        "fingerprint": dataset.fingerprint(),
        "tool_version": __version__,
    }
    with target.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(_dump(header) + "\n")
        for case in dataset.cases:
            fh.write(_dump(case.to_dict()) + "\n")
    return target


def write_experiment(experiment: Experiment, path: str | Path) -> Path:
    """Header line, then one result per line, refusals included.

    JSONL rather than one JSON array for the reason `sweep.write_jsonl` gives:
    a run killed half way leaves a file whose complete lines still read. The
    header's `n_results` is what lets the reader SAY it was killed rather than
    hand back a shorter complete-looking run.

    Two fields here are written exactly as the experiment states them, empty
    included, and both used to be filled in by this function instead:

    `started_at` no longer falls back to now. See `write_dataset` -- a writer
    is not a producer and does not know when the run it is writing happened.

    `truncated` is written at all. Every reader in this module SETS it and no
    writer carried it, so a gap somebody had already been told about died in
    the file: `read_experiment` recomputes the field from `n_results` against
    the rows under it, and those agree -- 3 declared, 3 written -- so a run
    three rows into a six-sample eval came back looking whole, and
    `compare_experiments`, which builds its notes from `before.truncated` and
    `after.truncated`, had nothing to say about it. The two gaps are
    different facts and the reader below carries both: this one is what the
    READER left behind, `_truncation` is what the WRITER never finished.
    """
    from . import __version__

    experiment = experiment.validated()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "modelmri_schema": SCHEMA_VERSION,
        "kind": EXPERIMENT,
        "name": experiment.name,
        "label": experiment.label,
        "dataset_name": experiment.dataset_name,
        "dataset_fingerprint": experiment.dataset_fingerprint,
        "started_at": experiment.started_at,
        "n_results": len(experiment.results),
        "n_measured": experiment.n_measured,
        "metric_floors": dict(experiment.metric_floors),
        "meta": dict(experiment.meta),
        "truncated": experiment.truncated,
        "tool_version": __version__,
    }
    with target.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(_dump(header) + "\n")
        for row in experiment.results:
            try:
                fh.write(_dump(row.to_dict()) + "\n")
            except ValueError:
                # `allow_nan=False` raises on a NaN or an infinity anywhere in
                # the row. Re-raised naming the CASE, because the default
                # message says "Out of range float values are not JSON
                # compliant" and leaves somebody to find which of forty rows
                # it meant. A NaN score is a real thing a metric produces; it
                # belongs in `could_not_measure` with a sentence, not in a
                # scores dict where it will sort as a number.
                raise BadRequest(
                    f"case {row.case_id} carries a value that is not a finite "
                    f"number (a NaN or an infinity), which is not JSON and "
                    f"cannot be ordered against anything. Record it as "
                    f"`could_not_measure` with the sentence saying why the "
                    f"metric came back non-finite."
                ) from None
    return target


def _read_header(fh, name: str, *, expect: str) -> dict:
    """The first line, checked before anything under it is trusted."""
    line = fh.readline()
    if not line:
        raise BadRequest(
            f"{name} is empty. An empty file is not a dataset where every "
            f"case passed — it is a file nothing was ever written to."
        )
    if not line.strip():
        raise BadRequest(
            f"{name} begins with a blank line where its header should be, so "
            f"nothing here knows what schema the rows under it are in."
        )
    try:
        header = json.loads(line)
    except json.JSONDecodeError:
        raise BadRequest(
            f"{name} line 1 is not JSON, so it carries no header. Every file "
            f"this module writes opens with one naming the schema version and "
            f"the kind."
        ) from None
    if not isinstance(header, dict):
        raise BadRequest(f"{name} line 1 is JSON but not a set of fields.")

    version = header.get("modelmri_schema")
    # bool is an int in Python, and `"modelmri_schema": true` must not read as
    # version 1. Every int check in this file carries the same guard.
    if not isinstance(version, int) or isinstance(version, bool):
        raise BadRequest(
            f"{name} states no schema version on its header line. Guessing "
            f"that it is version {SCHEMA_VERSION} would be reading unknown "
            f"fields as known ones, which is how a row silently loses the "
            f"field that made it interesting."
        )
    if version < 1:
        raise BadRequest(
            f"{name} states schema version {version}, which is not a version "
            f"anything ever wrote."
        )
    if version > SCHEMA_VERSION:
        raise BadRequest(
            f"{name} was written by a newer ModelMRI (schema {version}; this "
            f"one reads {SCHEMA_VERSION}). Reading it under the older rules "
            f"would ignore whatever the newer version added, without saying "
            f"so. Upgrade modelmri to open it."
        )

    kind = header.get("kind")
    if kind != expect:
        found = kind if isinstance(kind, str) and kind else "nothing"
        raise BadRequest(
            f"{name} says it is a {found} file and this is reading it as "
            f"a {expect}. The two carry different rows, and reading one as "
            f"the other produces a set of cases with no results or the "
            f"reverse."
        )
    return header


def _read_line(line: str, name: str, number: int) -> dict | None:
    """One row, or `None` for a blank line. A blank line is not a row."""
    if not line.strip():
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        raise BadRequest(
            f"{name} line {number} is not JSON. A file with an unreadable row "
            f"is not a file with one fewer row — that row was a case somebody "
            f"meant to measure."
        ) from None
    if not isinstance(obj, dict):
        raise BadRequest(f"{name} line {number} is JSON but not a set of fields.")
    return obj


def _truncation(name: str, declared, actual: int, noun: str) -> str:
    """Does the header's count agree with what was there? A sentence, or "".

    The whole reason the count is in the header. A run killed at case 31 of 40
    leaves a file that reads perfectly as a complete 31-case run, and every
    percentage computed from it is right about a denominator nobody chose.
    """
    if not isinstance(declared, int) or isinstance(declared, bool) or declared < 0:
        return (
            f"{name}'s header does not say how many {noun} it should hold, so "
            f"nothing here can tell whether it is complete. A run killed part "
            f"way through leaves a file that reads as a shorter finished one."
        )
    if declared > actual:
        return (
            f"{name} declares {declared} {noun} in its header and {actual} "
            f"were read, so {declared - actual} are missing: this file was "
            f"not finished being written. The {actual} it carries are real "
            f"measurements; the {declared - actual} it does not are unknown, "
            f"not zero."
        )
    if actual > declared:
        return (
            f"{name} declares {declared} {noun} in its header and {actual} "
            f"were read, so {actual - declared} were appended after it was "
            f"written. Nothing here knows what wrote them or under which "
            f"schema."
        )
    return ""


def read_dataset(path: str | Path) -> Dataset:
    """Read a dataset file. Refuses a version it cannot read, by name."""
    p = Path(path)
    cases: list[Case] = []
    try:
        with p.open("r", encoding="utf-8") as fh:
            header = _read_header(fh, p.name, expect=DATASET)
            # Iterated line by line rather than through `read_text().split`,
            # which would hold the file and the split copy at once. A set of
            # ten thousand prompts is not large, but nothing here needs to be
            # the reason a machine chosen for 8 GB of VRAM starts swapping.
            for number, line in enumerate(fh, start=2):
                obj = _read_line(line, p.name, number)
                if obj is None:
                    continue
                cases.append(_case_from(obj, p.name, number))
    except UnicodeDecodeError:
        # NOT an OSError, so the arm below never saw it. `UnicodeDecodeError`
        # is a ValueError, and a binary file handed to a route expecting JSONL
        # therefore escaped as a 500 — while a MISSING file, one line away,
        # answered with a sentence. Same reader, two shapes of wrong file, two
        # completely different experiences.
        raise BadRequest(
            f"{p.name} is not text. A dataset or an experiment is JSONL — one "
            f"JSON object per line, UTF-8 — and this file has bytes in it that "
            f"are not. If it is a `.mri` or an archive, it belongs to a "
            f"different reader."
        ) from None
    except OSError as err:
        raise BadRequest(
            f"{p.name} could not be read ({err.strerror or type(err).__name__})"
        ) from None

    dataset = Dataset(
        name=str(header.get("name") or p.stem),
        cases=cases,
        description=str(header.get("description") or ""),
        created_at=str(header.get("created_at") or ""),
    ).validated()
    dataset.truncated = _truncation(p.name, header.get("n_cases"), len(cases), "cases")

    # The fingerprint is COMPUTED from the cases and never trusted from the
    # header, so a hand-edited file cannot keep claiming to be the set the
    # experiments ran on. The header's copy is there for whoever reads the
    # file with `head`, and a disagreement between the two is itself a finding.
    stamped = header.get("fingerprint")
    if isinstance(stamped, str) and stamped and stamped != dataset.fingerprint():
        dataset.edited = (
            f"{p.name}'s header records fingerprint {stamped} and its cases "
            f"hash to {dataset.fingerprint()}, so it was edited after it was "
            f"written. Experiments that ran against {stamped} do not describe "
            f"these cases."
        )
    return dataset


def _case_from(obj: dict, name: str, number: int) -> Case:
    case_id = obj.get("case_id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise BadRequest(f"{name} line {number} has no `case_id`.")
    text = obj.get("input_text")
    if not isinstance(text, str):
        raise BadRequest(
            f"{name} line {number} (case {case_id}) has no `input_text` string."
        )
    reference = obj.get("reference")
    if reference is not None and not isinstance(reference, str):
        raise BadRequest(
            f"{name} line {number} (case {case_id}) has a reference that is "
            f"neither text nor absent."
        )
    tags = obj.get("tags")
    meta = obj.get("meta")
    return Case(
        case_id=case_id,
        input_text=text,
        # Absent stays absent. Coercing a missing reference to "" here would
        # erase the one distinction the fingerprint is built to keep.
        reference=reference,
        tags=[str(t) for t in tags] if isinstance(tags, list) else [],
        meta=meta if isinstance(meta, dict) else {},
    )


def read_experiment(path: str | Path) -> Experiment:
    """Read an experiment file, refusals and all."""
    p = Path(path)
    rows: list[Result] = []
    try:
        with p.open("r", encoding="utf-8") as fh:
            header = _read_header(fh, p.name, expect=EXPERIMENT)
            for number, line in enumerate(fh, start=2):
                obj = _read_line(line, p.name, number)
                if obj is None:
                    continue
                rows.append(_result_from(obj, p.name, number))
    except UnicodeDecodeError:
        # NOT an OSError, so the arm below never saw it. `UnicodeDecodeError`
        # is a ValueError, and a binary file handed to a route expecting JSONL
        # therefore escaped as a 500 — while a MISSING file, one line away,
        # answered with a sentence. Same reader, two shapes of wrong file, two
        # completely different experiences.
        raise BadRequest(
            f"{p.name} is not text. A dataset or an experiment is JSONL — one "
            f"JSON object per line, UTF-8 — and this file has bytes in it that "
            f"are not. If it is a `.mri` or an archive, it belongs to a "
            f"different reader."
        ) from None
    except OSError as err:
        raise BadRequest(
            f"{p.name} could not be read ({err.strerror or type(err).__name__})"
        ) from None

    floors = header.get("metric_floors")
    meta = header.get("meta")
    experiment = Experiment(
        name=str(header.get("name") or p.stem),
        dataset_name=str(header.get("dataset_name") or ""),
        dataset_fingerprint=str(header.get("dataset_fingerprint") or ""),
        results=rows,
        label=str(header.get("label") or ""),
        started_at=str(header.get("started_at") or ""),
        metric_floors=floors if isinstance(floors, dict) else {},
        meta=meta if isinstance(meta, dict) else {},
    ).validated()
    # Both gaps, joined, never one replacing the other. What the WRITER of
    # this file had already left out is in the header; what the file itself
    # is missing is computed from the header's count against the rows read.
    # Assigning only the second is how the first died: a capped import
    # declares 3 and writes 3, so the recomputed sentence is empty and
    # overwrote a warning somebody had authored.
    experiment.truncated = " ".join(
        said
        for said in (
            str(header.get("truncated") or ""),
            _truncation(p.name, header.get("n_results"), len(rows), "results"),
        )
        if said
    )
    return experiment


def _result_from(obj: dict, name: str, number: int) -> Result:
    case_id = obj.get("case_id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise BadRequest(f"{name} line {number} has no `case_id`.")
    scores = obj.get("scores")
    receipt = obj.get("receipt")
    internals = obj.get("internals")
    output = obj.get("output")
    return Result(
        case_id=case_id,
        # `None` survives as `None`; `""` survives as `""`. A run that emitted
        # nothing and a run that emitted the empty string are different runs.
        output=output if isinstance(output, str) else None,
        scores=scores if isinstance(scores, dict) else {},
        could_not_measure=str(obj.get("could_not_measure") or ""),
        receipt=receipt if isinstance(receipt, dict) else {},
        internals=internals if isinstance(internals, dict) else {},
    )


def stream_results(path: str | Path):
    """Yield results one at a time, holding one row at a time.

    For a file too large to want in memory — the comparison below needs both
    runs indexed by id and so cannot stream, but counting, filtering and
    re-scoring a single run all can.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        _read_header(fh, p.name, expect=EXPERIMENT)
        for number, line in enumerate(fh, start=2):
            obj = _read_line(line, p.name, number)
            if obj is None:
                continue
            yield _result_from(obj, p.name, number)


# -------------------------------------------------------------- the internals


def _shape(value) -> str:
    """What KIND of internal this is, from its structure alone.

    Never a table of known key names. `imaging.py` records why: a name map
    identifies exactly what was written into it and silently declines to
    compare everything added since, which looks identical to "nothing changed".

    A list of names is a ranking. A mapping of names to signed numbers is a set
    of sites. Anything else is not something this can compare, and says so.
    """
    if isinstance(value, list):
        for item in value:
            if isinstance(item, bool) or not isinstance(item, (str, int, float)):
                return ""
        return "ranking"
    if isinstance(value, dict):
        for key, inner in value.items():
            if not isinstance(key, str):
                return ""
            # bool is an int, and a mapping of flags is not a set of signed
            # magnitudes — `True` would read as +1 and flip against `False`.
            if isinstance(inner, bool) or not isinstance(inner, (int, float)):
                return ""
            if not math.isfinite(float(inner)):
                return ""
        return "sites"
    return ""


def _names(items: list[str]) -> str:
    """`a, b, c and 4 more` — bounded, and what was dropped is counted."""
    head = [str(i) for i in items[:NAMED]]
    if len(items) > NAMED:
        return f"{', '.join(head)} and {len(items) - NAMED} more"
    return ", ".join(head)


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


@dataclass
class InternalDelta:
    """One internal, compared structurally or explicitly not compared."""

    key: str
    kind: str
    status: str
    detail: str
    # The evidence: ranks, values, which names moved. Reported whatever the
    # status, because a reader chasing a regression wants the numbers even
    # when nothing crossed a line.
    measured: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def compare_internals(
    before: dict, after: dict, *, top_k: int = TOP_K
) -> list[InternalDelta]:
    """What the two runs SAW, compared key by key.

    This is the part no other experiment table has, and it is deliberately
    structural rather than magnitudinal: rankings are compared for membership
    and order, signed sites for sign flips and for keys that appeared or
    vanished. Magnitudes travel in `measured` and decide nothing, because this
    module has no floor for an internal — a floor for a score is stated by
    whoever ran the experiment, and no such statement exists for these.
    """
    out: list[InternalDelta] = []
    for key in sorted(set(before) | set(after)):
        if key not in before or key not in after:
            missing = "before" if key not in before else "after"
            out.append(
                InternalDelta(
                    key=key,
                    kind="",
                    status=NOT_COMPARABLE,
                    detail=(
                        f"only the {'later' if missing == 'before' else 'earlier'} "
                        f"run recorded `{key}`, so there is nothing to compare "
                        f"it against. A missing internal is not an unchanged "
                        f"one and it is certainly not a zero."
                    ),
                    measured={
                        "present_in": "after" if missing == "before" else "before"
                    },
                )
            )
            continue

        b, a = before[key], after[key]
        kb, ka = _shape(b), _shape(a)
        if kb != ka or not kb:
            out.append(
                InternalDelta(
                    key=key,
                    kind="",
                    status=NOT_COMPARABLE,
                    detail=(
                        f"`{key}` is a {type(b).__name__} in the earlier run "
                        f"and a {type(a).__name__} in the later one, and this "
                        f"compares two shapes: a list of names as a ranking, "
                        f"and a mapping of names to signed numbers as a set of "
                        f"sites. A bare number belongs in `scores`, where a "
                        f"floor can be stated for it."
                    ),
                )
            )
            continue
        out.append(
            _ranking_delta(key, b, a, top_k)
            if kb == "ranking"
            else _sites_delta(key, b, a)
        )
    return out


def _ranking_delta(key: str, before, after, top_k: int) -> InternalDelta:
    b = [str(x) for x in before]
    a = [str(x) for x in after]
    b_top, a_top = b[:top_k], a[:top_k]
    entered = [x for x in a_top if x not in b_top]
    left = [x for x in b_top if x not in a_top]
    moved = {
        x: [b.index(x), a.index(x)]
        for x in b_top
        if x in a_top and b.index(x) != a.index(x)
    }
    changed = bool(entered or left or moved)
    if changed:
        bits = []
        if entered:
            bits.append(f"{_names(entered)} entered the top {top_k}")
        if left:
            bits.append(f"{_names(left)} left it")
        if moved:
            bits.append(f"{_names(list(moved))} moved rank without leaving")
        detail = (
            f"`{key}` is not the same ranking: {'; '.join(bits)}. That is a "
            f"different set of components carrying this case, which is a lead "
            f"a score alone does not give."
        )
    else:
        detail = (
            f"`{key}` has the same top {top_k}, in the same order. Positions "
            f"past {top_k} are not judged here — a ranking is compared where a "
            f"reader looks at it."
        )
    return InternalDelta(
        key=key,
        kind="ranking",
        status=CHANGED if changed else SAME,
        detail=detail,
        measured={
            "top_k": top_k,
            "before_top": b_top,
            "after_top": a_top,
            "entered": entered,
            "left": left,
            "moved": moved,
        },
    )


def _sites_delta(key: str, before: dict, after: dict) -> InternalDelta:
    only_before = sorted(set(before) - set(after))
    only_after = sorted(set(after) - set(before))
    shared = sorted(set(before) & set(after))
    flipped = {
        site: [float(before[site]), float(after[site])]
        for site in shared
        if _sign(float(before[site])) * _sign(float(after[site])) < 0
    }
    changed = bool(flipped or only_before or only_after)
    if changed:
        bits = []
        if flipped:
            bits.append(f"{_names(list(flipped))} flipped sign")
        if only_before:
            bits.append(f"{_names(only_before)} is no longer recorded")
        if only_after:
            bits.append(f"{_names(only_after)} appeared")
        detail = (
            f"`{key}` changed structurally: {'; '.join(bits)}. A site that "
            f"flipped sign is now pushing the answer the other way, and a site "
            f"that vanished is unknown rather than zero."
        )
    else:
        detail = (
            f"`{key}` has the same sites and none of them changed sign. Their "
            f"magnitudes are in `measured` and are NOT judged here: this "
            f"module has no floor for an internal, and calling a move "
            f"significant without one would be inventing the threshold."
        )
    return InternalDelta(
        key=key,
        kind="sites",
        status=CHANGED if changed else SAME,
        detail=detail,
        measured={
            "flipped": flipped,
            "only_in_before": only_before,
            "only_in_after": only_after,
            "before": {k: float(before[k]) for k in shared},
            "after": {k: float(after[k]) for k in shared},
        },
    )


# ------------------------------------------------------------ the comparison


@dataclass
class RowDelta:
    """One case, across two runs. Always present, whatever happened to it."""

    case_id: str
    status: str
    # Always a sentence, including for `unchanged` — a reader scanning forty
    # rows should never have to reconstruct why one of them says what it says.
    detail: str
    before: float | None = None
    after: float | None = None
    # `None` when either side is unmeasurable. NEVER 0.0, which would read as
    # "no change" and would be counted into any distribution built from it.
    delta: float | None = None
    # `None` when no dataset was supplied to resolve references, which is not
    # the same as False.
    has_reference: bool | None = None
    internals: list[InternalDelta] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "status": self.status,
            "detail": self.detail,
            "before": self.before,
            "after": self.after,
            "delta": self.delta,
            "has_reference": self.has_reference,
            "internals": [i.to_dict() for i in self.internals],
        }


@dataclass
class Comparison:
    """Counts and deltas. Deliberately not a score."""

    metric: str
    higher_is_better: bool
    before_name: str
    after_name: str
    dataset_name: str
    rows: list[RowDelta] = field(default_factory=list)
    counts: dict = field(default_factory=dict)
    n_cases: int = 0
    floor: float = 0.0
    floor_note: str = ""
    # `None` means no dataset was supplied, so nothing here looked. `0` means
    # the dataset was read and has none. Different answers.
    references: int | None = None
    delta_distribution: dict | None = None
    # Every metric name a measured row on either side carries, this one
    # included. The gate that refuses an unrecorded metric already computes
    # this set and then threw it away, so a caller who picked a metric that
    # exists but is not orderable -- an Inspect log's "C"/"I" marker is the
    # case that made this necessary -- got a table of unmeasurable rows and
    # nowhere to read what else was in the file.
    metrics_present: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def internals_changed(self) -> list[RowDelta]:
        """Rows where something the measurement SAW moved. The lead list."""
        return [r for r in self.rows if any(i.status == CHANGED for i in r.internals)]

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "higher_is_better": self.higher_is_better,
            "before_name": self.before_name,
            "after_name": self.after_name,
            "dataset_name": self.dataset_name,
            "n_cases": self.n_cases,
            "counts": dict(self.counts),
            "floor": self.floor,
            "floor_note": self.floor_note,
            "references": self.references,
            "delta_distribution": self.delta_distribution,
            "metrics_present": list(self.metrics_present),
            "rows": [r.to_dict() for r in self.rows],
            "notes": list(self.notes),
            "means": self.means(),
        }

    def means(self) -> str:
        counts = self.counts
        direction = "higher is better" if self.higher_is_better else "lower is better"
        spread = (
            (
                f" Over the {self.delta_distribution['n']} cases measured on "
                f"BOTH sides the delta has median "
                f"{self.delta_distribution['median']:,.6g} and IQR "
                f"{self.delta_distribution['iqr']:,.6g}, ranging "
                f"{self.delta_distribution['lo']:,.6g} to "
                f"{self.delta_distribution['hi']:,.6g}. That `n` is NOT the "
                f"{self.n_cases} cases in this comparison."
            )
            if self.delta_distribution
            else (
                " No case was measured on both sides, so there is no delta "
                "distribution at all — which is not a distribution centred on "
                "zero."
            )
        )
        refs = (
            (f" {self.references} of {self.n_cases} cases carry an expected output.")
            if self.references is not None
            else (
                " No dataset was supplied, so nothing here knows which cases "
                "have an expected output — that count is unknown rather than "
                "zero."
            )
        )
        leads = len(self.internals_changed)
        internals = (
            f" On {leads} of these cases something the measurement SAW moved — "
            f"a ranking reordered or a site flipped sign — and those rows carry "
            f"what it was. That is the lead a score on its own does not give."
            if leads
            else (
                " No row recorded internals that changed, so this comparison "
                "is scores only."
            )
        )
        return (
            f"{self.after_name} against {self.before_name} on "
            f"{self.dataset_name or 'this dataset'}, over {self.metric} where "
            f"{direction}: {counts[BETTER]} better, {counts[WORSE]} worse, "
            f"{counts[UNCHANGED]} unchanged, {counts[UNMEASURABLE]} could not "
            f"be measured — {self.n_cases} cases, and those four always sum to "
            f"it. {self.floor_note}{spread}{refs}{internals}\n\n"
            f"THERE IS NO SINGLE NUMBER HERE ON PURPOSE. Two cases collapsing "
            f"and three improving slightly average out to fine, which is "
            f"exactly the regression an aggregate hides. The counts and the "
            f"per-case rows are the answer."
            + ("\n\n" + " ".join(self.notes) if self.notes else "")
        )


def _score_of(row: Result | None, metric: str, side: str) -> tuple[float | None, str]:
    """One side's score, or `None` and the sentence saying why there is none."""
    if row is None:
        return None, (
            f"the {side} run wrote no row for this case at all, so it is "
            f"unknown here rather than unchanged. A run that stopped early "
            f"looks exactly like this."
        )
    if not row.measured:
        return None, f"the {side} run could not measure it: {row.could_not_measure}"
    if metric not in row.scores:
        present = ", ".join(sorted(row.scores)) or "no metrics at all"
        return None, (
            f"the {side} run measured this case but recorded no `{metric}` "
            f"score for it — it carries {present}."
        )
    value = row.scores[metric]
    # bool is an int. `{"passed": True}` must not be read as 1.0 and ranked.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, (
            f"the {side} run's `{metric}` is a {type(value).__name__}, which "
            f"is not a number this can order against another."
        )
    number = float(value)
    if not math.isfinite(number):
        return None, (
            f"the {side} run's `{metric}` is not a finite number, so it "
            f"cannot be ordered. A non-finite score is a measurement that did "
            f"not work and belongs in `could_not_measure` with its reason."
        )
    return number, ""


def _resolve_floor(
    before: Experiment, after: Experiment, metric: str, given
) -> tuple[float, str]:
    """The smallest difference this comparison will call a change.

    Never invented. Either the caller states one, or the files do, and when
    both files do the COARSER wins — two files cannot be compared more finely
    than the coarser of them can represent, which is the rule `mri_diff`
    already keeps for quantised blocks.
    """
    if given is not None:
        if isinstance(given, bool) or not isinstance(given, (int, float)):
            raise BadRequest(
                f"a floor of {given!r} is not a number, so nothing can be "
                f"compared against it."
            )
        value = float(given)
        if not math.isfinite(value) or value < 0:
            raise BadRequest(
                f"a floor of {given} is not a smallest meaningful difference. "
                f"A negative floor makes every row 'changed' including the "
                f"identical ones, and a non-finite one cannot be compared "
                f"against anything. Pass zero for an exact comparison."
            )
        return value, (
            f"A difference of {value:,.6g} or less in {metric} is reported as "
            f"unchanged, on the caller's instruction."
        )

    stated = []
    for exp in (before, after):
        raw = (
            exp.metric_floors.get(metric)
            if isinstance(exp.metric_floors, dict)
            else None
        )
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            continue
        value = float(raw)
        if math.isfinite(value) and value >= 0:
            stated.append((exp.name, value))

    if len(stated) == 2:
        name, value = max(stated, key=lambda pair: pair[1])
        other = min(stated, key=lambda pair: pair[1])
        if value == other[1]:
            return value, (
                f"Both runs state a floor of {value:,.6g} for {metric}, so a "
                f"difference at or under it is reported as unchanged."
            )
        return value, (
            f"{name} states a floor of {value:,.6g} for {metric} and "
            f"{other[0]} states {other[1]:,.6g}; the coarser is used, because "
            f"two runs cannot be compared more finely than the coarser of them "
            f"can represent."
        )
    if len(stated) == 1:
        name, value = stated[0]
        return value, (
            f"Only {name} states a floor for {metric} ({value:,.6g}), so that "
            f"is what is used; the other run states none, and this comparison "
            f"is no finer than the one statement it has."
        )
    return 0.0, (
        f"Neither run states a floor for {metric} and none was given, so this "
        f"comparison is EXACT: a difference in the last representable digit "
        f"counts as better or worse. That is a fact about float arithmetic "
        f"rather than about the model — pass `floor=` with the smallest "
        f"difference in {metric} you would act on."
    )


def _dataset_mismatch(
    before: Experiment, after: Experiment, dataset: Dataset | None
) -> str:
    """The sentence refusing this pair, or `""` when they may be compared."""
    fb, fa = before.dataset_fingerprint, after.dataset_fingerprint
    if not fb or not fa:
        which = []
        if not fb:
            which.append(before.name)
        if not fa:
            which.append(after.name)
        return (
            f"{' and '.join(which)} did not record which dataset it ran on, so "
            f"nothing here can tell whether these two runs covered the same "
            f"cases. Row-by-row comparison of two different sets is a table of "
            f"real numbers about nothing. Re-run with the dataset's "
            f"fingerprint recorded, or compare the files by hand."
        )
    if fb != fa:
        ids_b = {r.case_id for r in before.results}
        ids_a = {r.case_id for r in after.results}
        gone = sorted(ids_b - ids_a)
        new = sorted(ids_a - ids_b)
        if gone or new:
            parts = []
            if gone:
                parts.append(f"{len(gone)} only in {before.name} ({_names(gone)})")
            if new:
                parts.append(f"{len(new)} only in {after.name} ({_names(new)})")
            return (
                f"{before.name} ran on dataset {before.dataset_name!r} "
                f"(fingerprint {fb}) and {after.name} ran on "
                f"{after.dataset_name!r} (fingerprint {fa}). The case ids "
                f"differ: {'; '.join(parts)}. These are two different sets."
            )
        return (
            f"{before.name} and {after.name} cover the same "
            f"{len(ids_b)} case ids, but the datasets they name hash "
            f"differently ({fb} against {fa}), so an input or a reference "
            f"output was edited between the two runs. A score computed "
            f"against a changed answer is not comparable to one computed "
            f"against the old answer, however identical the ids look."
        )
    if dataset is not None and dataset.fingerprint() != fb:
        return (
            f"the dataset supplied ({dataset.name}, fingerprint "
            f"{dataset.fingerprint()}) is not the one these runs were measured "
            f"on (fingerprint {fb}). Resolving references from it would attach "
            f"the wrong expected output to every case."
        )
    return ""


def compare_experiments(
    before: Experiment,
    after: Experiment,
    *,
    metric: str,
    higher_is_better: bool,
    dataset: Dataset | None = None,
    floor: float | None = None,
    top_k: int = TOP_K,
) -> Comparison:
    """Two runs of one dataset, case by case. Counts and deltas, no verdict.

    `higher_is_better` has no default on purpose. There is no way to tell from
    a metric's name which direction is good — KL divergence is better lower,
    faithfulness is better higher — and a wrong guess inverts every conclusion
    while producing output that looks entirely reasonable.
    """
    if not isinstance(higher_is_better, bool):
        raise BadRequest(
            f"`higher_is_better` is a {type(higher_is_better).__name__} and "
            f"has to be True or False. It decides the sign of every conclusion "
            f"in this comparison, so it is stated rather than inferred."
        )
    if not isinstance(metric, str) or not metric.strip():
        raise BadRequest("no metric was named, so there is nothing to compare.")

    before = before.validated()
    after = after.validated()

    mismatch = _dataset_mismatch(before, after, dataset)
    if mismatch:
        raise DifferentDatasets(mismatch)

    # A metric no measured row on either side carries is a typo, and reporting
    # forty `unmeasurable` rows for a typo buries it. The refusal names what is
    # actually there.
    present: set[str] = set()
    for row in list(before.results) + list(after.results):
        if row.measured:
            present.update(k for k in row.scores if isinstance(k, str))
    if metric not in present:
        raise Refusal(
            f"no measured row in either run carries a `{metric}` score. What "
            f"is recorded: {', '.join(sorted(present)) or 'no metrics at all'}."
        )

    floor_value, floor_note = _resolve_floor(before, after, metric, floor)

    by_before = {r.case_id: r for r in before.results}
    by_after = {r.case_id: r for r in after.results}
    references = (
        {c.case_id: c.reference for c in dataset.cases} if dataset is not None else None
    )

    # THE DENOMINATOR IS A UNION. The dataset's own order first, because that
    # is the order somebody authored, then anything either run has that the
    # dataset does not -- which is itself a fact worth seeing rather than a
    # reason to drop a row.
    order: list[str] = []
    seen: set[str] = set()
    for case_id in (
        [c.case_id for c in dataset.cases] if dataset is not None else []
    ) + sorted(set(by_before) | set(by_after)):
        if case_id not in seen:
            seen.add(case_id)
            order.append(case_id)

    rows: list[RowDelta] = []
    for case_id in order:
        b_row, a_row = by_before.get(case_id), by_after.get(case_id)
        b_value, b_why = _score_of(b_row, metric, "earlier")
        a_value, a_why = _score_of(a_row, metric, "later")

        internals: list[InternalDelta] = []
        if b_row is not None and a_row is not None:
            if b_row.internals or a_row.internals:
                internals = compare_internals(
                    b_row.internals, a_row.internals, top_k=top_k
                )

        has_reference = (
            None if references is None else references.get(case_id) is not None
        )

        if b_value is None or a_value is None:
            rows.append(
                RowDelta(
                    case_id=case_id,
                    status=UNMEASURABLE,
                    detail=" ".join(s for s in (b_why, a_why) if s),
                    before=b_value,
                    after=a_value,
                    # NOT 0.0. A missing side means the difference is unknown,
                    # and a zero here would be averaged into the distribution
                    # as though the change had been measured and was nil.
                    delta=None,
                    has_reference=has_reference,
                    internals=internals,
                )
            )
            continue

        delta = a_value - b_value
        if abs(delta) <= floor_value:
            status = UNCHANGED
            detail = (
                f"{metric} moved from {b_value:,.6g} to {a_value:,.6g}, a "
                f"difference of {delta:,.6g}, which is at or under the floor "
                f"of {floor_value:,.6g}."
            )
        else:
            improved = (delta > 0) == higher_is_better
            status = BETTER if improved else WORSE
            detail = (
                f"{metric} moved from {b_value:,.6g} to {a_value:,.6g} "
                f"({delta:+,.6g}), which is {status} for a metric where "
                f"{'higher' if higher_is_better else 'lower'} is better."
            )
        moved = [i.key for i in internals if i.status == CHANGED]
        if moved:
            detail += f" What the measurement saw also moved: {_names(moved)}."
        rows.append(
            RowDelta(
                case_id=case_id,
                status=status,
                detail=detail,
                before=b_value,
                after=a_value,
                delta=delta,
                has_reference=has_reference,
                internals=internals,
            )
        )

    counts = {
        # Every key present at zero. A caller reading `counts.get("worse", 0)`
        # cannot tell an absent key from a real zero, and this file exists to
        # keep those apart.
        BETTER: sum(1 for r in rows if r.status == BETTER),
        WORSE: sum(1 for r in rows if r.status == WORSE),
        UNCHANGED: sum(1 for r in rows if r.status == UNCHANGED),
        UNMEASURABLE: sum(1 for r in rows if r.status == UNMEASURABLE),
    }

    notes = [n for n in (before.truncated, after.truncated) if n]
    if dataset is not None:
        notes.extend(n for n in (dataset.truncated, dataset.edited) if n)
        # A run may carry a row for a case the dataset does not contain, and
        # that row is COUNTED rather than dropped: a result nobody can point
        # at an input for is a finding about the run, and dropping it would
        # shrink the denominator to make the file tidy.
        known = {c.case_id for c in dataset.cases}
        unknown = [r.case_id for r in rows if r.case_id not in known]
        if unknown:
            notes.append(
                f"{len(unknown)} case(s) appear in the runs and not in the "
                f"dataset supplied ({_names(unknown)}). They are counted here "
                f"rather than dropped, because a row nobody can explain is a "
                f"finding."
            )

    return Comparison(
        metric=metric,
        higher_is_better=higher_is_better,
        before_name=before.name,
        after_name=after.name,
        dataset_name=dataset.name if dataset is not None else before.dataset_name,
        rows=rows,
        counts=counts,
        n_cases=len(rows),
        floor=floor_value,
        floor_note=floor_note,
        references=(
            sum(1 for r in rows if r.has_reference) if references is not None else None
        ),
        delta_distribution=_distribution(
            [r.delta for r in rows if r.delta is not None]
        ),
        metrics_present=sorted(present),
        notes=notes,
    )


def _distribution(deltas: list[float]) -> dict | None:
    """Median, IQR and range — never a mean, and `None` for nothing at all.

    `sweep` states the rule and the reason: a mean over a set hides the one
    case that carried it, and that case is usually the interesting one. `None`
    rather than a dict of zeros, because a distribution over no measurements is
    not a distribution centred on zero.
    """
    if not deltas:
        return None
    ordered = sorted(deltas)
    n = len(ordered)
    if n >= 2:
        q1, _, q3 = statistics.quantiles(ordered, n=4, method="inclusive")
    else:
        # With one point the median IS the whole distribution, and q1/q3 are
        # that same number — reported as such with n=1 beside it, rather than
        # as a spread that was never measured.
        q1 = q3 = ordered[0]
    return {
        "n": n,
        "median": statistics.median(ordered),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "lo": ordered[0],
        "hi": ordered[-1],
    }


# ---------------------------------------------------------------- the terminal


# The widest a case id column gets before an id is shortened for display. A
# content-addressed id is `receipts.DIGEST_CHARS` wide and fits comfortably;
# an authored one can be a sentence, and a 200-column row is unreadable.
MAX_ID_COLUMN = 40


def render(comparison: Comparison, *, limit: int = 12) -> str:
    """The table a terminal reads. Worst first, refusals never off the end."""
    counts = comparison.counts
    out = [
        f"{comparison.after_name} vs {comparison.before_name} · "
        f"{comparison.metric} "
        f"({'higher' if comparison.higher_is_better else 'lower'} is better)",
        f"  {counts[BETTER]} better · {counts[WORSE]} worse · "
        f"{counts[UNCHANGED]} unchanged · {counts[UNMEASURABLE]} could not be "
        f"measured · {comparison.n_cases} cases",
    ]
    # The other metrics, on the second line, because the commonest way to
    # misread this table is to have picked a metric that exists and cannot be
    # ordered -- an imported eval's "C"/"I" marker produces exactly that, a
    # full page of `unmeasurable` with the reason repeated forty times. The
    # names of what else is in the files are the next command to run.
    others = [m for m in comparison.metrics_present if m != comparison.metric]
    if others:
        out.append(f"  also recorded: {', '.join(others)}")
    out.append("")
    # Sized to the ids actually present rather than to a fixed guess, so
    # nothing is cut unless an id is genuinely enormous -- and when one is, it
    # is cut VISIBLY and counted at the bottom. A silently shortened id is a
    # row a reader cannot look up.
    width = min(
        MAX_ID_COLUMN, max((len(r.case_id) for r in comparison.rows), default=8)
    )
    shortened = 0

    # Worse first: a comparison is read to find what broke.
    rank = {WORSE: 0, BETTER: 1, UNCHANGED: 2, UNMEASURABLE: 3}
    ordered = sorted(
        comparison.rows,
        key=lambda r: (rank.get(r.status, 4), -abs(r.delta or 0.0)),
    )
    for row in ordered[:limit]:
        shown = f"{row.delta:+,.6g}" if row.delta is not None else "unknown"
        label = row.case_id
        if len(label) > width:
            label = label[: width - 1] + "…"
            shortened += 1
        out.append(f"  {label:<{width}} {row.status:<13} {shown:>12}")
        if row.status == UNMEASURABLE:
            # The sentence, in the terminal, not only in the JSON. `sweep`
            # prints its refusals for the same reason: "unmeasurable" with no
            # reason is the same non-answer as a gap, and the reason is
            # already written.
            out.append(f"      {row.detail}")
        for finding in row.internals:
            if finding.status == CHANGED:
                out.append(f"      {finding.key}: {finding.detail}")
    if len(ordered) > limit:
        out.append(
            f"  … {len(ordered) - limit} more rows, not shown here — the "
            f"comparison itself carries all {comparison.n_cases}"
        )
    if shortened:
        out.append(
            f"  {shortened} case id(s) were shortened to fit this column; the "
            f"full ids are in the comparison."
        )
    out.append("")
    out.append("  " + comparison.means().replace("\n\n", "\n  "))
    return "\n".join(out)


# ------------------------------------------------ an eval log, as a run


def _score_names(run) -> set:
    """Every metric name the LOG itself uses, anywhere in the run.

    Computed once for the whole run and not per row, because ownership of a
    name is a fact about the log and not about one sample. A scorer that ran
    on sample `b` and errored on sample `a` still owns `match_correct` on
    both -- and a per-row check sees a free name on `a`, writes a derived 1.0
    into it, and produces one column holding two kinds of number: half
    measured by somebody's scorer and half transcribed from a marker here,
    with nothing on either row saying which. `--metric match_correct` then
    ranks the two against each other, `derived_scores` calls the whole column
    derived, and `score_summary` takes a median across the mixture.
    """
    return {
        str(name)
        for row in getattr(run, "rows", [])
        for name in getattr(row, "scores", {})
    }


def _inspect_scores(row, owned: set) -> tuple[dict, dict, list, dict]:
    """One sample's scores, what could not be recorded, and what was derived.

    `owned` is every name the log uses anywhere in the run -- see
    `_score_names`. Returns `(kept, dropped, derived, not_derived)`, the last
    being marker names that got no numeric column, each with the sentence
    saying why: "there is no `match_correct` to compare" is otherwise
    indistinguishable from this converter having forgotten.

    Three things happen here and each one is a decision:

    1. A NON-FINITE value is dropped with a sentence. `write_experiment`
       refuses a NaN anywhere in a row and its message says the row should
       have carried `could_not_measure` instead -- so putting it there is this
       converter's job, not the file writer's job to raise half way through a
       file it has already begun.
    2. Everything else survives BYTE FOR BYTE. A string stays a string.
    3. Inspect's C/I marker gets a numeric companion under its own name, and
       only when the log does not use that name ANYWHERE. A scorer literally
       called `match_correct` owns it, and overwriting it would replace a
       number somebody measured with one derived here.
    """
    kept: dict = {}
    dropped: dict = dict(row.skipped_scores)
    for name, value in row.scores.items():
        if isinstance(value, float) and not math.isfinite(value):
            dropped[name] = (
                "the log recorded a non-finite number (a NaN or an infinity), "
                "which is a measurement that did not work rather than a value"
            )
            continue
        kept[name] = value

    derived: list[str] = []
    not_derived: dict = {}
    for name, value in list(kept.items()):
        if not isinstance(value, str):
            continue
        number = SCORE_MARKERS.get(value.strip().upper())
        if number is None:
            continue
        companion = f"{name}{DERIVED_SUFFIX}"
        if companion in kept or companion in owned:
            not_derived[name] = (
                f"this log has its own `{companion}` scorer, so the marker "
                f"was left as the log wrote it rather than given a number "
                f"under a name somebody else's scorer already owns"
            )
            continue
        kept[companion] = number
        derived.append(companion)
    return kept, dropped, derived, not_derived


def _inspect_could_not_measure(row, kept: dict, dropped: dict) -> str:
    """Why a row has no scores, or `""` when it has some.

    `Result.could_not_measure` is documented as "the sentence saying why this
    row has no scores. Empty when it does", and that contract is what decides
    this: an unscored sample is not a measured row with an empty dict, it is a
    row nothing measured. The alternative -- leaving it empty and letting
    `_score_of` say "measured this case but recorded no `X` score" -- reads as
    though a scorer ran and declined, and makes `n_measured` count samples the
    eval never scored.
    """
    if kept:
        return ""
    if row.error:
        return (
            f"the log records this sample as errored and no score survived "
            f"it: {row.error}"
        )
    if dropped:
        named = "; ".join(f"`{k}` — {why}" for k, why in sorted(dropped.items()))
        return (
            f"no score on this sample could be recorded: {named}. An "
            f"unreadable score is not a score of zero."
        )
    return (
        "the log recorded no score for this sample at all, which is not a "
        "score of zero — nothing measured it."
    )


def from_inspect(run, *, name: str = "", label: str = "") -> tuple[Experiment, Dataset]:
    """An Inspect eval log's scores, as an experiment and the set it ran on.

    `run` is an already-read `inspect_io.ScoredRun`. Taken rather than read,
    for the reason `from_traces` takes already-fetched documents: this module
    never opens somebody else's format, which is what lets both be tested
    without one and keeps `inspect_io` a leaf the server can import lazily
    inside its route.

    Returns `(experiment, dataset)`. Both, because they are only useful
    together: the experiment records this run's scores and the dataset records
    what each case ASKED, and `compare_experiments` refuses to compare two
    runs that do not record the same dataset fingerprint. Building only the
    experiment would produce a file that cannot be compared to anything.

    ## What it does not do

    It does not decide whether the eval went well. Inspect's own markers are
    transcribed (see `SCORE_MARKERS`), everything else is copied, and nothing
    here invents a verdict, a threshold or a metric floor -- `metric_floors`
    stays empty because a floor is a claim about a metric's precision that
    only whoever computed it can make.
    """
    from . import __version__

    head = run.header
    task = str(getattr(head, "task", "") or "")
    model = str(getattr(head, "model", "") or "")
    log_name = str(getattr(run, "log_name", "") or "")
    # The receipt writer's truncation, applied here rather than in the reader:
    # `DIGEST_CHARS` is this side of the fence's rule about how short a
    # provenance label gets to be, and `inspect_io` does not import receipts.
    log_sha = str(getattr(run, "log_sha256", "") or "")[: receipts.DIGEST_CHARS]

    rows: list[Result] = []
    cases: list[Case] = []
    skipped_scores: dict = {}
    derived_names: set[str] = set()
    not_derived: dict = {}
    sample_errors: dict = {}
    # Decided once for the whole run, before any row is converted: which names
    # the log itself owns is a fact about the log, not about one sample.
    owned = _score_names(run)

    for row in run.rows:
        kept, dropped, derived, left = _inspect_scores(row, owned)
        derived_names.update(derived)
        not_derived.update(left)
        if dropped:
            skipped_scores[row.case_id] = dropped
        if row.error and kept:
            # An Inspect sample can be errored AND scored -- a scorer that ran
            # on a partial transcript. The scores are real, so the row stays
            # measured and `could_not_measure` (which flips `measured`) is the
            # wrong place for the error; without this the one thing the log
            # recorded about a row somebody is about to compare reached
            # nowhere in the experiment at all. `Result` has no per-row note
            # field to put it on, which is the schema friction worth naming.
            sample_errors[row.case_id] = row.error

        rows.append(
            Result(
                case_id=row.case_id,
                output=row.output,
                scores=kept,
                could_not_measure=_inspect_could_not_measure(row, kept, dropped),
                receipt=receipts.Receipt(
                    op="inspect_import",
                    # Through the same sanitiser `stamp()` puts its request
                    # block through. Nothing here should be able to carry a
                    # path -- `log_name` is a bare filename by construction --
                    # but this is the one receipt in the tree built without
                    # `stamp`, and skipping the reduction would make it the one
                    # the leak test's rule was never applied to.
                    request=receipts._request(
                        {
                            "task": task,
                            "log_name": log_name,
                            "log_sha256": log_sha,
                            "log_format_version": getattr(head, "version", 0),
                            "sample_id": row.id,
                            "epoch": row.epoch,
                            "n_samples": getattr(head, "n_samples", 0),
                        }
                    ),
                    tool_version=__version__,
                    # `public_name` even though an Inspect model id is normally
                    # `provider/model`: it can be a local path for a locally
                    # served model, and a receipt is the part of a finding most
                    # likely to be forwarded to a stranger.
                    model=receipts.public_name(model) or None,
                    prompt_sha256=(
                        receipts.digest(row.input_text) if row.input_text else None
                    ),
                    # When the eval ran, from the log -- NOT now. Stamping the
                    # import time here would date somebody else's measurement
                    # to the moment this machine happened to read it.
                    measured_at=str(getattr(head, "created", "") or ""),
                ).to_dict(),
            )
        )
        cases.append(
            Case(
                case_id=row.case_id,
                input_text=row.input_text,
                # Inspect's `target` is a real answer key, and absent stays
                # absent: `None` and `""` serialise differently and the
                # fingerprint is built to keep them apart.
                reference=row.target,
                meta={"sample_id": row.id, "epoch": row.epoch},
            )
        )

    dataset = Dataset(
        name=task or log_name or "inspect-log",
        cases=cases,
        description=(
            f"Every sample in {log_name or 'an Inspect log'}, with the target "
            f"the log states as its expected output."
        ),
        created_at=str(getattr(head, "created", "") or ""),
    ).validated()

    n_total = int(getattr(run, "n_total", len(run.rows)) or len(run.rows))
    experiment = Experiment(
        name=name or task or log_name or "inspect-import",
        dataset_name=dataset.name,
        dataset_fingerprint=dataset.fingerprint(),
        results=rows,
        label=label or model,
        started_at=str(getattr(head, "created", "") or ""),
        meta={
            "source": INSPECT,
            "task": task,
            "model": model,
            "log_name": log_name,
            "log_sha256": log_sha,
            "log_format_version": getattr(head, "version", 0),
            "n_samples_total": n_total,
            "n_samples_read": len(run.rows),
            # Stated in the FILE, not only in this module, so a reader holding
            # the .jsonl a year from now can see where a `_correct` column came
            # from without this source in front of them.
            "score_markers": dict(SCORE_MARKERS),
            "derived_scores": sorted(derived_names),
            "markers_not_derived": not_derived,
            "skipped_scores": skipped_scores,
            "sample_errors": sample_errors,
            # The log's own creation time, recorded whether or not it has one.
            # An empty string here is the file SAYING the log stated none,
            # which is what separates it from a converter that lost it.
            "log_created": str(getattr(head, "created", "") or ""),
            "means": _from_inspect_means(
                task,
                model,
                log_name,
                rows,
                n_total,
                sorted(derived_names),
                skipped_scores,
                not_derived,
                sample_errors,
                str(getattr(head, "created", "") or ""),
            ),
        },
    ).validated()

    if getattr(run, "truncated", False):
        # The same correction the listing path already carries. An experiment
        # 5,000 rows into a 6,000-sample eval that says nothing about the gap
        # is one somebody compares as though it were the whole run.
        experiment.truncated = (
            f"this log carries {n_total} samples and only the first "
            f"{len(run.rows)} were read, so this run is "
            f"{n_total - len(run.rows)} row(s) short of the eval it names."
        )
    return experiment, dataset


def _from_inspect_means(
    task: str,
    model: str,
    log_name: str,
    rows: list,
    n_total: int,
    derived: list,
    skipped: dict,
    not_derived: dict,
    sample_errors: dict,
    created: str,
) -> str:
    measured = sum(1 for r in rows if r.measured)
    where = f" ({task})" if task else ""
    who = f" on {model}" if model else ""
    head = (
        f"{len(rows)} of {n_total} sample(s) from "
        f"{log_name or 'an Inspect log'}{where}{who}: {measured} carry a score "
        f"and {len(rows) - measured} do not, and each of those says why."
    )
    made = (
        (
            f" {len(derived)} column(s) were DERIVED here rather than read from "
            f"the log — {', '.join(derived)} — each one Inspect's own C/I "
            f"marker written as {SCORE_MARKERS['C']}/{SCORE_MARKERS['I']} under "
            f"a new name, because a string cannot be ordered against another. "
            f"The marker itself is still in the row exactly as the log wrote it."
        )
        if derived
        else (
            " No column was derived: nothing in this log used Inspect's C/I "
            "marker, so every score here is one the log itself recorded."
        )
    )
    left_alone = ", ".join(
        f"{marker} (would have been {marker}{DERIVED_SUFFIX})"
        for marker in sorted(not_derived)
    )
    kept_as_written = (
        (
            f" {len(not_derived)} marker(s) got NO derived number — "
            f"{left_alone} — because this log has its own scorer under that "
            f"name on some row, and a column somebody measured is not one to "
            f"write into. Compare on the log's own column, not on a derived "
            f"one that is not in this file."
        )
        if not_derived
        else ""
    )
    lost = (
        f" {len(skipped)} sample(s) carried a score entry this reader could not "
        f"take a value from; they are listed under `skipped_scores` with the "
        f"reason, rather than counted as unscored."
        if skipped
        else ""
    )
    errored = (
        f" {len(sample_errors)} scored sample(s) are ALSO marked errored in the "
        f"log — a scorer that ran on a transcript that crashed. Their scores "
        f"are real and the rows are measured; what the log said went wrong is "
        f"under `sample_errors`, keyed by case id."
        if sample_errors
        else ""
    )
    # An absent date is stated rather than left to be noticed. The file's own
    # `started_at` is empty in this case, and a reader who finds a run with no
    # date on it has to be able to tell "the log stated none" from "whatever
    # wrote this dropped it".
    when = (
        ""
        if created
        else (
            " This log states no time at which it ran, so this run carries no "
            "start time either — an import moment is not when somebody else's "
            "eval happened."
        )
    )
    return (
        f"{head}{made}{kept_as_written}{lost}{errored}{when} No verdict, no "
        f"floor and no threshold is set here: what the eval measured is a "
        f"fact, and what counts as a regression is a decision for whoever "
        f"compares two of these."
    )


# ------------------------------------------------- what one run measured


def score_summary(experiment: Experiment) -> dict:
    """Every metric one run recorded, counted — the run's own table of scores.

    `compare_experiments` answers "what moved between two runs". This answers
    the question that comes before it and had no reader at all: what is IN
    this file. A converted eval log is the case that needs it — somebody
    holding a fresh experiment has no way to learn which metric to pass to
    `--metric` without opening the JSONL by hand.

    Never an aggregate ACROSS metrics, for the reason `Comparison.means`
    gives: two collapsing and three improving average out to fine.
    """
    rows = experiment.results
    values: dict[str, list] = {}
    for row in rows:
        if not row.measured:
            continue
        for metric, value in row.scores.items():
            if isinstance(metric, str):
                values.setdefault(metric, []).append(value)

    metrics = []
    for metric in sorted(values):
        got = values[metric]
        # bool is an int, and this project refuses to rank one — so a metric
        # carrying a boolean is summarised by its VALUES, not by a median.
        numbers = [
            float(v)
            for v in got
            if not isinstance(v, bool) and isinstance(v, (int, float))
        ]
        all_numeric = len(numbers) == len(got) and bool(numbers)
        metrics.append(
            {
                "metric": metric,
                "n": len(got),
                # Counted, not inferred: a metric on 3 of 40 rows is a
                # different fact from one on all 40, and the gap is the story.
                "n_missing": len(rows) - len(got),
                "numbers": all_numeric,
                # `None`, never 0, for a metric that is not a number. A median
                # of zero and no median at all are different answers.
                "median": statistics.median(numbers) if all_numeric else None,
                "min": min(numbers) if all_numeric else None,
                "max": max(numbers) if all_numeric else None,
                "values": (
                    None
                    if all_numeric
                    else _value_counts([_score_label(v) for v in got])
                ),
            }
        )

    unmeasured = [r for r in rows if not r.measured]
    return {
        "name": experiment.name,
        "label": experiment.label,
        "dataset_name": experiment.dataset_name,
        "dataset_fingerprint": experiment.dataset_fingerprint,
        "n_results": len(rows),
        "n_measured": experiment.n_measured,
        "n_unmeasured": len(unmeasured),
        "metrics": metrics,
        # The distinct REASONS, counted. Forty copies of one sentence is one
        # finding, and printing it forty times buries the second one.
        "why_unmeasured": _value_counts([r.could_not_measure for r in unmeasured]),
        "truncated": experiment.truncated,
        "means": _score_summary_means(experiment, metrics, len(unmeasured)),
    }


def _score_label(value) -> str:
    """A score value as one short readable token."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:,.6g}"
    return str(value)


def _value_counts(labels: list) -> dict:
    out: dict = {}
    for label in labels:
        out[label] = out.get(label, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def _score_summary_means(experiment: Experiment, metrics: list, unmeasured: int) -> str:
    if not metrics:
        return (
            f"{experiment.name} carries {len(experiment.results)} row(s) and no "
            f"metric at all — nothing in it was scored, which is not every case "
            f"scoring zero. Each row says why it has none."
        )
    named = ", ".join(f"{m['metric']} ({m['n']})" for m in metrics)
    orderable = [m["metric"] for m in metrics if m["numbers"]]
    gate = (
        f" {', '.join(orderable)} can be passed to `--metric`; the rest are not "
        f"numbers and cannot be ordered against another run."
        if orderable
        else (
            " None of these is a number, so none can be ordered against another "
            "run — a comparison would report every row as unmeasurable."
        )
    )
    return (
        f"{experiment.name}: {len(experiment.results)} row(s), {unmeasured} of "
        f"them with no score. Metrics recorded, with how many rows carry each: "
        f"{named}.{gate}"
    )


def render_scores(experiment: Experiment, *, limit: int = 20) -> str:
    """One run's metrics as a terminal table — the reader `score_summary` needs."""
    summary = score_summary(experiment)
    label = f" · {summary['label']}" if summary["label"] else ""
    out = [
        f"{summary['name']}{label} · {summary['dataset_name'] or 'no dataset named'}",
        f"  {summary['n_results']} row(s) · {summary['n_measured']} scored · "
        f"{summary['n_unmeasured']} with no score",
        "",
    ]
    if not summary["metrics"]:
        out.append("  no metric was recorded for any row in this run")
    width = min(
        MAX_ID_COLUMN, max((len(m["metric"]) for m in summary["metrics"]), default=8)
    )
    for metric in summary["metrics"][:limit]:
        if metric["numbers"]:
            shown = (
                f"median {metric['median']:,.6g} "
                f"(range {metric['min']:,.6g} to {metric['max']:,.6g})"
            )
        else:
            shown = ", ".join(f"{v}x{n}" for v, n in metric["values"].items())
        out.append(f"  {metric['metric']:<{width}} {metric['n']:>4} rows  {shown}")
        if metric["n_missing"]:
            # Named rather than implied. A metric on 3 of 40 rows read as a
            # metric on 40 is how a partial scorer looks like a complete one.
            blank = ""
            out.append(
                f"  {blank:<{width}}        {metric['n_missing']} row(s) do "
                f"not carry it"
            )
    if len(summary["metrics"]) > limit:
        out.append(
            f"  … {len(summary['metrics']) - limit} more metric(s), not shown "
            f"here — the summary itself carries all of them"
        )
    for why, n in summary["why_unmeasured"].items():
        out.append(f"\n  {n} row(s): {why}")
    if summary["truncated"]:
        out.append(f"\n  {summary['truncated']}")
    out.append("")
    out.append("  " + summary["means"])
    return "\n".join(out)
