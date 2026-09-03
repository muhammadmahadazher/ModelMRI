# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""Read a UK AISI Inspect `.eval` log onto the ModelMRI timeline.

Inspect is where eval interop is consolidating — Docent integrates natively,
Apollo publicly adopted it. Speaking it turns this timeline into a second
viewer for logs people already have, and stops `.mri` being a private dialect.

An `.eval` is a zip of JSON. So: `zipfile` and `json`, no new dependency, works
with the network off.

## Reader only, on purpose

There is no writer here and there will not be one. Inspect's schema is not
frozen; committing to track an unfrozen format in BOTH directions forever is
not solo-maintainer work, and somebody with Inspect logs already has Inspect's
viewer. Reading is the half that adds something they do not have.

## Version-gated, and named

The log states its own format version. An unrecognised one is refused WITH THE
VERSION IN THE MESSAGE — the same discipline `session.parse` applies to
`format_version`. Guessing at a schema that moved produces a timeline full of
real-looking steps in the wrong places, which is worse than a refusal.

## What was dropped is reported, not implied

Only fields actually present are mapped, and every event kind this does not
understand is COUNTED and named in the result. A reader who sees 40 steps from
a 90-event sample needs to know that, and "we showed you what we understood"
is only honest if the rest is on screen too.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .errors import BadRequest
from .step_kinds import VALID_KINDS

# `meta["source"]` for a trace that came out of somebody else's eval log, as
# opposed to one this app performed (`traces.SOURCE_APP`) or one an
# instrumented program posted. Named here rather than imported from `traces`
# because this module is a leaf -- it imports `errors` and `step_kinds` and
# nothing else, and the server imports it lazily inside the route precisely to
# keep that cost off the boot path. The string is what `traces.list_traces`
# reads and it is the same vocabulary either way.
SOURCE = "inspect"

# Inspect's own log format version, from the header. 2 is the version in
# current use; 1 predates the zip layout entirely and is a different reader.
#
# Listed rather than "anything >= 2": a version bump is how a schema says it
# moved, and silently accepting 3 would be this module guessing.
SUPPORTED_VERSIONS = (2,)

# Where things live inside the archive.
HEADER = "header.json"
SAMPLES_DIR = "samples/"

# A single sample is one trace. Reading them lazily matters: an eval log can
# carry thousands, and parsing every one to show a list of names would hold
# the whole run in memory to answer a question about its table of contents.
MAX_SAMPLES_LISTED = 5_000

# Event kinds Inspect emits, mapped onto the kinds a step may have. Anything
# not here is counted as dropped and named, never quietly folded into a
# neighbouring kind.
EVENT_KINDS = {
    "model": "llm_call",
    "tool": "tool_call",
    "subtask": "subagent",
    "error": "error",
}

# Message roles, for the messages fallback when a sample carries no events.
ROLE_KINDS = {
    "user": "user_turn",
    "system": "user_turn",
    "assistant": "llm_call",
    "tool": "tool_call",
}

assert set(EVENT_KINDS.values()) <= VALID_KINDS
assert set(ROLE_KINDS.values()) <= VALID_KINDS


class InspectError(BadRequest):
    """This log cannot be read honestly, and the message says why."""


@dataclass
class Mapping:
    """What became a step, and what did not."""

    mapped: dict = field(default_factory=dict)
    dropped: dict = field(default_factory=dict)

    def count(self, kind: str, *, ok: bool) -> None:
        target = self.mapped if ok else self.dropped
        target[kind] = target.get(kind, 0) + 1

    def to_dict(self) -> dict:
        return asdict(self) | {"means": self.means()}

    def means(self) -> str:
        if not self.mapped and not self.dropped:
            return "Nothing in this sample became a step."
        got = ", ".join(f"{n} {k}" for k, n in sorted(self.mapped.items()))
        parts = [f"Mapped {got}." if got else "Mapped nothing."]
        if self.dropped:
            lost = ", ".join(f"{n} {k}" for k, n in sorted(self.dropped.items()))
            parts.append(
                f"NOT SHOWN: {lost}. Inspect's schema is not frozen and these "
                f"event kinds have no step kind here, so they are counted "
                f"rather than folded into a neighbouring one."
            )
        return " ".join(parts)


@dataclass
class SampleRef:
    """One sample's identity, without having parsed it."""

    name: str  # the path inside the archive
    id: str
    epoch: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvalHeader:
    """What the log says about itself."""

    version: int = 0
    task: str = ""
    model: str = ""
    created: str = ""
    status: str = ""
    n_samples: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _open(path) -> zipfile.ZipFile:
    where = Path(str(path)).expanduser()
    if not where.is_file():
        raise InspectError(
            "that .eval log does not exist. An Inspect log is the file "
            "`inspect eval` writes into its log directory."
        )
    try:
        return zipfile.ZipFile(where)
    except (OSError, zipfile.BadZipFile) as err:
        raise InspectError(
            "that file is not a readable Inspect log. An `.eval` is a zip "
            "archive of JSON, and this one could not be opened as one."
        ) from err


def _load(archive: zipfile.ZipFile, name: str) -> dict:
    try:
        raw = archive.read(name)
    except KeyError as err:
        raise InspectError(
            f"this log has no {name} inside it, so there is nothing to read."
        ) from err
    try:
        doc = json.loads(raw)
    except ValueError as err:
        raise InspectError(f"{name} inside this log is not readable JSON.") from err
    if not isinstance(doc, dict):
        raise InspectError(f"{name} inside this log is not an object.")
    return doc


def header(path) -> EvalHeader:
    """What the log says about itself, and a refusal if the version moved."""
    with _open(path) as archive:
        doc = _load(archive, HEADER)
        version = doc.get("version")
        if not isinstance(version, int) or isinstance(version, bool):
            raise InspectError(
                "this log does not state a format version. Refusing rather "
                "than assuming: a schema guessed wrong produces a timeline "
                "full of real-looking steps in the wrong places."
            )
        if version not in SUPPORTED_VERSIONS:
            raise InspectError(
                f"this is an Inspect log format version {version}, and this "
                f"reader speaks {', '.join(str(v) for v in SUPPORTED_VERSIONS)}. "
                f"Refusing rather than guessing at a schema that moved."
            )
        spec = doc.get("eval") if isinstance(doc.get("eval"), dict) else {}
        results = doc.get("results") if isinstance(doc.get("results"), dict) else {}
        return EvalHeader(
            version=version,
            task=str(spec.get("task") or ""),
            model=str(spec.get("model") or ""),
            created=str(spec.get("created") or ""),
            status=str(doc.get("status") or ""),
            n_samples=(
                int(results["total_samples"])
                if isinstance(results.get("total_samples"), int)
                else 0
            ),
        )


class SampleList(list):
    """The listed samples, and how many there really are.

    A plain list could not carry the cap, so `MAX_SAMPLES_LISTED` was applied
    and then reported as the log's size: a 6,000-sample file rendered "5000
    samples" as a fact about the reader's own archive, and the "open another
    sample" dropdown silently held an arbitrary subset in whatever order the
    zip happened to store them — a later sample simply unselectable, with
    nothing saying why.

    The same shape `weights_scan.ScanTree` and `custom.Candidates` use, and the
    same reason. This module's REFUSAL path already got this right and says so
    in its own comment; the listing path did not.
    """

    def __init__(self, refs=(), *, n_total: int = 0) -> None:
        super().__init__(refs)
        #: Every sample in the archive, counted — not just the listed ones.
        self.n_total = n_total or len(self)

    @property
    def truncated(self) -> bool:
        return self.n_total > len(self)


def samples(path) -> SampleList:
    """Every sample's identity, WITHOUT parsing any of them.

    Reading the names off the archive's directory is the difference between
    listing a 4,000-sample eval instantly and holding the whole run in memory
    to answer a question about its table of contents.
    """
    out = []
    n_total = 0
    with _open(path) as archive:
        for name in archive.namelist():
            if not name.startswith(SAMPLES_DIR) or not name.endswith(".json"):
                continue
            stem = name[len(SAMPLES_DIR) : -len(".json")]
            # Inspect names these `<id>_epoch_<n>.json`; older writers used
            # `<id>_<n>.json`. Split from the RIGHT so an id containing an
            # underscore survives.
            sample_id, epoch = stem, 1
            for marker in ("_epoch_", "_"):
                head, sep, tail = stem.rpartition(marker)
                if sep and tail.isdigit():
                    sample_id, epoch = head, int(tail)
                    break
            # COUNTED before the cap is applied. Reading one more name off
            # the archive's directory costs nothing — the expensive thing
            # this function avoids is parsing samples, not seeing that they
            # exist — so there is no reason for the total to be unknown.
            n_total += 1
            if len(out) < MAX_SAMPLES_LISTED:
                out.append(SampleRef(name=name, id=sample_id, epoch=epoch))
    out.sort(key=lambda s: (s.id, s.epoch))
    return SampleList(out, n_total=n_total)


def _text(value) -> str:
    """Inspect content is a string, or a list of typed content blocks."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for block in value:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # `text` for text blocks; anything else is named rather than
                # rendered as an empty string, so an image does not read as a
                # message with nothing in it.
                if isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif block.get("type"):
                    parts.append(f"[{block['type']}]")
        return "\n".join(parts)
    if value is None:
        return ""
    return str(value)


def _steps_from_events(events, mapping: Mapping) -> list:
    """Inspect events, in order, as steps."""
    steps = []
    base = None
    # Where the synthetic ladder has reached, so a real timestamp arriving
    # later CONTINUES it instead of restarting the axis at zero.
    next_synthetic = 0
    for event in events:
        if not isinstance(event, dict):
            continue
        name = str(event.get("event") or "")
        kind = EVENT_KINDS.get(name)
        if kind is None:
            mapping.count(name or "unnamed", ok=False)
            continue
        mapping.count(name, ok=True)

        # Timestamps are ISO strings; the timeline wants milliseconds from the
        # start of the run. A sample whose events carry no timestamp gets
        # sequential offsets rather than all-zero, which would stack every
        # block on top of each other at x=0.
        started = _epoch_ms(event.get("timestamp"))
        if started is not None and base is None:
            # ANCHORED TO THE LADDER, not to itself. `base = started` made the
            # first timestamped event's offset 0 -- so in a sample whose first
            # three events carry no timestamp and whose fourth does, the
            # fourth rendered on top of the first and BEFORE the two between
            # them. The comment above promises the opposite ("rather than
            # all-zero, which would stack every block on top of each other at
            # x=0"), and the two schemes were simply not on one origin.
            base = started - next_synthetic
        offset = (
            (started - base) if (started is not None and base is not None) else None
        )
        if offset is None:
            offset = next_synthetic
        # Untimestamped events after a real one continue from where the last
        # block ended, for the same reason.
        next_synthetic = offset + 10

        step: dict = {
            "id": f"{name}-{len(steps)}",
            "kind": kind,
            "name": _event_name(event, name),
            "started_ms": offset,
            "error": bool(event.get("error")),
        }
        if name == "model":
            step["input"] = _model_input(event)
            step["output"] = _model_output(event)
            usage = _usage(event)
            step.update(usage)
            model = event.get("model")
            if isinstance(model, str) and model:
                step["meta"] = {"model": model}
        elif name == "tool":
            step["input"] = _text(event.get("arguments"))
            step["output"] = _text(event.get("result"))
            if event.get("error"):
                step["error"] = True
                step["output"] = _text(event.get("error")) or step["output"]
        else:
            step["input"] = _text(event.get("input"))
            step["output"] = _text(event.get("result") or event.get("output"))
        steps.append(step)
    return steps


def _event_name(event: dict, kind: str) -> str:
    for key in ("function", "model", "name"):
        value = event.get(key)
        if isinstance(value, str) and value:
            return value
    return kind


def _model_input(event: dict) -> str:
    value = event.get("input")
    if isinstance(value, list):
        lines = []
        for message in value:
            if isinstance(message, dict):
                role = str(message.get("role") or "")
                lines.append(f"{role}: {_text(message.get('content'))}".strip(": "))
        return "\n".join(lines)
    return _text(value)


def _model_output(event: dict) -> str:
    out = event.get("output")
    if not isinstance(out, dict):
        return _text(out)
    choices = out.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                return _text(message.get("content"))
    return _text(out.get("completion"))


def _usage(event: dict) -> dict:
    """Token counts, when the log carries them.

    Absent stays ABSENT — no key rather than a zero. The trace store's columns
    are nullable so "the provider reported nothing" is recordable, and writing
    0 here would claim a report that never happened.
    """
    out = event.get("output")
    usage = out.get("usage") if isinstance(out, dict) else None
    if not isinstance(usage, dict):
        return {}
    got = {}
    for ours, theirs in (
        ("tokens_in", "input_tokens"),
        ("tokens_out", "output_tokens"),
        ("tokens_cache_read", "input_tokens_cache_read"),
        ("tokens_cache_write", "input_tokens_cache_write"),
        ("tokens_reasoning", "reasoning_tokens"),
    ):
        value = usage.get(theirs)
        if isinstance(value, int) and not isinstance(value, bool):
            got[ours] = value
    return got


def _steps_from_messages(messages, mapping: Mapping) -> list:
    """The fallback when a sample carries no event log."""
    steps = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        kind = ROLE_KINDS.get(role)
        if kind is None:
            mapping.count(f"message:{role or 'unnamed'}", ok=False)
            continue
        mapping.count(f"message:{role}", ok=True)
        steps.append(
            {
                "id": f"msg-{len(steps)}",
                "kind": kind,
                "name": role,
                "input": _text(message.get("content")) if role != "assistant" else "",
                "output": _text(message.get("content")) if role == "assistant" else "",
                "started_ms": len(steps) * 10,
                "error": False,
            }
        )
    return steps


def _epoch_ms(value):
    if not isinstance(value, str) or not value:
        return None
    from datetime import datetime

    text = value.replace("Z", "+00:00")
    try:
        return int(datetime.fromisoformat(text).timestamp() * 1000)
    except ValueError:
        return None


@dataclass
class Imported:
    """One Inspect sample, as a trace this tool can draw."""

    trace: dict
    mapping: Mapping
    scores: dict = field(default_factory=dict)
    #: Score entries this reader could not take a value from, keyed by the
    #: scorer's name, each with the sentence saying what was there instead.
    #: Its own field rather than a `Mapping.count(..., ok=False)` entry: the
    #: mapping's vocabulary is EVENT KINDS and its sentence explains that an
    #: unknown kind has no step kind here, which is not true of a score and
    #: would put the wrong reason on screen.
    skipped_scores: dict = field(default_factory=dict)
    failed: bool = False
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "trace": self.trace,
            "mapping": self.mapping.to_dict(),
            "scores": self.scores,
            "skipped_scores": self.skipped_scores,
            "failed": self.failed,
            "error": self.error,
            "means": self.means(),
        }

    def means(self) -> str:
        parts = [self.mapping.means()]
        if self.scores:
            named = ", ".join(f"{k}={v}" for k, v in sorted(self.scores.items()))
            parts.append(f"Scored {named}.")
        elif not self.skipped_scores:
            # NOT "scored 0". A sample nobody scored and a sample scored zero
            # are different facts, and only one of them is a mark.
            parts.append(
                "This log records no score for this sample, which is not a "
                "score of zero — nothing measured it."
            )
        if self.skipped_scores:
            lost = ", ".join(
                f"{name} ({why})" for name, why in sorted(self.skipped_scores.items())
            )
            parts.append(
                f"NOT SCORED HERE: {lost}. A score whose value this reader "
                f"cannot take is counted rather than dropped, for the same "
                f"reason an unmapped event kind is."
            )
        if self.failed:
            parts.append(
                "This sample is marked as failed in the log, which is why it "
                "is the one on screen."
            )
        return " ".join(parts)


def read_sample(
    path, sample_id: str = "", epoch: int = 0, head: EvalHeader | None = None
) -> Imported:
    """One sample, as a trace. Empty `sample_id` reads the first failing one.

    Reading the FAILING one by default is the point: "same timeline, failing
    sample highlighted" is what makes this worth opening rather than scrolling.

    `head` is the already-read header, for a caller that has one -- the route
    reads it a line before calling this. Omitted, it is read here, and that is
    not only for the trace's provenance: this function drew steps out of a log
    of ANY format version, so the module's central promise (an unrecognised
    schema is refused with the version named) held for `header()` and for the
    sample-not-found path and nowhere else. A caller reaching straight for a
    sample got a timeline full of real-looking steps out of a schema that had
    moved, which is the exact failure the version gate exists to prevent.
    """
    if head is None:
        head = header(path)
    refs = samples(path)
    if not refs:
        raise InspectError("this log carries no samples, so there is no run to draw.")

    wanted = None
    if sample_id:
        for ref in refs:
            if ref.id == sample_id and (not epoch or ref.epoch == epoch):
                wanted = ref
                break
        if wanted is None:
            # The listing is capped at MAX_SAMPLES_LISTED and this sentence
            # used to quote `len(refs)` as the log's size -- so a 6,000-sample
            # log told the reader it "carries 5000 sample(s)", a false claim
            # about their file, while looking like the authoritative answer to
            # "is my sample in here?". The header's own total is the real
            # count; when the two differ the cap is stated rather than
            # substituted.
            total = head.n_samples or len(refs)
            capped = (
                ""
                if len(refs) >= total
                else (
                    f" Only the first {len(refs)} are listed, so a sample past "
                    f"that point is not searched here."
                )
            )
            raise InspectError(
                f"no sample {sample_id!r} in this log. It carries "
                f"{total} sample(s).{capped}"
            )

    with _open(path) as archive:
        if wanted is None:
            # First failing, else first. Parsed one at a time and released, so
            # finding the failure in a 4,000-sample log costs one sample of
            # memory rather than all of them.
            for ref in refs:
                doc = _load(archive, ref.name)
                if doc.get("error") or _failed(doc):
                    wanted = ref
                    break
            else:
                wanted = refs[0]
            doc = _load(archive, wanted.name)
        else:
            doc = _load(archive, wanted.name)

    mapping = Mapping()
    events = doc.get("events")
    if isinstance(events, list) and events:
        steps = _steps_from_events(events, mapping)
    else:
        messages = doc.get("messages")
        steps = (
            _steps_from_messages(messages, mapping)
            if isinstance(messages, list)
            else []
        )
    if not steps:
        raise InspectError(
            f"sample {wanted.id!r} has neither an event log nor messages this "
            f"reader understands, so there is nothing to draw."
        )

    scores, skipped = _scores(doc)
    return Imported(
        trace={
            "id": f"inspect-{wanted.id}-{wanted.epoch}",
            "name": f"{wanted.id} (Inspect sample)",
            "started_at": "",
            "steps": steps,
            # The store has always had a `meta` column and `import_trace` has
            # always written `doc.get("meta", {})` into it -- so every Inspect
            # sample ever imported landed in it with `{}`, and the scores, the
            # task and the model the reader had already parsed stopped at the
            # route's return value. A trace in the list could not say which
            # eval it came from, and the one thing an eval produces was not in
            # the store at all.
            "meta": {
                "source": SOURCE,
                "task": head.task,
                "model": head.model,
                "log_format_version": head.version,
                "sample_id": wanted.id,
                "epoch": wanted.epoch,
                # Clipped like `error` below and for the same reason: this
                # travels into a database column and an answer key can be a
                # whole document.
                "target": _text(doc.get("target"))[:400],
                "scores": scores,
                "skipped_scores": skipped,
            },
        },
        mapping=mapping,
        scores=scores,
        skipped_scores=skipped,
        failed=bool(doc.get("error")) or _failed(doc),
        error=_text(doc.get("error"))[:400],
    )


def _scores(doc: dict) -> tuple[dict, dict]:
    """The scores this reader can take a value from, and the ones it cannot.

    Returns `(scores, skipped)` -- `skipped` keyed by the scorer's name with
    the sentence saying what was there instead. It used to return the first
    half alone and throw the second away, which made it the one place in this
    module that dropped something without counting it, against the file's own
    docstring: "what was dropped is reported, not implied". A rubric whose
    value is a nested object then read as a sample that carried no rubric.
    """
    out: dict = {}
    skipped: dict = {}
    scores = doc.get("scores")
    if not isinstance(scores, dict):
        return out, skipped
    for name, entry in scores.items():
        if isinstance(entry, dict) and "value" not in entry:
            skipped[str(name)] = "the log's entry for it has no `value` field"
            continue
        value = entry.get("value") if isinstance(entry, dict) else entry
        if isinstance(value, (str, int, float, bool)):
            out[str(name)] = value
        else:
            # Named by TYPE rather than by content: the thing that was there
            # can be arbitrarily large, and a reader only needs to know the
            # shape they would have to change to get it read.
            skipped[str(name)] = (
                f"its value is a {type(value).__name__}, and a score is a "
                f"number, a string or a boolean here"
            )
    return out, skipped


def _failed(doc: dict) -> bool:
    """A sample counts as failed when a score says so, or an error is set.

    `I`/`C` are Inspect's incorrect/correct markers. A numeric 0 is NOT read as
    failure: a score of 0 on a 0-10 rubric is a low mark, not an error, and
    guessing otherwise would highlight the wrong sample.
    """
    for value in _scores(doc)[0].values():
        if isinstance(value, str) and value.upper() == "I":
            return True
    return False


# ------------------------------------------------------ the scores, in bulk


@dataclass
class ScoredSample:
    """One sample's identity, its inputs and what the eval scored it.

    Deliberately NOT a trace. Reading every sample's steps to answer "what did
    this eval score" would parse the whole run to build thousands of timelines
    nobody asked to see; this carries the columns an experiment row is made of
    and nothing else.
    """

    id: str
    epoch: int
    input_text: str = ""
    #: The expected answer the log states, or `None` when it states none.
    #: `None` and `""` are different claims and the dataset fingerprint tells
    #: them apart, so an absent target must not arrive here as an empty one.
    target: str | None = None
    #: What the model finally said, or `None` for a run that produced nothing.
    output: str | None = None
    scores: dict = field(default_factory=dict)
    skipped_scores: dict = field(default_factory=dict)
    error: str = ""
    failed: bool = False

    @property
    def case_id(self) -> str:
        """The id an experiment row joins on -- WITH THE EPOCH IN IT.

        A multi-epoch eval runs every sample id more than once, and an
        experiment refuses two rows under one case id because a comparison
        would silently pick whichever it read last. The epoch is what makes
        those runs distinct rows rather than a refusal on an ordinary log.
        """
        return f"{self.id}:{self.epoch}"

    def to_dict(self) -> dict:
        return asdict(self) | {"case_id": self.case_id}


@dataclass
class ScoredRun:
    """Every sample's scores, and the log's identity.

    `log_sha256` is the FULL digest of the file's bytes. Truncating it is the
    receipt writer's decision, not this reader's -- `receipts.DIGEST_CHARS`
    owns how short a provenance label gets to be, and this module does not
    import `receipts`.
    """

    header: EvalHeader
    rows: list = field(default_factory=list)
    n_total: int = 0
    log_name: str = ""
    log_sha256: str = ""

    @property
    def truncated(self) -> bool:
        return self.n_total > len(self.rows)

    def to_dict(self) -> dict:
        return {
            "header": self.header.to_dict(),
            "rows": [r.to_dict() for r in self.rows],
            "n_listed": len(self.rows),
            "n_total": self.n_total,
            "truncated": self.truncated,
            "log_name": self.log_name,
            "log_sha256": self.log_sha256,
        }


def _sha256(path: Path) -> str:
    """The file's own identity, read in chunks.

    In chunks because the route's limit is 200 MB and this is the one place
    that touches every byte -- `read_scores` otherwise reads the archive's
    directory and the members it wants. A digest is what lets two experiment
    files say whether they came from the same log, which no filename can.
    """
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample_output(doc: dict) -> str | None:
    """What the run finally said, or `None` when the log carries nothing.

    `None` rather than `""`: a run that emitted nothing and a run that emitted
    the empty string are different runs, and `datasets.Result.output` is
    documented to keep them apart.
    """
    text = _model_output({"output": doc.get("output")})
    if not text:
        events = doc.get("events")
        for event in reversed(events if isinstance(events, list) else []):
            if isinstance(event, dict) and event.get("event") == "model":
                text = _model_output(event)
                if text:
                    break
    return text or None


def read_scores(path) -> ScoredRun:
    """Every sample in the log, with the scores the eval gave it.

    The half of this reader that `read_sample` is not. `read_sample` draws ONE
    sample as a timeline, which is what the panel wants; this reads what the
    whole run measured, which is what an experiment is. Both are version-gated
    through `header()` and both leave no open handle behind.

    Capped by `samples()` at `MAX_SAMPLES_LISTED`, and the cap is REPORTED
    through `truncated` rather than substituted for the log's size — the same
    correction the listing path already carries, because an experiment 5,000
    rows into a 6,000-sample eval that says nothing is a run somebody would
    compare as if it were complete.
    """
    where = Path(str(path)).expanduser()
    head = header(where)
    refs = samples(where)
    if not refs:
        raise InspectError(
            "this log carries no samples, so there is nothing here that was scored."
        )

    rows: list[ScoredSample] = []
    with _open(where) as archive:
        for ref in refs:
            doc = _load(archive, ref.name)
            scores, skipped = _scores(doc)
            raw_target = doc.get("target")
            rows.append(
                ScoredSample(
                    id=ref.id,
                    epoch=ref.epoch,
                    input_text=_model_input({"input": doc.get("input")}),
                    target=None if raw_target is None else _text(raw_target),
                    output=_sample_output(doc),
                    scores=scores,
                    skipped_scores=skipped,
                    error=_text(doc.get("error"))[:400],
                    failed=bool(doc.get("error")) or _failed(doc),
                )
            )

    return ScoredRun(
        header=head,
        rows=rows,
        n_total=refs.n_total,
        # The NAME, never the path. A receipt built from this travels to
        # whoever the finding is forwarded to, and `receipts.public_name`
        # exists because the `.mri` exporter learned that the hard way.
        log_name=where.name,
        log_sha256=_sha256(where),
    )
