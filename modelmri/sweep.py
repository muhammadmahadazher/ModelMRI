"""One prompt is an anecdote. This is the loop that makes it a distribution.

"A number measured once is a sample, not a property" is the line this project
opens with, and until now it was a line. Every panel measures one prompt and
shows one number: layer 6 head 9 scores 0.41 nats, and nothing on screen says
whether that is what this head does or what it did that once.

`modelmri sweep` runs the same measurement over a set of prompts and reports
each head as **median, IQR, n, and how often it reached the top five** instead
of as a number. A head that tops one prompt and sits at rank 40 on the other
nineteen displays as exactly that, which is the fact you actually wanted.

THREE RULES THIS ENFORCES RATHER THAN DOCUMENTS

1. **Never a mean without a spread.** Every aggregate here is an order
   statistic. A mean over twenty prompts hides the head that carried one of
   them entirely, and that head is usually the interesting one.

2. **A refusal is a row, not a gap.** A prompt the measurement cannot be taken
   on is written out with the sentence saying why, in `could_not_measure`. If
   refusals were skipped, the output file would quietly describe only the
   prompts that happened to work, and its `n` would be a different number for
   every head with nothing saying so.

3. **Position metrics are not aggregated across prompts.** Head and feature
   identities mean the same thing in every prompt -- layer 6 head 9 is the
   same head, feature 4021 is the same feature. Token POSITION 3 is a
   different token in every prompt, so averaging over it produces a number
   about nothing. That boundary is a check in the code, not a warning in a
   docstring.

It runs headless: a `ModelRuntime` directly, no FastAPI and no browser, so it
works over SSH.
"""

from __future__ import annotations

import json
import sqlite3
import statistics
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import paths, receipts
from .errors import BadRequest, Refusal

# What a head has to reach to count as "top". Five because that is what the
# panel shows and what a reader means by "the heads that matter"; it travels
# in the output so a different reader can disagree with it.
TOP_K = 5

# Metrics whose row identity is stable across prompts. Anything not here is
# per-prompt only -- see rule 3 above.
AGGREGATABLE = {"heads", "features"}

METRICS = ("heads", "tokens", "features")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sweep (
    id          TEXT PRIMARY KEY,
    started_at  TEXT NOT NULL,
    model       TEXT,
    metric      TEXT NOT NULL,
    n_prompts   INTEGER NOT NULL,
    n_measured  INTEGER NOT NULL,
    n_refused   INTEGER NOT NULL,
    job         TEXT NOT NULL,
    rows_json   TEXT NOT NULL
);
"""


@dataclass
class Job:
    """What to run, over what. Instantiated from a CLI or a small JSON file."""

    model: str
    prompts: list[str]
    metric: str = "heads"
    baseline: str = "zero"
    # None sweeps every layer, which is the expensive one -- the projection
    # below says so before anything runs.
    layer: int | None = None
    max_new_tokens: int = 8
    top_k: int = TOP_K
    # One `.mri` per prompt lands here when set, so a single prompt from the
    # sweep can be opened, forwarded or verified like any other finding.
    out_dir: Path | None = None

    def validated(self) -> Job:
        if self.metric not in METRICS:
            raise BadRequest(
                f"unknown metric {self.metric!r} — use one of {', '.join(METRICS)}"
            )
        kept = [p for p in self.prompts if isinstance(p, str) and p.strip()]
        if not kept:
            raise BadRequest(
                "this sweep has no prompts. Pass a .jsonl or .txt with one "
                "prompt per line."
            )
        return Job(**{**asdict(self), "prompts": kept})


@dataclass
class Row:
    """One prompt's outcome. Written whether or not the measurement worked."""

    index: int
    prompt_sha256: str
    n_prompt_tokens: int | None = None
    # (layer, head) or feature id, as a string key, to whatever was scored.
    scores: dict = field(default_factory=dict)
    top: list = field(default_factory=list)
    # The sentence saying why this prompt has no scores. Empty when it does.
    # NOT a bool: "it failed" is not actionable and this always is.
    could_not_measure: str = ""
    receipt: dict = field(default_factory=dict)
    mri: str = ""

    @property
    def measured(self) -> bool:
        return not self.could_not_measure

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Stat:
    """One head or feature, over every prompt it was measured on."""

    key: str
    n: int
    median: float
    q1: float
    q3: float
    lo: float
    hi: float
    top_k_hits: int
    # How often it reached the top k, as a fraction of the prompts it was
    # actually measured on -- NOT of the prompts in the job. A head measured
    # on 3 of 20 prompts that topped all 3 is 1.0 here and `n=3` beside it,
    # and reporting 3/20 would be a different claim.
    top_k_rate: float

    @property
    def iqr(self) -> float:
        return self.q3 - self.q1

    def to_dict(self) -> dict:
        return {**asdict(self), "iqr": self.iqr}


# ------------------------------------------------------------------ planning


def plan(job: Job, runtime) -> dict:
    """What this will cost, before any of it is paid.

    The pass count is the portable part of the cost -- `ablate.py` records the
    same model measuring between 12 and 71 ms/pass across sessions on one
    RTX 4060, so a figure in seconds would be fiction. This reports passes and
    the per-pass rate only if the machine has actually measured one.
    """
    job = job.validated()
    model = runtime.model
    if model is None:
        raise Refusal("No model is loaded, so there is nothing to project.")

    n_layers = int(getattr(model.config, "num_hidden_layers", 0) or 0)
    n_heads = int(getattr(model.config, "num_attention_heads", 0) or 0)

    if job.metric == "heads":
        # REFUSED, not defaulted. `getattr(config, ..., 0) or 0` turns an
        # architecture that names these fields differently into a model with
        # no layers and no heads, and the arithmetic below then quotes a
        # whole-model head sweep at 2 passes per prompt instead of
        # n_layers x n_heads + 2 -- 2 against 450 on Qwen3-0.6B.
        #
        # `model_diff.head_pass_estimate` already refuses exactly this and
        # says why: "A preflight that under-quotes is worse than no
        # preflight, because it is the number somebody plans around." Same
        # defect, same answer.
        if n_layers <= 0 or n_heads <= 0:
            raise Refusal(
                f"this model's config does not state how many layers and "
                f"attention heads it has (read {n_layers} and {n_heads}), so "
                f"the cost of a head sweep cannot be projected. Running it "
                f"blind is the thing this preflight exists to prevent."
            )
        per_prompt = (n_layers * n_heads if job.layer is None else n_heads) + 2
        # The resample baseline draws several times per head, and the draws
        # multiply through every prompt. This is the number the caveat in the
        # roadmap is about.
        if job.baseline == "resample":
            from . import ablate

            per_prompt = per_prompt * int(getattr(ablate, "RESAMPLE_DRAWS", 1) or 1)
    elif job.metric == "tokens":
        per_prompt = 2
    else:
        per_prompt = 2

    # Every prompt is generated first, and a generation is one pass per token.
    per_prompt += job.max_new_tokens

    return {
        "prompts": len(job.prompts),
        "passes_per_prompt": per_prompt,
        "passes_total": per_prompt * len(job.prompts),
        "metric": job.metric,
        "baseline": job.baseline if job.metric == "heads" else None,
        "layer": job.layer,
        "aggregatable": job.metric in AGGREGATABLE,
    }


# ------------------------------------------------------------------- running


def _score_heads(result: dict) -> tuple[dict, list]:
    rows = result.get("ranked") or []
    scores = {f"L{r['layer']}H{r['head']}": float(r["kl"]) for r in rows}
    top = [k for k, _ in sorted(scores.items(), key=lambda kv: -kv[1])]
    return scores, top


def _score_features(result: dict) -> tuple[dict, list]:
    rows = result.get("ranked") or []
    scores = {}
    for r in rows:
        fid = r.get("feature") if "feature" in r else r.get("feature_id")
        if fid is None:
            continue
        scores[f"F{fid}"] = float(r.get("kl") or 0.0)
    top = [k for k, _ in sorted(scores.items(), key=lambda kv: -kv[1])]
    return scores, top


def _score_tokens(result: dict) -> tuple[dict, list]:
    """Token attribution rows key on `index`, and carry the token itself.

    The key is index AND text, because the index alone is what makes this
    metric unaggregatable -- position 3 is a different word in every prompt --
    and a reader scanning the per-prompt JSONL should be able to see that
    without cross-referencing anything.
    """
    rows = result.get("ranked") or []
    scores = {}
    for r in rows:
        index = r.get("index")
        if index is None:
            continue
        token = str(r.get("token") or "")
        scores[f"P{index}{token!r}"] = float(r.get("kl") or 0.0)
    top = [k for k, _ in sorted(scores.items(), key=lambda kv: -kv[1])]
    return scores, top


def run(job: Job, runtime, *, on_row=None, cancel: threading.Event | None = None):
    """Measure every prompt. Returns the rows, refusals included.

    `cancel` is honoured between prompts rather than inside one: a measurement
    stopped half way has hooks hanging on the model, and the partial numbers
    it would leave are worse than the wait.
    """
    job = job.validated()
    rows: list[Row] = []

    for index, prompt in enumerate(job.prompts):
        if cancel is not None and cancel.is_set():
            # A ROW, NOT A GAP -- the same rule the `except` arm below states
            # and this did not follow. Breaking out left the unrun prompts
            # with no row and no note, so a sweep cancelled after 3 of 20
            # prompts was saved and rendered as a complete 3-prompt sweep. The
            # aggregate, the top-k and the "measured on N prompts" line were
            # all true of a run nobody meant to take as final, and nothing on
            # screen or in the JSONL said it had been stopped.
            #
            # `measured` is False for these, so `aggregate` skips them exactly
            # as it skips a refusal.
            for later, unrun in enumerate(job.prompts[index:], start=index):
                rows.append(
                    Row(
                        index=later,
                        prompt_sha256=receipts.digest(unrun),
                        could_not_measure=(
                            "the sweep was cancelled before this prompt ran"
                        ),
                    )
                )
            break
        row = Row(index=index, prompt_sha256=receipts.digest(prompt))
        try:
            # Greedy, always. A sweep exists to compare prompts against each
            # other, and sampling would put a second source of variation into
            # a measurement whose entire subject is variation.
            list(runtime.generate_stream(prompt, job.max_new_tokens, temperature=0.0))
            row.n_prompt_tokens = runtime.last_n_prompt_tokens

            if job.metric == "heads":
                result = runtime.ablate_heads(layer=job.layer, baseline=job.baseline)
                row.scores, row.top = _score_heads(result)
            elif job.metric == "features":
                result = runtime.rank_features()
                row.scores, row.top = _score_features(result)
            else:
                result = runtime.attribute_tokens()
                row.scores, row.top = _score_tokens(result)
            row.receipt = result.get("receipt") or {}

            if job.out_dir is not None:
                # One `.mri` per prompt, so any single row of the sweep can be
                # opened, forwarded or verified like any other finding.
                target = Path(job.out_dir) / f"{index:04d}.mri"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(
                    runtime.export_session(note=f"sweep row {index}: {prompt[:60]}")
                )
                row.mri = target.name
        except (BadRequest, Refusal) as err:
            # A ROW, NOT A GAP. These sentences already say what would make the
            # measurement work -- a control token at the attribution position,
            # a model with no SAE loaded -- and dropping them would leave an
            # output file that silently describes only the prompts that
            # happened to work.
            row.could_not_measure = str(err)  # leak-ok: authored
        rows.append(row)
        if on_row is not None:
            on_row(row, len(job.prompts))
    return rows


# --------------------------------------------------------------- aggregating


def aggregate(rows: list[Row], *, metric: str, top_k: int = TOP_K) -> list[Stat]:
    """Order statistics per head, over the prompts each was measured on.

    Refuses outright for a metric whose rows are not the same thing in every
    prompt. Layer 6 head 9 is the same head in every prompt and feature 4021
    is the same feature, so those aggregate. Token position 3 is a different
    token in every prompt -- averaging it produces a number about nothing, and
    it is refused here rather than computed and captioned with a warning.
    """
    if metric not in AGGREGATABLE:
        raise Refusal(
            f"a {metric} sweep is not aggregated across prompts. Position 3 is "
            f"a different token in every prompt, so a median over it would be "
            f"a number about nothing. The per-prompt rows are in the JSONL."
        )

    measured = [r for r in rows if r.measured]
    collected: dict[str, list[float]] = {}
    hits: dict[str, int] = {}
    for row in measured:
        for key, value in row.scores.items():
            collected.setdefault(key, []).append(value)
        for key in row.top[:top_k]:
            hits[key] = hits.get(key, 0) + 1

    stats: list[Stat] = []
    for key, values in collected.items():
        ordered = sorted(values)
        n = len(ordered)
        # `quantiles` needs at least two points. With one, the median IS the
        # whole distribution and q1/q3 are that same number -- reported as
        # such, with n=1 beside it, rather than as a spread that was never
        # measured.
        if n >= 2:
            q1, _, q3 = statistics.quantiles(ordered, n=4, method="inclusive")
        else:
            q1 = q3 = ordered[0]
        stats.append(
            Stat(
                key=key,
                n=n,
                median=statistics.median(ordered),
                q1=q1,
                q3=q3,
                lo=ordered[0],
                hi=ordered[-1],
                top_k_hits=hits.get(key, 0),
                top_k_rate=(hits.get(key, 0) / n) if n else 0.0,
            )
        )
    # By median, then by how reliably it reaches the top. A head with a high
    # median on two prompts and a head with the same median on twenty are not
    # the same finding, and `n` travels with every row so the reader can see
    # which they are looking at.
    stats.sort(key=lambda s: (-s.median, -s.top_k_rate))
    return stats


# ------------------------------------------------------------------ output


def write_jsonl(rows: list[Row], path: str | Path) -> Path:
    """One row per prompt, refusals included.

    JSONL rather than CSV precisely so a refusal row can carry its sentence:
    a CSV column holding "the attribution position is a control token, so
    there is nothing to attribute" needs quoting rules nobody gets right, and
    the row would end up dropped instead.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row.to_dict(), allow_nan=False) + "\n")
    return target


def _db() -> sqlite3.Connection:
    path = paths.trace_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(path))
    db.executescript(_SCHEMA)
    return db


def save(job: Job, rows: list[Row], *, started_at: str, sweep_id: str) -> str:
    """Persist beside the traces, so a sweep is findable after the shell closes."""
    db = _db()
    try:
        db.execute(
            "INSERT OR REPLACE INTO sweep "
            "(id, started_at, model, metric, n_prompts, n_measured, n_refused, "
            " job, rows_json) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                sweep_id,
                started_at,
                job.model,
                job.metric,
                len(rows),
                sum(1 for r in rows if r.measured),
                sum(1 for r in rows if not r.measured),
                json.dumps(
                    {**asdict(job), "out_dir": str(job.out_dir) if job.out_dir else ""},
                    allow_nan=False,
                ),
                json.dumps([r.to_dict() for r in rows], allow_nan=False),
            ),
        )
        db.commit()
    finally:
        db.close()
    return sweep_id


def saved_sweeps(limit: int = 50) -> list[dict]:
    """Every sweep on this machine, newest first, with how far each one got.

    `save` has existed since the sweep did and nothing ever read it back, so a
    saved sweep was write-only: findable in the database and unreachable from
    the tool that wrote it.
    """
    db = _db()
    try:
        rows = db.execute(
            "SELECT id, started_at, model, metric, n_prompts, n_measured, "
            "n_refused FROM sweep ORDER BY started_at DESC LIMIT ?",
            (max(1, int(limit)),),
        ).fetchall()
    finally:
        db.close()
    return [
        {
            "sweep_id": r[0],
            "started_at": r[1],
            "model": r[2],
            "metric": r[3],
            "n_prompts": r[4],
            "n_measured": r[5],
            "n_refused": r[6],
            # The number that decides whether resuming is worth anything.
            "n_remaining": max(0, r[4] - r[5]),
            "complete": r[4] > 0 and r[5] >= r[4],
        }
        for r in rows
    ]


def load_sweep(sweep_id: str) -> tuple:
    """`(job, rows)` for a saved sweep, or a refusal naming the id."""
    db = _db()
    try:
        found = db.execute(
            "SELECT job, rows_json FROM sweep WHERE id = ?", (str(sweep_id),)
        ).fetchone()
    finally:
        db.close()
    if found is None:
        raise BadRequest(
            f"there is no saved sweep with id {sweep_id!r} on this machine. "
            f"Sweeps are stored beside the traces, so one run on another "
            f"machine is not here."
        )
    raw_job = json.loads(found[0])
    out_dir = raw_job.get("out_dir") or ""
    try:
        job = Job(**{**raw_job, "out_dir": Path(out_dir) if out_dir else None})
        rows = [Row(**r) for r in json.loads(found[1])]
    except TypeError as err:
        # A sweep saved by a version whose `Job` or `Row` had different fields.
        # `Job(**raw)` raises TypeError on both an unexpected key and a missing
        # one, and that reached the route as an unexplained 500 — a fact about
        # the FILE reported as a fault in the server.
        raise BadRequest(
            f"the saved sweep {sweep_id!r} was written by a different version "
            f"of ModelMRI and its stored shape no longer matches this one "
            f"({type(err).__name__}). Re-run it rather than resuming it; the "
            f"prompts are in the record and the measurements are not "
            f"transferable across a schema change."
        ) from None
    return job, rows


def _started_at(sweep_id: str) -> str:
    """When the sweep FIRST ran, kept across a resume.

    Finishing a run does not change when it began, and overwriting this with
    the resume's own clock would reorder `saved_sweeps` — putting a sweep
    started on Monday and finished on Friday above one that ran wholly on
    Thursday.
    """
    db = _db()
    try:
        found = db.execute(
            "SELECT started_at FROM sweep WHERE id = ?", (str(sweep_id),)
        ).fetchone()
    finally:
        db.close()
    return found[0] if found else ""


def remaining(job: Job, rows: list[Row]) -> list[int]:
    """Which prompt indices still need running.

    A row that was MEASURED is done. A row that was not is not a result — "the
    sweep was cancelled before this prompt ran" and "this prompt could not be
    measured" are both reasons to try again, and a resume that skipped them
    would report a partial sweep as finished.

    Retrying a prompt that genuinely cannot be measured costs one more attempt
    per resume and writes the same sentence again, which is the honest
    outcome: the alternative is a sweep that silently never covers it.
    """
    done = {r.index for r in rows if r.measured}
    return [i for i in range(len(job.prompts)) if i not in done]


def resume_plan(sweep_id: str, runtime=None) -> dict:
    """What finishing a saved sweep would cost, before it starts.

    Priced first like everything else here, and it also checks the three ways
    a resume can be WRONG rather than merely expensive — see `_resumable`.
    """
    job, rows = load_sweep(sweep_id)
    left = remaining(job, rows)
    blocked = _resumable(job, rows, runtime)
    return {
        "sweep_id": sweep_id,
        "model": job.model,
        "metric": job.metric,
        "n_prompts": len(job.prompts),
        "n_measured": sum(1 for r in rows if r.measured),
        "n_remaining": len(left),
        "remaining_indices": left,
        # `None` when nothing blocks it. A string is the reason it must not
        # run, and it is never merely a warning.
        "blocked": blocked,
        "means": (
            f"{len(job.prompts) - len(left)} of {len(job.prompts)} prompt(s) "
            f"already measured, {len(left)} left."
            + (f" This cannot be resumed: {blocked}" if blocked else "")
        ),
    }


def _resumable(job: Job, rows: list[Row], runtime) -> str | None:
    """Why this sweep must not be resumed, or `None`.

    Three checks, and each exists because failing it produces one table of
    numbers that came from two different runs — which looks exactly like a
    table of numbers that came from one.

    The prompt check is by DIGEST rather than by count. A set with the same
    number of prompts and one of them edited would otherwise attach the old
    row to the new prompt, and every number in it would be about text that is
    no longer there.
    """
    for row in rows:
        # Checked for EVERY row, not only measured ones. An unmeasured row
        # whose index is past the end is still evidence the prompt set moved,
        # and skipping it let a shortened set through the guard.
        if not isinstance(row.index, int) or isinstance(row.index, bool):
            return f"a saved row has index {row.index!r}, which is not a position"
        if row.index < 0:
            # Python indexes backwards from a negative, so -1 would quietly
            # read the LAST prompt and compare its digest against a row about
            # the first.
            return (
                f"a saved row has index {row.index}, and a negative position "
                f"reads backwards from the end rather than failing"
            )
        if row.index >= len(job.prompts):
            return (
                f"a saved row is index {row.index} and this set has only "
                f"{len(job.prompts)} prompt(s), so the prompts have changed. "
                f"Run this set as a new sweep — the saved rows measured a "
                f"different set and cannot be carried into it"
            )
        # Only a MEASURED row's digest matters: an unmeasured one carries no
        # numbers to attach to the wrong prompt.
        if not row.measured:
            continue
        if row.prompt_sha256 != receipts.digest(job.prompts[row.index]):
            return (
                f"prompt {row.index} is not the text that was measured — the "
                f"set has been edited, and reusing the old row would attach a "
                f"measurement to a prompt it was never about. Restore that "
                f"prompt to its original text to resume, or run the edited set "
                f"as a new sweep"
            )
    if runtime is not None:
        live = getattr(runtime, "hf_id", "") or ""
        if live and job.model and live != job.model:
            return (
                f"the saved run measured `{job.model}` and `{live}` is loaded "
                f"now. Finishing it would put two models' numbers in one "
                f"table. Load `{job.model}` to resume this one"
            )
    return None


def resume(sweep_id: str, runtime, *, on_row=None, cancel=None) -> tuple:
    """Finish a saved sweep, keeping what was already measured.

    A sweep that died at prompt 180 of 200 currently starts over, and losing
    four hours to a sleeping laptop is a worse failure than any missing
    feature. This runs the remainder and returns `(job, rows)` in prompt
    order, so the result is indistinguishable from a run that never stopped —
    except that it happened faster.
    """
    job, rows = load_sweep(sweep_id)
    blocked = _resumable(job, rows, runtime)
    if blocked is not None:
        raise BadRequest(f"this sweep cannot be resumed: {blocked}.")

    keep = {r.index: r for r in rows if r.measured}
    left = remaining(job, rows)
    if not left:
        return job, [keep[i] for i in sorted(keep)]

    # PRICED, like a fresh run. A resume that skipped the projection let
    # somebody finish a 40,000-pass remainder without ever being shown the
    # number — the one thing `plan` exists to prevent, bypassed by the path
    # that is most likely to be long.
    partial = Job(**{**asdict(job), "prompts": [job.prompts[i] for i in left]})
    plan(partial, runtime)

    # Every prompt kept its ORIGINAL text and its original position. `run`
    # numbers rows from 0 over the job it is given, so the indices are
    # restored afterwards — without that, every resumed row would claim to be
    # prompt 0..n of the original set and the join back would be silently
    # wrong.
    #
    # `out_dir` is dropped for the same reason: `run` names one `.mri` per
    # prompt by POSITION, so a resume writing prompt 180 as position 0 would
    # overwrite the first prompt's file and leave two rows pointing at one
    # analysis. The files for the prompts being re-run are rewritten below,
    # under their real positions.
    resumed = Job(**{**asdict(partial), "out_dir": None})
    fresh = run(resumed, runtime, on_row=on_row, cancel=cancel)

    if len(fresh) != len(left):
        # `Job.validated` drops prompts that are empty or whitespace, so `run`
        # can return fewer rows than positions asked for. `zip(strict=True)`
        # turned that into a bare ValueError; it is a fact about the saved
        # prompt set and it gets a sentence.
        raise BadRequest(
            f"this sweep asked to finish {len(left)} prompt(s) and the run "
            f"produced {len(fresh)} row(s), which happens when a saved prompt "
            f"is empty or whitespace. Re-run it rather than resuming: the "
            f"positions no longer line up and pairing them would attach each "
            f"measurement to the wrong prompt."
        )

    for row, original in zip(fresh, left, strict=True):
        row.index = original

    merged = {**keep, **{r.index: r for r in fresh}}
    ordered = [merged[i] for i in sorted(merged)]

    # PERSISTED under the same id. Without this the database still advertised
    # the work that had just been done — `saved_sweeps` kept reporting
    # `n_remaining: 2` for a sweep that was finished, and resuming it again
    # would re-run the same prompts.
    save(job, ordered, started_at=_started_at(sweep_id), sweep_id=sweep_id)
    return job, ordered


def load_prompts(path: str | Path) -> list[str]:
    """Prompts from a .jsonl (one object with `prompt`) or a .txt (one a line).

    Both, because the file people already have is usually one of the two and
    asking them to convert it is a reason not to run the sweep at all.
    """
    target = Path(path)
    try:
        text = target.read_text(encoding="utf-8")
    except OSError as err:
        raise BadRequest(
            f"{target.name} could not be read ({err.strerror or type(err).__name__})"
        ) from None

    prompts: list[str] = []
    for n, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                raise BadRequest(
                    f"{target.name} line {n} starts with '{{' but is not JSON."
                ) from None
            value = obj.get("prompt") if isinstance(obj, dict) else None
            if not isinstance(value, str) or not value.strip():
                raise BadRequest(
                    f"{target.name} line {n} is JSON with no `prompt` string."
                )
            prompts.append(value)
        else:
            prompts.append(line)
    if not prompts:
        raise BadRequest(f"{target.name} has no prompts in it.")
    return prompts


def render(job: Job, rows: list[Row], stats: list[Stat], *, limit: int = 12) -> str:
    """The table a terminal reads."""
    measured = sum(1 for r in rows if r.measured)
    refused = len(rows) - measured
    out = [
        f"{job.metric} over {len(rows)} prompts on {job.model}"
        + (f" · baseline {job.baseline}" if job.metric == "heads" else ""),
        f"  {measured} measured · {refused} could not be measured",
        "",
    ]
    if job.metric not in AGGREGATABLE:
        # NOT "nothing to aggregate": these prompts were measured, and saying
        # they were refused would be false. They are per-prompt by design,
        # which is a different sentence and the one a reader needs.
        out.append(
            f"  a {job.metric} sweep is not aggregated across prompts —"
            f" position 3 is a different"
        )
        out.append(
            "  token in every prompt, so a median over it would describe nothing. The"
        )
        out.append(f"  {measured} per-prompt rows carry the scores.")
    elif not stats:
        out.append("  nothing to aggregate — every prompt was refused.")
    else:
        out.append(
            f"  {'head':<10} {'median':>9} {'IQR':>9} {'range':>19}  "
            f"{'n':>3}  top{job.top_k}"
        )
        for stat in stats[:limit]:
            out.append(
                f"  {stat.key:<10} {stat.median:>9.5f} {stat.iqr:>9.5f} "
                f"{stat.lo:>8.5f}–{stat.hi:<10.5f} {stat.n:>3}  "
                f"{stat.top_k_hits}/{stat.n} ({stat.top_k_rate:.0%})"
            )
        if len(stats) > limit:
            out.append(f"  … {len(stats) - limit} more in the JSONL")
    if refused:
        out.append("")
        out.append("  could not be measured:")
        for row in rows:
            if not row.measured:
                out.append(f"    prompt {row.index}: {row.could_not_measure}")
    out.append("")
    out.append(
        "  Median and IQR, never a mean: a head that carries one prompt and "
        "nothing else\n  is the interesting one, and a mean hides it. `n` is "
        "how many prompts each head\n  was measured on, which is not always "
        "the number of prompts in the job."
    )
    return "\n".join(out)
