"""What does feature 14203 fire on, and what does it push toward?

"Feature 14203 fired" is not a finding. This gives that feature three
independent readouts, and they are independent on purpose — a claim that
survives all three is worth something that a claim resting on one is not:

  what it fires on     top-activating spans in YOUR corpus, its firing rate,
                       and the shape of its activations
  what it pushes at    the tokens it promotes and suppresses in vocabulary
                       space — needing no corpus and EXACT, because it is
                       pure weight math
  what removing it does the causal ranking `feature_ablate` already produces

NOTHING IS DOWNLOADED. The corpus is a local `.txt` or `.jsonl` you point at.
Every dashboard in this category shows features from a model and an SAE
somebody else chose, on a corpus somebody else picked; the numbers here are
about the text you handed it, and that text is named beside every one of them.

THE CORPUS IS PART OF THE RESULT

A top activation is a top activation IN THIS CORPUS. Its name, its token count
and the fraction of features that never fired sit next to the dashboard rather
than in a footnote, because "feature 14203 fires on legal citations" and
"feature 14203's highest activation in the 40,000 tokens you gave it was on a
legal citation" are different claims and only the second one was measured.

**A feature with zero activations here is "not seen in this corpus", never
"dead".** The difference matters: dead means the feature does nothing, not seen
means you did not show it anything it responds to, and only one of those is
about the model.

NO NATURAL-LANGUAGE LABELS. This shows what fired and what it promotes. Naming
the concept is the reader's job, and a generated label would be the one part of
the page nothing measured.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

from . import paths
from .errors import BadRequest, Refusal

# Activations are one-sided and heavy-tailed -- most are zero and the
# interesting ones are far out. Fixed bins over [0, max] so two features in the
# same corpus are directly comparable, which quantile bins would not be.
HISTOGRAM_BINS = 20

# Tokens either side of the peak in a reported span. Enough to see what the
# feature is responding to, short enough that twenty of them fit on a screen.
SPAN_CONTEXT = 6

# A cap on what one sweep will read, so pointing this at a 2 GB log does not
# quietly become a job. What it drops is NAMED in the response.
MAX_TOKENS = 200_000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS feature_activation (
    corpus_sha256 TEXT NOT NULL,
    model         TEXT NOT NULL,
    sae           TEXT NOT NULL,
    layer         INTEGER NOT NULL,
    feature       INTEGER NOT NULL,
    n_fired       INTEGER NOT NULL,
    max_act       REAL NOT NULL,
    PRIMARY KEY (corpus_sha256, model, sae, layer, feature)
);
"""


@dataclass
class Span:
    """One place a feature fired, with enough context to read it."""

    text: str
    token: str
    activation: float
    position: int
    sequence: int
    # Where the firing token STARTS inside `text`, in characters.
    #
    # Not derivable by searching for `token`: "The appeals court disagreed
    # with the trial court's" contains "court" twice and only one of those
    # positions fired. A reader shown both highlighted is being told the
    # feature fired twice there. The index is free here and unrecoverable
    # downstream, so it travels with the span.
    offset: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CorpusStats:
    corpus_label: str
    corpus_sha256: str
    n_sequences: int
    n_tokens: int
    n_features: int
    n_never_fired: int
    truncated: bool = False
    layer: int = 0

    @property
    def never_fired_share(self) -> float:
        return (self.n_never_fired / self.n_features) if self.n_features else 0.0

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "never_fired_share": round(self.never_fired_share, 4),
            "means": self.means(),
        }

    def means(self) -> str:
        return (
            f"Measured on {self.corpus_label}: {self.n_tokens:,} tokens in "
            f"{self.n_sequences} sequences, read at layer {self.layer}. "
            f"{self.n_never_fired:,} of {self.n_features:,} features "
            f"({self.never_fired_share:.1%}) never fired here — that is NOT "
            f"SEEN IN THIS CORPUS, not dead: it means this text never showed "
            f"them anything they respond to. Every activation below is a top "
            f"activation IN THIS TEXT, which is a different claim from what "
            f"the feature fires on generally."
            + (
                f" The corpus was cut at {MAX_TOKENS:,} tokens; what came "
                f"after it was not read."
                if self.truncated
                else ""
            )
        )


def _inside(candidate: str, roots: list) -> bool:
    """Is `candidate` at or under one of `roots`? Pure string comparison.

    `normcase` because Windows paths compare case-insensitively and neither
    `normpath` nor `resolve` reliably fixes the case of every component, so
    two spellings of one user's home directory that differ only in case are
    the same place, and a raw `startswith` would refuse one of them.

    The `+ sep` is the classic prefix bug and is not optional: without it a
    root of `/home/ana` accepts `/home/anabel`, which is a different person's
    home directory. There is a test for exactly that, added after removing
    this guard left every other test green.
    """
    here = os.path.normcase(candidate)
    for root in roots:
        prefix = os.path.normcase(str(root))
        if here == prefix or here.startswith(prefix.rstrip(os.sep) + os.sep):
            return True
    return False


def resolve_corpus(path: str | Path) -> Path:
    """The path, normalised, or a refusal — for a path that arrived over HTTP.

    THE BOUNDARY IS HERE AND NOT IN `sweep.load_prompts`, and the difference
    is who is asking. `modelmri sweep --prompts ~/corpus.txt` is the person at
    the keyboard naming their own file; they can already read anything that
    process can, and refusing a path its own user just typed protects nobody.
    A path that arrived in a request body is a different thing, even on
    loopback, and `..` in one is not a corpus.

    TWO CHECKS, IN THIS ORDER, AND THE ORDER IS THE POINT.

    First a LEXICAL one. `expanduser` + `abspath` + `normpath` are pure string
    operations — they collapse `..` and anchor a relative path without asking
    the filesystem anything — and the result is checked against the roots
    before this function touches a disk at all.

    Then a SYMLINK one. `Path.resolve()` follows links, so a symlink sitting
    inside your home directory and pointing at `/etc/shadow` passes the
    lexical check and fails this one. Only a path that survives both is
    returned.

    WHY THE LEXICAL CHECK COMES FIRST, rather than just resolving and checking
    once. `Path.resolve()` reads the filesystem, so it IS a path access, and
    CodeQL flags it as the sink: #418 closed on `sweep.py` and reopened as
    #431 pointing at the `resolve()` call in the first version of this
    function. No check placed after that line can clear it, because the access
    has already happened. Guarding first is what makes the boundary visible to
    a taint tracker — and it is better security besides, since an unchecked
    `resolve()` on a hostile path is a filesystem probe in its own right.
    """
    try:
        expanded = os.path.expanduser(str(path))
    except (OSError, ValueError, RuntimeError):
        raise BadRequest(_unresolvable(path)) from None

    if chr(0) in expanded:
        # `abspath` raises ValueError on an embedded NUL on some platforms and
        # silently truncates on others. Named here rather than left to differ.
        raise BadRequest(_unresolvable(path))

    try:
        lexical = os.path.normpath(os.path.abspath(expanded))
    except (OSError, ValueError, RuntimeError):
        raise BadRequest(_unresolvable(path)) from None

    roots = paths.corpus_roots()
    if not _inside(lexical, roots):
        raise BadRequest(_outside(os.path.basename(lexical) or lexical, roots))

    # Past the guard, so the filesystem may be asked. `strict=False` keeps a
    # path that does not exist yet answerable — the reader below is what says
    # "no such file", with the name in it.
    try:
        target = Path(lexical).resolve(strict=False)
    except (OSError, ValueError, RuntimeError):
        raise BadRequest(_unresolvable(path)) from None

    if not _inside(str(target), roots):
        # Lexically inside, actually outside: a symlink pointing out of the
        # roots. The refusal says which, because "it is a link" is the fact
        # the reader needs and cannot see from the path they typed.
        raise BadRequest(
            f"{os.path.basename(lexical)!r} is a link that leads outside the "
            f"directories a corpus may be read from over HTTP: "
            f"{', '.join(str(r) for r in roots)}. The path itself is inside "
            f"one of them; what it points at is not."
        )
    return target


def _unresolvable(path: object) -> str:
    return (
        f"{str(path)!r} is not a path this machine can resolve. Check the "
        f"drive, and that no link in it points at itself."
    )


def _outside(name: str, roots: list) -> str:
    return (
        f"{name!r} resolves outside the directories a corpus may be "
        f"read from over HTTP: {', '.join(str(r) for r in roots)}. Move the "
        f"file under one of those, or name its directory in "
        f"MODELMRI_CORPUS_DIRS and restart. (`modelmri sweep --prompts` has "
        f"no such boundary — it is you naming your own file.)"
    )


def load_corpus(path: str | Path) -> tuple[list[str], str]:
    """(sequences, label) from a local `.txt` or `.jsonl`. Never a download.

    The same reader `modelmri sweep` and the tuned lens use, so a corpus file
    that works for one works for all three rather than being three
    nearly-identical formats.

    Every caller of this is a ROUTE, so the path is checked against
    `resolve_corpus` before it is opened.
    """
    from . import sweep as sweep_mod

    target = resolve_corpus(path)
    return sweep_mod.load_prompts(target), target.name


def corpus_hash(texts: list[str]) -> str:
    from . import tuned_lens

    return tuned_lens.corpus_hash(texts)


# ------------------------------------------------------- the exact half


def logit_weights(model, tokenizer, sae, feature_id: int, *, top_k: int = 10) -> dict:
    """Which tokens this feature promotes and suppresses.

    EXACT, and needs no corpus. A feature's decoder row is a direction in the
    residual stream; pushing it through the final norm and the unembedding says
    what it does to every logit. That is pure weight math -- no sampling, no
    text, nothing to be a sample OF.

    The one approximation is the norm's scale, which depends on the stream this
    direction would be added to. It is applied at unit scale here, so these are
    the RELATIVE tokens rather than absolute logit amounts, and the response
    says so.
    """
    import torch

    from .lens import _final_norm

    if sae is None:
        raise Refusal("No SAE loaded, so there are no feature directions to read.")
    n_features = int(sae.W_dec.shape[0])
    if not 0 <= feature_id < n_features:
        raise BadRequest(f"feature {feature_id} is outside this SAE's {n_features:,}.")

    head = model.get_output_embeddings()
    if head is None:
        raise Refusal(
            "this model has no output embedding, so a feature's effect on the "
            "vocabulary cannot be read."
        )
    norm = _final_norm(model)

    direction = sae.W_dec[feature_id].detach().float()
    with torch.no_grad():
        projected = norm(
            direction.to(next(model.parameters()).device).to(
                next(norm.parameters()).dtype
            )
        )
        logits = head(projected).float()
        # Against the vocabulary mean, for the same reason `ablate.py` uses KL
        # and `dla.py` subtracts it: softmax ignores a constant, so a direction
        # that lifts every logit equally has changed nothing.
        centred = logits - logits.mean()
        top = torch.topk(centred, min(top_k, centred.shape[-1]))
        bottom = torch.topk(-centred, min(top_k, centred.shape[-1]))

    return {
        "feature": feature_id,
        "promotes": [
            {"token": tokenizer.decode([int(i)]), "logit": round(float(v), 5)}
            for i, v in zip(top.indices, top.values, strict=True)
        ],
        "suppresses": [
            {"token": tokenizer.decode([int(i)]), "logit": round(float(-v), 5)}
            for i, v in zip(bottom.indices, bottom.values, strict=True)
        ],
        "exact": True,
        "means": (
            "What this feature's decoder direction does to the vocabulary, "
            "read straight through the final norm and the unembedding. NO "
            "CORPUS AND NO SAMPLING — this is weight arithmetic and it is the "
            "same every time. Values are relative to the vocabulary mean and "
            "at unit scale, so they rank tokens rather than predict logit "
            "amounts: the norm's real scale depends on the stream this "
            "direction would be added to."
        ),
    }


# ------------------------------------------------------ the corpus half


def _activations(model, block, tokenizer, sae, texts: list[str], device):
    """Yield (sequence index, token strings, [S, d_sae] activations).

    One sequence at a time and never held together. `[n_tokens, d_sae]` for a
    65k-feature SAE over a 200k-token corpus is 52 GB in float32, so the sweep
    accumulates per-feature statistics as it goes and keeps no history.
    """
    import torch

    # AT THE SAE'S OWN HOOK POINT. This used to call `patch._capture`, which
    # is a forward PRE hook -- the stream ENTERING the block, i.e. resid_pre --
    # for every SAE regardless of where it was trained. `saes.py` records that
    # exact bug being found and fixed ("the hook POINT was previously
    # discarded... for a resid_post SAE that is the wrong side of the block"),
    # and the fix reached `runtime.py` and `feature_ablate.py` and not this
    # module. So a resid_post SAE loaded here was encoded from the block's
    # input: no error, no warning, and every number downstream -- the
    # never-fired count, the firing-rate table, the evidence spans and
    # histogram, and the rows written to the feature_activation table --
    # described activations the SAE was never trained on.
    #
    # `encode` does not catch it. It calibrates the input CONVENTION, which is
    # the same on both sides of a block; which side is a different question.
    from .feature_ablate import _register_capture

    for index, text in enumerate(texts):
        ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
        if ids.shape[-1] == 0:
            continue
        sink: list = []
        handle = _register_capture(block, sae.point, sink)
        try:
            with torch.no_grad():
                model(ids)
        finally:
            handle.remove()
        if not sink:
            raise Refusal("this model produced no residual stream at the SAE's layer.")
        acts = sae.encode(sink[0][0].float().cpu())
        yield index, [tokenizer.decode([int(t)]) for t in ids[0]], acts


def sweep(
    model,
    block,
    tokenizer,
    sae,
    texts: list[str],
    *,
    device,
    layer: int,
    corpus_label: str = "",
    corpus_sha: str = "",
) -> tuple[CorpusStats, dict]:
    """Firing rate and peak activation for every feature, over the corpus.

    Returns `(stats, per_feature)` where per_feature maps id -> (n_fired,
    max_act). Blocking; call from a worker thread.
    """
    import torch

    if sae is None:
        raise Refusal("No SAE loaded, so there are no features to sweep.")

    n_features = int(sae.W_dec.shape[0])
    fired = torch.zeros(n_features, dtype=torch.int64)
    peak = torch.zeros(n_features, dtype=torch.float32)

    n_tokens = 0
    n_sequences = 0
    truncated = False
    for _, tokens, acts in _activations(model, block, tokenizer, sae, texts, device):
        if n_tokens >= MAX_TOKENS:
            truncated = True
            break
        fired += (acts > 0).sum(dim=0).cpu()
        peak = torch.maximum(peak, acts.max(dim=0).values.cpu())
        n_tokens += len(tokens)
        n_sequences += 1

    stats = CorpusStats(
        corpus_label=corpus_label or f"{len(texts)} sequences",
        corpus_sha256=corpus_sha or corpus_hash(texts),
        n_sequences=n_sequences,
        n_tokens=n_tokens,
        n_features=n_features,
        n_never_fired=int((fired == 0).sum()),
        truncated=truncated,
        layer=layer,
    )
    per_feature = {
        i: (int(fired[i]), float(peak[i])) for i in range(n_features) if fired[i] > 0
    }
    return stats, per_feature


def evidence(
    model,
    block,
    tokenizer,
    sae,
    texts: list[str],
    feature_id: int,
    *,
    device,
    top_k: int = 10,
) -> dict:
    """Top-activating spans and the activation histogram for ONE feature.

    A second pass, because keeping the top spans of every feature at once is
    what makes this the memory-heaviest thing a dashboard can do. The sweep
    answers "which features fired"; this answers "show me that one".
    """
    import torch

    if sae is None:
        raise Refusal("No SAE loaded, so there is nothing to show.")
    n_features = int(sae.W_dec.shape[0])
    if not 0 <= feature_id < n_features:
        raise BadRequest(f"feature {feature_id} is outside this SAE's {n_features:,}.")

    spans: list[Span] = []
    values: list[float] = []
    n_tokens = 0
    truncated = False
    for index, tokens, acts in _activations(
        model, block, tokenizer, sae, texts, device
    ):
        # THE SAME CAP `sweep` APPLIES. It stopped at MAX_TOKENS and this
        # walked the whole corpus, so one response reported two different
        # sizes for "this corpus" -- the sweep's 200,000 beside this one's
        # full count -- and the firing rate shown for a feature was computed
        # over a different denominator from the rates it sits next to in the
        # table. Two numbers that look comparable and are not.
        if n_tokens >= MAX_TOKENS:
            truncated = True
            break
        column = acts[:, feature_id]
        n_tokens += len(tokens)
        for position in (column > 0).nonzero(as_tuple=True)[0].tolist():
            activation = float(column[position])
            values.append(activation)
            lo = max(0, position - SPAN_CONTEXT)
            hi = min(len(tokens), position + SPAN_CONTEXT + 1)
            spans.append(
                Span(
                    text="".join(tokens[lo:hi]),
                    token=tokens[position],
                    activation=round(activation, 5),
                    position=position,
                    sequence=index,
                    offset=len("".join(tokens[lo:position])),
                )
            )

    spans.sort(key=lambda s: -s.activation)
    histogram: list[int] = []
    edges: list[float] = []
    if values:
        column = torch.tensor(values)
        top = float(column.max())
        # Fixed bins over [0, max] rather than quantiles, so two features in
        # the same corpus have comparable shapes.
        histogram = torch.histc(column, bins=HISTOGRAM_BINS, min=0.0, max=top)
        histogram = [int(v) for v in histogram]
        edges = [round(top * i / HISTOGRAM_BINS, 5) for i in range(HISTOGRAM_BINS + 1)]

    return {
        "feature": feature_id,
        "spans": [s.to_dict() for s in spans[:top_k]],
        "n_fired": len(values),
        "n_tokens": n_tokens,
        "truncated": truncated,
        "firing_rate": round(len(values) / n_tokens, 6) if n_tokens else 0.0,
        "max_activation": round(max(values), 5) if values else 0.0,
        "histogram": histogram,
        "bin_edges": edges,
        "selective": bool(values and len(values) / max(1, n_tokens) < 0.2),
        "means": (
            f"Fired on {len(values):,} of {n_tokens:,} tokens "
            f"({(len(values) / n_tokens if n_tokens else 0):.2%}) in this "
            f"corpus. These are its highest activations HERE — a different "
            f"claim from what it fires on generally, and the corpus is named "
            f"beside them for that reason."
            + (
                f" The corpus was cut at {MAX_TOKENS:,} tokens, the same cut "
                f"the feature sweep makes, so this rate and the sweep's are "
                f"over the same denominator."
                if truncated
                else ""
            )
            + (
                f" A FIRING RATE THIS HIGH IS NOT A CONCEPT: a feature active "
                f"on {(len(values) / max(1, n_tokens)):.0%} of tokens is not "
                f"selecting anything, and its top activations will look like "
                f"whatever the corpus is mostly made of. Measured: the most "
                f"FREQUENTLY firing feature in an SAE fired on most of the "
                f"tokens and promoted an unrelated scatter of vocabulary — "
                f"all three readouts agreeing that it is not a clean concept, "
                f"which is why there are three."
                if values and len(values) / max(1, n_tokens) >= 0.2
                else ""
            )
            if values
            else (
                f"This feature did not fire once in {n_tokens:,} tokens. That "
                f"is NOT SEEN IN THIS CORPUS, not dead — this text never "
                f"showed it anything it responds to, which is a fact about "
                f"the text as much as about the feature."
                # The cut belongs HERE most of all. "this text never showed
                # it anything" is a claim about the whole corpus, and a
                # truncated pass only read the front of it -- so an unread
                # tail is exactly where the missing activation would be.
                + (
                    f" READ ONLY TO {MAX_TOKENS:,} TOKENS, though: the rest of "
                    f"this corpus was not looked at, so "
                    f'"not in this corpus" means "not in the part that was '
                    f'read".'
                    if truncated
                    else ""
                )
            )
        ),
        # NO LABEL. Naming the concept is the reader's job; a generated label
        # would be the one thing on this page that nothing measured.
        "label": None,
    }


# ------------------------------------------------------------ persistence


def _db() -> sqlite3.Connection:
    path = paths.trace_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(path))
    db.executescript(_SCHEMA)
    return db


def save(stats: CorpusStats, per_feature: dict, *, model: str, sae: str) -> int:
    """Persist a sweep beside the traces, so it survives the process."""
    db = _db()
    try:
        db.executemany(
            "INSERT OR REPLACE INTO feature_activation "
            "(corpus_sha256, model, sae, layer, feature, n_fired, max_act) "
            "VALUES (?,?,?,?,?,?,?)",
            [
                (stats.corpus_sha256, model, sae, stats.layer, f, n, a)
                for f, (n, a) in per_feature.items()
            ],
        )
        db.commit()
    finally:
        db.close()
    return len(per_feature)


def stored(corpus_sha: str, *, model: str, sae: str, layer: int, limit: int = 50):
    """The features that fired most often in a saved sweep."""
    db = _db()
    try:
        rows = db.execute(
            "SELECT feature, n_fired, max_act FROM feature_activation "
            "WHERE corpus_sha256=? AND model=? AND sae=? AND layer=? "
            "ORDER BY n_fired DESC LIMIT ?",
            (corpus_sha, model, sae, layer, limit),
        ).fetchall()
    finally:
        db.close()
    return [
        {"feature": f, "n_fired": n, "max_activation": round(a, 5)} for f, n, a in rows
    ]


def write_jsonl(rows: list[dict], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, allow_nan=False) + "\n")
    return target
