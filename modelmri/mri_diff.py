"""Did this change the model, or did it change the last digits?

`modelmri diff before.mri after.mri` compares two saved analyses of the same
prompt and says which heads moved in the ablation ranking, which patching sites
changed sign, which edges of the traced circuit stopped clearing their
controls, and whether the model still says the same thing. With
`--fail-over` it exits non-zero, so a repo can check in a baseline `.mri` and
have CI say **"your quantisation changed which heads carry this answer"** in
the pull request that did it.

Nothing in the category has a regression concept for model internals. The state
of the art for "did my quant change the model" is a Reddit thread and somebody
eyeballing two completions.

THE FLOOR IS NOT INVENTED

Every delta is compared against a floor the FILES supply. `session._quantise`
stores each attention matrix as uint8 against that matrix's own maximum, so
that block's `scale` is the smallest difference it can represent -- and a
comparison of two files cannot be finer than the coarser of the two. A
patching graph carries its own: `patch_graph` prunes at the trace's recovery
resolution, records the number and records the sentence saying where it came
from. There is no epsilon in this module that somebody chose.

And where no file supplies one, none is borrowed. A forwarded attribution
graph records no resolution of any kind, so this compares what needs no floor
-- which edges are there and which changed sign -- and reports the rest as
unavailable rather than as a magnitude of unknown significance. Which edges are
STRONGEST belongs to the second group and not the first: it is an ordering of
the same unjudgeable weights, and two a hair apart rank in whatever order the
last digits fell.

WHAT IT REFUSES TO COMPARE

- **Different prompts.** Different tokens, `n_prompt`, `n_layers` or `n_heads`
  means these are not two measurements of the same thing, and a diff of them is
  a category error rather than a small one.
- **Sampled runs.** Two `.mri` at temperature > 0 differ for reasons that are
  not the model. The `generate` receipt records `greedy`, so this is checked
  rather than assumed -- and a file too old to carry it is labelled unknown
  rather than presumed greedy.
- **Different dtype or device.** `patch.py` documents bf16 moving the reference
  gap from 4.000 to 4.467 and changing the reference token itself. A diff
  across float formats measures the formats.

A MISSING BLOCK IS NOT A ZERO. Each section is compared only when BOTH files
carry it, and its absence is reported as "not comparable" naming which side
lacked it. Treating an absent section as unchanged is exactly the bug class
this project keeps finding -- a 0.10 defect showed 206 robot episodes as one
video because a missing value became a default.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import fmt
from . import session as session_mod
from .errors import BadRequest
from .verify import dequantise, max_abs_diff

SAME = "same"
CHANGED = "changed"
NOT_COMPARABLE = "not comparable"

# How many moved heads to name in the terminal before saying "and N more".
NAMED_HEADS = 6


@dataclass
class Delta:
    """One section, compared or explicitly not compared."""

    name: str
    status: str
    detail: str
    # The magnitude and the floor it was judged against, in this metric's own
    # units. Units differ per metric and are named rather than blended into one
    # score that would mean nothing; which units a report contains is read off
    # the deltas themselves, so no list of them is written down twice and none
    # can fall behind the sections.
    #
    # `floor` stays None where no file supplies one -- a forwarded attribution
    # graph records no resolution at all -- and a `magnitude` of None means
    # the finding is categorical rather than small: a changed generation, a
    # flipped control verdict, a leader that moved. `exit_code` fails those
    # unconditionally for exactly that reason.
    magnitude: float | None = None
    floor: float | None = None
    unit: str = ""
    measured: dict = field(default_factory=dict)
    # WHETHER `--fail-over` CAN GATE THIS SECTION AT ALL, which is a property
    # of the section and not of how this particular run came out. Three
    # sections are categorical by construction -- a changed generation, a
    # logit-lens leader that moved, an attribution graph with no resolution
    # behind its weights -- and every CHANGED they can produce carries
    # `magnitude=None`, which `exit_code` fails at every threshold. Reading
    # eligibility off this run's magnitude instead selects almost exactly the
    # wrong set: their SAME verdicts carry `magnitude=0.0` and would be named
    # as gated, and their CHANGED verdicts, the ones that always fail, would
    # not be.
    gated: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DiffReport:
    a: str
    b: str
    model_a: str | None = None
    model_b: str | None = None
    deltas: list[Delta] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def changed(self) -> list[Delta]:
        return [d for d in self.deltas if d.status == CHANGED]

    def to_dict(self) -> dict:
        return {
            "a": self.a,
            "b": self.b,
            "model_a": self.model_a,
            "model_b": self.model_b,
            "deltas": [d.to_dict() for d in self.deltas],
            "notes": self.notes,
            "totals": {
                SAME: sum(1 for d in self.deltas if d.status == SAME),
                CHANGED: len(self.changed),
                NOT_COMPARABLE: sum(
                    1 for d in self.deltas if d.status == NOT_COMPARABLE
                ),
            },
        }

    def exit_code(self, fail_over: float | None = None) -> int:
        """Non-zero when something changed by more than the caller allows.

        With no `--fail-over`, anything that moved past the files' own floor
        fails: that is the strictest honest threshold, because a difference
        below the floor is not one the files can represent. With a number, it
        is compared in each metric's own units -- and a changed GENERATION
        always fails, because there is no magnitude at which "the model now
        says something else" is within tolerance.
        """
        for delta in self.changed:
            if delta.magnitude is None:
                return 1
            if fail_over is None or delta.magnitude > fail_over:
                return 1
        return 0


# ------------------------------------------------------------------ helpers


def _receipt(parsed, op: str) -> dict:
    for receipt in parsed.receipts or []:
        if receipt.get("op") == op:
            return receipt
    return {}


def _is_number(value) -> bool:
    """A real number, and `True` is not one."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _comparable(a, b) -> tuple[list[str], list[str]]:
    """(hard refusals, soft notes) about whether these two describe one thing."""
    refuse: list[str] = []
    notes: list[str] = []

    if a.tokens != b.tokens:
        # Named precisely, because "the prompts differ" and "the model
        # generated something else" are both possible and want different
        # actions from the reader.
        # strict=False deliberately: the lengths differ, which is exactly what
        # is being reported, and the useful number is how far they agreed
        # before diverging.
        shared = sum(1 for x, y in zip(a.tokens, b.tokens, strict=False) if x == y)
        refuse.append(
            f"these two files are not about the same run: {len(a.tokens)} "
            f"tokens against {len(b.tokens)}, agreeing on the first {shared}. "
            f"A diff needs two measurements of the same prompt."
        )
    for field_name, label in (
        ("n_prompt", "prompt length"),
        ("n_layers", "layer count"),
        ("n_heads", "head count"),
    ):
        left, right = getattr(a, field_name), getattr(b, field_name)
        if left != right:
            refuse.append(
                f"the {label} differs ({left} against {right}), so these are "
                f"not two measurements of the same thing."
            )

    gen_a, gen_b = _receipt(a, "generate"), _receipt(b, "generate")
    greedy = [
        (gen_a.get("request") or {}).get("greedy"),
        (gen_b.get("request") or {}).get("greedy"),
    ]
    if False in greedy:
        refuse.append(
            "one of these runs was sampled (temperature > 0), so the two "
            "differ for reasons that are not the model. Only greedy runs can "
            "be diffed."
        )
    elif None in greedy:
        notes.append(
            "at least one file does not record whether its generation was "
            "sampled, so a difference here cannot be told apart from the "
            "sampler. Files written by newer versions carry it."
        )

    for key, label in (("dtype", "dtype"), ("device", "device")):
        left = gen_a.get(key) or (a.meta or {}).get(key)
        right = gen_b.get(key) or (b.meta or {}).get(key)
        if left and right and left != right:
            notes.append(
                f"these were measured in different {label}s ({left} against "
                f"{right}). `patch.py` records bf16 moving the reference gap "
                f"from 4.000 to 4.467 and changing the reference token, so "
                f"every number below is a comparison of {label}s as much as of "
                f"models."
            )

    rev_a, rev_b = gen_a.get("revision"), gen_b.get("revision")
    if rev_a and rev_b and rev_a != rev_b:
        notes.append(
            f"different commits: {rev_a[:12]} against {rev_b[:12]}. These are "
            f"different weights, which is often the point of the diff."
        )
    return refuse, notes


# ------------------------------------------------------------------ sections


def _diff_generation(a, b) -> Delta:
    if a.generation == b.generation:
        return Delta(
            "generation",
            SAME,
            f"both files continue with {a.generation!r}.",
            magnitude=0.0,
            unit="text",
            gated=False,
        )
    return Delta(
        "generation",
        CHANGED,
        f"the model said {a.generation!r} and now says {b.generation!r}.",
        # No magnitude. There is no number at which "it says something else"
        # is within tolerance, and `exit_code` treats a magnitude of None as
        # unconditionally failing for exactly that reason. `gated=False` says
        # the same thing to `render`: "0.01 in text units" is not a threshold
        # anybody could set, so the sentence explaining `--fail-over` must not
        # offer it as one.
        magnitude=None,
        unit="text",
        measured={"a": a.generation, "b": b.generation},
        gated=False,
    )


def _diff_attention(a, b) -> Delta:
    shared = sorted(set(a.attention) & set(b.attention))
    if not shared:
        only_a, only_b = len(a.attention), len(b.attention)
        return Delta(
            "attention",
            NOT_COMPARABLE,
            f"these files have no head map in common ({only_a} in the first, "
            f"{only_b} in the second) — an export captures a slice of the cube "
            f"and these two slices do not overlap.",
        )

    # Seeded as "nothing looked at yet", NOT as a zero-margin pair. The
    # replacement test below is `gap - floor > worst_gap - worst_floor`, so a
    # (0.0, 0.0) seed reduces to `gap > floor` -- and in the SAME case, where
    # by definition no block exceeds its floor, nothing ever replaced it. The
    # answer was then reported as `max_abs_diff 0.0, floor 0.0`: a difference
    # of exactly zero and a quantisation step of exactly zero, neither of
    # which was measured and the second of which cannot exist. Two files whose
    # worst block differs by 1.97e-03 against a 3.92e-03 step agree, and are
    # entitled to have those numbers said about them.
    worst_key: str | None = None
    worst_gap = worst_floor = 0.0
    for key in shared:
        block_a, block_b = a.attention[key], b.attention[key]
        # The coarser of the two scales. Neither file can represent a
        # difference finer than its own quantisation step, so a comparison of
        # the pair cannot either.
        floor = max(float(block_a["scale"]), float(block_b["scale"]))
        gap = max_abs_diff(dequantise(block_a), dequantise(block_b))
        if gap is None:
            return Delta(
                "attention",
                NOT_COMPARABLE,
                f"head map {key} is a different shape in the two files.",
            )
        if worst_key is None or gap - floor > worst_gap - worst_floor:
            worst_key, worst_gap, worst_floor = key, gap, floor

    measured = {
        "blocks_compared": len(shared),
        "worst_block": worst_key,
        "max_abs_diff": worst_gap,
        "floor": worst_floor,
        "floor_from": "the coarser of the two files' uint8 quantisation steps",
    }
    if worst_gap <= worst_floor:
        return Delta(
            "attention",
            SAME,
            f"all {len(shared)} shared head maps agree to within the files' "
            f"own storage precision ({worst_gap:.2e} at {worst_key}, floor "
            f"{worst_floor:.2e}).",
            magnitude=worst_gap,
            floor=worst_floor,
            unit="attention weight",
            measured=measured,
        )
    return Delta(
        "attention",
        CHANGED,
        f"{worst_key} moved by {worst_gap:.2e}, past the {worst_floor:.2e} "
        f"these files can represent ({len(shared)} maps compared).",
        magnitude=worst_gap,
        floor=worst_floor,
        unit="attention weight",
        measured=measured,
    )


def _top_k_line(top: int, entered: list, left: list) -> str:
    """Which heads entered and left the top K, as a sentence.

    Extracted because two paths need it and a second copy of a sentence is a
    second chance for the two to disagree: the missing-floor branch reports
    exactly this finding, which does not depend on a floor at all.
    """
    return (
        f"the top {top} changed: "
        + ", ".join(f"L{h[0]}H{h[1]}" for h in entered)
        + " entered"
        + (
            " and " + ", ".join(f"L{h[0]}H{h[1]}" for h in left) + " left"
            if left
            else ""
        )
        + "."
    )


def _diff_ranking(a, b) -> Delta:
    rows_a = (a.ranking or {}).get("ranked") or []
    rows_b = (b.ranking or {}).get("ranked") or []
    if not rows_a or not rows_b:
        which = "the first" if not rows_a else "the second"
        return Delta(
            "head ranking",
            NOT_COMPARABLE,
            f"{which} file carries no head ranking. A missing section is not "
            f"a zero, so this comparison is unavailable rather than clean.",
        )

    base_a = (a.ranking or {}).get("baseline")
    base_b = (b.ranking or {}).get("baseline")
    if base_a != base_b:
        return Delta(
            "head ranking",
            NOT_COMPARABLE,
            f"these rankings used different baselines ({base_a} against "
            f"{base_b}), and the baselines disagree with each other — "
            f"`ablate.py` measures only a weak rank correlation between them. "
            f"Comparing across them would measure the baseline.",
        )

    scores_a = {(r["layer"], r["head"]): r["kl"] for r in rows_a}
    scores_b = {(r["layer"], r["head"]): r["kl"] for r in rows_b}
    shared = sorted(scores_a.keys() & scores_b.keys())
    if not shared:
        return Delta(
            "head ranking",
            NOT_COMPARABLE,
            "the two rankings cover no head in common.",
        )

    # The floor for a KL is whichever noise floor the files recorded, and the
    # LARGER of the two: a difference below the coarser measurement is not one
    # either file could distinguish from arithmetic.
    #
    # AN ABSENT FLOOR IS NOT A FLOOR OF ZERO. This read `... or 0.0` on both
    # sides, which turned "this file never recorded what it could resolve"
    # into "this file recorded that it could resolve everything" — and then
    # labelled the invented number "the coarser of the two files' recorded
    # noise floors". Because 0.0 is a legal recorded value, nothing downstream
    # could tell the fabrication from a measurement.
    #
    # Measured on two files whose rankings carry rows and a baseline and no
    # floor: `abs(gap) > 0.0` is true for any non-identical KL, so last-digit
    # drift reported as "L0H0 moved 0.10000 → 0.10000. 1 of 2 heads moved past
    # the 0.00e+00 noise floor", and `exit_code()` returned 1 — a CI failure
    # attributed to a floor no file ever claimed.
    #
    # This module states the rule three times for other sections ("A missing
    # section is not a zero") and broke it here.
    floor_a = (a.ranking or {}).get("noise_floor_kl")
    floor_b = (b.ranking or {}).get("noise_floor_kl")

    # WHICH HEADS ARE IN THE TOP FIVE needs no floor. It is a comparison of
    # two orderings, and it is the finding a reader acts on: "L5H3 entered and
    # L0H0 left" says the answer moved to a different head. Computed BEFORE
    # the floor is required, because the first version of this guard returned
    # NOT_COMPARABLE for a missing floor and took that finding down with it —
    # a real top-five change, reported as nothing, exit 0.
    top = min(5, len(shared))
    top_a = sorted(shared, key=lambda h: -scores_a[h])[:top]
    top_b = sorted(shared, key=lambda h: -scores_b[h])[:top]
    entered = [h for h in top_b if h not in top_a]
    left = [h for h in top_a if h not in top_b]

    if floor_a is None or floor_b is None:
        which = "the first" if floor_a is None else "the second"
        why = (
            f"{which} file's ranking records no noise floor, so no per-head "
            f"score difference can be judged against one. An absent floor is "
            f"not a floor of zero — with one, every last-digit difference "
            f"counts as a change. Re-run `modelmri ablate` on that side to "
            f"record one, or diff two files that both carry it."
        )
        if entered:
            # The ordering DID change, and that does not depend on a floor.
            # Reported as a change, so the exit code is the one the reader
            # expects, with the magnitude question named as unanswered.
            return Delta(
                "head ranking",
                CHANGED,
                f"{_top_k_line(top, entered, left)} {why}",
                magnitude=None,
                floor=None,
                unit="nats",
                measured={
                    "heads_compared": len(shared),
                    "top_k": top,
                    "entered_top_k": [f"L{h[0]}H{h[1]}" for h in entered],
                    "left_top_k": [f"L{h[0]}H{h[1]}" for h in left],
                    "floor": None,
                    "floor_from": "neither file recorded one",
                },
            )
        return Delta(
            "head ranking",
            NOT_COMPARABLE,
            f"the top {top} are the same heads in both files. Beyond that, {why}",
        )
    floor = max(float(floor_a), float(floor_b))

    moved = []
    for head in shared:
        gap = scores_b[head] - scores_a[head]
        if abs(gap) > floor:
            moved.append((abs(gap), head, scores_a[head], scores_b[head]))
    moved.sort(reverse=True)

    measured = {
        "heads_compared": len(shared),
        "heads_moved": len(moved),
        "max_abs_kl_diff": moved[0][0] if moved else 0.0,
        "floor": floor,
        "floor_from": "the coarser of the two files' recorded noise floors",
        "top_k": top,
        "entered_top_k": [f"L{h[0]}H{h[1]}" for h in entered],
        "left_top_k": [f"L{h[0]}H{h[1]}" for h in left],
        "moved": [
            {"head": f"L{h[0]}H{h[1]}", "from": was, "to": now, "delta": now - was}
            for _, h, was, now in moved[:NAMED_HEADS]
        ],
    }

    if not moved and not entered:
        return Delta(
            "head ranking",
            SAME,
            f"all {len(shared)} heads score within the recorded noise floor "
            f"({floor:.2e}) and the top {top} are the same heads.",
            magnitude=0.0,
            floor=floor,
            unit="nats",
            measured=measured,
        )

    # The headline is the ORDER when it changed, and the magnitude when it did
    # not. "Which head carries this answer" is what a reader acts on; a score
    # that drifted while the order held is a smaller claim and reads as one.
    if entered:
        head_line = _top_k_line(top, entered, left)
    else:
        _, head, was, now = moved[0]
        # THE MOVEMENT, not just the pair. `:.5f` on both sides printed a
        # head that moved by 1e-06 as "moved 0.10000 → 0.10000" — the same
        # number twice, in a sentence whose entire content is that it changed.
        # `fmt.measured` does not fix that on its own: 0.100001 is not below
        # five places' rounding floor, so it still renders "0.10000". Any
        # fixed precision has a pair it cannot separate.
        #
        # The delta is the quantity this sentence is about and it is never
        # ambiguous, so it is stated outright and the pair keeps its place as
        # context.
        step = now - was
        head_line = (
            f"the top {top} are unchanged, but L{head[0]}H{head[1]} moved by "
            f"{'+' if step >= 0 else '−'}{fmt.measured(abs(step), 5)}, from "
            f"{fmt.measured(was, 5)} to {fmt.measured(now, 5)}."
        )
    return Delta(
        "head ranking",
        CHANGED,
        f"{head_line} {len(moved)} of {len(shared)} heads moved past the "
        f"{floor:.2e} noise floor.",
        magnitude=moved[0][0] if moved else 0.0,
        floor=floor,
        unit="nats",
        measured=measured,
    )


def _diff_ground(a, b) -> Delta:
    """Did the same passage still carry the answer, and by the same margin?

    The regression this exists for: a finetune that stops reading the document
    and starts answering from its weights. That is invisible in every other
    section here — the generation can be word-for-word identical while the
    thing producing it has moved from the passage to the prior.

    Two questions, kept apart because they fail differently. WHICH passage
    tops the list is what a reader acts on; HOW FAR it moved the answer is the
    number they quote. A finetune can hold the ordering while every score
    halves, and it can hold the scores while the top passage changes.
    """
    rows_a = (a.ground or {}).get("chunks") or []
    rows_b = (b.ground or {}).get("chunks") or []
    if not rows_a or not rows_b:
        which = "the first" if not rows_a else "the second"
        return Delta(
            "grounding",
            NOT_COMPARABLE,
            f"{which} file carries no grounding. A missing section is not a "
            f"zero, so this comparison is unavailable rather than clean.",
        )

    question_a = (a.ground or {}).get("question") or ""
    question_b = (b.ground or {}).get("question") or ""
    if question_a != question_b:
        return Delta(
            "grounding",
            NOT_COMPARABLE,
            "these two groundings answer different questions, so a passage "
            "that mattered for one need not have been asked about in the "
            "other. Comparing them would measure the question.",
        )

    by_a = {int(r["index"]): r for r in rows_a}
    by_b = {int(r["index"]): r for r in rows_b}
    shared = sorted(by_a.keys() & by_b.keys())
    if not shared:
        return Delta(
            "grounding",
            NOT_COMPARABLE,
            "the two groundings share no passage index, so the documents were "
            "split differently and no passage can be compared.",
        )

    # Indices are only meaningful if they name the SAME passage. The preview
    # is the only text a `.mri` carries, and comparing it is what stops a
    # reordered document from being reported as a change in the model.
    moved = [
        i
        for i in shared
        if (by_a[i].get("preview") or "") != (by_b[i].get("preview") or "")
    ]
    if moved:
        return Delta(
            "grounding",
            NOT_COMPARABLE,
            f"passage #{moved[0]} is different text in the two files, so the "
            f"indices do not name the same passages. The document changed "
            f"between these runs, and a per-index comparison would report "
            f"that as a change in the model.",
        )

    # The coarser of the two recorded floors, for the same reason the ranking
    # takes the larger one: a difference below the blunter measurement is not
    # one either file could tell from arithmetic. And the same guard, because
    # this was the same substitution written twice — fixing one and leaving
    # the other is how the ground path keeps fabricating.
    g_floor_a = (a.ground or {}).get("noise_floor")
    g_floor_b = (b.ground or {}).get("noise_floor")
    if g_floor_a is None or g_floor_b is None:
        which = "the first" if g_floor_a is None else "the second"
        return Delta(
            "grounding",
            NOT_COMPARABLE,
            f"{which} file's grounding records no noise floor, so there is "
            f"nothing to judge a passage's movement against. An absent floor "
            f"is not a floor of zero. Re-run the grounding on that side to "
            f"record one, or diff two files that both carry it.",
        )
    floor = max(float(g_floor_a), float(g_floor_b))

    worst_index, worst_gap = shared[0], 0.0
    for i in shared:
        gap = abs(
            float(by_a[i].get("dependence") or 0.0)
            - float(by_b[i].get("dependence") or 0.0)
        )
        if gap > worst_gap:
            worst_index, worst_gap = i, gap

    top_a = max(shared, key=lambda i: float(by_a[i].get("dependence") or 0.0))
    top_b = max(shared, key=lambda i: float(by_b[i].get("dependence") or 0.0))

    # Grounded-ness itself, which is the headline. A file where nothing
    # cleared and one where something did are different findings even if every
    # score moved less than the floor.
    ungrounded_a = bool((a.ground or {}).get("ungrounded"))
    ungrounded_b = bool((b.ground or {}).get("ungrounded"))

    measured = {
        "passages_compared": len(shared),
        "max_abs_dependence_diff": worst_gap,
        "worst_passage": worst_index,
        "tolerance": floor,
        "top_passage_a": top_a,
        "top_passage_b": top_b,
        "ungrounded_a": ungrounded_a,
        "ungrounded_b": ungrounded_b,
    }

    if ungrounded_a != ungrounded_b:
        became = "stopped" if ungrounded_b else "started"
        return Delta(
            "grounding",
            CHANGED,
            f"the answer {became} depending on the document. One of these "
            f"files has no passage clearing its floor and the other does — "
            f"which is the regression this section exists to catch, and it is "
            f"invisible in a generation that may be word for word the same.",
            magnitude=worst_gap,
            floor=floor,
            unit="nats",
            measured=measured,
        )
    if top_a != top_b:
        return Delta(
            "grounding",
            CHANGED,
            f"a different passage now carries the answer: #{top_a} became "
            f"#{top_b}. The scores moved by at most {worst_gap:.4f}, so this "
            f"is a change in WHICH passage the answer rests on rather than in "
            f"how much any of them matters.",
            magnitude=worst_gap,
            floor=floor,
            unit="nats",
            measured=measured,
        )
    if worst_gap > floor:
        return Delta(
            "grounding",
            CHANGED,
            # TWO PRECISIONS ON ONE LINE, and the wrong way round. This
            # branch is only entered when `worst_gap > floor`, and it printed
            # the mover at four places and the floor it beat at six — so a
            # passage moving 5e-06 against a floor of 3e-06 read "moved by
            # 0.0000 nats against a floor of 0.000003", where the number that
            # cleared the bar looks like zero and the bar looks larger than it.
            f"passage #{worst_index} moved by {fmt.measured(worst_gap, 4)} nats "
            f"against a floor of {fmt.measured(floor, 6)}. #{top_a} still "
            f"carries the answer, so the "
            f"document is being read the same way and read to a different "
            f"degree.",
            magnitude=worst_gap,
            floor=floor,
            unit="nats",
            measured=measured,
        )
    if floor <= 0.0:
        # Both files recorded a zero floor, so `worst_gap <= floor` only holds
        # when the numbers are bit-identical. Saying SAME is right; implying a
        # tolerance was applied is not.
        return Delta(
            "grounding",
            SAME,
            f"all {len(shared)} passages score identically and #{top_a} still "
            f"carries the answer. Both files recorded a floor of exactly "
            f"zero, so this is bit-for-bit equality rather than agreement "
            f"within a tolerance.",
            magnitude=0.0,
            floor=floor,
            unit="nats",
            measured=measured,
        )
    return Delta(
        "grounding",
        SAME,
        f"all {len(shared)} passages score within {worst_gap:.4f} of each "
        f"other, under the {floor:.6f} floor these files recorded, and "
        f"#{top_a} carries the answer in both.",
        magnitude=worst_gap,
        floor=floor,
        unit="nats",
        measured=measured,
    )


def _diff_patch(a, b) -> Delta:
    grids_a = (a.patch or {}).get("grids") or {}
    grids_b = (b.patch or {}).get("grids") or {}
    if not grids_a or not grids_b:
        which = "the first" if not grids_a else "the second"
        return Delta(
            "patching",
            NOT_COMPARABLE,
            f"{which} file carries no patching trace, so there is nothing to "
            f"compare. A missing section is not a zero.",
        )
    shared = sorted(set(grids_a) & set(grids_b))
    if not shared:
        return Delta(
            "patching",
            NOT_COMPARABLE,
            "these traces have no component in common.",
        )

    worst_component, worst_gap = shared[0], 0.0
    flipped: list[str] = []
    for name in shared:
        grid_a, grid_b = grids_a[name], grids_b[name]
        gap = max_abs_diff(grid_a, grid_b)
        if gap is None:
            return Delta(
                "patching",
                NOT_COMPARABLE,
                f"the {name!r} grid is a different shape in the two files.",
            )
        if gap > worst_gap:
            worst_component, worst_gap = name, gap
        # A SITE CHANGING SIGN is the finding, not the magnitude: a cell that
        # recovered the clean answer and now pushes away from it is a
        # different causal story, however small the numbers are.
        for r, (row_a, row_b) in enumerate(zip(grid_a, grid_b, strict=False)):
            for c, (x, y) in enumerate(zip(row_a, row_b, strict=False)):
                if (x > 0) != (y > 0) and abs(x) > 0 and abs(y) > 0:
                    flipped.append(f"{name}[{r},{c}]")

    measured = {
        "components": shared,
        "worst_component": worst_component,
        "max_abs_diff": worst_gap,
        "sites_changed_sign": len(flipped),
        "flipped": flipped[:NAMED_HEADS],
    }
    if not flipped and worst_gap == 0.0:
        return Delta(
            "patching",
            SAME,
            f"all {len(shared)} grids are identical.",
            magnitude=0.0,
            unit="nats",
            measured=measured,
        )
    detail = (
        f"{len(flipped)} patching sites changed sign"
        if flipped
        else f"the {worst_component!r} grid moved by {worst_gap:.2e}"
    )
    return Delta(
        "patching",
        CHANGED,
        f"{detail} — a site that recovered the clean answer and now pushes "
        f"away from it is a different causal story."
        if flipped
        else f"{detail}.",
        magnitude=worst_gap,
        unit="nats",
        measured=measured,
    )


def _diff_lens(a, b) -> Delta:
    if not a.lens or not b.lens:
        which = "the first" if not a.lens else "the second"
        return Delta(
            "logit lens",
            NOT_COMPARABLE,
            f"{which} file carries no logit-lens trajectory.",
        )
    rows = min(len(a.lens), len(b.lens))

    def leader(row: dict) -> str | None:
        tokens = row.get("tokens") or []
        return tokens[0] if tokens else None

    for i in range(rows):
        was, now = leader(a.lens[i]), leader(b.lens[i])
        if was != now:
            # The LAYER is the finding. "The answer used to be decided by
            # layer 8 and now is not decided until 11" is what a reader acts
            # on; which token it was at that layer is the supporting detail.
            return Delta(
                "logit lens",
                CHANGED,
                f"the trajectory first diverges at layer {a.lens[i].get('layer', i)}: "
                f"{was!r} led there and now {now!r} does.",
                magnitude=None,
                unit="token",
                measured={
                    "first_divergence_layer": a.lens[i].get("layer", i),
                    "was": was,
                    "now": now,
                    "layers_compared": rows,
                },
                # Which token leads is a name, not a size. Every CHANGED this
                # section can produce reports `magnitude=None`, so there is no
                # threshold in tokens or in layers for `--fail-over` to hold.
                gated=False,
            )

    settled_a = (a.lens_info or {}).get("settled_at")
    settled_b = (b.lens_info or {}).get("settled_at")
    if settled_a != settled_b:
        return Delta(
            "logit lens",
            CHANGED,
            f"the same token leads at every layer, but the answer settles at "
            f"layer {settled_b} rather than {settled_a}.",
            magnitude=None,
            unit="layer",
            measured={"settled_at_a": settled_a, "settled_at_b": settled_b},
            gated=False,
        )
    return Delta(
        "logit lens",
        SAME,
        f"the same token leads at all {rows} layers.",
        magnitude=0.0,
        unit="token",
        measured={"layers_compared": rows},
        gated=False,
    )


def _named(items: list[str], cap: int = NAMED_HEADS) -> str:
    """The first `cap` names, then a count of the ones not printed.

    A terminal line naming forty edges is a line nobody reads, and one that
    silently stops at six is one that hides thirty-four. The count is the
    difference between the two.
    """
    rest = len(items) - cap
    return ", ".join(items[:cap]) + (f" and {rest} more" if rest > 0 else "")


def _plural(count: int, noun: str) -> str:
    """`1 edge`, `2 edges` — a count and its noun, agreeing.

    `_diff_patch` writes "3 patching sites changed sign" and every other differ
    either pluralises or phrases around the question; the two lines that wrote
    "1 edge(s)" were the only places in this module that made the reader do it.
    """
    return f"{count} {noun}" + ("" if count == 1 else "s")


def _sentence(text: str) -> str:
    """The same clause, capitalised for use after another sentence.

    Every detail in this module opens lowercase -- "both files continue
    with...", "the first file carries no..." -- so a clause that is sometimes
    the first sentence of one and sometimes the third is built once in that
    voice and cased where it is used.
    """
    return text[:1].upper() + text[1:]


def _set_line(noun: str, entered: list[str], left: list[str]) -> str:
    """Which of something the second file gained and lost, as a sentence.

    A sibling of `_top_k_line` rather than a reuse of it. That one builds its
    names out of `(layer, head)` pairs, and everything here arrives already
    named by the file that carried it -- so the two share a shape and not a
    body, and the shape is the part a reader notices.
    """
    parts = []
    if entered:
        parts.append(f"{_named(entered)} entered")
    if left:
        parts.append(f"{_named(left)} left")
    return f"the {noun} changed: " + " and ".join(parts) + "."


def _top_edges_line(
    top: int, entered: list[str], left: list[str], noun: str = "strongest"
) -> str:
    """`_top_k_line`'s sentence for edges that are already named.

    `noun` because the two graph sections order their edges by two different
    quantities and only one of them is a strength. An attribution weight is
    ranked by magnitude, so "strongest" is what it is; a patching recovery is a
    SIGNED fraction of the gap, ranked the way `path_trace` and the walk's own
    seeding rank it, so the edge at the top is the one that recovers the most
    and an edge at -0.9 is a sender pushing the answer away rather than a
    strong one. Calling that ordering "the strongest" would be this module
    naming a verdict the file never made.
    """
    return (
        f"the {noun} {top} changed: "
        + _named(entered)
        + " entered"
        + (" and " + _named(left) + " left" if left else "")
        + "."
    )


def _keyed(edges: list[dict], key) -> tuple[dict, object]:
    """Edges by key, plus the first key that turned up twice.

    Both graph sections are joined edge-to-edge on a key their readers do not
    require to be unique. A pair carried twice makes the list a multiset, and
    joining two multisets on it compares an arbitrary member of one against an
    arbitrary member of the other -- reported, rather than collapsed to
    whichever entry happened to be last.
    """
    out: dict = {}
    duplicate = None
    for edge in edges:
        k = key(edge)
        if k in out and duplicate is None:
            duplicate = k
        out[k] = edge
    return out, duplicate


def _changed_sign(was: float, now: float) -> bool:
    """The same rule `_diff_patch` applies cell by cell, in one place.

    A quantity that pushed toward the clean answer and now pushes away from it
    is a different causal story however small the numbers are. Zero on either
    side is excluded: it has no direction to have changed.
    """
    return (was > 0) != (now > 0) and abs(was) > 0 and abs(now) > 0


def _diff_patch_graph(a, b) -> Delta:
    """Does the same circuit still carry the answer, edge by edge?

    `patch_graph` is a graph THIS tool measured: `patch.path_trace` walked back
    from the sites the node grid flagged and every drawn edge was run against
    eight same-norm control draws. So unlike `graph` below it comes with its
    own floor -- the prune threshold, which the walk read off the dtype's
    recovery resolution rather than choosing -- and the two files carry it.

    THREE FINDINGS, IN DESCENDING STRENGTH OF CLAIM, because they fail
    differently and a reader acts on them differently:

      1. A VERDICT THAT FLIPPED. `clears_control` is the section's whole
         guarantee: an edge is drawn because it beat its controls. One that
         used to and no longer does is the loudest thing this section can say,
         and it is a boolean -- there is no magnitude in recovery fractions at
         which it is within tolerance, which is why it reports `None` and
         fails at any `--fail-over`, exactly as a changed generation does.
         BOTH DIRECTIONS ARE THAT FINDING and only the verb differs: an edge
         that failed its controls is kept and marked `clears_control: false`
         rather than dropped, so one that now clears them is an ordinary
         outcome and saying it "no longer clears" them would be this module
         reporting the opposite of what it read.
      2. A RECOVERY THAT CHANGED SIGN, the same finding `_diff_patch` leads
         with on the node grid and in the same units, so an edge here and a
         cell there can still be read together.
      3. AN EDGE THAT ENTERED OR LEFT. Weaker than it looks, and the sentence
         says so: the file that dropped an edge does not record what that edge
         scored, so a recovery that fell below the prune threshold and a
         circuit that genuinely rerouted are indistinguishable from here. An
         absent edge is not a zero, and it is not a regression either until
         something says which of the two it was.

    Only then the magnitude, judged against the coarser of the two thresholds.
    """
    graph_a, graph_b = a.patch_graph or {}, b.patch_graph or {}
    if not graph_a or not graph_b:
        which = "the first" if not graph_a else "the second"
        return Delta(
            "patching graph",
            NOT_COMPARABLE,
            f"{which} file carries no patching graph. A missing section is "
            f"not a zero, so this comparison is unavailable rather than clean.",
        )

    # THE PAIR THE RECOVERIES WERE MEASURED AGAINST. Recovery is a fraction of
    # the gap between the clean run and the corrupted one, so two graphs over
    # two different pairs are two measurements rather than one that moved --
    # the same category error `_diff_ground` refuses for two questions.
    #
    # BOTH SIDES OR NEITHER. `session._patch_graph` defaults an absent `clean`
    # or `corrupt` to "", so a file that never recorded the pair arrives here
    # indistinguishable from one that recorded an empty string -- and comparing
    # that against a file that did record it produced a refusal asserting the
    # two walks ran over different prompts, quoting '' as the second one. That
    # is an unknown rendered as a value, in a sentence claiming to have
    # measured a difference. One side recording nothing is not a disagreement;
    # both sides travel in `measured` and this only refuses what both files
    # actually said.
    for key, label in (("clean", "clean"), ("corrupt", "corrupted")):
        left, right = graph_a.get(key) or "", graph_b.get(key) or ""
        if left and right and left != right:
            return Delta(
                "patching graph",
                NOT_COMPARABLE,
                f"these two graphs were walked over different {label} prompts "
                f"({left!r} against {right!r}). Recovery is a fraction of the "
                f"gap between one run and the other, so a graph over a "
                f"different pair is a different measurement and not a changed "
                f"one.",
            )

    # HOW FAR THE WALK WENT. Edge count is quadratic in sites, so every such
    # graph is a subset by construction and the rule that chose the subset has
    # to match before the subsets can be compared. `depth` is the part of that
    # rule the reader keeps: `max_receivers` is dropped by `session._patch_graph`
    # and the `seeding` sentence embeds `n_scored` and `n_pruned`, which move
    # between two honest runs of the same walk -- refusing on the sentence
    # would refuse nearly every real diff, so both sentences travel in
    # `measured` instead and only the reach is a refusal.
    depth_a, depth_b = graph_a.get("depth", 0), graph_b.get("depth", 0)
    if depth_a != depth_b:
        return Delta(
            "patching graph",
            NOT_COMPARABLE,
            f"these two walks went back {depth_a} level(s) and {depth_b}. Edge "
            f"count is quadratic in sites, so each graph is a subset by "
            f"construction — and two walks of different reach hold different "
            f"edges for a reason that is not the model.",
        )

    def _pair(edge: dict) -> tuple:
        return (edge["source"], edge["target"])

    edges_a, dup_a = _keyed(graph_a.get("edges") or [], _pair)
    edges_b, dup_b = _keyed(graph_b.get("edges") or [], _pair)
    if dup_a or dup_b:
        which = "the first" if dup_a else "the second"
        source, target = dup_a or dup_b
        return Delta(
            "patching graph",
            NOT_COMPARABLE,
            f"{which} file's patching graph carries {source} → {target} more "
            f"than once, so its edge list is a multiset and a per-edge "
            f"comparison would pick an arbitrary one of them.",
        )

    ids_a = {n["id"] for n in graph_a.get("nodes") or []}
    ids_b = {n["id"] for n in graph_b.get("nodes") or []}
    nodes_entered, nodes_left = sorted(ids_b - ids_a), sorted(ids_a - ids_b)

    # THE FLOOR THE FILES SUPPLY. `patch_graph.build` sets `prune_threshold`
    # from the trace's own recovery resolution -- one representable step of the
    # gap between the two runs' answers, which is why it is per model and per
    # pair and never a constant -- and records `prune_from` saying so. The
    # coarser of the two is the finest difference this comparison can claim,
    # for the reason the ranking takes the larger noise floor.
    thr_a = float(graph_a.get("prune_threshold") or 0.0)
    thr_b = float(graph_b.get("prune_threshold") or 0.0)
    floor = max(thr_a, thr_b)
    coarser = (graph_b if thr_b > thr_a else graph_a).get("prune_from") or ""
    floor_from = (
        f"the coarser of the two files' prune thresholds ({coarser})"
        if coarser
        else "the coarser of the two files' prune thresholds, neither of "
        "which says where it came from"
    )

    shared = sorted(edges_a.keys() & edges_b.keys())
    edges_entered = [f"{s} → {t}" for s, t in sorted(edges_b.keys() - edges_a.keys())]
    edges_left = [f"{s} → {t}" for s, t in sorted(edges_a.keys() - edges_b.keys())]

    # BY DIRECTION, BECAUSE ONLY ONE DIRECTION HAS A VERB. Both are a flipped
    # verdict and both are equally a change, but "no longer clears its
    # controls" and "now clears the controls it used to fail" are opposite
    # claims about the model -- and `patch_graph` keeps an edge that LOST its
    # control and marks it `clears_control: false` rather than dropping it, so
    # a file carrying `false` is an ordinary file and False -> True is an
    # ordinary outcome. Collected with a bare `!=` and printed with the losing
    # verb, this section published the opposite of what it measured, in the
    # strongest sentence it can say.
    lost_control: list[str] = []
    gained_control: list[str] = []
    lost_position: list[str] = []
    gained_position: list[str] = []
    changed_sign: list[str] = []
    moved: list[tuple[float, str, float, float]] = []
    for key in shared:
        edge_a, edge_b = edges_a[key], edges_b[key]
        name = f"{key[0]} → {key[1]}"
        if edge_a["clears_control"] != edge_b["clears_control"]:
            (lost_control if edge_a["clears_control"] else gained_control).append(name)
        # THREE-VALUED, AND ONLY THE TWO-VALUED TRANSITION IS A FINDING.
        # `clears_position` is a separate pass a file may legitimately not have
        # run, and the reader keeps that as None on purpose. None becoming True
        # is a pass that was skipped becoming one that was run -- a change in
        # what the walk did, not in what the model does -- and reading None as
        # False here would turn "not run" into "run, and failed".
        pos_a, pos_b = edge_a.get("clears_position"), edge_b.get("clears_position")
        if isinstance(pos_a, bool) and isinstance(pos_b, bool) and pos_a != pos_b:
            (lost_position if pos_a else gained_position).append(name)
        was, now = float(edge_a["recovery"]), float(edge_b["recovery"])
        if _changed_sign(was, now):
            changed_sign.append(name)
        if abs(now - was) > floor:
            moved.append((abs(now - was), name, was, now))
    moved.sort(reverse=True)

    # WHICH EDGES RECOVER THE MOST needs no floor, exactly as the ranking's top
    # five does not: it is a comparison of two orderings, and "this edge no
    # longer leads" is what a reader acts on. Ranked by SIGNED recovery, which
    # is how `path_trace` ranks its senders and how the walk seeds its next
    # level -- so this ordering is the file's own and not a second one invented
    # here. An edge at -0.9 is therefore at the BOTTOM of it, which is right:
    # it is a sender pushing the answer away, and the sign finding above is
    # what says so.
    top = min(5, len(shared))
    top_a = sorted(shared, key=lambda k: -edges_a[k]["recovery"])[:top]
    top_b = sorted(shared, key=lambda k: -edges_b[k]["recovery"])[:top]
    entered_top = [f"{s} → {t}" for s, t in top_b if (s, t) not in top_a]
    left_top = [f"{s} → {t}" for s, t in top_a if (s, t) not in top_b]

    measured = {
        "nodes_compared": len(ids_a & ids_b),
        "nodes_entered": nodes_entered,
        "nodes_left": nodes_left,
        "edges_compared": len(shared),
        "edges_entered": edges_entered,
        "edges_left": edges_left,
        "edges_moved": len(moved),
        # A COUNT AND THE NAMES BEHIND IT, PER DIRECTION. The first version
        # published a bare count for the position flips and named only the
        # control ones, so an edge whose position verdict moved appeared
        # nowhere -- not in the terminal, not in `--json` -- and the count was
        # a number with no receipt. Uncapped on purpose: the terminal sentence
        # elides with `_named` and says how many it elided, and this is where
        # the reader who wants all of them looks.
        "verdicts_flipped": len(lost_control) + len(gained_control),
        "control_verdicts_lost": lost_control,
        "control_verdicts_gained": gained_control,
        "position_verdicts_flipped": len(lost_position) + len(gained_position),
        "position_verdicts_lost": lost_position,
        "position_verdicts_gained": gained_position,
        "edges_changed_sign": len(changed_sign),
        "changed_sign": changed_sign[:NAMED_HEADS],
        "max_abs_recovery_diff": moved[0][0] if moved else 0.0,
        "worst_edge": moved[0][1] if moved else None,
        "floor": floor,
        "floor_from": floor_from,
        "top_k": top,
        "entered_top_k": entered_top,
        "left_top_k": left_top,
        "depth": depth_a,
        "seeding_a": graph_a.get("seeding"),
        "seeding_b": graph_b.get("seeding"),
        # THE PAIR EACH WALK RAN OVER, both sides, because the guard above
        # refuses only when both files recorded one. "" is what the reader
        # leaves behind for a file that recorded nothing AND for one that
        # recorded an empty string, and it cannot tell those apart -- so it
        # travels as `None`, which says "this file does not say" rather than
        # naming a prompt nobody ran.
        "clean_a": graph_a.get("clean") or None,
        "clean_b": graph_b.get("clean") or None,
        "corrupt_a": graph_a.get("corrupt") or None,
        "corrupt_b": graph_b.get("corrupt") or None,
        # HOW MANY SENDERS EACH WALK SCORED AND PRUNED. An edge list is a
        # subset of what was looked at, and these two are what say how big the
        # subset was -- without them "neither walk drew an edge" is a fact with
        # nothing behind it.
        "n_scored_a": graph_a.get("n_scored"),
        "n_scored_b": graph_b.get("n_scored"),
        "n_pruned_a": graph_a.get("n_pruned"),
        "n_pruned_b": graph_b.get("n_pruned"),
        # Carried through as `None` where the file recorded nothing. "Nothing
        # was too weak" is a result and `0` says it; a walk that never wrote
        # the number is a different fact, and folding the second into the first
        # is the defect class this whole module keeps documenting.
        "n_weak_a": graph_a.get("n_weak"),
        "n_weak_b": graph_b.get("n_weak"),
        "n_untested_a": graph_a.get("n_untested"),
        "n_untested_b": graph_b.get("n_untested"),
        # Where each walk stopped asking. An edge missing from a graph whose
        # frontier names its receiver was never looked for, which is a
        # different fact from one that was looked for and pruned.
        "frontier_a": graph_a.get("frontier") or [],
        "frontier_b": graph_b.get("frontier") or [],
    }

    if not edges_a and not edges_b:
        # THE FACT, AND NOT A CAUSE FOR IT. This sentence used to say no sender
        # in either file cleared both its prune threshold and its controls,
        # which is a reason nothing in the section records and which its own
        # numbers can contradict: `patch_graph` KEEPS an edge that failed its
        # controls and marks it `clears_control: false`, so a graph with no
        # edges is not a graph of edges that lost, and `n_scored` minus
        # `n_pruned` in `measured` can sit well above zero right beside the
        # claim that nothing survived the threshold. What both files do say is
        # that the list is empty.
        return Delta(
            "patching graph",
            SAME,
            "neither walk drew an edge. Two graphs agreeing that there was "
            "nothing to draw is a finding, and this is it — how many senders "
            "each walk scored and pruned on the way to that is in "
            "`n_scored_a`/`n_pruned_a` and their counterparts, which is where "
            "the reason lives if either file recorded one.",
            magnitude=0.0,
            floor=floor,
            unit="recovery fraction",
            measured=measured,
        )

    # THE SET CHANGE TRAVELS WITH WHATEVER OUTRANKS IT rather than instead of
    # it. The branches below are ordered by strength of claim, and the first
    # draft returned on the first one that matched -- so a run where an edge
    # lost its verdict AND another edge vanished printed only the verdict, and
    # the vanished edge lived in `measured` where a terminal reader never
    # looks. One headline, and the rest of the findings appended to it.
    #
    # THE EDGE CAVEAT BELONGS TO THE EDGE LINE. Attached to the joined block
    # instead, a run where only a NODE entered the walk was explained by a
    # sentence about an edge that neither entered nor left -- an answer to a
    # question the finding did not ask.
    lines = []
    if edges_entered or edges_left:
        lines.append(
            _set_line("edge set", edges_entered, edges_left)
            + f" An edge carried by one file and not the other is either a "
            f"circuit that rerouted or a recovery that crossed the "
            f"{floor:.6g} prune threshold — the file that dropped it does not "
            f"record what it scored, so an absent edge is not a zero and this "
            f"cannot tell the two apart."
        )
    if nodes_entered or nodes_left:
        lines.append(_set_line("nodes walked", nodes_entered, nodes_left))
    membership = (
        lines[0] + "".join(f" {_sentence(x)}" for x in lines[1:]) if lines else ""
    )

    # ONE CLAUSE PER VERDICT CLASS THAT FIRED, joined -- not the first
    # non-empty list. `flipped_control or flipped_position` returns whichever
    # is truthy first, so a run where one edge lost its control verdict and
    # another its position verdict named only the first, and the second was
    # a bare count in `measured` with no name anywhere. That is the same
    # defect the membership block above documents, one branch lower.
    verdicts = []
    if lost_control:
        verdicts.append(
            f"{_named(lost_control)} no longer clears the eight same-norm "
            f"control draws behind it"
        )
    if gained_control:
        verdicts.append(
            f"{_named(gained_control)} now clears the eight same-norm control "
            f"draws it previously failed"
        )
    if lost_position:
        verdicts.append(
            f"{_named(lost_position)} no longer clears the shifted-position control"
        )
    if gained_position:
        verdicts.append(
            f"{_named(gained_position)} now clears the shifted-position "
            f"control it previously failed"
        )
    if verdicts:
        headline = verdicts[0] + "".join(f". {_sentence(v)}" for v in verdicts[1:])
        return Delta(
            "patching graph",
            CHANGED,
            f"{headline}. An edge is drawn only because it was "
            f"tested, so a verdict that flipped is this section's strongest "
            f"finding — and a verdict is a boolean, with no size in recovery "
            f"fractions at which it is within tolerance. {_sentence(membership)}".strip(),
            # None, and not the recovery gap sitting in `measured`: nothing
            # here was judged against the floor, and publishing one next to a
            # magnitude of None would say it had been.
            magnitude=None,
            floor=None,
            unit="recovery fraction",
            measured=measured,
        )

    if changed_sign:
        return Delta(
            "patching graph",
            CHANGED,
            f"{_plural(len(changed_sign), 'edge')} changed sign "
            f"({_named(changed_sign)}) "
            f"— a sender that moved the answer toward the clean run and now "
            f"moves it away is a different causal story, however small the "
            f"numbers are. {_sentence(membership)}".strip(),
            magnitude=moved[0][0] if moved else 0.0,
            floor=floor,
            unit="recovery fraction",
            measured=measured,
        )

    if membership:
        return Delta(
            "patching graph",
            CHANGED,
            # The threshold named again rather than referred back to: with the
            # edge caveat now attached to the edge line, a report whose only
            # membership finding is a NODE never mentions a threshold, and
            # "that threshold" pointed at nothing.
            f"{membership} {len(moved)} of {len(shared)} shared edges also "
            f"moved past the {floor:.2e} prune threshold.",
            magnitude=moved[0][0] if moved else 0.0,
            floor=floor,
            unit="recovery fraction",
            measured=measured,
        )

    if entered_top or left_top:
        return Delta(
            "patching graph",
            CHANGED,
            f"the same {len(shared)} edges are drawn, but "
            f"{_top_edges_line(top, entered_top, left_top, 'highest-recovering')} "
            f"{len(moved)} of them moved past the {floor:.2e} prune threshold.",
            magnitude=moved[0][0] if moved else 0.0,
            floor=floor,
            unit="recovery fraction",
            measured=measured,
        )

    if moved:
        _, name, was, now = moved[0]
        step = now - was
        return Delta(
            "patching graph",
            CHANGED,
            f"the circuit is the same {len(shared)} edges in the same order, "
            f"but {name} recovered "
            f"{'+' if step >= 0 else '−'}{fmt.measured(abs(step), 5)} more of "
            f"the gap, from {fmt.measured(was, 5)} to {fmt.measured(now, 5)}. "
            f"{len(moved)} of {len(shared)} edges moved past the "
            f"{floor:.2e} prune threshold.",
            magnitude=moved[0][0],
            floor=floor,
            unit="recovery fraction",
            measured=measured,
        )

    if floor <= 0.0:
        # Neither file carried a threshold above zero -- and the reader
        # defaults a missing one to 0.0, so "recorded zero" and "never
        # recorded" arrive here identical and this cannot tell them apart.
        # With a floor of zero `abs(now - was) > floor` only holds for
        # identical numbers, so SAME is right; implying a tolerance was
        # applied to reach it is not.
        return Delta(
            "patching graph",
            SAME,
            f"all {len(shared)} edges carry identical recoveries and the same "
            f"verdicts. Both files carry a prune threshold of exactly zero — "
            f"recorded as zero or never recorded, which this cannot tell "
            f"apart — so that is bit-for-bit equality rather than agreement "
            f"within a tolerance.",
            magnitude=0.0,
            floor=floor,
            unit="recovery fraction",
            measured=measured,
        )
    return Delta(
        "patching graph",
        SAME,
        f"all {len(shared)} edges hold their verdicts and score within the "
        f"{floor:.2e} prune threshold these files recorded, and the same "
        f"edges recover the most.",
        magnitude=0.0,
        floor=floor,
        unit="recovery fraction",
        measured=measured,
    )


def _diff_graph(a, b) -> Delta:
    """Two attribution graphs SOMEBODY ELSE'S tool computed.

    The section exists because a `.mri` can forward a transcoder attribution
    graph, and `session._graph` refuses one that does not say who measured it.
    That disclaimer is the whole reason the key is separate from `patch_graph`,
    and it governs this comparison too: everything below is a difference
    between two runs of another tool, and where that tool's own resolution
    would be needed there is no number to reach for.

    WHAT THE READER ACTUALLY HANDS THIS FUNCTION is narrower than what the
    writer wrote. `_graph` builds its output from scratch and never copies the
    node list, so an edge here names its endpoints by INDEX into a list this
    file does not carry. There is no id to join two graphs on -- only the
    assumption that two graphs from the same tool, over the same model and the
    same prompt, with the same node count, numbered their nodes the same way.
    Every refusal below is one leg of that assumption, checked rather than
    assumed, and the comparison says out loud that the rest of it is an
    assumption.

    AND THERE IS NO FLOOR. `circuit.Graph.summary` records `density` and
    `max_abs_weight`; nothing anywhere says what the producing tool could
    resolve. So a weight that merely moved is reported as not comparable with
    the number printed, and only the floor-independent findings -- membership
    and sign -- are ever called a change. Which edges are strongest travels
    with that refusal rather than as a change, because ranking near-equal
    weights is exactly the comparison the missing resolution forbids.
    Inventing an epsilon here is the one thing this module has never done.
    """
    graph_a, graph_b = a.graph or {}, b.graph or {}
    if not graph_a or not graph_b:
        which = "the first" if not graph_a else "the second"
        return Delta(
            "attribution graph",
            NOT_COMPARABLE,
            f"{which} file carries no attribution graph. A missing section is "
            f"not a zero, so this comparison is unavailable rather than clean.",
        )

    prov_a = graph_a.get("provenance") or {}
    prov_b = graph_b.get("provenance") or {}
    for key, label in (("producer", "tool"), ("model", "model")):
        left, right = prov_a.get(key), prov_b.get(key)
        if left != right:
            return Delta(
                "attribution graph",
                NOT_COMPARABLE,
                f"these graphs name a different {label} ({left!r} against "
                f"{right!r}). ModelMRI computed neither of them, so a diff "
                f"across two {label}s measures the {label} rather than "
                f"anything either file recorded.",
            )

    # ONLY WHERE BOTH FILES CARRY IT. `prompt` is optional the whole way down:
    # `circuit.to_session` forwards `graph.prompt`, which may be None, and
    # `session._graph` writes the key only when the value it found was a
    # string. Read as `.get("prompt") or ""`, a file that never recorded one
    # became a file that recorded the empty prompt, and the refusal said in so
    # many words that these graphs were computed over different prompts,
    # quoting '' as the second one -- an unknown rendered as a value inside a
    # sentence claiming to have measured a difference. It is the rule this
    # same function applies to `edge_limit` and `truncated` twenty lines down.
    if (
        "prompt" in graph_a
        and "prompt" in graph_b
        and graph_a["prompt"] != graph_b["prompt"]
    ):
        return Delta(
            "attribution graph",
            NOT_COMPARABLE,
            f"these graphs were computed over different prompts "
            f"({graph_a['prompt']!r} against {graph_b['prompt']!r}), so their "
            f"nodes are different tokens and an edge in one names nothing in "
            f"the other.",
        )

    nodes_a, nodes_b = graph_a.get("n_nodes"), graph_b.get("n_nodes")
    if nodes_a != nodes_b:
        return Delta(
            "attribution graph",
            NOT_COMPARABLE,
            f"these graphs declare {nodes_a} nodes and {nodes_b}. An edge here "
            f"names its endpoints by index into a node list the section does "
            f"not carry, so with two different node counts the indices do not "
            f"name the same nodes and a per-edge comparison would be "
            f"arithmetic over two different graphs.",
        )

    # WHAT DECIDED MEMBERSHIP. `circuit.Graph.edges` returns only the strongest
    # `edge_limit`, and the summary records both the limit and whether it bit.
    # Two lists cut at different points, or one cut and one whole, differ
    # because of the cut -- so the set comparison below would report the export
    # setting. Compared only where BOTH files carry the field: an absent
    # `truncated` is not a `False`.
    sum_a = graph_a.get("summary") or {}
    sum_b = graph_b.get("summary") or {}
    limit_a, limit_b = sum_a.get("edge_limit"), sum_b.get("edge_limit")
    if _is_number(limit_a) and _is_number(limit_b) and limit_a != limit_b:
        return Delta(
            "attribution graph",
            NOT_COMPARABLE,
            f"these graphs were exported at different edge limits ({limit_a} "
            f"against {limit_b}), so which edges each list holds was decided "
            f"by the export and not by the graph.",
        )
    trunc_a, trunc_b = sum_a.get("truncated"), sum_b.get("truncated")
    if isinstance(trunc_a, bool) and isinstance(trunc_b, bool) and trunc_a != trunc_b:
        which = "the first" if trunc_a else "the second"
        return Delta(
            "attribution graph",
            NOT_COMPARABLE,
            f"{which} file's edge list was truncated at its export limit and "
            f"the other's was not, so one is the strongest slice of a graph "
            f"and the other is a whole one. Their edge sets differ because of "
            f"that rather than because of the model.",
        )

    def _pair(edge: dict) -> tuple:
        return (edge["source"], edge["target"])

    edges_a, dup_a = _keyed(graph_a.get("edges") or [], _pair)
    edges_b, dup_b = _keyed(graph_b.get("edges") or [], _pair)
    if dup_a or dup_b:
        which = "the first" if dup_a else "the second"
        source, target = dup_a or dup_b
        return Delta(
            "attribution graph",
            NOT_COMPARABLE,
            f"{which} file's graph carries the edge #{source} → #{target} more "
            f"than once, so its edge list is a multiset and a per-edge "
            f"comparison would pick an arbitrary one of them.",
        )

    shared = sorted(edges_a.keys() & edges_b.keys())
    entered = [f"#{s} → #{t}" for s, t in sorted(edges_b.keys() - edges_a.keys())]
    left = [f"#{s} → #{t}" for s, t in sorted(edges_a.keys() - edges_b.keys())]

    changed_sign: list[str] = []
    worst_edge, worst_gap = None, 0.0
    for key in shared:
        was = float(edges_a[key]["weight"])
        now = float(edges_b[key]["weight"])
        if _changed_sign(was, now):
            changed_sign.append(f"#{key[0]} → #{key[1]}")
        if abs(now - was) > worst_gap:
            worst_edge, worst_gap = f"#{key[0]} → #{key[1]}", abs(now - was)

    top = min(5, len(shared))
    top_a = sorted(shared, key=lambda k: -abs(edges_a[k]["weight"]))[:top]
    top_b = sorted(shared, key=lambda k: -abs(edges_b[k]["weight"]))[:top]
    entered_top = [f"#{s} → #{t}" for s, t in top_b if (s, t) not in top_a]
    left_top = [f"#{s} → #{t}" for s, t in top_a if (s, t) not in top_b]

    measured = {
        "n_nodes": nodes_a,
        # BOTH PROMPTS, because the guard above refuses only when both files
        # carry one. `session._graph` writes the key only for a string, so
        # `None` here is "this file did not record a prompt" and not "this file
        # was computed over the empty prompt" -- which is the distinction the
        # guard now keeps and the receipt has to keep with it.
        "prompt_a": graph_a.get("prompt"),
        "prompt_b": graph_b.get("prompt"),
        "edges_compared": len(shared),
        "edges_entered": entered,
        "edges_left": left,
        "edges_changed_sign": len(changed_sign),
        "changed_sign": changed_sign[:NAMED_HEADS],
        "top_k": top,
        "entered_top_k": entered_top,
        "left_top_k": left_top,
        "max_abs_weight_diff": worst_gap,
        "worst_edge": worst_edge,
        # There is no floor to publish, and `None` says that rather than a
        # zero saying the tool could resolve everything.
        "floor": None,
        "floor_from": "no attribution graph records what its producer could "
        "resolve, so a weight difference has nothing to be judged against",
        # BOTH SIDES, because the guard above compares only `producer` and
        # `model`. Two files may disclaim in two different sentences and still
        # be compared, so publishing one of them under an unqualified key
        # reports the first file's disclaimer as the pair's.
        "measured_by_a": prov_a.get("measured_by"),
        "measured_by_b": prov_b.get("measured_by"),
        # Unqualified because the refusal above proves the two are equal.
        "producer": prov_a.get("producer"),
        # THE PRODUCING TOOL'S OWN NUMBERS, carried and not judged: the summary
        # is that tool's block, its keys are open ended, and deriving a verdict
        # from one would report a producer's version bump as a change in the
        # model. Named field by field rather than copied wholesale, because
        # `report.to_dict()` is dumped with `allow_nan=False` and
        # `session._graph` checks a summary value for finiteness only at the
        # top level -- so a NESTED block could still carry a NaN, and
        # `modelmri diff --json` would end in a serialiser crash rather than in
        # a wrong number. These four are top level and are already checked.
        #
        # `_a`/`_b` on all of them, including the two the guards above touched:
        # those guards fire only when BOTH files carry the field, so what
        # survives them is exactly the one-sided case -- a pair where the first
        # file records no `edge_limit` and the second was cut at 2000 passes,
        # the differ reports the second's missing edges as "left", and a
        # receipt saying `edge_limit: None` hides the export setting that
        # decided membership. Which is the confusion the guards exist for.
        "edge_limit_a": limit_a,
        "edge_limit_b": limit_b,
        "truncated_a": trunc_a,
        "truncated_b": trunc_b,
        "nonzero_edges_a": sum_a.get("nonzero_edges"),
        "nonzero_edges_b": sum_b.get("nonzero_edges"),
        "density_a": sum_a.get("density"),
        "density_b": sum_b.get("density"),
        "max_abs_weight_a": sum_a.get("max_abs_weight"),
        "max_abs_weight_b": sum_b.get("max_abs_weight"),
    }

    joined = (
        "These two graphs are joined on node index — the same tool, model, "
        "prompt and node count, which is the strongest link this section "
        "carries and still an assumption rather than a check, because the "
        "node list itself does not travel."
    )

    if not edges_a and not edges_b:
        return Delta(
            "attribution graph",
            SAME,
            f"neither graph carries an edge, so there is nothing here that "
            f"could have moved. {joined}",
            magnitude=0.0,
            unit="attribution weight",
            measured=measured,
            gated=False,
        )

    # MEMBERSHIP AND SIGN, AND NOT ORDER. The first two need no floor: an edge
    # is in a list or it is not, and a weight that pushed toward the answer and
    # now pushes away is a different claim at any size. WHICH EDGES ARE
    # STRONGEST IS NOT LIKE THEM -- it is an ordering of the very magnitudes
    # this section says it cannot judge, so two weights a hair apart rank in
    # whatever order the last digits fell. Called a change here, and the
    # arithmetic decided a CI outcome: `exit_code` fails a CHANGED delta with
    # no magnitude at EVERY `--fail-over`, so seven edges differing by 1e-10
    # swapped ranks 5 and 6 and failed `--fail-over 1e9`, while a weight that
    # moved by 0.5 without reordering anything returned NOT_COMPARABLE and
    # exited 0. The reorder is reported below instead, in the branch that
    # already says the magnitudes have nothing to be judged against.
    if changed_sign or entered or left:
        lines = []
        if changed_sign:
            lines.append(
                f"{_plural(len(changed_sign), 'edge')} changed sign "
                f"({_named(changed_sign)})."
            )
        if entered or left:
            lines.append(_set_line("edge set", entered, left))
        # Every finding that fired, not the first one: both need no floor and
        # neither is a stronger claim than the other, so neither is a headline
        # the other hides behind.
        joined_lines = lines[0] + "".join(f" {_sentence(x)}" for x in lines[1:])
        return Delta(
            "attribution graph",
            CHANGED,
            f"{joined_lines} These are findings that need no floor — "
            f"membership and sign — because nothing in either file says what "
            f"the producing tool could resolve, so how far a weight moved "
            f"cannot be judged here at all. {joined}",
            # No magnitude, for the same reason a changed generation has none:
            # the finding is categorical, and the only number available would
            # be one no file gave a scale for.
            magnitude=None,
            floor=None,
            unit="attribution weight",
            measured=measured,
            gated=False,
        )

    if worst_gap > 0.0:
        if entered_top or left_top:
            head = (
                f"the same {len(shared)} edges are here with the same signs, "
                f"and {_top_edges_line(top, entered_top, left_top)} "
                f"{worst_edge} moved by {fmt.measured(worst_gap, 5)} in the "
                f"producing tool's own units."
            )
            tail = (
                "Which edges are strongest is an ordering of those same "
                "weights, so the reorder above is reported here rather than "
                "as a change: two of them a hair apart rank in whatever order "
                "the last digits fell. Re-export both graphs from a tool that "
                "records a resolution and this becomes answerable."
            )
        else:
            head = (
                f"the same {len(shared)} edges are here in the same order and "
                f"with the same signs, and {worst_edge} moved by "
                f"{fmt.measured(worst_gap, 5)} in the producing tool's own "
                f"units."
            )
            tail = (
                "Re-export both graphs from a tool that records one, or read "
                "this as the ordering holding."
            )
        return Delta(
            "attribution graph",
            NOT_COMPARABLE,
            f"{head} Nothing in either file says what that tool could resolve, "
            f"and an absent resolution is not a resolution of zero — with one, "
            f"every last digit counts as a change. {tail}",
            # The one branch of this function where a number was actually
            # computed, and the first draft was the only one that published no
            # receipt for it: `worst_edge`, `max_abs_weight_diff` and both
            # files' summary scalars were all in scope and dropped, so a
            # `--json` reader got the sentence and nothing to check it against.
            unit="attribution weight",
            measured=measured,
            gated=False,
        )
    return Delta(
        "attribution graph",
        SAME,
        f"all {len(shared)} edges carry identical weights, so these two "
        f"graphs are the same graph rather than two that agree within "
        f"something. {joined}",
        magnitude=0.0,
        unit="attribution weight",
        measured=measured,
        gated=False,
    )


def _named_list(names: list[str]) -> str:
    """`a`, `a and b`, `a, b and c` — an English list, for one sentence."""
    if len(names) < 2:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


# ------------------------------------------------------------------- driver


def diff(path_a: str | Path, path_b: str | Path) -> DiffReport:
    """Compare two `.mri` of the same prompt."""
    left, right = Path(path_a), Path(path_b)
    parsed = []
    for target in (left, right):
        try:
            data = target.read_bytes()
        except OSError as err:
            raise BadRequest(
                f"{target.name} could not be read ({err.strerror or type(err).__name__})"
            ) from None
        parsed.append(session_mod.parse(data))
    a, b = parsed

    report = DiffReport(
        a=left.name,
        b=right.name,
        model_a=(a.meta or {}).get("model"),
        model_b=(b.meta or {}).get("model"),
    )

    refuse, notes = _comparable(a, b)
    report.notes.extend(notes)
    if refuse:
        # Refused as a whole rather than per section: when the two files are
        # not about the same run, every per-section number would be a
        # comparison of different things dressed as a regression.
        raise BadRequest(" ".join(refuse))

    report.deltas.append(_diff_generation(a, b))
    report.deltas.append(_diff_ranking(a, b))
    report.deltas.append(_diff_attention(a, b))
    report.deltas.append(_diff_patch(a, b))
    # Straight after the node grid it was walked out of, and in the same units,
    # so a reader who has just been told which cells moved is told next which
    # edges between them did.
    report.deltas.append(_diff_patch_graph(a, b))
    report.deltas.append(_diff_lens(a, b))
    report.deltas.append(_diff_ground(a, b))
    # Last, because it is the one section here ModelMRI did not measure.
    report.deltas.append(_diff_graph(a, b))
    return report


def render(report: DiffReport, fail_over: float | None = None) -> str:
    mark = {SAME: "=", CHANGED: "≠", NOT_COMPARABLE: "–"}
    lines = [f"{report.a} → {report.b}"]
    if report.model_a and report.model_a != report.model_b:
        lines.append(f"  {report.model_a} → {report.model_b}")
    elif report.model_a:
        lines.append(f"  {report.model_a}")
    lines.append("")

    for delta in report.deltas:
        lines.append(f"  {mark.get(delta.status, '?')} {delta.name}: {delta.status}")
        lines.append(f"      {delta.detail}")

    totals = report.to_dict()["totals"]
    lines.append("")
    lines.append(
        f"  {totals[SAME]} unchanged · {totals[CHANGED]} changed · "
        f"{totals[NOT_COMPARABLE]} not comparable"
    )
    if fail_over is not None:
        # THE UNITS, READ OFF THE REPORT RATHER THAN REMEMBERED. This sentence
        # named three of them from a literal -- "nats for the ranking and
        # patching, attention weight for attention" -- and the moment a
        # section was added, `--fail-over` was being compared in units the
        # sentence explaining `--fail-over` did not mention. A hardcoded list
        # of what a report contains drifts from the report by construction;
        # the deltas already carry their own units, so they are the source.
        #
        # Only the sections the threshold can actually gate -- and that is a
        # property of the section, which is why it is read off `delta.gated`
        # and not off this run's magnitude. Read off the magnitude, the
        # sentence selected almost exactly the wrong set: a SAME generation
        # carries `magnitude=0.0` and was named, offering "0.01 in text units";
        # a CHANGED patching graph whose verdict flipped carries `None`, is the
        # delta that produced the exit 1, and its "recovery fraction" was left
        # out of the sentence explaining the threshold that failed it.
        by_unit: dict[str, list[str]] = {}
        for delta in report.deltas:
            if delta.gated and delta.unit:
                by_unit.setdefault(delta.unit, []).append(delta.name)
        named = ", ".join(
            f"{unit} for {_named_list(names)}" for unit, names in by_unit.items()
        )
        lines.append(
            f"  failing over {fail_over} in each metric's own units"
            + (f" ({named})" if named else "")
        )
    for note in report.notes:
        lines.append(f"  note: {note}")
    return "\n".join(lines)
