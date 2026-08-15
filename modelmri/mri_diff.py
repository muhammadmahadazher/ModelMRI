"""Did this change the model, or did it change the last digits?

`modelmri diff before.mri after.mri` compares two saved analyses of the same
prompt and says which heads moved in the ablation ranking, which patching sites
changed sign, and whether the model still says the same thing. With
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
comparison of two files cannot be finer than the coarser of the two. There is
no epsilon in this module that somebody chose.

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
    # units -- nats for the ranking and patching, attention weight for
    # attention. Units differ per metric and are named rather than blended
    # into one score that would mean nothing.
    magnitude: float | None = None
    floor: float | None = None
    unit: str = ""
    measured: dict = field(default_factory=dict)

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
        )
    return Delta(
        "generation",
        CHANGED,
        f"the model said {a.generation!r} and now says {b.generation!r}.",
        # No magnitude. There is no number at which "it says something else"
        # is within tolerance, and `exit_code` treats a magnitude of None as
        # unconditionally failing for exactly that reason.
        magnitude=None,
        unit="text",
        measured={"a": a.generation, "b": b.generation},
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
            f"`ablate.py` measures Spearman 0.34-0.47 between them on gpt2 "
            f"layer 0. Comparing across them would measure the baseline.",
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
    floor = max(
        float((a.ranking or {}).get("noise_floor_kl") or 0.0),
        float((b.ranking or {}).get("noise_floor_kl") or 0.0),
    )

    moved = []
    for head in shared:
        gap = scores_b[head] - scores_a[head]
        if abs(gap) > floor:
            moved.append((abs(gap), head, scores_a[head], scores_b[head]))
    moved.sort(reverse=True)

    top = min(5, len(shared))
    top_a = sorted(shared, key=lambda h: -scores_a[h])[:top]
    top_b = sorted(shared, key=lambda h: -scores_b[h])[:top]
    entered = [h for h in top_b if h not in top_a]
    left = [h for h in top_a if h not in top_b]

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
        head_line = (
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
    else:
        _, head, was, now = moved[0]
        head_line = (
            f"the top {top} are unchanged, but L{head[0]}H{head[1]} moved "
            f"{was:.5f} → {now:.5f}."
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
    # one either file could tell from arithmetic.
    floor = max(
        float((a.ground or {}).get("noise_floor") or 0.0),
        float((b.ground or {}).get("noise_floor") or 0.0),
    )

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
            f"passage #{worst_index} moved by {worst_gap:.4f} nats against a "
            f"floor of {floor:.6f}. #{top_a} still carries the answer, so the "
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
        )
    return Delta(
        "logit lens",
        SAME,
        f"the same token leads at all {rows} layers.",
        magnitude=0.0,
        unit="token",
        measured={"layers_compared": rows},
    )


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
                f"{target.name} could not be read ({err.strerror or err})"
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
    report.deltas.append(_diff_lens(a, b))
    report.deltas.append(_diff_ground(a, b))
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
        lines.append(
            f"  failing over {fail_over} in each metric's own units "
            f"(nats for the ranking and patching, attention weight for "
            f"attention)"
        )
    for note in report.notes:
        lines.append(f"  note: {note}")
    return "\n".join(lines)
