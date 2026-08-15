"""Re-run somebody else's findings on your machine.

`modelmri verify run.mri` takes a recording, re-runs every measurement in it
that this machine can re-run, and reports per number whether it came back the
same. It is the other half of #17: a receipt says what produced a number, and
this is the thing that acts on one.

WHY THIS CLASSIFIES INSTEAD OF PASSING OR FAILING

Bit-exact reproduction across two machines is not achievable and pretending
otherwise would make the output worthless. Kernel selection, cuDNN version,
TF32, and the order a reduction happens in all move the last digits, and none
of that is the model changing. `ablate.py` records measuring 4.863085746765137
against 4.863086102936881 for the identical computation.

So there are three verdicts and no pass/fail:

  reproduced      the difference is inside a tolerance that was MEASURED on
                  this machine, not asserted from a constant
  differs         the difference is outside it, and both numbers are printed
  not verifiable  something about this machine or this file makes the
                  comparison meaningless, and the sentence says which

EVERY TOLERANCE IS MEASURED BEFORE IT IS CLAIMED

There is not a single hardcoded epsilon in this module. Each check establishes
its own floor by running the same computation twice on this machine and taking
the spread, which is the technique `ablate.rank_heads` already uses for its
noise floor (the same forward pass twice with nothing ablated). For attention
there is a second floor the file supplies itself: `session._quantise` stores
each matrix as uint8 against its own maximum, so that block's `scale` IS the
smallest difference the file is capable of representing, and no comparison
against it can be finer than that.

A clean report does NOT mean the finding is right. It means the numbers came
back the same on this machine. That distinction is printed in the output
rather than left for the reader to remember.
"""

from __future__ import annotations

import base64
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import receipts as receipts_mod
from . import session as session_mod
from .errors import BadRequest, Refusal

log = logging.getLogger("modelmri")

REPRODUCED = "reproduced"
DIFFERS = "differs"
NOT_VERIFIABLE = "not verifiable"

# A full export is every layer and head -- 144 blocks on gpt2, 900+ on a
# larger model -- and each is compared cell by cell in Python. The bound keeps
# `verify` from taking longer than the analysis it is checking. Whatever it
# drops is NAMED in the check's own sentence: a report that quietly checked
# 512 of 900 and said "reproduced" would read as having checked all of them.
MAX_ATTENTION_BLOCKS = 512


@dataclass
class Check:
    """One measurement, re-run or explicitly not re-run."""

    name: str
    verdict: str
    detail: str
    # The arithmetic behind the verdict, so a reader can disagree with the
    # threshold rather than having to trust it. Empty for a check that never
    # ran, which is itself the honest record of a check that never ran.
    measured: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Report:
    file: str
    model: str | None
    checks: list[Check] = field(default_factory=list)
    # What was true of the machine that wrote the file versus this one. Read
    # before any number is compared, because a mismatch here decides whether
    # comparing numbers means anything at all.
    setup: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def reproduced(self) -> int:
        return sum(1 for c in self.checks if c.verdict == REPRODUCED)

    @property
    def differed(self) -> int:
        return sum(1 for c in self.checks if c.verdict == DIFFERS)

    @property
    def unverifiable(self) -> int:
        return sum(1 for c in self.checks if c.verdict == NOT_VERIFIABLE)

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "model": self.model,
            "setup": self.setup,
            "checks": [c.to_dict() for c in self.checks],
            "notes": self.notes,
            "totals": {
                REPRODUCED: self.reproduced,
                DIFFERS: self.differed,
                NOT_VERIFIABLE: self.unverifiable,
            },
        }

    def exit_code(self) -> int:
        """Non-zero only for a real disagreement.

        A file this machine CANNOT check is not a failure of the file, and
        exiting non-zero for it would make `verify` unusable in CI the moment
        somebody runs it on a machine with a different accelerator. Nothing
        verified and nothing differing is exit 0 with a report that says so.
        """
        return 1 if self.differed else 0


# ------------------------------------------------------------------ helpers


def _receipt_for(parsed, op: str) -> dict:
    for receipt in parsed.receipts or []:
        if receipt.get("op") == op:
            return receipt
    return {}


def dequantise(block: dict) -> list[list[float]]:
    """A stored attention matrix back to floats, at the file's own precision."""
    raw = base64.b64decode(block["q"])
    scale = float(block["scale"])
    side = int(math.isqrt(len(raw)))
    return [[raw[r * side + c] * scale for c in range(side)] for r in range(side)]


def max_abs_diff(a, b) -> float | None:
    """Largest cell-wise gap between two equally shaped grids, or None."""
    if len(a) != len(b):
        return None
    worst = 0.0
    # strict=True: two grids of different lengths are two different
    # measurements, and zip's default would silently compare the overlap and
    # report a clean match on the part that happened to line up.
    for row_a, row_b in zip(a, b, strict=True):
        if len(row_a) != len(row_b):
            return None
        for x, y in zip(row_a, row_b, strict=True):
            if not (math.isfinite(x) and math.isfinite(y)):
                return None
            worst = max(worst, abs(x - y))
    return worst


def _setup_gap(parsed, runtime) -> tuple[dict, str]:
    """How this machine differs from the one that wrote the file.

    The second return value is a sentence naming a difference that makes
    numeric comparison meaningless, or "" when there is none. Dtype is the
    example the roadmap names and `patch.py` documents: on the reference pair
    bf16 moves the gap from 4.000 to 4.467 and changes the reference token
    itself, so a difference measured across dtypes is a fact about float
    formats and not about the model.
    """
    meta = parsed.meta or {}
    generate = _receipt_for(parsed, "generate")
    # Prefer the receipt: it was stamped at the moment of the run. `meta` is
    # written at export and is the older, coarser record of the same facts.
    file_dtype = generate.get("dtype") or meta.get("dtype")
    file_device = generate.get("device") or meta.get("device")
    file_revision = generate.get("revision")

    here_dtype = None
    if runtime.model is not None:
        try:
            here_dtype = str(next(runtime.model.parameters()).dtype).removeprefix(
                "torch."
            )
        except (StopIteration, AttributeError):
            here_dtype = None
    here_revision, here_revision_note = receipts_mod.revision_of(runtime.hf_id)

    setup = {
        "file_dtype": file_dtype,
        "here_dtype": here_dtype,
        "file_device": file_device,
        "here_device": runtime.device,
        "file_revision": file_revision,
        "here_revision": here_revision,
        "here_revision_note": here_revision_note,
    }

    if file_dtype and here_dtype and file_dtype != here_dtype:
        return setup, (
            f"could not verify here — this machine runs {here_dtype}, the file "
            f"says {file_dtype}. A difference measured across two float "
            f"formats is a fact about the formats, not about the model."
        )
    if file_revision and here_revision and file_revision != here_revision:
        return setup, (
            f"could not verify here — the file was measured against commit "
            f"{file_revision[:12]} and this machine has {here_revision[:12]}. "
            f"These are different weights, so a difference between them is "
            f"expected and says nothing about reproducibility."
        )
    return setup, ""


# ------------------------------------------------------------------- checks


def _check_generation(parsed, runtime, blocked: str) -> Check:
    """Did the model say the same thing again?"""
    generate = _receipt_for(parsed, "generate")
    request = generate.get("request") or {}
    greedy = request.get("greedy")

    if greedy is False:
        temperature = request.get("temperature")
        return Check(
            "generation",
            NOT_VERIFIABLE,
            f"this generation was sampled (temperature {temperature}), so a "
            f"different continuation is the sampler and not the model. Only a "
            f"greedy run can be compared.",
        )
    if greedy is None:
        return Check(
            "generation",
            NOT_VERIFIABLE,
            "this file does not record whether the generation was sampled, so "
            "a difference here cannot be told apart from the sampler. Files "
            "written by newer versions carry it.",
        )
    if blocked:
        return Check("generation", NOT_VERIFIABLE, blocked)

    want = parsed.generation
    asked = int(request.get("n_generated_tokens") or 0)
    if not asked:
        return Check(
            "generation",
            NOT_VERIFIABLE,
            "this file does not record how many tokens were generated, so the "
            "re-run has no length to match.",
        )

    got = "".join(runtime.generate_stream(parsed.prompt, asked, temperature=0.0))
    if got == want:
        return Check(
            "generation",
            REPRODUCED,
            f"greedy decoding produced the same {asked} tokens.",
            {"tokens": asked},
        )
    return Check(
        "generation",
        DIFFERS,
        f"the file says {want!r}, this machine says {got!r}.",
        {"file": want, "here": got, "tokens": asked},
    )


def _check_attention(parsed, runtime, blocked: str) -> Check:
    """Is the same head looking at the same tokens?

    TWO floors, and the larger wins. The file's own quantisation step is the
    finest difference it can represent -- `session._quantise` stores uint8
    against each matrix's maximum -- and this machine's own run-to-run spread
    is the finest difference this machine can distinguish. Claiming to
    reproduce a number more precisely than either would be claiming precision
    neither side has.
    """
    if not parsed.attention:
        return Check(
            "attention", NOT_VERIFIABLE, "this file carries no attention maps."
        )
    if blocked:
        return Check("attention", NOT_VERIFIABLE, blocked)

    keys = sorted(parsed.attention)
    probe_layer, probe_head = (int(x) for x in keys[0].split(":"))

    # THE SPREAD IS MEASURED BY FORCING TWO REAL CAPTURES. `runtime.attention`
    # memoises into `_attn_variants`, so calling it twice returns the same
    # tensor and any "spread" taken that way is a measurement of the cache --
    # it would report 0.0 forever, and a fabricated 0.0 floor in the module
    # whose entire subject is not fabricating tolerances is the worst possible
    # place for one. Clearing between the calls makes the model run again.
    runtime._attn_variants.clear()
    first = runtime.attention(probe_layer, probe_head)["matrix"]
    runtime._attn_variants.clear()
    again = runtime.attention(probe_layer, probe_head)["matrix"]
    spread = max_abs_diff(first, again)
    if spread is None:
        return Check(
            "attention",
            NOT_VERIFIABLE,
            "two identical captures on this machine came back differently "
            "shaped or non-finite, so no tolerance can be established.",
        )

    checked = 0
    # None, not (0.0, 0.0). A sentinel of "gap 0 against tolerance 0" has a
    # margin of exactly zero, which beats every block that actually passed --
    # so the report printed 0.00e+00 off 0.00e+00 for a block measured at
    # 1.97e-03 off 3.92e-03, publishing the sentinel as if it were the
    # measurement. In this module above all others, a displayed number has to
    # be one that was taken.
    worst: tuple[float, str, float] | None = None
    for key in keys[:MAX_ATTENTION_BLOCKS]:
        layer, head = (int(x) for x in key.split(":"))
        block = parsed.attention[key]
        # One block dequantised at a time. The stored cube can be every layer
        # and head, and holding all of them as Python floats alongside the
        # model is how this would become the memory-heaviest command here.
        stored = dequantise(block)
        here = runtime.attention(layer, head)["matrix"]
        if len(here) != len(stored):
            return Check(
                "attention",
                NOT_VERIFIABLE,
                f"the file's map is {len(stored)}x{len(stored)} and this run "
                f"is {len(here)}x{len(here)} — a different number of tokens is "
                f"a different measurement.",
            )
        gap = max_abs_diff(here, stored)
        if gap is None:
            return Check(
                "attention",
                NOT_VERIFIABLE,
                f"layer {layer} head {head} came back non-finite on the "
                f"re-run, so the comparison would be arithmetic rather than a "
                f"measurement.",
            )
        # Per block, because the quantisation step is per block: each matrix
        # was stored against ITS OWN maximum, so a sparse head has a finer
        # step than a diffuse one and a single tolerance would be wrong for
        # both.
        tolerance = max(float(block["scale"]), spread)
        checked += 1
        # Ranked by MARGIN rather than by raw gap: the block closest to
        # failing is the one worth naming, and because each block carries its
        # own quantisation step a larger gap under a larger tolerance is the
        # healthier of the two.
        if worst is None or (gap - tolerance) > (worst[0] - worst[2]):
            worst = (gap, key, tolerance)

    if worst is None:
        return Check(
            "attention",
            NOT_VERIFIABLE,
            "this file's attention section carries no readable blocks.",
        )
    worst_gap, worst_key, worst_tolerance = worst
    dropped = len(keys) - checked
    measured = {
        "blocks_in_file": len(keys),
        "blocks_checked": checked,
        "worst_block": worst_key,
        "max_abs_diff": worst_gap,
        "tolerance": worst_tolerance,
        "machine_spread": spread,
        "tolerance_from": (
            "the file's own uint8 quantisation step for that block"
            if worst_tolerance > spread
            else "this machine's spread over two real captures"
        ),
    }
    # Never a silent cap: a report that checked 512 of 900 blocks and says
    # "reproduced" reads as having checked all of them.
    capped = (
        f" (the other {dropped} were not checked — this reads at most "
        f"{MAX_ATTENTION_BLOCKS})"
        if dropped > 0
        else ""
    )
    exact = (
        " This machine reproduces itself exactly, so the tolerance is entirely "
        "the file's own storage precision."
        if spread == 0
        else ""
    )
    if worst_gap <= worst_tolerance:
        return Check(
            "attention",
            REPRODUCED,
            f"all {checked} stored head maps match{capped}. The worst, "
            f"{worst_key}, is off by {worst_gap:.2e} against a "
            f"{worst_tolerance:.2e} tolerance.{exact}",
            measured,
        )
    return Check(
        "attention",
        DIFFERS,
        f"{checked} head maps checked{capped}; {worst_key} is off by "
        f"{worst_gap:.2e}, outside its {worst_tolerance:.2e} tolerance.{exact}",
        measured,
    )


def _check_patch(parsed, runtime, blocked: str) -> Check:
    """Is the answer still decided in the same place?"""
    stored = parsed.patch or {}
    grids = stored.get("grids") or {}
    clean, corrupt = stored.get("clean"), stored.get("corrupt")
    if not grids:
        return Check("patching", NOT_VERIFIABLE, "this file carries no patch trace.")
    if not (isinstance(clean, str) and isinstance(corrupt, str)):
        return Check(
            "patching",
            NOT_VERIFIABLE,
            "this file's patch trace does not carry the prompt pair it was "
            "measured on, so it cannot be re-run.",
        )
    if blocked:
        return Check("patching", NOT_VERIFIABLE, blocked)

    first = runtime.patch_trace(clean, corrupt)
    second = runtime.patch_trace(clean, corrupt)

    worst_gap = 0.0
    worst_component = ""
    floor = 0.0
    for component, grid in grids.items():
        here = (first.get("grids") or {}).get(component)
        again = (second.get("grids") or {}).get(component)
        if here is None or again is None:
            return Check(
                "patching",
                NOT_VERIFIABLE,
                f"this machine produced no `{component}` grid, which the file "
                f"has — the architectures are being read differently.",
            )
        if len(here) != len(grid):
            return Check(
                "patching",
                NOT_VERIFIABLE,
                f"the `{component}` grid is {len(grid)} rows in the file and "
                f"{len(here)} here, so the two describe different runs.",
            )
        gap = max_abs_diff(here, grid)
        spread = max_abs_diff(here, again)
        if gap is None or spread is None:
            return Check(
                "patching",
                NOT_VERIFIABLE,
                f"the `{component}` grid came back non-finite or misshapen on "
                f"the re-run.",
            )
        if gap > worst_gap:
            worst_gap, worst_component = gap, component
        floor = max(floor, spread)

    measured = {
        "max_abs_diff": worst_gap,
        "worst_component": worst_component or next(iter(grids)),
        "tolerance": floor,
        "tolerance_from": "this machine's spread over two identical patch runs",
        "components": sorted(grids),
    }
    if worst_gap <= floor:
        return Check(
            "patching",
            REPRODUCED,
            f"all {len(grids)} grids match to {worst_gap:.2e}, inside a "
            f"{floor:.2e} floor measured by running the same trace twice here.",
            measured,
        )
    return Check(
        "patching",
        DIFFERS,
        f"the `{worst_component}` grid is off by {worst_gap:.2e}, outside the "
        f"{floor:.2e} floor measured by running the same trace twice here.",
        measured,
    )


def _check_lens(parsed, runtime, blocked: str) -> Check | None:
    """Does the answer still arrive at the same layer?

    Compared as a TRAJECTORY of leading tokens rather than cell by cell. The
    per-layer probabilities move in the last digits for the same reasons every
    other number does, but "which token leads at layer 8" is discrete: it
    either reproduces or it does not, and there is no tolerance to invent for
    it. `settled_at` is checked separately because the whole trajectory can
    lead with the same token while the layer the model commits at moves.
    """
    stored = parsed.lens or []
    if not stored:
        if not _receipt_for(parsed, "logit_lens"):
            return None
        return Check(
            "logit lens",
            NOT_VERIFIABLE,
            "this file records that a lens was read but does not carry it. "
            "Files written before the lens travelled have nothing here.",
        )
    if blocked:
        return Check("logit lens", NOT_VERIFIABLE, blocked)

    try:
        fresh = runtime.logit_lens(len(stored[0].get("tokens") or []) or 5)
    except (BadRequest, Refusal) as err:
        return Check(
            "logit lens",
            NOT_VERIFIABLE,
            f"the lens could not be read here — {err}",  # leak-ok: authored
        )

    rows = fresh.get("layers") or []
    if len(rows) != len(stored):
        return Check(
            "logit lens",
            NOT_VERIFIABLE,
            f"the file carries {len(stored)} layers and this model has "
            f"{len(rows)} — a different depth is a different model.",
        )

    def leader(row: dict):
        tokens = row.get("tokens") or []
        return tokens[0] if tokens else None

    for i, (here, there) in enumerate(zip(rows, stored, strict=True)):
        if leader(here) != leader(there):
            return Check(
                "logit lens",
                DIFFERS,
                f"layer {there.get('layer', i)} led with {leader(there)!r} in "
                f"the file and leads with {leader(here)!r} here.",
                {"first_divergence_layer": there.get("layer", i)},
            )

    settled_there = (parsed.lens_info or {}).get("settled_at")
    settled_here = fresh.get("settled_at")
    if settled_there is not None and settled_there != settled_here:
        return Check(
            "logit lens",
            DIFFERS,
            f"every layer leads with the same token, but the answer settles at "
            f"layer {settled_here} here against {settled_there} in the file.",
            {"settled_at_file": settled_there, "settled_at_here": settled_here},
        )
    return Check(
        "logit lens",
        REPRODUCED,
        f"the same token leads at all {len(rows)} layers"
        + (f", settling at layer {settled_here}." if settled_here is not None else "."),
        {"layers_compared": len(rows), "settled_at": settled_here},
    )


def _check_model_diff(parsed, runtime, blocked: str) -> Check | None:
    """Is the recorded comparison consistent with its own numbers?

    NEVER RE-RUN, and for a blunter reason than grounding's. Grounding needs a
    document the file deliberately does not carry; this needs TWO OTHER
    MODELS, neither of which is the one this file describes and neither of
    which is necessarily on this machine at all. Loading them to check would
    be a different and much larger operation than verifying a `.mri`.

    What it can check is that the section agrees with itself: the consensus
    layer against what the per-prompt rows actually say, and the spread counts
    against the number of prompts. A file edited to move the headline without
    moving the rows fails here.
    """
    stored = parsed.model_diff or {}
    rows = stored.get("prompts") or []
    if not rows:
        if not _receipt_for(parsed, "model_diff"):
            return None
        return Check(
            "model diff",
            NOT_VERIFIABLE,
            "this file records that a model comparison ran but does not carry it.",
        )

    model_a = stored.get("model_a") or "?"
    model_b = stored.get("model_b") or "?"
    measured = {
        "prompts": len(rows),
        "model_a": model_a,
        "model_b": model_b,
        "consensus_layer": stored.get("consensus_layer"),
    }

    problems: list[str] = []

    # The headline against the rows it claims to summarise.
    firsts = [r.get("first_divergent_layer") for r in rows]
    diverged = [f for f in firsts if isinstance(f, int)]
    claimed = stored.get("consensus_layer")
    if diverged:
        commonest = max(set(diverged), key=diverged.count)
        if claimed != commonest:
            problems.append(
                f"the file names layer {claimed} as where the streams come "
                f"apart, and its own per-prompt rows name layer {commonest} "
                f"most often"
            )
        share = diverged.count(commonest) / len(rows)
        stated = stored.get("consensus_share")
        if isinstance(stated, (int, float)) and abs(stated - share) > 1e-3:
            problems.append(
                f"the file says that layer was first on {stated:.0%} of "
                f"prompts and its rows say {share:.0%}"
            )
    elif claimed is not None:
        problems.append(
            f"the file names layer {claimed} as where the streams come apart "
            f"and not one of its prompt rows reports a divergence at all"
        )

    # The spreads against the prompt count they claim to be over.
    for key in ("kl", "flips"):
        spread = stored.get(key)
        if isinstance(spread, dict) and spread.get("n") != len(rows):
            problems.append(
                f"the {key} spread says it is over {spread.get('n')} prompts "
                f"and the file carries {len(rows)}"
            )

    if problems:
        measured["problems"] = problems
        return Check(
            "model diff",
            DIFFERS,
            "this comparison does not agree with itself — "
            + "; ".join(problems[:3])
            + ". The file has been edited, or written by something that did "
            "not derive its headline from its own rows.",
            measured,
        )

    return Check(
        "model diff",
        NOT_VERIFIABLE,
        f"{model_a} against {model_b} over {len(rows)} prompts, and the "
        f"summary agrees with the per-prompt rows underneath it. NOT "
        f"RE-MEASURED: this compares two models, neither of which is the one "
        f"this file describes and neither of which need be on this machine. "
        f"Re-run it against the same pair and the same prompts to reproduce "
        f"it.",
        measured,
    )


def _check_ground(parsed, runtime, blocked: str) -> Check | None:
    """Does the same passage still carry the answer, by the same margin?

    THE ONE CHECK HERE THAT NEEDS NO GENERATION. Grounding runs its own
    document and its own question, so unlike attention, the lens and the head
    ranking it does not need this file's run re-established first — it needs
    the document, and the document is not in the file.

    That is the whole difficulty. A `.mri` carries the passage PREVIEWS, which
    are about 120 characters each and deliberately not the passages: a
    grounding document is usually the private half of the pair, and a format
    that quietly shipped somebody's source material to whoever they forwarded
    the file to would be a worse failure than not verifying. So this cannot
    re-take the measurement, and it does not pretend to.

    What it CAN check is the part that is about the model rather than about
    the text: whether the recorded verdicts are internally consistent with the
    floor recorded beside them, and whether the passage the file names as
    carrying the answer is still the one its own numbers point at. A file
    edited to move the verdict without moving the numbers fails here.
    """
    stored = parsed.ground or {}
    rows = stored.get("chunks") or []
    if not rows:
        if not _receipt_for(parsed, "ground"):
            return None
        # Written before the grounding section existed, or exported from a
        # state where the epoch had moved. Named rather than skipped: silence
        # would read as "it reproduced".
        return Check(
            "grounding",
            NOT_VERIFIABLE,
            "this file records that a grounding ran but does not carry it. "
            "Files written before the grounding section existed have nothing "
            "here to compare against.",
        )

    floor = float(stored.get("noise_floor") or 0.0)
    degenerate = bool(stored.get("floor_degenerate"))
    measured = {
        "passages": len(rows),
        "noise_floor": floor,
        "floor_degenerate": degenerate,
        "attention_available": bool(stored.get("attention_available")),
    }

    # The document is not in the file, so this is never REPRODUCED. Saying so
    # in the same words every time is the point: a reader scanning a verify
    # report should not have to work out which of these checks re-ran the
    # model and which one read the file.
    inconsistent = [
        row["index"]
        for row in rows
        if bool(row.get("depended_on")) != (float(row.get("dependence") or 0.0) > floor)
    ]
    if inconsistent:
        measured["inconsistent_passages"] = inconsistent
        return Check(
            "grounding",
            DIFFERS,
            f"{len(inconsistent)} passage(s) carry a verdict their own "
            f"numbers do not support — {inconsistent[:4]} say the answer did "
            f"or did not depend on them, against a recorded floor of "
            f"{floor:.6f} that says the opposite. The file has been edited, "
            f"or written by something that did not use the floor it stored.",
            measured,
        )

    top = max(rows, key=lambda r: float(r.get("dependence") or 0.0))
    detail = (
        f"the {len(rows)} recorded passages are internally consistent with "
        f"the floor stored beside them ({floor:.6f}), and #{top['index']} "
        f"carries the answer at {float(top['dependence']):.4f} nats. "
        f"NOT RE-MEASURED: a `.mri` carries passage previews rather than your "
        f"passages, deliberately — a grounding document is usually the "
        f"private half of the pair — so there is nothing here to mask out and "
        f"run again. Re-run it against the same document to reproduce it."
    )
    if degenerate:
        detail += (
            " The recorded floor is exactly zero, so 'depended on' there means "
            "'moved the answer at all' and no significance test was taken."
        )
    return Check("grounding", NOT_VERIFIABLE, detail, measured)


def _check_ranking(parsed, runtime, blocked: str) -> Check | None:
    """Does the same head still carry the answer, by the same margin?

    TWO questions, because they fail differently. The scores can drift while
    the order holds, which is float noise; the order can change while the
    scores barely move, which is a different claim about the model. A reader
    who is told only "reproduced" cannot tell those apart, so both are
    measured and the sentence names whichever broke.

    The tolerance is `rank_heads`'s own noise floor, taken from THIS run: the
    same forward pass twice with nothing ablated. It was built for exactly
    this purpose -- "anything at or below this is arithmetic, not the model" --
    and using the freshly measured one rather than the file's is what makes it
    a measurement of this machine.
    """
    stored = parsed.ranking or {}
    rows = stored.get("ranked") or []
    if not rows:
        if not _receipt_for(parsed, "ablate_heads"):
            return None
        # Written before the ranking section existed. Named rather than
        # skipped: silence would read as "it reproduced".
        return Check(
            "head ranking",
            NOT_VERIFIABLE,
            "this file records that a head ranking ran but does not carry it. "
            "Files written before the ranking section existed have nothing "
            "here to compare against.",
        )
    if blocked:
        return Check("head ranking", NOT_VERIFIABLE, blocked)

    from . import ablate

    baseline = stored.get("baseline")
    layer = stored.get("layer")
    try:
        fresh = runtime.ablate_heads(layer=layer, baseline=baseline)
    except (BadRequest, Refusal) as err:
        return Check(
            "head ranking",
            NOT_VERIFIABLE,
            f"the ranking could not be re-taken here — {err}",  # leak-ok: authored
        )

    here = {(r["layer"], r["head"]): r["kl"] for r in fresh.get("ranked") or []}
    there = {(r["layer"], r["head"]): r["kl"] for r in rows}
    shared = sorted(here.keys() & there.keys())
    if not shared:
        return Check(
            "head ranking",
            NOT_VERIFIABLE,
            "the re-run ranked a different set of heads entirely, so there is "
            "no head the two rankings have in common to compare.",
        )

    floor = float(fresh.get("noise_floor_kl") or 0.0)
    worst_head, worst_gap = shared[0], 0.0
    for head in shared:
        gap = abs(here[head] - there[head])
        if gap > worst_gap:
            worst_head, worst_gap = head, gap

    # Order, separately. The top-k set is what a reader acts on -- "layer 6
    # head 9 carries this" -- and it can change while every score stays inside
    # the floor.
    top = min(5, len(shared))
    top_here = [h for h in sorted(here, key=lambda h: -here[h])][:top]
    top_there = [h for h in sorted(there, key=lambda h: -there[h])][:top]
    shared_top = len(set(top_here) & set(top_there))
    rank_correlation = ablate.spearman(
        [here[h] for h in shared], [there[h] for h in shared]
    )

    measured = {
        "heads_compared": len(shared),
        "heads_in_file": len(there),
        "max_abs_kl_diff": worst_gap,
        "worst_head": f"L{worst_head[0]}H{worst_head[1]}",
        "tolerance": floor,
        "tolerance_from": (
            "this run's own noise floor — the same forward pass twice with "
            "nothing ablated"
        ),
        "top_k": top,
        "top_k_shared": shared_top,
        "spearman": rank_correlation,
        "baseline": baseline,
    }

    scores_hold = worst_gap <= floor
    order_holds = shared_top == top
    if scores_hold and order_holds:
        return Check(
            "head ranking",
            REPRODUCED,
            f"all {len(shared)} heads score within {worst_gap:.2e} of the "
            f"file against a {floor:.2e} noise floor, and the top {top} are "
            f"the same heads.",
            measured,
        )
    if order_holds:
        return Check(
            "head ranking",
            DIFFERS,
            f"the top {top} heads are unchanged, but L{worst_head[0]}"
            f"H{worst_head[1]} scores {here[worst_head]:.5f} here against "
            f"{there[worst_head]:.5f} in the file — a gap of {worst_gap:.2e} "
            f"outside the {floor:.2e} noise floor.",
            measured,
        )
    return Check(
        "head ranking",
        DIFFERS,
        f"the ranking moved: {top - shared_top} of the top {top} heads are "
        f"different"
        + (
            f" (Spearman {rank_correlation:.2f} across all {len(shared)} heads)"
            if rank_correlation is not None
            else ""
        )
        + ". This is a different claim about which head carries the answer, "
        "not a difference in the last digits.",
        measured,
    )


# ---------------------------------------------------------------- the driver


def verify(path: str | Path, runtime) -> Report:
    """Re-run what this machine can, and say what it could not.

    `runtime` is an unloaded `ModelRuntime`; this loads the file's model into
    it and leaves it loaded for the caller to unload. Taking it as an argument
    rather than building one means `verify` runs through exactly the same code
    paths the server calls, which is the only thing that makes the comparison
    worth anything -- a second implementation would be verifying itself.
    """
    target = Path(path)
    try:
        data = target.read_bytes()
    except OSError as err:
        raise BadRequest(
            f"{target.name} could not be read ({err.strerror or err})"
        ) from None

    parsed = session_mod.parse(data)
    model = (parsed.meta or {}).get("model")
    report = Report(file=target.name, model=model)

    report.notes.append(
        "Bit-exact reproduction across two machines is not achievable — kernel "
        "selection, cuDNN version and TF32 all move the last digits — so every "
        "tolerance below was measured on THIS machine rather than assumed, and "
        "the result is a classification and not a pass."
    )
    report.notes.append(
        "A number reproducing does not make the finding right. It means the "
        "same setup produced the same number here."
    )

    if not model:
        report.checks.append(
            Check(
                "model",
                NOT_VERIFIABLE,
                "this file does not name the model it was measured on, so "
                "nothing in it can be re-run.",
            )
        )
        return report

    try:
        runtime.load(model, confirm=True)
    except (BadRequest, Refusal) as err:
        # This project's own sentences, written for this reader, and safe to
        # publish: "not enough VRAM", "not in the local cache", "unsupported
        # architecture". They are the most useful thing the report can say.
        report.checks.append(
            Check(
                "model",
                NOT_VERIFIABLE,
                f"{model} could not be loaded here — {err}",  # leak-ok: authored
            )
        )
        return report
    except Exception as err:
        # A library's exception, which is NOT safe to publish. transformers and
        # huggingface_hub routinely put absolute paths in theirs, and a report
        # is the part of this feature most likely to be pasted into an issue.
        # The type is named because it is the actionable half, and the full
        # text stays in the log where it does not travel.
        log.warning("loading %s for verification failed", model, exc_info=err)
        report.checks.append(
            Check(
                "model",
                NOT_VERIFIABLE,
                f"{model} could not be loaded here — the loader raised "
                f"{type(err).__name__}. The full text is in the server log; it "
                f"is kept out of this report because a library's exception can "
                f"carry a path from this machine, and a report is made to be "
                f"forwarded.",
            )
        )
        return report

    setup, blocked = _setup_gap(parsed, runtime)
    report.setup = setup
    if blocked:
        report.notes.append(blocked)

    # Generation first, and not only because it reads first: re-running it is
    # what puts the model into the state the attention section describes.
    generation = _check_generation(parsed, runtime, blocked)
    report.checks.append(generation)

    # ATTENTION DEPENDS ON THE GENERATION; PATCHING DOES NOT. `attention()`
    # reads off `last_ids` and refuses outright when there is none, so a file
    # whose generation could not be re-established here has nothing for it to
    # read -- the first version let it through and the refusal came back as a
    # crash instead of a verdict. `patch_trace` takes both prompts and runs
    # its own forwards, so it is checkable either way, and collapsing the two
    # into one dependency would throw away a verifiable measurement.
    if generation.verdict == REPRODUCED:
        report.checks.append(_check_attention(parsed, runtime, blocked))
    elif generation.verdict == DIFFERS:
        report.checks.append(
            Check(
                "attention",
                NOT_VERIFIABLE,
                "this machine produced a different continuation, so its "
                "attention describes a different sequence of tokens than the "
                "file's. Two maps over two different sentences are not "
                "comparable, and reporting them as differing would blame the "
                "attention for the generation.",
            )
        )
    else:
        report.checks.append(
            Check(
                "attention",
                NOT_VERIFIABLE,
                "attention is read off a generation, and this file's could not "
                f"be re-established here — {generation.detail}",
            )
        )

    report.checks.append(_check_patch(parsed, runtime, blocked))

    if generation.verdict == REPRODUCED:
        lens = _check_lens(parsed, runtime, blocked)
    elif parsed.lens or _receipt_for(parsed, "logit_lens"):
        lens = Check(
            "logit lens",
            NOT_VERIFIABLE,
            "the lens is read off a generation, and this file's could not be "
            f"re-established here — {generation.detail}",
        )
    else:
        lens = None
    if lens is not None:
        report.checks.append(lens)

    # Like attention, the ranking is measured at a position inside the run, so
    # it needs that run re-established. Unlike attention it is expensive --
    # n_layers x n_heads forward passes -- which is why it goes last: a reader
    # watching the output has already seen everything cheap by the time this
    # starts.
    if generation.verdict == REPRODUCED:
        ranking = _check_ranking(parsed, runtime, blocked)
    elif parsed.ranking.get("ranked") or _receipt_for(parsed, "ablate_heads"):
        ranking = Check(
            "head ranking",
            NOT_VERIFIABLE,
            "a ranking is measured at a position inside a generation, and "
            f"this file's could not be re-established here — {generation.detail}",
        )
    else:
        ranking = None
    if ranking is not None:
        report.checks.append(ranking)

    # Deliberately NOT gated on the generation. Grounding runs its own
    # document and question, so this file's run being unreproducible here says
    # nothing about it — and it is a file check rather than a re-measurement,
    # so `blocked` does not apply either.
    ground = _check_ground(parsed, runtime, blocked)
    if ground is not None:
        report.checks.append(ground)

    # Deliberately NOT gated on the generation, and not on `blocked` either.
    # A model diff is about two OTHER models; this file's own run being
    # unreproducible here says nothing about it.
    model_diff_check = _check_model_diff(parsed, runtime, blocked)
    if model_diff_check is not None:
        report.checks.append(model_diff_check)

    return report


def render(report: Report) -> str:
    """The report as a terminal reads it."""
    mark = {REPRODUCED: "✓", DIFFERS: "✗", NOT_VERIFIABLE: "–"}
    lines = [f"{report.file} — measured on {report.model or 'an unnamed model'}"]

    setup = report.setup
    if setup.get("file_dtype") or setup.get("here_dtype"):
        lines.append(
            f"  file: {setup.get('file_dtype') or '?'} on "
            f"{setup.get('file_device') or '?'}"
            f"    here: {setup.get('here_dtype') or '?'} on "
            f"{setup.get('here_device') or '?'}"
        )
    if setup.get("file_revision") or setup.get("here_revision"):
        same = setup.get("file_revision") and setup.get("file_revision") == setup.get(
            "here_revision"
        )
        lines.append(
            f"  commit: {(setup.get('file_revision') or 'unrecorded')[:12]}"
            + (
                "  (same weights)"
                if same
                else f"  vs {(setup.get('here_revision') or 'unknown')[:12]} here"
            )
        )
    lines.append("")

    for check in report.checks:
        lines.append(f"  {mark.get(check.verdict, '?')} {check.name}: {check.verdict}")
        lines.append(f"      {check.detail}")
    lines.append("")
    lines.append(
        f"  {report.reproduced} reproduced · {report.differed} differ · "
        f"{report.unverifiable} not verifiable"
    )
    for note in report.notes:
        lines.append(f"  note: {note}")
    return "\n".join(lines)
