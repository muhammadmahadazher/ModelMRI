"""Token counts a provider reported, rolled up — and never a price we guessed.

Two rules, and the whole module is them.

**A count the provider did not return is not zero.** Anthropic returns cache
fields only when a cache was in play and reasoning tokens only from models that
reason. Storing the absence as `0` makes a silent provider indistinguishable
from one that answered zero, which is the `.get(name, 0.0)` bug class that made
206 robot episodes show the same video. Every field here is `int | None` end to
end, and a rollup over a subtree where nobody reported says so rather than
summing to a confident 0.

**A price we are not certain of is not a price.** Every competitor derives cost
from a bundled map with regex model matching, and a regex matching the wrong
model produces a plausible dollar figure with no signal it is wrong — Langfuse
matches by regex with priority-ordered tiers. OTel deliberately defines no cost
attribute for exactly this reason. So:

  * no bundled `prices.json` — a map goes stale between releases and a user on
    a six-month-old install would see six-month-old prices with no way to know,
  * `MODELMRI_PRICES` points at the user's own file or there is no cost column,
  * lookup is EXACT-STRING only. No prefix, no regex, no normalisation, no
    "claude-3-5-sonnet-20241022 is probably claude-3-5-sonnet". `test_ledger`
    asserts each of those does not match,
  * an unpriced call reads "no price on file for <model>", and a run with any
    unpriced call reads "partial — 3 of 11 calls unpriced" rather than a total
    that looks complete.

For an audience running local models the honest unit is tokens, and tokens are
free.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import fmt
from .errors import BadRequest
from .step_kinds import VALID_KINDS

# The only kind that carries provider token counts. A subagent's tokens come
# from its own `llm_call` children, so counting the subagent step as well
# would double every nested run.
#
# CHECKED against `step_kinds.VALID_KINDS` rather than just written down: a
# kind name that does not exist matches no step, so the rollup reports zero
# LLM calls — which reads exactly like a run that made none. The first draft
# of this module said `("llm",)` and would have done that on every real trace.
TOKEN_KINDS = ("llm_call",)
assert set(TOKEN_KINDS) <= VALID_KINDS, "a token kind that no step can have"

# The five counts, in the order a reader wants them. `tokens_in`/`tokens_out`
# predate this module; the other three arrived with it.
FIELDS = (
    "tokens_in",
    "tokens_out",
    "tokens_cache_read",
    "tokens_cache_write",
    "tokens_reasoning",
)

# Which fields a price file may name. Cache read and cache write are priced
# separately and asymmetrically per provider, and getting the write-vs-read
# multiplier backwards is a silent error — so a file that prices one and not
# the other is told which one is missing rather than having it inferred.
PRICED = ("tokens_in", "tokens_out", "tokens_cache_read", "tokens_cache_write")

ENV_VAR = "MODELMRI_PRICES"

# Prices are per million tokens, the unit every provider publishes.
PER = 1_000_000


def _is_count(value) -> bool:
    """Is this a token count this module will use?

    ONE rule, because there were two. `Count.add` refused a bool, float or
    string as a recorder bug rather than coercing it into a plausible number
    -- and `Price.cost` multiplied whatever was in the field. So a step
    recording `tokens_in: 1500.7` had its tokens reported as "not reported by
    provider" in one column and billed in the next, off the same value.

    `tokens_in: "n/a"` was worse. Nothing validates these on the way in --
    `import_trace` hands them to SQLite, whose INTEGER affinity leaves a
    non-numeric string as TEXT, so it comes back out a str -- and `"n/a" *
    3.0` raises TypeError from inside a cost column that is documented as
    optional. The route catches BadRequest only, so the whole trace view
    answered 500 over an optional column.

    `isinstance(True, int)` is True, which is why bool is excluded by name.
    """
    return isinstance(value, int) and not isinstance(value, bool)


class PriceError(BadRequest):
    """The price file cannot be read, and the message says which part."""


@dataclass
class Count:
    """One token field summed over some set of steps.

    `total` is None when NOTHING in the set reported it — not 0. `reported` and
    `silent` are both carried so "3 of 11 calls said nothing" is answerable
    without re-walking the steps.
    """

    field: str
    total: int | None = None
    reported: int = 0
    silent: int = 0

    def add(self, value) -> None:
        if not _is_count(value):
            self.silent += 1
            return
        self.reported += 1
        self.total = value if self.total is None else self.total + value

    def merge(self, other: Count) -> None:
        """Fold another set's count into this one.

        Two Nones stay None. One None and one number is that number — a
        subtree where nobody reported does not drag a parent that did back to
        "not reported".
        """
        if other.total is not None:
            self.total = other.total if self.total is None else self.total + other.total
        self.reported += other.reported
        self.silent += other.silent

    def to_dict(self) -> dict:
        return asdict(self)

    def means(self) -> str:
        if self.total is None:
            return f"{self.field}: not reported by provider"
        if self.silent:
            return (
                f"{self.field}: {self.total:,} across {self.reported} call(s) — "
                f"{self.silent} reported nothing and are not counted as zero"
            )
        return f"{self.field}: {self.total:,}"


@dataclass
class Rollup:
    """Every token field over one subtree or one whole run."""

    counts: dict = field(default_factory=dict)
    n_steps: int = 0
    n_llm_steps: int = 0

    def merge(self, other: Rollup) -> None:
        for name, count in other.counts.items():
            self.counts.setdefault(name, Count(field=name)).merge(count)
        self.n_steps += other.n_steps
        self.n_llm_steps += other.n_llm_steps

    def to_dict(self) -> dict:
        return {
            "counts": {k: v.to_dict() for k, v in self.counts.items()},
            "n_steps": self.n_steps,
            "n_llm_steps": self.n_llm_steps,
        }

    def means(self) -> str:
        if not self.n_llm_steps:
            return "No LLM calls in this subtree, so there are no tokens to count."
        lines = [c.means() for c in self.counts.values()]
        return f"{self.n_llm_steps} LLM call(s). " + "; ".join(lines) + "."


def roll_up(steps, *, kinds=TOKEN_KINDS) -> Rollup:
    """Sum every token field over `steps`, keeping absence distinguishable.

    `kinds` names the step kinds that carry tokens at all. A tool call has no
    token count and must not be counted among the ones that "reported nothing",
    or a run of 40 tool calls and 2 LLM calls would read as 40 silent providers.
    """
    out = Rollup(counts={name: Count(field=name) for name in FIELDS})
    for step in steps:
        out.n_steps += 1
        if str(step.get("kind") or "") not in kinds:
            continue
        out.n_llm_steps += 1
        for name in FIELDS:
            out.counts[name].add(step.get(name))
    return out


def subtree_rollups(steps, *, kinds=TOKEN_KINDS) -> dict:
    """A rollup per step id, covering that step and everything beneath it.

    One pass to build the child lists, then one post-order walk, so a deep
    trace costs O(n) rather than O(n) per node. A cycle in `parent_id` — which
    a hand-written document can contain — is broken rather than recursed into.
    """
    by_id = {}
    children: dict = {}
    for step in steps:
        sid = str(step.get("id") or "")
        if not sid:
            continue
        by_id[sid] = step
        children.setdefault(sid, [])
    for step in steps:
        parent = step.get("parent_id")
        sid = str(step.get("id") or "")
        if parent and str(parent) in children and str(parent) != sid:
            children[str(parent)].append(sid)

    out: dict = {}
    # Post-order without recursion, merging each child's finished rollup into
    # its parent. Re-flattening every subtree instead would be O(n^2) AND — as
    # the first draft of this function was — recursive, which a 4,000-step
    # chain blows the stack on. `test_a_deep_trace_does_not_blow_the_stack`
    # caught exactly that.
    #
    # 0 unseen, 1 in progress, 2 done. A back-edge to an in-progress node is a
    # cycle and is skipped rather than followed.
    state: dict = {}
    for start in by_id:
        if state.get(start):
            continue
        stack = [(start, False)]
        while stack:
            sid, expanded = stack.pop()
            if expanded:
                roll = roll_up([by_id[sid]], kinds=kinds)
                for kid in children.get(sid, []):
                    if state.get(kid) == 2:
                        roll.merge(out[kid])
                out[sid] = roll
                state[sid] = 2
                continue
            if state.get(sid):
                continue
            state[sid] = 1
            stack.append((sid, True))
            for kid in children.get(sid, []):
                if not state.get(kid):
                    stack.append((kid, False))
    return out


# --------------------------------------------------------------- the prices


@dataclass
class Price:
    """What one model costs per million tokens, as the user's file states it."""

    model: str
    per_million: dict

    def cost(self, counts: dict) -> tuple[float | None, list]:
        """Cost for one call, and the fields that had no rate.

        Returns (None, [...]) when NOT ONE field could be priced — a zero here
        would read as a free call.
        """
        total = 0.0
        priced_any = False
        unrated = []
        for name in PRICED:
            count = counts.get(name)
            # Unusable is unreported, exactly as `Count.add` treats it -- so
            # the token column saying "not reported by provider" and the cost
            # column omitting it are the same statement about the same field,
            # rather than two answers to one question.
            if not _is_count(count):
                continue
            rate = self.per_million.get(name)
            if rate is None:
                unrated.append(name)
                continue
            priced_any = True
            total += count * rate / PER
        return (round(total, 6) if priced_any else None), unrated


def load_prices(path=None) -> dict:
    """The user's price file, or {} when they have not pointed at one.

    Absent is NOT an error: no price file means no cost column, which is the
    default and the honest one. A file that exists and cannot be parsed IS an
    error — silently falling back to "no prices" would hide a typo in the very
    file whose whole purpose is being exact.
    """
    raw = path if path is not None else os.environ.get(ENV_VAR, "")
    if not raw:
        return {}
    where = Path(str(raw)).expanduser()
    if not where.is_file():
        raise PriceError(
            f"{ENV_VAR} points at something that is not a file. Unset it for "
            f"token counts with no cost column, or point it at a JSON file of "
            f"{{model: {{tokens_in: <per million>, ...}}}}."
        )
    try:
        doc = json.loads(where.read_text("utf-8"))
    except (OSError, ValueError) as err:
        raise PriceError(
            f"the price file named by {ENV_VAR} could not be read as JSON. A "
            f"cost column built from a file this tool could not parse would "
            f"be a number with no source."
        ) from err
    if not isinstance(doc, dict):
        raise PriceError(
            f"the price file named by {ENV_VAR} must be an object keyed by "
            f"exact model id, and this one is a {type(doc).__name__}."
        )

    out: dict = {}
    for model, rates in doc.items():
        if not isinstance(rates, dict):
            raise PriceError(
                f"the price for {model!r} must be an object of per-million "
                f"rates, and it is a {type(rates).__name__}."
            )
        clean = {}
        for name, rate in rates.items():
            if name not in PRICED:
                raise PriceError(
                    f"{model!r} prices {name!r}, which is not one of "
                    f"{', '.join(PRICED)}. A rate this tool does not know how "
                    f"to apply is a rate somebody expects to be charged."
                )
            if isinstance(rate, bool) or not isinstance(rate, (int, float)):
                raise PriceError(
                    f"the {name} rate for {model!r} must be a number of "
                    f"currency units per million tokens."
                )
            if rate < 0:
                raise PriceError(f"the {name} rate for {model!r} is negative.")
            clean[name] = float(rate)
        out[str(model)] = Price(model=str(model), per_million=clean)
    return out


def price_for(model, prices: dict):
    """The price for EXACTLY this model id, or None.

    No prefix, no regex, no normalisation, no case folding. A near-miss returns
    None and the call reads "no price on file", because the alternative —
    matching `claude-sonnet-4-5-20250929` against a `claude-sonnet-4-5` entry —
    produces a plausible dollar figure with no signal it is wrong, and that is
    the failure this module exists to refuse.
    """
    if not isinstance(model, str) or not model:
        return None
    return prices.get(model)


@dataclass
class Bill:
    """What a run cost, and how much of it is actually known."""

    total: float | None = None
    currency: str = ""
    n_calls: int = 0
    n_priced: int = 0
    unpriced_models: list = field(default_factory=list)

    @property
    def partial(self) -> bool:
        return self.n_priced < self.n_calls

    def to_dict(self) -> dict:
        out = asdict(self)
        out["partial"] = self.partial
        out["means"] = self.means()
        return out

    def means(self) -> str:
        if not self.n_calls:
            return "No LLM calls in this run."
        if self.total is None:
            missing = ", ".join(sorted(set(self.unpriced_models))[:4])
            return (
                f"No price on file for any of this run's {self.n_calls} call(s)"
                + (f" ({missing})" if missing else "")
                + f". Point {ENV_VAR} at a JSON file of exact model ids to get "
                f"a cost column; tokens above are complete either way."
            )
        if self.partial:
            missing = ", ".join(sorted(set(self.unpriced_models))[:4])
            return (
                f"PARTIAL — {self.n_calls - self.n_priced} of {self.n_calls} "
                f"call(s) unpriced ({missing}). The "
                f"{fmt.measured(self.total, 4)} covers "
                f"only the {self.n_priced} with an exact price on file, so it "
                f"is a floor and not the total."
            )
        # `fmt.measured`, not `:,.4f`. `Price.cost` and `bill` both keep
        # six decimals deliberately so the small end survives, and
        # `CostBanner` renders the same field through `measured()` — so a run
        # costing $2.5e-5 showed "$2.5e-5" in bold with "0.0000 across all 1
        # call(s)" on the line beside it. The branch above is worse: there the
        # fabricated zero is explicitly called a floor, which is a claim about
        # a measurement.
        return f"{fmt.measured(self.total, 4)} across all {self.n_calls} call(s)."


def bill(steps, prices: dict, *, kinds=TOKEN_KINDS, currency: str = "") -> Bill:
    """Cost a run against the user's own price file.

    The model id comes from the step's `meta.model`, which is what the recorder
    stored, not from the step name — a name is a label somebody chose and two
    different models can share one.
    """
    out = Bill(currency=currency)
    if not prices:
        # No file: every call is unpriced, and `means()` says how to fix it.
        for step in steps:
            if str(step.get("kind") or "") in kinds:
                out.n_calls += 1
                out.unpriced_models.append(_model_of(step) or "unnamed model")
        return out

    for step in steps:
        if str(step.get("kind") or "") not in kinds:
            continue
        out.n_calls += 1
        model = _model_of(step)
        price = price_for(model, prices)
        if price is None:
            out.unpriced_models.append(model or "unnamed model")
            continue
        cost, _unrated = price.cost({name: step.get(name) for name in PRICED})
        if cost is None:
            out.unpriced_models.append(model or "unnamed model")
            continue
        out.n_priced += 1
        out.total = cost if out.total is None else out.total + cost
    if out.total is not None:
        out.total = round(out.total, 6)
    return out


def _model_of(step: dict):
    meta = step.get("meta")
    if isinstance(meta, dict):
        value = meta.get("model")
        if isinstance(value, str) and value:
            return value
    return None
