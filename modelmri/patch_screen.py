"""A cheap approximation of `patch.trace`, published with its own error.

`patch.trace` pays a real forward pass for every cell of a COMPONENTS x layers
x positions grid. On a 12-layer model over an 11-token prompt that is 396
passes for the grid alone, and the panel that opens it is the one people wait
on. The grid is also almost entirely uninteresting: the whole reason anybody
reads it is the handful of bright cells, and 380-odd passes were spent
establishing that the rest are dim.

ATTRIBUTION PATCHING is the first-order Taylor expansion of that grid. Run the
corrupted prompt once with the gradient of the metric taken with respect to
every site's activation, and the effect of replacing that activation with the
clean one is approximated by a dot product:

    recovery(L, p)  ~=  (clean_act[L, p] - corrupt_act[L, p]) . dM/dact[L, p]
                        -------------------------------------------------
                                              gap

Two forward passes and one backward, for the entire grid, at every site at
once. That is the feature. It is also, exactly, a picture rather than a
measurement, and this module is built around that sentence rather than around
the speed-up.

THE NUMBERS LOOK IDENTICAL TO `patch.trace`'S AND ARE NOT THE SAME QUANTITY.
They share a scale — the share of the clean-to-corrupt logit gap — so they
plot on the same colour ramp and read the same way, and that is the danger.
A payload from here therefore cannot be mistaken for one from there by any
consumer that does not look: it carries `approximate: True`, its grids are
`screen_grids` and not `grids`, its per-site number is `attribution` and not
`recovery`, and there is no `sites`, no `passes` and no `controlled` key for a
frontend to read out of habit. A reader who wires this into the exact panel
gets a KeyError, which is the correct outcome.

MEASURE THE APPROXIMATION'S OWN ERROR, EVERY RUN. A screen whose agreement
with the thing it screens for was never measured is a guess with a
leaderboard. So the top of the screen's own ranking is patched EXACTLY — a
handful of real passes — and the rank correlation, the largest disagreement,
the sign flips and the worst rank move between the two are in the payload
beside the ranking they qualify. `ablate.spearman` does the correlation, and
returns `None` rather than 0.0 when one side is constant, which is the
difference between "these disagree" and "one of them is not a ranking".

`largest_disagreement` is over EVERY site the run patched exactly, near-zero
probes included, and not only over the verified top-k. It is published as the
screen's own resolution, and a resolution computed from a subset of the run's
own measurements is not one. MEASURED, on `_fixture(curved=True, seed=25)`:
the verified top-k disagreed by at most 0.3717, and in the same payload the
probe at `mlp L0@3` — screened at +0.0473, exactly patched at +1.4166 —
disagreed by 1.3693. Quoting 0.3717 as the resolution there understated it by
3.7x, at the site with the largest exact recovery in the run. The top-k figure
is still reported, under `largest_disagreement_verified_only`, because that is
the one that belongs beside `spearman`: both are over a range restricted by
construction.

Those verified rows are not wasted, either: an exact patch on a shortlisted
site is the first passes of the shortlist run this screen exists to make
possible. `exact_recovery` on those rows is a measurement. On every other row
it is `None`, never 0.0.

A SITE THIS RANKS NEAR ZERO IS NOT "DOES NOT MATTER". A first-order
approximation is worst exactly where the effect is non-linear, which is where
a saturating attention pattern or a gate flips — and those are the sites an
interpretability reader most wants. Ranking near zero here means the linear
term is small, and nothing more. So sites the screen ranked NEAREST ZERO are
also patched exactly, and the largest exact recovery among them is reported: a
number in the payload that a reader can hold against the top of the shortlist.
MEASURED on the hand-built non-linear fixture in tests/test_patch_screen.py,
the site the screen ranked nearest zero with a live gradient — `resid L2@2`,
screened at +0.000109 — recovered +0.000591 under the real patch, 5.4 times
its screened value. On the linear fixture, where the approximation is the
answer, the same three probes agreed to seven places.

HOW WRONG IT GETS, MEASURED. On that same non-linear fixture the screen ranked
`resid L0@1` at +1.596 and the exact patch measured it at exactly +1.000 — and
that cell is the one whose true value is known by construction, because layer
0's input IS the embedding and replacing the only token the two prompts differ
in restores the clean prompt outright. A 60% overshoot on the cell that cannot
be wrong. Over the six verified sites Spearman came back +0.7143 against a
resolution of 0.0571, the worst rank move was 2 places (`mlp L0@1` screened
fourth at +0.714 and measured sixth at +0.253), and no site had its sign wrong.
The same six on the linear fixture: Spearman exactly 1.0, nothing moved, and
the largest disagreement was 2.98e-07 — float32's own noise. Those are toy
numbers from a toy model and they are labelled as such; what they establish is
that the disagreement is real and that the payload carries it.

It is NOT reliably largest at the top of the ranking, which is why the
resolution is taken over every exactly-patched site rather than over the
verified ones. On seed 25 of the same fixture family the largest disagreement
in the run is at a near-zero probe and is 3.7x the worst one at the top. Nor
is the sign always right: sweeping seeds 0-59 of `_fixture(curved=True)`,
seeds 10 and 24 produce 3 and 2 verified sites whose exact recovery has the
opposite sign to the screen's, against `sign_flips: 0` everywhere else.

PASSES GO DOWN; PEAK MEMORY GOES UP, BY A FACTOR THIS COUNTS. `patch.trace`
runs every pass under `torch.no_grad` and holds ONE component's activation
cache at a time. This holds FOUR caches that each span every component and
every layer — the clean activations, the corrupt ones, one gradient-carrying
zero per site, and the gradients `autograd.grad` returns — and, on top of all
four, a full backward graph for one forward pass, which is the same shape of
allocation training does.

That sentence used to read "holds a full backward graph for one forward pass"
and stop there, which named the smallest of the five things and skipped the
four the module allocates itself. MEASURED, by summing distinct tensor storage
at the end of the gradient pass, at two sizes, float32, 4 positions:

    6 layers,  d=128:  36,864 + 36,864 + 36,864 + 26,624 = 137,216 B held,
                       against 12,288 B for one component's cache -> 11.2x
    12 layers, d=256: 147,456 + 147,456 + 147,456 + 102,400 = 544,768 B held,
                       against 49,152 B for one component's cache -> 11.1x

The ratio is structural rather than a property of those two sizes: three
grid-sized caches plus the gradients, against `patch.trace`'s one component.
It would be exactly 12x if every tap took a gradient; it comes in a little
under because a tap on a path that cannot reach the metric returns `None`
rather than a tensor of zeros. Every run reports its own four numbers in
`cost.activation_bytes_held`, which is counted from the tensors and therefore
exists on CPU, where `cost.memory.peak_bytes` is `None` because there is no
allocator to ask.

The saving is in passes and in seconds, and it is not in memory, and both
halves are in `cost` rather than only the flattering one. The backward is
`torch.autograd.grad` against the taps and NOT `metric.backward()`, which
would accumulate a gradient into every parameter of the model — 3.4 GB on a
1.7B model in bfloat16, allocated for nothing, and left behind on the caller's
model afterwards. A test asserts no parameter's `.grad` survives a screen.

NaN AND inf ARE REFUSED HERE AND NOT THERE. `patch.trace`'s gap floor is
`gap < MIN_GAP`, and no non-finite value fails a `<`: every comparison against
NaN is False, and `inf < 0.5` is False too. A model whose head overflows one
logit therefore walks through that guard, and what comes out the other side is
a grid of 0.0 at every site — a finite delta divided by an infinite gap — with
`exact_recovery: 0.0` on the verified rows and a `p: nan` in the answer, which
is the fabricated "this site does nothing" this module's notes say it never
emits. So finiteness is checked explicitly here, on the clean logits, the
corrupt logits and the gap, and per site on the score and both norms; a site
that scores non-finite is a null cell and is excluded from the ranking with
its name and the count in the payload. This is a deliberate divergence from
`patch.trace` rather than an inherited refusal, and it is the one place the
two do not refuse the same pairs.

THE SAVING IS REPORTED IN BOTH UNITS, AND IT IS NOT ALWAYS A SAVING. Passes
are portable and a backward pass is not a forward pass, so the two are counted
separately rather than added into a single flattering integer; seconds are the
unit where a backward is priced honestly, and the per-pass figure is measured
here, on this machine, from the verification passes this run actually spent.
On a small enough model the screen costs MORE than the grid it replaces — two
passes and a backward against a grid of a few dozen cells — and that case is
reported rather than hidden, because it is the case a fixture and a toy model
are in. MEASURED, four screens back to back on the non-linear fixture in one
process: the gradient pass read 0.5924 s, then 0.0051, 0.0027, 0.0025, and the
projected saving flipped from -0.53 s to +0.03 s between the first reading and
the second. The first screen in a process pays autograd's warm-up and is not
the cost of a screen; `seconds_basis` says so in the payload, beside the
number, rather than here where only a maintainer would find it.
"""

from __future__ import annotations

import math
import time
from typing import Any

import torch

from . import fmt, patch
from .errors import BadRequest, Refusal

# How many sites come back as candidates for the exact grid. Not a measured
# threshold and it does not pretend to be one: it is the length of a list a
# person reads, and it is reported beside the number of sites that were scored
# so the cap is visible rather than implied.
DEFAULT_SHORTLIST = 12

# How many of the shortlist get a real patch, to measure this method's own
# error. Each one is a genuine forward pass, so this is the price of the
# honesty and it is the first thing a caller in a hurry will want to lower.
# It cannot go below MIN_VERIFY.
DEFAULT_VERIFY = 6

# How many of the sites the screen ranked NEAREST ZERO also get a real patch.
# This is the control on the method's known failure, not a sample of the grid:
# first-order attribution is least reliable exactly where the effect is
# non-linear, and a screen that only ever verifies its own top is untestable
# in the direction it is most likely to be wrong.
DEFAULT_NEAR_ZERO_PROBES = 3

# Below two verified sites there is no rank correlation to compute — Spearman
# is undefined on one point — and a screen with no measured agreement is the
# thing this module refuses to publish. Stated as a constant so the refusal
# below can name it.
MIN_VERIFY = 2


def _as_count(name: str, value: Any, floor: int, hint: str = "") -> int:
    """`value` as a whole count, or a `BadRequest` naming what arrived.

    The bool check comes FIRST and is separate, because `isinstance(True, int)`
    is True in Python: `verify=True` would otherwise pass an int check and go
    on to mean 1, which is a different request from the one anybody typed.

    A float lands here rather than at the slice it would have broken. It used
    to raise `TypeError: slice indices must be integers` from
    `shortlisted[:verify]` — AFTER the gradient pass and the whole grid had
    been paid for, with a message about slices at a caller who asked for a
    screen.
    """
    if isinstance(value, bool):
        raise BadRequest(
            f"{name}={value} is a boolean, not a count. Python treats True as "
            f"1 and False as 0, so this would quietly become {int(value)} — "
            f"which is not what anybody types when they mean it. Pass a whole "
            f"number. {hint}".strip()
        )
    if not isinstance(value, int):
        raise BadRequest(
            f"{name}={value!r} is not a whole number, and every one of these "
            f"counts a thing that cannot be fractional — a site in a list, a "
            f"forward pass. Round it yourself, so the rounding is your "
            f"decision rather than a slice's. {hint}".strip()
        )
    if value < floor:
        raise BadRequest(
            f"{name}={value} is below {floor}, which is the smallest value "
            f"that means anything here. {hint}".strip()
        )
    return value


def _nonfinite(t: torch.Tensor) -> int:
    """How many entries of `t` are NaN or an infinity.

    Counted rather than tested, because the count is what gets reported. A
    guard written as `if x < LIMIT` does not catch either one — every
    comparison against NaN is False, and `inf < 0.5` is False as well — so a
    finiteness check has to be explicit or it is not there at all.
    """
    return int((~torch.isfinite(t)).sum())


def _cache_bytes(cache: dict[str, dict[int, torch.Tensor]]) -> int:
    """Bytes of distinct storage held by one of this module's caches.

    By storage and de-duplicated by `data_ptr`, not by `numel * element_size`:
    two entries can be views of one allocation, and counting both would
    inflate the number this is published as.
    """
    seen: set[int] = set()
    total = 0
    for layer_map in cache.values():
        for tensor in layer_map.values():
            storage = tensor.untyped_storage()
            if storage.data_ptr() in seen:
                continue
            seen.add(storage.data_ptr())
            total += storage.nbytes()
    return total


def _targets(component: str, blocks: list) -> list[torch.nn.Module]:
    """Every module this component will be read at, resolved before any hook.

    Resolving the whole list first is not tidiness. `patch._sublayer` raises
    for an architecture that does not expose the sublayer under either
    spelling, and a Mixtral-style stack can expose it at layer 0 and not at
    layer 9. Registering as we go would leave nine live hooks on the caller's
    model when the tenth raised, and a forward hook nobody holds a handle for
    is a leak that changes the next measurement rather than one that shows up
    as a leak.
    """
    if component == "resid":
        return list(blocks)
    return [patch._sublayer(block, component) for block in blocks]


def _tap_in(block: torch.nn.Module, layer: int, values: dict, taps: dict):
    """Record a block's residual input and make it differentiable.

    The trick is the zero. We need `dM/dx` at this point, and `x` is a
    non-leaf whose `.grad` autograd will not fill in — and detaching it to
    make a leaf would sever the graph, so the gradient at every layer but the
    last would come back as zero, silently and plausibly. Adding a zero tensor
    that is itself a leaf leaves the forward arithmetic exactly unchanged
    (`x + 0` is `x`, at every dtype) while giving autograd an input it will
    answer about: `dM/dzero` is `dM/dx` by construction.

    `values[layer]` is the corrupt run's activation, detached and cloned for
    the same reason `patch._capture` clones — the delta it goes into is
    computed after the graph is freed.
    """

    def pre(module, args):
        x = args[0]
        zero = torch.zeros_like(x, requires_grad=True)
        values[layer] = x.detach().clone()
        taps[layer] = zero
        return (x + zero,) + args[1:]

    return block.register_forward_pre_hook(pre)


def _tap_out(module_: torch.nn.Module, layer: int, values: dict, taps: dict):
    """The same tap on a sublayer's OUTPUT, tuple and all.

    Rebuilds the tuple rather than returning a bare tensor, for the reason
    `patch._splice_out` states: attention returns the key/value cache
    alongside the hidden states on several transformers versions, and dropping
    the tail changes what the rest of the block sees without raising.
    """

    def post(module, args, output):
        is_tuple = isinstance(output, tuple)
        y = output[0] if is_tuple else output
        zero = torch.zeros_like(y, requires_grad=True)
        values[layer] = y.detach().clone()
        taps[layer] = zero
        tapped = y + zero
        return (tapped,) + output[1:] if is_tuple else tapped

    return module_.register_forward_hook(post)


def _name(component: str, layer: int, position: int) -> str:
    """The stable name for one site. Same shape as `patch_graph.node_id`."""
    return f"{component} L{layer}@{position}"


def _rank_resolution(n: int) -> float | None:
    """The smallest change in Spearman's rho that `n` items can express.

    Swapping one adjacent pair of ranks changes the sum of squared rank
    differences by exactly 2, and rho is `1 - 6 * sum(d^2) / (n(n^2-1))`, so
    the finest step this statistic has on `n` points is `12 / (n(n^2-1))`.
    On the default six verified sites that is 0.0571: two screens whose rho
    differ by less than that are the same screen, and a rho quoted to four
    places without this number beside it reads as far more precise than six
    points can be. Assumes no ties, which is the case the formula is exact
    for; ties can only make the statistic coarser.
    """
    if n < 2:
        return None
    return 12.0 / (n * (n * n - 1))


def estimate(
    n_layers: int,
    n_positions: int,
    *,
    n_components: int = len(patch.COMPONENTS),
    shortlist: int = DEFAULT_SHORTLIST,
    verify: int = DEFAULT_VERIFY,
    near_zero_probes: int = DEFAULT_NEAR_ZERO_PROBES,
) -> dict:
    """What the screen costs and what it is being compared against, in passes.

    Nobody should discover the price of an analysis by waiting for it — the
    same reason `patch_graph.estimate` and `ablate.estimate_cost` exist. The
    counts here are exact arithmetic on this module's own loop and on
    `patch.trace`'s, read from `patch`'s constants rather than restated, so a
    change to `MAX_CONTROLLED` or `CONTROL_DRAWS` there moves this projection
    with it.

    No seconds. A pass costs what it costs on this machine; `budget.py`
    measured one card between 12 and 71 ms/pass across sessions, so a figure
    in seconds quoted from anywhere but this machine would be fiction. The
    finished screen reports measured seconds because by then it has some.
    """
    n_layers = _as_count("n_layers", n_layers, 0)
    n_positions = _as_count("n_positions", n_positions, 0)
    n_components = _as_count("n_components", n_components, 0)
    shortlist = _as_count("shortlist", shortlist, 1)
    verify = _as_count(
        "verify", verify, 0, f"The floor for a measured screen is {MIN_VERIFY}."
    )
    near_zero_probes = _as_count("near_zero_probes", near_zero_probes, 0)

    if n_layers <= 0 or n_positions <= 0 or n_components <= 0:
        raise BadRequest(
            f"a grid of {n_components} component(s) x {n_layers} layer(s) x "
            f"{n_positions} position(s) has nothing in it, so there is no cost "
            f"to project. These come from the model's config and the prompt's "
            f"tokenization — load a model and pass a prompt that has tokens in "
            f"it."
        )
    if verify < MIN_VERIFY:
        raise BadRequest(
            f"verify={verify} would publish a screen whose agreement with the "
            f"exact patch was never measured. Spearman is undefined on fewer "
            f"than {MIN_VERIFY} points. Ask for at least {MIN_VERIFY}."
        )
    if verify > shortlist:
        raise BadRequest(
            f"verify={verify} is larger than shortlist={shortlist}, and only "
            f"shortlisted sites are patched exactly. Raise the shortlist or "
            f"lower verify."
        )

    cells = n_components * n_layers * n_positions
    # +2 for the two baseline passes `patch.trace` spends before its grid, and
    # +n_components for the clean-cache pass it spends per component. That
    # third term used to be missing here, which under-priced the baseline by
    # exactly `n_components` passes and so under-stated the saving. MEASURED
    # by counting real `model.forward` calls on the curved fixture:
    # `patch.trace` makes 257 and its own `passes` field reports 254 — the
    # three cache passes are outside its counter. This one counts them.
    exact_grid = 2 + n_components + cells
    # And what `patch.trace` ACTUALLY costs, controls included, computed with
    # its own rule: `max(1, max_controlled // len(grids))` sites per component,
    # each paying `draws` random draws plus one shifted-position pass.
    per_component = max(1, patch.MAX_CONTROLLED // n_components)
    controls = per_component * n_components * (patch.CONTROL_DRAWS + 1)
    exact_trace = exact_grid + controls

    screen_forward = 2 + verify + near_zero_probes
    remaining = max(0, shortlist - verify)
    return {
        "approximate": True,
        "screen_forward_passes": screen_forward,
        # NOT added to the forward count. A backward is a different unit of
        # work and folding it into a pass count would quote a saving in a
        # currency this module made up.
        "screen_backward_passes": 1,
        "verification_passes": verify + near_zero_probes,
        "shortlist_remaining_passes": remaining,
        "exact_grid_passes": exact_grid,
        "exact_trace_passes": exact_trace,
        "exact_passes_basis": (
            f"{exact_grid} = 2 baselines + {n_components} clean-cache pass(es), "
            f"one per component + {cells} cell(s), one pass each. The cache "
            f"passes are real forward passes that `patch.trace`'s own `passes` "
            f"counter does not include, so this figure is {n_components} "
            f"higher than that field reports for the same run."
        ),
        "passes_saved_against_exact_grid": exact_grid - (screen_forward + remaining),
        "passes_saved_against_exact_trace": exact_trace - (screen_forward + remaining),
        "seconds": None,
        "seconds_from": "passes are portable; milliseconds per pass are not",
        "memory": (
            "the screen holds a backward graph for one forward pass, which "
            "`patch.trace` never does — it runs everything under no_grad. "
            "Passes go down and peak memory goes up. The finished screen "
            "reports the measured peak; this projection cannot."
        ),
        "means": (
            f"The screen scores all {cells} sites from 2 forward passes and 1 "
            f"backward, then spends {verify + near_zero_probes} exact patches "
            f"measuring its own error, and hands back a shortlist of "
            f"{shortlist}. The exact grid it replaces is {exact_grid} passes, "
            f"or {exact_trace} as `patch.trace` runs it with controls. On a "
            f"small model the screen can cost more than the grid; the saving "
            f"is a function of grid size and is reported either way."
        ),
    }


def screen(
    model: Any,
    tokenizer: Any,
    blocks: list,
    clean: str,
    corrupt: str,
    *,
    device: Any,
    shortlist: int = DEFAULT_SHORTLIST,
    verify: int = DEFAULT_VERIFY,
    near_zero_probes: int = DEFAULT_NEAR_ZERO_PROBES,
    device_kind: str = "cpu",
) -> dict:
    """Rank every patching site by a first-order approximation, and say how wrong it is.

    `blocks` is the decoder block list, passed in rather than found here so
    that the one place which knows the architecture layouts stays the one
    place — the same contract `patch.trace` has, and the same argument order,
    so the two are call-compatible from a runtime that already binds one.

    THIS IS NOT `patch.trace` AND THE RETURN SHAPE SAYS SO. See the module
    docstring for the field names that were chosen to make a mix-up raise
    rather than mislead.

    Refuses on every pair `patch.trace` refuses on, because the approximation
    is of that measurement and inherits its arithmetic: two prompts of
    different token lengths do not have comparable positions, and a pair that
    predicts the same token makes the denominator zero. Those come back as
    `BadRequest` (422) rather than `patch.PatchError`, because the caller has
    to change the request — `errors.py` has the two words for this and a new
    module can raise the right one at the source instead of having its runtime
    translate.

    AND ON THREE MORE, WHICH `patch.trace` DOES NOT REFUSE. Each is a
    `Refusal` (409), because the request is well formed and it is this module
    declining:

      * a non-finite logit, gap, or exact recovery. `patch.trace`'s gap floor
        is a `<` comparison and NaN and inf both pass it; see the module
        docstring.
      * a grid too small to measure the approximation on. `verify` is checked
        against `MIN_VERIFY` as an argument above, but what gets measured is
        `shortlisted[:verify]`, and on a one-site grid that is one site
        however large `verify` was — a published screen with `verified: 1`,
        `spearman: None` and no agreement behind it at all. The achievable
        count is checked before the tapped pass, from the component and site
        counts, and again once the rows exist.
      * a model whose forward produces no gradient.
    """
    from . import ablate, budget

    t0 = time.perf_counter()

    # Every count checked BEFORE a single pass is spent. A `verify=2.7` used
    # to survive as far as `shortlisted[:verify]`, which is after the two
    # forward passes, the backward and the whole grid — and then raised a
    # TypeError about slice indices.
    shortlist = _as_count("shortlist", shortlist, 1)
    verify = _as_count(
        "verify", verify, 0, f"The floor for a measured screen is {MIN_VERIFY}."
    )
    near_zero_probes = _as_count(
        "near_zero_probes",
        near_zero_probes,
        0,
        "Pass 0 to skip the near-zero control, knowing that it is the only "
        "thing here that tests the approximation where it is weakest.",
    )

    if not clean.strip() or not corrupt.strip():
        raise BadRequest("Both prompts have to have something in them.")
    if clean == corrupt:
        raise BadRequest(
            "The two prompts are identical, so there is nothing to screen. "
            "Change one fact in the second one — a name, a number, a place — "
            "and keep everything else the same."
        )
    if verify < MIN_VERIFY:
        raise BadRequest(
            f"verify={verify} would publish a screen whose agreement with the "
            f"exact patch was never measured, which is the one thing this "
            f"module will not do — a screen nobody checked is a guess with a "
            f"leaderboard. Spearman is undefined on fewer than {MIN_VERIFY} "
            f"points. Ask for at least {MIN_VERIFY}, or run `patch.trace` and "
            f"get exact numbers for every site."
        )
    if verify > shortlist:
        raise BadRequest(
            f"verify={verify} is larger than shortlist={shortlist}, and only "
            f"shortlisted sites are patched exactly. Raise shortlist to at "
            f"least {verify}, or lower verify."
        )
    # An `inference_mode` region cannot produce a tensor that requires grad,
    # and the failure surfaces deep inside the forward pass as a message about
    # a view, not about a screen. Caught here, where there is something to say.
    if torch.is_inference_mode_enabled():
        raise Refusal(
            "This screen needs a gradient, and it is being called inside a "
            "torch.inference_mode() region where no tensor can carry one. Run "
            "it outside that region — `patch.trace` works there because every "
            "pass it takes is a plain forward."
        )

    clean_ids = tokenizer(clean, return_tensors="pt").input_ids.to(device)
    corrupt_ids = tokenizer(corrupt, return_tensors="pt").input_ids.to(device)
    n_pos = int(clean_ids.shape[1])
    if int(corrupt_ids.shape[1]) != n_pos:
        raise BadRequest(
            f"The two prompts tokenize to different lengths ({n_pos} and "
            f"{int(corrupt_ids.shape[1])}), so position 3 of one is not "
            f"position 3 of the other and there is no site for the screen to "
            f"score. Clean:  {patch._tokens(tokenizer, clean_ids)}. Corrupt: "
            f"{patch._tokens(tokenizer, corrupt_ids)}. Change the second "
            f"prompt so it splits into the same number of pieces — a shorter "
            f"or longer name is usually all it takes."
        )

    n_layers = len(blocks)
    skipped: list[str] = []

    # ------------------------------------------------------------ clean pass
    #
    # ONE pass for all three components, where `patch.trace` spends three.
    # Capture hooks are read-only, so reading the residual input and both
    # sublayer outputs in the same forward changes nothing about any of them —
    # `trace` runs three because it interleaves each capture with that
    # component's whole grid, and this has no grid to interleave with.
    clean_cache: dict[str, dict[int, torch.Tensor]] = {}
    handles: list = []
    for component in patch.COMPONENTS:
        sink: dict[int, torch.Tensor] = {}
        try:
            targets = _targets(component, blocks)
        except patch.PatchError as err:
            # `trace` does the same and for the same reason: a model with no
            # submodule of this name still has a residual stream, and refusing
            # the whole screen would throw away the two thirds that work. The
            # note travels in the payload.
            skipped.append(f"{component}: {err}")  # leak-ok: PatchError is authored
            continue
        for i, target in enumerate(targets):
            handles.append(
                patch._capture(target, i, sink)
                if component == "resid"
                else patch._capture_out(target, i, sink)
            )
        clean_cache[component] = sink
    if not clean_cache:
        raise Refusal(
            "None of this model's components could be read, so there is "
            "nothing to screen. " + " ".join(skipped)
        )

    # THE ACHIEVABLE VERIFY COUNT, checked here rather than the requested one.
    # `verify >= MIN_VERIFY` above is a check on the ARGUMENT; what ends up
    # measured is `shortlisted[:verify]`, and a grid with one site in it
    # publishes `verified: 1` however large `verify` was — which is exactly
    # the unmeasured screen the refusal above says this module will not
    # publish. This is the upper bound (null layers can only lower it) and it
    # is known before the tapped pass and the backward, so a screen that
    # cannot be measured costs one forward pass to find out rather than all of
    # them. The exact count is checked again once the rows exist.
    possible = len(clean_cache) * n_layers * n_pos
    if possible < MIN_VERIFY:
        raise Refusal(
            f"This model and prompt have at most {possible} patching site(s) "
            f"— {len(clean_cache)} readable component(s) x {n_layers} layer(s) "
            f"x {n_pos} position(s) — and a screen is only publishable here "
            f"when at least {MIN_VERIFY} of its sites can be patched exactly "
            f"to measure the approximation against. Below that there is no "
            f"rank correlation to compute and nothing qualifying the ranking. "
            f"At this size run `patch.trace` instead: the whole exact grid is "
            f"{2 + len(clean_cache) + possible} passes, which is fewer than "
            f"this screen would spend."
        )

    try:
        with torch.no_grad():
            clean_logits = model(clean_ids).logits[0, -1].float()
    finally:
        for h in handles:
            h.remove()

    # NON-FINITE IS NOT A NUMBER, AND IT IS NOT SMALL. An fp16 head can
    # overflow a logit to +inf, and every guard below is a comparison — `a ==
    # b`, `gap < MIN_GAP` — which NaN and inf both walk straight through,
    # because every comparison against NaN is False and `inf < 0.5` is False.
    # Left alone, this run publishes a grid of 0.0 at every site (a finite
    # delta over an infinite gap), `exact_recovery: 0.0` on the verified rows,
    # a clean answer with `p: nan`, and a payload that `json.dumps` refuses.
    # The count is in the refusal because "one logit overflowed" and "the head
    # is all NaN" are different findings.
    bad_clean = _nonfinite(clean_logits)
    if bad_clean:
        raise Refusal(
            f"{bad_clean} of this model's {clean_logits.numel()} final logits "
            f"on the clean prompt are NaN or infinite, so there is no gap to "
            f"divide by and no ranking to take — a score against an infinite "
            f"gap is 0.0 at every site, which would read as 'nothing here "
            f"matters'. This is usually a narrow dtype overflowing: load the "
            f"model in float32 or bfloat16 and run it again. Note that "
            f"`patch.trace` does NOT refuse this pair — its gap floor is a "
            f"`<` comparison, which no non-finite value fails — so this is one "
            f"place the screen is stricter than the measurement it screens for."
        )

    # -------------------------------------------------- corrupt pass, tapped
    corrupt_values: dict[str, dict[int, torch.Tensor]] = {}
    taps: dict[str, dict[int, torch.Tensor]] = {}
    handles = []
    for component in clean_cache:
        values: dict[int, torch.Tensor] = {}
        holes: dict[int, torch.Tensor] = {}
        for i, target in enumerate(_targets(component, blocks)):
            handles.append(
                _tap_in(target, i, values, holes)
                if component == "resid"
                else _tap_out(target, i, values, holes)
            )
        corrupt_values[component] = values
        taps[component] = holes

    grads: dict[str, dict[int, torch.Tensor]] = {c: {} for c in clean_cache}
    try:
        # `enable_grad` explicitly: this is called from a runtime that runs
        # most things under no_grad, and a screen that silently produced a
        # metric with no graph would refuse below with a message about
        # inference_mode that was not true.
        with torch.enable_grad():
            corrupt_logits_full = model(corrupt_ids).logits[0, -1].float()

        bad_corrupt = _nonfinite(corrupt_logits_full.detach())
        if bad_corrupt:
            raise Refusal(
                f"{bad_corrupt} of this model's "
                f"{corrupt_logits_full.numel()} final logits on the CORRUPT "
                f"prompt are NaN or infinite. The clean run was finite, so "
                f"this is the corrupted prompt itself overflowing the head — "
                f"and the metric this screen differentiates is a difference "
                f"of two of those logits, which would make every gradient it "
                f"takes NaN. Load the model in a wider dtype, or corrupt a "
                f"different token."
            )

        a = int(clean_logits.argmax())
        b = int(corrupt_logits_full.detach().argmax())
        if a == b:
            raise BadRequest(
                f"Both prompts predict the same next token "
                f"({tokenizer.decode([a])!r}), so there is no difference for a "
                f"patch to restore and nothing for the screen to rank. Pick a "
                f"pair whose answers actually differ."
            )

        corrupt_logits = corrupt_logits_full.detach()
        ld_clean = float(clean_logits[a] - clean_logits[b])
        ld_corrupt = float(corrupt_logits[a] - corrupt_logits[b])
        gap = ld_clean - ld_corrupt
        # Finiteness FIRST, and explicitly. `gap < patch.MIN_GAP` below is the
        # floor, and a floor written as a `<` is not a guard against NaN or
        # inf: both come back False and sail through. The two logit
        # differences are checked as well as the gap, because a NaN in one and
        # a NaN in the other can cancel into a finite-looking difference.
        if not (
            math.isfinite(ld_clean) and math.isfinite(ld_corrupt) and math.isfinite(gap)
        ):
            raise Refusal(
                f"The clean-to-corrupt logit gap came out non-finite "
                f"(clean {ld_clean}, corrupt {ld_corrupt}, gap {gap}), and "
                f"every score this module publishes is a fraction of that gap. "
                f"Dividing by it would put 0.0 or NaN in every cell of the "
                f"grid, which reads as a measurement that every site does "
                f"nothing. Load the model in a wider dtype and run it again."
            )
        if gap < patch.MIN_GAP:
            raise BadRequest(
                f"The two prompts disagree by only {gap:.4f} logits, which is "
                f"too little to divide by: a screen score is a fraction of "
                f"that gap, so a movement of a fraction of it would read as a "
                f"large share. Pick a pair whose answers differ more clearly."
            )

        # THE METRIC, differentiated at the corrupted run. Attribution
        # patching linearises around the point it is patching INTO, so this is
        # the corrupt run's logits and not the clean run's — taking the
        # gradient at the clean point would answer a different question
        # (what would break it) with numbers that look identical.
        metric = corrupt_logits_full[a] - corrupt_logits_full[b]
        if not metric.requires_grad:
            raise Refusal(
                "This model's forward pass produced no gradient to take: the "
                "metric came back detached from everything upstream of it. "
                "That happens when a model is wrapped so its activations are "
                "not differentiable — a quantised or compiled forward, or one "
                "that detaches between blocks. There is no first-order screen "
                "to take here. `patch.trace` measures the same grid exactly, "
                "at one forward pass per site."
            )

        flat_taps: list[torch.Tensor] = []
        flat_keys: list[tuple[str, int]] = []
        for component, holes in taps.items():
            for layer, tensor in holes.items():
                flat_taps.append(tensor)
                flat_keys.append((component, layer))

        def gradient_pass() -> None:
            # `torch.autograd.grad`, NOT `metric.backward()`. Backward
            # accumulates into `.grad` on every parameter that requires one,
            # which is a second copy of the whole model — 3.4 GB on a 1.7B
            # model in bfloat16 — allocated for a number this screen never
            # reads, and left behind on the caller's model afterwards.
            # `autograd.grad` propagates only along the paths to the taps and
            # writes nothing into the model. A test asserts no parameter's
            # `.grad` survives a screen.
            got = torch.autograd.grad(metric, flat_taps, allow_unused=True)
            for (component, layer), g in zip(flat_keys, got, strict=True):
                if g is not None:
                    grads[component][layer] = g.detach()

        # Timed and measured through `budget.probe_pass`, because this is the
        # allocation that is new in kind: everything `patch.trace` does runs
        # under no_grad, and the honest way to publish "cheaper" is to publish
        # what got more expensive beside it. Returns `None` with a reason on a
        # backend that will not report a peak, and `None` is not zero.
        probe = budget.probe_pass(gradient_pass, device_kind)

        # WHAT THIS RUN IS HOLDING, COUNTED, while it is still holding it.
        # `probe.memory` is the allocator's peak and it is `None` on CPU,
        # where there is no allocator to ask — so on the machine most of this
        # runs on, the module's memory claim had nothing behind it. These
        # bytes are countable anywhere: they are the caches this function
        # itself keeps alive across the backward, summed from the tensors.
        #
        # There are FOUR of them, not one. `clean_cache` spans every component
        # and every layer, `corrupt_values` is a second of the same shape, the
        # `taps` are a third (one zeros tensor per site, allocated to be
        # differentiated against), and `grads` is a fourth as `autograd.grad`
        # returns. `patch.trace` holds ONE component's cache at a time, which
        # is what `patch_trace_equivalent` below is.
        held = {
            "clean_cache": _cache_bytes(clean_cache),
            "corrupt_values": _cache_bytes(corrupt_values),
            "taps": _cache_bytes(taps),
            "grads": _cache_bytes(grads),
        }
        one_component = (
            max(_cache_bytes({c: m}) for c, m in clean_cache.items())
            if clean_cache
            else 0
        )
    finally:
        for h in handles:
            h.remove()

    dtype = str(next(model.parameters()).dtype).removeprefix("torch.")

    # ------------------------------------------------------------- the grids
    #
    # Vectorised per layer rather than per site: one [n_pos, d] temporary at a
    # time, so the peak here is one layer's activation and not a grid of them.
    screen_grids: dict[str, list[list[float | None] | None]] = {}
    delta_norms: dict[tuple[str, int, int], float] = {}
    grad_norms: dict[tuple[str, int, int], float] = {}
    rows: list[dict] = []
    nonfinite_sites: list[str] = []
    for component in clean_cache:
        grid: list[list[float | None] | None] = []
        for layer in range(n_layers):
            if (
                layer not in clean_cache[component]
                or layer not in corrupt_values[component]
                or layer not in grads[component]
            ):
                # NOT a row of zeros. A block that never ran during the
                # forward pass — a block list longer than the stack the model
                # actually walks, or a routed expert that was not selected —
                # has no activation to attribute and no gradient to attribute
                # it with. "We did not measure this" and "this scored 0" are
                # different findings and a zero would publish the wrong one.
                grid.append(None)
                skipped.append(
                    f"{component} layer {layer}: this block produced no "
                    f"activation during the forward pass, so there is nothing "
                    f"to attribute. Its row is null rather than zero."
                )
                continue
            # float() before the dot product: a bfloat16 sum over 2048 terms
            # loses most of what the small terms carry, and this number's
            # whole job is to be compared against an exact one.
            clean_act = clean_cache[component][layer][0].float()
            corrupt_act = corrupt_values[component][layer][0].float()
            grad = grads[component][layer][0].float()
            delta = clean_act - corrupt_act
            scores = (delta * grad).sum(-1) / gap
            dn = delta.norm(dim=-1)
            gn = grad.norm(dim=-1)
            row: list[float | None] = []
            for pos in range(n_pos):
                value = float(scores[pos])
                dn_pos = float(dn[pos])
                gn_pos = float(gn[pos])
                if not (
                    math.isfinite(value)
                    and math.isfinite(dn_pos)
                    and math.isfinite(gn_pos)
                ):
                    # NOT a zero and NOT ranked. A NaN sorts wherever the
                    # comparison happens to put it and would take a place near
                    # the top of a shortlist on no evidence at all; an inf
                    # would take the top outright. The cell is null — the same
                    # word this grid already uses for "not measured" — the
                    # site is out of `rows`, and the count is reported.
                    row.append(None)
                    nonfinite_sites.append(_name(component, layer, pos))
                    continue
                row.append(round(value, 6))
                delta_norms[(component, layer, pos)] = dn_pos
                grad_norms[(component, layer, pos)] = gn_pos
                rows.append(
                    {
                        "component": component,
                        "layer": layer,
                        "position": pos,
                        "name": _name(component, layer, pos),
                        # UNROUNDED here, where `screen_grids` is rounded to
                        # six places for the wire. The agreement figures below
                        # are differences between this and an exact number,
                        # and a reader has to be able to recompute them from
                        # the rows they are quoted against.
                        "attribution": value,
                        "delta_norm": float(dn[pos]),
                        "grad_norm": float(gn[pos]),
                        # Filled in only for sites that were actually patched.
                        # None means "not measured", for every other site.
                        "exact_recovery": None,
                        "exact_error": None,
                    }
                )
            grid.append(row)
        screen_grids[component] = grid

    if nonfinite_sites:
        skipped.append(
            f"{len(nonfinite_sites)} site(s) scored non-finite (NaN or "
            f"infinite) and were excluded from the ranking rather than ranked "
            f"or zeroed: {', '.join(nonfinite_sites[:8])}"
            + (" ..." if len(nonfinite_sites) > 8 else "")
        )

    # Free the graph's residue before spending a single exact pass. The
    # verification loop below is a plain no_grad forward per site and should
    # cost what `patch.trace` costs, not what it costs plus a backward graph
    # nobody is reading any more.
    #
    # EMPTIED, not `del`ed and not rebound. `gradient_pass` above closes over
    # all four names, so deleting them leaves that closure reading names a
    # static reader cannot see are bound — the landmine the previous version
    # of this comment was right to avoid.
    #
    # But it avoided it by REBINDING (`taps = {}`), which is worse than it
    # looks in two ways. It leaves four locals nothing ever reads again, which
    # is what `py/unused-local-variable` correctly flagged; and it drops only
    # THIS scope's reference, while the closure's cell still points at the
    # original dict. `.clear()` empties the very object the closure sees, so
    # the tensors go now rather than when the frame does.
    taps.clear()
    grads.clear()
    corrupt_values.clear()
    flat_taps.clear()
    # `metric = corrupt_logits_full = None` used to sit here and released
    # nothing. `metric` is assigned only INSIDE `gradient_pass`, so it is a
    # local of that function and the name here had never been bound at all —
    # dead on arrival. And `corrupt_logits` is `corrupt_logits_full.detach()`,
    # which SHARES its storage, so dropping one name while the other lives
    # frees a Python object header and not one byte of the tensor.
    resolution = patch.recovery_resolution(model, clean_logits, gap)

    # --------------------------------------------------------- the shortlist
    #
    # Ranked by SIGNED attribution, descending, exactly as `patch.trace` ranks
    # its sites by recovery: the top of this list is "most likely to restore
    # the answer", not "largest effect either way". That rule can exclude a
    # site with a large negative screen score, so the most negative one is
    # named in the payload rather than left out of it.
    #
    # Ties broken by name so the order is total and the same run twice is the
    # same list. Every attribution here is finite — the non-finite ones never
    # entered `rows` — so this comparison means what it says.
    by_score = sorted(rows, key=lambda r: (-r["attribution"], r["name"]))
    shortlisted = by_score[:shortlist]

    # `by_score[-1]` is the LAST of the ranking, which is only "the most
    # negative" when something scored below zero. On a run where every site
    # scored >= 0 it named the smallest positive one and the payload called it
    # the most negative; `strongest_negative` is now None there, with the
    # reason beside it, because "no site scored negative" is a finding and a
    # mislabelled positive number is not.
    least = by_score[-1] if by_score else None
    strongest_negative = least if least and least["attribution"] < 0 else None
    shortlisted_names = {r["name"] for r in shortlisted}
    strongest_negative_on_shortlist = (
        strongest_negative["name"] in shortlisted_names if strongest_negative else None
    )
    strongest_negative_reason = (
        ""
        if strongest_negative
        else (
            f"no site scored below zero in this run, so the exclusion this "
            f"ranking rule can cause did not happen. The lowest score was "
            f"{least['name']} at {fmt.measured(least['attribution'], 4)}."
            if least
            else "no sites were scored."
        )
    )

    # THE FLOOR, ON THE COUNT THAT WILL ACTUALLY BE MEASURED. Checked before
    # any exact pass is spent, and against `shortlisted[:verify]` rather than
    # `verify`, which is the number the refusal at the top of this function
    # was really about.
    if len(shortlisted[:verify]) < MIN_VERIFY:
        raise Refusal(
            f"Only {len(shortlisted[:verify])} site(s) can be patched exactly "
            f"here — {len(rows)} site(s) were scored, the shortlist is "
            f"{len(shortlisted)} long and verify={verify} — and a screen "
            f"whose agreement with the exact patch rests on fewer than "
            f"{MIN_VERIFY} sites has no rank correlation behind it at all. "
            f"That is the one thing this module will not publish. Run "
            f"`patch.trace` on a grid this small: it is "
            f"{2 + len(screen_grids) + len(rows)} passes for exact numbers at "
            f"every site."
        )

    def exact(component: str, layer: int, pos: int) -> float:
        """One real patch at one site. The same splice `patch.trace` makes.

        Deliberately `patch`'s own helpers rather than a copy of them: a
        second definition of "the exact patch" living here would make the
        agreement figures a comparison between this module and itself, which
        is the one thing they must not be.
        """
        target = (
            blocks[layer]
            if component == "resid"
            else patch._sublayer(blocks[layer], component)
        )
        vector = clean_cache[component][layer][:, pos, :]
        handle = (
            patch._splice(target, pos, vector)
            if component == "resid"
            else patch._splice_out(target, pos, vector)
        )
        try:
            with torch.no_grad():
                out = model(corrupt_ids).logits[0, -1].float()
        finally:
            handle.remove()
        return (float(out[a] - out[b]) - ld_corrupt) / gap

    exact_seconds = 0.0
    exact_passes = 0
    nonfinite_exact: list[str] = []

    def verify_row(row: dict) -> None:
        nonlocal exact_seconds, exact_passes
        started = time.perf_counter()
        measured = exact(row["component"], row["layer"], row["position"])
        exact_seconds += time.perf_counter() - started
        exact_passes += 1
        if not math.isfinite(measured):
            # The pass was spent and it did not produce a number. Left as
            # None, which is what every unpatched row says, and counted — a
            # NaN written into this column would poison the max, the Spearman
            # and the sign count downstream, all of which compare against it.
            nonfinite_exact.append(row["name"])
            return
        row["exact_recovery"] = measured
        row["exact_error"] = measured - row["attribution"]

    verified = shortlisted[:verify]
    for row in verified:
        verify_row(row)

    # ----------------------------------------------------- the near-zero control
    #
    # Chosen from sites with a non-zero delta AND a non-zero gradient. A site
    # where the clean and corrupt activations are identical, or where the
    # gradient is structurally zero — a sublayer output at an earlier position
    # in the last layer, which cannot reach the final prediction at all — is
    # zero for a reason that has nothing to do with the approximation, and
    # verifying it would only confirm the geometry.
    verified_names = {r["name"] for r in verified}
    candidates = [
        r
        for r in rows
        if r["name"] not in verified_names
        and r["delta_norm"] > 0
        and r["grad_norm"] > 0
    ]
    candidates.sort(key=lambda r: (abs(r["attribution"]), r["name"]))
    probes = candidates[:near_zero_probes]
    for row in probes:
        verify_row(row)

    # ---------------------------------------------------------- the agreement
    #
    # Only rows that came back with a finite exact number. A pass that was
    # spent and produced a NaN is counted in `verification_passes` and
    # excluded here, by name, rather than carried into a statistic.
    verified = [r for r in verified if r["exact_recovery"] is not None]
    probes = [r for r in probes if r["exact_recovery"] is not None]
    if len(verified) < MIN_VERIFY:
        raise Refusal(
            f"{len(nonfinite_exact)} of the exact patches came back non-finite "
            f"({', '.join(nonfinite_exact[:8])}), leaving {len(verified)} "
            f"measured site(s) — fewer than the {MIN_VERIFY} this module needs "
            f"before it will publish a ranking with an agreement figure "
            f"attached. The splice itself is `patch.trace`'s, so "
            f"`patch.trace` will produce the same NaNs at the same sites; a "
            f"wider dtype is the fix."
        )

    screen_side = [r["attribution"] for r in verified]
    exact_side = [r["exact_recovery"] for r in verified]
    rho = ablate.spearman(screen_side, exact_side)
    rho_reason = ""
    if rho is None:
        # WHICH undefined. `ablate.spearman` returns None for two different
        # reasons and this used to report the second one for both, so a screen
        # with one measured site printed a sentence about a constant ranking.
        rho_reason = (
            f"undefined on {len(verified)} site(s): a rank correlation needs "
            f"at least 2 points and there are fewer here."
            if len(verified) < 2
            else (
                f"undefined on these {len(verified)} sites: one of the two "
                f"rankings is constant, so there is no order for the other to "
                f"agree or disagree with."
            )
        )

    # Rank movement, over the verified set only. Both orderings are of the
    # same sites, so a site's two ranks are comparable and the largest move is
    # the plainest statement of disagreement there is — rho compresses the
    # whole ordering into one number and this does not.
    screen_order = [
        r["name"] for r in sorted(verified, key=lambda r: -r["attribution"])
    ]
    exact_order = [
        r["name"] for r in sorted(verified, key=lambda r: -r["exact_recovery"])
    ]
    # `None`, not 0, below two points. One site's rank cannot move, and a 0
    # printed there says the two rankings agreed perfectly — a claim from no
    # comparison at all.
    worst_rank_move = (
        max(abs(screen_order.index(n) - exact_order.index(n)) for n in screen_order)
        if len(screen_order) >= 2
        else None
    )

    # EVERY SITE THIS RUN PATCHED EXACTLY, not just the top-k.
    # `largest_disagreement` is published as "the screen's own resolution",
    # and a resolution computed over a subset of the run's own measurements is
    # not one. The near-zero probes are exact patches on shortlist-eligible
    # sites, measured in this same run, and leaving them out understated the
    # resolution by 3.7x on a fixture from this module's own family: the
    # verified top-k disagreed by at most 0.3717 while `mlp L0@3`, screened at
    # +0.0473 and probed at +1.4166, disagreed by 1.3693 in the same payload.
    # The top-k figure is still here, under its own name, because a rank
    # correlation over a restricted range is a different statement.
    measured_rows = verified + probes
    disagreements = [(abs(r["exact_error"]), r) for r in measured_rows]
    largest = max(disagreements, key=lambda z: z[0]) if disagreements else None
    verified_disagreements = [(abs(r["exact_error"]), r) for r in verified]
    largest_verified = (
        max(verified_disagreements, key=lambda z: z[0])
        if verified_disagreements
        else None
    )
    # A sign flip is only a finding when the exact number is bigger than what
    # this dtype can express. Below that, both signs are the same measurement.
    # Over the VERIFIED rows, which is what the sentence quoting it says. The
    # probes get their own count rather than being folded in, because "of the
    # sites this screen put at the top, N had the sign wrong" and "of the
    # sites it put near zero, N did" are different findings.
    sign_flips = sum(
        1
        for r in verified
        if r["attribution"] * r["exact_recovery"] < 0
        and abs(r["exact_recovery"]) > resolution
    )
    near_zero_sign_flips = sum(
        1
        for r in probes
        if r["attribution"] * r["exact_recovery"] < 0
        and abs(r["exact_recovery"]) > resolution
    )
    near_zero_worst = (
        max(probes, key=lambda r: abs(r["exact_recovery"])) if probes else None
    )

    # ---------------------------------------------------------------- cost
    screen_forward = 2 + exact_passes
    cells = sum(
        sum(1 for r in grid if r is not None) * n_pos for grid in screen_grids.values()
    )
    # 2 baselines + one clean-cache pass per component + one pass per cell.
    # The cache passes were missing and they are real: `patch.trace` calls
    # `cache_for(component)` once per component and its own `passes` counter
    # does not include them. MEASURED on the curved fixture by counting
    # `model.forward` calls — `patch.trace` makes 257 where it reports 254 —
    # so the baseline this screen prices itself against was 3 passes short.
    exact_grid_passes = 2 + len(screen_grids) + cells
    per_component = max(1, patch.MAX_CONTROLLED // len(screen_grids))
    control_passes = per_component * len(screen_grids) * (patch.CONTROL_DRAWS + 1)
    exact_trace_passes = exact_grid_passes + control_passes
    remaining = max(0, len(shortlisted) - len(verified))
    seconds_per_exact = exact_seconds / exact_passes if exact_passes else None
    projected_grid_seconds = (
        seconds_per_exact * exact_grid_passes if seconds_per_exact is not None else None
    )
    elapsed = time.perf_counter() - t0
    projected_saving = (
        projected_grid_seconds - elapsed - seconds_per_exact * remaining
        if seconds_per_exact is not None and projected_grid_seconds is not None
        else None
    )

    # WHAT THAT RULE ACTUALLY EXCLUDED, IN THIS RUN. The sentence used to say
    # "so a site with a large negative score is not on it" unconditionally and
    # then name a site that was on it — which happens with DEFAULT arguments
    # on any model with fewer sites than the shortlist length, and on every
    # run where `shortlist` covers the grid. A claim about an exclusion has to
    # check whether anything was excluded.
    if not by_score:
        exclusion_sentence = "No sites were scored, so nothing was ranked."
    elif len(shortlisted) == len(by_score):
        exclusion_sentence = (
            "The shortlist is every site that was scored, so that rule "
            "excluded nothing here — the sites with the most negative scores "
            "are on it too, at the bottom"
            + (
                f" ({strongest_negative['name']} at "
                f"{fmt.measured(strongest_negative['attribution'], 4)} is "
                f"last)."
                if strongest_negative
                else "."
            )
        )
    elif strongest_negative is None:
        exclusion_sentence = (
            f"The rule excludes sites with a large negative score, and "
            f"nothing scored below zero in this run: the lowest was "
            f"{least['name']} at {fmt.measured(least['attribution'], 4)}, and "
            f"{len(by_score) - len(shortlisted)} site(s) fell off the end of "
            f"the list on rank alone."
        )
    elif strongest_negative_on_shortlist:
        exclusion_sentence = (
            f"{len(by_score) - len(shortlisted)} site(s) fell off the end of "
            f"it — but not the most negative one, which is "
            f"{strongest_negative['name']} at "
            f"{fmt.measured(strongest_negative['attribution'], 4)} and is on "
            f"the shortlist anyway, because fewer than {len(shortlisted)} "
            f"sites outrank it."
        )
    else:
        exclusion_sentence = (
            f"So a site with a large negative score is not on it: the most "
            f"negative was {strongest_negative['name']} at "
            f"{fmt.measured(strongest_negative['attribution'], 4)}, and it is "
            f"one of {len(by_score) - len(shortlisted)} site(s) the cut "
            f"excluded."
        )

    def row_out(r: dict) -> dict:
        return {
            "name": r["name"],
            "component": r["component"],
            "layer": r["layer"],
            "position": r["position"],
            "attribution": r["attribution"],
            "delta_norm": r["delta_norm"],
            "grad_norm": r["grad_norm"],
            "exact_recovery": r["exact_recovery"],
            "exact_error": r["exact_error"],
        }

    return {
        # THE FLAG, FIRST. Every other field in here is a number that looks
        # exactly like one `patch.trace` produces.
        "approximate": True,
        "method": (
            "attribution patching — the first-order Taylor approximation of "
            "activation patching. (clean_act - corrupt_act) . d(metric)/d(act), "
            "taken at the corrupted run, divided by the clean-to-corrupt gap."
        ),
        "screens": "patch.trace",
        "clean": {
            "prompt": clean,
            "tokens": patch._tokens(tokenizer, clean_ids),
            "answer": patch._answer(clean_logits, tokenizer),
        },
        "corrupt": {
            "prompt": corrupt,
            "tokens": patch._tokens(tokenizer, corrupt_ids),
            "answer": patch._answer(corrupt_logits, tokenizer),
        },
        "gap": round(gap, 6),
        "n_layers": n_layers,
        "n_positions": n_pos,
        "components": list(screen_grids),
        "skipped": skipped,
        "dtype": dtype,
        # `screen_grids`, not `grids`. A null row is a layer that produced no
        # activation and a null CELL is a site that scored non-finite; neither
        # is a zero.
        "screen_grids": screen_grids,
        "n_sites_scored": len(rows),
        # THE EXCLUSION, WITH ITS COUNT. Sites dropped from the ranking for
        # scoring NaN or infinite. Zero on every healthy run, and it is here
        # on every run so that a non-zero one is visible without diffing.
        "n_sites_nonfinite": len(nonfinite_sites),
        "nonfinite_sites": nonfinite_sites,
        "shortlist": [row_out(r) for r in shortlisted],
        "shortlist_size": len(shortlisted),
        "shortlist_requested": shortlist,
        # THE CAP, REPORTED. The list is `shortlist` long out of this many.
        "shortlist_capped_from": len(rows),
        # None when nothing scored below zero, with the reason beside it —
        # never the least positive site wearing a "most negative" label.
        "strongest_negative": row_out(strongest_negative)
        if strongest_negative
        else None,
        "strongest_negative_on_shortlist": strongest_negative_on_shortlist,
        "strongest_negative_reason": strongest_negative_reason,
        "near_zero_probes": [row_out(r) for r in probes],
        # THE OTHER CAP, REPORTED, for the same reason the shortlist's is: a
        # request for 1,000,000 probes used to come back silently truncated to
        # however many candidates existed, with nothing saying so.
        "near_zero_requested": near_zero_probes,
        "near_zero_capped_from": len(candidates),
        "agreement": {
            "verified": len(verified),
            "spearman": rho,
            "spearman_reason": rho_reason,
            # Two rho values closer than this are the same rho. See
            # `_rank_resolution`. Unrounded, for the reason `patch.py` gives
            # about `recovery_resolution`: a resolution rounded away asserts a
            # precision the statistic does not have.
            "spearman_resolution": _rank_resolution(len(verified)),
            # Over every site this run patched exactly — the verified top-k
            # AND the near-zero probes — because this is published as the
            # screen's own resolution and a resolution that skips half of the
            # run's own measurements is not one.
            "largest_disagreement": largest[0] if largest else None,
            "largest_disagreement_at": largest[1]["name"] if largest else None,
            "largest_disagreement_screen": largest[1]["attribution"]
            if largest
            else None,
            "largest_disagreement_exact": (
                largest[1]["exact_recovery"] if largest else None
            ),
            "largest_disagreement_measured_on": len(measured_rows),
            "largest_disagreement_scope": (
                f"the {len(measured_rows)} site(s) patched exactly in this "
                f"run: {len(verified)} from the top of the shortlist and "
                f"{len(probes)} near-zero probe(s)"
            ),
            # And the top-k-only figure, kept under its own name: it is the
            # one that belongs beside `spearman`, which is also over the
            # verified rows and over a deliberately restricted range.
            "largest_disagreement_verified_only": (
                largest_verified[0] if largest_verified else None
            ),
            "largest_disagreement_verified_only_at": (
                largest_verified[1]["name"] if largest_verified else None
            ),
            "sign_flips": sign_flips,
            "near_zero_sign_flips": near_zero_sign_flips,
            "worst_rank_move": worst_rank_move,
            # The exact patches that were spent and came back non-finite.
            # Their rows carry `exact_recovery: null` and are excluded from
            # every figure above, so this is the count that says a pass was
            # paid for and produced nothing.
            "nonfinite_exact": len(nonfinite_exact),
            "nonfinite_exact_at": nonfinite_exact,
            # The dtype's own floor, on the EXACT numbers. The screen's own
            # resolution is `largest_disagreement`, which is a bigger number
            # and the one that matters here.
            "exact_recovery_resolution": resolution,
            "near_zero_probed": len(probes),
            "near_zero_largest_exact": (
                near_zero_worst["exact_recovery"] if near_zero_worst else None
            ),
            "near_zero_largest_exact_at": (
                near_zero_worst["name"] if near_zero_worst else None
            ),
            "near_zero_largest_screen": (
                near_zero_worst["attribution"] if near_zero_worst else None
            ),
            "means": (
                f"The {len(verified)} sites this screen ranked highest were "
                f"patched EXACTLY, at one forward pass each, and `spearman`, "
                f"`worst_rank_move` and `sign_flips` compare the two "
                f"orderings over those sites. Rank correlation over a top-k "
                f"is the hard half of the comparison and not a summary of the "
                f"whole grid: the range is restricted by construction, "
                f"because every site in it already scored near the top. "
                f"`largest_disagreement` is over a wider set — all "
                f"{len(measured_rows)} site(s) this run patched exactly, the "
                f"{len(probes)} near-zero probe(s) included — because it is "
                f"read as the screen's own resolution: two screened sites "
                f"closer together than "
                f"{fmt.measured(largest[0], 4) if largest else 'that'} are not "
                f"ordered by this method, whatever their scores say. Over the "
                f"verified rows alone it would be "
                f"{fmt.measured(largest_verified[0], 4) if largest_verified else 'undefined'}"
                f", which is the number that belongs beside `spearman` and is "
                f"in `largest_disagreement_verified_only`."
            ),
        },
        "cost": {
            "screen_forward_passes": screen_forward,
            # Counted separately and never added in. A backward is not a
            # forward pass and quoting one number would price it at zero.
            "screen_backward_passes": 1,
            "verification_passes": exact_passes,
            "shortlist_remaining_passes": remaining,
            "exact_grid_passes": exact_grid_passes,
            "exact_trace_passes": exact_trace_passes,
            "exact_passes_basis": (
                f"{exact_grid_passes} = 2 baselines + {len(screen_grids)} "
                f"clean-cache pass(es), one per component + {cells} cell(s), "
                f"one pass each. Those cache passes are real forward passes "
                f"that `patch.trace`'s own `passes` field does not count, so "
                f"this figure is {len(screen_grids)} higher than that field "
                f"reports for the same grid. MEASURED by counting "
                f"`model.forward` calls on the test fixture: `patch.trace` "
                f"makes 257 and reports 254."
            ),
            "passes_saved_against_exact_grid": exact_grid_passes
            - (screen_forward + remaining),
            "passes_saved_against_exact_trace": exact_trace_passes
            - (screen_forward + remaining),
            "seconds": round(elapsed, 4),
            "seconds_gradient_pass": round(probe.seconds, 4),
            "seconds_per_exact_pass": seconds_per_exact,
            "seconds_exact_grid_projected": projected_grid_seconds,
            "seconds_saved_projected": projected_saving,
            "seconds_basis": (
                f"`seconds_per_exact_pass` is {exact_passes} exact patch "
                f"pass(es) timed in this run, on this machine, and "
                f"`seconds_exact_grid_projected` is that figure multiplied out "
                f"— a projection, not a measurement of a grid nobody ran. One "
                f"pass is one sample. `seconds_gradient_pass` is ONE sample of "
                f"a kind of work this process may never have done before, and "
                f"it pays autograd's first-call warm-up in full. MEASURED on "
                f"the fixture, four screens back to back in one process: "
                f"0.5924 s, then 0.0051, 0.0027, 0.0025 — a factor of 116 "
                f"between the first and the second, and "
                f"`seconds_saved_projected` flipping from -0.53 s to +0.03 s "
                f"with it. In a separate process the first reading was 1.6785 "
                f"s. Do not quote the first screen's seconds as the cost of a "
                f"screen."
            ),
            # From `budget.probe_pass`. `None` with a reason on a backend that
            # will not report an allocator peak, and `None` is not zero.
            "memory": probe.memory.to_dict(),
            # COUNTED, not projected, and available where `memory` is not —
            # these are bytes of real tensors this function held alive across
            # the backward, summed by storage. `memory.peak_bytes` is the
            # allocator's figure and is `None` on CPU, which is where most of
            # this runs; without these the module's whole memory claim had
            # nothing behind it on the machine making it.
            "activation_bytes_held": {
                **held,
                "total": sum(held.values()),
                "patch_trace_equivalent": one_component,
                "ratio_vs_patch_trace": (
                    round(sum(held.values()) / one_component, 2)
                    if one_component
                    else None
                ),
                "means": (
                    f"At the moment of the backward this screen is holding "
                    f"FOUR caches, not one: the clean activations of every "
                    f"component at every layer ({held['clean_cache']:,} B), "
                    f"the corrupt ones ({held['corrupt_values']:,} B), one "
                    f"gradient-carrying zero per site "
                    f"({held['taps']:,} B), and the gradients "
                    f"`autograd.grad` hands back ({held['grads']:,} B). "
                    f"`patch.trace` holds one component's clean cache — "
                    f"{one_component:,} B here — so this is "
                    f"{round(sum(held.values()) / one_component, 1) if one_component else '?'}x "
                    f"that, before the autograd graph's own saved tensors, "
                    f"which only an allocator can see and which "
                    f"`memory.peak_bytes` reports where there is one."
                ),
            },
            "means": (
                f"The screen scored all {len(rows)} sites from "
                f"{screen_forward - exact_passes} forward passes and 1 "
                f"backward, then spent {exact_passes} exact patches measuring "
                f"its own error. The grid it approximates is "
                f"{exact_grid_passes} passes, or {exact_trace_passes} as "
                f"`patch.trace` runs it with its controls. Running the exact "
                f"patch on the rest of the shortlist costs {remaining} more. "
                f"PASSES ARE NOT THE ONLY COST: the gradient pass holds a "
                f"backward graph, which `patch.trace` never does, and four "
                f"activation caches where `patch.trace` holds one — "
                f"{sum(held.values()):,} bytes against {one_component:,}, "
                f"counted in `activation_bytes_held`. Passes go down and peak "
                f"memory goes up, and both are here."
            ),
        },
        # WHICH SITES WERE EVEN LOOKED AT. `patch_graph.seeding()` treats this
        # as part of the answer and so does this.
        "seeding": (
            f"Every one of the {len(rows)} sites in "
            f"{len(screen_grids)} component(s) x {n_layers} layer(s) x "
            f"{n_pos} position(s) was scored — the screen has no shortlist of "
            f"its own to choose, which is the point of it"
            + (
                f", except for {len(nonfinite_sites)} site(s) that scored "
                f"non-finite and are excluded from the ranking rather than "
                f"placed in it"
                if nonfinite_sites
                else ""
            )
            + f". The returned shortlist is the top {len(shortlisted)} by "
            f"SIGNED attribution, descending. " + exclusion_sentence + " "
            f"The first {min(verify, len(shortlisted))} of the shortlist were "
            f"patched exactly"
            + (
                f", of which {len(verified)} came back finite"
                if len(verified) != min(verify, len(shortlisted))
                else ""
            )
            + f". The near-zero control is the {len(probes)} site(s) with "
            f"the smallest |attribution| among those with a non-zero delta AND "
            f"a non-zero gradient — a site that is zero because the clean and "
            f"corrupt activations agree, or because a sublayer output cannot "
            f"reach the final prediction from where it sits, is zero for a "
            f"reason that has nothing to do with this approximation"
            + (
                f", chosen out of {len(candidates)} such candidate(s)"
                if candidates
                else ""
            )
            + "."
        ),
        "notes": [
            (
                "THIS IS A SCREEN, NOT A MEASUREMENT. Every number in "
                "`screen_grids` is a first-order approximation of the recovery "
                "fraction `patch.trace` measures exactly. It is on the same "
                "scale and it is not the same quantity. Use it to pick which "
                "sites are worth the exact grid; do not report it as one."
            ),
            (
                "A SITE RANKED NEAR ZERO IS NOT A SITE THAT DOES NOT MATTER. "
                "A first-order approximation is worst exactly where the effect "
                "is non-linear — a saturating attention pattern, a gate that "
                "flips — which is where the interesting circuits are. Near "
                "zero here means the linear term is small, and nothing more. "
                + (
                    f"MEASURED IN THIS RUN: of the {len(probes)} site(s) the "
                    f"screen ranked nearest zero, the largest exact recovery "
                    f"was {fmt.measured(near_zero_worst['exact_recovery'], 4)} "
                    f"at {near_zero_worst['name']}, against a screened "
                    f"{fmt.measured(near_zero_worst['attribution'], 4)}."
                    if near_zero_worst
                    else "The near-zero control was not run, so nothing here "
                    "tests the approximation where it is weakest."
                )
            ),
            (
                "The rows carrying an `exact_recovery` are EXACT — a real "
                "forward pass with the clean activation spliced in, the same "
                "one `patch.trace` takes. Every other row's `exact_recovery` "
                "is null, meaning not measured. It is never 0."
            ),
            (
                "A null row in a grid is a layer that produced no activation "
                "during the forward pass, not a layer that scored zero. See "
                "`skipped` for which and why."
            ),
            (
                "A sublayer's output at an earlier position cannot reach the "
                "prediction from the final layer, so the gradient there is "
                "exactly zero and so is the attribution. That is the geometry, "
                "the same zero `patch.trace` reports in the same cells, and it "
                "is not a finding."
            ),
            (
                f"Scores depend on the dtype the model is loaded in, for the "
                f"same reason `patch.trace`'s do — the reference tokens "
                f"themselves can change. These were taken in {dtype} against "
                f"{tokenizer.decode([a])!r} and {tokenizer.decode([b])!r}, and "
                f"a gradient in a narrow float carries its own error on top of "
                f"that. Compare within a dtype, not across one."
            ),
            (
                "NO CONTROLS. `patch.trace` runs eight same-norm random draws "
                "and a shifted-position patch at each of its top sites, and "
                "nothing here does — a control is a forward pass, which is the "
                "thing this method exists to avoid. A shortlisted site has no "
                "verdict behind it, only a rank. The controls happen when the "
                "exact grid runs on the shortlist."
            ),
        ],
        "means": (
            f"An approximate ranking of all {len(rows)} patching sites, from "
            f"{screen_forward - exact_passes} forward passes and one backward, "
            f"against the {exact_grid_passes} the exact grid costs — plus "
            f"{exact_passes} real patches spent here on measuring this "
            f"approximation's own error. Every score is "
            f"(clean - corrupt) . gradient / gap, the first-order term of what "
            f"`patch.trace` measures in full. Its agreement with that "
            f"measurement was checked here, on {len(verified)} site(s) from "
            f"the top of the ranking, "
            + (
                f"and Spearman came back {rho:+.4f} "
                f"(+/- {fmt.measured(_rank_resolution(len(verified)) or 0.0, 4)}, "
                f"which is all {len(verified)} points can express)"
                if rho is not None
                else f"and Spearman is undefined on them — {rho_reason}"
            )
            + f", and {sign_flips} of those had the sign wrong. Across all "
            f"{len(measured_rows)} site(s) patched exactly in this run — the "
            f"{len(probes)} near-zero probe(s) included — the largest "
            f"disagreement between the screen and the exact patch was "
            f"{fmt.measured(largest[0], 4) if largest else 'not measurable'}"
            + (f" at {largest[1]['name']}" if largest else "")
            + ". Read the shortlist as candidates for the exact grid, which "
            "is what it is."
        ),
    }
