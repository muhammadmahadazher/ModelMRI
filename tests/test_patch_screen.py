# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""The screen's arithmetic, and the places it is allowed to be wrong.

No model download. Two hand-built models instead, and the reason is that a
real checkpoint cannot check this at all:

  * A model that is LINEAR end to end has a constant gradient, so the
    first-order approximation is not an approximation — it is the answer.
    Attribution and the exact patch must agree to float precision, and any
    disagreement is a bug in the taps, the metric or the delta. A real
    checkpoint can never make that assertion, because there is no separating
    a small modelling error from a small implementation one.

  * A model with a tanh and a softmax attention is non-linear, so they must
    DISAGREE — and the whole product claim is that the disagreement is
    measured and published rather than assumed small.

The rest of this file is about the payload being unmistakable for an exact
patch result, about `None` never becoming `0`, and about the backward pass
leaving nothing behind on the caller's model.
"""

from __future__ import annotations

import json

import pytest
import torch

from modelmri import patch, patch_screen
from modelmri.errors import BadRequest, Refusal

# A minimal pair in the shape `patch.py`'s reference pair has: one fact
# changed, everything else identical, and both sides tokenizing to four
# whitespace words so the position alignment refusal does not fire.
VOCAB = ["the", "eiffel", "colosseum", "tower", "is", "in", "paris", "rome"]
CLEAN = "the eiffel tower is"
CORRUPT = "the colosseum tower is"


class _Tok:
    """Whitespace tokenizer. Enough to exercise alignment and decoding.

    Same shape as the one in test_patch.py — deliberately, so a fixture that
    exercises the screen exercises `patch.trace` on identical ids and the two
    can be compared cell for cell.
    """

    def __init__(self, vocab: list[str] | None = None) -> None:
        self.vocab = list(vocab or [])

    def __call__(self, text: str, return_tensors=None):
        words = text.split()
        for w in words:
            if w not in self.vocab:
                self.vocab.append(w)
        ids = torch.tensor([[self.vocab.index(w) for w in words]])
        return type("Enc", (), {"input_ids": ids})()

    def decode(self, ids) -> str:
        return "".join(self.vocab[int(i)] for i in ids)


class _LinearAttention(torch.nn.Module):
    """Causal mixing with fixed weights: linear, and it moves information.

    The mixing matters. An "attention" that does not mix positions makes every
    site but the last column exactly zero in both the screen and the exact
    patch, and a test over a grid of zeros passes without touching the
    arithmetic it claims to check.

    Returns a tuple, like a real attention module, so `_tap_out` and
    `patch._splice_out` are exercised on the shape they actually meet.
    """

    def __init__(self, d: int, n_pos: int) -> None:
        super().__init__()
        self.value = torch.nn.Linear(d, d, bias=False)
        self.out = torch.nn.Linear(d, d, bias=False)
        mix = torch.tril(torch.ones(n_pos, n_pos))
        self.register_buffer("mix", mix / mix.sum(-1, keepdim=True))

    def forward(self, x):
        n = x.shape[1]
        return (self.out(self.mix[:n, :n] @ self.value(x)), "cache-sentinel")


class _SoftmaxAttention(_LinearAttention):
    """The same module with a real, input-dependent, saturating mixture.

    This is what makes the model non-linear in its own residual stream: the
    weights the mixture uses depend on the activation being patched, so
    replacing that activation changes both what is mixed and how.
    """

    def __init__(self, d: int, n_pos: int) -> None:
        super().__init__(d, n_pos)
        self.query = torch.nn.Linear(d, d, bias=False)

    def forward(self, x):
        n = x.shape[1]
        scores = self.query(x) @ x.transpose(-1, -2)
        scores = scores.masked_fill(self.mix[:n, :n] == 0, float("-inf"))
        weights = scores.softmax(-1)
        return (self.out(weights @ self.value(x)), "cache-sentinel")


class _Block(torch.nn.Module):
    """attn and mlp under the GPT-2 spelling, so all three grids are readable."""

    def __init__(self, d: int, n_pos: int, *, curved: bool = False) -> None:
        super().__init__()
        self.attn = (
            _SoftmaxAttention(d, n_pos) if curved else _LinearAttention(d, n_pos)
        )
        self.mlp = torch.nn.Linear(d, d, bias=False)
        self.curved = curved

    def forward(self, x):
        x = x + self.attn(x)[0]
        inner = self.mlp(x)
        return x + (torch.tanh(inner) if self.curved else inner)


class _Model(torch.nn.Module):
    def __init__(
        self,
        n_layers: int = 3,
        d: int = 8,
        # Exactly the tokenizer's vocabulary. A model that can predict an id
        # the tokenizer cannot name turns a refusal that quotes the predicted
        # token into an IndexError — which happened, and hid the refusal it
        # was raised from.
        vocab: int = len(VOCAB),
        n_pos: int = 4,
        *,
        curved: bool = False,
    ) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            _Block(d, n_pos, curved=curved) for _ in range(n_layers)
        )
        self.embed = torch.nn.Embedding(vocab, d)
        self.head = torch.nn.Linear(d, vocab, bias=False)

    def forward(self, ids):
        x = self.embed(ids)
        for b in self.blocks:
            x = b(x)
        return type("Out", (), {"logits": self.head(x)})()


# Seed and scale, both chosen by search rather than by taste, and both load
# bearing. `patch.MIN_GAP` refuses a pair whose answers differ by less than 0.5
# logits and `patch.trace` refuses a pair that predicts the SAME token, and a
# default-initialised toy fails one or the other most of the time — measured,
# over seeds 0-59 at five scales, this is the smallest scale where one seed
# clears both refusals in the linear AND the curved fixture. Gaps at this
# setting: 4.5958 linear, 3.5506 curved. Picking a bigger scale instead pushes
# the logits to a magnitude where float32's own step starts to show in the
# agreement figures.
SEED = 1
SCALE = 1.5


def _fixture(*, curved: bool = False, seed: int = SEED):
    """A model, a tokenizer and the block list, with a usable gap."""
    torch.manual_seed(seed)
    model = _Model(curved=curved)
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(SCALE)
    return model, _Tok(VOCAB), list(model.blocks)


def _run(**kwargs):
    model, tok, blocks = _fixture(**{k: v for k, v in kwargs.items() if k == "curved"})
    extra = {k: v for k, v in kwargs.items() if k != "curved"}
    return patch_screen.screen(
        model, tok, blocks, CLEAN, CORRUPT, device="cpu", **extra
    )


def test_a_linear_model_makes_the_first_order_screen_the_exact_answer():
    """The gradient of a linear metric is constant, so the Taylor expansion
    terminates: `(clean - corrupt) . grad` IS the effect of the patch, not an
    approximation of it.

    This is the only assertion in this file that checks the arithmetic itself
    — the taps reading the right tensor, the metric differentiated at the
    corrupt run rather than the clean one, the delta pointing the right way,
    and the division by the same gap `patch.trace` divides by. Any one of
    those wrong and a linear model disagrees. A real checkpoint cannot make
    this assertion, because on one the modelling error and an implementation
    error look identical.
    """
    out = _run(curved=False)
    largest = out["agreement"]["largest_disagreement"]
    assert largest is not None
    assert largest < 1e-4, (
        f"a linear model must make the screen exact; largest disagreement was "
        f"{largest!r} at {out['agreement']['largest_disagreement_at']!r}"
    )
    assert out["agreement"]["sign_flips"] == 0
    assert out["agreement"]["worst_rank_move"] == 0


def test_the_screen_is_wrong_on_a_curved_model_and_publishes_how_wrong():
    """The product claim. A saturating attention and a tanh make the effect of
    a patch non-linear in the activation being patched, so the first-order
    term is not the answer — and the number that says by how much has to be in
    the payload rather than in this docstring."""
    out = _run(curved=True)
    agree = out["agreement"]
    assert agree["verified"] >= patch_screen.MIN_VERIFY
    assert agree["largest_disagreement"] > 1e-3, (
        "a non-linear fixture that agrees to float precision is not exercising "
        "the approximation at all"
    )
    # And it is attributed to a site, not left as a floating quantity.
    assert agree["largest_disagreement_at"] in {
        r["name"] for r in out["shortlist"] + out["near_zero_probes"]
    }
    assert agree["largest_disagreement_screen"] is not None
    assert agree["largest_disagreement_exact"] is not None


def test_the_exact_half_is_the_same_number_patch_trace_reports():
    """The verified rows claim to be exact patches. This checks they are the
    SAME exact patch, cell for cell, against `patch.trace` run on the same
    fixture with the same ids.

    It matters because the agreement figures are the whole point: if this
    module spliced differently — the wrong hook point, a replacement where
    `trace` adds, the clean cache captured at a different moment — then
    `largest_disagreement` would be measuring this module against itself and
    would be a number about nothing.
    """
    model, tok, blocks = _fixture(curved=True)
    out = patch_screen.screen(model, tok, blocks, CLEAN, CORRUPT, device="cpu")
    exact = patch.trace(model, tok, blocks, CLEAN, CORRUPT, device="cpu")

    checked = 0
    for row in out["shortlist"] + out["near_zero_probes"]:
        if row["exact_recovery"] is None:
            continue
        cell = exact["grids"][row["component"]][row["layer"]][row["position"]]
        assert cell == pytest.approx(row["exact_recovery"], abs=1e-5), (
            f"{row['name']}: the screen's exact patch says "
            f"{row['exact_recovery']}, patch.trace says {cell}"
        )
        checked += 1
    assert checked >= patch_screen.MIN_VERIFY


def test_a_site_the_screen_ranks_near_zero_is_patched_exactly_too():
    """First-order attribution is worst exactly where the effect is
    non-linear, so a screen that only ever verifies its own top is untested in
    the direction it is most likely to be wrong. The near-zero control is the
    only thing in the payload that measures that, and it has to come from live
    sites: a site whose delta is zero, or whose gradient is structurally zero
    because a sublayer output cannot reach the final prediction, is zero for a
    reason that has nothing to do with the approximation."""
    out = _run(curved=True)
    probes = out["near_zero_probes"]
    assert len(probes) == patch_screen.DEFAULT_NEAR_ZERO_PROBES
    for row in probes:
        assert row["delta_norm"] > 0 and row["grad_norm"] > 0
        assert row["exact_recovery"] is not None
    assert out["agreement"]["near_zero_largest_exact"] is not None
    assert out["agreement"]["near_zero_largest_exact_at"] is not None
    # And the warning is not left implicit in a field name.
    assert any(
        "NOT a site that does not matter" in n.upper()
        or "NOT A SITE THAT DOES NOT MATTER" in n.upper()
        for n in out["notes"]
    )


def test_the_payload_cannot_be_read_as_an_exact_patch_result():
    """The danger this module was written around: the numbers are on the same
    scale as `patch.trace`'s, plot on the same ramp and read the same way. A
    consumer that was wired to the exact payload and pointed at this one must
    break loudly rather than render an approximation as a measurement."""
    out = _run(curved=True)
    exact = patch.trace(*_fixture(curved=True), CLEAN, CORRUPT, device="cpu")

    assert out["approximate"] is True
    assert "approximate" not in exact
    # The score-bearing keys of an exact result are absent here, by name.
    for key in ("grids", "sites", "recovery", "passes", "controlled"):
        assert key in exact or key == "recovery", f"{key} is not an exact-result key"
        assert key not in out, (
            f"{key!r} appears in a screen payload; a consumer reading it would "
            f"silently render an approximation as a measurement"
        )
    assert "screen_grids" in out and "attribution" in out["shortlist"][0]
    assert "recovery" not in out["shortlist"][0]


def test_an_unverified_site_says_not_measured_rather_than_zero():
    """`exact_recovery: 0.0` on a site nobody patched would be a fabricated
    measurement sitting in the same column as real ones, and it would read as
    'this site does nothing' — the strongest possible claim, from no data."""
    out = _run(curved=True, shortlist=8, verify=3, near_zero_probes=0)
    measured = [r for r in out["shortlist"] if r["exact_recovery"] is not None]
    unmeasured = [r for r in out["shortlist"] if r["exact_recovery"] is None]
    assert len(measured) == 3
    assert len(unmeasured) == 5
    for row in unmeasured:
        assert row["exact_recovery"] is None and row["exact_error"] is None


def test_the_backward_leaves_no_gradient_on_the_model():
    """`metric.backward()` accumulates a gradient into every parameter that
    requires one — a second copy of the whole model, allocated for a number
    this screen never reads and left behind on the caller's model afterwards.
    On a 1.7B model in bfloat16 that is another 3.4 GB. `torch.autograd.grad`
    against the taps propagates only along the paths to them and writes
    nothing. This test is the only thing standing between that distinction and
    a future edit that finds `backward()` more familiar."""
    model, tok, blocks = _fixture(curved=True)
    for p in model.parameters():
        assert p.grad is None
    patch_screen.screen(model, tok, blocks, CLEAN, CORRUPT, device="cpu")
    leaked = [n for n, p in model.named_parameters() if p.grad is not None]
    assert leaked == [], f"a gradient was left on {leaked}"


def test_the_tap_does_not_change_the_forward_pass():
    """`x + zero` has to be `x`. The tap exists to give autograd an input it
    will answer about, and if it perturbed the activation it would be scoring
    a model nobody asked about — silently, since the perturbation is a
    gradient-carrying zero and reads as zero everywhere it is printed."""
    model, tok, blocks = _fixture(curved=True)
    ids = tok(CORRUPT, return_tensors="pt").input_ids
    with torch.no_grad():
        untouched = model(ids).logits.clone()

    values: dict = {}
    taps: dict = {}
    handles = [patch_screen._tap_in(b, i, values, taps) for i, b in enumerate(blocks)]
    try:
        with torch.enable_grad():
            tapped = model(ids).logits
    finally:
        for h in handles:
            h.remove()
    assert torch.equal(untouched, tapped.detach())
    assert tapped.requires_grad, "the tap did not make the output differentiable"
    assert set(taps) == set(range(len(blocks)))


def test_a_block_that_never_ran_is_a_null_row_and_not_a_row_of_zeros():
    """A block list longer than the stack the model actually walks — a routed
    expert that was not selected, a wrongly derived list — leaves a layer with
    no activation to attribute and no gradient to attribute it with. A row of
    zeros there says every site in that layer was measured and found to do
    nothing, which is the opposite of what happened."""
    model, tok, blocks = _fixture(curved=False)
    phantom = _Block(8, 4)
    out = patch_screen.screen(
        model, tok, [*blocks, phantom], CLEAN, CORRUPT, device="cpu"
    )
    assert out["n_layers"] == len(blocks) + 1
    for component, grid in out["screen_grids"].items():
        assert grid[-1] is None, f"{component}: the phantom layer scored numbers"
        assert all(row is not None for row in grid[:-1])
    assert any("produced no activation" in s for s in out["skipped"])
    # And the sites of that layer are absent from the count rather than in it
    # as zeros.
    assert out["n_sites_scored"] == 3 * len(blocks) * out["n_positions"]


def test_the_cap_on_the_shortlist_travels_with_it():
    """A truncated list with no count beside it reads as the whole answer.
    Every other ranking in this package carries the true total; so does this
    one."""
    out = _run(curved=True, shortlist=5, verify=2, near_zero_probes=1)
    assert out["shortlist_size"] == 5
    assert out["shortlist_capped_from"] == out["n_sites_scored"]
    assert out["shortlist_capped_from"] > 5
    # The selection rule itself, not just the count.
    assert "SIGNED attribution" in out["seeding"]
    assert out["strongest_negative"] is not None


def test_the_saving_is_reported_in_passes_and_is_allowed_to_be_negative():
    """The reason the feature exists is the saving, so it is in the payload —
    and on a small enough model it is a LOSS, because two passes and a
    backward are not free against a grid of a few dozen cells. Reporting only
    the flattering case would make the number decoration."""
    out = _run(curved=True)
    cost = out["cost"]
    forward = cost["screen_forward_passes"]
    assert cost["screen_backward_passes"] == 1
    assert forward == 2 + cost["verification_passes"]
    # 2 baselines + one clean-cache pass per component + one pass per cell.
    # The per-component term is the one that was missing; the test below
    # counts the real forward calls rather than restating this arithmetic.
    assert cost["exact_grid_passes"] == (
        2 + len(out["components"]) + out["n_sites_scored"]
    )
    assert cost["exact_trace_passes"] > cost["exact_grid_passes"]
    assert cost["passes_saved_against_exact_grid"] == cost["exact_grid_passes"] - (
        forward + cost["shortlist_remaining_passes"]
    )
    # Measured on this machine in this run, not quoted from anywhere else.
    assert cost["seconds_per_exact_pass"] > 0
    assert cost["seconds_gradient_pass"] > 0
    assert str(cost["verification_passes"]) in cost["seconds_basis"]


def test_the_memory_the_screen_costs_is_named_even_when_it_cannot_be_read():
    """Passes go down and peak memory goes up: this holds a backward graph and
    `patch.trace` never does. On CPU there is no allocator peak to read, and
    the honest answer is `None` with a reason — a zero there would say the
    screen costs no memory, which is the one thing it certainly does."""
    out = _run(curved=True)
    memory = out["cost"]["memory"]
    assert memory["peak_bytes"] is None
    assert memory["reason"], "an unmeasured figure with no reason beside it"
    assert "memory" in out["cost"]["means"].lower()


def test_the_rank_correlation_carries_the_resolution_of_its_own_sample_size():
    """Spearman over six points cannot express a difference smaller than one
    adjacent swap. Quoting rho to four places without that number beside it
    claims a precision six points do not have — the same rule `patch.py`
    applies to its recovery resolution, in a different statistic."""
    assert patch_screen._rank_resolution(6) == pytest.approx(12 / (6 * 35))
    assert patch_screen._rank_resolution(2) == pytest.approx(2.0)
    assert patch_screen._rank_resolution(1) is None
    out = _run(curved=True)
    assert out["agreement"]["spearman_resolution"] == pytest.approx(
        patch_screen._rank_resolution(out["agreement"]["verified"])
    )


def test_an_undefined_correlation_is_none_with_a_reason_not_zero():
    """`ablate.spearman` returns None when one side is constant, and this has
    to carry that through rather than flattening it: 'the screen and the exact
    patch are uncorrelated' and 'one of them is not a ranking' are different
    findings and only one of them is bad news."""
    from modelmri import ablate

    assert ablate.spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None
    out = _run(curved=True)
    rho = out["agreement"]["spearman"]
    if rho is None:
        assert out["agreement"]["spearman_reason"]
    else:
        assert out["agreement"]["spearman_reason"] == ""


def test_the_pairs_patch_trace_refuses_are_refused_here_too():
    """The approximation is OF that measurement and inherits its arithmetic:
    positions only correspond when the two prompts tokenize alike, and the
    denominator is the gap between two answers that have to differ. A screen
    that accepted a pair the exact grid rejects would rank sites of a
    quantity that does not exist."""
    model, tok, blocks = _fixture()
    with pytest.raises(BadRequest) as err:
        patch_screen.screen(model, tok, blocks, "the eiffel", CORRUPT, device="cpu")
    assert "different lengths" in err.value.sentence
    assert "same number of pieces" in err.value.sentence

    with pytest.raises(BadRequest) as err:
        patch_screen.screen(model, tok, blocks, CLEAN, CLEAN, device="cpu")
    assert "identical" in err.value.sentence

    with pytest.raises(BadRequest):
        patch_screen.screen(model, tok, blocks, "   ", CORRUPT, device="cpu")


def test_a_pair_that_agrees_is_refused_rather_than_divided_by():
    """Two of three casually-written pairs predicted the same next token when
    `patch.py` measured them, which makes the gap exactly 0.000000 and every
    attribution a division by zero."""

    class _Flat(_Model):
        def forward(self, ids):
            out = super().forward(ids)
            out.logits = out.logits * 0.0
            out.logits[..., 2] = 1.0
            return out

    torch.manual_seed(SEED)
    flat = _Flat()
    with pytest.raises(BadRequest) as err:
        patch_screen.screen(
            flat, _Tok(VOCAB), list(flat.blocks), CLEAN, CORRUPT, device="cpu"
        )
    assert "same next token" in err.value.sentence


def test_a_screen_whose_error_was_never_measured_is_refused():
    """The one refusal this module has that `patch.trace` does not. A screen
    published without a measured agreement is a guess with a leaderboard, and
    the cheapest way to get one is to ask for zero verification passes. The
    refusal names the floor and names the alternative."""
    model, tok, blocks = _fixture()
    with pytest.raises(BadRequest) as err:
        patch_screen.screen(model, tok, blocks, CLEAN, CORRUPT, device="cpu", verify=1)
    assert str(patch_screen.MIN_VERIFY) in err.value.sentence
    assert "patch.trace" in err.value.sentence

    with pytest.raises(BadRequest) as err:
        patch_screen.screen(
            model, tok, blocks, CLEAN, CORRUPT, device="cpu", shortlist=3, verify=5
        )
    assert "shortlist" in err.value.sentence


def test_a_model_with_no_gradient_is_refused_with_somewhere_to_go():
    """A quantised, compiled or otherwise detached forward has no first-order
    term to take. The failure without this guard is a torch message about a
    tensor that does not require grad, raised from inside the model, at a
    reader who asked for a patching screen."""

    class _Detached(_Model):
        def forward(self, ids):
            out = super().forward(ids)
            out.logits = out.logits.detach()
            return out

    torch.manual_seed(SEED)
    model = _Detached()
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(SCALE)
    with pytest.raises(Refusal) as err:
        patch_screen.screen(
            model, _Tok(VOCAB), list(model.blocks), CLEAN, CORRUPT, device="cpu"
        )
    assert "no gradient" in err.value.sentence
    assert "patch.trace" in err.value.sentence


def test_inference_mode_is_named_rather_than_crashed_into():
    """Inside `torch.inference_mode()` no tensor can carry a gradient, and the
    failure surfaces deep in the forward pass as a message about a view. The
    caller has a real choice to make — move the call out of the region, or run
    the exact grid, which works there — and can only make it if told."""
    model, tok, blocks = _fixture()
    with torch.inference_mode(), pytest.raises(Refusal) as err:
        patch_screen.screen(model, tok, blocks, CLEAN, CORRUPT, device="cpu")
    assert "inference_mode" in err.value.sentence
    assert "patch.trace" in err.value.sentence


def test_the_estimate_prices_the_screen_against_the_grid_it_replaces():
    """Nobody should learn the cost of an analysis by waiting for it, and the
    comparison has to be against what `patch.trace` ACTUALLY costs — controls
    included — rather than the flattering half. Read from `patch`'s own
    constants so a change there moves this with it, instead of leaving a
    second copy of the number to drift."""
    plan = patch_screen.estimate(12, 11)
    assert plan["approximate"] is True
    assert plan["exact_grid_passes"] == 2 + 3 + 3 * 12 * 11
    per_component = max(1, patch.MAX_CONTROLLED // 3)
    assert plan["exact_trace_passes"] == plan[
        "exact_grid_passes"
    ] + per_component * 3 * (patch.CONTROL_DRAWS + 1)
    assert plan["screen_forward_passes"] == 2 + plan["verification_passes"]
    assert plan["screen_backward_passes"] == 1
    assert plan["passes_saved_against_exact_grid"] > 0
    # No seconds, for the reason patch_graph.estimate gives.
    assert plan["seconds"] is None and plan["seconds_from"]
    # And the memory trade is stated even where there is nothing to measure.
    assert "memory" in plan and "backward graph" in plan["memory"]


def test_the_estimate_refuses_a_grid_that_does_not_exist():
    with pytest.raises(BadRequest) as err:
        patch_screen.estimate(0, 11)
    assert "nothing in it" in err.value.sentence
    with pytest.raises(BadRequest):
        patch_screen.estimate(12, 11, verify=1)
    with pytest.raises(BadRequest):
        patch_screen.estimate(12, 11, shortlist=2, verify=6)


def test_the_module_says_what_it_is_everywhere_a_reader_looks():
    """The payload is read in four places — a flag, a name, a sentence per
    field group and a notes list — and the claim has to survive a reader who
    only looks at one of them."""
    out = _run(curved=True)
    assert out["approximate"] is True
    assert "first-order" in out["method"]
    assert out["screens"] == "patch.trace"
    assert "SCREEN, NOT A MEASUREMENT" in " ".join(out["notes"]).upper()
    assert "approximate ranking" in out["means"]
    assert "first-order term" in out["means"]
    assert "no verdict" not in out
    # No fabricated verdict anywhere: counts and receipts, and the reader
    # decides whether the agreement is good enough.
    for key in ("verdict", "quality", "reliable", "trustworthy"):
        assert key not in out and key not in out["agreement"]


# ---------------------------------------------------------------------------
# The tests below exist because a skeptic showed that the suite above passed
# unchanged with the thing each one checks broken. Every one of them was
# written to fail against a specific mutation, and each names it.
# ---------------------------------------------------------------------------


def test_the_resolution_is_measured_over_every_site_the_run_patched_exactly():
    """`largest_disagreement` is published as the screen's own resolution —
    "two sites closer together than this are not ordered by this method" — and
    it was computed over the verified top-k only, silently dropping the
    near-zero probes the SAME run patched exactly.

    On seed 25 of the curved fixture that publishes 0.3717 as the resolution
    of a run that measured a 1.3693 error at `mlp L0@3`, which is also the
    largest exact recovery in the whole run: the number is 3.7x too small at
    the site that most needed it.
    """
    for seed in (SEED, 25):
        model, tok, blocks = _fixture(curved=True, seed=seed)
        out = patch_screen.screen(model, tok, blocks, CLEAN, CORRUPT, device="cpu")
        agree = out["agreement"]
        # Every row this run patched exactly, deduplicated: a near-zero probe
        # can also be a shortlisted row that was not verified.
        measured = {
            r["name"]: r
            for r in out["shortlist"] + out["near_zero_probes"]
            if r["exact_recovery"] is not None
        }
        worst = max(measured.values(), key=lambda r: abs(r["exact_error"]))
        assert agree["largest_disagreement"] == pytest.approx(
            abs(worst["exact_error"])
        ), (
            f"seed {seed}: the run measured an error of "
            f"{abs(worst['exact_error'])} at {worst['name']} and published "
            f"{agree['largest_disagreement']} as its resolution"
        )
        assert agree["largest_disagreement_at"] == worst["name"]
        assert agree["largest_disagreement_measured_on"] == len(measured)
        assert str(len(measured)) in agree["largest_disagreement_scope"]

    # And the top-k figure is still there, under a name that says so.
    assert agree["largest_disagreement_verified_only"] <= agree["largest_disagreement"]
    # Seed 25 is the case that makes the difference load bearing: the worst
    # disagreement of the run is at a probe, not at the top of the ranking.
    assert agree["largest_disagreement_at"] in {
        r["name"] for r in out["near_zero_probes"]
    }
    assert (
        agree["largest_disagreement"] > 3 * agree["largest_disagreement_verified_only"]
    )


def test_the_shortlist_is_ranked_and_is_the_top_of_the_whole_ranking():
    """The ranking IS the product, and replacing the sort with the identity
    passed the whole suite: the payload then shipped a "top 12 by SIGNED
    attribution, descending" whose first six rows scored +0.000000, and those
    six were the ones patched exactly and used for every agreement figure."""
    out = _run(curved=True)
    scores = [r["attribution"] for r in out["shortlist"]]
    assert scores == sorted(scores, reverse=True), (
        f"the shortlist is not in descending order: {scores}"
    )

    # Sorted is not enough — it has to be the TOP of the full ranking.
    everything = _run(curved=True, shortlist=36, verify=2, near_zero_probes=0)
    assert everything["shortlist_size"] == everything["n_sites_scored"]
    full = sorted((r["attribution"] for r in everything["shortlist"]), reverse=True)
    assert scores == full[: len(scores)]

    # And the sites that got the exact passes are the highest scoring ones,
    # which is the claim the agreement figures rest on.
    verified = [r for r in out["shortlist"] if r["exact_recovery"] is not None]
    assert len(verified) == patch_screen.DEFAULT_VERIFY
    assert [r["name"] for r in verified] == [
        r["name"] for r in out["shortlist"][: len(verified)]
    ]
    assert min(r["attribution"] for r in verified) >= max(
        r["attribution"] for r in out["shortlist"][len(verified) :]
    )


def _finite_difference_gradient(model, tok, blocks, out, name, eps=1e-3):
    """dM/d(activation) at one site, from real forward passes only.

    Central differences along each basis direction, spliced in with `patch`'s
    own helpers. It never touches `patch_screen`'s taps, its autograd call or
    its norms, so it is an independent measurement of the gradient rather than
    a second copy of the code that computed it. d is 8 in these fixtures, so
    the whole vector costs 16 passes.
    """
    row = {r["name"]: r for r in out["shortlist"]}[name]
    component, layer, pos = row["component"], row["layer"], row["position"]
    a = out["clean"]["answer"]["id"]
    b = out["corrupt"]["answer"]["id"]
    corrupt_ids = tok(CORRUPT, return_tensors="pt").input_ids
    target = (
        blocks[layer]
        if component == "resid"
        else patch._sublayer(blocks[layer], component)
    )

    sink: dict = {}
    handle = (
        patch._capture(target, layer, sink)
        if component == "resid"
        else patch._capture_out(target, layer, sink)
    )
    try:
        with torch.no_grad():
            model(corrupt_ids)
    finally:
        handle.remove()
    base = sink[layer][:, pos, :].clone()

    def metric(vec):
        h = (
            patch._splice(target, pos, vec)
            if component == "resid"
            else patch._splice_out(target, pos, vec)
        )
        try:
            with torch.no_grad():
                logits = model(corrupt_ids).logits[0, -1].float()
        finally:
            h.remove()
        return float(logits[a] - logits[b])

    grad = torch.zeros(base.shape[-1])
    for i in range(base.shape[-1]):
        up, down = base.clone(), base.clone()
        up[0, i] += eps
        down[0, i] -= eps
        grad[i] = (metric(up) - metric(down)) / (2 * eps)
    return float(grad.norm())


def test_the_published_grad_norm_is_the_gradient_and_not_a_copy_of_something():
    """`grad_norm` is a published column and the only assertion on it read the
    column itself — so making it a copy of `delta_norm` passed all 22 tests,
    and the near-zero control then picked exactly the structurally-dead sites
    the module promises it excludes.

    Two independent checks. Finite differences, which know nothing about the
    taps, measure the same gradient to a fraction of a percent. And the sites
    whose gradient is zero BY CONSTRUCTION — a sublayer output at an earlier
    position in the last layer cannot reach the final prediction — report
    exactly 0.0 while their delta is large, which no copy of `delta_norm`
    could do.
    """
    model, tok, blocks = _fixture(curved=True)
    out = patch_screen.screen(
        model,
        tok,
        blocks,
        CLEAN,
        CORRUPT,
        device="cpu",
        shortlist=36,
        verify=2,
        near_zero_probes=0,
    )
    rows = {r["name"]: r for r in out["shortlist"]}
    assert len(rows) == out["n_sites_scored"] == 36

    for name in ("resid L0@1", "attn L2@3", "mlp L1@1"):
        measured = _finite_difference_gradient(model, tok, blocks, out, name)
        assert rows[name]["grad_norm"] == pytest.approx(measured, rel=0.02), (
            f"{name}: the payload says grad_norm={rows[name]['grad_norm']}, "
            f"finite differences measure {measured}"
        )
        assert rows[name]["grad_norm"] != pytest.approx(
            rows[name]["delta_norm"], rel=0.02
        )

    # The last layer's sublayer outputs at earlier positions: gradient zero by
    # geometry, delta emphatically not.
    last = len(blocks) - 1
    for name in (f"attn L{last}@1", f"attn L{last}@2", f"mlp L{last}@1"):
        assert rows[name]["grad_norm"] == 0.0, (
            f"{name} cannot reach the prediction, so its gradient is exactly "
            f"zero; the payload says {rows[name]['grad_norm']}"
        )
        assert rows[name]["delta_norm"] > 1.0
        assert rows[name]["attribution"] == 0.0

    # And the near-zero control never picks one of those.
    for row in _run(curved=True)["near_zero_probes"]:
        assert row["grad_norm"] > 0 and row["delta_norm"] > 0
        assert _finite_difference_gradient(model, tok, blocks, out, row["name"]) > 0, (
            f"{row['name']} was probed as a live site and its gradient is 0"
        )


def test_the_most_negative_site_is_named_only_when_a_site_scored_negative():
    """Two claims that were made unconditionally and are not unconditionally
    true. The shortlist rule "so a site with a large negative score is not on
    it" is false whenever the shortlist covers the grid — which happens with
    DEFAULT arguments on any model with fewer than 12 sites. And `by_score[-1]`
    is only the most negative site if the minimum is negative; swapping it for
    `by_score[0]`, the most POSITIVE site, passed all 22 tests, because the
    only assertion on the field was that it was not None."""
    # (a) the shortlist covers the grid: nothing was excluded and it says so.
    covering = _run(curved=True, shortlist=36)
    names = {r["name"] for r in covering["shortlist"]}
    assert covering["shortlist_size"] == covering["n_sites_scored"]
    assert covering["strongest_negative"]["name"] in names
    assert covering["strongest_negative_on_shortlist"] is True
    assert "is not on it" not in covering["seeding"]
    assert "excluded nothing" in covering["seeding"]

    # (b) the default cut really does exclude it, and then it says THAT.
    cut = _run(curved=True)
    assert cut["strongest_negative"]["attribution"] < 0
    assert cut["strongest_negative"]["name"] not in {
        r["name"] for r in cut["shortlist"]
    }
    assert cut["strongest_negative_on_shortlist"] is False
    assert "not on it" in cut["seeding"]
    # The most negative, not the most positive: it is the LAST of the ranking.
    assert cut["strongest_negative"]["attribution"] < min(
        r["attribution"] for r in cut["shortlist"]
    )

    # (c) a run where nothing scored below zero: None with a reason, never a
    # positive number wearing a "most negative" label. Seed 6 of the linear
    # fixture is such a run — measured, sweeping seeds 0-59 of both fixtures.
    model, tok, blocks = _fixture(curved=False, seed=6)
    flat = patch_screen.screen(model, tok, blocks, CLEAN, CORRUPT, device="cpu")
    assert min(r["attribution"] for r in flat["shortlist"]) >= 0
    assert flat["strongest_negative"] is None
    assert flat["strongest_negative_on_shortlist"] is None
    assert "scored below zero" in flat["strongest_negative_reason"]
    assert "nothing scored below zero" in flat["seeding"]


class _ResidOnlyBlock(torch.nn.Module):
    """A block with no `attn` and no `mlp`, so only the residual grid exists."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(d, d, bias=False)

    def forward(self, x):
        return x + torch.tanh(self.lin(x))


class _ResidOnly(torch.nn.Module):
    def __init__(self, n_layers: int = 1, d: int = 8) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList(_ResidOnlyBlock(d) for _ in range(n_layers))
        self.embed = torch.nn.Embedding(len(VOCAB), d)
        self.head = torch.nn.Linear(d, len(VOCAB), bias=False)

    def forward(self, ids):
        x = self.embed(ids)
        for b in self.blocks:
            x = b(x)
        return type("Out", (), {"logits": self.head(x)})()


def _counting(model):
    """Wrap `model.forward` so the test can count real forward passes."""
    calls = {"n": 0}
    inner = model.forward

    def counted(ids):
        calls["n"] += 1
        return inner(ids)

    model.forward = counted
    return calls


def test_a_grid_too_small_to_measure_the_screen_on_is_refused():
    """The `verify < MIN_VERIFY` refusal binds on the REQUESTED count, and
    what gets measured is `shortlisted[:verify]`. A model with one site
    therefore published a screen with `verified: 1`, `spearman: None` and
    `worst_rank_move: 0` — an agreement figure from one point — which is
    precisely the unmeasured screen that refusal says this module will not
    publish.

    And it refuses before the tapped pass and the backward, so the cost of
    finding out is one forward pass rather than all of them."""
    torch.manual_seed(0)
    model = _ResidOnly()
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(SCALE)
    calls = _counting(model)

    with pytest.raises(Refusal) as err:
        patch_screen.screen(
            model,
            _Tok(VOCAB),
            list(model.blocks),
            "eiffel",
            "colosseum",
            device="cpu",
        )
    assert str(patch_screen.MIN_VERIFY) in err.value.sentence
    assert "patch.trace" in err.value.sentence
    assert calls["n"] == 0, (
        f"the refusal cost {calls['n']} forward pass(es); the site count is "
        f"knowable from the component list and the tokenization, before any "
        f"pass at all"
    )


def test_an_undefined_correlation_says_which_undefined_it_is():
    """`ablate.spearman` returns None for two different reasons and the
    payload printed the second one for both: a screen with one measured site
    claimed "one of the two rankings is constant", which was not what
    happened. The refusal above means one point can no longer be published at
    all, and the sentence still has to be right about the case it covers."""
    out = _run(curved=True)
    if out["agreement"]["spearman"] is None:
        assert "constant" in out["agreement"]["spearman_reason"]
    # Two verified sites is the floor, and the resolution says so: 2 points
    # can only express rho = +1 or -1.
    two = _run(curved=True, shortlist=4, verify=2, near_zero_probes=1)
    assert two["agreement"]["verified"] == 2
    assert two["agreement"]["spearman_resolution"] == pytest.approx(2.0)
    assert two["agreement"]["worst_rank_move"] is not None


class _Overflowing(_Model):
    """A head that overflows one logit to +inf, the way an fp16 head does."""

    def forward(self, ids):
        out = super().forward(ids)
        out.logits = out.logits.clone()
        out.logits[..., 1] = float("inf")
        return out


def test_a_nonfinite_logit_is_refused_rather_than_scored_as_zero_everywhere():
    """NaN and inf compare False against every bound, so `gap < MIN_GAP` is
    not a guard against either. Left alone this publishes the entire grid as
    0.0 — a finite delta over an infinite gap — with `exact_recovery: 0.0` on
    the verified rows, `p: nan` in the answer, and a payload `json.dumps`
    refuses. A grid of zeros reads as "measured, and nothing here matters",
    which is the strongest possible claim from no data at all."""
    torch.manual_seed(SEED)
    model = _Overflowing(curved=True)
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(SCALE)
    with pytest.raises(Refusal) as err:
        patch_screen.screen(
            model, _Tok(VOCAB), list(model.blocks), CLEAN, CORRUPT, device="cpu"
        )
    assert "NaN" in err.value.sentence
    assert "1 of" in err.value.sentence, "the count of bad logits is not named"

    # And a healthy run is strict-JSON serialisable, which that one was not.
    json.dumps(_run(curved=True), allow_nan=False)


class _InfAtPositionZero(torch.nn.Linear):
    """An mlp whose output is +inf at position 0 only.

    Under the curved block's `tanh` the residual stream stays finite, so the
    logits are finite and every refusal above passes — but three sites have a
    non-finite activation, which is the case the per-site check is for."""

    def forward(self, x):
        y = super().forward(x).clone()
        y[:, 0, :] = float("inf")
        return y


def test_a_site_that_scores_nonfinite_is_a_null_cell_and_is_counted():
    """A NaN sorts wherever the comparison happens to leave it and an inf
    sorts to the top, so a non-finite site left in the ranking takes a place
    on the shortlist on no evidence. It is excluded, the cell is null — the
    same word this grid already uses for "not measured" — and the count and
    the names are in the payload, because an exclusion nobody reports is a
    silent one."""

    class _InfModel(_Model):
        def __init__(self, **kw):
            super().__init__(**kw)
            for block in self.blocks:
                block.mlp = _InfAtPositionZero(8, 8, bias=False)

    torch.manual_seed(SEED)
    model = _InfModel(curved=True)
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(SCALE)
    out = patch_screen.screen(
        model, _Tok(VOCAB), list(model.blocks), CLEAN, CORRUPT, device="cpu"
    )
    assert out["n_sites_nonfinite"] == 3
    assert out["nonfinite_sites"] == ["mlp L0@0", "mlp L1@0", "mlp L2@0"]
    assert out["n_sites_scored"] == 33
    for row in out["screen_grids"]["mlp"]:
        assert row[0] is None, "a non-finite cell was published as a number"
        assert all(v is not None for v in row[1:])
    excluded = set(out["nonfinite_sites"])
    assert all(
        r["name"] not in excluded for r in out["shortlist"] + out["near_zero_probes"]
    )
    assert any("non-finite" in s for s in out["skipped"])
    json.dumps(out, allow_nan=False)


def test_sign_flips_are_counted_and_the_count_is_not_always_zero():
    """Hardcoding `sign_flips` to the flattering 0 passed all 22 tests. It is
    not trivially zero: sweeping seeds 0-59 of the curved fixture, seeds 10
    and 24 produce 3 and 2 verified sites whose exact recovery has the
    opposite sign to the screen's."""
    seen = {}
    for seed in (10, 24):
        model, tok, blocks = _fixture(curved=True, seed=seed)
        out = patch_screen.screen(model, tok, blocks, CLEAN, CORRUPT, device="cpu")
        agree = out["agreement"]
        rows = [r for r in out["shortlist"] if r["exact_recovery"] is not None]
        flipped = [
            r
            for r in rows
            if r["attribution"] * r["exact_recovery"] < 0
            and abs(r["exact_recovery"]) > agree["exact_recovery_resolution"]
        ]
        assert flipped, f"seed {seed} was chosen because it flips signs"
        assert agree["sign_flips"] == len(flipped)
        seen[seed] = agree["sign_flips"]
    assert seen == {10: 3, 24: 2}
    # A linear model cannot flip a sign: the approximation is the answer.
    assert _run(curved=False)["agreement"]["sign_flips"] == 0


def test_the_output_tap_hands_back_the_tuple_it_was_given():
    """`_tap_out` rebuilds the tuple because attention returns the key/value
    cache alongside the hidden states, and dropping the tail changes what the
    rest of the block sees without raising. Dropping the rebuild passed all 22
    tests and the fixture could not see it: `_Block.forward` takes `[0]` of the
    result, and on a bare [1, n_pos, d] tensor that is the batch slice, which
    broadcasts back to identical logits. So this asserts on the tap's return
    value directly, where the tail is either there or it is not."""
    module = _LinearAttention(8, 4)
    values: dict = {}
    taps: dict = {}
    handle = patch_screen._tap_out(module, 0, values, taps)
    try:
        with torch.enable_grad():
            out = module(torch.randn(1, 4, 8))
    finally:
        handle.remove()

    assert isinstance(out, tuple), (
        "the tap returned a bare tensor; a real attention module's key/value "
        "cache would have been dropped here"
    )
    assert len(out) == 2 and out[1] == "cache-sentinel"
    assert out[0].requires_grad, "the tap did not make the output differentiable"
    assert torch.equal(out[0].detach(), values[0])
    assert taps[0].requires_grad
    assert float(taps[0].detach().abs().sum()) == 0.0


def test_the_four_caches_the_screen_holds_are_counted_in_the_payload():
    """The docstring said the extra memory was "a full backward graph for one
    forward pass" against `patch.trace`'s "one activation cache". At the
    backward this holds THREE full caches spanning every component and layer —
    clean, corrupt, and one gradient-carrying zero per site — with the
    gradients a fourth as `autograd.grad` returns. That is ~11x one
    component's cache, not 1x, and on CPU `cost.memory.peak_bytes` is None so
    the payload could not say so.

    The expected byte counts here are closed-form from the model's own shapes
    — components x layers x positions x d x 4 — and not a restatement of how
    the module sums them."""
    for seed, n_layers, d in ((SEED, 3, 8), (0, 4, 16)):
        torch.manual_seed(seed)
        model = _Model(n_layers=n_layers, d=d)
        with torch.no_grad():
            for p in model.parameters():
                p.mul_(SCALE)
        out = patch_screen.screen(
            model, _Tok(VOCAB), list(model.blocks), CLEAN, CORRUPT, device="cpu"
        )
        held = out["cost"]["activation_bytes_held"]
        one_cache = n_layers * out["n_positions"] * d * 4  # float32, 1 component
        assert held["patch_trace_equivalent"] == one_cache
        for key in ("clean_cache", "corrupt_values", "taps"):
            assert held[key] == 3 * one_cache, (
                f"{key} spans all three components at every layer; expected "
                f"{3 * one_cache} bytes and the payload says {held[key]}"
            )
        assert 0 < held["grads"] <= 3 * one_cache
        assert held["total"] == sum(
            held[k] for k in ("clean_cache", "corrupt_values", "taps", "grads")
        )
        assert held["ratio_vs_patch_trace"] >= 9.0, (
            "the screen holds three grid-sized caches plus the gradients "
            "against patch.trace's one component"
        )
        assert "FOUR caches" in held["means"]
        # The flattering half is still unavailable on CPU, and still None.
        assert out["cost"]["memory"]["peak_bytes"] is None


def test_the_baseline_it_prices_against_is_what_patch_trace_really_spends():
    """`exact_grid_passes` restated `patch.trace`'s own `passes` counter,
    which does not include the clean-cache pass it spends per component — so
    the baseline was 3 passes short and the saving was understated. Counted
    here rather than re-derived: this wraps `model.forward` and compares
    against the number of times it is actually called."""
    out = _run(curved=True)

    model, tok, blocks = _fixture(curved=True)
    calls = _counting(model)
    patch.trace(model, tok, blocks, CLEAN, CORRUPT, device="cpu")
    assert out["cost"]["exact_trace_passes"] == calls["n"], (
        f"the screen prices itself against {out['cost']['exact_trace_passes']} "
        f"passes and `patch.trace` really spends {calls['n']}"
    )
    assert str(out["cost"]["exact_grid_passes"]) in out["cost"]["exact_passes_basis"]

    # The screen's own count is honest too, by the same measurement.
    model2, tok2, blocks2 = _fixture(curved=True)
    mine = _counting(model2)
    again = patch_screen.screen(model2, tok2, blocks2, CLEAN, CORRUPT, device="cpu")
    assert again["cost"]["screen_forward_passes"] == mine["n"]


def test_a_count_that_is_not_a_count_is_refused_before_a_pass_is_spent():
    """`verify=2.7` used to raise a raw `TypeError` about slice indices from
    `shortlisted[:verify]` — after two forward passes, a backward and the
    whole grid had been paid for. And bools are ints in Python, so `True`
    silently means 1; the bool check has to come first or it never fires."""
    model, tok, blocks = _fixture(curved=True)
    calls = _counting(model)

    for kwargs, word in (
        ({"verify": 2.7}, "whole number"),
        ({"shortlist": True}, "boolean"),
        ({"verify": True}, "boolean"),
        ({"near_zero_probes": -1}, "below 0"),
        ({"near_zero_probes": 1.5}, "whole number"),
    ):
        with pytest.raises(BadRequest) as err:
            patch_screen.screen(
                model, tok, blocks, CLEAN, CORRUPT, device="cpu", **kwargs
            )
        assert word in err.value.sentence, f"{kwargs}: {err.value.sentence}"
    assert calls["n"] == 0, (
        f"a malformed argument cost {calls['n']} forward pass(es) before it was refused"
    )

    for bad in ((12.5, 11), (12, 11.5), (True, True)):
        with pytest.raises(BadRequest):
            patch_screen.estimate(*bad)


def test_the_near_zero_cap_travels_with_the_probes_like_the_shortlist_cap():
    """A request for a million probes came back silently truncated to however
    many candidates existed, with nothing in the payload saying a cap had been
    applied. The shortlist reports its own cap; so does this."""
    out = _run(curved=True, near_zero_probes=1_000_000)
    assert out["near_zero_requested"] == 1_000_000
    assert out["agreement"]["near_zero_probed"] == len(out["near_zero_probes"])
    assert out["agreement"]["near_zero_probed"] < 1_000_000
    assert out["near_zero_capped_from"] == out["agreement"]["near_zero_probed"]
    assert str(out["near_zero_capped_from"]) in out["seeding"]

    # Under the cap, the request is met exactly and the two agree.
    small = _run(curved=True, near_zero_probes=2)
    assert small["near_zero_requested"] == 2
    assert small["agreement"]["near_zero_probed"] == 2
    assert small["near_zero_capped_from"] > 2


class _Narrow(_Model):
    """The same model with every logit scaled down: the argmaxes are
    unchanged, so the pair still disagrees, and the gap falls under the
    floor."""

    def forward(self, ids):
        out = super().forward(ids)
        out.logits = out.logits * 0.1
        return out


def test_a_gap_too_small_to_divide_by_is_refused():
    """`patch.MIN_GAP` is the floor this module claims to inherit, and
    removing the check entirely passed all 22 tests. Every score here is a
    fraction of the gap, so a pair that disagrees by a third of a logit would
    turn a third of a logit of movement into "100% recovery"."""
    torch.manual_seed(SEED)
    model = _Narrow(curved=True)
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(SCALE)
    with pytest.raises(BadRequest) as err:
        patch_screen.screen(
            model, _Tok(VOCAB), list(model.blocks), CLEAN, CORRUPT, device="cpu"
        )
    assert "too little to divide by" in err.value.sentence
    # The gap it refused is the full-scale one scaled down, and it is named.
    assert "0.3551" in err.value.sentence
