"""An anchor is a sufficiency claim, and sufficiency is the easy thing to fake.

The failure this file exists to prevent is a panel that highlights three words
and says "these are enough", built on a search that never checked, a precision
with no sample size behind it, or a greedy result presented as the minimal set.
Each of those produces a confident, ordered, wrong answer about the reader's own
text — the same shape of wrongness `attribute.rank_tokens` is guarded against,
pointed at the opposite question.

The model below is a hand-built decider whose next token is a stated function of
which ids it can see, so every precision here is 0 of n or n of n EXACTLY. That
is the point of a toy: a real checkpoint could only ever produce a number that
looks about right, and "about right" is what a broken sufficiency search also
produces. The arithmetic-only numbers (Wilson bounds, pass counts) are checked
against values worked out by hand and quoted in the assertions.

Everything here runs fp32 on CPU with an all-ones mask.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from modelmri import anchors, attribute  # noqa: E402
from modelmri.errors import BadRequest  # noqa: E402

# The ids the rules below react to. Kept out of the donor pool so that a
# perturbation can never reinstate one by luck — that is what makes every
# precision in this file exactly 0 or exactly n rather than approximately so.
TRIGGER_A, TRIGGER_B = 7, 9
POOL = [1, 2, 3, 4, 5, 11, 12, 13]


# ------------------------------------------------- a model that can be checked


class Decider:
    """Next-token argmax is a stated function of the ids visible so far.

    Causal by construction: the row at position t is decided by the ids at
    indices <= t only, so "tokens after the position cannot matter" is a
    property of the model rather than something this file has to remember not
    to test.

    On top of the decision sits a small position-dependent term. It never moves
    the argmax — the winner leads by `gap` = 10 logits and the term is bounded
    by about 0.5 — but it makes the DISTRIBUTION depend on position_ids, which
    is what `attribute.rank_tokens` requires before it will produce a ranking at
    all: without it that function refuses (correctly) on the grounds that it
    cannot tell a position-blind model from one that derives its own positions
    from the attention mask. Both questions therefore run against one model,
    which is the only way the necessity/sufficiency comparison below means
    anything.

    Every call is recorded. Most of what this file proves is about the arguments
    a pass was given and the ids it was handed, not the number it returned.
    """

    def __init__(self, rule, *, vocab=16, gap=10.0, amplitude=0.05, flaky=False):
        self.rule = rule
        self.vocab = vocab
        self.gap = gap
        self.amplitude = amplitude
        self.flaky = flaky
        torch.manual_seed(0)
        self.phase_weights = torch.randn(64, vocab)
        self.calls: list[dict] = []

    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        output_attentions=False,
        **kw,
    ):
        seq = int(input_ids.shape[1])
        self.calls.append(
            {
                "ids": input_ids.clone(),
                "mask": None if attention_mask is None else attention_mask,
                "position_ids": position_ids,
            }
        )
        visible = (
            torch.ones(seq)
            if attention_mask is None
            else attention_mask[0].to(torch.float32)
        )
        pos = (
            visible.cumsum(0) - 1.0
            if position_ids is None
            else position_ids[0].to(torch.float32)
        )
        phase = torch.sin(0.9 * pos + 0.3)

        rows, seen = [], set()
        for t in range(seq):
            if visible[t]:
                seen.add(int(input_ids[0, t]))
            row = torch.zeros(self.vocab)
            winner = self.rule(frozenset(seen))
            if self.flaky and len(self.calls) > 1:
                # Two identical passes, two different answers. Nothing here can
                # hold, and the run must say so rather than count coin flips.
                winner = (winner + 1) % self.vocab
            row[winner] = self.gap
            extra = (
                self.phase_weights[: t + 1]
                * (visible[: t + 1] * phase[: t + 1]).unsqueeze(-1)
            ).sum(0)
            rows.append(row + self.amplitude * extra)
        return type("Out", (), {"logits": torch.stack(rows).unsqueeze(0)})()


def or_rule(seen):
    """Either trigger alone decides it. Neither is necessary; both are
    sufficient."""
    return 1 if (TRIGGER_A in seen or TRIGGER_B in seen) else 0


def and_rule(seen):
    """Both are necessary and neither is sufficient — the mirror case."""
    return 1 if (TRIGGER_A in seen and TRIGGER_B in seen) else 0


def leaky_or_rule(seen):
    """Like `or_rule`, but id 3 is IN the donor pool, so a perturbation can
    reinstate the answer by luck and precision becomes a genuine sample."""
    return 1 if (TRIGGER_A in seen or 3 in seen) else 0


# ids[0] and ids[position] are held fixed by rule, so they are deliberately
# neutral: an answer carried by a token the search cannot touch is a different
# finding, and it gets its own test rather than contaminating these.
PROMPT = [2, 4, TRIGGER_A, 5, TRIGGER_B, 12, 4]
AT_A, AT_B = PROMPT.index(TRIGGER_A), PROMPT.index(TRIGGER_B)


def run(rule=or_rule, *, ids=None, model=None, position=None, **kw):
    model = model or Decider(rule)
    row = torch.tensor([ids or PROMPT])
    settings = {
        "pool": POOL,
        "n_samples": 24,
        "n_search": 6,
        "target": 0.8,
        "max_candidates": 5,
        "seed": 0,
    }
    settings.update(kw)
    out = anchors.find_anchor(
        model,
        row,
        position=int(row.shape[1] - 1) if position is None else position,
        **settings,
    )
    return model, out


# ------------------------------------------------- the distinction itself


def test_a_token_can_be_sufficient_without_being_necessary():
    """The whole reason this module exists beside `attribute.rank_tokens`.

    On an OR of two triggers, masking either one changes nothing — the other
    still carries the answer — so a NECESSITY ranking puts both at the floor
    and a panel built on it reports that neither word mattered. Each of them
    ALONE holds the prediction through every perturbation of everything else.
    Same model, same prompt, same position, opposite readings.
    """
    ids = torch.tensor([PROMPT])
    necessity = attribute.rank_tokens(Decider(or_rule), ids, position=len(PROMPT) - 1)
    by_index = {r["index"]: r["kl"] for r in necessity["ranked"]}
    # Measured unrounded: 4.755e-07 for the trigger at index 2 and 8.052e-07 at
    # index 4, against a noise floor of 0.0 — both print as 0.00000 in the five
    # decimals `rank_tokens` publishes. Worse than flat: index 1 holds a token
    # nothing depends on and scores 4.164e-06, 8.8x the trigger, because below
    # the floor the list is ordering the position term.
    assert by_index[AT_A] == 0.0 and by_index[AT_B] == 0.0
    assert necessity["noise_floor_kl"] == 0.0

    _, out = run(or_rule)
    assert out["found"] is True
    assert out["size"] == 1 and out["anchor_indices"][0] in (AT_A, AT_B)
    assert out["precision"]["held"] == out["precision"]["samples"] == 24
    # And the receipt that this is not a fluke of the tie-break: both triggers
    # screened at 1.0 on the first step.
    first = {r["index"]: r["point"] for r in out["steps"][0]["screened"]}
    assert first[AT_A] == 1.0 and first[AT_B] == 1.0


def test_a_token_can_be_necessary_without_being_sufficient():
    """The mirror. On an AND, masking either trigger destroys the answer, so
    necessity ranks both at the top — and neither one alone holds anything."""
    ids = torch.tensor([PROMPT])
    necessity = attribute.rank_tokens(Decider(and_rule), ids, position=len(PROMPT) - 1)
    by_index = {r["index"]: r["kl"] for r in necessity["ranked"]}
    # Measured: 10.00451 and 9.92551 nats. A real, large, correct necessity
    # signal, on the same tokens that anchor at 0 of 24 each.
    assert by_index[AT_A] > 1.0 and by_index[AT_B] > 1.0

    _, out = run(and_rule)
    assert sorted(out["anchor_indices"]) == sorted([AT_A, AT_B])
    assert out["size"] == 2 and out["found"] is True
    assert out["precision"]["held"] == 24
    # Every single-token anchor screened at exactly zero. A search that stopped
    # when a step bought no improvement would have returned nothing here.
    assert all(r["point"] == 0.0 for r in out["steps"][0]["screened"])
    assert all(r["point"] == 0.0 for r in out["steps"][1]["screened"])


# ------------------------------------------------- precision is a sample


def test_eight_of_ten_and_eight_hundred_of_a_thousand_do_not_print_the_same():
    """Both are "80%" and they are not the same evidence. A point estimate
    alone would publish one number for a nineteen-fold difference in width."""
    small = anchors.wilson_interval(8, 10)
    large = anchors.wilson_interval(800, 1000)
    assert round(small[0], 3) == 0.490 and round(small[1], 3) == 0.943
    assert round(large[0], 3) == 0.774 and round(large[1], 3) == 0.824
    assert (small[1] - small[0]) > 8 * (large[1] - large[0])


def test_a_perfect_run_is_not_certainty():
    """The reason Wilson rather than the normal approximation: at k == n the
    normal interval has width zero, which is a claim of certainty out of a
    finite sample."""
    low, high = anchors.wilson_interval(64, 64)
    assert high == 1.0
    assert round(low, 4) == 0.9434
    # And that bound is exactly n / (n + z^2), which is what makes the
    # sample-size guard below arithmetic rather than a rule of thumb.
    assert round(anchors.reachable_target(64), 4) == 0.9434


def test_a_target_no_sample_of_this_size_could_reach_is_refused():
    """Not a warning. With 64 draws and a 0.95 bar every search runs to
    max_size and returns `found: false` at a precision of 1.000 — a payload
    that reads "this model has no anchors" when it means "these draws cannot
    carry that bound"."""
    with pytest.raises(BadRequest, match="cannot be met with 64 draws"):
        run(or_rule, n_samples=64, n_search=6, target=0.95)
    with pytest.raises(BadRequest, match="at least 73"):
        run(or_rule, n_samples=64, n_search=6, target=0.95)


def test_every_precision_carries_its_sample_size_and_interval():
    _, out = run(or_rule)
    for block in (out["precision"], out["base_rate"], out["steps"][0]["verified"]):
        assert block["samples"] > 0
        assert block["method"] == anchors.INTERVAL_METHOD
        assert 0.0 <= block["low"] <= block["point"] <= block["high"] <= 1.0
        assert block["confidence"] == 0.95
    assert out["target"] == 0.8
    assert out["target_ceiling"] == round(anchors.reachable_target(24), 4)


# ------------------------------------------------- the two ends


def test_the_base_rate_is_measured_as_the_floor():
    """55% precision means nothing without what the prediction does when
    nothing is held. Here the pool contains no trigger, so it is 0 of 24 —
    and 0 of 24 is an interval, not a point at zero."""
    _, out = run(or_rule)
    assert out["base_rate"]["held"] == 0 and out["base_rate"]["samples"] == 24
    assert out["base_rate"]["point"] == 0.0
    assert out["base_rate"]["high"] > 0.0


def test_the_ceiling_is_measured_when_the_cap_leaves_something_outside():
    """An anchor cannot beat the whole candidate set. When the trigger sits
    outside the search window the ceiling is 0 of 24, no anchor could ever
    clear the target, and the run says so instead of paying for a greedy sweep
    that is guaranteed to fail."""
    ids = [2, TRIGGER_A, 4, 5, 12, 11, 13, 4, 2, 5]  # the trigger is index 1
    _, out = run(or_rule, ids=ids, max_candidates=3, position=len(ids) - 1)
    assert out["ceiling"]["measured"] is True
    assert out["ceiling"]["held"] == 0 and out["ceiling"]["n_perturbed"] > 0
    assert out["stopped_because"] == "ceiling-below-target"
    assert out["found"] is False and out["anchor_indices"] == []
    # Priced exactly: three fixed passes, the base rate and the ceiling. No
    # screening was paid for at all — measured 51 passes against the 243 the
    # same shape of run costs when the search proceeds.
    assert out["passes"] == anchors.cost_passes(
        n_candidates=3, steps=0, n_samples=24, n_search=6, ceiling_measured=True
    )


def test_a_ceiling_that_was_never_sampled_is_not_printed_as_a_number():
    """When the candidate set covers everything perturbable, holding all of it
    perturbs nothing — there is no sample. `point` stays None and the 1.0 goes
    in `implied`, where a reader and a chart can tell it from evidence."""
    _, out = run(or_rule, max_candidates=99)
    assert out["ceiling"]["measured"] is False
    assert out["ceiling"]["point"] is None
    assert out["ceiling"]["implied"] == 1.0
    assert out["ceiling"]["n_perturbed"] == 0
    assert "no sample to take" in out["ceiling"]["reason"]


def test_a_prediction_held_outside_the_search_returns_an_empty_anchor():
    """The token at `position` is held fixed by rule, so a model that reads
    only it survives every perturbation. That is a real finding — none of your
    words is holding this — and it comes back as an anchor of size 0 rather
    than as a failure or as a token picked to have something to show."""
    _, out = run(lambda seen: 1 if 12 in seen else 0, ids=[2, 4, 5, 11, 12])
    assert out["base_rate"]["held"] == 24
    assert out["stopped_because"] == "empty-anchor-sufficient"
    assert out["found"] is True and out["anchor_indices"] == []
    # Nothing was dropped, so irreducibility is UNKNOWN. Not False.
    assert out["minimality"]["irreducible_under_single_removal"] is None
    assert out["passes"] == anchors.cost_passes(
        n_candidates=3, steps=0, n_samples=24, n_search=6, ceiling_measured=False
    )


# ------------------------------------------------- minimality, honestly


def test_a_greedy_result_is_never_called_the_minimal_set():
    _, out = run(or_rule)
    assert out["minimality"]["search"] == (
        "greedy forward selection, then backward elimination"
    )
    assert out["minimality"]["smaller_may_exist"] is True
    assert "not the minimal set" in out["minimality"]["note"]


def test_backward_elimination_undoes_what_the_tie_break_cost():
    """Forward selection overshoots here and it is worth seeing why.

    On the AND both triggers screen at 0.0 on the first step, so the tie-break
    picks by recency and lands on a token carrying nothing; greedy then needs
    THREE tokens to clear a bar that two of them clear alone. Reporting that
    third token as part of the anchor would be reporting the tie-break. The
    elimination sweep removes it and the removal is in the payload rather than
    only its conclusion.
    """
    _, out = run(and_rule)
    minimality = out["minimality"]
    assert len(out["steps"]) == 3  # forward selection took three
    assert out["size"] == 2  # elimination gave one back
    assert minimality["removed_by_elimination"] == [5]
    removed = [d for d in minimality["drops"] if d["removed"]]
    assert len(removed) == 1 and removed[0]["held"] == 24
    assert removed[0]["anchor_at_the_time"] == [2, 4, 5]


def test_the_elimination_sweep_says_only_what_it_measured():
    """The terminating sweep tested every element against the FINAL anchor, so
    the claim is exactly "no SUBSET of this is sufficient" — not "no other set
    of this size is"."""
    _, out = run(and_rule)
    minimality = out["minimality"]
    assert minimality["drop_one_checked"] is True
    assert minimality["irreducible_under_single_removal"] is True
    final_sweep = [d for d in minimality["drops"] if not d["removed"]]
    assert {d["dropped_index"] for d in final_sweep} == {AT_A, AT_B}
    assert all(d["held"] == 0 for d in final_sweep)
    assert all(d["anchor_at_the_time"] == [AT_A, AT_B] for d in final_sweep)
    assert "no SUBSET" in minimality["note"]


def test_dropping_the_only_element_reuses_the_base_rate_rather_than_re_paying():
    """It leaves the empty anchor, which was measured before the search
    started. Spending another n_samples passes to reproduce a number already in
    the payload is the kind of cost nobody notices."""
    _, out = run(or_rule)
    drop = out["minimality"]["drops"][0]
    assert drop["held"] == out["base_rate"]["held"] == 0
    assert drop["removed"] is False
    assert "already measured" in drop["reused"]
    assert out["passes"] == anchors.cost_passes(
        n_candidates=5,
        steps=1,
        n_samples=24,
        n_search=6,
        ceiling_measured=False,
        prune_size=0,
    )


def test_prune_off_leaves_irreducibility_unknown_not_false():
    _, out = run(and_rule, prune=False)
    assert out["minimality"]["irreducible_under_single_removal"] is None
    assert out["minimality"]["drop_one_checked"] is False


# ------------------------------------------------- what gets perturbed


def test_control_tokens_are_never_perturbed_and_never_anchored():
    """Replacing a chat template's own tokens with a word from a corpus about
    coffee measures the template coming apart. Every anchor would look weak and
    the base rate would sit at zero on every templated prompt."""
    ids = [2, 4, TRIGGER_A, 99, TRIGGER_B, 12, 4]  # 99 stands in for <|im_start|>
    model, out = run(or_rule, ids=ids, control_ids=[99])
    assert 3 in out["held_fixed"]["indices"]
    assert 3 not in out["anchor_indices"]
    for call in model.calls:
        assert int(call["ids"][0, 3]) == 99


def test_index_zero_the_position_and_everything_after_it_are_held():
    _, out = run(or_rule, position=4)
    fixed = set(out["held_fixed"]["indices"])
    assert {0, 4, 5, 6} <= fixed
    assert "attention sink" in out["held_fixed"]["why"]


def test_only_the_perturbable_positions_ever_change():
    """Every pass in the run, not just the ones behind the final number: the
    screening passes hold different anchors and must still respect the same
    held-fixed set."""
    model, _ = run(or_rule)
    original = torch.tensor([PROMPT])
    changed_anywhere = set()
    for call in model.calls:
        changed = (call["ids"] != original).nonzero()[:, 1].tolist()
        assert all(1 <= i < len(PROMPT) - 1 for i in changed)
        changed_anywhere |= set(changed)
    assert changed_anywhere == {1, 2, 3, 4, 5}


def test_replacements_only_ever_come_from_the_pool():
    """The perturbation distribution is the claim. A replacement from anywhere
    else would make the reported corpus a fiction."""
    model, _ = run(or_rule)
    original = torch.tensor([PROMPT])
    for call in model.calls:
        for index in (call["ids"] != original).nonzero()[:, 1].tolist():
            assert int(call["ids"][0, index]) in POOL


# ------------------------------------------------- the forward passes


def test_every_pass_is_handed_the_same_mask_and_the_same_position_ids():
    """attribute.py's discipline, kept where it is not strictly required.
    Substitution cannot re-phase RoPE the way masking can — the mask stays
    all-ones and the length never changes — but "cannot arise" is a claim about
    today's code and the failure it would cause is silent."""
    model, _ = run(or_rule)
    supplied = [c for c in model.calls if c["position_ids"] is not None]
    assert len(supplied) == len(model.calls) - 1  # the one plain model(ids)
    first = supplied[0]
    assert all(c["position_ids"] is first["position_ids"] for c in supplied)
    assert all(c["mask"] is first["mask"] for c in supplied)
    assert torch.equal(first["position_ids"], torch.arange(len(PROMPT)).unsqueeze(0))
    assert int(first["mask"].sum()) == len(PROMPT)


def test_substitution_never_changes_the_length_of_the_sequence():
    """The alternative — deleting the token — renumbers every position after
    it, and the measurement becomes about the renumbering."""
    model, _ = run(or_rule)
    assert {tuple(c["ids"].shape) for c in model.calls} == {(1, len(PROMPT))}


def test_the_pass_count_matches_the_cost_function_exactly():
    """The count is the portable half of the cost. If the loop and the
    projection disagree, `estimate_cost` is pricing a run nobody makes."""
    _, out = run(and_rule)
    paid_for = [d for d in out["minimality"]["drops"] if not d["reused"]]
    assert out["passes"] == anchors.cost_passes(
        n_candidates=5,
        steps=len(out["steps"]),
        n_samples=24,
        n_search=6,
        ceiling_measured=False,
        prune_size=len(paid_for),
    )
    assert (len(out["steps"]), len(paid_for)) == (3, 3)


def test_estimate_cost_brackets_the_run_it_prices():
    """Cost before spending it. Greedy stops when the bar is met, so the honest
    projection is a range with both ends named rather than an average that
    describes neither."""
    model = Decider(and_rule)
    ids = torch.tensor([PROMPT])
    priced = anchors.estimate_cost(
        model,
        ids,
        position=len(PROMPT) - 1,
        pool=POOL,
        max_candidates=5,
        max_size=4,
        n_samples=24,
        n_search=6,
    )
    _, out = run(and_rule)
    assert (
        priced["passes"]["minimum"] <= out["passes"] <= priced["passes"]["worst_case"]
    )
    assert priced["ceiling_measured"] is False
    assert priced["probe"]["seconds"] > 0
    # Kilobytes, and named. Nothing here materialises a vocabulary-square.
    assert priced["retained_bytes"] < 100_000


# ------------------------------------------------- reproducibility


def test_the_same_seed_reproduces_and_a_different_seed_is_a_different_sample():
    """A precision is a sample. Two runs of the same seed must agree exactly,
    and two seeds must be visibly two samples rather than one number wearing
    two hats — otherwise nothing about the interval is honest."""
    ids = [2, 4, TRIGGER_A, 5, 12, 11, 4]
    _, first = run(leaky_or_rule, ids=ids, seed=0)
    _, again = run(leaky_or_rule, ids=ids, seed=0)
    _, other = run(leaky_or_rule, ids=ids, seed=5)
    assert first["base_rate"] == again["base_rate"]
    assert first["seed"] == 0 and other["seed"] == 5
    # id 3 is in the pool, so the answer comes back by luck at a rate that
    # depends on the draws. Measured: 11 of 24 at seed 0 and 9 of 24 at seed 5,
    # whose intervals overlap heavily — which is what 24 draws buys and exactly
    # why the point estimate never travels alone.
    assert first["base_rate"]["held"] != other["base_rate"]["held"]
    assert 0 < first["base_rate"]["held"] < 24


# ------------------------------------------------- refusals


def test_a_model_whose_own_top_token_is_not_stable_is_refused():
    """Every precision here is an argmax compared against the base pass's
    argmax. If that flips between two identical passes there is nothing to
    hold, and the counts would be counting non-determinism."""
    with pytest.raises(anchors.AnchorError, match="two identical forward passes"):
        run(or_rule, model=Decider(or_rule, flaky=True))


def test_anchoring_a_control_token_is_refused():
    """A chat template all but guarantees a control token at the end of a
    prompt. An anchor for it would be the set of your words that keeps the
    template working."""
    with pytest.raises(anchors.AnchorError, match="control token"):
        run(lambda seen: 5, control_ids=[5])


def test_a_position_with_nothing_ordinary_before_it_is_refused():
    with pytest.raises(anchors.AnchorError, match="attention sink"):
        run(or_rule, position=1)


def test_a_position_outside_the_sequence_is_a_bad_request():
    with pytest.raises(BadRequest, match="outside a sequence of 7 tokens"):
        run(or_rule, position=99)


def test_screening_may_not_read_draws_the_verification_did_not_see():
    with pytest.raises(BadRequest, match="cannot exceed n_samples"):
        run(or_rule, n_samples=8, n_search=16, target=0.5)


def test_bools_are_rejected_before_they_become_numbers():
    """`isinstance(True, int)` is True, so `position=True` would sail through
    every range check and quietly anchor at index 1."""
    with pytest.raises(BadRequest, match="not a boolean"):
        run(or_rule, position=True)
    with pytest.raises(BadRequest, match="not a boolean"):
        run(or_rule, n_samples=True)
    with pytest.raises(BadRequest, match="not booleans"):
        anchors.wilson_interval(True, 10)


def test_an_empty_pool_is_a_bad_request_naming_the_fix():
    with pytest.raises(BadRequest, match="donor_pool"):
        run(or_rule, pool=[])


def test_a_pool_the_model_cannot_embed_is_refused_before_the_sweep():
    class Config:
        vocab_size = 16

    model = Decider(or_rule)
    model.config = Config()
    with pytest.raises(BadRequest, match="vocabulary is 16 wide"):
        run(or_rule, model=model, pool=[1, 2, 999])


def test_a_batched_sequence_is_this_packages_own_bug_not_the_callers():
    """runtime.py builds `ids` itself, so a batch here is ModelMRI
    contradicting itself — a 500 with a traceback rather than a sentence a
    reader cannot act on."""
    with pytest.raises(RuntimeError, match=r"\[1, S\]"):
        anchors.find_anchor(
            Decider(or_rule), torch.tensor([PROMPT, PROMPT]), position=3, pool=POOL
        )


# ------------------------------------------------- the caps and the corpus


def test_the_cap_is_reported_with_the_true_count_beside_it():
    """ "Not offered to the search" and "tried and rejected" are different
    findings, and a truncated list cannot tell them apart on its own."""
    ids = [2, 4, 5, 11, 12, 13, 1, 2, 3, 4, TRIGGER_A, 5, 4]
    _, out = run(or_rule, ids=ids, max_candidates=3, position=len(ids) - 1)
    counts = out["candidates"]
    assert counts["n_tested"] == 3 and counts["n_candidates"] == 11
    assert counts["truncated"] is True
    assert counts["max_candidates"] == 3
    assert "never offered to the search" in counts["coverage"]
    assert counts["tested_span"] == [9, 12]


def test_the_held_fixed_list_is_cut_and_the_true_count_travels():
    """Everything after `position` is held — a long generation puts dozens of
    indices in this list, and a list silently cut to 32 would read as the whole
    of what the measurement left alone."""
    long_prompt = [2, 4, TRIGGER_A, 5, 11, 12] + [4] * 40
    _, out = run(or_rule, ids=long_prompt, position=5)
    held = out["held_fixed"]
    assert held["count"] > anchors.HELD_FIXED_LISTED
    assert held["listed"] == anchors.HELD_FIXED_LISTED
    assert len(held["indices"]) == anchors.HELD_FIXED_LISTED


def test_the_perturbation_distribution_travels_with_the_precision():
    """ "The prediction survived perturbation" is not a statement until you say
    perturbed with WHAT. corpus.py already argues this for resampled heads."""
    from modelmri import corpus

    tokenizer = _Tokenizer()
    pool, description = anchors.donor_pool(tokenizer, control_ids=[99])
    assert description["corpus"] == corpus.BUILT_IN_LABEL
    assert description["distinct_ids"] >= anchors.MIN_DISTINCT_IDS
    assert description["draws_in_pool"] == len(pool)
    assert "frequency-weighted" in description["weighting"]

    _, out = run(or_rule, perturbation=description)
    assert out["perturbation"]["corpus"] == corpus.BUILT_IN_LABEL
    assert corpus.BUILT_IN_LABEL in out["perturbation"]["sentence"]
    assert out["perturbation"]["seed"] == 0
    assert out["perturbation"]["samples"] == 24


def test_a_pool_too_small_to_be_a_distribution_is_refused():
    """With a handful of distinct ids every draw has a real chance of putting
    back the very token it was meant to remove, and precision rises for that
    reason rather than because an anchor holds anything."""
    with pytest.raises(anchors.AnchorError, match="distinct ids"):
        anchors.donor_pool(
            _Tokenizer(), sentences=["one two three", "two three one"], label="tiny (2)"
        )


def test_a_pool_may_not_wear_another_corpus_label():
    with pytest.raises(BadRequest, match="both `sentences` and `label`"):
        anchors.donor_pool(_Tokenizer(), sentences=["one two"])


def test_control_tokens_are_counted_out_of_the_pool_rather_than_dropped_silently():
    tokenizer = _Tokenizer()
    ids, description = anchors.donor_pool(
        tokenizer, sentences=list(_MANY), label="fixture", control_ids=[3]
    )
    assert 3 not in ids
    assert description["control_ids_dropped"] > 0


class _Tokenizer:
    """Word -> id by first appearance, with no vocabulary file anywhere.

    A real tokenizer would need a download and this file must run air-gapped.
    Insertion order rather than `hash()`: Python randomises string hashing per
    process unless PYTHONHASHSEED is pinned, so a hash-based id would make the
    pool — and every count taken from it — a different fixture on every run.
    What `donor_pool` is being tested on is which ids it keeps and what it says
    about them, not how the ids were derived.
    """

    def __init__(self):
        self.vocab: dict[str, int] = {}

    def __call__(self, text, return_tensors=None):
        ids = [
            self.vocab.setdefault(word, len(self.vocab) + 1) for word in text.split()
        ]
        return type("Encoded", (), {"input_ids": torch.tensor([ids])})()


_MANY = tuple(f"word{i} three shared tail" for i in range(40))


def test_the_shipped_defaults_are_reachable_by_the_shipped_sample_size():
    """THE INVARIANT THE WHOLE MODULE RESTS ON, and it was unguarded.

    Wilson's lower bound for a run with no failures cancels to exactly
    `n / (n + z**2)`, so at 64 draws and 95% confidence the highest lower bound
    obtainable is 0.9434 — the literature's customary 0.95 target is NOT
    REACHABLE there. A build shipping `PRECISION_TARGET = 0.95` against
    `N_SAMPLES = 64` would not fail: every search would walk to `max_size`,
    return an over-large anchor with `found: false`, and report a precision of
    1.000 beside it. A confident, ordered, wrong answer.

    MEASURED, and the reason this test exists: raising `PRECISION_TARGET` to
    0.95 in the module left all 39 other tests in this file green. The
    `reachable_target` / `_impossible_target` machinery was written for exactly
    this and nothing checked that the SHIPPED PAIR obeyed it.

    The ceilings, computed rather than recalled:

        n =  16   0.8064        n =  64   0.9434
        n =  32   0.8928        n =  73   0.9500   <- the first n that reaches 0.95
    """
    ceiling = anchors.reachable_target(anchors.N_SAMPLES, confidence=anchors.CONFIDENCE)
    assert anchors.PRECISION_TARGET <= ceiling, (
        f"the shipped target {anchors.PRECISION_TARGET} cannot be reached in "
        f"{anchors.N_SAMPLES} draws at {anchors.CONFIDENCE} confidence — the "
        f"highest lower bound available is {ceiling:.4f}. Either lower the "
        f"target or raise N_SAMPLES to at least "
        f"{next(n for n in range(2, 4096) if anchors.reachable_target(n) >= anchors.PRECISION_TARGET)}."
    )
    # And the ceiling is a real function of n, not a constant: a
    # `reachable_target` that always returned 1.0 would satisfy the assertion
    # above while destroying the guard it exists for.
    assert anchors.reachable_target(16) < anchors.reachable_target(64)
    assert anchors.reachable_target(64) < anchors.reachable_target(128)
    assert anchors.reachable_target(73) >= 0.95 > anchors.reachable_target(72)


def test_an_unreachable_target_is_refused_up_front_rather_than_walked_into():
    """The other half of the same guard. Asking for 0.95 at 64 draws must be a
    refusal naming the minimum sample size — not a search that quietly walks to
    `max_size` and returns the largest anchor it is allowed to build, with
    `found: false` and a precision of 1.000 beside it.
    """
    assert 0.95 > anchors.reachable_target(64), (
        "the pair this test is built on is no longer impossible"
    )
    # `BadRequest`, not `AnchorError`: an impossible target/sample pair is the
    # caller asking for something that cannot exist, which is a 422, rather
    # than this module declining to answer a well-formed question.
    with pytest.raises(BadRequest) as caught:
        anchors.find_anchor(
            Decider(or_rule),
            torch.tensor([PROMPT]),
            position=3,
            pool=POOL,
            n_samples=64,
            target=0.95,
        )
    said = caught.value.sentence
    # The refusal has to carry BOTH numbers a reader needs to act: what is
    # actually reachable here, and how many draws would reach what they asked
    # for. One without the other leaves them guessing.
    assert "0.95" in said or "0.9500" in said, said
    assert "73" in said, f"the refusal does not name the sample size that works: {said}"
