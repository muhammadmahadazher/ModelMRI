# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""A counterfactual is a claim about a change, and a change is easy to fake.

Three ways this module could produce a confident, wrong answer, and each has a
test below rather than a comment:

  1. Report an edit that does not actually reach the target — nothing here
     re-runs the model on `edited_ids` unless a test does.
  2. Clear the controls by construction. The search skips a donor that is
     already the token at a position; a control that did NOT skip them would be
     drawing weaker edits than the one it is controlling, and a fraction of its
     draws would be no-ops. `beats_controls` would then be measuring the
     asymmetry rather than the prompt. That is `test_controls_never_substitute_
     a_token_for_itself`, and it reads the ids of every control pass rather than
     trusting the payload's own summary.
  3. Miscount what it spent. The scan short-circuits on a hit and skips
     self-substitutions, so `passes` and `passes_expected` agree only if both
     are accounted for. An unexplained gap between them is indistinguishable
     from a miscount, so the equality is asserted on found runs, on unfound
     runs, and on a run whose first step hits immediately.

The model is a hand-built decider whose next token is a stated function of the
ids it can see, so every count here is exact rather than approximately right —
"about right" is also what a broken search produces. Everything runs fp32 on
CPU with an all-ones mask.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from modelmri import counterfactual  # noqa: E402
from modelmri.errors import BadRequest  # noqa: E402

# The id that moves the answer. It IS in the pool — a trigger the search cannot
# draw is a search that cannot succeed — and the tests that need controls to
# come back empty use a rule that also requires a specific POSITION, which the
# controls hit only by coincidence.
FLIP = 9
POOL = [1, 2, 3, 4, 5, FLIP, 11, 12]

QUIET = 0  # what the model says when nothing has triggered
MOVED = 2  # what it says once the rule fires

# Index 0 is held as an attention sink and the queried position is its own
# query, so both are deliberately neutral ids: an answer carried by a token the
# search cannot touch is a different finding and gets its own test.
PROMPT = [1, 4, 11, 12, 3, 5, 4]
POSITION = 6
FLIP_AT = 3  # the position `only_at_3` cares about


class Decider:
    """Next-token argmax is a stated function of the ids visible so far.

    Causal by construction: the row at position t is decided by indices <= t
    only. A small position-dependent term rides on top; it never moves the
    argmax (the winner leads by `gap` and the term is bounded well below it)
    but it makes the distribution depend on position_ids, which is what the
    agreement check in `find_counterfactual` needs in order to mean anything.

    Every call is recorded. Most of what this file proves is about the ids a
    pass was handed, not the number it returned.
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

    def __call__(self, input_ids=None, attention_mask=None, position_ids=None, **kw):
        seq = int(input_ids.shape[1])
        self.calls.append({"ids": input_ids.clone()})
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

        rows = []
        for t in range(seq):
            prefix = [int(input_ids[0, i]) for i in range(t + 1) if visible[i]]
            row = torch.zeros(self.vocab)
            winner = self.rule(prefix)
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


def anywhere(prefix):
    """FLIP anywhere in the visible prefix moves the answer.

    A control drawing FLIP at any position reaches the target too, which is the
    honest result: this rule has nothing position-specific to isolate.
    """
    return MOVED if FLIP in prefix else QUIET


def only_at_3(prefix):
    """Only FLIP at index 3 moves it. Any other position is inert."""
    return MOVED if len(prefix) > FLIP_AT and prefix[FLIP_AT] == FLIP else QUIET


def never(prefix):
    """Nothing reachable moves it, so the search must spend its budget and
    say so."""
    return QUIET


def run(rule=only_at_3, *, ids=None, model=None, **kw):
    model = model if model is not None else Decider(rule)
    row = torch.tensor([ids if ids is not None else PROMPT])
    settings = {
        "position": POSITION,
        "target_token_id": MOVED,
        "pool": POOL,
        "n_donors": 4,
        "n_controls": 8,
        "max_edits": 2,
        "candidates": "pool",
        "seed": 0,
    }
    settings.update(kw)
    return model, counterfactual.find_counterfactual(model, row, **settings)


# ------------------------------------------------------- the pass accounting


def test_pass_count_is_exact_on_a_found_run():
    model, out = run()
    assert out["found"] is True
    assert out["passes"] == out["passes_expected"]
    # And the payload's own arithmetic reproduces it, so a reader can check the
    # total rather than take it on faith.
    assert out["passes"] == counterfactual.cost_passes(
        n_positions=len(out["candidate_window"]["perturbable_indices"]),
        steps=out["size"],
        n_donors=4,
        control_passes=(
            out["controls"]["same_positions"]["samples"]
            + out["controls"]["any_positions"]["samples"]
        ),
        trials_not_run=out["trials_skipped_self"] + out["trials_short_circuited"],
    )
    # The model was called exactly as many times as the payload says.
    assert len(model.calls) == out["passes"]


def test_pass_count_is_exact_when_nothing_is_found():
    _, out = run(never)
    assert out["found"] is False
    assert out["passes"] == out["passes_expected"]
    # No edit exists, so no control was drawn and none was priced.
    assert out["controls"]["measured"] is False
    assert out["controls"]["same_positions"]["samples"] == 0
    assert out["controls"]["any_positions"]["samples"] == 0


def test_short_circuited_trials_are_counted_not_lost():
    """A step that hits stops scanning; the pairs it never reached are named."""
    _, out = run()
    assert out["found"] is True
    assert out["trials_short_circuited"] > 0, (
        "a hit on anything but the very last (position, donor) pair leaves "
        "unreached trials, and if they are not counted `passes_expected` "
        "overstates what the run spent"
    )
    assert out["passes"] == out["passes_expected"]


# ------------------------------------------------------------- the controls


def test_controls_never_substitute_a_token_for_itself():
    """The bias this module would otherwise have, checked at the ids.

    If a control were allowed to draw the token already at a position, some of
    its draws would be no-ops and the control would be systematically weaker
    than the edit it is controlling. `beats_controls` would then clear its bar
    by construction. So: every control pass must differ from the base prompt at
    exactly as many indices as the edit it controls.
    """
    model, out = run()
    assert out["found"] is True
    size = out["size"]
    taken = (
        out["controls"]["same_positions"]["samples"]
        + out["controls"]["any_positions"]["samples"]
    )
    assert taken > 0

    base = torch.tensor([PROMPT])
    for call in model.calls[-taken:]:
        differing = int((call["ids"] != base).sum())
        assert differing == size, (
            f"a control pass changed {differing} indices where the edit it "
            f"controls changes {size}. A control that substitutes a token for "
            "itself is not a control."
        )


def test_a_control_that_reaches_the_target_denies_the_finding():
    """`anywhere` can be reached by a random draw, and then it is not a
    finding."""
    _, out = run(anywhere)
    assert out["found"] is True
    reached = (
        out["controls"]["same_positions"]["successes"]
        + out["controls"]["any_positions"]["successes"]
    )
    assert reached > 0, (
        "with FLIP in the pool and any position sufficient, some control draw "
        "must reach the target; if none does, the control is not drawing from "
        "the pool the search draws from"
    )
    assert out["beats_controls"] is False


def test_zero_samples_is_an_absence_not_a_rate_of_zero():
    _, out = run(never)
    for arm in ("same_positions", "any_positions"):
        interval = out["controls"][arm]
        assert interval["measured"] is False
        assert interval["point"] is None, (
            "a control nobody drew must not render as 0.00 — that reads as "
            "'measured, and it never happened'"
        )
        assert interval["interval"] is None


def test_intervals_are_counts_with_bounds_not_bare_rates():
    _, out = run()
    arm = out["controls"]["same_positions"]
    assert arm["measured"] is True
    assert arm["samples"] == 8
    low, high = arm["interval"]
    assert 0.0 <= low <= high <= 1.0
    assert arm["method"] == "Wilson score interval"


# -------------------------------------------------------------- the answer


def test_edited_ids_actually_reach_the_target():
    """The payload's headline claim, re-run rather than believed."""
    model, out = run()
    assert out["found"] is True
    row = torch.tensor([out["edited_ids"]])
    seq = row.shape[1]
    fresh = Decider(only_at_3)
    logits = fresh(
        input_ids=row,
        attention_mask=torch.ones((1, seq), dtype=torch.long),
        position_ids=torch.arange(seq).unsqueeze(0),
    ).logits
    assert int(logits[0, POSITION].argmax()) == MOVED


def test_the_edit_names_where_and_what_it_changed():
    _, out = run()
    assert out["size"] == len(out["edit"]) == 1
    step = out["edit"][0]
    assert step["index"] == FLIP_AT
    assert step["to_token_id"] == FLIP
    assert step["from_token_id"] == PROMPT[FLIP_AT]
    # `edited_ids` and the trail must describe the same edit.
    assert out["edited_ids"][FLIP_AT] == FLIP


def test_greedy_never_claims_minimality():
    _, out = run()
    assert out["minimality"]["smaller_may_exist"] is True


def test_a_step_that_moves_away_from_the_target_is_never_committed():
    """Greedy picks the best on offer; the best on offer can still be worse.

    Measured on Qwen3-1.7B before the guard existed, searching "The Eiffel
    Tower is in the city of" toward its own runner-up: the three committed
    steps took p(target) 0.004418 -> 0.002118 -> 0.000441 and the run then
    reported that edit as its best effort. Every step was real and every step
    went the wrong way.

    Under `never` nothing can raise the target, so the correct run commits
    NOTHING rather than spending its budget going downhill.
    """
    _, out = run(never)
    assert out["found"] is False
    assert out["size"] == 0
    assert out["edit"] == []
    assert "moves away from the target" in out["stopped_because"]
    # No edit was committed, so there is no best effort to report. None rather
    # than a zero-size edit dressed up as an attempt.
    assert out["best_effort"] is None


# ------------------------------------------------------------- the refusals


def test_refuses_a_target_the_model_already_predicts():
    with pytest.raises(BadRequest, match="already predicts"):
        run(only_at_3, target_token_id=QUIET)


def test_refuses_a_target_outside_the_vocabulary():
    with pytest.raises(BadRequest, match="outside this model's vocabulary"):
        run(only_at_3, target_token_id=9999)


def test_refuses_when_nothing_before_the_position_is_editable():
    with pytest.raises(counterfactual.CounterfactualError, match="nothing editable"):
        run(only_at_3, position=1)


def test_refuses_a_model_that_cannot_reproduce_its_own_top_token():
    model = Decider(only_at_3, flaky=True)
    with pytest.raises(
        counterfactual.CounterfactualError, match="two identical forward passes"
    ):
        run(model=model)


def test_refuses_a_batch():
    model = Decider(only_at_3)
    with pytest.raises(BadRequest, match=r"\[1, S\]"):
        counterfactual.find_counterfactual(
            model,
            torch.tensor([PROMPT, PROMPT]),
            position=POSITION,
            target_token_id=MOVED,
            pool=POOL,
        )


def test_rejects_bools_where_counts_are_expected():
    model = Decider(only_at_3)
    with pytest.raises(BadRequest, match="not a boolean"):
        counterfactual.find_counterfactual(
            model,
            torch.tensor([PROMPT]),
            position=True,
            target_token_id=MOVED,
            pool=POOL,
        )


def test_refuses_an_empty_donor_pool():
    model = Decider(only_at_3)
    with pytest.raises(BadRequest, match="nothing to substitute IN"):
        counterfactual.find_counterfactual(
            model,
            torch.tensor([PROMPT]),
            position=POSITION,
            target_token_id=MOVED,
            pool=[],
        )


# ------------------------------------------------------------ the arithmetic


def test_cost_passes_refuses_more_steps_than_positions():
    with pytest.raises(BadRequest, match="each step consumes one position"):
        counterfactual.cost_passes(n_positions=3, steps=4, n_donors=2, control_passes=0)


def test_cost_passes_refuses_skipping_more_trials_than_the_scan_holds():
    with pytest.raises(BadRequest, match="cannot happen"):
        counterfactual.cost_passes(
            n_positions=3, steps=1, n_donors=2, control_passes=0, trials_not_run=99
        )


def test_cost_passes_is_the_arithmetic_it_documents():
    # 3 fixed + (3 + 2) trials * 2 donors + 6 control passes, nothing skipped.
    assert (
        counterfactual.cost_passes(n_positions=3, steps=2, n_donors=2, control_passes=6)
        == 3 + 10 + 6
    )


def test_estimate_cost_prices_both_ends_and_says_which_is_which():
    priced = counterfactual.estimate_cost(n_positions=5, max_edits=2, n_donors=4)
    assert priced["shortest"] < priced["most_expensive"]
    assert "controls" in priced["shortest_is"]
    assert "no control passes" in priced["longest_search_is"]
    # Every figure is a ceiling; a real run skips trials and comes in under it.
    _, out = run()
    live = counterfactual.estimate_cost(
        n_positions=len(out["candidate_window"]["perturbable_indices"]),
        max_edits=2,
        n_donors=4,
        n_controls=8,
    )
    assert out["passes"] <= live["most_expensive"]


# ------------------------------------------- what makes a finding a finding


def _arm(successes, samples):
    return counterfactual._interval(successes, samples, confidence=0.95)


def test_a_finding_needs_both_arms_empty():
    assert counterfactual.is_finding(_arm(0, 8), _arm(0, 8)) is True
    assert counterfactual.is_finding(_arm(1, 8), _arm(0, 8)) is False
    assert counterfactual.is_finding(_arm(0, 8), _arm(3, 8)) is False


def test_an_unmeasured_arm_is_never_evidence_for_the_finding():
    """Zero out of zero is an absence, and an absence must not confirm.

    An arm whose every draw was abandoned carries successes 0 and samples 0. A
    rule written as "no control reached the target" reads that as the strongest
    possible support for the finding, which is the `?? 0` bug pointed at a
    conclusion rather than at a number.
    """
    assert counterfactual.is_finding(_arm(0, 0), _arm(0, 8)) is False
    assert counterfactual.is_finding(_arm(0, 8), _arm(0, 0)) is False
    assert counterfactual.is_finding(_arm(0, 0), _arm(0, 0)) is False


# ------------------------------------------------------ the gradient screen


class Tiny(torch.nn.Module):
    """Embedding -> causal running mean -> linear. Small, real, differentiable.

    `Decider` above cannot be used here: it has no embedding table, so a
    substitution gradient has no continuous space to live in — which is itself
    a case worth testing, and `test_gradient_mode_refuses_a_model_with_no_
    embeddings` tests it. This one is a genuine torch module, so the gradient
    the screen takes is the model's own rather than a stand-in for one.
    """

    def __init__(self, vocab=16, dim=8):
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(vocab, dim)
        self.out = torch.nn.Linear(dim, vocab, bias=False)
        self.eval()

    def get_input_embeddings(self):
        return self.emb

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, **kw):
        x = self.emb(input_ids) if inputs_embeds is None else inputs_embeds
        counts = torch.arange(
            1, x.shape[1] + 1, device=x.device, dtype=x.dtype
        ).reshape(1, -1, 1)
        return type("Out", (), {"logits": self.out(x.cumsum(1) / counts)})()


def _tiny_target(model, row, position):
    """A token the model does NOT currently predict, chosen by looking."""
    with torch.no_grad():
        probs = model(input_ids=row).logits[0, position]
    return int(probs.argsort(descending=True)[1])


def test_propose_by_gradient_ranks_and_never_proposes_a_no_op():
    model = Tiny()
    row = torch.tensor([PROMPT])
    target = _tiny_target(model, row, POSITION)
    proposals = counterfactual.propose_by_gradient(
        model,
        row,
        position=POSITION,
        target_token_id=target,
        indices=[1, 2, 3, 4, 5],
        k=10,
    )
    assert len(proposals) == 10
    estimates = [p["estimate"] for p in proposals]
    assert estimates == sorted(estimates, reverse=True), "best estimate first"
    for p in proposals:
        assert p["token_id"] != int(row[0, p["index"]]), (
            "a token cannot be replaced by itself; the estimate for doing so "
            "is exactly zero and would sit in the middle of the ranking"
        )


def test_gradient_mode_pass_count_is_exact():
    model = Tiny()
    row = torch.tensor([PROMPT])
    target = _tiny_target(model, row, POSITION)
    out = counterfactual.find_counterfactual(
        model,
        row,
        position=POSITION,
        target_token_id=target,
        pool=POOL,
        candidates="gradient",
        n_proposals=6,
        n_controls=5,
        max_edits=2,
    )
    assert out["passes"] == out["passes_expected"]
    assert out["candidates"] == "gradient"
    assert out["screen"]["backward_passes"] >= 1
    assert out["screen"]["proposals_per_step"] == 6


def test_the_screen_publishes_how_often_its_first_choice_won():
    model = Tiny()
    row = torch.tensor([PROMPT])
    target = _tiny_target(model, row, POSITION)
    out = counterfactual.find_counterfactual(
        model,
        row,
        position=POSITION,
        target_token_id=target,
        pool=POOL,
        candidates="gradient",
        n_proposals=6,
        n_controls=5,
        max_edits=2,
    )
    won = out["screen"]["top_choice_won"]
    assert won["measured"] is True
    assert 0 <= won["successes"] <= won["samples"]
    assert won["samples"] == len(out["edit"])


def test_gradient_mode_refuses_a_model_with_no_embeddings():
    with pytest.raises(counterfactual.CounterfactualError, match="embedding table"):
        counterfactual.find_counterfactual(
            Decider(only_at_3),
            torch.tensor([PROMPT]),
            position=POSITION,
            target_token_id=MOVED,
            pool=POOL,
            candidates="gradient",
        )


def test_an_unknown_candidate_source_is_refused_by_name():
    with pytest.raises(BadRequest, match="candidates must be one of"):
        counterfactual.find_counterfactual(
            Tiny(),
            torch.tensor([PROMPT]),
            position=POSITION,
            target_token_id=MOVED,
            pool=POOL,
            candidates="vibes",
        )


def test_decode_is_called_with_one_id_not_a_list():
    """The contract `anchors.py` uses, pinned.

    Every caller in `runtime.py` passes `lambda t: tokenizer.decode([t])` — the
    wrapping is already done. This module briefly passed `[id]` into that, so
    every token in the payload came back as `["The"]`: a one-element LIST where
    a string belonged. Nothing in the unit suite noticed, because the tests
    asserted on `*_token_id` and left `decode` unset. The live route noticed
    immediately, which is the argument for this test existing.
    """
    seen = []

    def decode(token_id):
        seen.append(token_id)
        return f"<{token_id}>"

    _, out = run(decode=decode)
    assert seen, "decode was never called"
    for token_id in seen:
        assert isinstance(token_id, int), (
            f"decode got {token_id!r}; the contract is one id, not a sequence"
        )
    assert isinstance(out["base_token"], str)
    assert out["base_token"] == f"<{out['base_token_id']}>"
    for step in out["edit"]:
        assert isinstance(step["from_token"], str)
        assert isinstance(step["to_token"], str)


def test_without_a_decoder_a_token_is_its_id_not_a_null():
    _, out = run()
    assert out["base_token"] == str(out["base_token_id"])
    assert isinstance(out["target_token"], str)
