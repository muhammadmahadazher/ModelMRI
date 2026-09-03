# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""Token counts that stay nullable, and prices that never guess.

The roadmap's caveat for this feature is the specification: "Token counts the
provider did not return read 'not reported by provider', never 0", and "exact-
string model matching and no regex".

The second is the one worth testing hardest. Langfuse matches model names by
regex with priority-ordered tiers, and a regex matching the wrong model
produces a plausible dollar figure with no signal it is wrong. Every near-miss
below must return None.
"""

from __future__ import annotations

import json

import pytest

from modelmri import ledger
from modelmri.errors import BadRequest


def _step(sid, kind="llm_call", parent=None, model="m", **counts):
    out = {
        "id": sid,
        "parent_id": parent,
        "kind": kind,
        "meta": {"model": model} if model else {},
    }
    out.update(counts)
    return out


# ------------------------------------------------- absence is not a zero


def test_a_field_nobody_reported_totals_to_none_not_zero():
    """The `.get(name, 0.0)` bug class, in the shape it takes here."""
    roll = ledger.roll_up([_step("a", tokens_in=10), _step("b", tokens_in=5)])
    assert roll.counts["tokens_in"].total == 15
    assert roll.counts["tokens_cache_read"].total is None
    assert "not reported by provider" in roll.counts["tokens_cache_read"].means()


def test_a_partial_report_says_how_many_stayed_silent():
    """Summing 3 of 11 and printing it as the total is the same lie one step
    up from storing 0."""
    steps = [_step("a", tokens_cache_read=100), _step("b"), _step("c")]
    roll = ledger.roll_up(steps)
    count = roll.counts["tokens_cache_read"]
    assert count.total == 100
    assert count.reported == 1 and count.silent == 2
    assert "2 reported nothing and are not counted as zero" in count.means()


def test_a_reported_zero_is_kept_as_a_zero():
    """The other half: a provider that genuinely says 0 must not be folded
    into 'said nothing'."""
    roll = ledger.roll_up([_step("a", tokens_cache_read=0)])
    count = roll.counts["tokens_cache_read"]
    assert count.total == 0
    assert count.reported == 1 and count.silent == 0
    assert "not reported" not in count.means()


def test_a_tool_call_is_not_counted_as_a_silent_provider():
    """40 tool calls beside 2 LLM calls would otherwise read as 40 providers
    that reported nothing."""
    steps = [_step("a", tokens_in=5)] + [
        _step(f"t{i}", kind="tool_call", model=None) for i in range(40)
    ]
    roll = ledger.roll_up(steps)
    assert roll.n_steps == 41 and roll.n_llm_steps == 1
    assert roll.counts["tokens_in"].silent == 0


def test_a_non_integer_token_field_is_not_laundered_into_a_number():
    """A string in a token field is a recorder bug; coercing it would hide
    that behind a plausible total."""
    roll = ledger.roll_up([_step("a", tokens_in="1200"), _step("b", tokens_in=3)])
    assert roll.counts["tokens_in"].total == 3
    assert roll.counts["tokens_in"].silent == 1


def test_a_boolean_is_not_an_integer_here():
    """`isinstance(True, int)` is True in Python, so this needs saying."""
    roll = ledger.roll_up([_step("a", tokens_in=True)])
    assert roll.counts["tokens_in"].total is None


# ------------------------------------------------------------- subtrees


def test_a_subtree_rollup_covers_the_step_and_everything_beneath_it():
    steps = [
        _step("root", kind="chain", model=None),
        _step("a", parent="root", tokens_in=10),
        _step("b", parent="root", tokens_in=5),
        _step("b1", parent="b", tokens_in=2),
    ]
    rolls = ledger.subtree_rollups(steps)
    assert rolls["root"].counts["tokens_in"].total == 17
    assert rolls["b"].counts["tokens_in"].total == 7
    assert rolls["b1"].counts["tokens_in"].total == 2


def test_a_parent_cycle_is_broken_rather_than_recursed_into():
    """A hand-written trace document can contain one."""
    steps = [
        _step("a", parent="b", tokens_in=1),
        _step("b", parent="a", tokens_in=1),
    ]
    rolls = ledger.subtree_rollups(steps)
    assert set(rolls) == {"a", "b"}
    for roll in rolls.values():
        assert roll.counts["tokens_in"].total in (1, 2)


def test_a_deep_trace_does_not_blow_the_stack():
    steps = [_step("s0", tokens_in=1)]
    for i in range(1, 4000):
        steps.append(_step(f"s{i}", parent=f"s{i - 1}", tokens_in=1))
    rolls = ledger.subtree_rollups(steps)
    assert rolls["s0"].counts["tokens_in"].total == 4000


# ---------------------------------------------------- exact prices only


@pytest.fixture
def prices(tmp_path):
    path = tmp_path / "prices.json"
    path.write_text(
        json.dumps(
            {"claude-sonnet-4-5-20250929": {"tokens_in": 3.0, "tokens_out": 15.0}}
        ),
        encoding="utf-8",
    )
    return ledger.load_prices(path)


@pytest.mark.parametrize(
    "near_miss",
    [
        "claude-sonnet-4-5",  # a prefix
        "claude-sonnet-4-5-20250929-v2",  # a suffix
        "CLAUDE-SONNET-4-5-20250929",  # different case
        "claude-sonnet-4.5-20250929",  # a dot for a dash
        " claude-sonnet-4-5-20250929",  # leading space
        "claude-sonnet-4-5-.*",  # a regex that WOULD match
        "",
    ],
)
def test_a_near_miss_is_not_a_match(prices, near_miss):
    """This is the whole feature. A regex matching the wrong model produces a
    plausible dollar figure with no signal it is wrong."""
    assert ledger.price_for(near_miss, prices) is None


def test_the_exact_id_does_match(prices):
    price = ledger.price_for("claude-sonnet-4-5-20250929", prices)
    assert price is not None and price.per_million["tokens_in"] == 3.0


def test_the_cost_is_per_million_tokens(prices):
    price = ledger.price_for("claude-sonnet-4-5-20250929", prices)
    cost, unrated = price.cost({"tokens_in": 1_000_000, "tokens_out": 1_000_000})
    assert cost == pytest.approx(18.0)
    assert unrated == []


def test_a_field_the_file_does_not_rate_is_named_not_assumed(prices):
    """Cache pricing is asymmetric per provider and getting the write-vs-read
    multiplier wrong is a silent error."""
    price = ledger.price_for("claude-sonnet-4-5-20250929", prices)
    cost, unrated = price.cost({"tokens_in": 1_000_000, "tokens_cache_read": 500})
    assert cost == pytest.approx(3.0)
    assert unrated == ["tokens_cache_read"]


def test_a_call_with_nothing_priceable_costs_none_not_zero(prices):
    price = ledger.price_for("claude-sonnet-4-5-20250929", prices)
    cost, _ = price.cost({"tokens_cache_read": 500})
    assert cost is None, "a zero here reads as a free call"


# ------------------------------------------------------- the price file


def test_no_price_file_is_not_an_error(monkeypatch):
    """No cost column is the default and the honest one."""
    monkeypatch.delenv(ledger.ENV_VAR, raising=False)
    assert ledger.load_prices() == {}


def test_a_file_that_does_not_exist_is_an_error(tmp_path):
    with pytest.raises(BadRequest, match="not a file"):
        ledger.load_prices(tmp_path / "nope.json")


def test_unparseable_json_is_an_error_not_a_silent_fallback(tmp_path):
    """Falling back to 'no prices' would hide a typo in the very file whose
    purpose is being exact."""
    path = tmp_path / "p.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(BadRequest, match="could not be read as JSON"):
        ledger.load_prices(path)


def test_a_rate_this_tool_cannot_apply_is_refused(tmp_path):
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"m": {"tokens_thinking": 1.0}}), encoding="utf-8")
    with pytest.raises(BadRequest, match="which is not one of"):
        ledger.load_prices(path)


def test_a_negative_rate_is_refused(tmp_path):
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"m": {"tokens_in": -1.0}}), encoding="utf-8")
    with pytest.raises(BadRequest, match="negative"):
        ledger.load_prices(path)


# ------------------------------------------------------------ the bill


def test_a_run_with_no_price_file_says_how_to_get_one():
    out = ledger.bill([_step("a", tokens_in=10)], {})
    assert out.total is None
    assert out.n_calls == 1
    assert ledger.ENV_VAR in out.means()


def test_a_partial_run_reads_as_partial(prices):
    steps = [
        _step("a", model="claude-sonnet-4-5-20250929", tokens_in=1_000_000),
        _step("b", model="some-local-model", tokens_in=500),
        _step("c", model="another-one", tokens_in=500),
    ]
    out = ledger.bill(steps, prices)
    assert out.partial
    assert out.n_calls == 3 and out.n_priced == 1
    means = out.means()
    assert "PARTIAL — 2 of 3 call(s) unpriced" in means
    assert "floor and not the total" in means


def test_a_fully_priced_run_does_not_say_partial(prices):
    steps = [_step("a", model="claude-sonnet-4-5-20250929", tokens_in=1_000_000)]
    out = ledger.bill(steps, prices)
    assert not out.partial
    assert out.total == pytest.approx(3.0)
    assert "PARTIAL" not in out.means()


def test_the_model_comes_from_meta_not_the_step_name(prices):
    """A name is a label somebody chose and two models can share one."""
    step = _step("a", model="claude-sonnet-4-5-20250929", tokens_in=1_000_000)
    step["name"] = "some-other-model"
    out = ledger.bill([step], prices)
    assert out.n_priced == 1


def test_the_token_kind_is_one_a_step_can_actually_have():
    """A kind name that no step has matches nothing, so the rollup reports
    zero LLM calls — indistinguishable from a run that made none. The first
    draft of `ledger.py` said "llm" and would have done that on every trace."""
    from modelmri.step_kinds import VALID_KINDS

    assert set(ledger.TOKEN_KINDS) <= VALID_KINDS
    assert "llm_call" in ledger.TOKEN_KINDS


def test_a_subagent_is_not_counted_beside_its_own_llm_children():
    """Counting both would double every nested run."""
    steps = [
        _step("sub", kind="subagent", model=None),
        _step("a", parent="sub", tokens_in=10),
    ]
    assert ledger.roll_up(steps).counts["tokens_in"].total == 10
    assert ledger.subtree_rollups(steps)["sub"].counts["tokens_in"].total == 10


# ------------------------- one rule for what counts as a count


@pytest.mark.parametrize("bad", [1500.7, True, "n/a", "1500", [1500], None])
def test_a_value_the_rollup_will_not_count_is_not_billed_either(bad):
    """`Count.add` refuses a bool, float or string as a recorder bug rather
    than coercing it; `Price.cost` multiplied whatever was in the field.

    So a step recording `tokens_in: 1500.7` had its tokens reported as "not
    reported by provider" in one column and billed at 0.0045 in the next, off
    that same value. Two answers, same module, same number.
    """
    price = ledger.Price(model="m", per_million={"tokens_in": 3.0})
    count = ledger.Count(field="tokens_in")
    count.add(bad)

    cost, _unrated = price.cost({"tokens_in": bad})
    assert count.reported == 0, "the rollup counts this as a report"
    assert cost is None, "billed a value the rollup refuses to count"


def test_a_string_token_count_does_not_take_down_the_trace_view():
    """Nothing validates these on the way in — `import_trace` hands them to
    SQLite, whose INTEGER affinity leaves a non-numeric string as TEXT — so a
    str comes back out of `get_trace`. `"n/a" * 3.0` is a TypeError, the route
    catches BadRequest only, and the whole trace disappeared behind a 500 over
    a cost column documented as optional."""
    price = ledger.Price(model="m", per_million={"tokens_in": 3.0})
    assert price.cost({"tokens_in": "n/a"}) == (None, [])


def test_a_real_count_is_still_billed():
    price = ledger.Price(model="m", per_million={"tokens_in": 3.0})
    cost, unrated = price.cost({"tokens_in": 1500})
    assert cost == pytest.approx(0.0045)
    assert unrated == []


def test_one_usable_field_beside_one_unusable_still_prices():
    """The unusable field is skipped, not fatal — and it is skipped the same
    way an absent one is, which is what the token column already reports."""
    price = ledger.Price(model="m", per_million={"tokens_in": 3.0, "tokens_out": 15.0})
    cost, _ = price.cost({"tokens_in": 1000, "tokens_out": "unknown"})
    assert cost == pytest.approx(0.003)
