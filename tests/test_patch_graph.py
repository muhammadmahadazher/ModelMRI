"""A patching graph, and the four things it must never quietly do.

The graph is a subset by construction — edge count is quadratic in sites — so
every one of these is about what the picture says about its own limits:

  * it is a PATCHING graph and says so, never an attribution graph,
  * the seeding rule and the prune threshold travel with it,
  * every drawn edge was controlled, and one that FAILED its control is drawn
    and marked rather than dropped — two rules that are easy to confuse,
  * where the walk stopped is reported, because "nothing wrote this" and "we
    did not ask" are different findings.

`build` takes a `trace_fn` rather than a model, so all of that is testable
without loading anything.
"""

from __future__ import annotations

import pytest

from modelmri import patch_graph as pg


def _sender(layer, head, recovery, *, controlled=True, clears=True):
    row = {
        "layer": layer,
        "head": head,
        "name": f"L{layer}H{head}" if head is not None else f"L{layer} MLP",
        "recovery": recovery,
    }
    if controlled:
        row.update(
            {
                "control_max": recovery * (0.5 if clears else 2.0),
                "control_draws": 8,
                "clears_control": clears,
                "clears_position": clears,
            }
        )
    return row


def _tracer(by_receiver, *, resolution=0.01):
    """A stand-in `path_trace`: {(layer, position): [sender, ...]}."""
    calls = []

    def trace_fn(layer, position):
        calls.append((layer, position))
        return {
            "receiver": {"layer": layer, "position": position},
            "senders": list(by_receiver.get((layer, position), [])),
            "recovery_resolution": resolution,
            "passes": 10,
            "seconds": 0.5,
        }

    trace_fn.calls = calls
    return trace_fn


SITES = [{"layer": 8, "position": 3, "recovery": 0.9, "clears_control": True}]


# ------------------------------------------------------------ what it builds


def test_it_walks_backwards_from_the_seed():
    """One level finds the seed's senders; the second asks the same question
    of the senders that beat their controls."""
    trace_fn = _tracer(
        {
            (8, 3): [_sender(6, 2, 0.7), _sender(5, None, 0.4)],
            (6, 3): [_sender(2, 1, 0.6)],
            (5, 3): [_sender(1, 0, 0.5)],
        }
    )
    graph = pg.build(trace_fn, SITES, depth=2, max_receivers=4)

    ids = {n.id for n in graph.nodes}
    assert "L8 MLP@3" in ids, "the seed is a node"
    assert "L6H2@3" in ids and "L5 MLP@3" in ids, "its senders are nodes"
    assert graph.n_receivers_expanded == 3
    assert any(e.source == "L6H2@3" and e.target == "L8 MLP@3" for e in graph.edges)


def test_the_depth_bounds_the_walk():
    trace_fn = _tracer(
        {(8, 3): [_sender(6, 2, 0.7)], (6, 3): [_sender(2, 1, 0.6)]},
    )
    shallow = pg.build(trace_fn, SITES, depth=1)
    assert shallow.n_receivers_expanded == 1
    assert shallow.depth == 1


def test_a_seed_at_layer_zero_is_not_expanded():
    """`path_trace` refuses layer 0 — nothing earlier can have written into
    it — so asking would spend a refusal per seed."""
    trace_fn = _tracer({})
    graph = pg.build(
        trace_fn,
        [{"layer": 0, "position": 1, "recovery": 0.5, "clears_control": True}],
        depth=2,
    )
    assert trace_fn.calls == []
    assert graph.n_receivers_expanded == 0


# ------------------------------------------- what it says about its own limits


def test_it_is_a_patching_graph_and_never_an_attribution_graph():
    """circuit-tracer's attribution graphs are built from transcoders. This is
    a different object from a different measurement, and borrowing the more
    famous name for it would be the claim this project exists not to make."""
    trace_fn = _tracer({(8, 3): [_sender(6, 2, 0.7)]})
    said = pg.build(trace_fn, SITES).means()

    assert "PATCHING graph" in said
    assert "not a transcoder attribution graph" in said
    assert "attribution graph," not in said.replace(
        "not a transcoder attribution graph,", ""
    )


def test_the_seeding_rule_travels_with_the_graph():
    """A graph whose edges were chosen by an undisclosed rule is a picture,
    not a measurement."""
    trace_fn = _tracer({(8, 3): [_sender(6, 2, 0.7), _sender(5, None, 0.001)]})
    graph = pg.build(trace_fn, SITES, depth=1)
    rule = graph.seeding()

    assert "quadratic" in rule
    assert "strongest edges LOOKED AT, not the strongest edges there are" in rule
    assert str(graph.n_scored) in rule
    assert str(graph.n_pruned) in rule


def test_the_prune_threshold_comes_from_the_dtype_not_from_here():
    """Two senders closer than the dtype can express are tied, so the cut is
    `path_trace`'s own resolution rather than a constant invented here."""
    trace_fn = _tracer({(8, 3): [_sender(6, 2, 0.7)]}, resolution=0.25)
    graph = pg.build(trace_fn, SITES, depth=1)

    assert graph.prune_threshold == 0.25
    assert "resolution" in graph.prune_from


def test_a_sender_below_the_resolution_is_pruned_and_counted():
    trace_fn = _tracer(
        {(8, 3): [_sender(6, 2, 0.7), _sender(4, 1, 0.005)]}, resolution=0.01
    )
    graph = pg.build(trace_fn, SITES, depth=1)

    assert graph.n_scored == 2
    assert graph.n_pruned == 1
    assert len(graph.edges) == 1


# ------------------------------------------- an edge that failed is still drawn


def test_an_edge_that_lost_to_its_control_is_kept_and_marked():
    """Dropping it silently turns "we tested this and it did not survive" into
    "we never saw this", and those are different findings."""
    trace_fn = _tracer(
        {(8, 3): [_sender(6, 2, 0.7, clears=True), _sender(4, 1, 0.5, clears=False)]}
    )
    graph = pg.build(trace_fn, SITES, depth=1)

    assert len(graph.edges) == 2, "the losing edge was dropped"
    weak = graph.weak
    assert len(weak) == 1
    assert weak[0].source == "L4H1@3"
    assert weak[0].clears_control is False
    assert weak[0].tested is True
    assert "drawn differently rather than dropped" in graph.means()


def test_an_uncontrolled_sender_is_not_drawn_at_all():
    """Controls run on the strongest few per receiver. An edge without them has
    a score and no verdict — nothing behind it to click — so it is pruned and
    counted rather than drawn as though it had passed."""
    trace_fn = _tracer(
        {(8, 3): [_sender(6, 2, 0.7), _sender(4, 1, 0.65, controlled=False)]}
    )
    graph = pg.build(trace_fn, SITES, depth=1)

    assert [e.source for e in graph.edges] == ["L6H2@3"]
    assert graph.n_scored == 2 and graph.n_pruned == 1
    assert graph.untested == [], "an edge was drawn with no verdict behind it"
    assert "NOT drawn" in graph.means()


def test_the_prune_does_not_read_as_a_verdict_against_what_it_dropped():
    """ "We did not test this" is not "this does nothing"."""
    trace_fn = _tracer({(8, 3): [_sender(6, 2, 0.7, controlled=False)]})
    graph = pg.build(trace_fn, SITES, depth=1)

    assert graph.edges == []
    assert "Absence from the picture is not a verdict" in graph.means()


def test_only_an_edge_that_beat_its_control_is_expanded():
    """A losing edge is drawn, but spending a whole path_trace on it would
    walk the graph into noise."""
    trace_fn = _tracer(
        {
            (8, 3): [_sender(6, 2, 0.7, clears=True), _sender(4, 1, 0.5, clears=False)],
            (6, 3): [_sender(1, 0, 0.3)],
            (4, 3): [_sender(1, 1, 0.3)],
        }
    )
    pg.build(trace_fn, SITES, depth=2)

    assert (6, 3) in trace_fn.calls
    assert (4, 3) not in trace_fn.calls, "expanded an edge its control beat"


# ------------------------------------------------------- where the walk stopped


def test_the_frontier_names_where_the_walk_ran_out():
    """ "Nothing wrote this" and "we did not ask" are different findings."""
    trace_fn = _tracer(
        {(8, 3): [_sender(6, 2, 0.7)], (6, 3): [_sender(2, 1, 0.6)]},
    )
    graph = pg.build(trace_fn, SITES, depth=1)

    assert graph.frontier, "the walk stopped and said nothing about it"
    assert "The walk stopped at" in graph.means()


def test_a_receiver_with_no_senders_is_reported_not_silently_empty():
    trace_fn = _tracer({(8, 3): []})
    graph = pg.build(trace_fn, SITES, depth=1)

    assert graph.edges == []
    assert "L8 MLP@3" in graph.frontier


# -------------------------------------------------------------- the refusals


def test_a_walk_with_no_seeds_is_refused():
    with pytest.raises(pg.GraphError, match="no sites"):
        pg.build(_tracer({}), [], depth=1)


def test_a_zero_depth_walk_is_refused():
    with pytest.raises(pg.GraphError, match="zero levels"):
        pg.build(_tracer({}), SITES, depth=0)


def test_the_edge_cap_is_a_prune_not_a_silent_truncation():
    many = [_sender(5, h, 0.5) for h in range(20)]
    trace_fn = _tracer({(8, 3): many})
    graph = pg.build(trace_fn, SITES, depth=1, max_edges=5)

    assert len(graph.edges) == 5
    assert graph.n_pruned == 15, "the dropped edges were not counted"
    assert graph.n_scored == 20


# ------------------------------------------------------------- the projection


def test_the_cost_is_projected_before_anybody_waits():
    out = pg.estimate(12, 12, depth=2, max_receivers=4)

    assert out["passes_total"] > 0
    assert out["receivers"] == 5
    assert out["seconds"] is None, "a duration measured elsewhere is fiction here"
    assert "not" in out["seconds_from"]


def test_an_unreadable_config_is_refused_rather_than_projected_at_zero():
    """The same rule `sweep.plan` and `model_diff.head_pass_estimate` hold: a
    preflight that under-quotes is worse than no preflight."""
    with pytest.raises(pg.GraphError, match="cannot be projected"):
        pg.estimate(0, 0)
