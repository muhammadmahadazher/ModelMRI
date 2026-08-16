"""A `.mri` carries the PATCHING graph, and what it refuses to carry.

The graph costs thousands of forward passes to build — on Qwen3-1.7B at depth
2 it walked 35 nodes and 48 edges out of 153 scored senders — and the recipient
of a `.mri` has activations rather than weights, so there is nothing on their
machine to rebuild it with. A section
that travels is the difference between a finding somebody can show a colleague
and one that dies on the laptop that took it.

It is a SEPARATE key from `graph`. That one carries a transcoder attribution
graph this tool did not compute and is gated on provenance saying so; this one
carries a graph it did. Two different objects from two different measurements,
and one key for both would make the disclaimer on each unreadable.

Three refusals here are about the claim rather than the bytes, and each one is
a rule the builder already enforces — restated at the reader because `parse`
runs on bytes a stranger sent, and a hand-written file is not obliged to have
come from `patch_graph.build`:

  * an edge with no verdict against its controls,
  * a graph with no seeding sentence,
  * an edge naming a node the file does not carry.
"""

from __future__ import annotations

import gzip
import json

import pytest

from modelmri import session


def _edge(source, target, recovery=0.5, **over):
    edge = {
        "source": source,
        "target": target,
        "recovery": recovery,
        "control_max": recovery / 2,
        "control_draws": 8,
        "clears_control": True,
        "clears_position": True,
    }
    edge.update(over)
    return edge


GRAPH = {
    "nodes": [
        {"id": "L11 MLP@9", "layer": 11, "head": None, "position": 9, "role": "seed"},
        {"id": "L10 MLP@9", "layer": 10, "head": None, "position": 9, "depth": 1},
        {"id": "L9H6@9", "layer": 9, "head": 6, "position": 9, "depth": 2},
    ],
    "edges": [
        _edge("L10 MLP@9", "L11 MLP@9", 0.25),
        _edge("L9H6@9", "L10 MLP@9", 0.25),
    ],
    "seeding": "Seeded from the 4 strongest sites the node grid flagged and "
    "walked back 2 level(s). 153 senders in total, 105 pruned.",
    "means": "A PATCHING graph, not a transcoder attribution graph.",
    "clean": "The Eiffel Tower is in the city of",
    "corrupt": "The Colosseum is in the city of",
    "depth": 2,
    "n_scored": 153,
    "n_pruned": 105,
    # Measured on Qwen3-1.7B at depth 2 over the reference pair. `passes` is
    # deliberately absent: it was not captured from that run, and a plausible
    # number written into a fixture reads exactly like a measured one.
    "prune_threshold": 0.006231,
    "prune_from": "the dtype's own recovery resolution",
    "frontier": ["L9H6@9"],
}


def make(patch_graph=None, **over) -> bytes:
    kw = dict(
        model_id="Qwen/Qwen3-1.7B",
        device="cuda",
        dtype="bfloat16",
        n_params=1_720_574_976,
        tokens=["a", "b"],
        prompt="a",
        generation="b",
        attention={(0, 0): [[1.0, 0.0], [0.5, 0.5]]},
        n_layers=2,
        n_heads=2,
    )
    kw.update(over)
    return session.build(patch_graph=patch_graph, **kw)


def repack(raw: bytes, graph) -> bytes:
    """A file whose patching-graph section says exactly what we want it to.

    `build` runs the reader's validator, so a file that breaks one of these
    rules cannot be written through it — which is the point. This is how a
    hand-made one gets in.
    """
    doc = json.loads(gzip.decompress(raw))
    doc["patch_graph"] = graph
    return gzip.compress(json.dumps(doc).encode("utf-8"))


# --------------------------------------------------------------- carrying it


def test_a_graph_survives_the_round_trip():
    got = session.parse(make(GRAPH))

    assert got.has_patch_graph()
    assert [n["id"] for n in got.patch_graph["nodes"]] == [
        "L11 MLP@9",
        "L10 MLP@9",
        "L9H6@9",
    ]
    assert got.patch_graph["edges"][0]["source"] == "L10 MLP@9"
    assert got.patch_graph["edges"][0]["control_draws"] == 8
    assert got.patch_graph["n_scored"] == 153
    assert got.patch_graph["n_pruned"] == 105
    assert got.patch_graph["prune_threshold"] == pytest.approx(0.006231)


def test_the_seeding_rule_travels_with_the_file():
    """Edge count is quadratic in sites, so every such graph is a subset. One
    whose rule for choosing edges was stripped is a picture, not a
    measurement — and it must not survive a forward."""
    got = session.parse(make(GRAPH))
    assert "105 pruned" in got.patch_graph["seeding"]


def test_the_frontier_travels_because_stopping_is_not_a_finding():
    got = session.parse(make(GRAPH))
    assert got.patch_graph["frontier"] == ["L9H6@9"]


def test_it_is_a_separate_section_from_somebody_elses_attribution_graph():
    """`graph` is a circuit-tracer file this tool did not compute and `parse`
    demands provenance saying so. Merging the two would put ours under that
    banner and theirs under none."""
    doc = json.loads(gzip.decompress(make(GRAPH)))

    assert "patch_graph" in doc
    assert "graph" not in doc
    assert session.parse(make(GRAPH)).has_graph() is False


def test_a_session_without_a_graph_does_not_claim_one():
    raw = make()
    assert "patch_graph" not in json.loads(gzip.decompress(raw))
    assert session.parse(raw).has_patch_graph() is False


def test_an_empty_edge_list_is_not_written():
    raw = make({**GRAPH, "edges": []})
    assert "patch_graph" not in json.loads(gzip.decompress(raw))


def test_older_files_still_open():
    """Additive, which is why the format version does not move."""
    doc = json.loads(gzip.decompress(make(GRAPH)))
    doc.pop("patch_graph")
    reopened = session.parse(gzip.compress(json.dumps(doc).encode("utf-8")))

    assert reopened.has_patch_graph() is False
    assert reopened.tokens == ["a", "b"]


# ------------------------------------------------- what it refuses to carry


def test_an_edge_with_no_verdict_is_refused():
    """The section's whole guarantee is that every drawn edge was run against
    the eight same-norm draws behind it. One carrying a score and no verdict
    would render as though it had passed."""
    broken = {**GRAPH, "edges": [_edge("L10 MLP@9", "L11 MLP@9", clears_control=None)]}
    with pytest.raises(session.SessionError, match="no verdict against its controls"):
        session.parse(repack(make(GRAPH), broken))


def test_an_edge_claiming_a_verdict_with_no_control_is_refused():
    """A verdict without its reference is the bare claim this section exists
    to replace — the same rule `_ground` applies to a passage."""
    broken = {**GRAPH, "edges": [_edge("L10 MLP@9", "L11 MLP@9", control_max=None)]}
    with pytest.raises(session.SessionError, match="no control behind it"):
        session.parse(repack(make(GRAPH), broken))


def test_an_edge_with_no_draw_count_is_refused():
    broken = {**GRAPH, "edges": [_edge("L10 MLP@9", "L11 MLP@9", control_draws=0)]}
    with pytest.raises(session.SessionError, match="how many control draws"):
        session.parse(repack(make(GRAPH), broken))


def test_a_graph_with_no_seeding_rule_is_refused():
    with pytest.raises(session.SessionError, match="picture rather than a measurement"):
        session.parse(repack(make(GRAPH), {**GRAPH, "seeding": "  "}))


def test_an_edge_between_nodes_the_file_does_not_carry_is_refused():
    """It reaches the viewer as a lookup, and a dangling one draws as an edge
    from nowhere."""
    broken = {**GRAPH, "edges": [_edge("L4H2@9", "L11 MLP@9")]}
    with pytest.raises(session.SessionError, match="edge between nodes it does not"):
        session.parse(repack(make(GRAPH), broken))


def test_a_non_finite_recovery_is_refused_rather_than_drawn_as_nothing():
    """NaN and infinity survive JSON round-trips through most writers and
    colour-scale to nothing visible."""
    broken = {
        **GRAPH,
        "edges": [_edge("L10 MLP@9", "L11 MLP@9", recovery=float("nan"))],
    }
    raw = json.loads(gzip.decompress(make(GRAPH)))
    raw["patch_graph"] = broken
    packed = gzip.compress(json.dumps(raw).encode("utf-8"))
    with pytest.raises(session.SessionError, match="no finite recovery"):
        session.parse(packed)


def test_a_node_that_does_not_say_where_it_is_is_refused():
    broken = {**GRAPH, "nodes": [{"id": "L11 MLP@9", "head": None}]}
    with pytest.raises(session.SessionError, match="which layer and position"):
        session.parse(repack(make(GRAPH), broken))


def test_a_boolean_layer_is_not_read_as_layer_one():
    """`isinstance(True, int)` is True in Python, so a bare int check lets
    `{"layer": true}` through and then indexes as layer 1."""
    broken = {
        **GRAPH,
        "nodes": [{"id": "L11 MLP@9", "layer": True, "position": 9, "head": None}],
    }
    with pytest.raises(session.SessionError, match="which layer and position"):
        session.parse(repack(make(GRAPH), broken))


@pytest.mark.parametrize("key", ["nodes", "edges"])
def test_a_graph_far_past_what_anybody_measured_is_refused(key):
    """Every edge costs eight control passes to earn, so a file claiming tens
    of thousands is claiming a run nobody sat through."""
    broken = {**GRAPH, key: [{"id": f"n{i}"} for i in range(9_000)]}
    with pytest.raises(session.SessionError, match="not one anybody measured"):
        session.parse(repack(make(GRAPH), broken))


def test_the_writer_is_not_laxer_than_the_reader():
    """A writer that accepts what `parse` refuses builds files this tool signs
    its name to and then cannot open."""
    with pytest.raises(session.SessionError, match="no verdict against its controls"):
        make({**GRAPH, "edges": [_edge("L10 MLP@9", "L11 MLP@9", clears_control=None)]})


def test_an_unrun_position_control_stays_unknown():
    """The shifted-position control is a separate pass. Coercing None to False
    would turn "not run" into "run, and failed"."""
    graph = {
        **GRAPH,
        "edges": [_edge("L10 MLP@9", "L11 MLP@9", clears_position=None)],
    }
    got = session.parse(make(graph))
    assert got.patch_graph["edges"][0]["clears_position"] is None
    assert got.patch_graph["edges"][0]["clears_control"] is True


# --------------------------------------------- the eight draws behind an edge


def test_the_individual_draws_travel_so_the_spread_is_readable():
    """ROADMAP #52 asks for the eight controls to be clickable behind an edge.
    A verdict quoted as "beat 0.28" reads differently once you can see that
    seven of the eight were nowhere near it."""
    draws = [0.0, 0.02, 0.0, 0.125, 0.0, 0.05, 0.0, 0.0]
    graph = {
        **GRAPH,
        "edges": [
            _edge("L10 MLP@9", "L11 MLP@9", 0.25, control_max=0.125, controls=draws)
        ],
    }
    got = session.parse(make(graph))
    assert got.patch_graph["edges"][0]["controls"] == draws


def test_draws_that_disagree_with_the_verdicts_own_number_are_refused():
    """Two numbers answering "what did noise recover here" differently is a
    defect even when each is individually plausible — and the one the reader
    clicks would be the one that is wrong."""
    graph = {
        **GRAPH,
        "edges": [
            _edge("L10 MLP@9", "L11 MLP@9", 0.5, control_max=0.1, controls=[0.1, 0.4])
        ],
    }
    with pytest.raises(session.SessionError, match="rests on one of those"):
        session.parse(repack(make(GRAPH), graph))


def test_an_edge_with_no_draw_list_still_carries_its_verdict():
    """Optional, unlike the verdict: `control_max` is what the verdict rests
    on and is complete without them."""
    got = session.parse(make(GRAPH))
    assert got.patch_graph["edges"][0]["controls"] == []
    assert got.patch_graph["edges"][0]["clears_control"] is True
