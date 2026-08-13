"""Reading an attribution graph somebody else computed.

Two things carry the weight.

**A `.pt` is a pickle, and unpickling runs code.** The reader must open a file
from a stranger without importing anything the file names, so the tests below
build a real pickled graph against a real module, delete that module, and read
the file with it absent — which is the actual condition someone is in when a
colleague sends them a graph.

**A graph is nodes x nodes.** At 10,000 nodes the adjacency matrix is 400 MB
and `.tolist()` on it is several gigabytes of Python floats. So the tests
assert that nothing materialises it: the summary reduces on the tensor and the
edge list is bounded by `topk`, and a large graph is checked to produce a
small payload rather than a large one.

The rest is the provenance banner, which is not decoration. A rendered graph
ModelMRI did not compute must never be mistakable for one it did.
"""

from __future__ import annotations

import json
import sys
import textwrap

import pytest

torch = pytest.importorskip("torch")

from modelmri import circuit  # noqa: E402
from modelmri.errors import BadRequest, Refusal  # noqa: E402


@pytest.fixture
def producer(tmp_path):
    """A real `circuit_tracer` package, importable only while saving.

    The classes have to genuinely exist to be pickled — `torch.save` resolves
    them by name and refuses a class whose module does not import. So the
    module is created, used, and then removed from `sys.path` and
    `sys.modules`, leaving a file that names classes the reader cannot import.
    That is the condition that matters.
    """
    pkg = tmp_path / "pkg" / "circuit_tracer"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(
        textwrap.dedent(
            """
            import dataclasses

            class UnifiedConfig:
                def __init__(self, model_name="google/gemma-2-2b"):
                    self.model_name = model_name
                    self.n_layers = 26

            @dataclasses.dataclass
            class LogitTarget:
                token_id: int
                probability: float
            """
        ),
        encoding="utf-8",
    )
    root = str(tmp_path / "pkg")
    sys.path.insert(0, root)
    import circuit_tracer  # type: ignore[import-not-found]

    yield circuit_tracer

    sys.modules.pop("circuit_tracer", None)
    if root in sys.path:
        sys.path.remove(root)


def _graph_file(tmp_path, producer, *, nodes=32, **over):
    a = torch.zeros(nodes, nodes)
    a[3, 1] = 0.9
    a[5, 2] = -0.7
    doc = {
        "input_string": "The capital of France is",
        "adjacency_matrix": a,
        "cfg": producer.UnifiedConfig(),
        "logit_targets": [producer.LogitTarget(1, 0.5)],
        "input_tokens": torch.tensor([1, 2, 3, 4, 5]),
        "scan_name": "gemma-2-2b-transcoders",
        "vocab_size": 256000,
    }
    doc.update(over)
    path = tmp_path / "graph.pt"
    torch.save(doc, path)
    sys.modules.pop("circuit_tracer", None)
    return path


# ------------------------------------------------- reading a stranger's file


def test_a_graph_reads_without_importing_the_tool_that_made_it(tmp_path, producer):
    """The whole security posture. `weights_only=True` refuses this file
    outright because `cfg` is a UnifiedConfig; `weights_only=False` would run
    whatever the file says. The reader does neither."""
    path = _graph_file(tmp_path, producer)
    sys.modules.pop("circuit_tracer", None)

    g = circuit.read(path)

    assert "circuit_tracer" not in sys.modules, "the reader imported the file's module"
    assert g.n_nodes == 32
    assert g.prompt == "The capital of France is"


def test_the_foreign_classes_are_named_but_not_imported(tmp_path, producer):
    """They are the evidence for `producer`, and the list of things the reader
    declined to import."""
    g = circuit.read(_graph_file(tmp_path, producer))
    assert g.foreign_classes == [
        "circuit_tracer.LogitTarget",
        "circuit_tracer.UnifiedConfig",
    ]
    assert "circuit_tracer" not in sys.modules


def test_the_model_name_is_recovered_from_the_config(tmp_path, producer):
    """Without this the banner cannot name the model, which is the reason the
    reader does not simply refuse every file with a custom class in it."""
    g = circuit.read(_graph_file(tmp_path, producer))
    assert g.model == "google/gemma-2-2b"
    assert g.scan == "gemma-2-2b-transcoders"


def test_a_config_with_a_different_spelling_still_yields_a_model(tmp_path, producer):
    cfg = producer.UnifiedConfig()
    del cfg.model_name
    cfg.name = "meta-llama/Llama-3.2-1B"
    g = circuit.read(_graph_file(tmp_path, producer, cfg=cfg))
    assert g.model == "meta-llama/Llama-3.2-1B"


def test_an_unnamed_model_is_reported_as_unnamed_not_guessed(tmp_path, producer):
    cfg = producer.UnifiedConfig()
    del cfg.model_name
    g = circuit.read(_graph_file(tmp_path, producer, cfg=cfg))
    assert g.model is None
    assert any("does not name the model" in n for n in g.notes)


def test_the_producer_is_read_from_the_classes_the_file_names(tmp_path, producer):
    g = circuit.read(_graph_file(tmp_path, producer))
    assert g.producer == "circuit-tracer"


def test_a_file_naming_no_foreign_classes_is_unattested(tmp_path, producer):
    """The shape can match without anything proving who wrote it, and that
    difference is stated rather than assumed away."""
    path = tmp_path / "bare.pt"
    torch.save(
        {"adjacency_matrix": torch.zeros(4, 4), "input_tokens": torch.zeros(3)}, path
    )
    g = circuit.read(path)
    assert "unknown" in g.producer
    assert any("unattested" in n for n in g.notes)


# ---------------------------------------------------------------- provenance


def test_provenance_always_says_modelmri_did_not_measure_it(tmp_path, producer):
    """Not optional chrome, and not a flag the UI has to remember to read: a
    sentence, inside the payload, on every graph."""
    p = circuit.read(_graph_file(tmp_path, producer)).provenance
    assert "ModelMRI did not run the model" in p["measured_by"]
    assert "cannot vouch" in p["measured_by"]


def test_provenance_is_present_even_when_the_file_names_nothing(tmp_path):
    path = tmp_path / "bare.pt"
    torch.save(
        {"adjacency_matrix": torch.zeros(4, 4), "input_tokens": torch.zeros(3)}, path
    )
    p = circuit.read(path).provenance
    assert p["model"] is None
    assert "ModelMRI did not run the model" in p["measured_by"]


def test_the_file_name_travels_with_the_provenance(tmp_path, producer):
    assert (
        circuit.read(_graph_file(tmp_path, producer)).provenance["file"] == "graph.pt"
    )


# ------------------------------------------------------------------ refusals


def test_a_missing_file_is_a_bad_request(tmp_path):
    with pytest.raises(BadRequest, match="no such file"):
        circuit.read(tmp_path / "nope.pt")


def test_something_that_is_not_a_torch_archive_is_refused(tmp_path):
    path = tmp_path / "not-a-graph.pt"
    path.write_bytes(b"this is not a pickle")
    with pytest.raises(Refusal, match="could not be read as a torch archive"):
        circuit.read(path)


def test_a_torch_file_that_is_not_a_dict_is_refused(tmp_path):
    path = tmp_path / "tensor.pt"
    torch.save(torch.zeros(4), path)
    with pytest.raises(Refusal, match="not the dict"):
        circuit.read(path)


def test_a_dict_without_the_required_keys_names_what_it_holds(tmp_path):
    path = tmp_path / "other.pt"
    torch.save({"weights": torch.zeros(2), "bias": torch.zeros(2)}, path)
    with pytest.raises(Refusal) as err:
        circuit.read(path)
    assert "adjacency_matrix" in str(err.value)
    assert "bias" in str(err.value)  # what it DOES hold


def test_a_ragged_adjacency_matrix_stops_at_the_reader(tmp_path):
    """ "A ragged graph stops at the reader, not in the recipient's browser.\""""
    path = tmp_path / "ragged.pt"
    torch.save(
        {"adjacency_matrix": torch.zeros(4, 7), "input_tokens": torch.zeros(3)}, path
    )
    with pytest.raises(Refusal, match="not square"):
        circuit.read(path)


def test_a_one_dimensional_adjacency_is_refused(tmp_path):
    path = tmp_path / "flat.pt"
    torch.save(
        {"adjacency_matrix": torch.zeros(16), "input_tokens": torch.zeros(3)}, path
    )
    with pytest.raises(Refusal, match="1-dimensional"):
        circuit.read(path)


def test_a_non_tensor_adjacency_is_refused_by_type(tmp_path):
    path = tmp_path / "listy.pt"
    torch.save(
        {"adjacency_matrix": [[0, 1], [1, 0]], "input_tokens": torch.zeros(3)}, path
    )
    with pytest.raises(Refusal, match="not a tensor"):
        circuit.read(path)


def test_an_implausible_node_count_is_refused_before_squaring_it(tmp_path):
    """`nodes * nodes` on a corrupt header is how a reader hangs instead of
    refusing. The bound is checked against the shape, not against a product."""
    path = tmp_path / "huge.pt"
    torch.save(
        {"adjacency_matrix": torch.zeros(4, 4), "input_tokens": torch.zeros(3)}, path
    )
    with pytest.raises(Refusal, match="above the"):
        circuit.read(path, max_nodes=2)


def test_non_finite_weights_are_reported_not_cleaned(tmp_path):
    a = torch.zeros(4, 4)
    a[0, 1] = float("nan")
    path = tmp_path / "nan.pt"
    torch.save({"adjacency_matrix": a, "input_tokens": torch.zeros(3)}, path)
    g = circuit.read(path)
    assert any("non-finite" in n for n in g.notes)


def test_an_unknown_key_is_noted_rather_than_refused(tmp_path, producer):
    """A newer circuit-tracer adding a field should still open — but an unread
    field must never be mistaken for an absent one."""
    g = circuit.read(_graph_file(tmp_path, producer, some_new_field=torch.zeros(2)))
    assert any("some_new_field" in n for n in g.notes)


# ------------------------------------------------------------------ the size


def test_the_summary_reduces_on_the_tensor(tmp_path):
    a = torch.zeros(100, 100)
    a[1, 2] = 3.0
    a[4, 5] = -5.0
    path = tmp_path / "g.pt"
    torch.save({"adjacency_matrix": a, "input_tokens": torch.zeros(3)}, path)
    s = circuit.read(path).summary()
    assert s["possible_edges"] == 10_000
    assert s["nonzero_edges"] == 2
    assert s["max_abs_weight"] == pytest.approx(5.0)
    assert s["density"] == pytest.approx(2 / 10_000)


def test_edges_are_the_strongest_by_absolute_weight(tmp_path):
    a = torch.zeros(10, 10)
    a[1, 2] = 0.1
    a[3, 4] = -0.9
    a[5, 6] = 0.5
    path = tmp_path / "g.pt"
    torch.save({"adjacency_matrix": a, "input_tokens": torch.zeros(3)}, path)
    edges = circuit.read(path).edges(limit=2)
    assert [e["weight"] for e in edges] == pytest.approx([-0.9, 0.5], abs=1e-6)


def test_an_edge_carries_its_source_and_target(tmp_path):
    a = torch.zeros(10, 10)
    a[7, 3] = 1.0  # row is the target, column the source
    path = tmp_path / "g.pt"
    torch.save({"adjacency_matrix": a, "input_tokens": torch.zeros(3)}, path)
    e = circuit.read(path).edges(limit=1)[0]
    assert e["source"] == 3 and e["target"] == 7


def test_zero_edges_are_not_padded_into_the_list(tmp_path):
    """`topk` returns `limit` values whatever the data holds. A zero edge is
    the ABSENCE of an edge, not an edge of no weight."""
    a = torch.zeros(10, 10)
    a[1, 2] = 1.0
    path = tmp_path / "g.pt"
    torch.save({"adjacency_matrix": a, "input_tokens": torch.zeros(3)}, path)
    assert len(circuit.read(path).edges(limit=50)) == 1


def test_a_truncated_edge_list_says_so(tmp_path):
    a = torch.ones(20, 20)
    path = tmp_path / "g.pt"
    torch.save({"adjacency_matrix": a, "input_tokens": torch.zeros(3)}, path)
    s = circuit.read(path).summary(edge_limit=5)
    assert s["returned_edges"] == 5
    assert s["truncated"] is True


def test_a_whole_edge_list_says_it_is_whole(tmp_path):
    a = torch.zeros(20, 20)
    a[1, 2] = 1.0
    path = tmp_path / "g.pt"
    torch.save({"adjacency_matrix": a, "input_tokens": torch.zeros(3)}, path)
    assert circuit.read(path).summary(edge_limit=5)["truncated"] is False


def test_a_large_graph_produces_a_small_payload(tmp_path):
    """The point of the whole design. 1,000 nodes is a million possible edges;
    serialising them would be tens of megabytes of JSON in somebody's browser
    and is exactly what `.tolist()` on the matrix would do."""
    a = torch.zeros(1000, 1000)
    a[0, 1] = 1.0
    path = tmp_path / "big.pt"
    torch.save({"adjacency_matrix": a, "input_tokens": torch.zeros(3)}, path)
    payload = json.dumps(circuit.read(path).to_dict(edge_limit=100))
    assert circuit.read(path).summary()["possible_edges"] == 1_000_000
    assert len(payload) < 20_000, f"payload was {len(payload)} bytes"


def test_the_report_is_json_safe(tmp_path, producer):
    json.dumps(circuit.read(_graph_file(tmp_path, producer)).to_dict())


# ------------------------------------------------------------ the unpickler


def test_two_reads_do_not_share_a_foreign_registry(tmp_path, producer):
    """The registry is a class attribute, so it has to be a FRESH subclass per
    call — a module-level dict would be a data race the moment the server
    reads two graphs at once."""
    a = circuit.read(_graph_file(tmp_path, producer))
    bare = tmp_path / "bare.pt"
    torch.save(
        {"adjacency_matrix": torch.zeros(4, 4), "input_tokens": torch.zeros(3)}, bare
    )
    b = circuit.read(bare)
    assert a.foreign_classes  # the real one saw classes
    assert b.foreign_classes == []  # the bare one must not inherit them


def test_torch_classes_are_allowed_through(tmp_path):
    """Tensors need torch's own rebuild machinery, so the allow-list cannot be
    empty — the boundary is "torch and nothing else"."""
    assert "torch" in circuit._ALLOWED_MODULES
    a = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    path = tmp_path / "g.pt"
    torch.save({"adjacency_matrix": a, "input_tokens": torch.zeros(3)}, path)
    assert circuit.read(path).summary()["nonzero_edges"] == 8  # 0 is not an edge


# ---------------------------------------------------------------------------
# The security claim, with a control
# ---------------------------------------------------------------------------


class _Evil:
    """A pickle payload that executes a command when unpickled.

    The classic `__reduce__` attack. Module-level so it is picklable.
    """

    target = ""

    def __reduce__(self):
        import os

        return (os.system, (f'echo pwned > "{type(self).target}"',))


def test_a_hostile_pickle_does_not_execute(tmp_path):
    """The reason this module exists rather than a two-line `torch.load`.

    A `.pt` is a pickle and unpickling runs code, so a graph from a stranger
    is arbitrary code execution by default. The reader must neutralise that —
    either by refusing the file or by handing the payload an inert stub — and
    must never run it.

    The control below proves the payload is genuinely dangerous, so a pass
    here cannot be a payload that never worked.
    """
    marker = tmp_path / "PWNED"
    _Evil.target = str(marker)

    path = tmp_path / "evil.pt"
    torch.save(
        {
            "adjacency_matrix": torch.zeros(4, 4),
            "input_tokens": torch.zeros(3),
            "cfg": _Evil(),
        },
        path,
    )

    assert not marker.exists()
    try:
        circuit.read(path)
    except Refusal:
        pass  # refusing is one acceptable outcome; executing is not
    assert not marker.exists(), "the reader executed code from the file"


def test_the_control_shows_that_payload_really_is_dangerous(tmp_path):
    """Without this, the test above could pass against a payload that never
    worked in the first place — which would be a green tick for nothing."""
    import pickle as _pickle

    marker = tmp_path / "PWNED-control"
    _Evil.target = str(marker)
    _pickle.loads(_pickle.dumps(_Evil()))
    assert marker.exists(), (
        "the control payload did not fire; the test above proves nothing"
    )


# ---------------------------------------------------------------------------
# The .mri section: a graph travels like every other finding
# ---------------------------------------------------------------------------


def _session_bytes(tmp_path, producer, **over):
    from modelmri import session

    g = circuit.read(_graph_file(tmp_path, producer))
    doc = {
        "n_nodes": g.n_nodes,
        "edges": g.edges(limit=100),
        "provenance": g.provenance,
        "prompt": g.prompt,
        "summary": g.summary(),
        "notes": g.notes,
    }
    doc.update(over)
    return session.build(
        model_id=None,
        device=None,
        dtype=None,
        n_params=None,
        tokens=[],
        generation="",
        prompt=g.prompt,
        attention={},
        lens=[],
        n_layers=0,
        n_heads=0,
        graph=doc,
    )


def test_a_graph_survives_a_round_trip_through_a_mri(tmp_path, producer):
    from modelmri import session

    s = session.parse(circuit.to_session(circuit.read(_graph_file(tmp_path, producer))))
    assert s.has_graph()
    assert s.graph["n_nodes"] == 32
    assert s.graph["edges"][0]["weight"] == pytest.approx(0.9)


def test_the_provenance_survives_the_round_trip(tmp_path, producer):
    """The one field that must never be lost in transit."""
    from modelmri import session

    s = session.parse(circuit.to_session(circuit.read(_graph_file(tmp_path, producer))))
    assert "ModelMRI did not run the model" in s.graph["provenance"]["measured_by"]
    assert s.graph["provenance"]["producer"] == "circuit-tracer"
    assert s.graph["provenance"]["model"] == "google/gemma-2-2b"


def test_the_mri_header_does_not_claim_the_graphs_model_as_its_own(tmp_path, producer):
    """A `.mri` whose header names a model reads as one this tool loaded. The
    model belongs in the graph's provenance, labelled as the FILE's claim."""
    from modelmri import session

    s = session.parse(circuit.to_session(circuit.read(_graph_file(tmp_path, producer))))
    assert not s.meta.get("model")
    assert s.graph["provenance"]["model"] == "google/gemma-2-2b"


def test_a_session_without_a_graph_carries_no_empty_section(tmp_path):
    from modelmri import session

    blob = session.build(
        model_id=None,
        device=None,
        dtype=None,
        n_params=None,
        tokens=["a"],
        generation="",
        prompt="",
        attention={},
        lens=[],
        n_layers=1,
        n_heads=1,
    )
    assert session.parse(blob).has_graph() is False


def test_building_a_graph_without_provenance_is_refused(tmp_path, producer):
    """The WRITER is as strict as the reader. Dropping the section instead
    would hand back a file the caller believes carries a graph and which does
    not, silently — and the reason it was dropped is the one thing the section
    exists to guarantee."""
    from modelmri.errors import BadRequest as _BadRequest

    with pytest.raises(_BadRequest, match="provenance"):
        _session_bytes(tmp_path, producer, provenance={})


@pytest.mark.parametrize(
    "broken,match",
    [
        ({"edges": [{"source": 999, "target": 0, "weight": 1.0}]}, "outside the"),
        ({"edges": [{"source": 0, "target": 1, "weight": "big"}]}, "not a number"),
        ({"edges": [{"source": 0, "target": 1, "weight": float("inf")}]}, "non-finite"),
        ({"edges": "not a list"}, "missing or malformed"),
        ({"n_nodes": "many"}, "how many nodes"),
    ],
)
def test_a_malformed_graph_stops_at_the_reader(tmp_path, producer, broken, match):
    """Same posture as the rest of `session.parse`: this runs on bytes a
    stranger forwarded, and the indices reach a browser as array subscripts."""
    from modelmri import session
    from modelmri.errors import BadRequest as _BadRequest

    blob = _session_bytes(tmp_path, producer, **broken)
    with pytest.raises(_BadRequest, match=match):
        session.parse(blob)


def test_an_absurd_edge_count_is_refused(tmp_path, producer):
    from modelmri import session
    from modelmri.errors import BadRequest as _BadRequest

    many = [{"source": 0, "target": 1, "weight": 0.1}] * (session.MAX_GRAPH_EDGES + 1)
    with pytest.raises(_BadRequest, match="above the"):
        session.parse(_session_bytes(tmp_path, producer, n_nodes=4, edges=many))


def test_the_two_implementations_agree_on_the_graph_key(tmp_path, producer):
    """`session.py` and `frontend/src/viewer.ts` both read this section, and
    two implementations of one format drift. The names are pinned from the
    Python side here; `tests/viewer_check.py` compares the parsed result."""
    import re

    ts = (ROOT_TS := __import__("pathlib").Path("frontend/src/viewer.ts")).read_text(
        encoding="utf-8"
    )
    assert ROOT_TS.is_file()
    block = re.search(r"graph\?:\s*\{(.+?)\n  \};", ts, re.S)
    assert block, "the viewer's Doc has no graph section"
    for key in ("n_nodes", "edges", "provenance", "summary", "notes"):
        assert key in block.group(1), f"the viewer does not read {key}"
    # And the guard that matters must exist on the viewer side too.
    assert "measured_by" in ts
