"""`session.build` must refuse what `session.parse` would refuse.

Ten of the fourteen sections in a `.mri` already went through the READER's
own validator on the way out, and the comments around them say why: a writer
laxer than its reader is how you build a file that this tool signs its name to
and then cannot open, with the failure landing on the RECIPIENT, who has no
way to fix it.

Four did not. `lens`, `lens_info`, `patch` and `graph` were assigned straight
into the document. `_lens`'s own docstring records how that happened -- the
section was in the format from the beginning and nothing ever wrote it, so the
reader was hardened and the writer was never brought up to match.

The route bounds here are the same defect one step earlier: `/api/lens` and
`/api/features/summary` take `top_k` as a bare query integer while every POST
sibling declares `Field(ge=1, le=100)`, so the number that reaches the file --
or reaches `torch.topk` -- is whatever was typed.
"""

from __future__ import annotations

import pytest

from modelmri import session
from modelmri.session import MAX_DIM, SessionError


def _build(**over) -> bytes:
    args = dict(
        model_id="Qwen/Qwen3-1.7B",
        device="cpu",
        dtype="float32",
        n_params=1_720_000_000,
        tokens=["a", "b"],
        prompt="hello",
        generation="world",
        attention={(0, 0): [[1.0, 0.0], [0.5, 0.5]]},
        n_layers=1,
        n_heads=1,
        n_prompt=1,
    )
    args.update(over)
    return session.build(**args)


# ------------------------------------------------------------------- lens


def test_a_lens_row_too_wide_to_read_is_refused_at_write_time():
    """THE DEFECT, end to end. `/api/lens?top_k=5000` produced rows this
    file's own reader will not accept, and `build` wrote them anyway -- so the
    Share button handed back a file that 422s on open."""
    wide = [
        {"layer": 0, "tokens": ["x"] * (MAX_DIM + 1), "probs": [0.0] * (MAX_DIM + 1)}
    ]
    with pytest.raises(SessionError) as caught:
        _build(lens=wide)
    assert "too many predictions" in str(caught.value)


def test_a_lens_row_whose_tokens_and_probs_disagree_is_refused_at_write_time():
    """The panel zips these together: mismatched lengths render a token beside
    somebody else's probability."""
    with pytest.raises(SessionError):
        _build(lens=[{"layer": 0, "tokens": ["a", "b"], "probs": [1.0]}])


def test_an_ordinary_lens_still_round_trips():
    """The guard must not cost the ordinary path."""
    rows = [{"layer": 0, "tokens": ["a", "b"], "probs": [0.7, 0.3]}]
    parsed = session.parse(_build(lens=rows))
    assert len(parsed.lens) == 1
    assert parsed.lens[0]["tokens"] == ["a", "b"]


def _document(blob: bytes) -> dict:
    """The file as it was WRITTEN, not as the reader re-reads it.

    `parse` runs the same validator again on the way in, so it filters junk
    out whether or not the writer did -- which makes it blind to the thing
    under test here. The claim is about the bytes.
    """
    import gzip
    import json

    return json.loads(gzip.decompress(blob).decode("utf-8"))


def test_lens_info_is_written_as_the_reader_will_read_it():
    """`_lens` keeps the keys it understands and drops the rest. Writing the
    raw dict meant the FILE carried fields no reader would ever surface --
    weight in a document whose whole purpose is that its contents are exactly
    what somebody else can read."""
    rows = [{"layer": 0, "tokens": ["a"], "probs": [1.0]}]
    blob = _build(lens=rows, lens_info={"settled_at": 3, "not_a_real_field": "x"})
    written = _document(blob)["lens_info"]
    assert written.get("settled_at") == 3
    assert "not_a_real_field" not in written


def test_settled_at_none_survives_the_writer():
    """`null` is a RESULT here -- "never settles before the last layer" -- and
    coercing it to 0 would claim it settled immediately."""
    rows = [{"layer": 0, "tokens": ["a"], "probs": [1.0]}]
    parsed = session.parse(_build(lens=rows, lens_info={"settled_at": None}))
    assert parsed.lens_info.get("settled_at", "missing") is None


# ------------------------------------------------------------ patch, graph


def test_a_ragged_patch_grid_is_refused_at_write_time():
    """The grids reach the viewer as nested loop bounds. A ragged one is a
    crash in a stranger's browser, and the writer had no opinion about it."""
    with pytest.raises(SessionError):
        _build(
            patch={
                "clean": "a",
                "corrupt": "b",
                "grids": {"resid": [[0.1, 0.2], [0.3]]},
            }
        )


def test_a_well_formed_patch_still_round_trips():
    good = {"clean": "a", "corrupt": "b", "grids": {"resid": [[0.1, 0.2], [0.3, 0.4]]}}
    parsed = session.parse(_build(patch=good))
    assert parsed.patch["grids"]["resid"][1][1] == pytest.approx(0.4)


def test_a_graph_edge_with_no_verdict_is_refused_at_write_time():
    """A graph carries attributions THIS TOOL DID NOT COMPUTE. The provenance
    check was already here; the rest of the reader's rules were not."""
    with pytest.raises(SessionError):
        _build(
            graph={
                "provenance": {"measured_by": "circuit-tracer"},
                "n_nodes": 2,
                "edges": [{"source": 0, "target": 1, "weight": "not a number"}],
            }
        )


# ------------------------------------------------ the number that gets there


def test_an_unbounded_top_k_is_refused_before_it_reaches_the_lens():
    """The same defect one step earlier. `/api/lens` declared `top_k: int` with
    no bound while every POST sibling declares `Field(ge=1, le=100)`, so a
    query string decided how wide the rows in the exported file would be.

    422, not 500 and not a file nobody can open.
    """
    from fastapi.testclient import TestClient

    from modelmri.server import create_app

    client = TestClient(create_app())
    assert client.get("/api/lens?top_k=5000").status_code == 422
    assert client.get("/api/lens?top_k=0").status_code == 422


def test_features_top_k_is_bounded_by_the_sae_this_model_actually_has():
    """`feats.topk(top_k)` raises a torch error above the SAE's width, and
    that reached the caller as a 500 about an index rather than the 422 it is.

    The bound is READ from the tensor, not written down: two SAEs for one
    model routinely differ in width, so a constant here would be wrong for one
    of them.
    """
    import threading

    import pytest as _pytest

    torch = _pytest.importorskip("torch")

    from modelmri.errors import BadRequest
    from modelmri.runtime import ModelRuntime

    width = 6
    rt = ModelRuntime.__new__(ModelRuntime)
    rt._decoding = threading.Event()
    rt._feats = torch.zeros(3, width)
    rt.last_ids = torch.arange(3)
    rt.last_ids_epoch = rt.epoch = 1
    rt.sae = object()
    rt.tokenizer = type("T", (), {"decode": staticmethod(lambda ids: "x")})()

    with _pytest.raises(BadRequest) as caught:
        rt.features_summary(top_k=width + 1)
    said = caught.value.sentence
    assert f"[1,{width}]" in said
    assert f"{width} features" in said

    # And the ordinary case still answers.
    out = rt.features_summary(top_k=2)
    assert len(out["tokens"]) == 3
