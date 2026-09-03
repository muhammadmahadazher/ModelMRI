# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""A head's wiring, read off its weights — and the two ways that reads wrong.

`ablate.rank_heads` answers "does this head matter for THIS prompt". This
answers "what is this head wired to do", and the answer must be identical every
time it is asked, because it is arithmetic on weights rather than a measurement
of behaviour.

Two failures have no symptom and both are tested here rather than argued:

  the wrong value head   grouped-query attention gives 16 query heads 8 value
                         heads, so slicing `W_V` by the query index reads a
                         NEIGHBOURING head's values for every head past the
                         first group. Same shapes, same dtypes, plausible
                         numbers, wrong head.
  the wrong orientation  `nn.Linear` stores `[out, in]` and GPT-2's `Conv1D`
                         stores `[in, out]`. On a square projection the two
                         have identical shape, so reading one as the other is
                         invisible until the numbers are checked against a
                         hand-computed product.

Nothing here loads a real model. The quantities under test are exact products
of weights, so a synthetic module with known weights checks them exactly —
which a real checkpoint could not, since there would be nothing to compare
against but the code itself.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from modelmri import ov_circuits  # noqa: E402
from modelmri.errors import BadRequest, Refusal  # noqa: E402

D_MODEL = 32
HEAD_DIM = 8


def block(*, n_heads: int, n_kv_heads: int, fused: bool = False, seed: int = 0):
    """One transformer block's attention, with the geometry stated explicitly.

    `n_kv_heads < n_heads` is grouped-query attention, which is what every
    recent model does and what the head-to-value mapping exists for.
    """
    torch.manual_seed(seed)
    b = torch.nn.Module()
    attn = torch.nn.Module()
    attn.head_dim = HEAD_DIM
    if fused:
        # GPT-2's shape: one projection producing q, k and v concatenated, and
        # a `Conv1D` that stores its weight transposed relative to `nn.Linear`.
        c_attn = torch.nn.Module()
        c_attn.weight = torch.nn.Parameter(torch.randn(D_MODEL, 3 * n_heads * HEAD_DIM))
        attn.c_attn = c_attn
    else:
        attn.q_proj = torch.nn.Linear(D_MODEL, n_heads * HEAD_DIM, bias=False)
        attn.k_proj = torch.nn.Linear(D_MODEL, n_kv_heads * HEAD_DIM, bias=False)
        attn.v_proj = torch.nn.Linear(D_MODEL, n_kv_heads * HEAD_DIM, bias=False)
    attn.o_proj = torch.nn.Linear(n_heads * HEAD_DIM, D_MODEL, bias=False)
    b.self_attn = attn
    return b


# --------------------------------------------------------------- geometry


def test_the_value_head_a_query_head_reads_is_measured_not_assumed():
    """Qwen3-1.7B is 16 query heads over 8 value heads. Slicing `W_V` by the
    QUERY index reads head 8's values for head 8 — which does not exist — or,
    worse, a neighbour's for every head in the second half. The mapping is one
    line and it is the line grouped-query attention exists for."""
    geo = ov_circuits.geometry(
        block(n_heads=16, n_kv_heads=8), n_heads=16, d_model=D_MODEL
    )
    assert (geo.n_heads, geo.n_kv_heads, geo.group_size) == (16, 8, 2)
    assert [geo.kv_head(h) for h in range(16)] == [h // 2 for h in range(16)]

    # Ordinary multi-head attention is the degenerate case, not a special one.
    plain = ov_circuits.geometry(
        block(n_heads=4, n_kv_heads=4), n_heads=4, d_model=D_MODEL
    )
    assert plain.group_size == 1
    assert [plain.kv_head(h) for h in range(4)] == [0, 1, 2, 3]


def test_n_kv_heads_is_derived_from_the_projection_not_from_the_config():
    """`config.num_key_value_heads` is absent on several architectures, and on
    the ones that set it this division has to agree with it anyway. Deriving it
    from `v_proj`'s own width makes the disagreement impossible rather than
    unlikely — and there is no config object in this test at all."""
    for n_kv in (1, 2, 4, 8):
        geo = ov_circuits.geometry(
            block(n_heads=8, n_kv_heads=n_kv), n_heads=8, d_model=D_MODEL
        )
        assert geo.n_kv_heads == n_kv, n_kv


def test_a_value_projection_that_is_not_whole_heads_is_refused():
    """A width that is not a multiple of `head_dim` means nothing here can say
    where one value head ends. A guess would read a neighbouring head's values
    for half the heads, with no symptom."""
    b = block(n_heads=4, n_kv_heads=2)
    b.self_attn.v_proj = torch.nn.Linear(D_MODEL, 2 * HEAD_DIM + 3, bias=False)
    with pytest.raises(Refusal, match="whole number of"):
        ov_circuits.geometry(b, n_heads=4, d_model=D_MODEL)


def test_query_heads_that_do_not_divide_by_value_heads_are_refused():
    """Grouped-query attention needs the first to be a whole multiple of the
    second. 6 query heads over 4 value heads has no answer to "which value head
    does head 5 read", so nothing is invented."""
    b = block(n_heads=6, n_kv_heads=4)
    with pytest.raises(Refusal, match="whole multiple"):
        ov_circuits.geometry(b, n_heads=6, d_model=D_MODEL)


def test_a_block_with_no_attention_says_so():
    with pytest.raises(Refusal, match="neither `attn` nor `self_attn`"):
        ov_circuits.geometry(torch.nn.Module(), n_heads=4, d_model=D_MODEL)


def test_a_bool_head_count_is_not_read_as_one():
    """`isinstance(True, int)` is True, so `n_heads=True` would have been a
    one-head model rather than a refusal."""
    with pytest.raises(BadRequest, match="at least one attention head"):
        ov_circuits.geometry(block(n_heads=4, n_kv_heads=4), n_heads=True, d_model=32)


# ------------------------------------------------------------ the factors


def test_the_ov_factors_are_this_head_and_its_own_value_head():
    """The exactness check. Both factors are sliced out of the real weights,
    so they can be compared against the slices themselves — and the value
    factor must come from `kv(h)`, not from `h`."""
    b = block(n_heads=8, n_kv_heads=2)
    w_v_all = b.self_attn.v_proj.weight.detach().float()
    w_o_all = b.self_attn.o_proj.weight.detach().float()

    for head in range(8):
        w_o, w_v, geo = ov_circuits.ov_factors(b, head, n_heads=8, d_model=D_MODEL)
        assert w_o.shape == (D_MODEL, HEAD_DIM)
        assert w_v.shape == (HEAD_DIM, D_MODEL)
        kv = head // 4
        assert geo.kv_head(head) == kv
        assert torch.allclose(
            w_o, w_o_all[:, head * HEAD_DIM : (head + 1) * HEAD_DIM]
        ), head
        assert torch.allclose(w_v, w_v_all[kv * HEAD_DIM : (kv + 1) * HEAD_DIM, :]), (
            head
        )

    # And the heads in one group really do share a value factor — which is the
    # fact that makes the wrong slice invisible on the first group and wrong on
    # every later one.
    a0, v0, _ = ov_circuits.ov_factors(b, 0, n_heads=8, d_model=D_MODEL)
    a3, v3, _ = ov_circuits.ov_factors(b, 3, n_heads=8, d_model=D_MODEL)
    _, v4, _ = ov_circuits.ov_factors(b, 4, n_heads=8, d_model=D_MODEL)
    assert torch.allclose(v0, v3)
    assert not torch.allclose(v0, v4)
    assert not torch.allclose(a0, a3)


def test_the_qk_factors_use_the_same_value_head_mapping():
    """Keys are grouped exactly as values are. A QK circuit that sliced `W_K`
    by the query index would score the wrong pair for every head past the first
    group."""
    b = block(n_heads=8, n_kv_heads=2)
    w_k_all = b.self_attn.k_proj.weight.detach().float()
    for head in (0, 3, 4, 7):
        w_q, w_k, geo = ov_circuits.qk_factors(b, head, n_heads=8, d_model=D_MODEL)
        assert w_q.shape == (HEAD_DIM, D_MODEL)
        assert w_k.shape == (HEAD_DIM, D_MODEL)
        kv = geo.kv_head(head)
        assert torch.allclose(w_k, w_k_all[kv * HEAD_DIM : (kv + 1) * HEAD_DIM, :])


def test_a_fused_qkv_projection_is_split_in_q_k_v_order():
    """GPT-2 concatenates the three along the output dimension, and its
    `Conv1D` stores the weight transposed relative to `nn.Linear`. Read as a
    Linear, every number would be wrong while every shape stayed right."""
    b = block(n_heads=4, n_kv_heads=4, fused=True)
    fused = b.self_attn.c_attn.weight.detach().float().T  # [3*n*hd, d_model]
    third = fused.shape[0] // 3

    _, w_v, _ = ov_circuits.ov_factors(b, 1, n_heads=4, d_model=D_MODEL)
    assert torch.allclose(w_v, fused[2 * third :][HEAD_DIM : 2 * HEAD_DIM, :])

    w_q, w_k, _ = ov_circuits.qk_factors(b, 1, n_heads=4, d_model=D_MODEL)
    assert torch.allclose(w_q, fused[:third][HEAD_DIM : 2 * HEAD_DIM, :])
    assert torch.allclose(w_k, fused[third : 2 * third][HEAD_DIM : 2 * HEAD_DIM, :])


def test_a_head_outside_the_layer_is_refused_by_number():
    b = block(n_heads=4, n_kv_heads=4)
    for bad in (4, -1, 99):
        with pytest.raises(BadRequest, match="outside this layer"):
            ov_circuits.ov_factors(b, bad, n_heads=4, d_model=D_MODEL)
    with pytest.raises(BadRequest, match="a head is an index"):
        ov_circuits.ov_factors(b, True, n_heads=4, d_model=D_MODEL)


# ------------------------------------------------- what it says out loud


class _Tiny(torch.nn.Module):
    """A model just complete enough to have an embedding, a norm and a head."""

    def __init__(self, n_vocab: int = 40):
        super().__init__()
        torch.manual_seed(1)
        self.embed = torch.nn.Embedding(n_vocab, D_MODEL)
        self.norm = torch.nn.LayerNorm(D_MODEL)
        self.lm_head = torch.nn.Linear(D_MODEL, n_vocab, bias=False)
        self.config = type("C", (), {"hidden_size": D_MODEL})()

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head


class _Tok:
    def decode(self, ids):
        return f"<{ids[0]}>"


def test_the_vocabulary_readout_is_the_product_it_claims_to_be(monkeypatch):
    """Exactness, checked against the product computed by hand. This is the
    leg that would be wrong-but-plausible if the value slice or the weight
    orientation were off, so it is checked as arithmetic rather than as shape."""
    monkeypatch.setattr(ov_circuits, "_final_norm", None, raising=False)
    model, b = _Tiny(), block(n_heads=4, n_kv_heads=2)
    monkeypatch.setattr("modelmri.lens._final_norm", lambda m: m.norm)

    said = ov_circuits.ov_vocabulary(
        model, _Tok(), b, 3, n_heads=4, source_token_id=7, top_k=3
    )
    w_o, w_v, _ = ov_circuits.ov_factors(b, 3, n_heads=4, d_model=D_MODEL)
    with torch.no_grad():
        e = model.embed.weight[7].float()
        expect = model.lm_head(model.norm(w_o @ (w_v @ e))).float()
        expect = expect - expect.mean()
        best = torch.topk(expect, 3)

    assert said["promotes"][0]["token"] == f"<{int(best.indices[0])}>"
    assert said["promotes"][0]["score"] == pytest.approx(
        float(best.values[0]), abs=1e-4
    )
    assert said["exact"] is True
    assert said["kv_head"] == 1
    # The caveat is not optional chrome: at unit scale these RANK tokens, and a
    # reader who takes them for logit amounts is reading a number nobody
    # measured.
    assert "RANK tokens" in said["means"]
    assert "NO CORPUS AND NO SAMPLING" in said["means"]


def test_a_token_outside_the_vocabulary_is_refused_by_size(monkeypatch):
    monkeypatch.setattr("modelmri.lens._final_norm", lambda m: m.norm)
    model, b = _Tiny(), block(n_heads=4, n_kv_heads=4)
    with pytest.raises(BadRequest, match="outside this model's"):
        ov_circuits.ov_vocabulary(
            model, _Tok(), b, 0, n_heads=4, source_token_id=40, top_k=3
        )
    with pytest.raises(BadRequest, match="a source token is an id"):
        ov_circuits.ov_vocabulary(
            model, _Tok(), b, 0, n_heads=4, source_token_id=True, top_k=3
        )


def test_the_spectrum_reports_its_sample_rather_than_implying_the_vocabulary():
    """The full OV circuit is V x V — 92 TB on Qwen3-1.7B — so the eigenvalues
    are of a SAMPLE. A payload that said "positive fraction 0.83" without the
    sample size beside it would read as a property of the head, which is a
    claim about every token including the ones nobody drew."""
    model, b = _Tiny(n_vocab=40), block(n_heads=4, n_kv_heads=2)
    said = ov_circuits.ov_spectrum(model, b, 2, n_heads=4, n_samples=16, seed=3)

    assert said["n_sampled"] == 16
    assert said["n_vocab"] == 40
    assert said["seed"] == 3
    assert said["sample_capped"] is False
    assert 0.0 <= said["positive_fraction"] <= 1.0
    assert said["positive"] == round(said["positive_fraction"] * 16)
    assert "MEASURED OVER A SAMPLE" in said["means"]
    assert "16" in said["means"] and "40" in said["means"]
    # No label, ever. `head_types.py` gates every label it attaches on a
    # measured null; this has none, so it attaches none.
    for verdict in ("is a copying head", "induction head", "this head copies"):
        assert verdict not in said["means"]


def test_asking_for_more_tokens_than_exist_reports_the_cap():
    """Not an error — the honest answer is the whole vocabulary — but a result
    that said 100,000 when it measured 40 would be."""
    model, b = _Tiny(n_vocab=40), block(n_heads=4, n_kv_heads=4)
    said = ov_circuits.ov_spectrum(model, b, 0, n_heads=4, n_samples=100_000)
    assert said["n_sampled"] == 40
    assert said["sample_capped"] is True


def test_the_spectrum_is_reproducible_at_a_seed_and_moves_without_one():
    """It is weight arithmetic over a drawn sample: same seed, same answer, or
    the number is not a measurement of anything."""
    model, b = _Tiny(n_vocab=200), block(n_heads=4, n_kv_heads=4)
    a = ov_circuits.ov_spectrum(model, b, 1, n_heads=4, n_samples=32, seed=5)
    again = ov_circuits.ov_spectrum(model, b, 1, n_heads=4, n_samples=32, seed=5)
    other = ov_circuits.ov_spectrum(model, b, 1, n_heads=4, n_samples=32, seed=6)
    assert a["positive_fraction"] == again["positive_fraction"]
    assert a["trace"] == again["trace"]
    assert (a["n_sampled"], other["n_sampled"]) == (32, 32)


def test_a_spectrum_of_one_token_is_refused():
    """One eigenvalue is not a spectrum, and a "positive fraction" over a
    single draw is 0.0 or 1.0 — a number with the shape of a measurement and
    none of the content."""
    model, b = _Tiny(), block(n_heads=4, n_kv_heads=4)
    for bad in (1, 0, -4):
        with pytest.raises(BadRequest, match="at least 2 sampled tokens"):
            ov_circuits.ov_spectrum(model, b, 0, n_heads=4, n_samples=bad)


def test_a_model_with_no_embedding_table_says_what_still_works():
    """A refusal that names the thing that DOES work is worth twice one that
    does not: the causal answer needs no embedding table at all."""

    class NoEmbed(_Tiny):
        def get_input_embeddings(self):
            return None

    with pytest.raises(Refusal, match="ablate the head and measure"):
        ov_circuits.ov_spectrum(NoEmbed(), block(n_heads=4, n_kv_heads=4), 0, n_heads=4)


# ------------------------------------------------------ over the wire


def _client():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from modelmri.server import create_app

    return TestClient(create_app())


def test_the_routes_refuse_without_a_model_rather_than_500():
    """Both are weight-space readouts, so "nothing loaded" is the ordinary
    resting state of the panel they sit in — and a 500 there would say the tool
    is broken when the honest answer is that there are no weights to read."""
    c = _client()
    for url in (
        "/api/attention/ov?layer=0&head=0&token=Paris",
        "/api/attention/ov/spectrum?layer=0&head=0",
    ):
        r = c.get(url)
        assert r.status_code == 409, (url, r.status_code)
        assert r.json()["error"] == "No model loaded — pick one first.", url


def test_a_missing_source_token_is_the_callers_mistake_not_the_models():
    """422, not 409, and BEFORE the model check: asking what a head writes
    when it attends to nothing has no answer whether or not weights are
    resident, and answering "no model loaded" would send the reader to load
    one and get the same emptiness back."""
    c = _client()
    for query in ("", "token=", "token=%20%20"):
        r = c.get(f"/api/attention/ov?layer=0&head=0&{query}")
        assert r.status_code == 422, query
        assert "Name a token" in r.json()["error"], query


def test_the_spectrum_default_comes_from_the_module_not_the_route():
    """`n_samples=0` means "whatever the module uses", so the route cannot
    grow a second default that drifts from it. Checked at the module rather
    than over the wire, because the wire needs weights."""
    assert ov_circuits.SPECTRUM_SAMPLE >= 2
    source = (
        Path(__file__).resolve().parents[1] / "modelmri" / "runtime.py"
    ).read_text(encoding="utf-8")
    assert "n_samples or ov_circuits.SPECTRUM_SAMPLE" in source, (
        "the runtime stopped deferring to the module's own default"
    )
