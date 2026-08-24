"""A head's behaviour on real text, and the four ways that reads wrong.

`ablate.rank_heads` measures one prompt. `head_types` measures random repeated
tokens and says so. Neither is evidence about a corpus, and the SAE side of
this tool has had the corpus version since long before the attention side did.

The model here is synthetic and its heads are PLANTED: one writes hard at a
known position and attends to a known other one, the rest write nothing. That
is the only way to check a ranking is right — against a real checkpoint there
is nothing to compare the answer to except the code that produced it.

Four failures with no symptom, tested rather than argued:

  the wrong slice     `hidden_size // n_heads` is wrong by 2x on Qwen3-0.6B, so
                      the norms would be half of one head plus half of the next
                      and the ranking would be about nothing.
  the missing pair    reporting where a head WROTE without what it was READING
                      is half a sentence, and the half that gets quoted.
  a flat distribution twenty spans off the top of a flat distribution look
                      exactly like twenty findings.
  a silent cap        "never wrote hard" in the first 200k tokens of a 2M-token
                      log is not "never wrote hard".
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from modelmri import head_corpus  # noqa: E402
from modelmri.errors import BadRequest, Refusal  # noqa: E402

D_MODEL = 32
HEAD_DIM = 8
N_HEADS = 4
N_VOCAB = 40

#: Where the planted head writes, and what it reads. Both are checked exactly.
LOUD_POSITION = 3
LOUD_SOURCE = 1
LOUD_HEAD = 2


class _Attn(torch.nn.Module):
    """Attention whose weights and whose projection input are both PLANTED.

    Nothing is learned and nothing is random: head `LOUD_HEAD` writes a large
    vector at `LOUD_POSITION` and attends there to `LOUD_SOURCE`, and every
    other head writes zeros. So the expected ranking is known in advance,
    which is what makes the assertions below evidence rather than a
    restatement of the implementation.
    """

    def __init__(self):
        super().__init__()
        self.head_dim = HEAD_DIM
        self.q_proj = torch.nn.Linear(D_MODEL, N_HEADS * HEAD_DIM, bias=False)
        self.k_proj = torch.nn.Linear(D_MODEL, N_HEADS * HEAD_DIM, bias=False)
        self.v_proj = torch.nn.Linear(D_MODEL, N_HEADS * HEAD_DIM, bias=False)
        self.o_proj = torch.nn.Linear(N_HEADS * HEAD_DIM, D_MODEL, bias=False)

    def forward(self, x, output_attentions: bool = False):
        seq = x.shape[1]
        # The projection input `capture_projection_inputs` hooks. Only the loud
        # head's columns are non-zero, and only at the loud position.
        mixed = torch.zeros(1, seq, N_HEADS * HEAD_DIM, dtype=x.dtype)
        lo = LOUD_HEAD * HEAD_DIM
        if seq > LOUD_POSITION:
            mixed[0, LOUD_POSITION, lo : lo + HEAD_DIM] = 3.0
        out = self.o_proj(mixed)
        if not output_attentions:
            return out, None
        # Uniform everywhere except the loud head's loud row, which points at
        # LOUD_SOURCE. Rows sum to 1, like real attention.
        attn = torch.full((1, N_HEADS, seq, seq), 1.0 / seq)
        if seq > max(LOUD_POSITION, LOUD_SOURCE):
            attn[0, LOUD_HEAD, LOUD_POSITION] = 0.0
            attn[0, LOUD_HEAD, LOUD_POSITION, LOUD_SOURCE] = 1.0
        return out, attn


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attn()

    def forward(self, x, output_attentions: bool = False):
        out, attn = self.self_attn(x, output_attentions=output_attentions)
        return x + out, attn


class _Out:
    def __init__(self, attentions):
        self.attentions = attentions


class _Model(torch.nn.Module):
    def __init__(self, n_layers: int = 2, attentions: bool = True):
        super().__init__()
        self.embed = torch.nn.Embedding(N_VOCAB, D_MODEL)
        self.layers = torch.nn.ModuleList([_Block() for _ in range(n_layers)])
        self.config = type("C", (), {"hidden_size": D_MODEL})()
        self._attentions = attentions
        # The OV leg of `evidence()` reads these. A model that has an
        # attention stack and no embedding table is not a shape any real
        # checkpoint takes, so the fixture carries them rather than the
        # module growing a branch for a model that cannot exist.
        self.norm = torch.nn.LayerNorm(D_MODEL)
        self.lm_head = torch.nn.Linear(D_MODEL, N_VOCAB, bias=False)

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, ids, output_attentions: bool = False):
        x = self.embed(ids)
        maps = []
        for block in self.layers:
            x, attn = block(x, output_attentions=output_attentions)
            if output_attentions and attn is not None:
                maps.append(attn)
        if not output_attentions:
            return _Out(None)
        return _Out(tuple(maps) if self._attentions else None)


class _Tok:
    """Whitespace tokens, so a span's characters are checkable by eye."""

    def __call__(self, text, return_tensors=None):
        ids = self.encode(text)
        return type("B", (), {"input_ids": torch.tensor([ids])})()

    def encode(self, text, add_special_tokens=False):
        return [(abs(hash(w)) % (N_VOCAB - 1)) + 1 for w in str(text).split()]

    def decode(self, ids):
        return " ".join(f"w{int(i)}" for i in ids)


def _blocks(model):
    return lambda i: model.layers[i]


def run(texts=None, **kw):
    model = kw.pop("model", None) or _Model()
    return head_corpus.sweep(
        model,
        _Tok(),
        _blocks(model),
        texts if texts is not None else ["alpha beta gamma delta epsilon zeta"],
        layer=kw.pop("layer", 0),
        head=kw.pop("head", LOUD_HEAD),
        n_heads=N_HEADS,
        corpus_label=kw.pop("corpus_label", "planted corpus"),
        **kw,
    )


# --------------------------------------------------- the planted answer


def test_the_loud_position_is_found_and_the_quiet_heads_are_not():
    """The whole feature in one assertion. The synthetic head writes 3.0 across
    eight dimensions at exactly one position, so its norm there is
    sqrt(8*9) = 8.485 and zero everywhere else — a number that can be checked
    by hand, which a real checkpoint could not offer."""
    out = run()
    assert out.spans, "the planted write was not found at all"
    top = out.spans[0]
    assert top.position == LOUD_POSITION
    assert top.write_norm == pytest.approx((8 * 9) ** 0.5, abs=1e-4)
    assert out.write_norm_max == pytest.approx(top.write_norm, abs=1e-4)

    # A head that writes nothing writes nothing — not a small number, and not
    # a ranking of noise.
    quiet = run(head=0)
    assert quiet.write_norm_max == 0.0
    assert quiet.n_wrote == 0
    assert quiet.spans == [], "a zero write is not a span"
    assert "NOT SEEN IN THIS CORPUS" in quiet.means()


def test_the_slice_is_the_one_head_ablation_removes():
    """`hidden_size // n_heads` is wrong by 2x on Qwen3-0.6B and by a different
    factor on gemma-3-270m. Reading the wrong slice takes half of one head and
    half of the next, and the ranking is then about nothing at all — with no
    symptom, because the numbers still look like numbers.

    Checked by asking for each head in turn: only the planted one is loud, and
    a slice off by even one head_dim would light up a neighbour.
    """
    loud = [h for h in range(N_HEADS) if run(head=h).write_norm_max > 0]
    assert loud == [LOUD_HEAD], f"the slice found heads {loud}"


def test_a_span_says_what_the_head_was_reading_not_only_where_it_wrote():
    """The pair. `feature_corpus.Span` carries one offset because a feature
    fires in one place; a head writes at one position while attending to
    another, and a span that reported only the first is the half-sentence this
    module exists to stop."""
    top = run().spans[0]
    assert top.source_position == LOUD_SOURCE
    assert top.source_token is not None
    assert top.source_share == pytest.approx(1.0, abs=1e-5)
    assert top.position != top.source_position, "the pair collapsed to one place"


def test_a_model_that_will_not_give_attention_degrades_rather_than_failing():
    """The write norms cost a forward pass each and are already measured when
    the attention pass fails, so raising here would throw away half an answer
    that is real. `None`, never 0.0 — a share of zero would say the head looked
    nowhere, which is a measurement."""
    out = run(model=_Model(attentions=False))
    assert out.attention_read is False
    assert out.spans and out.spans[0].source_position is None
    assert out.spans[0].source_share is None
    assert out.spans[0].write_norm > 0, "the half that worked was thrown away"
    assert "ATTENTION WAS NOT READ" in out.means()


def test_asking_for_the_cheap_half_costs_one_pass_per_sequence():
    """`read_attention=False` is a supported answer, not a degraded one, and it
    is the difference between one pass per sequence and two."""
    both = run(texts=["alpha beta gamma delta epsilon"], read_attention=True)
    cheap = run(texts=["alpha beta gamma delta epsilon"], read_attention=False)
    assert cheap.attention_read is False
    assert cheap.passes < both.passes
    assert cheap.write_norm_max == pytest.approx(both.write_norm_max, abs=1e-6)


# ------------------------------------------------- what it refuses to imply


def test_the_shape_of_the_distribution_travels_with_the_top_spans():
    """ "The top 20" means nothing without it. A head whose largest write is
    barely above its median has no tail, and twenty spans off a flat
    distribution read as twenty findings."""
    # A corpus long enough that most positions are NOT kept, which is the
    # ordinary case and the one the ratio is for.
    out = run(texts=["alpha beta gamma delta epsilon zeta eta theta iota kappa"])
    assert out.n_positions > len(out.spans), (
        "every position was kept — either the corpus is too short to exercise "
        "this or zero-write positions are being reported as spans"
    )
    # And the ones kept are real: a span with a zero write norm is not a place
    # the head wrote, and listing it turns one finding into twenty.
    assert all(s.write_norm > 0 for s in out.spans)
    assert out.write_norm_mean >= 0.0
    # Sparse is its OWN sentence. The first version branched on the median, so
    # a head that wrote once in ten positions — median 0, largest 8.49 —
    # printed "NOT SEEN IN THIS CORPUS", reporting a sparse head as an absent
    # one. That is the exact collapse this module exists to undo.
    said = out.means()
    assert out.write_norm_max > 0 and out.write_norm_median == 0.0
    assert "SPARSE HERE" in said
    assert "NOT SEEN IN THIS CORPUS" not in said
    assert f"{out.n_wrote:,} of {out.n_positions:,} positions" in said


def test_no_label_is_ever_attached():
    """`head_types.py` gates every label it attaches on a null it measured on
    this model, and still says the label is behaviour on random tokens rather
    than a claim about text. Nothing here measured a null."""
    said = run().means()
    for verdict in ("induction head", "name mover", "copying head", "detects"):
        assert verdict not in said.lower()
    assert "attaches no label" in said


def test_the_corpus_is_named_counted_and_hashed():
    """Every number is about the text handed in. A result that cannot say which
    text is not one anybody can check or compare."""
    a = run(texts=["alpha beta gamma delta"], corpus_label="mine")
    b = run(texts=["zeta eta theta iota"], corpus_label="mine")
    assert a.corpus_label == "mine"
    assert a.n_tokens == 4 and a.n_sequences == 1
    assert a.corpus_sha256 and a.corpus_sha256 != b.corpus_sha256, (
        "two different corpora under one label are indistinguishable"
    )


def test_an_unnamed_corpus_is_refused():
    with pytest.raises(BadRequest, match="name the corpus"):
        run(corpus_label="   ")


def test_the_cap_is_reported_and_not_merely_applied():
    """ "Never wrote hard" in the first N tokens of a longer log is a different
    claim from "never wrote hard"."""
    out = run(texts=["alpha beta gamma delta epsilon zeta eta"], max_tokens=3)
    assert out.truncated is True
    assert out.n_tokens == 3
    assert "was cut at" in out.means()

    whole = run(texts=["alpha beta gamma"], max_tokens=1000)
    assert whole.truncated is False
    assert "was cut at" not in whole.means()


# ------------------------------------------------------------- refusals


def test_a_head_outside_the_layer_costs_nothing_to_refuse():
    """The caller's mistake, caught before a single forward pass — a head index
    that does not exist should not cost a corpus sweep to discover."""
    for bad in (N_HEADS, 99):
        with pytest.raises(BadRequest, match="outside this layer"):
            run(head=bad)
    with pytest.raises(BadRequest, match="whole number"):
        run(head=True)


def test_an_empty_corpus_is_refused_with_the_next_step():
    for bad in ([], "not a list", [b"bytes"]):
        with pytest.raises(BadRequest, match="list of strings"):
            run(texts=bad)


def test_text_that_tokenises_to_nothing_is_refused_rather_than_answered():
    """Zero positions is not a measurement of zero. The refusal names the two
    things that actually cause it."""
    with pytest.raises(Refusal, match="nothing to measure"):
        run(texts=["   ", ""])


# ----------------------------------------------------------- the price


def test_the_pass_count_is_knowable_before_it_is_spent():
    """Priced in passes and never in seconds, like everything else here: a
    duration guessed from another machine is the kind of number people plan
    around."""
    with_attn = head_corpus.sweep_cost(12, read_attention=True)
    without = head_corpus.sweep_cost(12, read_attention=False)
    assert with_attn["passes"] == 24
    assert without["passes"] == 12
    assert "not read" in without["means"].lower()
    assert "[heads x S x S]" in with_attn["means"]


def test_the_price_refuses_a_bool_and_a_negative():
    """`isinstance(True, int)` is True, so `n_sequences=True` would have priced
    a one-sequence run."""
    with pytest.raises(BadRequest, match="whole number"):
        head_corpus.sweep_cost(True, read_attention=False)
    with pytest.raises(BadRequest, match="at least 0"):
        head_corpus.sweep_cost(-3, read_attention=False)


# ------------------------------------------------- the three-legged answer


def test_the_causal_leg_is_named_rather_than_quietly_omitted(monkeypatch):
    """`feature_corpus.py` argues that a claim resting on one readout is worth
    less than one surviving three. The causal leg for a HEAD is hundreds of
    passes, so it is not run here — but a two-legged answer presented as the
    whole thing is exactly what that argument is against, so it is named."""
    monkeypatch.setattr("modelmri.lens._final_norm", lambda m: m.norm)
    model = _Model()
    out = head_corpus.evidence(
        model,
        _Tok(),
        _blocks(model),
        ["alpha beta gamma delta epsilon"],
        layer=0,
        head=LOUD_HEAD,
        n_heads=N_HEADS,
        corpus_label="planted corpus",
    )
    assert out["causal"]["available"] is False
    assert "--metric heads" in out["causal"]["how"]
    assert out["causal"]["why"]
    assert out["corpus"]["spans"]
    assert "was not run" in out["means"]
