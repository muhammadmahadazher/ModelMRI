"""Token attribution must be a measurement, not a highlighted sentence.

This one is read as "these are the words that mattered", which is a claim
about the user's own text and therefore the one they are least equipped to
doubt. Every test here guards one of the ways the list could be confidently
wrong: the mask changing something other than what it says, the position
arithmetic moving under it, a row that measures the mask's own geometry, or a
ranking taken at a position where there is nothing to rank.

Numbers quoted below were measured on this machine. The gpt2 ones are fp32 on
CPU, because that is what this file re-runs; the bf16/cuda values from the
same prompt are labelled where they differ, and they differ.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from modelmri import attribute  # noqa: E402

PROMPT = "The capital of France is"


# ------------------------------------------------- a model that can be checked


class Listener:
    """A model whose next-token logits are a sum over the keys still visible.

    Each visible key's contribution is scaled by a function of ITS POSITION,
    which is what makes this a model position attribution can be checked
    against at all. A bag of visible keys with no position term passes every
    position check trivially — it has no phase to shift — and the version of
    this class that had none let the mask-deriving failure below through.

    Positions come from `position_ids` when they are supplied, and otherwise
    from `attention_mask.cumsum(-1) - 1`, which is what an HF decoder does.
    Under an all-ones mask those two agree, which is exactly why the agreement
    check alone cannot separate a faithful model from `derives_positions`.

    Every call is recorded, because most of what this file has to prove is
    about the arguments a pass was given rather than the number it returned.
    """

    def __init__(
        self,
        weights,
        *,
        layers=2,
        leak=0.0,
        reads_positions=False,
        derives_positions=False,
    ):
        self.weights = weights  # [S, V]
        self.layers = layers
        self.leak = leak
        # Answers differently merely because position_ids were SUPPLIED. The
        # agreement check against a plain model(ids) catches this one.
        self.reads_positions = reads_positions
        # Throws the supplied position_ids away and derives its own from the
        # mask. Indistinguishable from a faithful model under an all-ones
        # mask, and the reversed-ordering probe is the only thing that sees it.
        self.derives_positions = derives_positions
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
                "ids_shape": tuple(input_ids.shape),
                "mask": None if attention_mask is None else attention_mask.clone(),
                "position_ids": position_ids,
                "output_attentions": output_attentions,
            }
        )
        visible = (
            torch.ones(seq)
            if attention_mask is None
            else attention_mask[0].to(torch.float32)
        )
        if position_ids is None or self.derives_positions:
            pos = visible.cumsum(0) - 1.0
        else:
            pos = position_ids[0].to(torch.float32)
        # Per-key, so the phase cannot cancel out of the sum: reversing the
        # ordering gives every key a different scale rather than reshuffling
        # the same set of addends.
        phase = torch.sin(0.9 * pos + 0.3)
        row = (self.weights * (visible * phase).unsqueeze(-1)).sum(0)
        if self.reads_positions and position_ids is not None:
            row = row + torch.linspace(0.0, 1.0, row.shape[0])
        logits = row.expand(1, seq, row.shape[0]).clone()
        fields = {"logits": logits}
        if output_attentions:
            fields["attentions"] = self._attentions(visible, seq)
        return type("Out", (), fields)()

    def _attentions(self, visible, seq):
        causal = torch.tril(torch.ones(seq, seq))
        weights = causal * visible.unsqueeze(0)
        # `leak` puts weight back into a masked column, which is exactly the
        # thing the mechanism check is supposed to notice.
        weights = weights + self.leak * causal * (1.0 - visible).unsqueeze(0)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)
        return tuple(weights.expand(1, 2, seq, seq).clone() for _ in range(self.layers))


def listener(seq: int = 6, vocab: int = 4, **kw) -> Listener:
    """Distinct per-token contributions, so every index gets its own score."""
    torch.manual_seed(0)
    return Listener(torch.randn(seq, vocab), **kw)


def run(model=None, *, seq=6, position=4, **kw) -> dict:
    model = model or listener(seq)
    ids = torch.arange(seq).unsqueeze(0)
    return attribute.rank_tokens(model, ids, position=position, **kw)


# ------------------------------------------------- what the mask may change


def test_masking_never_changes_the_shape_of_the_ids():
    """The alternative baseline — deleting the token — renumbers every
    position after it, and the ranking would be about the renumbering."""
    model = listener()
    out = run(model)
    assert out["baseline"] == "mask"
    assert {c["ids_shape"] for c in model.calls} == {(1, 6)}
    for call in model.calls:
        assert call["mask"] is None or tuple(call["mask"].shape) == (1, 6)


def test_each_masked_pass_switches_off_exactly_the_index_it_scores():
    model = listener()
    run(model)
    # base, floor, the plain comparison, then one pass per row.
    zeroed = [
        sorted((c["mask"][0] == 0).nonzero().flatten().tolist())
        for c in model.calls
        if c["mask"] is not None
    ]
    singles = [z for z in zeroed if len(z) == 1]
    assert sorted(s[0] for s in singles) == [0, 1, 2, 3, 3]  # rows, index 0, probe
    # and one joint pass over the three ranked rows
    assert [1, 2, 3] in zeroed


def test_the_same_position_ids_object_is_handed_to_every_scoring_pass():
    """Several HF decoders derive position_ids from
    `attention_mask.cumsum(-1) - 1`. Rebuilding the tensor per pass is how the
    RoPE phase of the whole suffix would end up billed to the masked token —
    silently, because the scores would still look like scores."""
    model = listener()
    run(model)
    supplied = [c["position_ids"] for c in model.calls if c["position_ids"] is not None]
    assert len(supplied) >= 9
    arange = torch.arange(6).unsqueeze(0)
    shared = [p for p in supplied if torch.equal(p, arange)]
    assert len(shared) == len(supplied) - 1
    assert all(p is shared[0] for p in shared)
    # Exactly one pass deliberately goes without: the plain model(ids) call
    # that the agreement check compares against.
    assert sum(1 for c in model.calls if c["position_ids"] is None) == 1
    # And exactly one is handed an ordering that is wrong on purpose.
    reversed_ = [p for p in supplied if not torch.equal(p, arange)]
    assert len(reversed_) == 1 and torch.equal(reversed_[0], arange.flip(-1))


def test_a_model_that_answers_differently_with_explicit_positions_is_refused():
    """Not a warning. If supplying position_ids moves the answer, every score
    below it contains that move, and no amount of labelling fixes it."""
    with pytest.raises(attribute.AttributionError, match="position semantics"):
        run(listener(reads_positions=True))


def test_a_model_that_derives_its_positions_from_the_mask_is_refused():
    """The failure this whole module is built around, and the one the
    agreement check above cannot see: under an all-ones mask
    `attention_mask.cumsum(-1) - 1` IS arange(S), so a model that throws the
    supplied position_ids away agrees with a plain forward pass exactly. Every
    score it then produced would be the suffix's position shift rather than
    the masked token — and before the reversed probe existed this model came
    back with floor 0.0, mask_verified True and a full ranking."""
    with pytest.raises(attribute.AttributionError, match="not reading them"):
        run(listener(derives_positions=True))


def test_a_model_with_no_position_dependence_at_all_is_refused_too():
    """Conservative on purpose. Such a model is safe from the re-phasing, but
    it is indistinguishable from the mask-deriving one from outside, and this
    file will not ship a guarantee it cannot check. The message says what
    would make it work."""
    model = listener()
    model.weights = torch.zeros(6, 4)
    model.weights[:, 1] = 1.0  # flat along the sequence: order cannot matter
    with pytest.raises(attribute.AttributionError, match="not reading them"):
        run(model)


# ------------------------------------------------- which indices are candidates


def test_tokens_after_the_position_are_never_candidates():
    """They contribute exactly zero under the causal mask. A zero row would
    read as "measured, and it did not matter"."""
    model = listener(seq=8)
    out = run(model, seq=8, position=4)
    assert [r["index"] for r in out["ranked"]] and all(
        r["index"] < 4 for r in out["ranked"]
    )
    for call in model.calls:
        if call["mask"] is not None:
            off = (call["mask"][0] == 0).nonzero().flatten().tolist()
            assert all(j < 4 for j in off)


def test_index_zero_is_reported_outside_the_order_and_labelled():
    """It is a sink: on gpt2 it scores 4.86309, and prepending <|endoftext|>
    holds index 0 at 4.76083 (2.1% away) while the word that was there falls
    to 0.46107 at index 1 — 10.5x. The score follows the position. Ranking it
    would put "The" at the top of the list of words that mattered."""
    out = run()
    assert out["index0"]["index"] == 0
    assert 0 not in [r["index"] for r in out["ranked"]]
    assert "sink" in out["index0"]["note"].lower()


def test_the_attribution_position_is_not_a_candidate():
    out = run(position=4)
    assert 4 not in [r["index"] for r in out["ranked"]]
    assert out["index0"]["index"] != 4


def test_a_position_with_nothing_but_the_sink_before_it_is_refused():
    with pytest.raises(attribute.AttributionError, match="attention sink"):
        run(position=1)


# ------------------------------------------------- how much of it was tested


def test_truncation_is_reported_rather_than_silent():
    out = run(seq=100, position=99, max_candidates=8)
    assert (out["n_tested"], out["n_candidates"]) == (8, 98)
    assert out["truncated"] is True
    assert out["coverage"].startswith("8 of 98 were tested")
    assert "not found unimportant" in out["coverage"]
    # The ones kept are the ones nearest the position, and nothing else.
    assert sorted(r["index"] for r in out["ranked"]) == list(range(91, 99))
    # And the window is reported, because a strip that marks only the tested
    # tokens leaves indices 1..90 looking exactly like tokens that were tested
    # and scored nothing. Without this the client cannot draw the difference.
    assert out["tested_span"] == [91, 99]


def test_a_complete_run_says_so_too():
    out = run()
    assert out["truncated"] is False
    assert out["n_tested"] == out["n_candidates"] == 3
    assert out["coverage"].startswith("3 of 3 were tested")


# ------------------------------------------------- the mechanism, checked once


def test_the_mask_is_verified_to_have_emptied_the_column():
    out = run()
    assert out["mask_verified"] is True
    assert out["max_residual_weight"] == 0.0
    assert "MECHANISM" in out["mask_check"]
    # It is one check of the masking, not a per-row certificate, and it runs
    # eager attention — so it must not be sold as the pass behind any KL.
    assert "not the pass that produced any KL" in out["mask_check"]


def test_attention_still_reaching_a_masked_column_fails_the_check():
    """If the mask does not actually empty the column, every KL above is
    measuring a partly-masked token and the list is wrong by an unknown
    amount. The run says so rather than dropping the flag."""
    out = run(listener(leak=0.5))
    assert out["mask_verified"] is False
    assert out["max_residual_weight"] > 0


def test_a_model_with_no_attention_weights_reports_an_unchecked_mechanism():
    class Blind(Listener):
        def _attentions(self, visible, seq):
            return ()

    out = run(Blind(listener().weights))
    assert out["mask_verified"] is False
    assert out["max_residual_weight"] is None
    assert "could not be checked" in out["mask_check"]


# ------------------------------------------------- what the answer says


def test_every_row_reports_what_happened_to_the_top_token():
    out = run(typed_span=(2, 4))
    for row in out["ranked"]:
        assert {"kl", "flips_top", "p_top_before", "p_top_after", "group"} <= set(row)
        assert 0.0 <= row["p_top_before"] <= 1.0
        assert 0.0 <= row["p_top_after"] <= 1.0
        assert row["group"] in ("typed", "template")


def test_the_typed_span_is_what_separates_your_words_from_the_template():
    """Without it, Qwen3's own '\\n' and 'assistant' are labelled as the user's
    words. Measured on Qwen3-0.6B (bf16/cuda, "The capital of France is"):
    those two score 6.24429 and 2.02161 while every typed token sits between
    3.1e-05 and 7.9e-05."""
    out = run(typed_span=(2, 4))
    groups = {r["index"]: r["group"] for r in out["ranked"]}
    assert groups[2] == groups[3] == "typed"
    assert groups[1] == "template"


def test_no_span_is_an_unknown_and_never_silently_becomes_your_words():
    """`runtime._user_span` returns None for "we could not locate your words"
    — a slow tokenizer, or a prompt that occurs twice in the templated text.
    Reusing "typed" for that put the chat template under a heading reading
    "what you typed" on exactly the models where the template outscores the
    user's words by four orders of magnitude."""
    out = run()
    assert out["typed_span"] is None
    assert all(r["group"] == "unknown" for r in out["ranked"])
    assert out["index0"]["group"] == "unknown"


def test_tokens_past_the_prompt_are_the_models_own_output_not_the_template():
    """Without n_prompt every generated token falls outside typed_span and is
    labelled "template" — on gpt2, which has no chat template at all.
    Measured there (bf16/cuda, "The capital of France is", 12 greedy tokens,
    attributing at index 16): 11 of the 15 ranked rows sit past the prompt,
    and the top row in the whole run is the model's own word."""
    out = run(seq=8, position=6, typed_span=(0, 4), n_prompt=4)
    groups = {r["index"]: r["group"] for r in out["ranked"]}
    assert [groups[i] for i in (1, 2, 3)] == ["typed"] * 3
    assert [groups[i] for i in (4, 5)] == ["generated"] * 2
    # `generated` wins over an absent span too: past the prompt there is no
    # user text left for the row to be inside of.
    unknown = run(seq=8, position=6, n_prompt=4)
    assert {r["group"] for r in unknown["ranked"]} == {"unknown", "generated"}


def test_the_ranking_is_sorted_and_the_sum_is_over_the_rows_shown():
    out = run()
    scores = [r["kl"] for r in out["ranked"]]
    assert scores == sorted(scores, reverse=True)
    assert out["sum_of_singles"] == pytest.approx(sum(scores), abs=1e-5)


def test_the_answer_says_the_scores_are_not_shares():
    """Measured: summing exactly the rows this list shows over-states one
    joint mask of those same rows by 1.82x on gpt2 and 1.58x on
    gemma-3-270m-it (bf16/cuda, "The capital of France is", last prompt
    token), while summing only the typed span under-states it by 0.35x on
    gemma. The direction is not even fixed, so no correction factor exists."""
    means = run()["means"].lower()
    assert "not shares" in means and "do not add up" in means
    assert "correction factor" in means


def test_an_unknown_baseline_is_refused_by_name():
    """No `substitute` baseline: gpt2 has pad=None and unk == bos == eos,
    Qwen3 has a pad and no unk or bos, gemma has a real <pad>. Three
    experiments under one word, and on gpt2 the substitute is a document
    boundary, which reverses the ranking."""
    with pytest.raises(attribute.AttributionError, match="unknown baseline"):
        run(baseline="substitute")


def test_a_batch_is_refused_because_the_floor_was_measured_unbatched():
    model = listener()
    with pytest.raises(attribute.AttributionError, match="unbatched"):
        attribute.rank_tokens(model, torch.zeros(2, 6, dtype=torch.long), position=4)


# ------------------------------------------------- control tokens


class FakeTokenizer:
    """A tokenizer whose scaffolding is spelled every way that matters."""

    def __init__(self, chat_template=None):
        self.all_special_ids = [0]
        self.additional_special_tokens = []
        self.chat_template = chat_template
        self.added_tokens_decoder = {
            1: _Added("<start_of_turn>", special=True),
            2: _Added("<think>", special=False),
            3: _Added("<div>", special=False),
        }

    def convert_tokens_to_ids(self, token):  # pragma: no cover - unused arm
        return -1

    def get_vocab(self):
        return {"<|im_start|>": 4, "hello": 5, "<div>": 3}


class _Added(str):
    def __new__(cls, text, *, special):
        self = super().__new__(cls, text)
        self.special = special
        return self


def test_the_detector_finds_the_templates_own_tokens_without_eating_html():
    """Both halves are measured. `<think>` (Qwen3 id 151667) is declared
    special nowhere and is only findable through the chat template, and it is
    what that model emits with p=0.999531 at the end of a templated prompt.
    Meanwhile the wider `^<\\|?.+\\|?>$` claims 6573 ids beyond gemma's 8
    declared specials, including <div>, <b>, <html>, <table> and <li>, which
    are ordinary content in that vocabulary."""
    found = attribute.control_token_ids(FakeTokenizer(chat_template="A<think>B"))
    assert {0, 1, 2, 4} <= found  # declared, added-special, template-named, pipe
    assert 3 not in found  # <div> is shaped like a tag and is not one
    # Without the template naming it, <think> is just an added token.
    assert 2 not in attribute.control_token_ids(FakeTokenizer())


def test_a_tokenizer_that_hides_its_special_tokens_does_not_kill_the_run():
    """The slow GPT2Tokenizer overrides __getattr__ and raises AttributeError
    for `additional_special_tokens`; a bare access killed the first run of the
    measurement script this file's numbers come from."""

    class Hostile(FakeTokenizer):
        def __getattr__(self, name):
            raise AttributeError(name)

    assert 0 in attribute.control_token_ids(Hostile())


def test_attributing_a_control_token_is_refused():
    """Qwen3 answers a templated prompt with <think> at p=0.999531. Ranking
    which of the user's words caused the template to do what the template
    always does is ranking noise."""
    model = listener()
    # Make the argmax at the attribution position id 3, and call 3 special.
    # The column varies along the sequence rather than being flat: with one
    # constant value the model's answer is a function of the SET of visible
    # positions and not their order, and the reversed-ordering probe — which
    # runs before this refusal — sees a position-blind model and refuses
    # first, for the wrong reason.
    model.weights = torch.zeros(6, 4)
    model.weights[:, 3] = torch.linspace(1.0, 2.0, 6)
    with pytest.raises(attribute.AttributionError, match="control token"):
        run(model, control_ids=attribute.control_token_ids(FakeTokenizer()) | {3})


# ------------------------------------------------- against the real model


@pytest.fixture(scope="module")
def gpt2():
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("gpt2")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            "gpt2", attn_implementation="eager"
        ).eval()
    except Exception as err:  # noqa: BLE001 - no weights here, nothing to test
        pytest.skip(f"gpt2 is not available: {err}")
    return tok, model


def test_an_all_ones_mask_reproduces_the_plain_logits(gpt2):
    """The floor is measured, not assumed. On gpt2 the two logit vectors are
    bit-identical (torch.equal) — checked here in fp32 on CPU, and separately
    in bf16 on cuda where it also holds, along with Qwen3-0.6B and
    gemma-3-270m-it. That is the argument for spending the pass rather than
    dropping it: it is what would notice a model whose positions come from the
    mask instead."""
    tok, model = gpt2
    ids = tok(PROMPT, return_tensors="pt").input_ids
    seq = int(ids.shape[1])
    with torch.no_grad():
        plain = model(ids).logits[0, -1]
        explicit = model(
            input_ids=ids,
            attention_mask=torch.ones((1, seq), dtype=torch.long),
            position_ids=torch.arange(seq).unsqueeze(0),
        ).logits[0, -1]
    from modelmri.ablate import distribution, kl_nats

    assert torch.equal(plain, explicit)
    assert kl_nats(distribution(plain), distribution(explicit)) == 0.0


def test_the_self_position_scores_far_below_the_list_it_is_kept_out_of(gpt2):
    """Excluding i == position is geometric — it is the only candidate whose
    own key is taken from its own query — but a refactor that started ranking
    it would want to be caught here. Measured on gpt2 fp32/CPU with the prompt
    "The capital of France is": the self position scores 0.04075 against a
    ranked maximum of 1.52739, 2.67% of it. In bf16 on cuda the same pair is
    0.06375 against 4.86309.

    This bound does NOT generalise and must not be turned into the reason for
    the rule: on Qwen3-0.6B the self position scores 6.24429 and is the
    LARGEST of all 13 candidates, and on gemma-3-270m-it 1.92183 against a max
    of 9.33529."""
    tok, model = gpt2
    ids = tok(PROMPT, return_tensors="pt").input_ids
    seq = int(ids.shape[1])
    position = seq - 1
    out = attribute.rank_tokens(
        model, ids, position=position, decode=lambda t: tok.decode([t])
    )
    assert position not in [r["index"] for r in out["ranked"]]

    from modelmri.ablate import distribution, kl_nats

    ones = torch.ones((1, seq), dtype=torch.long)
    pos_ids = torch.arange(seq).unsqueeze(0)
    with torch.no_grad():
        base = distribution(
            model(input_ids=ids, attention_mask=ones, position_ids=pos_ids).logits[
                0, position
            ]
        )
        self_mask = ones.clone()
        self_mask[0, position] = 0
        after = distribution(
            model(input_ids=ids, attention_mask=self_mask, position_ids=pos_ids).logits[
                0, position
            ]
        )
    self_kl = kl_nats(base, after)
    top = out["ranked"][0]["kl"]
    assert self_kl < 0.1 * top, (
        f"the self position scores {self_kl:.5f} against a ranked maximum of "
        f"{top:.5f}; it was 2.67% of it when this bound was measured"
    )


def test_the_real_ranking_carries_a_measured_floor_and_a_verified_mask(gpt2):
    """gpt2, fp32 on CPU, prompt "The capital of France is", attributing at
    the last prompt token: floor 0.0, 10 passes, ' France' 1.52739, ' capital'
    0.83529, ' of' 0.73057, and index 0 outside the list at 4.92884 — 3.23x
    the largest thing in it. In bf16 on cuda the same run gives 1.74563 and
    4.86309, a ratio of 2.79x; the two sets must not be quoted across each
    other."""
    tok, model = gpt2
    ids = tok(PROMPT, return_tensors="pt").input_ids
    out = attribute.rank_tokens(
        model,
        ids,
        position=int(ids.shape[1]) - 1,
        decode=lambda t: tok.decode([t]),
    )
    assert out["noise_floor_kl"] == 0.0
    # Ten, not nine: the reversed-position_ids probe is one of them.
    assert out["passes"] == 10
    assert out["mask_verified"] is True and out["max_residual_weight"] == 0.0
    assert [r["token"] for r in out["ranked"]] == [" France", " capital", " of"]
    assert out["ranked"][0]["kl"] == pytest.approx(1.52739, abs=2e-4)
    assert out["index0"]["kl"] == pytest.approx(4.92884, abs=2e-4)
    assert out["index0"]["kl"] > out["ranked"][0]["kl"] * 3
    assert out["target_token"] == " the"
