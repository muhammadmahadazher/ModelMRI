"""The logit lens's last row must be what the model actually said.

That row is the only one a reader can check, and it anchors everything else the
panel reports — `final`, `settled_at`, and the whole agreement column are all
taken from it. If it is wrong, every row above it is being compared against the
wrong answer.

It is easy to get wrong in a way that looks fine. HuggingFace decoders apply
the final norm inside the forward pass and record the result, so
`hidden_states[-1]` is ALREADY normed and `head(hidden_states[-1])` reproduces
`logits` exactly. Normalising it again computes `head(norm(norm(h)))`, and a
norm with a learned scale is not idempotent — it strips the per-dimension
scaling the unembedding was trained to read. Measured, prompt "The Eiffel Tower
is located in the city of", float32:

    gpt2                  head(h) ' Paris'   head(norm(h)) ' the'
    google/gemma-3-270m-it head(h) ' Paris'   head(norm(h)) ' pale'
    Qwen3-0.6B            head(h) ' Paris'   head(norm(h)) ' Paris'
    Qwen2.5-0.5B-Instruct head(h) ' Paris'   head(norm(h)) ' Paris'
    SmolLM2-360M-Instruct head(h) ' Paris'   head(norm(h)) ' Paris'

Two of five families give a confident, plausible, wrong token if the detection
is dropped, and three do not — so a change that removed it would pass a spot
check on Qwen and ship a broken lens for GPT-2 and Gemma. Hence a test rather
than a comment.

Synthetic models throughout, matching test_ablate.py: no weights are
downloaded, and both branches of the detection are exercised on purpose.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from modelmri.lens import _final_norm, logit_lens  # noqa: E402

D_MODEL = 16
VOCAB = 32
N_LAYERS = 3


class SkewedNorm(nn.Module):
    """LayerNorm with a deliberately uneven learned scale.

    The unevenness is the point: it makes the norm non-idempotent, so applying
    it twice is measurably different from applying it once. A norm with unit
    weight would hide the bug this file is about.
    """

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.linspace(0.2, 4.0, D_MODEL))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        centred = x - x.mean(-1, keepdim=True)
        return centred / (centred.std(-1, keepdim=True) + 1e-5) * self.weight


class _Out:
    def __init__(self, logits, hidden_states) -> None:
        self.logits = logits
        self.hidden_states = hidden_states


class FakeDecoder(nn.Module):
    """A causal LM shaped like the ones lens.py has to read.

    `last_is_normed=True` mimics HuggingFace: the forward applies the final
    norm and records the normed tensor as the last hidden state.
    `last_is_normed=False` mimics a layout that records the pre-norm stream,
    which the lens must then normalise itself.
    """

    def __init__(self, *, last_is_normed: bool) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.last_is_normed = last_is_normed
        self.embed = nn.Embedding(VOCAB, D_MODEL)
        self.blocks = nn.ModuleList(
            [nn.Linear(D_MODEL, D_MODEL) for _ in range(N_LAYERS)]
        )
        self.ln_f = SkewedNorm()
        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)

    def get_output_embeddings(self):
        return self.head

    def forward(self, ids, output_hidden_states: bool = False):
        hidden = [self.embed(ids)]
        for block in self.blocks:
            hidden.append(block(hidden[-1]))
        if self.last_is_normed:
            hidden[-1] = self.ln_f(hidden[-1])
            logits = self.head(hidden[-1])
        else:
            logits = self.head(self.ln_f(hidden[-1]))
        return _Out(logits, tuple(hidden))


class FakeTokenizer:
    def decode(self, ids) -> str:
        return f"<{int(ids[0])}>"


IDS = torch.tensor([[1, 2, 3, 4]])


def _truth(model) -> str:
    with torch.no_grad():
        out = model(IDS, output_hidden_states=True)
    return f"<{int(out.logits[0, -1].argmax())}>"


# ------------------------------------------------- the row anyone can check


@pytest.mark.parametrize("last_is_normed", [True, False])
def test_the_final_row_is_what_the_model_actually_said(last_is_normed):
    """Both layouts, one invariant. This is the whole contract."""
    model = FakeDecoder(last_is_normed=last_is_normed).eval()
    out = logit_lens(model, FakeTokenizer(), IDS, top_k=5)
    assert out["final"] == _truth(model), (
        f"lens says {out['final']}, model said {_truth(model)} "
        f"(last_is_normed={last_is_normed})"
    )


def test_double_normalising_would_change_the_answer():
    """The detection is load-bearing, not belt-and-braces.

    If this ever stops holding, the model above has become too gentle to catch
    the bug and the test is no longer protecting anything — which is worth
    failing over, because the next person would trust a green run.
    """
    model = FakeDecoder(last_is_normed=True).eval()
    with torch.no_grad():
        h = model(IDS, output_hidden_states=True).hidden_states[-1][:, -1, :]
        once = int(model.head(h)[0].argmax())
        twice = int(model.head(model.ln_f(h))[0].argmax())
    assert once != twice, (
        "normalising twice gave the same answer, so this fixture cannot detect "
        "the double-norm bug any more"
    )


def test_the_lens_reads_every_layer_plus_the_embedding():
    model = FakeDecoder(last_is_normed=True).eval()
    out = logit_lens(model, FakeTokenizer(), IDS, top_k=3)
    assert len(out["layers"]) == N_LAYERS + 1, "hidden_states includes the embedding"
    assert out["n_layers"] == N_LAYERS
    assert [row["layer"] for row in out["layers"]] == list(range(N_LAYERS + 1))
    for row in out["layers"]:
        assert len(row["tokens"]) == 3
        assert row["probs"] == sorted(row["probs"], reverse=True)
        assert row["entropy"] >= 0.0


def test_settled_at_is_the_last_unbroken_run_not_the_first_hit():
    """A layer that guesses right, changes its mind, then comes back must not
    be reported as the point the model decided."""
    model = FakeDecoder(last_is_normed=True).eval()
    out = logit_lens(model, FakeTokenizer(), IDS, top_k=1)
    final, settled = out["final"], out["settled_at"]
    tops = [row["tokens"][0] for row in out["layers"]]
    assert settled is not None
    # Everything from settled_at onward agrees, and the layer before it does not.
    assert all(t == final for t in tops[settled:])
    if settled > 0:
        assert tops[settled - 1] != final


# --------------------------------------------------------- refusing to guess


def test_a_model_with_no_recognisable_norm_is_refused():
    """Applying the wrong transform produces a plausible ranked list that
    describes nothing, so lens.py refuses instead of guessing."""

    class NoNorm(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = nn.Linear(D_MODEL, VOCAB, bias=False)

        def get_output_embeddings(self):
            return self.head

    with pytest.raises(RuntimeError, match="final norm"):
        _final_norm(NoNorm())


def test_the_norm_is_found_under_either_container():
    """Different families put it in different places; all are real layouts."""

    class Inner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = SkewedNorm()

    class LlamaShaped(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = Inner()

    class GPTShaped(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer = Inner()
            self.transformer.ln_f = SkewedNorm()

    assert _final_norm(LlamaShaped()) is not None
    assert _final_norm(GPTShaped()) is not None
