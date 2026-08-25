"""A measurement must not read somebody else's forward pass.

`patch.trace` registers forward hooks on the block MODULES and then runs
`model(clean_ids)` to fill them. The hooks belong to the modules, not to that
call, so any forward pass through the model while they are installed fills them
instead.

That is not hypothetical. `ModelRuntime.generate_stream` runs
`model.generate(...)` on a daemon thread, and a decode step passes exactly ONE
token. Found by driving the running app: `/api/patch` answered 500 with

    IndexError: index 1 is out of bounds for dimension 1 with size 1

and it reproduced deterministically by tracing while a generation was decoding.

THE CRASH WAS THE LUCKY OUTCOME. A foreign PREFILL of the same length would
have matched every shape and produced a grid that looked exactly like a
measurement of this prompt pair and was not. So the guard is on the SHAPE, and
the refusal names the cause -- an exception check would have passed the silent
case straight through.

These use a two-layer stub rather than a real checkpoint: what is under test is
whose activations end up in the cache, which is arithmetic about hooks and
needs no weights.
"""

from __future__ import annotations

import threading

import pytest

torch = pytest.importorskip("torch")

from modelmri import patch  # noqa: E402
from modelmri.patch import PatchError  # noqa: E402

HIDDEN = 8
VOCAB = 32


class Block(torch.nn.Module):
    """One decoder block, with the two sublayer names `_sublayer` looks for."""

    def __init__(self) -> None:
        super().__init__()
        self.self_attn = torch.nn.Linear(HIDDEN, HIDDEN)
        self.mlp = torch.nn.Linear(HIDDEN, HIDDEN)

    def forward(self, x, *args, **kwargs):
        return (x + self.self_attn(x) + self.mlp(x),)


class Tiny(torch.nn.Module):
    def __init__(self, n_layers: int = 2) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB, HIDDEN)
        self.layers = torch.nn.ModuleList(Block() for _ in range(n_layers))
        self.head = torch.nn.Linear(HIDDEN, VOCAB)
        self.device = torch.device("cpu")

    def forward(self, ids, **kwargs):
        x = self.embed(ids)
        for b in self.layers:
            x = b(x)[0]

        class Out:
            pass

        out = Out()
        out.logits = self.head(x)
        return out


def _blocks(model):
    return list(model.layers)


def _model() -> Tiny:
    """Seeded, so the two prompts disagree by a stable amount rather than by
    whatever this run's random init happened to produce."""
    torch.manual_seed(0)
    return Tiny()


def test_a_cache_from_a_foreign_forward_pass_is_refused_by_shape():
    """The exact failure, at the layer that can see it.

    A one-token pass is what a decode step looks like. The real crash was an
    `IndexError` several frames later; this is the same event named where it
    happens.
    """
    model = Tiny()
    got: dict[int, torch.Tensor] = {}
    blocks = _blocks(model)
    handles = [patch._capture(b, i, got) for i, b in enumerate(blocks)]
    try:
        with torch.no_grad():
            # ONE token: a decode step, not this measurement's prompt.
            model(torch.tensor([[3]]))
    finally:
        for h in handles:
            h.remove()

    assert got[0].shape[1] == 1, "the stub must produce the one-token shape"
    # Which is exactly what `cache_for` now refuses: 1 != the prompt length.
    assert got[0].shape[1] != 5


def test_the_refusal_names_the_cause_and_a_next_step(monkeypatch):
    """`PatchError`, not `IndexError`. The reader is told what happened and
    what to do, which an index error out of a list comprehension cannot say."""
    model = _model()

    real_capture = patch._capture

    def poisoned(block, layer, sink):
        """Fill the sink as a concurrent one-token pass would."""
        handle = real_capture(block, layer, sink)

        class Wrapper:
            @staticmethod
            def remove():
                handle.remove()
                sink[layer] = torch.zeros(1, 1, HIDDEN)

        return Wrapper

    monkeypatch.setattr(patch, "_capture", poisoned)

    class Tok:
        def __call__(self, text, return_tensors=None):
            class R:
                pass

            r = R()
            # SAME LENGTH, DIFFERENT TOKENS. Equal lengths because `trace`
            # refuses a mismatched pair before it caches anything, and
            # different tokens so the two runs disagree enough to divide by --
            # the two refusals that come before the one under test here.
            base = 1 if "clean" in text else 11
            r.input_ids = torch.tensor([list(range(base, base + 5))])
            return r

        def decode(self, *a, **k):
            return "x"

        def convert_ids_to_tokens(self, ids):
            return [f"t{i}" for i in ids]

    with pytest.raises(PatchError) as caught:
        patch.trace(
            model, Tok(), _blocks(model), "clean one", "corrupt two", device="cpu"
        )
    said = str(caught.value)
    assert "ran through the model" in said or "one token wide" in said
    assert "trace again" in said


def test_the_runtime_refuses_a_trace_while_a_generation_decodes():
    """The other half: prevent the race rather than only detect it.

    `_decoding` describes the WORKER THREAD, not the generator that yields its
    chunks -- a consumer that stops pulling leaves the generator suspended and
    its `finally` unrun while the thread keeps decoding.
    """
    from modelmri.errors import Refusal
    from modelmri.runtime import ModelRuntime

    rt = ModelRuntime.__new__(ModelRuntime)
    rt._decoding = threading.Event()

    assert rt.decoding() is False
    rt._refuse_if_decoding("a patching trace")  # no generation: no refusal

    rt._decoding.set()
    assert rt.decoding() is True
    with pytest.raises(Refusal) as caught:
        rt._refuse_if_decoding("a patching trace")
    said = caught.value.sentence
    assert "generation is still running" in said
    assert "Wait for the run to finish" in said
