"""The copy onto the accelerator, and the meter it now keeps.

`model.to(device)` was one opaque call reporting nothing, and on this
project's own machine it is where two thirds of a load's wall clock goes:
measured on `meta-llama/Llama-3.2-1B-Instruct` fully cached, 7.1 s inside
`from_pretrained` and 15.36 s moving 2,471,629,056 bytes onto an RTX 4060
Laptop. Safetensors are memory-mapped, so the downloaded file is not actually
read until something touches the tensors — which is why the step that looks
like a pointer copy is the step that takes the minutes, and why it is the
step a reader most needs a number for.

`meta` is the device under test because it needs no hardware. A move onto it
is a real `Module.to` traversal that every runner can run, and `is_meta` is
an unambiguous answer to "did this tensor actually go anywhere".
"""

from itertools import chain

import pytest

torch = pytest.importorskip("torch")

from modelmri import device_move, runtime  # noqa: E402  (after the torch guard)

META = torch.device("meta")


class Tiny(torch.nn.Module):
    """Two layers, a buffer and a tied weight — the three things the
    accounting has to get right, in the smallest model that has all of them.
    """

    def __init__(self, tie: bool = True) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(16, 8)
        self.mid = torch.nn.Linear(8, 8)
        self.head = torch.nn.Linear(8, 16, bias=False)
        if tie:
            self.head.weight = self.embed.weight
        self.register_buffer("scale", torch.ones(8))


def _placement(model) -> dict:
    """Where every tensor of a model is, and what it is."""
    return {
        name: (str(t.device), tuple(t.shape), str(t.dtype))
        for name, t in chain(model.named_parameters(), model.named_buffers())
    }


def test_it_leaves_the_model_exactly_where_module_to_would():
    """The equivalence the whole change rests on.

    Walking the modules by hand to count bytes is only safe if the end state
    is the one `Module.to` produces. Asserted against `Module.to` itself
    rather than against a description of it.
    """
    by_torch, by_us = Tiny(), Tiny()
    by_torch.to(META)
    runtime.move_to_device(by_us, META)

    assert _placement(by_us) == _placement(by_torch)
    assert all(t.is_meta for t in by_us.parameters())
    assert all(t.is_meta for t in by_us.buffers())


def test_it_ties_and_unties_exactly_where_module_to_does():
    """Whether a tie survives a move is torch's business, not ours.

    `Module.to` rewrites `param.data` in place only when the source and
    destination types are shallow-copy compatible, and otherwise installs a
    new `Parameter`. Measured on torch 2.11: cpu->cuda is compatible and the
    tie survives, cpu->meta is not and it does not. So the claim worth
    asserting is that this walk does whatever `Module.to` does, on whichever
    device the test can reach — not that ties always live, which is false, or
    that they always die, which is also false.
    """
    by_torch, by_us = Tiny(), Tiny()
    assert by_torch.head.weight is by_torch.embed.weight
    by_torch.to(META)
    runtime.move_to_device(by_us, META)

    tied_after_torch = by_torch.head.weight is by_torch.embed.weight
    tied_after_us = by_us.head.weight is by_us.embed.weight
    assert tied_after_us is tied_after_torch


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="a tie only survives a real device move"
)
def test_a_tied_weight_survives_a_real_device_move():
    """The case that actually ships.

    Untying an embedding from its output head would silently double a model's
    resident memory, and nothing about the load would look wrong — the answer
    would still be right, on a card with less room left than the tool said it
    had. Skipped where there is no accelerator, because `meta` cannot show it:
    torch breaks the tie there itself.
    """
    model = Tiny()
    runtime.move_to_device(model, torch.device("cuda"))
    assert model.head.weight is model.embed.weight
    assert model.head.weight.is_cuda


def test_the_denominator_counts_a_tied_weight_once():
    """The bar has to be able to reach its own end.

    `parameters()` de-duplicates, so a tied weight is one entry — which is
    also what the move actually costs. 16x8 embedding + 8x8 + 8 bias + 8
    buffer, at 4 bytes each: 512 + 256 + 32 + 32.
    """
    assert device_move.resident_bytes(Tiny(tie=True)) == 832
    # And untying really does add exactly the head's own copy back, so the
    # de-duplication above is doing something rather than agreeing by luck.
    assert device_move.resident_bytes(Tiny(tie=False)) == 832 + 16 * 8 * 4


def test_it_publishes_while_it_works_and_not_only_at_the_end(monkeypatch):
    """A number that arrives once the wait is over is not progress."""
    seen: list[tuple] = []

    def record(**kw):
        seen.append((kw.get("bytes_done"), kw.get("bytes_total")))
        return True

    monkeypatch.setattr(runtime.progress.TRACKER, "publish", record)
    # PATCHED ON `device_move`, NOT ON `runtime`, AND THE DIFFERENCE IS THE
    # WHOLE TEST. `move_to_device` moved to a leaf module to break the
    # package's one import cycle, and `runtime` re-exports the name — so
    # `setattr(runtime, "DEVICE_PUBLISH_EVERY_S", 0.0)` now rebinds a second
    # reference that the function never reads. It would still pass on the
    # publish at the start and the one at the end, and stop checking the thing
    # it exists for: that a move publishes WHILE it works. `assert len(seen) >
    # 2` is what would have caught it, and only just.
    monkeypatch.setattr(device_move, "DEVICE_PUBLISH_EVERY_S", 0.0)

    model = Tiny()
    total = device_move.resident_bytes(model)  # read before the move, like the code
    moved = runtime.move_to_device(model, META)

    assert moved == total
    assert len(seen) > 2, seen
    done = [d for d, _ in seen]
    assert done[0] == 0, "the bar starts empty rather than wherever it was"
    assert done == sorted(done), "a meter that goes backwards is worse than none"
    assert done[-1] == total, "and it reaches the end it promised"
    assert {t for _, t in seen} == {total}, "one denominator throughout"


def test_the_denominator_never_ends_up_smaller_than_the_numerator(monkeypatch):
    """Published bytes are clamped.

    De-duplication is by parameter identity, and torch has a switch
    (`overwrite_module_params_on_conversion`) that makes `.to()` replace the
    `Parameter` instead of rewriting it. Nothing here sets it, but if anything
    ever did, a tied weight would be counted twice — and a bar drawn past its
    own end is the failure this project keeps finding, so it is guarded rather
    than argued about.
    """
    seen: list[tuple] = []

    def record(**kw):
        seen.append((kw.get("bytes_done"), kw.get("bytes_total")))
        return True

    monkeypatch.setattr(runtime.progress.TRACKER, "publish", record)
    runtime.move_to_device(Tiny(), META)
    assert all(done <= total for done, total in seen), seen


def test_a_model_with_nothing_to_move_says_nothing(monkeypatch):
    """Zero is not a denominator. An empty module still has to move — it just
    has no meter to keep, and publishing 0/0 would draw an indeterminate bar
    over a step that finished instantly."""
    seen: list[tuple] = []

    def record(**kw):
        seen.append((kw.get("bytes_done"), kw.get("bytes_total")))
        return True

    monkeypatch.setattr(runtime.progress.TRACKER, "publish", record)

    empty = torch.nn.Module()
    assert runtime.move_to_device(empty, META) == 0
    assert seen == []


def test_a_model_whose_tensors_cannot_be_sized_still_moves(monkeypatch):
    """A load must never fail because the thing measuring it did.

    Real checkpoints wrap their weights — quantised parameters, sharded ones,
    custom `Parameter` subclasses — and any of them may decline
    `element_size()`. The answer to that is an indeterminate bar, not a load
    that dies on the reader's first click. Found by the suite rather than by
    argument: a stub model in `test_smoke.py` raised `AttributeError` here and
    took a whole load down with it.
    """
    published: list = []

    def record(**kw):
        published.append(kw)
        return True

    monkeypatch.setattr(runtime.progress.TRACKER, "publish", record)

    class Unsizable:
        """Answers `parameters()` with something that is not a tensor."""

        def __init__(self) -> None:
            self.moved_to = None

        def parameters(self):
            return [object()]

        def buffers(self):
            return []

        def to(self, device):
            self.moved_to = device
            return self

    class NoBuffers(Unsizable):
        """The shape that actually broke a load: no `buffers()` at all, so the
        walk raises before it reaches a single tensor."""

        def parameters(self):
            return []

        buffers = None  # type: ignore[assignment]

    for shape in (Unsizable(), NoBuffers()):
        assert device_move.resident_bytes(shape) == 0, type(shape).__name__
        assert runtime.move_to_device(shape, META) == 0
        assert shape.moved_to is META
    published.clear()

    model = Unsizable()
    assert device_move.resident_bytes(model) == 0, "unknown, not a partial sum"
    assert runtime.move_to_device(model, META) == 0
    assert model.moved_to is META, "the move still has to happen"
    assert published == [], "and nothing is claimed about how far along it got"


def test_one_unsizable_tensor_makes_the_whole_answer_unknown():
    """A partial denominator is worse than none: the bar would stop short of
    its own end, which is a meter that never finishes — the exact complaint
    this change started from."""

    class HalfKnown(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.real = torch.nn.Linear(8, 8)

        def parameters(self, recurse: bool = True):
            return [*super().parameters(recurse=recurse), object()]

    assert device_move.resident_bytes(HalfKnown()) == 0


def test_buffers_are_moved_and_counted_like_parameters():
    """A buffer is not a parameter and `parameters()` does not yield it. Left
    out of the walk it would stay on the host, and a model half on each device
    fails at the first forward pass with a device-mismatch error that names
    neither the buffer nor the load."""
    model = Tiny()
    assert device_move.resident_bytes(model) == 832  # the buffer's 32 included
    assert runtime.move_to_device(model, META) == 832
    assert model.scale.is_meta
