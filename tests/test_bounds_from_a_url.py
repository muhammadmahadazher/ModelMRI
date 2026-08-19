"""A number in a query string may not reach torch unchecked.

Every case here was reproduced against a live Qwen model before the guard
existed, and each came back as an error about somebody else's library:

    GET /api/attention?layer=5&head=0&variant=ablate:9999.0
        -> 500  IndexError: index 9999 is out of range   (torch ModuleList)
    GET /api/attention/types?seq_len=24&n_sequences=0
        -> 500  RuntimeError: stack expects a non-empty TensorList
    GET /api/attention/types?seq_len=-4&n_sequences=4
        -> 500  DefaultCPUAllocator: you tried to allocate 735830067168 bytes
    POST /api/probe {"examples": []}
        -> 500  RuntimeError: stack expects a non-empty TensorList
    POST /api/patchscope {"prompt":"hi","max_new_tokens":-1}
        -> 500 after 180 seconds

The fourth line is the one worth keeping in view: a negative sequence length
in a URL asked the process for 735 GB of RAM.

These are unit tests against the guards rather than route tests, because the
crashes need a model resident and the guards are where the true bound is
known — `_block` is the only thing that knows how many blocks this checkpoint
has, and `label_heads` is the only thing that knows what its sampler needs.
"""

from __future__ import annotations

import pytest

from modelmri import head_types
from modelmri.errors import BadRequest


class _FakeBlocks(list):
    """Stands in for a torch ModuleList, which is what raised the IndexError."""


def test_a_layer_outside_the_model_is_refused_by_name():
    """`blocks[9999]` raised IndexError from inside torch's container, at
    somebody who typed a number into a URL."""
    from modelmri import runtime as runtime_mod

    handle = runtime_mod.ModelRuntime.__new__(runtime_mod.ModelRuntime)
    blocks = _FakeBlocks(["b0", "b1", "b2"])

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(runtime_mod, "decoder_blocks", lambda _m: blocks)
        handle.model = object()

        assert handle._block(0) == "b0"
        assert handle._block(2) == "b2"

        with pytest.raises(BadRequest) as err:
            handle._block(9999)
        assert "9999" in str(err.value) and "3 of them" in str(err.value)

        # Negative indices are refused rather than quietly wrapping. Python
        # would read -1 as the LAST block, so a reader who meant "unset"
        # would silently measure the top of the network and be told nothing.
        with pytest.raises(BadRequest):
            handle._block(-1)

        # `isinstance(True, int)` is True.
        with pytest.raises(BadRequest):
            handle._block(True)


@pytest.mark.parametrize("n_sequences", [0, -1])
def test_labelling_heads_needs_at_least_one_sequence(n_sequences):
    with pytest.raises(BadRequest) as err:
        head_types.label_heads(object(), object(), seq_len=24, n_sequences=n_sequences)
    assert str(n_sequences) in str(err.value)


@pytest.mark.parametrize("seq_len", [0, -4, 3])
def test_a_sequence_too_short_to_repeat_is_refused(seq_len):
    """`seq_len=0` was ALREADY answered properly, further down, with "there
    were no positions to score". `seq_len=-4` reached the allocator and asked
    for 735 GB. One question, two answers, decided by the sign."""
    with pytest.raises(BadRequest) as err:
        head_types.label_heads(object(), object(), seq_len=seq_len, n_sequences=4)
    assert str(seq_len) in str(err.value)


def test_the_minimum_is_stated_not_guessed():
    assert head_types.MIN_SEQ_LEN >= 2
