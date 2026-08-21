"""Refusals that named the wrong cause, for the likeliest real failure.

A refusal's whole value is that it tells somebody what to do next. One that
names a cause the file does not have sends them to fix the wrong thing, which
is worse than a generic error — they will believe it.
"""

from __future__ import annotations

import gzip
import tempfile
from pathlib import Path

import pytest

from modelmri import behavdiff, datasets, session
from modelmri.errors import BadRequest
from modelmri.session import SessionError


def test_a_truncated_mri_is_not_reported_as_too_big():
    """MEASURED: the first half of a genuine `.mri`, and a two-byte body, both
    answered "this file expands to more than 512 MB".

    `engine.eof` is false for two OPPOSITE reasons — the stream is still going
    and we stopped listening, or the input ran out mid-stream — and the
    message assumed the first. So the likeliest real failure of this route, an
    interrupted download, told the reader their file was too big and sent them
    to shrink something that was already incomplete. `unconsumed_tail` is the
    discriminator: it holds what we refused to read, and is empty when there
    was nothing left to consume.
    """
    whole = gzip.compress(b'{"hello": "world"}' * 200)

    for blob in (whole[: len(whole) // 2], b"\x1f\x8b", b""):
        with pytest.raises(SessionError) as err:
            session._inflate(blob)
        assert "incomplete" in str(err.value)
        assert "more than" not in str(err.value)

    # The oversize sentence still fires for something that IS oversize.
    with pytest.raises(SessionError) as err:
        session._inflate(gzip.compress(b"\0" * (session.MAX_INFLATED + 4096)))
    assert "more than" in str(err.value)

    # And a whole file still opens.
    assert session._inflate(whole)


def test_a_binary_file_where_jsonl_is_expected_says_so():
    """`UnicodeDecodeError` is a `ValueError`, not an `OSError`, so it escaped
    the `except OSError` arm one line away — while a MISSING file, guarded by
    that same arm, answered with a sentence. Same reader, two shapes of wrong
    file, two completely different experiences."""
    d = Path(tempfile.mkdtemp(prefix="mri-binary-"))
    binary = d / "weights.jsonl"
    binary.write_bytes(bytes(range(256)) * 4)

    for read in (datasets.read_dataset, datasets.read_experiment):
        with pytest.raises(BadRequest) as err:
            read(binary)
        assert "not text" in str(err.value)

    # The missing-file sentence still works, which is the arm this sits beside.
    with pytest.raises(BadRequest):
        datasets.read_dataset(d / "absent.jsonl")


def test_a_single_file_that_is_not_a_gguf_is_not_a_huggingface_model():
    """`side`'s docstring is "Classify a side by what it is, not by what the
    caller called it", and it classified README.md as `hf` — then handed it to
    `AutoTokenizer.from_pretrained`, so the reader got transformers' own "It
    looks like the config file at '…README.md' is not a valid JSON file"."""
    d = Path(tempfile.mkdtemp(prefix="mri-side-"))
    readme = d / "README.md"
    readme.write_text("# not a model\n", encoding="utf-8")

    with pytest.raises(BadRequest) as err:
        behavdiff.side(str(readme))
    assert "README.md" in str(err.value)
    assert ".gguf" in str(err.value)

    # A GGUF file, a directory, and a Hub id are all still sides.
    gguf = d / "q4.gguf"
    gguf.write_bytes(b"GGUF")
    assert behavdiff.side(str(gguf)).kind == "gguf"
    assert behavdiff.side("Qwen/Qwen3-1.7B").kind == "hf"

    # A directory HOLDING a gguf is a gguf side — that is the documented rule,
    # so the plain-directory case needs a directory of its own rather than
    # this one, which now contains the file written above.
    plain = Path(tempfile.mkdtemp(prefix="mri-plain-"))
    (plain / "config.json").write_text("{}", encoding="utf-8")
    assert behavdiff.side(str(plain)).kind == "hf"
    assert behavdiff.side(str(d)).kind == "gguf"
