"""The `.mri` format: what it preserves, what it loses, and what it refuses.

The failure this guards against is not a crash. It is a session that opens
cleanly and shows a heat map that is subtly not the one the sender saw --
transposed, mis-scaled, or attributed to the wrong head. Every test here
compares against the numbers that went in.
"""

from __future__ import annotations

import gzip
import json

import pytest

from modelmri import session


def _matrix(n: int, *, seed: int = 0) -> list[list[float]]:
    """A causal, row-normalised attention matrix -- the real shape of one."""
    rows = []
    for r in range(n):
        raw = [((seed + r * 7 + c * 13) % 17) + 1.0 for c in range(r + 1)]
        raw += [0.0] * (n - r - 1)
        total = sum(raw)
        rows.append([v / total for v in raw])
    return rows


def _build(**over) -> bytes:
    args = dict(
        model_id="gpt2",
        device="cuda:0",
        dtype="float16",
        n_params=124_000_000,
        tokens=["The", " cat", " sat"],
        prompt="The cat",
        generation=" sat",
        attention={(0, 0): _matrix(3)},
        n_layers=1,
        n_heads=1,
    )
    args.update(over)
    return session.build(**args)


# ----------------------------------------------------------- round trip


def test_round_trip_keeps_what_the_panels_read():
    got = session.parse(_build())
    assert got.tokens == ["The", " cat", " sat"]
    assert got.prompt == "The cat"
    assert got.generation == " sat"
    assert got.meta["model"] == "gpt2"
    assert got.meta["n_params"] == 124_000_000
    assert got.attention_meta() == {
        "available": True,
        "n_layers": 1,
        "n_heads": 1,
        "n_tokens": 3,
        "replay": True,
    }


def test_matrix_survives_quantisation_within_the_stated_error():
    """uint8 against the matrix max: error must stay under one step."""
    original = _matrix(12)
    got = session.parse(
        _build(
            tokens=[f"t{i}" for i in range(12)],
            attention={(3, 2): original},
            n_layers=6,
            n_heads=4,
        )
    )
    matrix = got.attention_slice(3, 2)["matrix"]

    assert len(matrix) == 12 and all(len(row) == 12 for row in matrix)
    peak = max(max(row) for row in original)
    step = peak / 255.0
    worst = max(
        abs(matrix[r][c] - original[r][c]) for r in range(12) for c in range(12)
    )
    assert worst <= step, f"{worst} exceeds one quantisation step ({step})"


def test_orientation_is_not_transposed():
    """A causal matrix is lower-triangular. A transpose would still 'work'."""
    got = session.parse(
        _build(
            tokens=[f"t{i}" for i in range(8)],
            attention={(0, 0): _matrix(8)},
        )
    )
    matrix = got.attention_slice(0, 0)["matrix"]
    for r in range(8):
        for c in range(r + 1, 8):
            assert matrix[r][c] == 0.0, f"row {r} attends forward to {c}"
    assert matrix[7][0] > 0.0


def test_each_head_keeps_its_own_matrix():
    """The bug that reads plausibly: every head showing head 0's numbers."""
    got = session.parse(
        _build(
            tokens=[f"t{i}" for i in range(6)],
            attention={(0, 0): _matrix(6, seed=0), (1, 2): _matrix(6, seed=5)},
            n_layers=2,
            n_heads=3,
        )
    )
    assert got.attention_slice(0, 0)["matrix"] != got.attention_slice(1, 2)["matrix"]
    assert got.attention_slice(1, 2)["layer"] == 1
    assert got.attention_slice(1, 2)["head"] == 2


def test_rows_still_sum_to_about_one():
    got = session.parse(
        _build(tokens=[f"t{i}" for i in range(10)], attention={(0, 0): _matrix(10)})
    )
    for row in got.attention_slice(0, 0)["matrix"]:
        assert 0.97 <= sum(row) <= 1.03


def test_an_all_zero_matrix_does_not_divide_by_zero():
    got = session.parse(_build(attention={(0, 0): [[0.0] * 3 for _ in range(3)]}))
    assert got.attention_slice(0, 0)["matrix"] == [[0.0] * 3 for _ in range(3)]


def test_torch_and_python_paths_agree():
    """The tensor fast path must not quantise differently from the slow one."""
    torch = pytest.importorskip("torch")
    original = _matrix(9)
    from_lists = session.parse(
        _build(tokens=[f"t{i}" for i in range(9)], attention={(0, 0): original})
    ).attention_slice(0, 0)["matrix"]
    from_tensor = session.parse(
        _build(
            tokens=[f"t{i}" for i in range(9)],
            attention={(0, 0): torch.tensor(original, dtype=torch.float32)},
        )
    ).attention_slice(0, 0)["matrix"]
    assert from_tensor == from_lists


# ----------------------------------------------------------- honesty


def test_the_file_admits_the_precision_it_lost():
    meta = session.parse(_build()).meta
    assert "uint8" in meta["precision"]


def test_a_scoped_export_says_it_is_scoped():
    meta = session.parse(_build(scope="every layer at head 3")).meta
    assert meta["scope"] == "every layer at head 3"


def test_the_file_carries_no_weights():
    """The whole premise: shareable without shipping the model."""
    raw = gzip.decompress(_build())
    doc = json.loads(raw)
    assert set(doc) == {
        "format",
        "format_version",
        "created_at",
        "modelmri",
        "meta",
        "prompt",
        "generation",
        "tokens",
        "n_layers",
        "n_heads",
        "attention",
        "lens",
    }


# ----------------------------------------------------------- refusals


def test_a_missing_slice_says_what_it_has():
    got = session.parse(_build(attention={(0, 0): _matrix(3)}, n_layers=4, n_heads=4))
    with pytest.raises(session.SessionError) as err:
        got.attention_slice(2, 1)
    assert "layer 2 head 1" in str(err.value)
    assert "0:0" in str(err.value)


def test_a_future_format_version_tells_you_to_upgrade():
    doc = json.loads(gzip.decompress(_build()))
    doc["format_version"] = session.FORMAT_VERSION + 1
    with pytest.raises(session.SessionError, match="pip install -U modelmri"):
        session.parse(gzip.compress(json.dumps(doc).encode()))


def test_someone_elses_gzip_is_refused_by_name():
    junk = gzip.compress(json.dumps({"hello": "world"}).encode())
    with pytest.raises(session.SessionError, match="not a ModelMRI session"):
        session.parse(junk)


def test_a_truncated_matrix_is_caught_not_reshaped():
    """Silently reshaping short data would draw a real-looking wrong map."""
    doc = json.loads(gzip.decompress(_build()))
    doc["tokens"] = doc["tokens"] + ["extra"]  # claims 4x4, holds 3x3
    got = session.parse(gzip.compress(json.dumps(doc).encode()))
    with pytest.raises(session.SessionError, match="truncated"):
        got.attention_slice(0, 0)


@pytest.mark.parametrize(
    "data", [b"", b"not gzip at all", gzip.compress(b"\xff\xfe not utf8")]
)
def test_garbage_gets_a_reason_not_a_traceback(data):
    with pytest.raises(session.SessionError):
        session.parse(data)


def test_an_uncompressed_session_still_opens():
    """Some transports gunzip on the way through; do not punish the user."""
    raw = gzip.decompress(_build())
    assert session.parse(raw).tokens == ["The", " cat", " sat"]
