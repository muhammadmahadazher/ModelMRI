# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

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
from modelmri.errors import Refusal


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


def test_the_prompt_boundary_survives_a_round_trip():
    """The panel rests on the last prompt token, so a shared session that
    loses the boundary opens on the blank canvas the resting state replaced —
    someone else's analysis, arriving worse than your own."""
    got = session.parse(_build(n_prompt=2))
    assert got.n_prompt == 2
    assert got.attention_meta()["n_prompt"] == 2
    assert got.attention_slice(0, 0)["n_prompt"] == 2


def test_a_boundary_outside_the_token_list_is_discarded():
    """`n_prompt` arrives in a file a stranger sent. A value past the end
    would mark generated tokens as prompt, or index off the matrix.

    Zero means "unknown" — every reader treats it as "rest on nothing",
    which is the safe reading. It must NOT mean "all prompt"."""

    for hostile in (99, -1, "3", True, None):
        raw = json.loads(gzip.decompress(_build()))
        raw["n_prompt"] = hostile
        blob = gzip.compress(json.dumps(raw).encode())
        assert session.parse(blob).n_prompt == 0, f"accepted {hostile!r}"


def test_an_older_session_without_a_boundary_still_opens():
    """The field is additive. Files written before it exists must not become
    unreadable, and must report unknown rather than a guess."""

    raw = json.loads(gzip.decompress(_build()))
    del raw["n_prompt"]
    blob = gzip.compress(json.dumps(raw).encode())
    parsed = session.parse(blob)
    assert parsed.n_prompt == 0
    assert parsed.tokens == ["The", " cat", " sat"]


def test_round_trip_keeps_what_the_panels_read():
    got = session.parse(_build())
    assert got.tokens == ["The", " cat", " sat"]
    assert got.prompt == "The cat"
    assert got.generation == " sat"
    assert got.meta["model"] == "gpt2"
    assert got.meta["n_params"] == 124_000_000
    assert got.attention_meta() == {
        "available": True,
        "n_prompt": 0,
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
        "n_prompt",
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
    "data",
    [
        b"",
        b"not gzip at all",
        # `mtime=0` and an explicit id. gzip writes the CURRENT TIME into its
        # header, so `gzip.compress(...)` evaluated at collection time produced
        # a different parameter — and therefore a different test id — on every
        # run. Serially that was invisible churn. Under xdist it is fatal:
        # each worker collects independently and the ids must agree, so the
        # whole suite died with "Different tests were collected between gw1
        # and gw3".
        #
        # A test whose identity changes between runs also cannot be selected
        # with `-k`, cannot be compared across CI runs, and made the suite's
        # own collected-test count look unstable when nothing had changed.
        pytest.param(
            gzip.compress(b"\xff\xfe not utf8", mtime=0), id="gzip-of-non-utf8"
        ),
    ],
)
def test_garbage_gets_a_reason_not_a_traceback(data):
    with pytest.raises(session.SessionError):
        session.parse(data)


# --------------------------------------------- bounds on somebody else's file
#
# `parse` takes bytes a stranger sent — that is the entire premise of the
# format. Every bound below was reachable before it existed.


def test_a_gzip_bomb_is_refused_rather_than_allocated():
    """3 MB of gzip that becomes 3 GB of memory. The server's 64 MB body cap
    did not help: it bounds the compressed side."""
    bomb = gzip.compress(b"\0" * (3 * 1024 * 1024 * 1024), 9)
    assert len(bomb) < 10_000_000, "the bomb should be small — that is the point"
    with pytest.raises(session.SessionError, match="expands to more than"):
        session.parse(bomb)


def test_an_enormous_token_count_is_refused_before_any_matrix_is_built():
    """Cost is n^2 per slice, so a small file can ask for a hundred million
    Python floats — and the identical loop runs in the recipient's browser."""
    doc = json.loads(gzip.decompress(_build()))
    doc["tokens"] = ["t"] * 20_000  # 400 million cells per map
    with pytest.raises(session.SessionError, match="attention cells"):
        session.parse(gzip.compress(json.dumps(doc).encode()))


def test_a_file_far_larger_than_any_session_is_refused_on_sight():
    big = b"\x00" * (session.MAX_FILE + 1)
    with pytest.raises(session.SessionError, match="almost certainly not one"):
        session.parse(big)


# `{1: {}}` is deliberately absent: JSON object keys are always strings, so
# a non-string key cannot survive a round trip and testing for it would be
# testing the json module.
@pytest.mark.parametrize("bad", [[], "x", 7, {"0:0": "not-a-dict"}, {"0:0": []}])
def test_a_malformed_attention_index_is_refused(bad):
    """It reached the panels unvalidated, and every later request 500'd."""
    doc = json.loads(gzip.decompress(_build()))
    doc["attention"] = bad
    with pytest.raises(session.SessionError, match="attention index"):
        session.parse(gzip.compress(json.dumps(doc).encode()))


@pytest.mark.parametrize("bad", [-1, 1e20, "12", 10**9, 1.5, True])
def test_nonsense_layer_and_head_counts_are_refused(bad):
    """These reach the UI as loop bounds. 1e20 is not a shape."""
    doc = json.loads(gzip.decompress(_build()))
    doc["n_layers"] = bad
    with pytest.raises(session.SessionError, match="sensible number"):
        session.parse(gzip.compress(json.dumps(doc).encode()))


def test_non_finite_attention_is_refused_rather_than_exported_as_zeros():
    """NaN loses every comparison, so the peak became NaN, the scale became
    NaN, and every cell quantised to 0 — a plausible blank heat map with
    nothing saying the numbers were never there."""
    nan = float("nan")
    with pytest.raises(Refusal, match="non-finite"):
        _build(attention={(0, 0): [[1.0, 0.0], [nan, 0.5]]}, tokens=["a", "b"])


def test_non_finite_attention_is_refused_on_the_tensor_path_too():
    torch = pytest.importorskip("torch")
    bad = torch.tensor([[1.0, 0.0], [float("inf"), 0.5]])
    with pytest.raises(Refusal, match="non-finite"):
        _build(attention={(0, 0): bad}, tokens=["a", "b"])


def test_an_uncompressed_session_still_opens():
    """Some transports gunzip on the way through; do not punish the user."""
    raw = gzip.decompress(_build())
    assert session.parse(raw).tokens == ["The", " cat", " sat"]


def _mri(doc: dict) -> bytes:
    # Both are already imported at the top of this file. The local copies were
    # shadowing them to no purpose.
    return gzip.compress(json.dumps(doc).encode("utf-8"))


def test_a_hand_made_session_cannot_put_its_own_text_in_the_error():
    """`parse` takes bytes a stranger sent, and the version check interpolated
    the value BEFORE establishing it was a number.

    Measured before the fix: a `.mri` carrying
    `"format_version": "<img src=x onerror=alert(1)>ATTACKER"` came back with
    that string verbatim in the 422 body, and a dict came back as a Python
    repr. Not an XSS — the body is JSON and React renders it as a text node —
    but it is attacker-supplied content reflected by the one function whose
    docstring says it takes bytes a stranger sent you.
    """
    from modelmri.session import SessionError, parse

    for hostile in (
        "<img src=x onerror=alert(1)>ATTACKER",
        {"a": [1, 2]},
        ["x"],
        None,
        True,  # bool is an int subclass, and would otherwise pass isinstance
        1.5,
    ):
        with pytest.raises(SessionError) as err:
            parse(_mri({"format": "modelmri-session", "format_version": hostile}))
        message = str(err.value)
        assert "ATTACKER" not in message, message
        assert "{" not in message and "[" not in message, message
        assert "does not say which format version" in message, message


def test_a_damaged_gzip_does_not_republish_zlibs_own_words():
    """errors.py forbids interpolating a caught exception's text into a
    published message. This one did: `could not decompress the file: Error -3
    while decompressing data: unknown compression method`. zlib's strings are
    harmless C literals, but OSError carries its `filename` when set — the
    exact shape that leaked absolute paths to the browser before."""
    from modelmri.session import SessionError, parse

    with pytest.raises(SessionError) as err:
        parse(b"\x1f\x8b" + b"\x00" * 40)
    message = str(err.value)
    assert "Error -3" not in message and "zlib" not in message.lower(), message
    assert "could not be decompressed" in message, message


def test_malformed_metadata_is_a_refusal_rather_than_a_500():
    """`meta` was spread with `**` and never type-checked, so a `.mri` carrying
    `"meta": "hi"` raised a bare `TypeError: 'str' object is not a mapping` —
    not a BadRequest, so it fell past the 409 and 422 arms to the generic 500.
    Every other untrusted field in that function is checked."""
    from modelmri.session import SessionError, parse

    for hostile in ("hi", ["a"], 7):
        with pytest.raises(SessionError):
            parse(
                _mri(
                    {
                        "format": "modelmri-session",
                        "format_version": 1,
                        "meta": hostile,
                    }
                )
            )


# ------------------------------------------------------- the lens section
#
# Carried in the format from the beginning and never validated, because
# nothing wrote it: `export_session` did not pass it, so `lens` was always []
# and the hole was invisible. It reaches the viewer as a table of tokens and a
# bar per probability.


def _lens_doc(rows, info=None):
    import gzip
    import json

    from modelmri import session as s

    raw = s.build(
        model_id="gpt2",
        device="cpu",
        dtype="float32",
        n_params=1,
        tokens=["a", "b"],
        prompt="a",
        generation="b",
        attention={(0, 0): [[1.0, 0.0], [0.5, 0.5]]},
        n_layers=1,
        n_heads=1,
    )
    doc = json.loads(gzip.decompress(raw))
    doc["lens"] = rows
    if info is not None:
        doc["lens_info"] = info
    return gzip.compress(json.dumps(doc).encode())


def test_a_lens_row_with_mismatched_tokens_and_probs_is_refused():
    """The panel zips them together; mismatched lengths render a token with
    somebody else's probability beside it."""
    import pytest

    from modelmri import session as s

    raw = _lens_doc([{"layer": 0, "tokens": ["a", "b"], "probs": [1.0]}])
    with pytest.raises(s.SessionError, match="cannot be read together"):
        s.parse(raw)


def test_a_non_finite_lens_probability_is_refused():
    import pytest

    from modelmri import session as s

    raw = _lens_doc([{"layer": 0, "tokens": ["a"], "probs": [float("nan")]}])
    with pytest.raises(s.SessionError, match="non-finite"):
        s.parse(raw)


def test_a_lens_layer_with_no_index_is_refused():
    import pytest

    from modelmri import session as s

    with pytest.raises(s.SessionError, match="no index"):
        s.parse(_lens_doc([{"tokens": ["a"], "probs": [1.0]}]))


def test_settled_at_none_survives_because_it_is_a_finding():
    """ "The answer never settles before the last layer" is a result. Coercing
    it to 0 would claim it settled immediately."""
    from modelmri import session as s

    parsed = s.parse(
        _lens_doc(
            [{"layer": 0, "tokens": ["a"], "probs": [1.0]}],
            {"settled_at": None, "final": " a"},
        )
    )
    assert parsed.lens_info["settled_at"] is None
    assert parsed.lens_info["final"] == " a"


def test_a_file_with_no_lens_reads_as_an_empty_one():
    from modelmri import session as s

    parsed = s.parse(_lens_doc([]))
    assert parsed.lens == []
    assert parsed.lens_info == {}


def test_an_attention_block_with_nothing_to_draw_is_refused_at_parse():
    """`parse` checked that each block IS a dict and stopped there, so
    `{"0:0": {"layer": 0}}` opened cleanly and then
    `_dequantise(block["q"], block["scale"], ...)` raised KeyError on the
    first request for that head — a 500, out of a file the reader had just
    been told opened.

    That is the failure the comment above this check already describes for
    the index itself, one level further in.
    """
    for damaged in ({"layer": 0}, {"q": "AAAA"}, {"scale": 0.004}, {}):
        raw = json.loads(gzip.decompress(_build()))
        key = next(iter(raw["attention"]))
        raw["attention"][key] = damaged
        blob = gzip.compress(json.dumps(raw).encode())
        with pytest.raises(session.SessionError) as caught:
            session.parse(blob)
        assert "cannot be drawn" in str(caught.value), f"accepted {damaged!r}"


def test_a_scale_that_is_not_a_number_is_refused():
    """`True` is an int in Python, and a bool scale would dequantise every
    cell to 0 or 1."""
    for hostile in (True, "0.004", None, [0.004]):
        raw = json.loads(gzip.decompress(_build()))
        key = next(iter(raw["attention"]))
        raw["attention"][key] = {"q": "AAAA", "scale": hostile}
        blob = gzip.compress(json.dumps(raw).encode())
        with pytest.raises(session.SessionError):
            session.parse(blob)


def test_an_intact_file_still_opens():
    """The guard must not turn a good file away."""
    parsed = session.parse(_build())
    assert parsed.attention
    assert parsed.attention_slice(0, 0)["matrix"]


# ------------------------------------- the tool version is carried, not parsed


def test_a_development_version_survives_a_round_trip_unchanged(monkeypatch):
    """`main` versions itself `0.13.0.dev0`, and nothing may read that as X.Y.Z.

    The version reaches two places a reader later trusts: `session.build`
    stamps it into the `.mri` header (`session.py:2593`) and `receipts` stamps
    it into every measurement (`receipts.py:426`). Both are declared `str` and
    neither is compared, sorted or split anywhere — `mri_diff` says so in
    prose, carrying the producer's numbers "and not judged", because deriving a
    verdict from one "would report a producer's version bump as a change in the
    model".

    That is a property nothing tested. A PEP 440 development version is the
    cheapest way to break it: any code that reached for the middle number, or
    compared two versions as tuples, fails on `0.13.0.dev0` and passes on
    `0.12.0`. So the assertion is byte equality after a real build/parse cycle,
    not a `startswith`.
    """
    import modelmri

    dev = "0.13.0.dev0"
    monkeypatch.setattr(modelmri, "__version__", dev)

    parsed = session.parse(_build())
    assert parsed.meta["modelmri"] == dev, (
        f"the .mri header carries {parsed.meta['modelmri']!r}, "
        "not the version that wrote it"
    )


def test_a_receipt_carries_a_development_version_verbatim(monkeypatch):
    """The other half: a receipt is what a reader checks a number against, and
    it states the tool that produced it. Same rule, same reason."""
    import modelmri
    from modelmri import receipts

    dev = "0.13.0.dev0"
    monkeypatch.setattr(modelmri, "__version__", dev)

    receipt = receipts.Receipt(op="test", tool_version=dev)
    assert receipt.to_dict()["tool_version"] == dev
