"""Reading a GGUF header, and refusing the ones that are not.

Two things carry the weight here. Byte counts must be computed from whole
BLOCKS — a k-quant stores 256 elements in 144 bytes, so `elements * bpw` is
wrong for every quantised tensor in every file this will ever open. And an
unknown ggml type must stay unknown: ggml adds types regularly, and a size
guessed for one silently corrupts every roll-up that includes it.

The parser also reads files from the internet, so running off the end of a
truncated header has to be a refusal naming the file rather than an IndexError
arriving at the browser as a 500.
"""

from __future__ import annotations

import struct

import pytest

from modelmri import gguf_read
from modelmri.errors import BadRequest, Refusal


def _s(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def build(tmp_path, *, metadata=None, tensors=(), version=3, magic=b"GGUF"):
    """A real GGUF header: magic, version, counts, KV pairs, tensor table."""
    metadata = metadata or {}
    body = b""
    for key, (kind, value) in metadata.items():
        body += _s(key) + struct.pack("<I", kind)
        if kind == gguf_read._STRING:
            body += _s(value)
        elif kind == gguf_read._UINT32:
            body += struct.pack("<I", value)
        elif kind == gguf_read._ARRAY:
            inner, items = value
            body += struct.pack("<I", inner) + struct.pack("<Q", len(items))
            for it in items:
                body += _s(it) if inner == gguf_read._STRING else struct.pack("<I", it)

    for name, ggml_type, dims, offset in tensors:
        body += _s(name) + struct.pack("<I", len(dims))
        for d in dims:
            body += struct.pack("<Q", d)
        body += struct.pack("<I", ggml_type) + struct.pack("<Q", offset)

    head = magic + struct.pack("<IQQ", version, len(tensors), len(metadata))
    path = tmp_path / "model.gguf"
    path.write_bytes(head + body + b"\0" * 64)
    return path


# ------------------------------------------------------------------ parsing


def test_reads_metadata_and_the_tensor_table(tmp_path):
    p = build(
        tmp_path,
        metadata={
            "general.architecture": (gguf_read._STRING, "llama"),
            "general.name": (gguf_read._STRING, "tiny"),
            "llama.block_count": (gguf_read._UINT32, 4),
            "llama.context_length": (gguf_read._UINT32, 2048),
        },
        tensors=[
            ("token_embd.weight", 1, [512, 1000], 0),  # F16
            ("blk.0.attn_q.weight", 12, [512, 512], 4096),  # Q4_K
        ],
    )
    g = gguf_read.read(p)
    assert g.version == 3 and g.tensor_count == 2
    assert g.metadata["general.architecture"] == "llama"
    assert g.metadata["llama.context_length"] == 2048
    assert [t.name for t in g.tensors] == ["token_embd.weight", "blk.0.attn_q.weight"]
    assert g.tensors[0].type_name == "F16"
    assert g.tensors[1].type_name == "Q4_K"


def test_byte_counts_come_from_whole_blocks_not_a_rate(tmp_path):
    """A k-quant packs 256 elements into 144 bytes. `elements * 4.5 / 8` gives
    the same answer only when the count divides the block exactly, and silently
    differs when it does not."""
    p = build(tmp_path, tensors=[("w", 12, [256, 4], 0)])  # Q4_K, 1024 elements
    t = gguf_read.read(p).tensors[0]
    assert t.elements == 1024
    assert t.bytes == (1024 // 256) * 144
    assert t.bpw == pytest.approx(4.5)


def test_f16_is_sixteen_bits_per_weight(tmp_path):
    p = build(tmp_path, tensors=[("w", 1, [10, 10], 0)])
    t = gguf_read.read(p).tensors[0]
    assert t.bytes == 200 and t.bpw == pytest.approx(16.0)


def test_q6_k_matches_its_known_rate(tmp_path):
    p = build(tmp_path, tensors=[("w", 14, [256], 0)])
    assert gguf_read.read(p).tensors[0].bpw == pytest.approx(210 * 8 / 256)


# ------------------------------------------------ an unknown type stays unknown


def test_an_unknown_ggml_type_reports_no_size(tmp_path):
    """ggml adds types regularly. A guessed size corrupts every roll-up that
    includes the tensor, confidently."""
    p = build(tmp_path, tensors=[("w", 199, [256], 0)])
    g = gguf_read.read(p)
    t = g.tensors[0]
    assert t.bytes is None and t.bpw is None
    assert "unknown" in t.type_name and "199" in t.type_name
    assert g.unknown_types == [199]


def test_parameters_count_every_tensor_even_unsized_ones(tmp_path):
    """Regression. `elements` is read from `dims` BEFORE the ggml type is
    consulted, so it is as known for an unknown type as for F32. Excluding
    those made a 1.44B model with MXFP4 bulk tensors report 131,072
    parameters — wrong by 11,009x — while only an unrelated
    `unmeasured_tensors` field dissented."""
    p = build(
        tmp_path,
        tensors=[("known", 1, [100], 0), ("mystery", 199, [100], 0)],
    )
    s = gguf_read.read(p).summary()
    assert s["parameters"] == 200  # both — shapes do not depend on the type
    assert s["measured_parameters"] == 100
    assert s["unmeasured_tensors"] == 1


def test_byte_totals_are_withheld_when_any_tensor_could_not_be_sized(tmp_path):
    """A partial average printed as the file's headline is the confidently
    wrong number this module's docstring calls worse than an absent one."""
    p = build(
        tmp_path,
        tensors=[("known", 1, [100], 0), ("mystery", 199, [100], 0)],
    )
    s = gguf_read.read(p).summary()
    assert s["tensor_bytes"] is None
    assert s["effective_bpw"] is None
    assert s["dominant_type"] is None
    assert s["by_type_covers_whole_file"] is False
    assert "199" in s["why_unmeasured"]


def test_a_fully_unknown_file_does_not_report_zero_parameters(tmp_path):
    """Rule 1: unknown must not collapse into 0. The old code reported
    `parameters: 0` and `tensor_bytes: 0` for a file it understood nothing
    about, in the same dict where effective_bpw correctly refused."""
    p = build(tmp_path, tensors=[("a", 199, [256, 8], 0), ("b", 201, [128], 0)])
    s = gguf_read.read(p).summary()
    assert s["parameters"] == 256 * 8 + 128
    assert s["measured_parameters"] == 0
    assert s["tensor_bytes"] is None
    assert s["effective_bpw"] is None


def test_a_fully_measured_file_still_reports_everything(tmp_path):
    p = build(tmp_path, tensors=[("a", 12, [256], 0), ("b", 1, [256], 0)])
    s = gguf_read.read(p).summary()
    assert s["by_type_covers_whole_file"] is True
    assert s["why_unmeasured"] is None
    assert s["tensor_bytes"] == 144 + 512
    assert s["dominant_type"] is not None


# ------------------------------------------------------------------ roll-ups


def test_the_headline_names_the_tensors_that_sit_above_it(tmp_path):
    """Every runner shows `Q4_K_M` and a file size. The thing worth knowing is
    that the embedding and output layers are usually left much higher."""
    p = build(
        tmp_path,
        metadata={"general.architecture": (gguf_read._STRING, "llama")},
        tensors=[
            ("blk.0.ffn_down.weight", 12, [256, 400], 0),  # Q4_K, big
            ("blk.1.ffn_down.weight", 12, [256, 400], 0),
            ("token_embd.weight", 1, [256, 20], 0),  # F16, higher precision
        ],
    )
    s = gguf_read.read(p).summary()
    assert s["dominant_type"] == "Q4_K"
    names = [o["name"] for o in s["higher_precision_tensors"]]
    assert "token_embd.weight" in names
    assert s["higher_precision_tensors"][0]["bpw"] == pytest.approx(16.0)


def test_effective_bpw_is_the_whole_file_not_the_dominant_type(tmp_path):
    p = build(
        tmp_path,
        tensors=[("a", 12, [256], 0), ("b", 1, [256], 0)],  # Q4_K + F16
    )
    s = gguf_read.read(p).summary()
    total_bytes = 144 + 512
    assert s["effective_bpw"] == pytest.approx(total_bytes * 8 / 512, abs=1e-3)
    assert s["by_type"]["Q4_K"]["bpw"] == pytest.approx(4.5)
    assert s["by_type"]["F16"]["bpw"] == pytest.approx(16.0)


def test_the_summary_says_what_bpw_means(tmp_path):
    p = build(tmp_path, tensors=[("a", 12, [256], 0)])
    assert "not the quantisation label" in gguf_read.read(p).summary()["means"]


def test_architecture_keyed_metadata_is_surfaced(tmp_path):
    p = build(
        tmp_path,
        metadata={
            "general.architecture": (gguf_read._STRING, "qwen2"),
            "qwen2.context_length": (gguf_read._UINT32, 32768),
            "qwen2.attention.head_count_kv": (gguf_read._UINT32, 2),
        },
        tensors=[("a", 12, [256], 0)],
    )
    s = gguf_read.read(p).summary()
    assert s["architecture"] == "qwen2"
    assert s["context_length"] == 32768
    assert s["head_count_kv"] == 2


# ------------------------------------------------------------------ refusals


def test_a_file_without_the_magic_is_refused(tmp_path):
    p = build(tmp_path, magic=b"NOPE", tensors=[("a", 1, [4], 0)])
    with pytest.raises(BadRequest, match="not a GGUF file"):
        gguf_read.read(p)


def test_an_unsupported_version_refuses_rather_than_guessing(tmp_path):
    p = build(tmp_path, version=9, tensors=[("a", 1, [4], 0)])
    with pytest.raises(Refusal, match="version 9"):
        gguf_read.read(p)


def test_a_truncated_header_is_refused_by_name(tmp_path):
    p = build(tmp_path, tensors=[("a", 1, [4], 0)])
    raw = p.read_bytes()
    p.write_bytes(raw[:30])  # past the fixed head, into the table
    with pytest.raises(BadRequest, match="model.gguf"):
        gguf_read.read(p)


def test_a_file_too_short_for_the_fixed_head_is_refused(tmp_path):
    p = tmp_path / "model.gguf"
    p.write_bytes(b"GGUF" + b"\0" * 3)
    with pytest.raises(BadRequest, match="too short"):
        gguf_read.read(p)


def test_an_implausible_tensor_count_is_refused(tmp_path):
    p = tmp_path / "model.gguf"
    p.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 10**9, 0))
    with pytest.raises(BadRequest, match="not a file this will parse"):
        gguf_read.read(p)


def test_a_missing_file_refuses_in_words(tmp_path):
    with pytest.raises(Refusal, match="is not a file"):
        gguf_read.read(tmp_path / "absent.gguf")


# ---------------------------------------------------------------- big arrays


def test_a_long_metadata_array_is_truncated_and_says_so(tmp_path):
    """A tokeniser vocabulary is frequently 128,000 strings. Nobody wants that
    in a JSON response, and silently dropping it would be a lie."""
    p = build(
        tmp_path,
        metadata={
            "tokenizer.ggml.tokens": (
                gguf_read._ARRAY,
                (gguf_read._STRING, [f"tok{i}" for i in range(200)]),
            )
        },
        tensors=[("a", 12, [256], 0)],
    )
    value = gguf_read.read(p, max_array=10).metadata["tokenizer.ggml.tokens"]
    assert value["truncated"] is True
    assert value["length"] == 200
    assert len(value["shown"]) == 10
    assert "190 more not shown" in value["note"]


def test_a_short_array_is_returned_whole(tmp_path):
    p = build(
        tmp_path,
        metadata={
            "x": (gguf_read._ARRAY, (gguf_read._STRING, ["a", "b"])),
        },
        tensors=[("a", 12, [256], 0)],
    )
    assert gguf_read.read(p).metadata["x"] == ["a", "b"]


# ------------------------------------------------------------------- routing


def test_the_route_refuses_a_path_outside_the_allowed_roots(tmp_path):
    """A path arriving from a browser is only read if it sits under a root the
    server was told about. Same boundary as the adapter loader, not a second
    one — two implementations of a security check drift."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from modelmri.server import create_app

    client = TestClient(create_app(trace_db=str(tmp_path / "t.sqlite")))
    r = client.get("/api/gguf", params={"path": "/etc/shadow"})
    assert r.status_code in (409, 422), r.text
    assert "error" in r.json()


def test_the_route_is_not_shadowed_and_reports_a_bad_file(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from modelmri import custom
    from modelmri.server import create_app

    junk = tmp_path / "not-really.gguf"
    junk.write_bytes(b"NOPE" + b"\0" * 64)
    monkeypatch.setattr(custom, "allowed_roots", lambda: [tmp_path.resolve()])

    client = TestClient(create_app(trace_db=str(tmp_path / "t.sqlite")))
    r = client.get("/api/gguf", params={"path": str(junk)})
    # A 422 proves the handler RAN and reached the parser, rather than a 404
    # from some other route swallowing the path.
    assert r.status_code == 422, r.text
    assert "not a GGUF file" in r.json()["error"]


def test_the_route_reads_a_real_header(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from modelmri import custom
    from modelmri.server import create_app

    p = build(
        tmp_path,
        metadata={"general.architecture": (gguf_read._STRING, "llama")},
        tensors=[("blk.0.attn_q.weight", 12, [256, 4], 0)],
    )
    monkeypatch.setattr(custom, "allowed_roots", lambda: [tmp_path.resolve()])

    client = TestClient(create_app(trace_db=str(tmp_path / "t.sqlite")))
    r = client.get("/api/gguf", params={"path": str(p)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["summary"]["architecture"] == "llama"
    assert body["summary"]["dominant_type"] == "Q4_K"


def test_the_scanner_no_longer_calls_gguf_unopenable():
    """It cannot RUN one. That is a different claim from having nothing to
    say about it, and only the first was ever true."""
    from modelmri.discover import LOOSE_WEIGHTS

    loadable, note = LOOSE_WEIGHTS[".gguf"]
    assert loadable is False
    assert "inspectable" in note


def test_the_result_is_json_safe(tmp_path):
    import json

    p = build(
        tmp_path,
        metadata={"general.architecture": (gguf_read._STRING, "llama")},
        tensors=[("a", 12, [256], 0), ("b", 199, [10], 0)],
    )
    json.dumps(gguf_read.read(p).to_dict())
