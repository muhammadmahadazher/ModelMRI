"""What produced a number has to be right, or it is worse than absent.

A receipt is trusted by whoever receives it — that is the entire point of the
feature — so the failure mode that matters is not "a field is missing", it is
"a field is confidently wrong". Most of these tests are about the difference.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from modelmri import receipts
from modelmri.errors import BadRequest

# ------------------------------------------------------------------ fixtures


class _FastBackend:
    def __init__(self, blob):
        self._blob = blob

    def to_str(self):
        return self._blob


class _Tokenizer:
    """A tokenizer with an optional fast backend, like the real ones."""

    def __init__(self, vocab=None, backend=None, raises=False):
        self._vocab = vocab or {"a": 0, "b": 1, "c": 2}
        self.backend_tokenizer = backend
        self._raises = raises

    def get_vocab(self):
        if self._raises:
            raise RuntimeError("no vocabulary here")
        return dict(self._vocab)


class _Runtime:
    def __init__(self, **kw):
        self.hf_id = kw.get("hf_id", "gpt2")
        self.model = kw.get("model")
        self.tokenizer = kw.get("tokenizer", _Tokenizer())
        self.device = kw.get("device", "cpu")
        self.last_prompt = kw.get("last_prompt", "")
        self.last_ids = kw.get("last_ids")


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """An HF hub cache that is not this machine's."""
    root = tmp_path / "hub"
    root.mkdir(parents=True)
    monkeypatch.setenv("HF_HUB_CACHE", str(root))
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    for stale in ("HUGGINGFACE_HUB_CACHE", "XDG_CACHE_HOME"):
        monkeypatch.delenv(stale, raising=False)
    return root


def _cache_model(root, repo: str, *, snapshots: list[str], ref: str | None = None):
    base = root / ("models--" + repo.replace("/", "--"))
    for sha in snapshots:
        (base / "snapshots" / sha).mkdir(parents=True)
    if ref is not None:
        (base / "refs").mkdir(parents=True, exist_ok=True)
        (base / "refs" / "main").write_text(ref, encoding="utf-8")
    return base


# -------------------------------------------------------------------- digest


def test_the_same_text_hashes_the_same_way_every_time():
    assert receipts.digest("the capital of France is") == receipts.digest(
        "the capital of France is"
    )
    assert receipts.digest("a") != receipts.digest("b")


def test_a_digest_is_short_enough_to_read_in_a_terminal():
    assert len(receipts.digest("x")) == receipts.DIGEST_CHARS


# ---------------------------------------------------------------- model name


def test_a_hub_id_survives_intact():
    """`org/name` must NOT be reduced to `name` — that is a different model."""
    assert receipts.public_name("allenai/OLMo-2-0425-1B-Instruct") == (
        "allenai/OLMo-2-0425-1B-Instruct"
    )
    assert receipts.public_name("gpt2") == "gpt2"


def test_a_folder_model_is_reduced_to_its_name(tmp_path):
    folder = tmp_path / "Users" / "someone" / "my-net"
    folder.mkdir(parents=True)
    assert receipts.public_name(str(folder)) == "my-net"


def test_no_model_is_none_not_empty_string():
    assert receipts.public_name(None) is None
    assert receipts.public_name("") is None


# ------------------------------------------------------------------ revision


def test_the_revision_comes_from_refs_main(cache):
    """Not from the newest snapshot directory: that is a guess."""
    _cache_model(cache, "org/model", snapshots=["aaa", "bbb"], ref="aaa")
    sha, how = receipts.revision_of("org/model")
    assert sha == "aaa"
    assert "refs/main" in how


def test_a_canonical_model_with_no_owner_resolves(cache):
    """`gpt2` is a real repo, cached as `models--gpt2`, and is this package's
    own worked example. Requiring a slash reported it as "not a Hub repo"."""
    _cache_model(cache, "gpt2", snapshots=["deadbeef"], ref="deadbeef")
    sha, _ = receipts.revision_of("gpt2")
    assert sha == "deadbeef"


def test_one_cached_revision_needs_no_ref_file(cache):
    _cache_model(cache, "org/model", snapshots=["only"])
    sha, how = receipts.revision_of("org/model")
    assert sha == "only"
    assert "only revision" in how


def test_several_revisions_and_no_ref_refuses_to_pick_one(cache):
    """The whole reason this function exists. Naming one would be a guess, and
    a wrong revision on a receipt is worse than no revision."""
    _cache_model(cache, "org/model", snapshots=["aaa", "bbb", "ccc"])
    sha, how = receipts.revision_of("org/model")
    assert sha is None
    assert "3 revisions" in how and "guess" in how


def test_an_uncached_model_says_so(cache):
    sha, how = receipts.revision_of("org/never-downloaded")
    assert sha is None
    assert "not in the local cache" in how


@pytest.mark.parametrize(
    "hf_id",
    [
        "llama3.2:1b",  # an Ollama tag
        "C:\\models\\thing",  # a Windows path
        "./relative-dir",
        "~/models/thing",
        "a/b/c",  # too many segments to be a repo id
        None,
        "",
    ],
)
def test_things_that_are_not_hub_repos_get_no_commit(cache, hf_id):
    sha, how = receipts.revision_of(hf_id)
    assert sha is None
    assert "Hub repository" in how


def test_an_unreadable_cache_is_reported_not_swallowed(cache, monkeypatch):
    """ "the cache could not be read" and "this has no commit" are different
    facts, and a reader acts differently on each."""

    def boom():
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(receipts.paths, "hf_hub_cache", boom)
    sha, how = receipts.revision_of("org/model")
    assert sha is None
    assert "could not be read" in how


# ----------------------------------------------------------------- tokenizer


def test_the_fast_definition_is_preferred_and_named():
    tok = _Tokenizer(backend=_FastBackend('{"model":{"vocab":{"a":0}}}'))
    sha, how = receipts.tokenizer_fingerprint(tok)
    assert sha and "full fast-tokenizer" in how


def test_two_tokenizers_with_the_same_vocab_and_different_normalisers_differ():
    """The reason the fast definition is preferred: identical vocabularies
    with different normalisers produce different token ids."""
    a = _Tokenizer(backend=_FastBackend('{"normalizer":"NFC","vocab":{"a":0}}'))
    b = _Tokenizer(backend=_FastBackend('{"normalizer":"NFKC","vocab":{"a":0}}'))
    assert receipts.tokenizer_fingerprint(a)[0] != receipts.tokenizer_fingerprint(b)[0]


def test_the_vocabulary_fallback_says_what_it_could_not_see():
    sha, how = receipts.tokenizer_fingerprint(_Tokenizer())
    assert sha
    assert "vocabulary only" in how and "NOT covered" in how


def test_the_vocabulary_hash_does_not_depend_on_dict_order():
    """Python dicts are insertion-ordered, and two loads of the same files do
    not guarantee the same insertion order."""
    a = _Tokenizer(vocab={"a": 0, "b": 1, "c": 2})
    b = _Tokenizer(vocab={"c": 2, "a": 0, "b": 1})
    assert receipts.tokenizer_fingerprint(a)[0] == receipts.tokenizer_fingerprint(b)[0]


def test_a_different_vocabulary_hashes_differently():
    a = _Tokenizer(vocab={"a": 0, "b": 1})
    b = _Tokenizer(vocab={"a": 0, "b": 2})
    assert receipts.tokenizer_fingerprint(a)[0] != receipts.tokenizer_fingerprint(b)[0]


def test_a_backend_that_will_not_serialise_falls_back_rather_than_failing():
    class _Broken:
        def to_str(self):
            raise RuntimeError("nope")

    sha, how = receipts.tokenizer_fingerprint(_Tokenizer(backend=_Broken()))
    assert sha and "vocabulary only" in how


def test_no_tokenizer_is_none_with_a_reason():
    assert receipts.tokenizer_fingerprint(None) == (None, "no tokenizer was loaded")


def test_a_tokenizer_that_refuses_its_vocabulary_says_so():
    sha, how = receipts.tokenizer_fingerprint(_Tokenizer(raises=True))
    assert sha is None
    assert "would not report its vocabulary" in how


# --------------------------------------------------------------------- stamp


def test_a_stamp_records_what_is_actually_loaded(cache):
    _cache_model(cache, "org/model", snapshots=["abc123"], ref="abc123")
    receipt = receipts.stamp(
        _Runtime(hf_id="org/model", device="cuda:0", last_prompt="hello"),
        "ablate_heads",
        request={"layer": 3, "baseline": "resample"},
        seed=7,
    )
    assert receipt.op == "ablate_heads"
    assert receipt.model == "org/model"
    assert receipt.revision == "abc123"
    assert receipt.device == "cuda:0"
    assert receipt.seed == 7
    assert receipt.request == {"layer": 3, "baseline": "resample"}
    assert receipt.prompt_sha256 == receipts.digest("hello")
    assert receipt.tool_version


def test_an_unseeded_measurement_has_no_seed_rather_than_zero():
    """None is "this was not seeded". 0 is a draw that never happened."""
    assert receipts.stamp(_Runtime(), "attribute_tokens").seed is None


def test_a_receipt_cannot_be_edited_after_the_fact():
    """It describes something that already happened."""
    receipt = receipts.stamp(_Runtime(), "attribute_tokens")
    with pytest.raises(dataclasses.FrozenInstanceError):
        receipt.op = "something_else"


def test_the_named_prompt_wins_over_the_last_one():
    """`patch_trace` measures a PAIR and names the clean side explicitly."""
    receipt = receipts.stamp(
        _Runtime(last_prompt="the last generation"), "patch_trace", prompt="clean side"
    )
    assert receipt.prompt_sha256 == receipts.digest("clean side")


def test_the_summary_names_what_is_missing_rather_than_omitting_it():
    line = receipts.stamp(_Runtime(hf_id="org/nope"), "attribute_tokens").summary()
    assert "unknown-revision" in line


# ----------------------------------------------------------- paths in request


@pytest.mark.parametrize(
    "value,expected",
    [
        ("C:\\Users\\someone\\sae", "sae"),
        ("/home/someone/checkpoints/sae", "sae"),
        ("\\\\server\\share\\thing", "thing"),
        ("google/gemma-scope-2b", "google/gemma-scope-2b"),  # a repo id, untouched
        ("zero", "zero"),
        ("blocks.5.hook_resid_post", "blocks.5.hook_resid_post"),
    ],
)
def test_an_absolute_path_in_a_request_is_reduced_to_a_name(value, expected):
    """`rank_features` puts `sae_repo` in its receipt and an SAE can be loaded
    from a local directory, so that path travelled inside every exported
    `.mri`. Caught by the leak test, not by review."""
    out = receipts.stamp(_Runtime(), "rank_features", request={"sae_repo": value})
    assert out.request["sae_repo"] == expected


def test_a_path_inside_a_list_argument_is_reduced_too():
    out = receipts.stamp(
        _Runtime(), "sweep", request={"files": ["/home/me/a.jsonl", "plain"]}
    )
    assert out.request["files"] == ["a.jsonl", "plain"]


def test_an_object_argument_does_not_smuggle_a_path_through_its_repr(tmp_path):
    from pathlib import Path

    out = receipts.stamp(_Runtime(), "sweep", request={"corpus": Path(tmp_path)})
    assert str(tmp_path) not in json.dumps(out.to_dict())


# --------------------------------------------------------------------- parse


def test_no_receipts_section_is_fine():
    assert receipts.parse(None) == []
    assert receipts.parse([]) == []


def test_a_receipts_section_that_is_not_a_list_is_refused():
    with pytest.raises(BadRequest, match="not a list"):
        receipts.parse({"op": "ablate_heads"})


def test_a_receipt_without_an_op_is_refused():
    """A receipt that does not say what it describes describes nothing."""
    with pytest.raises(BadRequest, match="which measurement"):
        receipts.parse([{"model": "gpt2"}])
    with pytest.raises(BadRequest, match="which measurement"):
        receipts.parse([{"op": "   "}])


def test_a_receipt_that_is_not_fields_is_refused():
    with pytest.raises(BadRequest, match="not a set of fields"):
        receipts.parse(["ablate_heads"])


def test_none_survives_parse_rather_than_becoming_empty_string():
    """The writer was careful to record "could not be established". Coercing
    it to "" here would erase exactly that distinction."""
    out = receipts.parse([{"op": "x", "revision": None, "revision_note": "why not"}])
    assert out[0]["revision"] is None
    assert out[0]["revision_note"] == "why not"


def test_a_boolean_seed_does_not_read_as_a_number():
    """bool is an int in Python, so `seed: true` would arrive as seed 1."""
    assert receipts.parse([{"op": "x", "seed": True}])[0]["seed"] is None
    assert receipts.parse([{"op": "x", "seed": 0}])[0]["seed"] == 0


def test_a_hostile_request_block_is_bounded():
    big = {f"k{i}": i for i in range(receipts.MAX_REQUEST_KEYS + 5)}
    with pytest.raises(BadRequest, match="request fields"):
        receipts.parse([{"op": "x", "request": big}])


def test_a_request_that_is_not_fields_is_refused():
    with pytest.raises(BadRequest, match="request block"):
        receipts.parse([{"op": "x", "request": ["layer", 3]}])


def test_long_strings_are_truncated_not_rejected():
    out = receipts.parse([{"op": "x", "revision_note": "y" * 5000}])
    assert len(out[0]["revision_note"]) == receipts.MAX_REQUEST_TEXT


def test_a_stamped_receipt_survives_its_own_validator():
    """The writer must not be laxer than the reader — that is how you build
    files nobody can open."""
    stamped = receipts.stamp(
        _Runtime(), "ablate_heads", request={"layer": 0, "baseline": "zero"}
    ).to_dict()
    (out,) = receipts.parse([stamped])
    assert out["op"] == "ablate_heads"
    assert out["request"] == {"layer": 0, "baseline": "zero"}


# --------------------------------------------------------------------- jsonl


def test_receipts_write_one_line_each(tmp_path):
    rows = [receipts.stamp(_Runtime(), op) for op in ("a", "b", "c")]
    path = receipts.write_jsonl(rows, tmp_path / "out" / "receipts.jsonl")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    assert [json.loads(line)["op"] for line in lines] == ["a", "b", "c"]


# ------------------------------------------------------- inside a .mri file


def _mri(**over) -> bytes:
    from modelmri import session

    kw = dict(
        model_id="gpt2",
        device="cpu",
        dtype="float32",
        n_params=124_439_808,
        tokens=["a", "b"],
        prompt="a",
        generation="b",
        attention={(0, 0): [[1.0, 0.0], [0.5, 0.5]]},
        n_layers=2,
        n_heads=2,
    )
    kw.update(over)
    return session.build(**kw)


def test_a_file_without_receipts_carries_no_empty_section():
    """Additive like `patch` and `graph`: an empty key would make every older
    file claim a receipts section and every reader render one."""
    import gzip

    doc = json.loads(gzip.decompress(_mri()))
    assert "receipts" not in doc


def test_receipts_survive_the_round_trip():
    from modelmri import session

    stamped = receipts.stamp(
        _Runtime(), "ablate_heads", request={"layer": 0, "baseline": "zero"}
    ).to_dict()
    parsed = session.parse(_mri(receipts=[stamped]))
    assert [r["op"] for r in parsed.receipts] == ["ablate_heads"]
    assert parsed.receipts[0]["request"] == {"layer": 0, "baseline": "zero"}


def test_an_older_reader_ignoring_the_key_is_why_the_version_did_not_move():
    import gzip

    from modelmri import session

    stamped = receipts.stamp(_Runtime(), "x").to_dict()
    doc = json.loads(gzip.decompress(_mri(receipts=[stamped])))
    assert doc["format_version"] == session.FORMAT_VERSION


def test_a_hostile_receipts_section_is_refused_at_the_reader():
    import gzip

    from modelmri import session

    raw = _mri()
    doc = json.loads(gzip.decompress(raw))
    doc["receipts"] = [{"no_op_field": True}]
    with pytest.raises(BadRequest, match="which measurement"):
        session.parse(gzip.compress(json.dumps(doc).encode()))


def test_the_writer_is_not_laxer_than_the_reader():
    """A writer that accepts what the reader refuses builds files nobody can
    open, which is the lesson the `graph` section records two sections up."""

    with pytest.raises(BadRequest):
        _mri(receipts=[{"no_op_field": True}])


# ------------------------------------------- against a real model, on a run
#
# These two were found together and are the same bug: something derived from
# ONE generation surviving into the export of ANOTHER. Receipts introduced the
# risk; the patch trace already had it.


@pytest.fixture(scope="module")
def gpt2_runtime():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    from modelmri.runtime import ModelRuntime

    runtime = ModelRuntime()
    try:
        runtime.load("gpt2")
    except Exception as err:  # no local copy, no network
        pytest.skip(f"gpt2 is not available here: {err}")
    yield runtime
    runtime.unload()


def test_a_receipt_records_the_run_it_was_taken_on(gpt2_runtime):
    list(
        gpt2_runtime.generate_stream(
            "The capital of France is", max_new_tokens=3, temperature=0.0
        )
    )
    receipt = gpt2_runtime.ablate_heads(layer=0)["receipt"]

    assert receipt["model"] == "gpt2"
    # A real 40-character commit, read out of the cache rather than guessed.
    assert receipt["revision"] and len(receipt["revision"]) == 40
    assert receipt["attn_implementation"] == "eager"
    assert receipt["prompt_sha256"] == receipts.digest("The capital of France is")
    assert receipt["n_prompt_tokens"] and receipt["n_prompt_tokens"] > 0


def test_two_measurements_of_one_run_agree_on_the_setup(gpt2_runtime):
    """Which is the property that makes receipts comparable at all."""
    list(
        gpt2_runtime.generate_stream(
            "The capital of France is", max_new_tokens=3, temperature=0.0
        )
    )
    a = gpt2_runtime.ablate_heads(layer=0)["receipt"]
    b = gpt2_runtime.attribute_tokens()["receipt"]

    assert a["prompt_sha256"] == b["prompt_sha256"]
    assert a["tokenizer_sha256"] == b["tokenizer_sha256"]
    assert a["revision"] == b["revision"]
    assert a["op"] != b["op"]


def test_a_new_generation_drops_the_previous_runs_receipts(gpt2_runtime):
    list(
        gpt2_runtime.generate_stream(
            "The Eiffel Tower is in the city of", max_new_tokens=3, temperature=0.0
        )
    )
    gpt2_runtime.ablate_heads(layer=0)
    assert gpt2_runtime._receipts_for_export()

    list(
        gpt2_runtime.generate_stream(
            "Bananas are yellow because", max_new_tokens=3, temperature=0.0
        )
    )
    assert gpt2_runtime._receipts_for_export() == [], (
        "a receipt names the prompt it was taken on, so it cannot survive into "
        "an export describing a different one"
    )


def test_a_new_generation_drops_the_previous_runs_patch_trace(gpt2_runtime):
    """`_patch_for_export` guards on the epoch and its docstring says that is
    to stop "a trace measured on an earlier prompt" being written beside a
    different run's tokens. The epoch does NOT move on generation, so that
    guard never fired for the case it describes.

    Measured before the fix: patching the Eiffel Tower, then generating about
    bananas, produced a `.mri` whose tokens and attention were the bananas and
    whose patch section was the Eiffel Tower.
    """
    list(
        gpt2_runtime.generate_stream(
            "The Eiffel Tower is in the city of", max_new_tokens=3, temperature=0.0
        )
    )
    gpt2_runtime.patch_trace(
        "The Eiffel Tower is in the city of", "The Colosseum is in the city of"
    )
    assert gpt2_runtime._patch_for_export(), "the trace IS this run — keep it"

    list(
        gpt2_runtime.generate_stream(
            "Bananas are yellow because", max_new_tokens=3, temperature=0.0
        )
    )
    assert gpt2_runtime._patch_for_export() == {}, (
        "this trace describes a prompt that is no longer the run being exported"
    )
