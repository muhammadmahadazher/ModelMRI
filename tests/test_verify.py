"""A verifier that cannot fail is worthless.

Most of these build a `.mri` that says something the model does not, and check
that `verify` says so. The reproducing cases matter far less: a function that
returns "reproduced" unconditionally would pass every one of those.

The other half is about what `verify` REFUSES to claim. Bit-exact reproduction
across machines is not achievable, so the failure mode that would make this
feature actively harmful is reporting a pass it did not run.
"""

from __future__ import annotations

import base64
import gzip
import json
import os

import pytest

from modelmri import verify as verify_mod
from modelmri.errors import BadRequest

# ------------------------------------------------------------------ fixtures


def _doc(raw: bytes) -> dict:
    return json.loads(gzip.decompress(raw))


def _pack(doc: dict) -> bytes:
    return gzip.compress(json.dumps(doc).encode())


@pytest.fixture(scope="module")
def gpt2():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    # SKIP RATHER THAN DOWNLOAD. torch is a core dependency, so CI installs it
    # and this fixture would otherwise fetch ~500 MB of gpt2 weights on every
    # job of a 3-OS x 4-Python matrix. `revision_of` answers None for a model
    # that is not in the local cache, which is exactly the question here.
    from modelmri import receipts as _receipts

    if _receipts.revision_of("gpt2")[0] is None and not os.environ.get(
        "MODELMRI_TEST_DOWNLOAD"
    ):
        pytest.skip(
            "gpt2 is not in the local model cache — these are real-model "
            "integration tests. Set MODELMRI_TEST_DOWNLOAD=1 to fetch it."
        )

    from modelmri.runtime import ModelRuntime

    runtime = ModelRuntime()
    try:
        runtime.load("gpt2")
    except Exception as err:
        pytest.skip(f"gpt2 is not available here: {err}")
    yield runtime
    runtime.unload()


@pytest.fixture(scope="module")
def recording(gpt2, tmp_path_factory):
    """A real `.mri` from a real run, which every check below works against."""
    list(
        gpt2.generate_stream(
            "The capital of France is", max_new_tokens=4, temperature=0.0
        )
    )
    gpt2.ablate_heads(layer=0)
    gpt2.patch_trace("The capital of France is", "The capital of Italy is")
    path = tmp_path_factory.mktemp("mri") / "run.mri"
    path.write_bytes(gpt2.export_session(layer=0, head=0, note="test"))
    return path


@pytest.fixture(scope="module")
def light(recording, tmp_path_factory):
    """The same recording with its patch section removed.

    `_check_patch` runs the trace TWICE to measure its own floor, which is
    12.4s of the 19s a full verify costs here — measured, not assumed. Tests
    that are not about patching use this instead, so the file does real
    end-to-end work without paying for the same trace eighteen times. The
    patch tests use the full recording.
    """
    doc = _doc(recording.read_bytes())
    doc.pop("patch", None)
    path = tmp_path_factory.mktemp("light") / "light.mri"
    path.write_bytes(_pack(doc))
    return path


@pytest.fixture
def fresh():
    """An unloaded runtime, as `verify` expects to be handed one."""
    from modelmri.runtime import ModelRuntime

    runtime = ModelRuntime()
    yield runtime
    try:
        runtime.unload()
    except Exception:  # noqa: S110 - teardown, the test has already run
        pass


@pytest.fixture(scope="module")
def clean_report(recording):
    """One full verify of an untampered file, shared by every test that only
    reads it. Each call loads the model and re-runs every measurement."""
    from modelmri.runtime import ModelRuntime

    runtime = ModelRuntime()
    try:
        yield verify_mod.verify(recording, runtime)
    finally:
        try:
            runtime.unload()
        except Exception:  # noqa: S110 - teardown, the test has already run
            pass


def _verdicts(report) -> dict:
    return {c.name: c.verdict for c in report.checks}


# ------------------------------------------------------------- it reproduces


def test_a_file_verifies_against_the_machine_that_wrote_it(clean_report):
    report = clean_report
    verdicts = _verdicts(report)
    assert verdicts["generation"] == verify_mod.REPRODUCED
    assert verdicts["attention"] == verify_mod.REPRODUCED
    assert verdicts["patching"] == verify_mod.REPRODUCED
    assert report.differed == 0
    assert report.exit_code() == 0


def test_every_stored_head_map_is_checked_not_just_one(clean_report):
    """ "A number measured once is a sample" applies to this module too:
    checking one of 144 blocks and reporting "attention: reproduced" would be
    the same error the rest of the package exists to avoid."""
    report = clean_report
    attention = next(c for c in report.checks if c.name == "attention")
    assert attention.measured["blocks_in_file"] > 1
    assert attention.measured["blocks_checked"] == attention.measured["blocks_in_file"]


def test_the_reported_worst_block_is_one_that_was_measured(recording, clean_report):
    """The first version initialised the worst-block tracker to (0.0, 0.0),
    whose margin beats every block that actually passed — so it published the
    sentinel as though it were a measurement."""
    report = clean_report
    m = next(c for c in report.checks if c.name == "attention").measured
    doc = _doc(recording.read_bytes())
    assert m["worst_block"] in doc["attention"]
    # The tolerance must be at least that block's own quantisation step; a
    # reported 0.0 means the sentinel came back instead of a measurement.
    assert m["tolerance"] >= float(doc["attention"][m["worst_block"]]["scale"])


def test_the_tolerance_is_measured_and_says_where_it_came_from(clean_report):
    report = clean_report
    for name in ("attention", "patching"):
        check = next(c for c in report.checks if c.name == name)
        assert check.measured["tolerance_from"], f"{name} claims no source"
        assert "tolerance" in check.measured


# -------------------------------------------------------------- it can fail


def test_a_tampered_attention_map_is_caught(recording, fresh, tmp_path):
    """Every cell of one head pushed 40 quantisation steps — far past
    anything float noise or storage precision could explain."""
    doc = _doc(recording.read_bytes())
    key = sorted(doc["attention"])[0]
    raw = bytearray(base64.b64decode(doc["attention"][key]["q"]))
    for i in range(len(raw)):
        raw[i] = min(255, raw[i] + 40)
    doc["attention"][key]["q"] = base64.b64encode(bytes(raw)).decode("ascii")

    bad = tmp_path / "bad.mri"
    bad.write_bytes(_pack(doc))
    report = verify_mod.verify(bad, fresh)

    assert _verdicts(report)["attention"] == verify_mod.DIFFERS
    assert report.exit_code() == 1
    # And nothing else is dragged down with it.
    assert _verdicts(report)["generation"] == verify_mod.REPRODUCED
    assert _verdicts(report)["patching"] == verify_mod.REPRODUCED


def test_a_tampered_generation_is_caught(light, fresh, tmp_path):
    doc = _doc(light.read_bytes())
    doc["generation"] = " Berlin, obviously"
    bad = tmp_path / "bad.mri"
    bad.write_bytes(_pack(doc))

    report = verify_mod.verify(bad, fresh)
    check = next(c for c in report.checks if c.name == "generation")
    assert check.verdict == verify_mod.DIFFERS
    # Both sides printed, so the reader can see what changed rather than
    # being told that something did.
    assert "Berlin" in check.detail
    assert report.exit_code() == 1


def test_a_tampered_patch_grid_is_caught(recording, fresh, tmp_path):
    doc = _doc(recording.read_bytes())
    doc["patch"]["grids"]["resid"][0][0] += 5.0
    bad = tmp_path / "bad.mri"
    bad.write_bytes(_pack(doc))

    report = verify_mod.verify(bad, fresh)
    assert _verdicts(report)["patching"] == verify_mod.DIFFERS
    assert report.exit_code() == 1


# ------------------------------------------- what it refuses to claim it ran


def test_a_sampled_generation_is_not_compared(light, fresh, tmp_path):
    """A sampled run that comes back different is the sampler, not the model.
    Reporting that as `differs` would be a false accusation."""
    doc = _doc(light.read_bytes())
    for receipt in doc["receipts"]:
        if receipt["op"] == "generate":
            receipt["request"]["greedy"] = False
            receipt["request"]["temperature"] = 0.7
    doc["generation"] = " something else entirely"
    path = tmp_path / "sampled.mri"
    path.write_bytes(_pack(doc))

    report = verify_mod.verify(path, fresh)
    check = next(c for c in report.checks if c.name == "generation")
    assert check.verdict == verify_mod.NOT_VERIFIABLE
    assert "sampled" in check.detail
    # Not a failure: a file this machine cannot check is not a broken file.
    assert report.exit_code() == 0


def test_a_file_with_no_sampling_record_says_so(light, fresh, tmp_path):
    """Files written before the `generate` receipt existed carry no sampling
    configuration, and a difference in them cannot be told from the sampler."""
    doc = _doc(light.read_bytes())
    doc["receipts"] = [r for r in doc["receipts"] if r["op"] != "generate"]
    path = tmp_path / "old.mri"
    path.write_bytes(_pack(doc))

    report = verify_mod.verify(path, fresh)
    check = next(c for c in report.checks if c.name == "generation")
    assert check.verdict == verify_mod.NOT_VERIFIABLE
    assert "does not record" in check.detail


def test_a_dtype_difference_blocks_every_numeric_check(recording, fresh, tmp_path):
    """The example the roadmap names and `patch.py` documents: bf16 moves the
    reference gap from 4.000 to 4.467 and changes the reference token. A
    difference measured across two float formats is a fact about the formats."""
    doc = _doc(recording.read_bytes())
    for receipt in doc["receipts"]:
        receipt["dtype"] = "float64"  # nothing runs in float64 here
    doc["meta"]["dtype"] = "float64"
    path = tmp_path / "otherdtype.mri"
    path.write_bytes(_pack(doc))

    report = verify_mod.verify(path, fresh)
    assert set(_verdicts(report).values()) == {verify_mod.NOT_VERIFIABLE}
    assert any("float64" in n for n in report.notes)
    assert report.exit_code() == 0


def test_a_different_revision_blocks_the_comparison(light, fresh, tmp_path):
    """Different weights. A difference between them is expected and says
    nothing about reproducibility, so it must not be reported as `differs`."""
    doc = _doc(light.read_bytes())
    for receipt in doc["receipts"]:
        receipt["revision"] = "0" * 40
    doc["generation"] = " Berlin"  # would otherwise be a clear failure
    path = tmp_path / "othercommit.mri"
    path.write_bytes(_pack(doc))

    report = verify_mod.verify(path, fresh)
    assert _verdicts(report)["generation"] == verify_mod.NOT_VERIFIABLE
    assert report.exit_code() == 0


def test_the_head_ranking_is_re_run_not_merely_named(clean_report):
    """It used to be the one measurement in the file that could not be
    checked: the `.mri` recorded that a ranking had run and carried none of
    it, so `verify` could only name it."""
    report = clean_report
    check = next(c for c in report.checks if c.name == "head ranking")
    assert check.verdict == verify_mod.REPRODUCED
    assert check.measured["heads_compared"] > 1
    assert check.measured["tolerance_from"].startswith("this run's own noise floor")


def test_a_file_written_before_the_ranking_section_says_so(light, fresh, tmp_path):
    """Named rather than skipped — silence would read as "it reproduced"."""
    doc = _doc(light.read_bytes())
    doc.pop("ranking", None)
    path = tmp_path / "old.mri"
    path.write_bytes(_pack(doc))

    report = verify_mod.verify(path, fresh)
    check = next(c for c in report.checks if c.name == "head ranking")
    assert check.verdict == verify_mod.NOT_VERIFIABLE
    assert "does not carry it" in check.detail


def test_a_tampered_ranking_score_is_caught(light, fresh, tmp_path):
    """One head's KL moved, the order left alone. The scores broke and the
    order held, and the sentence has to say which."""
    doc = _doc(light.read_bytes())
    doc["ranking"]["ranked"][-1]["kl"] += 0.02
    path = tmp_path / "badscore.mri"
    path.write_bytes(_pack(doc))

    report = verify_mod.verify(path, fresh)
    check = next(c for c in report.checks if c.name == "head ranking")
    assert check.verdict == verify_mod.DIFFERS
    assert "top" in check.detail and "unchanged" in check.detail
    assert check.measured["max_abs_kl_diff"] > check.measured["tolerance"]


def test_a_reordered_ranking_is_caught_as_a_different_claim(light, fresh, tmp_path):
    """Scores can drift while the order holds, and the order can move while
    the scores barely do. They fail differently and are reported differently:
    a reader acts on "which head carries this", not on the last digits."""
    doc = _doc(light.read_bytes())
    rows = doc["ranking"]["ranked"]
    for row, kl in zip(rows, sorted(r["kl"] for r in rows), strict=True):
        row["kl"] = kl  # was descending, now ascending
    path = tmp_path / "badorder.mri"
    path.write_bytes(_pack(doc))

    report = verify_mod.verify(path, fresh)
    check = next(c for c in report.checks if c.name == "head ranking")
    assert check.verdict == verify_mod.DIFFERS
    assert "different claim about which head" in check.detail
    assert check.measured["spearman"] is not None
    assert check.measured["top_k_shared"] < check.measured["top_k"]


def test_a_model_this_machine_cannot_load_is_not_a_failure(light, fresh, tmp_path):
    doc = _doc(light.read_bytes())
    doc["meta"]["model"] = "not-a-real-org/not-a-real-model"
    path = tmp_path / "absent.mri"
    path.write_bytes(_pack(doc))

    report = verify_mod.verify(path, fresh)
    assert _verdicts(report) == {"model": verify_mod.NOT_VERIFIABLE}
    assert report.exit_code() == 0


def test_a_file_naming_no_model_is_refused_before_anything_loads(fresh, tmp_path):
    from modelmri import session

    raw = session.build(
        model_id=None,
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
    path = tmp_path / "anon.mri"
    path.write_bytes(raw)

    report = verify_mod.verify(path, fresh)
    assert _verdicts(report) == {"model": verify_mod.NOT_VERIFIABLE}
    assert fresh.model is None, "nothing should have been loaded"


def test_a_missing_file_says_so_rather_than_raising_a_stack(fresh, tmp_path):
    with pytest.raises(BadRequest, match="could not be read"):
        verify_mod.verify(tmp_path / "nope.mri", fresh)


# ------------------------------------------------------------------- output


def test_the_report_never_implies_a_clean_pass_means_correct(clean_report):
    """The honest weak point, stated in the output rather than left for the
    reader to remember."""
    report = clean_report
    joined = " ".join(report.notes)
    assert "does not make the finding right" in joined
    assert "not achievable" in joined


def test_the_rendered_report_names_both_setups(clean_report):
    text = verify_mod.render(clean_report)
    assert "here:" in text and "file:" in text
    assert "reproduced" in text


def test_the_report_survives_json(clean_report):
    """`--json` is the CI-facing surface, so it has to serialise cleanly."""
    report = clean_report
    round_tripped = json.loads(json.dumps(report.to_dict(), allow_nan=False))
    assert round_tripped["totals"]["reproduced"] >= 1
    assert {c["name"] for c in round_tripped["checks"]} >= {"generation", "attention"}


def test_a_recording_serves_its_ranking_with_no_model_loaded(recording):
    """The point of the format: a colleague sees the head you found without
    downloading 8 GB.

    The replay arm used to refuse this flatly — "ranking heads means running
    the model" — which was right when the file held nothing here. Once it
    carries the ranking, refusing a file that already holds the answer is the
    format failing rather than the reader asking too much. `patch_trace`
    records the same lesson.
    """
    from modelmri.runtime import ModelRuntime

    reader = ModelRuntime()  # nothing loaded, as on a stranger's machine
    reader.open_session(recording.read_bytes())

    out = reader.ablate_heads()
    assert out["recorded"] is True
    assert out["ranked"], "the recording carries rows"
    assert out["baseline"], "and says which baseline produced them"
    assert reader.model is None, "serving a recording must not load anything"


def test_a_recording_without_a_ranking_still_refuses_with_a_reason(recording, tmp_path):
    from modelmri.errors import Refusal
    from modelmri.runtime import ModelRuntime

    doc = _doc(recording.read_bytes())
    doc.pop("ranking", None)
    path = tmp_path / "noranking.mri"
    path.write_bytes(_pack(doc))

    reader = ModelRuntime()
    reader.open_session(path.read_bytes())
    with pytest.raises(Refusal, match="does not carry a head ranking"):
        reader.ablate_heads()
