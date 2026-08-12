"""Adopting a recorded step, and the four ways it must refuse instead.

This is the join nothing else in the category can build — the recorder and the
weights are in one process — so it is also the feature with the most room to
be quietly wrong. Every panel downstream reads `last_ids` and none of them
checks where it came from, so adopting the wrong ids points attention,
ablation, the lens and patching at a sequence the model never saw, and nothing
on screen would say so.

Hence: the ids are verified against the loaded tokenizer rather than trusted,
the model id must match, and a hosted-API step is refused with an explanation
rather than replayed through whatever happens to be loaded.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from modelmri.errors import Refusal
from modelmri.traces import TraceStore


# ------------------------------------------------------------- the migration


def test_a_store_written_before_meta_existed_still_opens(tmp_path):
    """`CREATE TABLE IF NOT EXISTS` does nothing to an existing table, so an
    upgrade that only edits the schema string breaks every existing user."""
    path = tmp_path / "old.sqlite"
    old = sqlite3.connect(str(path))
    old.executescript(
        """
        CREATE TABLE trace (id TEXT PRIMARY KEY, name TEXT NOT NULL,
          started_at TEXT NOT NULL, meta TEXT NOT NULL DEFAULT '{}');
        CREATE TABLE step (id TEXT PRIMARY KEY, trace_id TEXT NOT NULL,
          parent_id TEXT, kind TEXT NOT NULL, name TEXT NOT NULL DEFAULT '',
          started_ms INTEGER NOT NULL, duration_ms INTEGER NOT NULL DEFAULT 0,
          input TEXT NOT NULL DEFAULT '', output TEXT NOT NULL DEFAULT '',
          tokens_in INTEGER, tokens_out INTEGER,
          error INTEGER NOT NULL DEFAULT 0, seq INTEGER NOT NULL);
        """
    )
    old.execute(
        "INSERT INTO trace VALUES('t1','old run','2026-01-01T00:00:00Z','{}')"
    )
    old.execute(
        "INSERT INTO step VALUES('s1','t1',NULL,'llm_call','plan',0,5,'in','out',"
        "1,2,0,0)"
    )
    old.commit()
    old.close()

    store = TraceStore(path)
    doc = store.get_trace("t1")
    assert doc is not None
    assert doc["steps"][0]["name"] == "plan"
    # The new column exists and reads as "not adoptable", which is correct:
    # a step recorded before meta existed carries no token ids.
    assert doc["steps"][0]["meta"] == {}
    assert doc["steps"][0]["adoptable"] is False


def test_the_migration_is_idempotent(tmp_path):
    path = tmp_path / "t.sqlite"
    TraceStore(path)
    TraceStore(path)  # must not raise "duplicate column name"
    assert TraceStore(path).list_traces() == []


# ------------------------------------------------------------ round-tripping


def _doc(meta=None):
    return {
        "name": "run",
        "started_at": "2026-01-01T00:00:00Z",
        "steps": [
            {
                "id": "s1",
                "kind": "llm_call",
                "name": "plan",
                "input": "The capital of France is",
                "output": " Paris",
                "meta": meta or {},
            }
        ],
    }


def test_step_meta_round_trips(tmp_path):
    store = TraceStore(tmp_path / "t.sqlite")
    meta = {"model": "gpt2", "input_ids": [464, 3139], "n_prompt_tokens": 2}
    store.import_trace(dict(_doc(meta), id="t1"))
    step = store.get_trace(store.list_traces()[0]["id"])["steps"][0]
    assert step["meta"]["model"] == "gpt2"
    assert step["meta"]["input_ids"] == [464, 3139]
    assert step["adoptable"] is True


def test_a_step_with_no_meta_is_not_adoptable(tmp_path):
    store = TraceStore(tmp_path / "t.sqlite")
    store.import_trace(_doc())
    step = store.get_trace(store.list_traces()[0]["id"])["steps"][0]
    assert step["adoptable"] is False


def test_damaged_meta_reads_as_not_adoptable_rather_than_exploding(tmp_path):
    """One bad row must not take down the whole trace view."""
    path = tmp_path / "t.sqlite"
    store = TraceStore(path)
    tid = store.import_trace(_doc({"model": "gpt2", "input_ids": [1]}))
    raw = sqlite3.connect(str(path))
    raw.execute("UPDATE step SET meta='{not json' WHERE trace_id=?", (tid,))
    raw.commit()
    raw.close()

    step = TraceStore(path).get_trace(tid)["steps"][0]
    assert step["meta"] == {}
    assert step["adoptable"] is False


def test_meta_that_is_not_an_object_is_treated_as_absent(tmp_path):
    path = tmp_path / "t.sqlite"
    store = TraceStore(path)
    tid = store.import_trace(_doc({"model": "gpt2", "input_ids": [1]}))
    raw = sqlite3.connect(str(path))
    raw.execute("UPDATE step SET meta='[1,2,3]' WHERE trace_id=?", (tid,))
    raw.commit()
    raw.close()
    assert TraceStore(path).get_trace(tid)["steps"][0]["meta"] == {}


# ---------------------------------------------------------------- adopting


class FakeTokenizer:
    """Encodes by word index; decodes back. Enough to check the id contract."""

    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text, return_tensors=None):
        import torch

        ids = [self.vocab.index(w) for w in text.split() if w in self.vocab]
        return type("Enc", (), {"input_ids": torch.tensor([ids])})()

    def decode(self, ids):
        return " ".join(self.vocab[int(i)] for i in ids)


@pytest.fixture
def runtime():
    torch = pytest.importorskip("torch")  # noqa: F841
    from modelmri.runtime import ModelRuntime

    rt = ModelRuntime.__new__(ModelRuntime)
    import threading

    rt._lock = threading.RLock()
    rt.replay = None
    rt.backend = "hf"
    rt.model = object()
    rt.hf_id = "gpt2"
    rt.epoch = 3
    rt.last_ids = None
    rt.last_ids_epoch = -1
    rt.last_prompt = ""
    rt.last_n_prompt_tokens = 0
    rt.last_user_span = (1, 2)
    rt._attn_variants = {"live": ["stale"]}
    rt._attn_tokens = ["stale"]
    rt._last_patch = {"stale": True}
    rt.tokenizer = FakeTokenizer(["the", "capital", "of", "France", "is", "Paris"])
    return rt


def _step(**meta):
    base = {
        "model": "gpt2",
        "prompt": "the capital of France is",
        "input_ids": [0, 1, 2, 3, 4, 5],
        "n_prompt_tokens": 5,
    }
    base.update(meta)
    return {"id": "s1", "kind": "llm_call", "input": base["prompt"], "meta": base}


def test_adopting_sets_the_state_every_panel_reads(runtime):
    out = runtime.adopt_step(_step())
    assert out["adopted"] is True
    assert out["n_tokens"] == 6 and out["n_prompt_tokens"] == 5
    assert runtime.last_n_prompt_tokens == 5
    assert [int(t) for t in runtime.last_ids.tolist()] == [0, 1, 2, 3, 4, 5]
    assert runtime.last_ids_epoch == runtime.epoch
    assert out["generation"] == "Paris"


def test_adopting_clears_everything_derived_from_the_previous_generation(runtime):
    """A stale attention capture rendered against these tokens would be a
    difference between two different generations, drawn as if it were one."""
    runtime.adopt_step(_step())
    assert runtime._attn_variants == {}
    assert runtime._attn_tokens is None
    assert runtime._last_patch == {}
    assert runtime.last_user_span is None


def test_a_hosted_api_step_is_refused_with_an_explanation(runtime):
    step = {"id": "s1", "kind": "llm_call", "input": "hi", "meta": {}}
    with pytest.raises(Refusal, match="not produced by a model on this machine"):
        runtime.adopt_step(step)


def test_the_wrong_model_is_refused_by_name(runtime):
    """Reading one model's ids through another's weights produces numbers about
    nothing, and no panel here would show that it had."""
    with pytest.raises(Refusal, match="produced by qwen"):
        runtime.adopt_step(_step(model="qwen"))


def test_a_tokenisation_mismatch_is_refused_not_rounded(runtime):
    """A tokenizer upgrade between the recording and now is the usual cause.
    Adopting near-identical ids would point every panel at a sequence the
    model never saw."""
    with pytest.raises(Refusal, match="do not match"):
        runtime.adopt_step(_step(input_ids=[0, 1, 2, 9, 9, 5]))


def test_a_recording_cannot_adopt(runtime):
    runtime.replay = {"some": "mri"}
    with pytest.raises(Refusal, match="does not carry one"):
        runtime.adopt_step(_step())


def test_ollama_cannot_adopt(runtime):
    runtime.backend = "ollama"
    with pytest.raises(Refusal, match="text only"):
        runtime.adopt_step(_step())


def test_no_model_loaded_says_which_one_to_load(runtime):
    runtime.model = None
    with pytest.raises(Refusal, match="gpt2"):
        runtime.adopt_step(_step())


def test_the_response_is_json_safe(runtime):
    json.dumps(runtime.adopt_step(_step()))
