# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""`modelmri steer list` and `modelmri steer rm`.

The store lives on the user's own machine and, until this landed, there was no
way to see inside it that did not involve starting a server and opening a
browser. These two verbs are the terminal's half.

What the tests below are actually about:

  * **an empty store is the ordinary first state.** A machine that has never
    fitted a direction is not broken, and the useful thing to print there is
    where directions come from — not a blank screen and not an error;
  * **compatibility is a claim about a MODEL, so with no model named there is
    no claim to make.** `--model` reads a config and the column becomes real;
    without it the listing says so rather than inventing a verdict, which is
    the same three-state rule the panel keeps;
  * **`beats_null` absent is not `beats_null` false.** A direction saved
    before the store recorded one has no verdict, and printing "did not beat
    its null" for it would be a fabricated one;
  * **rm exits non-zero when it did not delete.** A script has to be able to
    tell "the user said no" from "it is gone".

Nothing here reaches the network or loads weights: `--model` is not passed in
any test that would need a config, and the store is `MODELMRI_HOME`.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from modelmri import steer_vectors as sv  # noqa: E402
from modelmri.cli import steer_list, steer_remove  # noqa: E402

D = 16


def _save(name: str, **meta):
    g = torch.Generator().manual_seed(0)
    vec = torch.randn(D, generator=g)
    sv.save(
        name,
        vec / vec.norm(),
        {
            "model": "Qwen/Qwen3-1.7B",
            "layer": 6,
            "hidden_size": D,
            "method": "caa",
            "dtype": "bfloat16",
            **meta,
        },
    )


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELMRI_HOME", str(tmp_path))
    return tmp_path


def test_an_empty_store_says_where_directions_come_from(store, capsys):
    assert steer_list() == 0
    out = capsys.readouterr().out
    assert "none saved yet" in out
    assert "Probe panel" in out, "the probe→steering connection is the useful line"


def test_the_listing_names_the_model_the_layer_and_the_null(store, capsys):
    _save("politeness", beats_null=True, p_value=0.111)
    assert steer_list() == 0
    out = capsys.readouterr().out
    assert "politeness" in out
    assert "Qwen/Qwen3-1.7B" in out
    assert "beat its null" in out and "p=0.111" in out


def test_a_direction_that_failed_its_null_is_not_dressed_up(store, capsys):
    _save("nothing", beats_null=False, p_value=0.444)
    steer_list()
    out = capsys.readouterr().out
    assert "did NOT beat its null" in out
    assert "p=0.444" in out


def test_a_direction_with_no_recorded_verdict_says_so(store, capsys):
    """Absent is not false. Printing "did not beat its null" for a direction
    that was never judged would be a fabricated verdict, which is the one
    thing this project's rules single out."""
    _save("older-vector")
    steer_list()
    out = capsys.readouterr().out
    assert "not recorded" in out
    assert "did NOT" not in out


def test_with_no_model_the_listing_says_it_cannot_judge_compatibility(store, capsys):
    _save("politeness")
    steer_list()
    out = capsys.readouterr().out
    assert "--model" in out
    assert "only means anything against the model it was fitted on" in out


def test_a_damaged_file_is_listed_rather_than_dropped(store, capsys):
    from modelmri import paths

    paths.ensure(sv.store_dir())
    (sv.store_dir() / "broken.json").write_text("{not json", encoding="utf-8")
    steer_list()
    assert "damaged" in capsys.readouterr().out


def test_the_json_form_carries_the_three_state_compatibility(store, capsys):
    _save("politeness")
    assert steer_list(as_json=True) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["model"] is None
    assert body["directions"][0]["compatible"] is None


def test_the_json_form_says_why_it_could_not_judge(store, capsys, monkeypatch):
    """`compatible: null` has two causes and a script has to tell them apart:
    no model was named, or the named model's config could not be read here.
    The human listing prints that reason; the JSON form dropped it, which
    left every row saying "unknown" with nothing saying why — the one thing
    an absent measurement is not allowed to do anywhere else in this tool.

    `_hidden_size_of` is stubbed rather than called: reading a real config is
    a network round trip, and what is under test is what `steer_list` does
    with the answer.
    """
    from modelmri import cli

    monkeypatch.setattr(
        cli,
        "_hidden_size_of",
        lambda model: (None, f"{model}'s config could not be read here (OSError)"),
    )
    _save("politeness")
    assert cli.steer_list(model="some/unreachable-model", as_json=True) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["model"] == "some/unreachable-model"
    assert body["directions"][0]["compatible"] is None
    assert "could not be read here" in body["unjudged"]


def test_the_json_form_leaves_unjudged_empty_when_it_could_judge(store, capsys):
    """No model named is not a failure to read one, so there is no reason to
    report. An empty string here is the difference between "nothing was
    asked" and "it was asked and could not be answered"."""
    _save("politeness")
    assert steer_list(as_json=True) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["unjudged"] == ""


def test_removing_with_yes_deletes_it(store, capsys):
    _save("politeness")
    assert steer_remove("politeness", yes=True) == 0
    assert sv.catalogue() == []
    assert "deleted 'politeness'" in capsys.readouterr().out


def test_declining_the_prompt_keeps_the_file_and_exits_non_zero(
    store, capsys, monkeypatch
):
    """A script has to be able to tell "said no" from "did it"."""
    _save("politeness")
    monkeypatch.setattr("builtins.input", lambda _: "n")
    assert steer_remove("politeness") == 1
    assert len(sv.catalogue()) == 1
    assert "nothing deleted" in capsys.readouterr().out


def test_a_closed_stdin_is_a_decline_rather_than_a_crash(store, monkeypatch):
    """`modelmri steer rm x < /dev/null` must not traceback."""
    _save("politeness")

    def eof(_):
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    assert steer_remove("politeness") == 1
    assert len(sv.catalogue()) == 1


def test_removing_one_that_is_not_there_reports_the_sentence(store, capsys):
    assert steer_remove("absent", yes=True) == 2
    assert "no saved direction called 'absent'" in capsys.readouterr().err


def test_listing_does_not_create_the_store(store):
    """The read-only rule, from a second door."""
    steer_list()
    assert not sv.store_dir().exists()
