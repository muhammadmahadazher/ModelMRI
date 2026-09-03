# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""What hub.py and discover.py say when something goes wrong.

Both modules answer questions about the world outside this process — a Hub
that may be unreachable, a token file that may be someone else's, a model
directory that may be half-synced. Every one of those has a right answer that
is neither a crash nor a confident wrong sentence, and none of them had a test.

The survey that preceded this pass put it plainly: `test_no_machine_leaks.py`
walks only 200-responses, so nothing guarded the leak-shaped error messages.
These are the guards for the two modules' worth of them.
"""

from __future__ import annotations

import json
import logging
import urllib.error
from pathlib import Path

import pytest

from modelmri import discover, hub
from modelmri.errors import BadRequest, Refusal

# ------------------------------------------------------- reaching the Hub


@pytest.mark.parametrize(
    "boom",
    [
        pytest.param(
            urllib.error.URLError(OSError(11001, "getaddrinfo failed")), id="urlerror"
        ),
        # NOT a URLError, and this is the whole reason the parameter exists.
        # urllib wraps a failure to CONNECT, but a server that accepts and then
        # never answers raises a bare TimeoutError out of `getresponse()`.
        # Measured against a socket that accepts and stalls; before this pass
        # it escaped `search` entirely and became a 500.
        pytest.param(TimeoutError("timed out"), id="bare-timeout"),
    ],
)
def test_an_unreachable_hub_is_a_refusal_that_leaks_nothing(monkeypatch, boom):
    def explode(*_a, **_k):
        raise boom

    monkeypatch.setattr(hub.urllib.request, "urlopen", explode)
    monkeypatch.setattr(hub, "token", lambda: None)

    with pytest.raises(Refusal) as caught:
        hub.search("qwen")

    body = str(caught.value)
    # 409, because nothing here broke — the network did not deliver an answer
    # and we said so. A 500 would blame ModelMRI for the reader's wifi.
    assert "Could not reach the HuggingFace Hub" in body
    # The published sentence must not carry the exception's own words.
    # `str(URLError)` is "<urlopen error [Errno 11001] getaddrinfo failed>",
    # which nobody wrote for a reader.
    for machinery in ("urlopen error", "getaddrinfo", "Errno", "timed out"):
        assert machinery not in body, f"raw exception text reached the browser: {body}"
    # A refusal earns its 409 by saying what to do instead.
    assert "On this machine" in body


def test_an_unreachable_hub_publishes_unknown_counts_rather_than_zeros(
    monkeypatch,
):
    """`suggested()` deliberately survives an outage — a picker with names and
    no metadata beats one with nothing in it. What it must not do is fill the
    gap with numbers.

    MEASURED before this: with the Hub down, `GET /api/hub/models` returned
    200 and every curated row carried `downloads: 0, likes: 0, updated: ""` —
    byte-identical to a real repo nobody has downloaded, and rendered beside a
    download button. `params` and `size_gb` in that same dict already said
    `None` for exactly this reason; these three had not caught up.
    """

    def explode(*_a, **_k):
        raise OSError("no route to host")

    monkeypatch.setattr(hub, "_api", explode)
    monkeypatch.setattr(hub, "token", lambda: None)

    rows = hub.suggested()
    assert rows, "an outage must still offer the names"
    for row in rows:
        assert row["id"]
        assert row["downloads"] is None, row["id"]
        assert row["likes"] is None, row["id"]
        assert row["updated"] is None, row["id"]


def test_a_repo_that_publishes_no_count_is_unknown_not_least_popular(monkeypatch):
    """The listing is SORTED by downloads, so an absent count rendered as 0
    sorts as the least popular thing on the page — a claim nobody made. And
    `isinstance(True, int)` is True, so a bool must not count as 1."""
    monkeypatch.setattr(hub, "token", lambda: None)
    monkeypatch.setattr(
        hub,
        "_api",
        lambda *_a, **_k: [
            {"id": "someone/quiet", "safetensors": {"total": 600_000_000}},
            {"id": "someone/odd", "downloads": True, "likes": "many"},
            {"id": "someone/real", "downloads": 0, "likes": 0, "lastModified": ""},
        ],
    )
    monkeypatch.setattr(hub, "_resolve_access", lambda entries, _tok: entries)

    quiet, odd, real = hub.search("x")
    assert quiet["downloads"] is None and quiet["likes"] is None
    assert quiet["updated"] is None
    assert odd["downloads"] is None, "a bool is not a count of one"
    assert odd["likes"] is None, "a string is not a count"
    # A PUBLISHED zero is still a zero. The point is telling the two apart.
    assert real["downloads"] == 0 and real["likes"] == 0
    assert real["updated"] is None, "an empty date string is not a date"


def test_the_real_hub_error_survives_in_the_log(monkeypatch, caplog):
    """Not pasting the exception is only right if it still exists somewhere."""

    def explode(*_a, **_k):
        raise urllib.error.URLError(OSError(11001, "getaddrinfo failed"))

    monkeypatch.setattr(hub.urllib.request, "urlopen", explode)
    monkeypatch.setattr(hub, "token", lambda: None)

    with caplog.at_level(logging.WARNING, logger="modelmri"), pytest.raises(Refusal):
        hub.search("qwen")

    record = next(r for r in caplog.records if r.name == "modelmri")
    assert record.exc_info is not None, "the traceback was discarded, not relocated"
    assert "getaddrinfo" in logging.Formatter().formatException(record.exc_info)


# ------------------------------------------------------------- signing in


def test_a_bad_token_is_a_bad_request_and_still_a_value_error():
    """`BadRequest` subclasses ValueError so a half-migrated handler that
    still catches ValueError keeps answering 422 rather than 500."""
    with pytest.raises(BadRequest) as caught:
        hub.sign_in("")
    assert isinstance(caught.value, ValueError)
    assert "huggingface.co/settings/tokens" in str(caught.value)


def test_a_rejected_token_is_a_bad_request(monkeypatch):
    monkeypatch.setattr(hub, "whoami", lambda tok=None: hub.HubAuth(signed_in=False))
    with pytest.raises(BadRequest, match="rejected"):
        hub.sign_in("hf_not_a_real_token")


# --------------------------------------------------- reading the token file


def test_an_unreadable_token_file_is_reported_instead_of_swallowed(
    tmp_path, monkeypatch, caplog
):
    """`_write_private`'s docstring records what this cost: truncated JSON
    "which the reader swallowed silently, so the user was signed out with no
    message and no way to tell why". The message has to exist somewhere."""
    truncated = tmp_path / "hub.json"
    truncated.write_text('{"token": "hf_ab', encoding="utf-8")
    monkeypatch.setattr(hub, "_config_path", lambda: truncated)
    monkeypatch.setattr(hub, "_cli_token_paths", lambda: [tmp_path / "absent"])
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    with caplog.at_level(logging.WARNING, logger="modelmri"):
        assert hub._read_stored_token() == (None, None)

    said = " ".join(r.getMessage() for r in caplog.records)
    assert "could not read the stored HuggingFace token" in said
    assert str(truncated) in said, "the message must name the file to look at"


def test_the_account_panel_survives_a_machine_with_no_home_directory(monkeypatch):
    """`Path.home()` RAISES RuntimeError where there is no home to expand `~`
    against — a container on an arbitrary UID, a Windows service account.

    This is not hypothetical tidying: `_read_stored_token` called it raw, and
    `whoami` calls `_read_stored_token` OUTSIDE its own try, so on such a
    machine `whoami` raised straight through its "Never raises" docstring and
    `/api/hub/auth` — which has no handler at all — answered 500 for a panel
    whose honest answer is "signed out".
    """

    def no_home():
        raise RuntimeError("Could not determine home directory.")

    monkeypatch.setattr(Path, "home", staticmethod(no_home))
    monkeypatch.setattr(hub, "_config_path", lambda: Path("nonexistent") / "hub.json")

    assert hub._cli_token_paths(), "hf_home() still gives one candidate"
    hub._read_stored_token()  # must not raise
    assert isinstance(hub.whoami(), hub.HubAuth)


def test_a_failed_sign_out_says_why_you_are_still_signed_in(
    tmp_path, monkeypatch, caplog
):
    """The answer stays honest — you ARE still signed in — but silence about
    it meant clicking Sign out did nothing, visibly, for no stated reason."""
    stubborn = tmp_path / "hub.json"
    stubborn.write_text(json.dumps({"token": "hf_x"}), encoding="utf-8")
    monkeypatch.setattr(hub, "_config_path", lambda: stubborn)
    monkeypatch.setattr(hub, "whoami", lambda tok=None: hub.HubAuth(signed_in=True))

    def wont_delete(*_a, **_k):
        raise PermissionError(32, "The process cannot access the file")

    monkeypatch.setattr(Path, "unlink", wont_delete)

    with caplog.at_level(logging.WARNING, logger="modelmri"):
        assert hub.sign_out().signed_in is True

    said = " ".join(r.getMessage() for r in caplog.records)
    assert "still signed in" in said
    assert str(stubborn) in said


# -------------------------------------------------------- reading a config


def test_a_config_we_could_not_read_is_not_called_absent(tmp_path):
    """ "not a transformers model (no config.json)" is a claim about the repo.
    A config that exists and would not parse — permissions, a cloud
    placeholder, a cache entry mid-write — used to produce that same sentence,
    so the picker told people their model was not a model because their sync
    client had not finished."""
    nothing_there = tmp_path / "empty"
    nothing_there.mkdir()
    config, unreadable = discover._read_config(nothing_there)
    assert (config, unreadable) == (None, False)
    assert discover._describe(config, "org/repo", unreadable)[1] == (
        "not a transformers model (no config.json)"
    )

    wont_parse = tmp_path / "half-synced"
    wont_parse.mkdir()
    (wont_parse / "config.json").write_text('{"architectures": [', encoding="utf-8")
    config, unreadable = discover._read_config(wont_parse)
    assert (config, unreadable) == (None, True)
    loadable, note = discover._describe(config, "org/repo", unreadable)
    assert loadable is False, "still not offered as something that will load"
    assert "could not read its config.json" in note
    assert "no config.json" not in note


def test_a_cache_entry_whose_snapshots_will_not_list_is_not_called_absent(
    tmp_path, monkeypatch
):
    """The failure the survey actually measured: `iterdir` on a HuggingFace
    cache entry raising while the entry is written or removed underneath us."""
    entry = tmp_path / "models--org--name"
    (entry / "snapshots").mkdir(parents=True)

    real_iterdir = Path.iterdir

    def refuse_snapshots(self):
        if self.name == "snapshots":
            raise PermissionError(13, "Permission denied")
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", refuse_snapshots)
    config, unreadable = discover._read_config(entry)
    assert (config, unreadable) == (None, True)


def test_a_real_causal_lm_is_still_loadable(tmp_path):
    """The other half of the above: the common path must not have moved."""
    good = tmp_path / "model"
    good.mkdir()
    (good / "config.json").write_text(
        json.dumps({"architectures": ["LlamaForCausalLM"]}), encoding="utf-8"
    )
    config, unreadable = discover._read_config(good)
    assert unreadable is False
    assert discover._describe(config, "org/repo", unreadable) == (
        True,
        "cached, loads offline",
    )
