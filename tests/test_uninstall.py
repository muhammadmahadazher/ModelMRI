"""Removing a tool should be as considerate as installing it.

`modelmri uninstall` deletes what ModelMRI wrote, and nothing else. The two
ways it could betray that are (a) taking the shared HuggingFace cache with it,
which belongs to `transformers` and every other tool on the machine, and
(b) deleting without saying what. Both are tested here.

Every path is resolved through `paths`, per-platform and per-account, so these
tests point MODELMRI_HOME at a temp directory rather than asserting any
literal location — the same reason the product has no hardcoded paths.
"""

from __future__ import annotations

import pytest

from modelmri import cli


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A throwaway ModelMRI home, plus a throwaway HuggingFace cache."""
    root = tmp_path / "mri"
    monkeypatch.setenv("MODELMRI_HOME", str(root))
    hub = tmp_path / "hf" / "hub"
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))

    from modelmri import paths

    for d in (paths.data_dir(), paths.config_dir(), paths.cache_dir()):
        d.mkdir(parents=True, exist_ok=True)
        (d / "something").write_bytes(b"x" * 1000)
    hub.mkdir(parents=True, exist_ok=True)
    (hub / "a-model.bin").write_bytes(b"y" * 4000)
    return root, hub


def test_it_removes_what_modelmri_wrote(home, capsys):
    root, hub = home
    assert root.exists()

    assert cli.uninstall(yes=True) == 0
    out = capsys.readouterr().out

    assert not root.exists() or not any(root.rglob("something"))
    assert "freed" in out
    # It cannot remove itself, and says so rather than implying it did.
    assert "pip uninstall modelmri" in out


def test_the_shared_model_cache_is_not_collateral(home, capsys):
    """The HuggingFace cache is not ours. Someone removing ModelMRI has not
    asked to lose the weights `transformers` also reads."""
    _, hub = home
    cli.uninstall(yes=True)
    assert (hub / "a-model.bin").exists()
    out = capsys.readouterr().out
    assert "SHARED" in out and "Left alone" in out


def test_the_model_cache_goes_only_when_asked(home, capsys):
    _, hub = home
    cli.uninstall(yes=True, models=True)
    assert not (hub / "a-model.bin").exists()
    assert "Deleting it, as asked" in capsys.readouterr().out


def test_it_says_what_it_will_delete_before_deleting(home, capsys, monkeypatch):
    """A destructive command that prints nothing is a destructive command
    nobody can check. Declining must also leave everything in place."""
    root, _ = home
    monkeypatch.setattr("builtins.input", lambda _="": "n")

    assert cli.uninstall() == 1
    out = capsys.readouterr().out

    assert str(root) in out  # the real, resolved location — not a guess
    assert "MB" in out  # with sizes, so the choice is informed
    assert "nothing deleted" in out
    assert any(root.rglob("something"))


def test_declining_is_the_default_on_a_closed_stdin(home, monkeypatch):
    """Piped input hits EOF. That must read as "no", never as consent."""
    root, _ = home

    def eof(_prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    assert cli.uninstall() == 1
    assert any(root.rglob("something"))


def test_it_is_calm_when_there_is_nothing_to_remove(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MODELMRI_HOME", str(tmp_path / "never-used"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "no-hub"))
    assert cli.uninstall(yes=True) == 0
    assert "nothing to remove" in capsys.readouterr().out
