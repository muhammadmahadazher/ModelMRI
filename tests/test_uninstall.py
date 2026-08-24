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


def test_the_venv_figure_comes_from_the_same_walk_as_the_data_directory(
    home, monkeypatch, capsys
):
    """The action expert's venv lives INSIDE the data directory, and both
    lines used to be measured by their own full traversal of it.

    MEASURED on this project's own machine: the data directory is 1.23 GB in
    38,664 entries, 38,635 of which ARE the venv. Walking it took 68.4 s and
    walking the venv took another 78.9 s — 147 of the ~124 s `uninstall` spent
    before printing anything. One walk answers both, and this asserts the tree
    is opened once rather than twice.
    """
    from modelmri import policy

    _root, _hub = home
    venv = policy.venv_dir()
    (venv / "lib").mkdir(parents=True, exist_ok=True)
    (venv / "lib" / "torch.so").write_bytes(b"z" * 7000)

    opened: list[str] = []
    real = cli.os.scandir

    def counting(path):
        opened.append(str(path))
        return real(path)

    def eof(_prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    monkeypatch.setattr(cli.os, "scandir", counting)
    # Declined, so this measures the DISCLOSURE only. The deletion loop walks
    # each target again to report what it freed, which is a real second walk
    # of a tree that is being removed either way.
    assert cli.uninstall() == 1
    said = capsys.readouterr().out

    assert "0.00 GB is the action expert's own" in said, said
    assert len(opened) == len(set(opened)), (
        f"a directory was walked twice while listing what would be deleted: "
        f"{sorted(x for x in opened if opened.count(x) > 1)}"
    )


def test_the_totals_survived_the_switch_to_scandir(tmp_path):
    """`rglob` + `lstat` + `is_symlink()` asked the OS twice per entry for one
    answer — `S_ISREG` is already false for a symlink under `lstat`, so the
    second call decided nothing. This asserts the replacement counts the same
    bytes, including the symlink rule the HuggingFace cache depends on: an
    8 GB cache was once reported at 20 GB by following `snapshots/` links back
    into `blobs/`.
    """
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    (blobs / "weights").write_bytes(b"w" * 5000)
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "small").write_bytes(b"s" * 300)

    snaps = tmp_path / "snapshots"
    snaps.mkdir()
    try:
        (snaps / "model.safetensors").symlink_to(blobs / "weights")
    except (OSError, NotImplementedError):
        pytest.skip("this account cannot create symlinks")

    total, marked = cli._walk_bytes(tmp_path, mark=tmp_path / "nested")
    assert total == 5300, "the symlink was followed and its target double-counted"
    assert marked == 300
    assert cli._tree_bytes(tmp_path) == total
