"""Where a corpus may be read from, when the request arrived over HTTP.

CodeQL raised `py/path-injection` against the flow that ends in
`sweep.load_prompts`'s `read_text` — twice, as #409 and then #418. The
previous answer was that the finding is a false positive because the routes
that reach it carry `_not_from_this_machine`. That is true and it is also not
the whole answer: that check settles WHO may ask, and this settles WHERE they
may point, which is a different question. A caller who is genuinely on this
machine can still send `../../../../etc/shadow`, and "a local tool that will
read any path on the filesystem on request" is a nastier primitive than it
looks.

So there is a boundary now, and these are the two halves of it:

  the feature still works   reading YOUR corpus is the thing people install
                            this for. Home, the working directory, the temp
                            directory and `MODELMRI_CORPUS_DIRS` are all in.
  traversal does not        `..` out of those is refused, and the refusal
                            names the roots and the escape hatch rather than
                            saying "denied".

And the CLI is deliberately outside it. `modelmri sweep --prompts` is the
person at the keyboard naming their own file; refusing a path its own user
just typed protects nobody and breaks the tool.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from modelmri import feature_corpus as fc
from modelmri import paths, sweep
from modelmri.errors import BadRequest


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    f = tmp_path / "corpus.txt"
    f.write_text("first prompt\nsecond prompt\n", encoding="utf-8")
    return f


# ---------------------------------------------------------- it still works


def test_a_corpus_under_a_root_is_read(corpus):
    """`tmp_path` is under the system temp directory on every runner, which is
    in the roots for exactly that reason — and because it is where a
    downloaded corpus lands."""
    texts, label = fc.load_corpus(corpus)
    assert texts == ["first prompt", "second prompt"]
    assert label == "corpus.txt"


def test_the_working_directory_is_a_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "here.txt").write_text("one\n", encoding="utf-8")
    assert fc.load_corpus("here.txt")[0] == ["one"]


def test_a_named_directory_is_a_root(tmp_path, monkeypatch):
    """The escape hatch the refusal advertises has to be real, or the sentence
    is telling the reader to do something that does not work."""
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "c.txt").write_text("x\n", encoding="utf-8")
    # Somewhere genuinely outside every default root, so the variable is the
    # only reason this can be read.
    monkeypatch.setattr(paths, "_home", lambda: None)
    monkeypatch.setattr(Path, "cwd", staticmethod(lambda: Path(__file__).parent))
    monkeypatch.setenv("MODELMRI_CORPUS_DIRS", str(outside))
    assert fc.load_corpus(outside / "c.txt")[0] == ["x"]


def test_the_variable_takes_several_directories(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        d.mkdir()
    monkeypatch.setenv("MODELMRI_CORPUS_DIRS", os.pathsep.join([str(a), str(b)]))
    roots = paths.corpus_roots()
    assert a.resolve() in roots and b.resolve() in roots


def test_a_tilde_in_the_variable_is_expanded(monkeypatch):
    """`models_dirs` records this exact bug: `~/models` became a literal
    directory named `~` under the cwd, and every file in it was refused."""
    monkeypatch.setenv("MODELMRI_CORPUS_DIRS", "~/corpora")
    assert not any("~" in str(r) for r in paths.corpus_roots())


# ------------------------------------------------------ traversal does not


@pytest.mark.parametrize(
    "escape",
    [
        "/etc/shadow",
        "/etc/passwd",
        "//server/share/secrets.txt",
    ],
)
def test_an_absolute_path_outside_every_root_is_refused(escape, monkeypatch):
    # Pinned so the test does not depend on where it happens to be run from:
    # under a cwd of `/` these would legitimately be inside a root.
    monkeypatch.setattr(Path, "cwd", staticmethod(lambda: Path(__file__).parent))
    with pytest.raises(BadRequest) as caught:
        fc.load_corpus(escape)
    assert "resolves outside" in caught.value.sentence


def test_the_refusal_names_the_roots_and_the_way_out(monkeypatch):
    """A refusal with no next step is a wall. This one has to say where a
    corpus MAY live and how to add somewhere else."""
    monkeypatch.setattr(Path, "cwd", staticmethod(lambda: Path(__file__).parent))
    with pytest.raises(BadRequest) as caught:
        fc.load_corpus("/etc/shadow")
    said = caught.value.sentence
    assert "MODELMRI_CORPUS_DIRS" in said
    # And the roots themselves, so the reader can see which one to move under.
    for root in paths.corpus_roots():
        assert str(root) in said


def test_dot_dot_is_collapsed_before_the_check_not_after(tmp_path, monkeypatch):
    """THE ORDER IS THE WHOLE THING. Checking the string as written and then
    opening the resolved path is how a traversal gets through: `<root>/../x`
    starts with `<root>` and opens `x`. Normalising first means the file that
    gets read is the file that gets checked."""
    # Every default root neutralised except the one being tested. `tmp_path`
    # lives UNDER the system temp directory, which is itself a root — so
    # without this the traversal target is legitimately inside one and the
    # test would be asserting against the wrong thing.
    import tempfile

    monkeypatch.setattr(paths, "_home", lambda: None)
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path / "nowhere"))
    monkeypatch.setattr(Path, "cwd", staticmethod(lambda: tmp_path))
    monkeypatch.delenv("MODELMRI_CORPUS_DIRS", raising=False)
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret\n", encoding="utf-8")
    try:
        with pytest.raises(BadRequest) as caught:
            fc.load_corpus(tmp_path / ".." / "outside.txt")
        # Named by what it RESOLVED to, not by what was typed.
        assert "outside.txt" in caught.value.sentence
    finally:
        outside.unlink(missing_ok=True)


def test_a_path_the_os_cannot_resolve_is_a_refusal_not_a_traceback():
    with pytest.raises(BadRequest) as caught:
        fc.load_corpus("\0no-such-thing")
    assert "resolve" in caught.value.sentence


# ------------------------------------------- and the CLI is not behind it


def test_the_library_reader_has_no_boundary(tmp_path, monkeypatch):
    """`modelmri sweep --prompts` is you naming your own file.

    If this ever starts refusing, the CLI has been given a boundary that
    protects nobody — the process can already read anything its user can —
    while breaking the one workflow that has no other way to pass a corpus.
    """
    import tempfile

    monkeypatch.setattr(paths, "_home", lambda: None)
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path / "nowhere"))
    monkeypatch.setattr(Path, "cwd", staticmethod(lambda: Path(__file__).parent))
    monkeypatch.delenv("MODELMRI_CORPUS_DIRS", raising=False)
    outside = tmp_path / "mine.txt"
    outside.write_text("a\nb\n", encoding="utf-8")
    assert sweep.load_prompts(outside) == ["a", "b"]


def test_a_sibling_directory_that_merely_starts_the_same_is_refused(
    tmp_path, monkeypatch
):
    """THE CLASSIC PREFIX BUG. A containment test written as a bare
    `startswith` accepts `/home/anabel` for a root of `/home/ana` — a
    different person's home directory, matched because one name is a prefix
    of the other. The separator guard is what stops it.

    This test exists because the guard was VACUOUS when it was written:
    removing it left all twelve tests green, so the comment beside it was a
    claim with nothing behind it.
    """
    import tempfile

    root = tmp_path / "ana"
    root.mkdir()
    sibling = tmp_path / "anabel"
    sibling.mkdir()
    (sibling / "theirs.txt").write_text("not yours\n", encoding="utf-8")

    monkeypatch.setattr(paths, "_home", lambda: None)
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path / "nowhere"))
    monkeypatch.setattr(Path, "cwd", staticmethod(lambda: root))
    monkeypatch.delenv("MODELMRI_CORPUS_DIRS", raising=False)

    with pytest.raises(BadRequest) as caught:
        fc.load_corpus(sibling / "theirs.txt")
    assert "resolves outside" in caught.value.sentence


def test_the_root_itself_is_inside_the_root(tmp_path, monkeypatch):
    """The other end of the same guard: `resolved == prefix` has to be
    accepted, or a corpus named as the root directory itself is refused for
    being exactly where it is allowed to be."""
    import tempfile

    monkeypatch.setattr(paths, "_home", lambda: None)
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path / "nowhere"))
    monkeypatch.setattr(Path, "cwd", staticmethod(lambda: tmp_path))
    monkeypatch.delenv("MODELMRI_CORPUS_DIRS", raising=False)
    # `resolve_corpus` answers about the path, not about what is at it, so a
    # directory resolves fine here and the reader below is what refuses it.
    assert fc.resolve_corpus(tmp_path) == tmp_path.resolve()
