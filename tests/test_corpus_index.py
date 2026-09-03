# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""A typed path still works, and it is never used to build a path.

Those two sentences look contradictory and are the whole design. A corpus
field that only takes an id takes away the way people actually point this at
their own file; one that takes a path and opens it is a path-injection
primitive with a text box in front of it. So the typed string is used ONLY as
the right-hand side of a name comparison: the server descends from a root, one
`os.scandir` per component, and the `Path` it returns came out of its own
directory reads.

Five attempts at the other design — fence the string with a normalise-and-
check barrier — all failed to satisfy CodeQL (#418, #431, #432), because
`Path.resolve()` reads the filesystem and is therefore the sink itself. No
check placed around a value that reaches it counts as a barrier.

What is pinned here is behaviour, not the analyser's opinion: a path inside
the roots opens, a path outside does not, a link that leads out does not, and
the value that gets opened is one `scandir` produced.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from modelmri import corpus_index as ci
from modelmri import paths
from modelmri.errors import BadRequest


@pytest.fixture
def only_root(tmp_path, monkeypatch):
    """`tmp_path` as the one corpus root, and nothing else."""
    import tempfile

    monkeypatch.setattr(paths, "_home", lambda: None)
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path / "nowhere"))
    # `chdir`, not a patched `Path.cwd`. `corpus_roots()` reads `Path.cwd()`
    # and `resolve_typed` anchors a relative path with `os.path.abspath`,
    # which reads `os.getcwd()` — patching one and not the other makes the
    # two disagree in the test and nowhere else, which is a fixture inventing
    # a bug rather than finding one.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MODELMRI_CORPUS_DIRS", raising=False)
    ci._INDEX.clear()
    return tmp_path


# ------------------------------------------------------- a path still works


def test_a_typed_path_inside_a_root_resolves(only_root):
    deep = only_root / "a" / "b" / "c"
    deep.mkdir(parents=True)
    f = deep / "corpus.txt"
    f.write_text("one\n", encoding="utf-8")
    assert ci.resolve_typed(str(f)) == Path(os.path.realpath(f))


def test_depth_is_not_capped_for_a_typed_path(only_root):
    """`MAX_DEPTH` bounds the LISTING, which is a different job. A path you
    typed is descended to however deep it is, one directory read per level —
    capping it would refuse a file the reader can see in their own explorer."""
    deep = only_root
    for part in ("one", "two", "three", "four", "five", "six"):
        deep = deep / part
    deep.mkdir(parents=True)
    f = deep / "deep.txt"
    f.write_text("x\n", encoding="utf-8")
    assert ci.MAX_DEPTH < 6
    assert ci.resolve_typed(str(f)).name == "deep.txt"


def test_a_relative_path_is_taken_against_the_working_directory(only_root):
    (only_root / "here.txt").write_text("x\n", encoding="utf-8")
    assert ci.resolve_typed("here.txt").name == "here.txt"


def test_the_returned_path_is_the_one_scandir_found(only_root):
    """The value that reaches the reader is the real entry on disk.

    Asked for by its exact name, so this runs the same everywhere. What it
    pins is that the object handed back is the directory's own entry rather
    than a `Path` assembled from the request; the mechanism test at the bottom
    of this file is what watches the `scandir` calls that make that true.
    """
    f = only_root / "Corpus.TXT"
    f.write_text("one\n", encoding="utf-8")
    got = ci.resolve_typed(str(only_root / "Corpus.TXT"))
    assert got.name == "Corpus.TXT"
    assert got == Path(os.path.realpath(f))


def test_a_name_differing_only_in_case_follows_this_platforms_rule(only_root):
    """AND THIS IS WHY THERE IS A CI MATRIX.

    The first version of this test asserted the Windows answer
    unconditionally. It passed on the machine it was written on and failed on
    Linux and on both macOS runners, which is the whole argument for not
    reporting a green local run as a green build.

    The descent compares names through `os.path.normcase`, which is the
    standard library's own statement of whether a platform folds case in
    paths: real folding on Windows, identity on POSIX. The rule is therefore
    not invented here, and this test asks `normcase` what to expect rather
    than asking `sys.platform` -- which would be a second opinion about the
    same question, free to disagree with the code.

    HONEST LIMIT, stated rather than papered over. `normcase` is a fact about
    the PLATFORM, not about the VOLUME. macOS's default APFS is
    case-insensitive, so a reader there who types `corpus.txt` for a file
    named `Corpus.TXT` is refused here even though every other tool on their
    machine opens it. Closing that would mean retrying case-insensitively and
    confirming with `os.path.samefile` on a path built from the request --
    which is exactly the tainted filesystem access this module was rewritten
    over six attempts to remove. A refusal naming the file it could not find
    is the cheaper wrong answer than reintroducing that.
    """
    (only_root / "Corpus.TXT").write_text("one\n", encoding="utf-8")
    asked = str(only_root / "corpus.txt")
    folds = os.path.normcase("A") == os.path.normcase("a")
    if folds:
        # The file's OWN spelling comes back, never the caller's.
        assert ci.resolve_typed(asked).name == "Corpus.TXT"
    else:
        # Refused, and NOT quietly resolved to the neighbour. On a
        # case-sensitive filesystem `corpus.txt` and `Corpus.TXT` are two
        # different files, and handing back the one nobody asked for is how a
        # reader ends up measuring the wrong text and never knowing.
        with pytest.raises(BadRequest) as caught:
            ci.resolve_typed(asked)
        assert "corpus.txt" in caught.value.sentence


# ------------------------------------------------------- and what does not


def test_a_path_outside_every_root_is_refused(only_root):
    with pytest.raises(BadRequest) as caught:
        ci.resolve_typed("/etc/shadow")
    assert "outside the directories" in caught.value.sentence
    assert "MODELMRI_CORPUS_DIRS" in caught.value.sentence


def test_dot_dot_out_of_a_root_is_refused(only_root):
    outside = only_root.parent / "escaped.txt"
    outside.write_text("secret\n", encoding="utf-8")
    try:
        with pytest.raises(BadRequest):
            ci.resolve_typed(str(only_root / ".." / "escaped.txt"))
    finally:
        outside.unlink(missing_ok=True)


def test_a_missing_component_names_the_component(only_root):
    """A refusal with no next step is a wall. This one says which segment of
    the path it could not find, which is the thing the reader can act on."""
    with pytest.raises(BadRequest) as caught:
        ci.resolve_typed(str(only_root / "nope" / "corpus.txt"))
    assert "nope" in caught.value.sentence


def test_a_link_that_leads_out_is_refused(only_root):
    target_dir = only_root.parent / "elsewhere"
    target_dir.mkdir(exist_ok=True)
    secret = target_dir / "secret.txt"
    secret.write_text("not yours\n", encoding="utf-8")
    link = only_root / "innocent.txt"
    try:
        link.symlink_to(secret)
    except (OSError, NotImplementedError):
        pytest.skip("this machine will not create a symlink without privileges")
    try:
        with pytest.raises(BadRequest) as caught:
            ci.resolve_typed(str(link))
        assert "leads outside" in caught.value.sentence
    finally:
        secret.unlink(missing_ok=True)


def test_an_empty_or_wrong_typed_value_is_refused(only_root):
    for bad in ("", "   ", None, 17):
        with pytest.raises(BadRequest):
            ci.resolve_typed(bad)


# --------------------------------------------------------- the id half


def test_the_listing_mints_an_id_that_resolves_back(only_root):
    f = only_root / "listed.txt"
    f.write_text("x\n", encoding="utf-8")
    listing = ci.scan()
    assert listing.n_found >= 1
    entry = next(c for c in listing.corpora if c.name == "listed.txt")
    assert len(entry.id) == 32
    ci._rebuild()
    assert ci.resolve(entry.id).name == "listed.txt"


def test_an_id_is_not_a_path_and_a_path_is_not_an_id(only_root):
    """`resolve_any` accepts both and must not confuse them. A 32-character
    hex string is an id; anything else is read as a path."""
    f = only_root / "both.txt"
    f.write_text("x\n", encoding="utf-8")
    listing = ci.scan()
    ci._rebuild()
    entry = next(c for c in listing.corpora if c.name == "both.txt")
    assert ci.resolve_any(entry.id).name == "both.txt"
    assert ci.resolve_any(str(f)).name == "both.txt"


def test_an_unknown_id_says_a_path_is_also_accepted(only_root):
    with pytest.raises(BadRequest) as caught:
        ci.resolve("0" * 32)
    assert "corpus/available" in caught.value.sentence


def test_only_the_two_suffixes_the_reader_handles_are_listed(only_root):
    for name in ("a.txt", "b.jsonl", "c.csv", "d.parquet"):
        (only_root / name).write_text("x\n", encoding="utf-8")
    found = {c.name for c in ci.scan().corpora}
    assert {"a.txt", "b.jsonl"} <= found
    assert "c.csv" not in found and "d.parquet" not in found


def test_every_cap_is_reported_rather_than_only_applied(only_root, monkeypatch):
    monkeypatch.setattr(ci, "MAX_FILES", 2)
    for i in range(5):
        (only_root / f"c{i}.txt").write_text("x\n", encoding="utf-8")
    listing = ci.scan()
    assert listing.truncated_files is True
    assert listing.n_found == 2
    said = listing.means()
    assert "CAPPED" in said and "MODELMRI_CORPUS_DIRS" in said


def test_the_descent_reads_a_directory_per_component(only_root, monkeypatch):
    """THE MECHANISM, pinned directly, because the OUTCOME cannot pin it.

    The claim this module rests on is that the path which ends up being opened
    came out of `os.scandir`, not out of the caller's string. This watches the
    calls: every `scandir` argument must be a path this module produced — the
    root, then a previous entry's own `.path` — and the walk must take one
    directory read per component.

    AND HERE IS WHAT IT STILL CANNOT SEE, stated because a test that quietly
    fails to pin its own claim is worse than no test. Replacing
    `step = Path(entry.path)` with `step = here / part` is an EQUIVALENT
    mutation at runtime: `part` only ever gets that far by matching a real
    entry, so on a case-insensitive filesystem the two spellings open the same
    file, and on a case-sensitive one the match required them to be identical.
    No behavioural test can separate them, and mutating it here is correctly
    reported as vacuous.

    The difference is in the DATA FLOW, not the behaviour: `here / part`
    builds a path out of the request's own string, and `Path(entry.path)`
    builds it out of a directory read. That is the whole property this module
    exists for, and it is visible to a taint tracker and to a reader — not to
    a runtime assertion.
    """
    deep = only_root / "one" / "two"
    deep.mkdir(parents=True)
    (deep / "corpus.txt").write_text("x\n", encoding="utf-8")

    seen: list[str] = []
    produced: set[str] = {os.path.normcase(str(only_root))}
    real_scandir = os.scandir

    class Recorder:
        def __init__(self, where):
            seen.append(str(where))
            self._it = real_scandir(where)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self._it.close()
            return False

        def __iter__(self):
            for entry in self._it:
                produced.add(os.path.normcase(entry.path))
                yield entry

    monkeypatch.setattr(ci.os, "scandir", Recorder)
    got = ci.resolve_typed(str(only_root / "one" / "two" / "corpus.txt"))
    assert got.name == "corpus.txt"

    # One read per component, and the first is the root itself.
    assert len(seen) == 3, seen
    assert os.path.normcase(seen[0]) == os.path.normcase(str(only_root))
    # And nothing was ever scanned that this module had not itself produced.
    for where in seen:
        assert os.path.normcase(where) in produced, where
