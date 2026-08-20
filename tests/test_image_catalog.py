"""Finding an image model, and the four things a row here must never claim.

`image_catalog` reads a listing and a disk. Neither is a measurement, which is
exactly what makes it dangerous: every number it produces is rendered beside a
button that starts a multi-gigabyte download, so a wrong one is acted on rather
than noticed.

The module's docstring states four invariants, and this file is written against
those rather than against how they are currently implemented:

  0 bytes is UNKNOWN, never small      -> "0.0 GB" is the click a size column
                                          exists to prevent
  a pipeline tag is a TASK, not an
  architecture                         -> a panel drawn for the wrong family is
                                          a picture of something that does not
                                          exist
  cached is answered from the disk     -> a half-finished download must not
                                          look like a model that is ready
  a Hub that did not answer is a
  refusal, in this project's words     -> and never the library's own text,
                                          which carries paths from this machine

Nothing here touches the network. `image_catalog` does `from . import hub`
INSIDE each function, so the Hub is stubbed by dotted string, and the tests
that assert a refusal arrives BEFORE a request make the stub explode rather
than merely returning nothing.
"""

from __future__ import annotations

import http.client
import json
import urllib.error
from functools import partial
from pathlib import Path

import pytest
from test_no_exception_leaks import CERT, WEIGHTS, leaked  # the same rule

from modelmri import image_catalog, imaging
from modelmri.errors import BadRequest, Refusal

# The families `imaging.detect` can actually return, named through the module's
# own constants so that a task table carrying a typo'd string fails rather than
# comparing equal to another typo.
REAL_FAMILIES = {
    imaging.UNET_DIFFUSION,
    imaging.DIT_DIFFUSION,
    imaging.VIT,
    imaging.CLIP,
    imaging.DETECTION,
    imaging.SEGMENTATION,
    imaging.VLM,
}


# --------------------------------------------------------------- the stubs


def _hub(monkeypatch, payload):
    """Answer every Hub request with `payload`, without opening a socket."""
    monkeypatch.setattr("modelmri.hub.token", lambda: None)
    monkeypatch.setattr("modelmri.hub._api", lambda *_a, **_k: payload)


def _hub_fails(monkeypatch, err):
    """The Hub does not answer, in one of the shapes it really fails in."""

    def _boom(*_a, **_k):
        raise err

    monkeypatch.setattr("modelmri.hub.token", lambda: None)
    monkeypatch.setattr("modelmri.hub._api", _boom)


def _no_request_may_happen(monkeypatch):
    """Reaching the network at all is the failure, not the response to it."""

    def _reached(*_a, **_k):
        raise AssertionError("a request was made when nothing should have been")

    monkeypatch.setattr("modelmri.hub._api", _reached)
    monkeypatch.setattr("urllib.request.urlopen", _reached)


def _cache(monkeypatch, *found):
    """What `imaging.scan_cache` reports, so the disk under test is `tmp_path`."""
    monkeypatch.setattr("modelmri.imaging.scan_cache", lambda *_a, **_k: list(found))


def _cache_entry(tmp_path, name, *, weights_mb=0, family=imaging.UNET_DIFFUSION):
    """A snapshot directory in the real shape: configs always, weights maybe.

    `weights_mb=0` is the state that matters — a directory of configs with no
    weight file in it, which is what an interrupted download leaves behind.
    """
    root = tmp_path / name.replace("/", "--")
    (root / "unet").mkdir(parents=True, exist_ok=True)
    (root / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "StableDiffusionPipeline",
                "unet": ["diffusers", "UNet2DConditionModel"],
            }
        ),
        encoding="utf-8",
    )
    (root / "unet" / "config.json").write_text(
        json.dumps({"_class_name": "UNet2DConditionModel", "cross_attention_dim": 768}),
        encoding="utf-8",
    )
    if weights_mb:
        (root / "unet" / "diffusion_pytorch_model.safetensors").write_bytes(
            b"\x00" * (weights_mb * 1_000_000)
        )
    return imaging.ImageModel(
        path=name,
        directory=str(root),
        family=family,
        architecture="UNet2DConditionModel",
        reason="" if family != imaging.UNKNOWN else "no class this knows",
    )


# A repo the Hub publishes real per-dtype counts for, and three that publish
# nothing to go on — the GGUF and pickle case, which is most of them.
SIZED = {
    "id": "someone/sdxl",
    "downloads": 900,
    "likes": 12,
    "lastModified": "2026-02-03T04:05:06.000Z",
    "safetensors": {"parameters": {"F16": 1_000_000}, "total": 1_000_000},
}
UNSIZED = {"id": "someone/gguf-only", "downloads": 40, "lastModified": ""}
UNSIZED_EMPTY = {"id": "someone/no-metadata", "safetensors": {}}
UNSIZED_ZERO = {"id": "someone/zero-total", "safetensors": {"total": 0}}


# ------------------------------------------------- refused before it is sent


def test_a_task_nothing_here_can_open_is_refused_by_name_and_costs_no_request(
    monkeypatch,
):
    """The refusal has to name the alternatives, because a caller who typed a
    tag that does not exist has no other way to learn which ones do — and it
    has to arrive before the request, or an unopenable task costs a round trip
    and returns checkpoints nothing here can load."""
    _no_request_may_happen(monkeypatch)

    with pytest.raises(BadRequest) as caught:
        image_catalog.search(task="text-to-hologram")
    said = str(caught.value)

    assert "text-to-hologram" in said
    for tag in image_catalog.TASKS:
        assert tag in said, f"the refusal did not offer `{tag}` as an alternative"


def test_pricing_a_model_nobody_named_is_refused_rather_than_defaulted(monkeypatch):
    """There is no default worth guessing, and a size quoted for a repo the
    caller did not ask about is a number they would plan around."""
    _no_request_may_happen(monkeypatch)

    for named in ("", "   "):
        with pytest.raises(BadRequest, match="nothing to price"):
            image_catalog.size_of(named)


# ------------------------------------------------- 0 is unknown, never small


def test_a_repo_the_hub_publishes_no_size_for_reports_none_and_never_zero(monkeypatch):
    """THE test in this file.

    `hub.weight_bytes` returns 0 for a repo that publishes nothing to go on —
    GGUF and pickle repos, which is most of them — and 0 is not a size. A row
    carrying 0 renders as "0.0 GB" beside a download button, which reads as
    "this one is tiny, click it" for a model whose real weight is unknown and
    may be 1.5 TB. That is exactly the click a size column exists to prevent.

    Asserted against `weight_bytes` itself so the test states the translation
    rather than the value: 0 in, `None` out.
    """
    from modelmri import hub

    _hub(monkeypatch, [SIZED, UNSIZED, UNSIZED_EMPTY, UNSIZED_ZERO])
    _cache(monkeypatch)

    rows = {r["id"]: r for r in image_catalog.search()}

    for raw in (UNSIZED, UNSIZED_EMPTY, UNSIZED_ZERO):
        assert hub.weight_bytes(raw) == 0, "the fixture is not an unsized repo"
        assert rows[raw["id"]]["size_bytes"] is None, (
            f"{raw['id']} published no size and was reported as a number"
        )

    assert hub.weight_bytes(SIZED) == 2_000_000
    assert rows["someone/sdxl"]["size_bytes"] == 2_000_000
    assert not any(r["size_bytes"] == 0 for r in rows.values()), (
        "a row carrying 0 renders as 0.0 GB, which is a claim and not an unknown"
    )


def test_a_repo_with_no_size_metadata_says_unknown_rather_than_small(monkeypatch):
    """The same rule at the other sink. `size_of` is what a reader asks before
    committing to a download, so "no metadata" has to be a sentence rather than
    a small number with no caveat."""
    _hub(monkeypatch, dict(UNSIZED))
    _cache(monkeypatch)

    priced = image_catalog.size_of("someone/gguf-only")
    assert priced["size_bytes"] is None
    assert "UNKNOWN rather than small" in priced["means"]
    assert "0.00 GB" not in priced["means"]


# ------------------------------------------- a tag is a task, not a family


def test_a_row_names_the_families_a_tag_allows_and_never_the_models_own(monkeypatch):
    """`text-to-image` covers a UNet and a DiT, and their cross-attention is in
    different places. Only the checkpoint's own config can settle which one a
    repo is, so a listing row states what the tag is CONSISTENT with — under a
    key whose name says so — and leaves the architecture to `imaging.detect`.

    The key name is load-bearing: a reader who sees `family` believes it.
    """
    _hub(monkeypatch, [SIZED])
    _cache(monkeypatch)

    (row,) = image_catalog.search(task="text-to-image")

    assert "families_possible" in row
    assert "family" not in row, "a row that names a family is claiming one"
    assert row["families_possible"] == list(
        image_catalog.TASKS["text-to-image"]["families"]
    )
    for family in row["families_possible"]:
        assert family in REAL_FAMILIES, f"`{family}` is not a family imaging returns"


def test_every_task_offered_names_families_imaging_can_actually_return():
    """Structural on purpose, so a task added with a typo'd family fails here
    rather than in a picker that renders it.

    `imaging.label` swallows a name it does not know — it falls through to the
    UNKNOWN sentence rather than echoing an identifier at a reader — so a
    misspelt family would otherwise reach the UI silently labelled "an
    architecture this does not recognise" against a task that works fine.
    """
    offered = image_catalog.tasks()

    assert [row["task"] for row in offered] == list(image_catalog.TASKS)
    assert image_catalog.DEFAULT_TASK in image_catalog.TASKS

    for row in offered:
        assert row["label"] and row["means"], row["task"]
        assert row["families"], f"{row['task']} offers no families at all"
        for family in row["families"]:
            assert family in REAL_FAMILIES, f"{row['task']} names `{family}`"
            assert family != imaging.UNKNOWN
            assert imaging.label(family) != imaging.label(imaging.UNKNOWN), (
                f"{row['task']} names a family `imaging.label` does not know"
            )


# ------------------------------------------- the Hub did not answer, in words


# The four failure shapes `hub.search` documents, each carrying text of its own
# that must not survive into the refusal. A captive portal, a corporate proxy
# and a flaky TLS terminator produce these; only the first is a `URLError`.
UNREACHABLE = [
    (urllib.error.URLError(f"getaddrinfo failed, certs at {CERT}"), "getaddrinfo"),
    (TimeoutError(f"the read timed out loading {WEIGHTS}"), "timed out"),
    (http.client.RemoteDisconnected(f"remote end closed, cache {WEIGHTS}"), "remote"),
    (http.client.BadStatusLine(f"HTTP/1.1 nonsense from {CERT}"), "nonsense"),
    (http.client.IncompleteRead(b"", 4096), "IncompleteRead"),
]


@pytest.mark.parametrize(
    ("err", "its_own_words"),
    UNREACHABLE,
    ids=["urlerror", "stalled", "closed-mid-read", "bad-status-line", "truncated"],
)
def test_an_unreachable_hub_is_a_refusal_that_quotes_none_of_the_library(
    monkeypatch, err, its_own_words
):
    """Two claims, and the second is the one `test_no_exception_leaks` exists
    for: the failure has to arrive as a `Refusal` rather than a crash, and the
    sentence has to be this project's own.

    `str(URLError)` is machinery talking to itself, and library text routinely
    carries absolute paths from the machine the server is running on — which is
    published straight to a browser at the other end of these routes. The real
    exception belongs in the terminal, and the refusal says where that is.

    Both callers are checked, because a rule enforced at one sink and not the
    other is a rule that half holds.
    """
    assert isinstance(
        err, (urllib.error.URLError, OSError, http.client.HTTPException)
    ), "the fixture is not a shape the Hub actually fails in"
    _hub_fails(monkeypatch, err)

    # The functions themselves rather than lambdas wrapping them: a
    # zero-argument `lambda: f()` IS `f`, and the indirection only hides which
    # sink a failure came from. `partial` carries the one argument the second
    # sink needs.
    for asking in (
        image_catalog.search,
        partial(image_catalog.size_of, "someone/sdxl"),
    ):
        with pytest.raises(Refusal) as caught:
            asking()
        said = str(caught.value)
        assert not leaked(said), f"the refusal leaked {leaked(said)}"
        assert its_own_words not in said, "the library's own text was relayed"
        # And it still says the actionable half: where the real error went.
        assert "terminal running `modelmri serve`" in said


def test_a_cache_that_cannot_be_read_marks_rows_uncached_rather_than_failing(
    monkeypatch,
):
    """Stated in `_cached_ids`: a cache that cannot be read is a reason to say
    "not cached" for every row, not a reason to fail a search. The listing is
    still the useful part, and the load will answer for itself."""

    def _unreadable(*_a, **_k):
        raise OSError("the cache directory is not readable")

    _hub(monkeypatch, [SIZED, UNSIZED])
    monkeypatch.setattr("modelmri.imaging.scan_cache", _unreadable)

    rows = image_catalog.search()
    assert [r["id"] for r in rows] == ["someone/sdxl", "someone/gguf-only"]
    assert not any(r["cached"] for r in rows)


# ------------------------------------------------- what is actually on a disk


def test_a_cache_entry_holding_configs_and_no_weights_is_not_reported_as_ready(
    tmp_path, monkeypatch
):
    """An interrupted download is a real state, and it is the one a browse list
    cannot show: the configs are there, so the entry identifies perfectly and
    looks like a model. Reporting it at 0 bytes would say it is ready and
    weighs nothing, and the load would then fail after the wait rather than
    before it."""
    _cache(monkeypatch, _cache_entry(tmp_path, "someone/interrupted", weights_mb=0))

    (row,) = image_catalog.local()
    assert row["path"] == "someone/interrupted"
    assert row["size_bytes"] is None, "0 bytes reads as a model that costs nothing"
    assert row["complete"] is False
    # And it is still LISTED. Dropping it would leave a directory on the disk
    # that nothing in the UI accounts for.
    assert row["known"] is True


def test_a_finished_download_reports_the_bytes_that_are_actually_there(
    tmp_path, monkeypatch
):
    """The other branch, so `complete` cannot quietly become "always false" and
    still pass the test above."""
    _cache(monkeypatch, _cache_entry(tmp_path, "someone/whole", weights_mb=4))

    (row,) = image_catalog.local()
    assert row["size_bytes"] == 4_000_000
    assert row["complete"] is True


def test_an_unknown_family_sorts_below_everything_that_was_identified(
    tmp_path, monkeypatch
):
    """A model this cannot open is the least useful row on the page, so it goes
    last — even when it is the biggest thing on the disk, which is the ordering
    a size-first sort would get backwards."""
    _cache(
        monkeypatch,
        _cache_entry(
            tmp_path, "aaa/unrecognised", weights_mb=8, family=imaging.UNKNOWN
        ),
        _cache_entry(tmp_path, "zzz/identified", weights_mb=0),
    )

    rows = image_catalog.local()
    assert [r["path"] for r in rows] == ["zzz/identified", "aaa/unrecognised"]
    assert rows[-1]["known"] is False
    assert rows[-1]["size_bytes"] == 8_000_000, (
        "it sorted last on merit, not because it looked empty"
    )


def test_a_repo_already_on_this_disk_says_nothing_would_be_downloaded(
    tmp_path, monkeypatch
):
    """Whether a repo is here is a question about this machine, so it is
    answered by looking rather than from the listing. A reader pricing
    something they already have needs to be told the download is zero, not the
    file size.

    The entry has REAL WEIGHTS on disk. An earlier version of this test stubbed
    `ImageModel(path=..., directory="")` — an object with no notion of
    completeness — and so codified the bug it was meant to guard: a cache entry
    holding one config was reported as a free download of a 55 GB model.
    """
    _hub(monkeypatch, dict(SIZED))
    _cache(monkeypatch, _cache_entry(tmp_path, "someone/sdxl", weights_mb=4))

    priced = image_catalog.size_of("someone/sdxl")
    assert priced["cached"] is True
    assert "nothing would be downloaded" in priced["means"]
    assert "a download would transfer" not in priced["means"]
    # The size is still reported — what changed is what it COSTS, not what it
    # weighs, and the two are different questions.
    assert priced["size_bytes"] == 2_000_000


def test_a_repo_name_cannot_walk_out_of_the_hub_models_endpoint(monkeypatch):
    """Found by the tests for this module, in this module.

    `size_of` puts the caller's name into an API PATH, and the request carries
    the reader's Hub token. `urllib.parse.quote(name, safe="/")` leaves `..`
    intact, so `../whoami-v2` walks out of `/models/` to a different endpoint
    entirely — with the token attached, from a route that is unauthenticated
    on a server started with `--host 0.0.0.0`.

    Refused on SHAPE before the URL is built, rather than trusted to quoting:
    quoting decides how characters are encoded, and this is about which
    characters are allowed at all.
    """
    import pytest

    from modelmri import image_catalog
    from modelmri.errors import BadRequest

    asked = []
    monkeypatch.setattr(
        "modelmri.hub._api", lambda path, tok=None, **k: asked.append(path) or {}
    )
    monkeypatch.setattr("modelmri.hub.token", lambda: "a-real-token")

    for hostile in ("../whoami-v2", "..%2fwhoami", "/models/x", "C:/x", "~/x", "a/b/c"):
        with pytest.raises(BadRequest) as caught:
            image_catalog.size_of(hostile)
        assert "not a Hub repo id" in str(caught.value)

    assert asked == [], f"a request was built for a hostile name: {asked}"


# ------------------------------------ what "already here" is allowed to mean


def test_a_cache_entry_with_no_weights_is_not_a_free_download(tmp_path, monkeypatch):
    """The worst finding of the review, reproduced live before the fix:

        GET /api/image/size?repo=Qwen/Qwen3.6-27B
        -> "already on this machine, so nothing would be downloaded"

    That repo's cache entry held one 4 KB `config.json`. 55.6 GB was quoted as
    free, and `/api/image/local` called the same repo `complete: false` in the
    same panel — the one route whose whole job is pricing a download before it
    is spent, disagreeing with the row beside it.

    `imaging.scan_cache` admits an entry as soon as a snapshot directory
    identifies, which a config alone satisfies. "Here" has to mean the WEIGHTS
    are here.
    """
    skeleton = _cache_entry(tmp_path, "someone/half-done", weights_mb=0)
    _cache(monkeypatch, skeleton)
    _hub(monkeypatch, SIZED)

    priced = image_catalog.size_of("someone/half-done")
    assert priced["cached"] is False, "an interrupted download is not 'already here'"
    assert priced["partial"] is True
    assert "NO WEIGHTS" in priced["means"]
    assert "nothing would be downloaded" not in priced["means"]


def test_a_complete_entry_still_reports_as_here(tmp_path, monkeypatch):
    """The other branch, so the fix cannot become 'nothing is ever cached'."""
    whole = _cache_entry(tmp_path, "someone/finished", weights_mb=4)
    _cache(monkeypatch, whole)
    _hub(monkeypatch, SIZED)

    priced = image_catalog.size_of("someone/finished")
    assert priced["cached"] is True
    assert priced["partial"] is False
    assert "nothing would be downloaded" in priced["means"]


def test_search_marks_a_skeleton_row_as_partial_not_cached(tmp_path, monkeypatch):
    """`search` shares `_cached_ids`, so the same repo was getting a Load
    button instead of a download, and its gated warning suppressed by
    `gated && !cached` — a licence prompt withheld for weights that have not
    in fact transferred."""
    _cache(monkeypatch, _cache_entry(tmp_path, "someone/sdxl", weights_mb=0))
    _hub(monkeypatch, [SIZED])

    row = image_catalog.search()[0]
    assert row["id"] == "someone/sdxl"
    assert row["cached"] is False
    assert row["partial"] is True


# ------------------------------------- absent is not zero, and caps are said


def test_a_repo_with_no_published_download_count_reports_none_not_zero(monkeypatch):
    """Sorting by downloads is the DEFAULT, so a repo whose count is simply
    absent would sort as the least popular thing on the page — a claim nobody
    made, rendered as a fact."""
    _hub(monkeypatch, [{"id": "someone/quiet"}])
    _cache(monkeypatch)

    row = image_catalog.search()[0]
    assert row["downloads"] is None
    assert row["likes"] is None


def test_a_published_zero_is_still_a_zero(monkeypatch):
    """The other half. A repo the Hub says has 0 downloads HAS 0 downloads,
    and turning that into `None` would lose a real measurement."""
    _hub(monkeypatch, [{"id": "someone/new", "downloads": 0, "likes": 0}])
    _cache(monkeypatch)

    row = image_catalog.search()[0]
    assert row["downloads"] == 0
    assert row["likes"] == 0


def test_a_boolean_count_is_not_read_as_one(monkeypatch):
    """`isinstance(True, int)` is True, which is how a bool becomes 1."""
    _hub(monkeypatch, [{"id": "someone/odd", "downloads": True}])
    _cache(monkeypatch)

    assert image_catalog.search()[0]["downloads"] is None


def test_the_result_cap_travels_with_the_rows(monkeypatch):
    """A cap nobody can see is reported as a complete list. `search` is the
    function that APPLIES the clamp, so it is the one that has to say so."""
    _hub(monkeypatch, [SIZED])
    _cache(monkeypatch)

    rows = image_catalog.search(limit=500)
    assert rows.limit_asked == 500
    assert rows.limit_used == image_catalog.MAX_RESULTS
    # And it is still a list, so nothing that iterates or indexes it changed.
    assert isinstance(rows, list)
    assert rows[0]["id"] == "someone/sdxl"


def test_a_limit_inside_the_cap_reports_no_clamp(monkeypatch):
    _hub(monkeypatch, [SIZED])
    _cache(monkeypatch)

    rows = image_catalog.search(limit=5)
    assert rows.limit_asked == rows.limit_used == 5


def test_a_truncated_list_does_not_compare_equal_to_a_complete_one():
    """Inheriting `list.__eq__` compared the ROWS alone, so a complete list of
    50 and a list of 50 truncated from 200 were equal — the silent-truncation
    defect this class exists to prevent, reappearing in the comparison
    operator. CodeQL flagged it as attributes added without `__eq__`."""
    from modelmri.image_catalog import _Rows

    rows = [{"id": "someone/sdxl"}]
    complete = _Rows(rows, limit_asked=50, limit_used=50, cache_capped=False)
    truncated = _Rows(rows, limit_asked=200, limit_used=50, cache_capped=False)
    half_read = _Rows(rows, limit_asked=50, limit_used=50, cache_capped=True)
    identical = _Rows(rows, limit_asked=50, limit_used=50, cache_capped=False)

    assert complete != truncated
    assert complete != half_read
    assert complete == identical
    # `!=` is inherited from `list` unless overridden, so it would have
    # disagreed with `__eq__` above.
    assert not (complete != identical)
    # Against a plain list it still compares as a list, so a test may write
    # `rows == [...]` and mean it.
    assert complete == rows


# ---------------------------------------------------------------------------
# Models in ORDINARY folders.
#
# `local()` reads the Hub cache, which is the wrong question for somebody who
# cloned a checkpoint into their project directory — exactly the models no
# registry knows about, and so exactly the ones a browse list most needs to
# find.
# ---------------------------------------------------------------------------


def _checkpoint(directory, *, weights=b"x" * 2048):
    """A directory that looks like a diffusers pipeline, on disk.

    Written rather than mocked: the thing under test is a filesystem walk, and
    a walk stubbed at the filesystem is a test of the stub.
    """
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model_index.json").write_text(
        json.dumps({"_class_name": "StableDiffusionPipeline"}), encoding="utf-8"
    )
    unet = directory / "unet"
    unet.mkdir(exist_ok=True)
    (unet / "config.json").write_text(json.dumps({}), encoding="utf-8")
    (unet / "diffusion_pytorch_model.bin").write_bytes(weights)
    return directory


def test_a_checkpoint_in_a_plain_folder_is_found(tmp_path):
    here = _checkpoint(tmp_path / "workdir" / "my-sd-checkpoint")

    out = image_catalog.discovered(roots=[tmp_path / "workdir"])

    paths = [row["path"] for row in out["models"]]
    assert any(Path(p) == here for p in paths), paths


def test_the_roots_it_looked_in_are_reported(tmp_path):
    """ "Nothing found" without "and here is where I looked" tells somebody
    their model is missing when the truth may be that the directory holding it
    was never searched."""
    empty = tmp_path / "nothing-here"
    empty.mkdir()

    out = image_catalog.discovered(roots=[empty])

    assert out["models"] == []
    assert out["roots"] == [str(empty)]


def test_a_folder_that_is_not_an_image_model_is_not_listed(tmp_path):
    """A causal LM is a DETERMINATION, not a gap. Listing every text model as
    an unidentified image model buries the one pipeline on the disk."""
    text = tmp_path / "some-llm"
    text.mkdir()
    (text / "config.json").write_text(
        json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}),
        encoding="utf-8",
    )
    (text / "model.safetensors").write_bytes(b"x" * 512)

    out = image_catalog.discovered(roots=[tmp_path])

    assert [row["path"] for row in out["models"]] == []


def test_a_walk_that_ran_out_of_budget_says_so(tmp_path, monkeypatch):
    """A list capped by a clock and reported as complete is the silent-cap
    defect. The flag is returned, not logged."""
    _checkpoint(tmp_path / "one")

    from modelmri import discover

    monkeypatch.setattr(discover, "scan", lambda root, **kw: ([], True), raising=True)
    out = image_catalog.discovered(roots=[tmp_path])

    assert out["truncated"] is True


def test_one_unreadable_cache_entry_does_not_take_down_the_listing(tmp_path):
    """MEASURED: a cache holding one healthy ViT plus one `model_index.json`
    whose contents were the JSON array `[1, 2, 3]` made GET
    /api/image/available and GET /api/image/local both answer 500.

    `json.loads` succeeds on an array, so a non-mapping sailed past every
    caller's `is None` check into `.get()`. One malformed directory therefore
    made the whole "what image models are on this disk" listing unusable, with
    a message naming nothing the reader could fix.
    """
    from modelmri import imaging

    good = tmp_path / "models--google--vit-base-patch16-224" / "snapshots" / "abc"
    good.mkdir(parents=True)
    (good / "config.json").write_text(
        json.dumps({"architectures": ["ViTForImageClassification"], "image_size": 224}),
        encoding="utf-8",
    )
    (good / "model.safetensors").write_bytes(b"\0" * 1024)

    bad = tmp_path / "models--auditco--badpipe" / "snapshots" / "abc"
    bad.mkdir(parents=True)
    # Valid JSON. Not an object. That is the whole test.
    (bad / "model_index.json").write_text("[1, 2, 3]", encoding="utf-8")

    found = imaging.scan_cache(hub=tmp_path)
    by_path = {m.path: m for m in found}

    # The walk completes. The healthy checkpoint identifies.
    assert by_path["google/vit-base-patch16-224"].known is True

    # The malformed one is LISTED, not hidden — `detect` returns a row
    # carrying a reason for anything it cannot read, and a directory that
    # silently vanishes from the listing is a worse answer than one that
    # explains itself.
    bad_row = by_path["auditco/badpipe"]
    assert bad_row.known is False
    # And the reason names the file that is actually there. Saying "it has no
    # model_index.json" about a directory that visibly contains one sends the
    # reader looking for a file they are staring at.
    assert "model_index.json" in bad_row.reason
    assert "not a JSON object" in bad_row.reason


def test_json_that_is_not_an_object_reads_as_unreadable(tmp_path):
    """Every caller of `_read_json` in that module wants a mapping and cannot
    use anything else, so the guarantee lives in the function rather than at
    four call sites — three of which would eventually be missed."""
    from modelmri import imaging

    for text in ("[1, 2, 3]", '"a string"', "42", "null", "true"):
        p = tmp_path / "x.json"
        p.write_text(text, encoding="utf-8")
        assert imaging._read_json(p) is None, f"{text!r} did not read as unreadable"

    p = tmp_path / "x.json"
    p.write_text('{"a": 1}', encoding="utf-8")
    assert imaging._read_json(p) == {"a": 1}


def test_a_cache_nobody_could_read_is_unknown_not_absent(monkeypatch):
    """The walk failing is a fact about the DISK. "You do not have this model"
    is a claim about the reader's machine, and only one of them was true.

    `_cached_ids` used to swallow the exception and return empty sets, so
    `/api/image/size` published `cached: false` — telling somebody to spend a
    download of a model sitting on their own disk.
    """
    from modelmri import image_catalog, imaging

    def boom(*a, **kw):
        raise OSError("the cache is gone")

    monkeypatch.setattr(imaging, "scan_cache", boom)
    with_weights, configs_only, unsizeable, _capped, readable = (
        image_catalog._cached_ids()
    )

    assert readable is False
    assert not with_weights and not configs_only and not unsizeable

    # And the sentence says so, rather than leaving it to a log line the
    # reader never sees.
    means = image_catalog._size_means("acme/thing", 350_000_000, None, None, readable)
    assert "unknown rather than no" in means
    assert "could not be read" in means


def test_an_entry_nobody_could_measure_is_not_an_interrupted_download(monkeypatch):
    """`except Exception: weighs = 0` filed an unsizeable entry under
    `configs_only`, so `/api/image/size` stated as fact that the repo "has a
    cache entry on this machine but NO WEIGHTS in it — an interrupted download
    rather than a model that is ready. Finishing it costs 0.35 GB, not
    nothing."

    Reproduced with a `PermissionError`. It sends the reader to re-download a
    model that may be sitting there complete — and `local()`, in this same
    module, answers the identical failure correctly with `complete: None`, so
    the two routes contradicted each other in one panel about one repo.
    """
    from modelmri import image_catalog, image_runtime, imaging

    row = imaging.ImageModel(
        path="acme/unmeasurable",
        directory="/nowhere/acme--unmeasurable",
        family=imaging.UNET_DIFFUSION,
        architecture="StableDiffusionPipeline",
    )
    monkeypatch.setattr(imaging, "scan_cache", lambda *a, **k: [row])

    def denied(*a, **kw):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(image_runtime, "_weights_bytes", denied)

    with_weights, configs_only, unsizeable, _capped, readable = (
        image_catalog._cached_ids()
    )

    assert readable is True
    assert "acme/unmeasurable" in unsizeable
    assert "acme/unmeasurable" not in configs_only, (
        "unmeasurable is not the same as holding no weights"
    )
    assert "acme/unmeasurable" not in with_weights


def test_the_sentence_for_an_unmeasurable_entry_claims_nothing_about_weights():
    from modelmri import image_catalog

    said = image_catalog._size_means(
        "acme/unmeasurable", 350_000_000, None, None, True, True
    )

    assert "could not be measured" in said
    assert "unknown rather than" in said
    assert "NO WEIGHTS" not in said
    assert "interrupted download" not in said


def test_a_size_under_a_gigabyte_is_not_rendered_as_zero():
    """`f"{weighs / 1e9:,.2f} GB"` rendered any real measurement under 5 MB as
    "0.00 GB", including inside "Finishing it costs 0.00 GB, not nothing." —
    which collides with this UI's own token for "could not measure". Three
    sibling modules carry a "fmt.bytes_si, not /1e9" comment; this one still
    divided."""
    from modelmri import image_catalog

    said = image_catalog._size_means("acme/small", 4_000_000, False, True, True)

    assert "0.00 GB" not in said
    assert "4.0 MB" in said or "4 MB" in said, said
