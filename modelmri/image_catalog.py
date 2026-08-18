"""Finding image models: what is downloadable, what is already here, what it costs.

The text side has had this for a long time — `hub.search` for what exists,
`discover.scan` for what is on disk, `hub.weight_bytes` for what a download
weighs, and `capacity.guard` to refuse before it starts. The image side had a
cache scan and nothing else, so the only way to open a diffusion model was to
already know its name.

This is the missing half, and it deliberately reuses the text side's machinery
rather than growing a second one: `hub._api` for the request, `hub.token` for
credentials, `hub.weight_bytes` for the size arithmetic. A second Hub client
with its own idea of what a timeout is, or its own idea of what "gated" means,
is two answers to one question.

## A pipeline tag is a TASK, not an architecture

This is the honest limit of anything read from a listing, and it is why the
rows here do not claim a family. `text-to-image` covers a UNet and a DiT, and
those two have their cross-attention in different places — so a row says what
the model DOES, names the families that tag is consistent with, and leaves the
architecture to `imaging.detect`, which reads the checkpoint's own config.

Claiming a family from a tag would put a confident wrong word in a list, and
`imaging`'s whole argument is that a panel drawn for the wrong family is a
picture of something that does not exist.

## Size is read, and 0 means unknown

`hub.weight_bytes` does arithmetic on the per-dtype parameter counts the Hub
publishes, and returns 0 for a repo that publishes nothing to go on — which is
most GGUF and pickle repos. Zero is passed through as `None` here rather than
as a number, because a picker that renders "0.0 GB" for an unknown invites
exactly the click that a size column exists to prevent.

## Cached is answered from the disk, not from the listing

Whether a repo is already here is a question about this machine, so it is
answered by looking — `imaging.scan_cache` already walks the local cache and
names what it finds. A listing that guessed would be wrong for the one user
who moved their cache.
"""

from __future__ import annotations

import http.client
import logging
import urllib.error
import urllib.parse

from . import imaging
from .errors import BadRequest, Refusal

log = logging.getLogger(__name__)

# The Hub pipeline tags this tool can actually open, each with the families
# `imaging.detect` might name once the config is read.
#
# A translation table like `imaging._BY_MODEL_TYPE`, and it earns its place the
# same way: these are the Hub's own vocabulary, not model names. A tag missing
# from here is simply not offered, which is better than offering a task whose
# checkpoints this cannot load.
TASKS: dict[str, dict] = {
    "text-to-image": {
        "label": "Text to image",
        "families": (imaging.UNET_DIFFUSION, imaging.DIT_DIFFUSION),
        "means": (
            "Generates a picture from a prompt. Cross-attention maps and word "
            "knockout apply once the checkpoint says which denoiser it uses."
        ),
    },
    "image-to-image": {
        "label": "Image to image",
        "families": (imaging.UNET_DIFFUSION, imaging.DIT_DIFFUSION),
        "means": "Transforms a picture under a prompt — upscalers, edits, depth.",
    },
    "unconditional-image-generation": {
        "label": "Unconditional generation",
        "families": (imaging.UNET_DIFFUSION, imaging.DIT_DIFFUSION),
        "means": (
            "Generates without a prompt. There is no cross-attention to a "
            "prompt, so there are no word-to-pixel maps to draw."
        ),
    },
    "image-classification": {
        "label": "Image classification",
        "families": (imaging.VIT,),
        "means": (
            "Names what is in a picture. Occlusion attribution applies — cover "
            "a region, re-run, and measure what the answer did."
        ),
    },
    "object-detection": {
        "label": "Object detection",
        "families": (imaging.DETECTION,),
        "means": "Finds and boxes things in a picture.",
    },
    "image-segmentation": {
        "label": "Segmentation",
        "families": (imaging.SEGMENTATION,),
        "means": "Labels a picture pixel by pixel, or cuts objects out of it.",
    },
    "mask-generation": {
        "label": "Mask generation",
        "families": (imaging.SEGMENTATION,),
        "means": "Segments anything it is pointed at, without a fixed label set.",
    },
    "zero-shot-image-classification": {
        "label": "Image-text embedding",
        "families": (imaging.CLIP,),
        "means": (
            "Scores a picture against arbitrary text, so its label set is "
            "whatever you type rather than what it was trained on."
        ),
    },
    "image-feature-extraction": {
        "label": "Image embedding",
        "families": (imaging.VIT, imaging.CLIP),
        "means": "Turns a picture into a vector, with no classifier on top.",
    },
    "image-text-to-text": {
        "label": "Vision-language",
        "families": (imaging.VLM,),
        "means": "Reads a picture and answers about it in words.",
    },
}

# What `search` asks for when no task is named. Every tag at once is not a
# valid Hub filter — the API ANDs repeated `filter` values — so the default is
# the one task most people mean by "image model", and the caller picks
# otherwise. Stated rather than silently one of ten.
DEFAULT_TASK = "text-to-image"

MAX_RESULTS = 50


def tasks() -> list[dict]:
    """Every task that can be searched, for a picker to render.

    Ordered as written rather than alphabetically: generation first because it
    is what most people arrive wanting, then the understanding tasks. A tag
    this tool cannot open is not in the table at all.
    """
    return [
        {
            "task": tag,
            "label": spec["label"],
            "families": list(spec["families"]),
            "means": spec["means"],
        }
        for tag, spec in TASKS.items()
    ]


def search(query: str = "", task: str = "", limit: int = 24) -> list[dict]:
    """Image models on the Hub, annotated with size and whether they are here.

    Nothing is downloaded. This reads a listing, and the one thing it touches
    on this machine is the cache index, to say which rows are already local.
    """
    from . import hub

    tag = (task or DEFAULT_TASK).strip()
    if tag not in TASKS:
        raise BadRequest(
            f"`{tag}` is not a task this reads. The ones it can open are: "
            f"{', '.join(TASKS)}. A tag outside that list would return "
            f"checkpoints nothing here can load."
        )

    asked = int(limit or 24)
    used = max(1, min(asked, MAX_RESULTS))

    params: list[tuple[str, str]] = [
        ("limit", str(used)),
        ("sort", "downloads"),
        ("direction", "-1"),
        ("filter", tag),
        # `expand[]` rather than `full=true` — the two are mutually exclusive
        # and `full` does NOT include `safetensors`, which is where the size
        # comes from. `hub.search` carries the same comment and the same scar:
        # a picker that cannot say how big a model is invites a click that
        # starts a 1.5 TB download on an 8 GB laptop.
        *[
            ("expand[]", k)
            for k in ("safetensors", "downloads", "gated", "lastModified", "likes")
        ],
    ]
    if query.strip():
        params.append(("search", query.strip()))

    tok = hub.token()
    try:
        raw = hub._api("/models?" + urllib.parse.urlencode(params), tok)
    except (urllib.error.URLError, OSError, http.client.HTTPException) as err:
        # The same four-way failure `hub.search` documents at length: URLError
        # for a failure to connect, OSError for a stall or a mid-read close,
        # HTTPException for a malformed status line or a truncated body. All
        # four mean "the Hub did not answer" to a reader.
        #
        # Deliberately does not interpolate `err`: this string is published to
        # the browser and `str(URLError)` is machinery talking to itself.
        log.warning("image hub search failed", exc_info=err)
        raise Refusal(
            "Could not reach the HuggingFace Hub. Check your connection — the "
            "full error is in the terminal running `modelmri serve`. Image "
            "models already downloaded still load: they are listed under what "
            "is on this machine."
        ) from err

    here, partial, cache_capped = _cached_ids()
    spec = TASKS[tag]
    out: list[dict] = []
    for m in raw if isinstance(raw, list) else []:
        repo = m.get("id")
        if not repo:
            continue
        gated = bool(m.get("gated", False))
        weighs = hub.weight_bytes(m)
        out.append(
            {
                "id": repo,
                "task": tag,
                "task_label": spec["label"],
                # What the tag is CONSISTENT with. Not a claim about this
                # checkpoint — only its own config can settle that, and
                # `imaging.detect` reads it at load time.
                "families_possible": list(spec["families"]),
                # `None`, not 0, when the Hub published no count. Sorting by
                # downloads is the default, so a repo whose count is simply
                # absent would sort as the least popular thing on the page —
                # a claim nobody made, rendered as a fact.
                "downloads": _count(m.get("downloads")),
                "likes": _count(m.get("likes")),
                "gated": gated,
                "updated": (m.get("lastModified") or "")[:10],
                # `None`, never 0. The Hub publishes nothing to go on for GGUF
                # and pickle repos, and a picker rendering "0.0 GB" for an
                # unknown invites the exact click a size column prevents.
                "size_bytes": weighs or None,
                # Answered by looking at the disk rather than guessed, and
                # "here" means the WEIGHTS are here. A repo whose cache entry
                # holds a config and nothing else still has its whole download
                # ahead of it.
                "cached": repo in here,
                # The third state, reported rather than folded into either:
                # an interrupted download looks cached to a directory listing
                # and costs the full size to finish.
                "partial": repo in partial,
            }
        )
    # Both caps ride on the rows, because a caller that cannot see them
    # reports a truncated list as a complete one. `search` is the function
    # that APPLIES them, so it is the function that has to say so.
    return _Rows(out, limit_asked=asked, limit_used=used, cache_capped=cache_capped)


class _Rows(list):
    """The rows, plus what was cut to produce them.

    A `list` subclass rather than a new return shape: every caller and test
    indexes and iterates this exactly as before, and the two caps are readable
    by the route that renders the sentence. A cap nobody can see is reported
    as a complete list.
    """

    def __init__(self, rows, *, limit_asked, limit_used, cache_capped):
        super().__init__(rows)
        self.limit_asked = limit_asked
        self.limit_used = limit_used
        self.cache_capped = cache_capped

    def __eq__(self, other) -> bool:
        """Equal only when the CAPS match too.

        Inheriting `list.__eq__` would have compared the rows alone, so a
        complete list of 50 and a list of 50 truncated from 200 compared
        equal — which is precisely the silent-truncation defect this class
        exists to prevent, reappearing in the comparison operator.

        Against a plain list this still compares as a list, so a test may
        write `rows == [...]` and mean it.
        """
        if not isinstance(other, _Rows):
            return list.__eq__(self, other)
        return (
            list.__eq__(self, other)
            and self.limit_asked == other.limit_asked
            and self.limit_used == other.limit_used
            and self.cache_capped == other.cache_capped
        )

    def __ne__(self, other) -> bool:
        # Python derives `!=` from `__eq__` only when `__ne__` is absent; `list`
        # defines one, so it would have been inherited and disagreed with the
        # `__eq__` above.
        result = self.__eq__(other)
        return result if result is NotImplemented else not result

    # Lists are unhashable and this stays unhashable: it is mutable, and a
    # hash over mutable state is the other half of the same trap.
    __hash__ = None


def _cached_ids() -> tuple:
    """`(with_weights, configs_only, capped)` — what is here, and whether the
    walk hit its own limit.

    Returns two sets, and the split is the whole point. `imaging.scan_cache`
    admits an entry the moment a snapshot directory identifies, which a lone
    4 KB `config.json` satisfies. Treating that as "already on this machine"
    is how `/api/image/size` came to answer:

        Qwen/Qwen3.6-27B  ->  "already on this machine, so nothing would be
                               downloaded"   (55.6 GB still to transfer)

    measured live, alongside `/api/image/local` reporting the same repo as
    `complete: false` in the same panel. Two routes answering one question
    differently, and the one whose entire job is pricing a download before it
    is spent was the one that was wrong.

    Never raises. A cache that cannot be read means "not cached" for every
    row — the listing is still useful and the load will answer for itself.
    """
    from . import image_runtime

    with_weights: set = set()
    configs_only: set = set()
    try:
        found = imaging.scan_cache()
    except Exception:
        log.warning("could not read the image cache", exc_info=True)
        return with_weights, configs_only, False

    # `scan_cache` stops at its own limit, and a repo past that point is
    # reported as NOT cached — which reads as "you do not have this" for a
    # model sitting on the disk. The cap travels so a caller can say the list
    # is partial rather than complete.
    capped = len(found) >= imaging.SCAN_CACHE_LIMIT

    for model in found:
        try:
            weighs = image_runtime._weights_bytes(_path_of(model))
        except Exception:
            # Unsizeable is not the same as absent, and it is not "here"
            # either — an entry nothing could measure must not be reported as
            # a download that costs nothing.
            log.warning("could not size %s", model.path, exc_info=True)
            weighs = 0
        (with_weights if weighs > 0 else configs_only).add(model.path)
    return with_weights, configs_only, capped


def local() -> list[dict]:
    """Every image model on this disk, with what it actually weighs.

    `imaging.scan_cache` says what each one IS. This adds what it costs, read
    off the files rather than from the Hub, because the question "will this
    fit" is about the copy on this machine.
    """
    from . import image_runtime

    out: list[dict] = []
    for found in imaging.scan_cache():
        bytes_on_disk = 0
        sized = True
        try:
            bytes_on_disk = image_runtime._weights_bytes(_path_of(found))
        except Exception:
            # A cache entry that cannot be sized is still worth listing, but it
            # must not be listed as an interrupted download. Those are two
            # different states — "the weights are not there" and "nobody could
            # tell" — and an earlier version collapsed both into
            # `complete: false`, which made the route's own summary sentence
            # say a permission error was a half-finished download.
            log.warning("could not size %s", found.path, exc_info=True)
            sized = False
        out.append(
            {
                "path": found.path,
                "family": found.family,
                "label": imaging.label(found.family),
                "known": found.known,
                "architecture": found.architecture,
                "capabilities": list(found.capabilities),
                "reason": found.reason,
                # Unknown stays unknown. A cache entry holding only configs is
                # a real state — an interrupted download — and reporting it as
                # 0 GB would say it is ready to load.
                "size_bytes": bytes_on_disk or None,
                # Three states, not two. True: weights are here. False: they
                # are not, which is an interrupted download. None: this entry
                # could not be sized at all, so neither claim can be made.
                "complete": (bytes_on_disk > 0) if sized else None,
            }
        )
    out.sort(key=lambda r: (not r["known"], -(r["size_bytes"] or 0), r["path"]))
    return out


def discovered(roots=None) -> dict:
    """Image models in ordinary folders — the running directory and elsewhere.

    `local()` answers "what has the Hub cache got". This answers "what is on
    this machine at all", which is the question somebody who cloned a
    checkpoint into their project directory is actually asking. The two are
    reported separately rather than merged, because where a model came from
    decides what you can do with it: a cache entry has a repo id to re-fetch,
    a loose folder has only a path.

    The roots looked in are RETURNED. A picker that says "nothing found"
    without saying where it looked is telling somebody their model is missing
    when the truth may be that the directory holding it was never searched.
    """
    from pathlib import Path

    from . import discover, image_runtime, imaging

    looked = [str(r) for r in (roots if roots is not None else discover.roots())]
    models, truncated = imaging.scan_dirs(roots)

    # A folder that is ALSO in the Hub cache is one model, not two. Compared
    # by resolved path rather than by name: two checkpoints can share a
    # directory name and be different weights.
    cached: set[str] = set()
    for row in imaging.scan_cache():
        try:
            cached.add(str(Path(_path_of(row)).resolve()))
        except OSError:
            continue

    out: list[dict] = []
    for found in models:
        try:
            here = Path(found.path).resolve()
        except OSError:
            here = Path(found.path)
        if str(here) in cached:
            continue
        bytes_on_disk = 0
        sized = True
        try:
            bytes_on_disk = image_runtime._weights_bytes(here)
        except Exception:
            log.warning("could not size %s", found.path, exc_info=True)
            sized = False
        out.append(
            {
                "path": found.path,
                "family": found.family,
                "label": imaging.label(found.family),
                "known": found.known,
                "architecture": found.architecture,
                "capabilities": list(found.capabilities),
                "reason": found.reason,
                "size_bytes": bytes_on_disk or None,
                "complete": (bytes_on_disk > 0) if sized else None,
            }
        )
    out.sort(key=lambda r: (not r["known"], -(r["size_bytes"] or 0), r["path"]))
    return {"models": out, "roots": looked, "truncated": truncated}


def _count(value):
    """A published count, or `None` when the Hub published none.

    Not `0`. Sorting by downloads is the default, so a repo whose count is
    simply absent would sort as the least popular thing on the page — a claim
    nobody made, rendered as a fact. `isinstance(True, int)` is True, so a
    bool is rejected rather than counted as 1.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _path_of(found):
    """The directory an `ImageModel` was read from.

    `scan_cache` renames `path` to the repo id — the name a reader recognises
    and the one `load` takes — so `directory` is what still points at the
    files. Reconstructing one from the other would mean rebuilding the cache's
    own layout in a second place.
    """
    from pathlib import Path

    return Path(found.directory or found.path)


def size_of(repo: str) -> dict:
    """What downloading `repo` would cost, before any of it moves.

    The counterpart to `capacity.guard`: that refuses against free memory, and
    this answers the question a reader asks first, which is how big the thing
    is at all.
    """
    from . import hub

    name = (repo or "").strip()
    if not name:
        raise BadRequest("no model was named, so there is nothing to price.")

    # The name goes into an API PATH, and the request carries the reader's Hub
    # token. `urllib.parse.quote(name, safe="/")` leaves `..` intact, so
    # `../whoami-v2` walks out of `/models/` to a different endpoint entirely —
    # with the token attached, from a route that is unauthenticated on a
    # server started with `--host 0.0.0.0`.
    #
    # `is_hub_id` is the same shape test `/api/image/load` uses, and it
    # rejects `..`, a leading separator and a drive letter. Checked BEFORE the
    # URL is built rather than trusting the quoting, because quoting decides
    # how characters are encoded and this is about which characters are
    # allowed at all.
    from .behavdiff import is_hub_id

    if not is_hub_id(name):
        raise BadRequest(
            f"`{name}` is not a Hub repo id, so there is nothing on the Hub to "
            f"price. An id is `name` or `owner/name` — no leading separator, "
            f"no drive letter and no `..`. A local directory has its size read "
            f"off the disk instead, and is already listed."
        )

    tok = hub.token()
    try:
        raw = hub._api(
            "/models/"
            + urllib.parse.quote(name, safe="/")
            + "?"
            + urllib.parse.urlencode(
                [("expand[]", k) for k in ("safetensors", "gated", "siblings")]
            ),
            tok,
        )
    except (urllib.error.URLError, OSError, http.client.HTTPException) as err:
        log.warning("image size lookup failed", exc_info=err)
        raise Refusal(
            f"Could not ask the Hub how big `{name}` is. The full error is in "
            f"the terminal running `modelmri serve`."
        ) from err

    if not isinstance(raw, dict):
        raise Refusal(f"The Hub did not describe `{name}` in a shape this reads.")

    weighs = hub.weight_bytes(raw)
    here, partial, _capped = _cached_ids()
    cached = name in here
    incomplete = name in partial
    return {
        "id": name,
        "size_bytes": weighs or None,
        "gated": bool(raw.get("gated", False)),
        "cached": cached,
        "partial": incomplete,
        "means": _size_means(name, weighs, cached, incomplete),
    }


def _size_means(name: str, weighs: int, cached: bool, partial: bool = False) -> str:
    if cached:
        where = f"`{name}` is already on this machine, so nothing would be downloaded."
    elif partial:
        # The state that used to be reported as "nothing would be downloaded".
        # A cache entry holding configs and no weights looks present to a
        # directory listing and has its entire transfer still ahead of it.
        size = f"{weighs / 1e9:,.2f} GB" if weighs else "an unpublished amount"
        where = (
            f"`{name}` has a cache entry on this machine but NO WEIGHTS in it — "
            f"an interrupted download rather than a model that is ready. "
            f"Finishing it costs {size}, not nothing."
        )
    elif weighs:
        where = (
            f"`{name}` publishes {weighs / 1e9:,.2f} GB of weights, which is "
            f"what a download would transfer and roughly what it would need "
            f"resident."
        )
    else:
        where = (
            f"`{name}` publishes no size metadata — usually a GGUF or pickle "
            f"repo — so how big it is is UNKNOWN rather than small. Nothing "
            f"here will guess it."
        )
    return (
        f"{where} Whether it fits is a separate question, answered against "
        f"this machine's free memory when you load it."
    )
