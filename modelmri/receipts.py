"""What produced a number, travelling with the number.

Every panel in this tool already prints its setup in prose somewhere near the
result — the model, the dtype, the device, which baseline, which corpus. That
prose is for a person reading the screen at the time. It does not survive being
exported, forwarded, pasted into an issue, or read six weeks later, and it
cannot be checked by anything.

A receipt is the same facts in a shape a machine can read, stamped onto the
individual number rather than onto the page. It is what makes `modelmri verify`
possible at all: you cannot re-run a measurement whose setup you have to infer.

WHAT THIS DELIBERATELY DOES NOT DO

It does not guess. Three of these fields — the revision, the tokenizer
fingerprint, the prompt — can genuinely fail to resolve on a given machine, and
each of them answers `None` with a sentence saying why rather than a plausible
default. A receipt that quietly reports the wrong revision is worse than one
that reports no revision, because the first is trusted and the second is
questioned. This is the same rule the rest of the package keeps: unknown is a
value, and it never collapses into zero, "", or the nearest thing lying around.

It carries no paths and no usernames. A receipt is the part of a finding most
likely to be forwarded to a stranger, and `tests/test_no_machine_leaks.py`
covers it for exactly that reason. The revision is a content hash; the
tokenizer is a content hash; the prompt is a content hash. Where something is
on this disk is never part of the answer.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import paths
from .errors import BadRequest

log = logging.getLogger("modelmri")

# Hashes are truncated for legibility -- these are provenance labels a person
# compares by eye in a terminal, not signatures defending against a forger.
# 16 hex characters is 64 bits: for the number of distinct tokenizers or
# prompts that could ever appear in one comparison, a collision is not a real
# risk, and a 64-character field would push everything else off the line.
DIGEST_CHARS = 16

# A receipt travels inside a `.mri` that a stranger may have written, so the
# request block is bounded on the way in. The numbers are generous for any
# real call and small enough that a hostile file cannot make the viewer
# allocate: `ablate_heads` passes two keys, `rank_features` four.
MAX_REQUEST_KEYS = 32
MAX_REQUEST_TEXT = 512


def digest(text: str) -> str:
    """A short, stable content hash. Same text, same answer, any machine."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:DIGEST_CHARS]


def public_name(hf_id: str | None) -> str | None:
    """The model's name with no filesystem path in it.

    `ModelRuntime.hf_id` is a Hub id for a Hub model and an ABSOLUTE PATH for
    one loaded from a local folder -- `discover.py` sets `id=str(d)`. A receipt
    is the part of a finding most likely to be forwarded to a stranger, so
    copying that field verbatim would publish `C:\\Users\\<their real name>\\...`
    to whoever received the file.

    `export_session` already learned this the hard way for the `.mri`'s own
    `model_id` field, and the comment there records that the leak-test never
    caught it because no test had loaded a folder model. Doing the same
    reduction here rather than only at export means it holds for every route
    that returns a receipt, not just the one that writes a file.
    """
    if not hf_id:
        return None
    try:
        if Path(hf_id).exists():
            return Path(hf_id).name
    except OSError:
        # A malformed path is not worth failing over, but it must not travel
        # either -- reduce it to its last component and move on.
        return Path(hf_id).name or None
    return hf_id


# ------------------------------------------------------------------ revision


def revision_of(hf_id: str | None) -> tuple[str | None, str]:
    """The commit a cached model resolves to, and how that was established.

    Returns `(sha, how)`. `sha` is None when it cannot be established, and
    `how` then says why in a sentence rather than leaving the caller to guess.

    THE CACHE IS ASKED, NOT THE NETWORK. Resolving a revision by calling the
    Hub would make an offline machine either hang or lie, and this package
    works air-gapped everywhere else.

    `refs/main` is consulted BEFORE the snapshot listing, and the difference
    matters. `huggingface_hub` writes the resolved commit of each branch into
    `refs/<branch>`, and a plain `from_pretrained("org/name")` resolves `main`.
    Picking the newest directory under `snapshots/` instead — the obvious
    implementation, and the one `vla._snapshot` uses for a different job where
    it is fine — is a guess the moment two revisions are cached, and a wrong
    revision on a receipt is the failure this whole module exists to prevent.
    """
    # A repo id is `owner/name` OR a bare canonical name. The first version of
    # this required the slash and so reported "not a Hub repository" for
    # `gpt2` -- which is a Hub repo, is cached here as `models--gpt2`, and is
    # the model this package's own docstrings use for every worked example. A
    # local directory or an Ollama tag is excluded by the characters it
    # contains, not by the absence of an owner.
    #
    # `\` and `:` are what actually separate the three cases: a Windows path
    # has a backslash, an Ollama tag has a colon (`llama3.2:1b`), and a repo
    # id may contain neither. A leading `.` or `~` is a relative or home path.
    bad = not hf_id or any(c in hf_id for c in "\\:") or hf_id[:1] in ".~"
    if bad or hf_id.count("/") > 1 or Path(hf_id).is_dir():
        # A local directory, an Ollama tag, or a GGUF path. None of those have
        # a Hub commit, and inventing one for them would be a fabrication.
        return None, (
            "this model did not come from a Hub repository, so it has no commit"
        )

    try:
        base = paths.hf_hub_cache() / ("models--" + hf_id.replace("/", "--"))
        ref = base / "refs" / "main"
        if ref.is_file():
            sha = ref.read_text(encoding="utf-8").strip()
            if sha:
                return sha, "the commit `refs/main` resolves to in the local cache"

        snaps = (
            sorted(p.name for p in (base / "snapshots").iterdir() if p.is_dir())
            if (base / "snapshots").is_dir()
            else []
        )
    except OSError as err:
        # An evicted network drive or a permission error. The reason is
        # reported rather than swallowed into a bare None, because "the cache
        # could not be read" and "this model has no commit" are different
        # facts and a reader acts differently on each.
        return (
            None,
            f"the local model cache could not be read ({err.strerror or type(err).__name__})",
        )

    if len(snaps) == 1:
        # Unambiguous even without a ref file: there is only one thing it
        # could have loaded.
        return snaps[0], "the only revision of this model in the local cache"
    if len(snaps) > 1:
        return None, (
            f"{len(snaps)} revisions of this model are cached and no `refs/main` "
            "says which one was loaded, so naming one would be a guess"
        )
    return None, "this model is not in the local cache, so its commit is unknown"


# ----------------------------------------------------------------- tokenizer


def tokenizer_fingerprint(tokenizer) -> tuple[str | None, str]:
    """A content hash of the tokenizer, and what was hashed to get it.

    Which material was used is reported alongside, because the sources are not
    equivalent and a reader comparing two receipts needs to know they were
    fingerprinted the same way. A fast tokenizer's serialised form covers the
    vocabulary AND the normaliser, pre-tokeniser and merges; the vocabulary
    alone does not. Two tokenizers with identical vocabularies and different
    normalisers produce different token ids, so a fingerprint that could not
    see the normaliser must not be compared against one that could.
    """
    if tokenizer is None:
        return None, "no tokenizer was loaded"

    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        try:
            # The complete definition: vocab, merges, normaliser, pre-tokeniser,
            # post-processor. Hashed and dropped immediately -- it is several
            # megabytes for a large vocabulary and there is no reason to hold it.
            return digest(backend.to_str()), "the full fast-tokenizer definition"
        except Exception as err:
            # A backend that will not serialise. Fall through to the vocabulary
            # rather than failing the measurement over its receipt -- and the
            # fallback SAYS it is one, in `tokenizer_note`, so a reader is not
            # left comparing a vocabulary-only hash against a full one without
            # knowing. Logged at debug: it is expected on slow tokenizers and
            # a warning would cry wolf on every gpt2-era model.
            log.debug("tokenizer would not serialise: %s", err)

    try:
        vocab = tokenizer.get_vocab()
    except Exception as err:
        return (
            None,
            f"this tokenizer would not report its vocabulary ({type(err).__name__})",
        )
    if not isinstance(vocab, dict):
        return None, "this tokenizer reported a vocabulary that is not a mapping"

    # Streamed into the hash in sorted order rather than built into one big
    # string. A 256k-entry vocabulary is a ~6 MB string if materialised, and
    # this runs on machines chosen for having 8 GB of VRAM and not much else.
    # Sorting makes it independent of dict order, which is insertion-ordered
    # and therefore not guaranteed equal across two loads of the same files.
    h = hashlib.sha256()
    for token in sorted(vocab):
        h.update(repr(token).encode("utf-8"))
        h.update(b"\x00")
        h.update(str(vocab[token]).encode("utf-8"))
        h.update(b"\x01")
    return h.hexdigest()[:DIGEST_CHARS], (
        "the vocabulary only — this tokenizer has no serialisable fast backend, "
        "so the normaliser and pre-tokeniser are NOT covered"
    )


# ------------------------------------------------------------------- receipt


@dataclass(frozen=True)
class Receipt:
    """The setup that produced one number.

    Frozen because a receipt describes something that already happened. A
    mutable receipt is one that can be edited to describe a run that never
    occurred, and the entire value of the object is that it cannot.
    """

    # What was asked for.
    op: str
    request: dict = field(default_factory=dict)

    # What answered.
    tool_version: str = ""
    model: str | None = None
    revision: str | None = None
    revision_note: str = ""
    dtype: str | None = None
    device: str | None = None
    attn_implementation: str | None = None

    # What the answer depended on.
    # None is "this measurement was not seeded", which is a real state and NOT
    # the same as seed 0 -- a deterministic greedy pass has no seed, and
    # writing 0 there would claim a draw that never happened.
    seed: int | None = None
    tokenizer_sha256: str | None = None
    tokenizer_note: str = ""
    prompt_sha256: str | None = None
    n_prompt_tokens: int | None = None

    measured_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        """One line for a terminal. Absent facts are named, not omitted."""
        rev = f"@{self.revision[:8]}" if self.revision else "@unknown-revision"
        bits = [f"{self.model or 'no model'}{rev}"]
        if self.dtype:
            bits.append(self.dtype)
        if self.device:
            bits.append(self.device)
        if self.attn_implementation:
            bits.append(f"attn={self.attn_implementation}")
        if self.seed is not None:
            bits.append(f"seed={self.seed}")
        return f"{self.op}: " + " · ".join(bits)


# An absolute path, by shape rather than by asking the filesystem: a drive
# letter, a UNC share, or a leading slash. Deliberately the same shape
# `tests/test_no_machine_leaks.py` scans responses for, so the writer and the
# test agree by construction instead of by two people remembering the same rule.
_ABSOLUTE = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\|/)")

# An absolute path appearing ANYWHERE in a string, for the repr arm below.
#
# The lookbehind keeps a URL intact. `https://example.com/Qwen/x` has four
# slashes and every one must be rejected: the first follows `:`, the second
# follows `/`, and the rest follow word characters. Excluding only `:` reduced
# that URL to `https:/x` — measured, not guessed. A drive letter is matched
# explicitly because `C:\Users` has no leading separator to anchor on.
_EMBEDDED = re.compile(r"(?<![A-Za-z0-9:~/\\])(?:[A-Za-z]:[\\/]|\\\\|/)[^\s'\"<>|,;]*")


def _no_path(text: str) -> str:
    """A request value with any absolute path in it reduced to a bare name.

    Matched by SHAPE, not by `Path.exists()`. Existence is the right test for
    `hf_id`, where `org/model` must survive intact and only a real directory
    may be reduced -- but a request block holds arbitrary arguments, several of
    which (`sae_repo`, a corpus override, a custom model) can be absolute paths
    that do not happen to exist on the machine reading the file. A shape test
    catches those too, and touches no disk.

    Caught by the leak test rather than by review: `rank_features` puts
    `sae_repo` in its receipt, an SAE can be loaded from a local directory, and
    that path travelled inside every `.mri` exported after one.
    """
    if _ABSOLUTE.match(text):
        # rsplit on both separators: a Windows path read on Linux, or the
        # reverse, still reduces. PurePath would pick one convention and miss
        # the other.
        return re.split(r"[\\/]", text.rstrip("\\/"))[-1] or text

    # EMBEDDED paths, not only ones at position 0. The `else` arm of `_request`
    # reduces `repr(value)`, and `repr(Path("/home/me/corpus.jsonl"))` is
    # `PosixPath('/home/me/corpus.jsonl')` — which does not START with a
    # separator, so the anchored match above let the whole path through.
    #
    # `test_an_object_argument_does_not_smuggle_a_path_through_its_repr` is
    # named for exactly this and passed anyway on Windows, where
    # `str(tmp_path)` uses backslashes while the repr renders forward slashes,
    # so the substring it looked for was never there. It failed the moment CI
    # ran it on macOS. The leak was real on every POSIX machine.
    return _EMBEDDED.sub(
        lambda m: re.split(r"[\\/]", m.group(0).rstrip("\\/"))[-1] or m.group(0),
        text,
    )


def _request(raw: dict | None) -> dict:
    """The call's own arguments, reduced to things that survive JSON."""
    out: dict = {}
    for key, value in (raw or {}).items():
        if not isinstance(key, str):
            continue
        if isinstance(value, bool) or value is None or isinstance(value, (int, float)):
            out[key] = value
        elif isinstance(value, str):
            out[key] = _no_path(value)[:MAX_REQUEST_TEXT]
        elif isinstance(value, (list, tuple)):
            # One level. A request argument is a scalar or a short list of
            # them; anything deeper is a data structure that belongs in the
            # measurement, not in the label describing it.
            out[key] = [
                _no_path(v)[:MAX_REQUEST_TEXT] if isinstance(v, str) else v
                for v in value
                if isinstance(v, (str, int, float, bool)) or v is None
            ]
        else:
            # repr() of an arbitrary object routinely embeds a path -- a Path,
            # a file handle, a module all do -- so this arm is reduced too.
            out[key] = _no_path(repr(value))[:MAX_REQUEST_TEXT]
    return out


def stamp(
    runtime,
    op: str,
    *,
    request: dict | None = None,
    prompt: str | None = None,
    seed: int | None = None,
    now: datetime | None = None,
) -> Receipt:
    """Read the live runtime and record what it currently is.

    Every field is read at the moment the measurement ran. Nothing here is
    passed in by a caller who might be describing an intention rather than
    what actually happened -- the dtype comes off the loaded parameters, the
    attention implementation off the model's own config. A caller that lied
    about its dtype would produce a receipt that disagreed with the run, which
    is the one thing a receipt must never do.

    `runtime` is duck-typed rather than annotated as ModelRuntime: importing
    it here would close a cycle (runtime imports this module to stamp its own
    results), and the only thing needed is a handful of attributes.
    """
    from . import __version__

    model = getattr(runtime, "model", None)
    tokenizer = getattr(runtime, "tokenizer", None)
    hf_id = getattr(runtime, "hf_id", None)

    dtype = None
    attn = None
    if model is not None:
        try:
            dtype = str(next(model.parameters()).dtype).removeprefix("torch.")
        except (StopIteration, AttributeError):
            # A model with no parameters is not a thing transformers produces,
            # but a receipt is not worth raising over.
            dtype = None
        config = getattr(model, "config", None)
        # `_attn_implementation` is private and transformers has moved it
        # before. getattr with a default keeps a receipt from taking down a
        # measurement when it moves again; the field simply reads as unknown.
        attn = getattr(config, "_attn_implementation", None) if config else None
        if attn is not None:
            attn = str(attn)

    revision, revision_note = revision_of(hf_id)
    tok_sha, tok_note = tokenizer_fingerprint(tokenizer)

    # The prompt the caller names, or the one the runtime last ran. Falling
    # back matters: most of these measurements take no prompt argument because
    # they operate on the last generation, and a receipt that left the prompt
    # blank for those would be blank on the majority of numbers.
    text = prompt if prompt is not None else getattr(runtime, "last_prompt", None)
    n_tokens = None
    ids = getattr(runtime, "last_ids", None)
    if prompt is None and ids is not None:
        try:
            n_tokens = int(ids.shape[-1])
        except (AttributeError, IndexError, TypeError):
            n_tokens = None

    return Receipt(
        op=op,
        request=_request(request),
        tool_version=__version__,
        model=public_name(hf_id),
        revision=revision,
        revision_note=revision_note,
        dtype=dtype,
        device=getattr(runtime, "device", None),
        attn_implementation=attn,
        seed=seed,
        tokenizer_sha256=tok_sha,
        tokenizer_note=tok_note,
        prompt_sha256=digest(text) if isinstance(text, str) else None,
        n_prompt_tokens=n_tokens,
        measured_at=(now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
    )


# ------------------------------------------------------- reading a stranger's


def parse(raw) -> list[dict]:
    """Validate the receipts section of an untrusted `.mri`.

    Same standard as every other section: absent is fine, malformed is
    refused. A damaged file shown as an intact one minus its provenance is
    precisely the outcome receipts exist to prevent -- a reader who sees no
    receipts concludes the numbers came without any, not that the section
    failed to load.
    """
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise BadRequest("this session's receipts section is not a list")

    out: list[dict] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise BadRequest(f"receipt {i} is not a set of fields")
        op = item.get("op")
        if not isinstance(op, str) or not op.strip():
            raise BadRequest(f"receipt {i} does not say which measurement it describes")

        clean: dict = {"op": op[:MAX_REQUEST_TEXT]}
        for key in (
            "tool_version",
            "revision",
            "revision_note",
            "model",
            "dtype",
            "device",
            "attn_implementation",
            "tokenizer_sha256",
            "tokenizer_note",
            "prompt_sha256",
            "measured_at",
        ):
            value = item.get(key)
            # None survives as None. It is the honest answer for a fact that
            # could not be established, and coercing it to "" here would erase
            # the distinction the writer was careful to record.
            clean[key] = value[:MAX_REQUEST_TEXT] if isinstance(value, str) else None

        for key in ("seed", "n_prompt_tokens"):
            value = item.get(key)
            # bool is an int in Python and `seed: true` must not read as 1.
            clean[key] = (
                value
                if isinstance(value, int) and not isinstance(value, bool)
                else None
            )

        request = item.get("request")
        if request is not None and not isinstance(request, dict):
            raise BadRequest(f"receipt {i} has a request block that is not fields")
        trimmed = _request(request)
        if len(trimmed) > MAX_REQUEST_KEYS:
            raise BadRequest(
                f"receipt {i} carries {len(trimmed)} request fields, above the "
                f"{MAX_REQUEST_KEYS} this reads."
            )
        clean["request"] = trimmed
        out.append(clean)
    return out


def write_jsonl(receipts: list[Receipt], path: str | Path) -> Path:
    """One receipt per line, for a sweep or a shell pipeline.

    JSONL rather than one JSON array so a run that is killed half way leaves a
    file whose complete lines are still readable, which is the whole reason
    the format exists.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as fh:
        for receipt in receipts:
            fh.write(json.dumps(receipt.to_dict(), allow_nan=False) + "\n")
    return target
