"""HuggingFace Hub access: sign in, search models, see what you can use.

ModelMRI never asks for your password. You paste a HuggingFace *access
token* (huggingface.co/settings/tokens) — a scoped credential you can
revoke at any time — never in the repo and never sent anywhere except
huggingface.co.

The token file is written owner-only: created at mode 0600 rather than
narrowed to it afterwards, and moved into place atomically. That mode is
enforced on POSIX. On Windows the file inherits your user profile's ACL,
because chmod there sets the read-only attribute and grants nothing.

Its location follows platform convention, so rather than trusting a path
written down in a docstring, run `modelmri where` for the resolved one.

Signing in unlocks gated models you have accepted the license for
(Gemma, Llama, ...) and your own private repos.
"""

from __future__ import annotations

import http.client
import json
import logging
import os
import stat
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from . import paths
from .errors import BadRequest, Refusal

HUB_API = "https://huggingface.co/api"

log = logging.getLogger("modelmri")


def _config_path() -> Path:
    """Where the token lives.

    Platform convention, with the pre-0.6 `~/.modelmri/hub.json` still read if
    it exists — moving the default without looking at the old place would sign
    people out on upgrade and leave a token file behind that nobody knows is
    there.
    """
    from . import paths

    return paths.token_path()


# Small, current, ungated models that actually fit an 8 GB GPU. Shown when
# the user has not searched for anything yet.
SUGGESTED = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "HuggingFaceTB/SmolLM3-3B",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "allenai/OLMo-2-0425-1B-Instruct",
    "microsoft/Phi-4-mini-instruct",
    "gpt2",
]


@dataclass
class HubAuth:
    signed_in: bool
    user: str | None = None
    source: str | None = None  # "modelmri" | "huggingface-cli" | "env"

    def to_dict(self) -> dict:
        return asdict(self)


def _cli_token_paths() -> list[Path]:
    """Where `huggingface-cli login` may have left a token.

    Built through `paths._home()` rather than `Path.home()`, because
    `Path.home()` **raises** — RuntimeError, not OSError — where there is no
    home directory to expand `~` against: a Linux container running as an
    arbitrary UID with no passwd entry, a Windows service account with no
    USERPROFILE. `paths._home()` is the package's one definition of that
    failure and documents it; this list was the last place calling
    `Path.home()` raw.

    It mattered. On such a machine `_read_stored_token` raised straight
    through `whoami`'s "Never raises" docstring — `whoami` calls it *outside*
    its own try — and `/api/hub/auth`, which has no handler, answered 500 for
    a panel whose honest answer is "signed out". Verified by clearing HOME,
    USERPROFILE, HOMEDRIVE and HOMEPATH and calling it: RuntimeError,
    "Could not determine home directory."
    """
    candidates = [paths.hf_home() / "token"]
    home = paths._home()
    if home is not None:
        candidates.append(home / ".huggingface" / "token")
    return candidates


def _read_stored_token() -> tuple[str | None, str | None]:
    """(token, source) — ours first, then the HF CLI's, then the env.

    Never raises. Every caller reads an unreadable token as "not signed in",
    and one of them (`whoami`) is on a route with no error handler at all.
    """
    try:
        if _config_path().is_file():
            token = json.loads(_config_path().read_text(encoding="utf-8")).get("token")
            if token:
                return token, "modelmri"
    except (OSError, ValueError, AttributeError, RecursionError) as err:
        # The complete set for reading our own token file, and it can be
        # complete because `_config_path()` cannot fail: `paths` swallows its
        # own OSError and RuntimeError and returns a path either way. So what
        # is left is PermissionError on the read (OSError); a file that is not
        # UTF-8 (UnicodeDecodeError, a ValueError); truncated JSON
        # (JSONDecodeError, also a ValueError); or JSON that parses to a list
        # or a number, where `.get` is not a method (AttributeError); or a
        # document nested thousands deep, where the recursive-descent decoder
        # runs out of stack (RecursionError, which is a RuntimeError and was in
        # none of the other three). That last one only needs a hand-edited or
        # corrupted file to reach, but the docstring above says "Never raises"
        # and `whoami` calls this OUTSIDE its own try, on the one route
        # (/api/hub/auth) that has no handler — so "never" has to be true.
        # Measured: `[` x 200000 in hub.json raised straight through before.
        #
        # Continuing is right — an unreadable token is not a token, and the
        # environment and the CLI's own file below may still have one. But
        # continuing *silently* is not: `_write_private`'s docstring records
        # what this shrug cost, "the user was signed out with no message and
        # no way to tell why". The message now exists, in the terminal, with
        # the path to look at.
        log.warning(
            "could not read the stored HuggingFace token at %s (%s: %s); "
            "treating this session as signed out",
            _config_path(),
            type(err).__name__,
            err,
        )

    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(var):
            return os.environ[var], "env"

    # whatever `huggingface-cli login` wrote
    for path in _cli_token_paths():
        try:
            if path.is_file():
                token = path.read_text(encoding="utf-8").strip()
                if token:
                    return token, "huggingface-cli"
        except (OSError, ValueError):
            # PermissionError on someone else's file, or a token file that is
            # not UTF-8 (UnicodeDecodeError, a ValueError). Not logged, unlike
            # our own file above: this one is the CLI's to own, we are only
            # borrowing it, and the next candidate may still answer.
            continue
    return None, None


def token() -> str | None:
    return _read_stored_token()[0]


def _api(path: str, tok: str | None = None, timeout: float = 15) -> object:
    req = urllib.request.Request(f"{HUB_API}{path}")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def whoami(tok: str | None = None) -> HubAuth:
    """Who the stored (or supplied) token belongs to. Never raises."""
    candidate, source = (tok, "supplied") if tok else _read_stored_token()
    if not candidate:
        return HubAuth(signed_in=False)
    try:
        me = _api("/whoami-v2", candidate)
        return HubAuth(
            signed_in=True,
            user=me.get("name") or me.get("fullname"),
            source=source,
        )
    except Exception as err:  # noqa: BLE001 - the contract, see below
        # Deliberately broad, and it stays broad. The contract is the first
        # line of the docstring, and `/api/hub/auth` calls this with no
        # handler at all — anything that escapes is an unhandled 500 on the
        # account panel. Every failure means one thing to the caller: we could
        # not confirm whose token this is, which is what signed_in=False says.
        #
        # Debug, not warning: the commonest arrival here is an HTTP 401 for a
        # token the user just typed wrong, and `sign_in` turns that into a
        # sentence of its own. A warning per rejected token would be noise
        # about working code.
        log.debug(
            "whoami failed (%s: %s); reporting signed out", type(err).__name__, err
        )
        return HubAuth(signed_in=False)


def _write_private(path: Path, text: str) -> None:
    """Write a credential: owner-only from the instant it exists, atomically.

    Two separate problems with `write_text` + `chmod`:

    * Order. `write_text` creates the file at 0666 & ~umask — 0644 on a
      typical POSIX box — and the chmod narrows it a moment later. On a
      multi-user host, any local account can read the token in between. The
      mode has to be passed to `open`, not applied after it.
    * Atomicity. A crash or a full disk mid-write left truncated JSON, which
      the reader swallowed silently, so the user was signed out with no
      message and no way to tell why.

    Written to a temp file in the same directory and moved into place, which
    on both POSIX and Windows is atomic for a same-volume rename.

    The 0600 is real on POSIX. On Windows `chmod` only toggles the read-only
    attribute, so the file inherits the user profile's ACL instead; that is
    the honest claim and it is the one the docs make.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        fd = os.open(
            tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR
        )
        try:
            handle = os.fdopen(fd, "w", encoding="utf-8")
        except BaseException:  # fdopen did not take ownership; we still own fd
            os.close(fd)
            raise
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            # On the success path `os.replace` already moved the temp file and
            # `missing_ok` covers that, so this only fires when the file is
            # there and undeletable: on Windows a PermissionError (WinError 32)
            # while an AV scanner or the search indexer still holds it open, or
            # the read-only attribute; on POSIX EACCES on a config directory
            # that turned read-only mid-write. Measured on this platform:
            # unlinking a file another handle has open raises exactly that.
            #
            # Swallowing it is right — the write has already either succeeded
            # or raised, and failing a sign-in over a leftover temp file would
            # be the worse outcome. The leftover is not an exposure either: it
            # was created 0600 through `os.open` on POSIX and inherits the
            # profile ACL on Windows, the same protection as the real token
            # file it was about to become.
            pass
    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        # A no-op on Windows, where chmod only toggles the read-only
        # attribute; on POSIX it re-asserts a mode `os.open` already set.
        # It raises on filesystems with no permission model (an exFAT stick,
        # some SMB and NFS mounts) and with EPERM when the file belongs to
        # another uid.
        #
        # Swallowing is safe for one specific reason: this call cannot be the
        # thing that makes the file private, because the file was *created*
        # 0600 four lines up. A failure here leaves the correct mode in place.
        pass


def sign_in(tok: str) -> HubAuth:
    """Validate a token, then store it privately.

    Raises `BadRequest` when the field is empty or the Hub rejects the token:
    both are facts about the credential in the request, not decisions of ours,
    and both reach the browser as 422 with the sentence below.
    """
    tok = (tok or "").strip()
    if not tok:
        raise BadRequest("Paste a token from huggingface.co/settings/tokens")
    auth = whoami(tok)
    if not auth.signed_in:
        raise BadRequest(
            "HuggingFace rejected that token. Create a fresh one with 'read' "
            "access at huggingface.co/settings/tokens."
        )
    _write_private(_config_path(), json.dumps({"token": tok}))
    auth.source = "modelmri"
    return auth


def sign_out() -> HubAuth:
    path = _config_path()
    try:
        path.unlink(missing_ok=True)
    except OSError as err:
        # `missing_ok` already covers "there was nothing to delete", so what
        # reaches here is a file that exists and will not go: PermissionError
        # on Windows while something else holds it open or it carries the
        # read-only attribute, EACCES on a config directory that turned
        # read-only.
        #
        # Continuing is right, but the old silent `pass` made this half a lie.
        # `whoami()` below re-reads the file we just failed to delete, so the
        # answer is signed_in=True — which is *true*, and which is why we do
        # not fake a sign-out we did not perform. What was missing is any
        # account of why the button did nothing. It exists now, with the path
        # to remove by hand.
        log.warning(
            "sign-out could not delete the token file at %s (%s: %s) — you are "
            "still signed in; delete that file to sign out",
            path,
            type(err).__name__,
            err,
        )
    return whoami()  # a CLI/env token may still be active — report honestly


def search(query: str = "", limit: int = 24) -> list[dict]:
    """Text-generation models, newest-relevant first, annotated with access."""
    tok = token()
    # `expand[]`, not `full=true`. The two are mutually exclusive, and
    # `full=true` does NOT include `safetensors` — so every row came back with
    # no size at all. A picker that cannot say how big a model is invites the
    # thing that actually happened here: a click on zai-org/GLM-5.2 started a
    # 1.5 TB download on an 8 GB laptop.
    params: list[tuple[str, str]] = [
        ("limit", str(max(1, min(limit, 50)))),
        ("sort", "downloads"),
        ("direction", "-1"),
        ("filter", "text-generation"),
        *[
            ("expand[]", k)
            for k in ("safetensors", "downloads", "gated", "lastModified", "likes")
        ],
    ]
    if query.strip():
        params.append(("search", query.strip()))
    try:
        raw = _api("/models?" + urllib.parse.urlencode(params), tok)
    except (urllib.error.URLError, OSError, http.client.HTTPException) as err:
        # Three, because "the Hub did not answer" arrives as at least four
        # different exceptions and only one of them was originally caught.
        # urllib wraps a failure to *connect* in URLError, but everything from
        # `getresponse()` and the body read onwards comes out raw. Measured
        # against local sockets that misbehave in the ways a captive portal,
        # a corporate proxy or a flaky TLS terminator does:
        #
        #   accepts then stalls      TimeoutError        (an OSError)
        #   accepts then closes      RemoteDisconnected  (an OSError AND a
        #                                                 BadStatusLine)
        #   malformed status line    BadStatusLine       (an HTTPException,
        #                                                 NOT an OSError)
        #   truncated body           IncompleteRead      (an HTTPException)
        #
        # OSError covers the first two and http.client.HTTPException the last
        # two; URLError is listed first because it is the one this was written
        # for and dropping it would read as an accident. All four mean the
        # same thing to the reader, and all four used to be reported as
        # "something inside ModelMRI failed".
        #
        # A refusal, not a failure: nothing here broke, and the sentence says
        # what to do instead. It deliberately does not interpolate `err` —
        # this string is published to the browser and `str(URLError)` is
        # machinery talking to itself ("<urlopen error [Errno 11001]
        # getaddrinfo failed>"). The real exception goes to the terminal,
        # which a local-first tool can assume the reader has open.
        log.warning("hub search failed", exc_info=err)
        raise Refusal(
            "Could not reach the HuggingFace Hub. Check your connection — the "
            "full error is in the terminal running `modelmri serve`. Models "
            "already downloaded still load: open the 'On this machine' tab."
        ) from err

    out: list[dict] = []
    for m in raw if isinstance(raw, list) else []:
        gated = bool(m.get("gated", False))
        out.append(
            {
                "id": m.get("id"),
                "downloads": m.get("downloads", 0),
                "likes": m.get("likes", 0),
                "gated": gated,
                # Filled in below. Never assume: a token is not access.
                "usable": not gated,
                "updated": (m.get("lastModified") or "")[:10],
                "params": _param_hint(m),
                "size_gb": weight_bytes(m) / 1e9 or None,
            }
        )
    return _resolve_access(out, tok)


def _has_access(repo: str, tok: str | None) -> bool:
    """Can this token actually download this gated repo?

    Being signed in is NOT access. Gating is per-repo license acceptance, so
    an account can hold a valid token and still be refused. We shipped
    `(not gated) or bool(tok)` and it labelled every Gemma build "gated ✓"
    for an account that had never accepted Google's terms — the picker
    promised a model the loader then refused.
    """
    if not tok:
        return False
    # Deliberately not _api(): auth-check answers 200 with an EMPTY body, so
    # json.load() raises and every repo -- gated, ungated, accepted or not --
    # came back False. The first version of this shipped that way and looked
    # correct, because the repos I tested were ones I could not access anyway.
    # The status code is the answer: 200 yes, 403 licence not accepted,
    # 404 no such repo.
    req = urllib.request.Request(f"{HUB_API}/models/{repo}/auth-check")
    req.add_header("Authorization", f"Bearer {tok}")
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            return 200 <= r.status < 300
    except Exception as err:  # noqa: BLE001 - see below
        # Broad on purpose. This runs inside a thread pool over every gated
        # row in the picker, and an exception here does not stay here — it
        # comes back out of `pool.map` and takes the whole search down with
        # it, so one odd repo id would empty a list of forty working ones.
        #
        # Not a swallowed bug: False *is* the answer being computed, and it is
        # the pessimistic one. An auth-check that did not answer means we have
        # not established access, and the row says "gated" rather than
        # promising a download the loader would refuse — the exact failure the
        # docstring above was written about.
        log.debug("auth-check for %s failed (%s); treating as no access", repo, err)
        return False


def _resolve_access(entries: list[dict], tok: str | None) -> list[dict]:
    """Check the gated entries for real, concurrently. Usually only a few."""
    gated = [e for e in entries if e["gated"]]
    if not gated or not tok:
        return entries
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=8) as pool:
        verdicts = pool.map(lambda e: _has_access(e["id"], tok), gated)
    for entry, ok in zip(gated, verdicts):
        entry["usable"] = ok
    return entries


# Bytes per parameter, by the dtype names the Hub reports. Weights are stored
# at their own precision, so a 753B-parameter model is 1.5 TB in BF16 and
# 750 GB in FP8 — the parameter count alone cannot tell you what you are
# about to download.
_DTYPE_BYTES: dict[str, float] = {
    "F64": 8,
    "I64": 8,
    "F32": 4,
    "I32": 4,
    "U32": 4,
    "BF16": 2,
    "F16": 2,
    "I16": 2,
    "U16": 2,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
    "F4": 0.5,
    "I4": 0.5,
    "U4": 0.5,
}


def weight_bytes(model: dict) -> int:
    """How many bytes of weights this repo holds, from its own metadata.

    Reads the per-dtype parameter counts the Hub publishes for safetensors
    repos, so it is arithmetic on the repo's own numbers rather than a guess
    from the model name. Falls back to assuming 2 bytes per parameter when
    only a total is available, which is right for the overwhelming majority
    of published checkpoints.

    Returns 0 when the repo publishes nothing to go on -- GGUF and pickle
    repos, mostly. Callers must treat 0 as "unknown", never as "small".
    """
    st = model.get("safetensors") or {}
    by_dtype = st.get("parameters") or {}
    if by_dtype:
        return int(
            sum(_DTYPE_BYTES.get(str(d).upper(), 2) * n for d, n in by_dtype.items())
        )
    total = st.get("total")
    return int(total * 2) if total else 0


def _param_hint(model: dict) -> str | None:
    """Best-effort size label from safetensors metadata, e.g. '0.6B'."""
    try:
        total = (model.get("safetensors") or {}).get("total")
        if not total:
            return None
        if total >= 1e9:
            return f"{total / 1e9:.1f}B"
        return f"{total / 1e6:.0f}M"
    except (AttributeError, TypeError):
        # The Hub's metadata, shaped however it arrives. `.get` is not a method
        # if `safetensors` decodes to a list (AttributeError), and both the
        # comparison and the division fail if `total` arrives as a string
        # (TypeError). Returning None is the honest answer and the caller
        # already handles it: a row with no size label is fine, a picker that
        # died on one malformed repo is not.
        return None


def suggested() -> list[dict]:
    """The curated starter list, annotated the same way as search results.

    Fetched concurrently. One repo at a time took 3.4 seconds for eight
    models, and this is the view the picker opens on — so the first thing
    anyone saw was three and a half seconds of skeleton rows.
    """
    tok = token()
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=8) as pool:
        entries = list(pool.map(lambda repo: _suggested_entry(repo, tok), SUGGESTED))
    return _resolve_access(entries, tok)


def _suggested_entry(repo: str, tok: str | None) -> dict:
    """One curated model, annotated. Never raises: offline still offers the
    name, because a picker with nothing in it is worse than one with names
    and no metadata."""
    entry = {
        "id": repo,
        "downloads": 0,
        "likes": 0,
        "gated": False,
        "usable": True,
        "updated": "",
        "params": None,
        "size_gb": None,
        "suggested": True,
    }
    try:
        info = _api(f"/models/{repo}", tok, timeout=6)
        entry["downloads"] = info.get("downloads", 0)
        entry["likes"] = info.get("likes", 0)
        entry["gated"] = bool(info.get("gated", False))
        entry["updated"] = (info.get("lastModified") or "")[:10]
        entry["params"] = _param_hint(info)
        entry["size_gb"] = weight_bytes(info) / 1e9 or None
    except (OSError, ValueError, TypeError, AttributeError, http.client.HTTPException):
        # One Hub lookup plus the field reads after it, enumerated:
        #   OSError      urllib's HTTPError and URLError, socket timeouts,
        #                RemoteDisconnected and ssl.SSLError are all OSError
        #                subclasses
        #   http.client.HTTPException
        #                a malformed status line (BadStatusLine) or a body
        #                shorter than its Content-Length (IncompleteRead).
        #                These are NOT OSErrors — measured, both escaped this
        #                handler, and because `suggested()` runs it through
        #                `pool.map` one bad response emptied all eight rows and
        #                turned the view the picker opens on into a 500
        #   ValueError   json.load raising JSONDecodeError or
        #                UnicodeDecodeError when a captive portal or a
        #                corporate proxy answers with HTML
        #   AttributeError / TypeError
        #                a body that decodes to the wrong shape — `.get` on a
        #                list, `[:10]` on a number, `weight_bytes` doing
        #                arithmetic on a string
        #
        # Not narrower than that, because the docstring above is a promise the
        # caller leans on: `suggested()` maps this over eight repos, and one
        # raise would empty the view the picker opens on. Offline you still
        # get the eight names with no metadata, which is the entire point.
        log.debug("no Hub metadata for %s; offering the name alone", repo)
    entry["usable"] = not entry["gated"]
    return entry
