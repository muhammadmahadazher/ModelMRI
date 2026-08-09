"""HuggingFace Hub access: sign in, search models, see what you can use.

ModelMRI never asks for your password. You paste a HuggingFace *access
token* (huggingface.co/settings/tokens) — a scoped credential you can
revoke at any time — and it is stored with 0600 permissions in
~/.modelmri/hub.json, never in the repo and never sent anywhere except
huggingface.co.

Signing in unlocks gated models you have accepted the license for
(Gemma, Llama, ...) and your own private repos.
"""

from __future__ import annotations

import json
import os
import stat
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from . import paths

HUB_API = "https://huggingface.co/api"


def _config_path() -> Path:
    """Where the token lives.

    Platform convention, with the pre-0.6 `~/.modelmri/hub.json` still read if
    it exists — moving the default without looking at the old place would sign
    people out on upgrade and leave a token file behind that nobody knows is
    there.
    """
    from . import paths

    if legacy := paths.legacy_file("hub.json"):
        return legacy
    return paths.config_dir() / "hub.json"


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


def _read_stored_token() -> tuple[str | None, str | None]:
    """(token, source) — ours first, then the HF CLI's, then the env."""
    try:
        if _config_path().is_file():
            token = json.loads(_config_path().read_text(encoding="utf-8")).get("token")
            if token:
                return token, "modelmri"
    except Exception:
        pass

    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(var):
            return os.environ[var], "env"

    # whatever `huggingface-cli login` wrote
    for path in (
        paths.hf_home() / "token",
        Path.home() / ".huggingface" / "token",
    ):
        try:
            if path.is_file():
                token = path.read_text(encoding="utf-8").strip()
                if token:
                    return token, "huggingface-cli"
        except Exception:
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
    except Exception:
        return HubAuth(signed_in=False)


def sign_in(tok: str) -> HubAuth:
    """Validate a token, then store it privately. Raises ValueError if bad."""
    tok = (tok or "").strip()
    if not tok:
        raise ValueError("Paste a token from huggingface.co/settings/tokens")
    auth = whoami(tok)
    if not auth.signed_in:
        raise ValueError(
            "HuggingFace rejected that token. Create a fresh one with 'read' "
            "access at huggingface.co/settings/tokens."
        )
    _config_path().parent.mkdir(parents=True, exist_ok=True)
    _config_path().write_text(json.dumps({"token": tok}), encoding="utf-8")
    try:  # best effort on Windows, meaningful on posix
        _config_path().chmod(stat.S_IRUSR | stat.S_IWUSR)
    except Exception:
        pass
    auth.source = "modelmri"
    return auth


def sign_out() -> HubAuth:
    try:
        _config_path().unlink(missing_ok=True)
    except Exception:
        pass
    return whoami()  # a CLI/env token may still be active — report honestly


def search(query: str = "", limit: int = 24) -> list[dict]:
    """Text-generation models, newest-relevant first, annotated with access."""
    tok = token()
    params = {
        "limit": str(max(1, min(limit, 50))),
        "sort": "downloads",
        "direction": "-1",
        "filter": "text-generation",
        "full": "true",
    }
    if query.strip():
        params["search"] = query.strip()
    try:
        raw = _api("/models?" + urllib.parse.urlencode(params), tok)
    except urllib.error.URLError as err:
        raise RuntimeError(f"Could not reach the HuggingFace Hub: {err}") from err

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
    except Exception:
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


def _param_hint(model: dict) -> str | None:
    """Best-effort size label from safetensors metadata, e.g. '0.6B'."""
    try:
        total = (model.get("safetensors") or {}).get("total")
        if not total:
            return None
        if total >= 1e9:
            return f"{total / 1e9:.1f}B"
        return f"{total / 1e6:.0f}M"
    except Exception:
        return None


def suggested() -> list[dict]:
    """The curated starter list, annotated the same way as search results."""
    tok = token()
    out = []
    for repo in SUGGESTED:
        entry = {
            "id": repo,
            "downloads": 0,
            "likes": 0,
            "gated": False,
            "usable": True,
            "updated": "",
            "params": None,
            "suggested": True,
        }
        try:
            info = _api(f"/models/{repo}", tok, timeout=6)
            entry["downloads"] = info.get("downloads", 0)
            entry["likes"] = info.get("likes", 0)
            entry["gated"] = bool(info.get("gated", False))
            entry["updated"] = (info.get("lastModified") or "")[:10]
            entry["params"] = _param_hint(info)
        except Exception:
            pass  # offline: still offer the name
        entry["usable"] = not entry["gated"]
        out.append(entry)
    return _resolve_access(out, tok)
