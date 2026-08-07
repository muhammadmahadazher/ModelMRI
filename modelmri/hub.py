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

HUB_API = "https://huggingface.co/api"
CONFIG = Path.home() / ".modelmri" / "hub.json"

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
        if CONFIG.is_file():
            token = json.loads(CONFIG.read_text(encoding="utf-8")).get("token")
            if token:
                return token, "modelmri"
    except Exception:
        pass

    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(var):
            return os.environ[var], "env"

    # whatever `huggingface-cli login` wrote
    for path in (
        Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        / "token",
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
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps({"token": tok}), encoding="utf-8")
    try:  # best effort on Windows, meaningful on posix
        CONFIG.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except Exception:
        pass
    auth.source = "modelmri"
    return auth


def sign_out() -> HubAuth:
    try:
        CONFIG.unlink(missing_ok=True)
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
        gated = m.get("gated", False)
        out.append(
            {
                "id": m.get("id"),
                "downloads": m.get("downloads", 0),
                "likes": m.get("likes", 0),
                "gated": bool(gated),
                # you can use a gated repo only once signed in AND accepted
                "usable": (not gated) or bool(tok),
                "updated": (m.get("lastModified") or "")[:10],
                "params": _param_hint(m),
            }
        )
    return out


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
            entry["usable"] = (not entry["gated"]) or bool(tok)
            entry["updated"] = (info.get("lastModified") or "")[:10]
            entry["params"] = _param_hint(info)
        except Exception:
            pass  # offline: still offer the name
        out.append(entry)
    return out
