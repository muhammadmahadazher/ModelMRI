"""Nothing the project did not author reaches the browser.

The rule this file enforces is one the codebase already states, in the arm
that handles a float32-on-CPU fallback: "only their exception CLASSES are
interpolated (never their text)". Refusal and BadRequest carry sentences
written for the reader and are relayed deliberately. Everything else — a
torch message, a urllib errno, an SSL failure, a safetensors header error —
is a library talking about a machine, and library text routinely carries
absolute paths and site-packages frames.

Five sites broke that rule, and none of them was on a route anybody had
hardened. `POST /api/model/load` correctly answers a fixed sentence at 500;
the same failure was simultaneously written into the progress snapshot, which
`GET /api/model/progress` returns verbatim at 200 once a second because the
load meter polls it. The other four were the same shape: a sink nobody
thought of as a response.

Each test below reproduces one of them with an exception carrying a path and
a frame, and asserts on the bytes that come back. They are written against
the SINKS rather than the source lines, so moving the code does not quietly
retire the coverage.
"""

from __future__ import annotations

import ssl
import urllib.error
from pathlib import Path

import pytest

# Text that can only appear if a library's own message was relayed. Kept
# deliberately broad: the failure this catches is "somebody pasted str(err)",
# and the specific words vary by library and platform.
FINGERPRINTS = (
    "site-packages",
    "AppData",
    "/home/",
    "Traceback",
    "cacert.pem",
    "serialization.py",
    "torch/nn/modules",
    "_ssl.c",
)

CERT = "/opt/hostedtoolcache/Python/lib/site-packages/certifi/cacert.pem"
WEIGHTS = "/home/someone/unreleased/model.pt"


def leaked(text: str) -> list[str]:
    return [f for f in FINGERPRINTS if f in text]


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from modelmri.server import create_app

    with TestClient(create_app()) as c:
        yield c


# ---------------------------------------------------- the progress snapshot


def test_a_failed_load_does_not_publish_the_exception(client, monkeypatch):
    """The one that started this.

    Measured before the fix, from a load failing and the very next poll:

        200 {"error": "RuntimeError: CUDA out of memory. Tried to allocate
             2.00 GiB. Loading /home/<name>/.../model.safetensors ...
             site-packages/torch/nn/modules/module.py, line 1518"}

    The 500 arm was hardened; this sibling GET was not.
    """
    import transformers

    boom = RuntimeError(
        f"CUDA out of memory. Tried to allocate 2.00 GiB. Loading {WEIGHTS} "
        f'File "{CERT}", line 1518'
    )

    def explode(*_a, **_k):
        raise boom

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", staticmethod(explode)
    )
    client.post("/api/model/load", json={"id": "gpt2", "source": "hf"})

    body = client.get("/api/model/progress").text
    assert not leaked(body), f"the progress snapshot leaked {leaked(body)}"
    # Still says something useful: which KIND of failure, and where to look.
    assert "RuntimeError" in body
    assert "terminal" in body


def test_the_load_route_itself_stays_generic(client, monkeypatch):
    """The arm that was already right, so a future edit cannot quietly undo
    it while this file is watching the other one."""
    import transformers

    def explode(*_a, **_k):
        raise RuntimeError(f"boom {WEIGHTS} {CERT}")

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", staticmethod(explode)
    )
    resp = client.post("/api/model/load", json={"id": "gpt2", "source": "hf"})
    assert resp.status_code == 500
    assert not leaked(resp.text)


# ------------------------------------------------------------------ ollama


def test_the_registry_lookup_answers_without_the_exception(client, monkeypatch):
    """`/api/ollama/resolve` returns its error as DATA on a 200, so no except
    arm anywhere sanitises it. An SSL failure put the CA bundle's absolute
    path in that body."""
    import urllib.request

    def explode(*_a, **_k):
        raise ssl.SSLCertVerificationError(
            f"[SSL: CERTIFICATE_VERIFY_FAILED] (_ssl.c:1000) CA bundle {CERT}"
        )

    monkeypatch.setattr(urllib.request, "urlopen", explode)
    resp = client.get("/api/ollama/resolve", params={"name": "llama3.2:1b"})
    assert resp.status_code == 200
    assert not leaked(resp.text), f"the registry answer leaked {leaked(resp.text)}"
    assert "SSLCertVerificationError" in resp.text


def test_unreachable_keeps_the_errno_and_drops_the_text():
    """The errno is the part a reader acts on and cannot carry a path. The
    rest of the message can and did."""
    import errno

    from modelmri import ollama

    refused = str(
        ollama._unreachable(
            "http://127.0.0.1:11434",
            urllib.error.URLError(
                # The NAME, not the number. errno values are not portable —
                # 111 is ECONNREFUSED on Linux and EIDRM on Windows, which is
                # how the first version of this test managed to fail on the
                # machine it was written on.
                ConnectionRefusedError(errno.ECONNREFUSED, "Connection refused")
            ),
        )
    )
    assert not leaked(refused)
    assert "ECONNREFUSED" in refused


def test_unreachable_names_the_class_when_there_is_no_errno():
    """An SSL failure has no errno, and its str carries the CA bundle path.
    The old code's docstring argued a reason "is an errno sentence, never a
    path from this machine" — true of the case it considered, and there was a
    second case."""
    from modelmri import ollama

    refused = str(
        ollama._unreachable(
            "https://ollama.internal",
            urllib.error.URLError(ssl.SSLCertVerificationError(f"CA bundle {CERT}")),
        )
    )
    assert not leaked(refused), f"the refusal leaked {leaked(refused)}"
    assert "SSLCertVerificationError" in refused


# ------------------------------------------------------------------ custom


def test_an_unreadable_checkpoint_names_the_class_only(tmp_path, monkeypatch):
    """The caller supplied a path, so echoing it back is not the leak — the
    leak is torch's own message, which carries a serialization.py frame."""
    import torch

    from modelmri import custom

    weights = tmp_path / "probe.pt"
    weights.write_bytes(b"not a checkpoint")

    def not_torchscript(*_a, **_k):
        raise RuntimeError("not a TorchScript archive")

    def unreadable(*_a, **_k):
        raise RuntimeError(
            f"PytorchStreamReader failed reading zip archive. Loading {WEIGHTS} "
            f'File "{CERT}", line 1487'
        )

    monkeypatch.setattr(torch.jit, "load", not_torchscript)
    monkeypatch.setattr(torch, "load", unreadable)

    with pytest.raises(Exception) as err:
        custom.load_torchscript(weights)
    said = str(err.value)
    assert not leaked(said), f"the adapter error leaked {leaked(said)}"
    assert "RuntimeError" in said
    # And it still gives the actual advice, which is the point of the message.
    assert "pickle" in said


# --------------------------------------------------------------- attribute


def test_a_model_without_attentions_is_reported_by_class(monkeypatch):
    """`why` is published to the caller. A flash-attn-only or state-space
    model raises from inside torch, and that text carries a module.py frame."""
    import torch

    from modelmri import attribute

    class NoAttentions(torch.nn.Module):
        """Reads position_ids, so it clears rank_tokens' own guard, and
        refuses to produce attentions — the case the arm exists for."""

        def __init__(self) -> None:
            super().__init__()
            self.emb = torch.nn.Embedding(64, 8)
            self.pos = torch.nn.Embedding(64, 8)
            self.head = torch.nn.Linear(8, 64)

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            output_attentions=False,
            **_kw,
        ):
            if output_attentions:
                raise RuntimeError(
                    f'eager attention is not implemented. File "{CERT}", line 1518'
                )
            if position_ids is None:
                position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
            x = self.emb(input_ids) + self.pos(position_ids.clamp(0, 63))
            if attention_mask is not None:
                x = x * attention_mask.unsqueeze(-1).float()
            return type("Out", (), {"logits": self.head(x), "attentions": None})()

    torch.manual_seed(0)
    out = attribute.rank_tokens(
        NoAttentions().eval(),
        torch.tensor([[1, 2, 3, 4, 5, 6]]),
        position=5,
        decode=lambda t: f"t{t}",
    )
    said = str(out.get("mask_check", {}))
    assert not leaked(said), f"the mask check leaked {leaked(said)}"
    assert "RuntimeError" in said


# ------------------------------------------------------- the rule, in source


def test_no_sink_interpolates_a_caught_exceptions_text():
    """A grep, deliberately, because the sinks are what future edits add to.

    Anything that formats a bare `{err}` into a string is suspect; the
    allowed form is `{type(err).__name__}`. Refusal and BadRequest raised
    from an authored sentence are fine and do not match this pattern.
    """
    import re

    root = Path(__file__).resolve().parents[1] / "modelmri"
    # `{err}` / `{exc}` / `{e}` inside an f-string, but not `{type(err)...}`.
    #
    # And the ATTRIBUTE forms, which this missed for a year. `{err.reason}` on
    # a `URLError` is the underlying OSError, whose text is whatever the
    # operating system wrote — the same leak as `{err}`, wearing an attribute
    # and sailing straight past a pattern that only looked for the bare name.
    #
    # CodeQL found it first (py/stack-trace-exposure, five routes at once,
    # tracing `URLError.reason` through `policy.status().reason` into the
    # `/api/policy` body). A check that a scanner has to catch for you is a
    # check that was not doing its job, so the pattern grew rather than the
    # finding being dismissed as a duplicate.
    #
    # `.args`, `.strerror`, `.filename` and `.reason` are the four that carry
    # host text; `type(err).__name__` and `err.name` stay allowed, the latter
    # being a module name and bounded.
    # And the FALLBACK form, which this missed until an audit went looking.
    #
    # Six sites wrote `{err.strerror or err}`. The bare `err` after the `or`
    # publishes `str(err)` — the exact thing this test exists to forbid — and
    # the old pattern did not match, because ` or err` sits inside the braces
    # and the regex only allowed an optional attribute there.
    #
    # It was not theoretical. A single-argument `OSError` has `strerror =
    # None`, so the fallback is what runs, and pyarrow raises exactly that:
    #
    #   pq.read_table("/definitely/not/here.parquet")
    #   -> FileNotFoundError, strerror None,
    #      fallback publishes "/definitely/not/here.parquet"
    #
    # `datasets.py` reads parquet, so two of the six were one bad path away
    # from putting an absolute path in a 422. The fix is
    # `err.strerror or type(err).__name__`, and the pattern now covers the
    # shape so the next one fails here instead of shipping.
    #
    # `{name}`, `{p.name}` and `{type(err).__name__}` still do not match: the
    # alternation is anchored to the exception NAMES, and `err.name` on an
    # ImportError is a module name and bounded.
    suspect = re.compile(
        r"\{(?:err|exc|error|cpu_err)"
        r"(?:\.(?:reason|args|strerror|filename))?"
        # an `or` fallback to the raw exception, with or without an attribute
        r"(?:\s+or\s+(?:err|exc|error|cpu_err)\b(?!\.))?"
        r"\}"
    )
    offenders = []
    for path in sorted(root.rglob("*.py")):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or "leak-ok:" in line:
                continue
            # The terminal is where the real exception BELONGS — `_internal`
            # and every fixed sink here logs it deliberately, because dropping
            # the text without recording it trades a leak for an erasure. Only
            # strings that can be published count.
            if "file=sys.stderr" in line or stripped.startswith(("log.", "print(")):
                continue
            if suspect.search(line):
                offenders.append(f"{path.relative_to(root)}:{n}  {stripped[:70]}")
    assert not offenders, (
        "these interpolate a caught exception's own text, which reaches the "
        "browser wherever the value is published:\n  " + "\n  ".join(offenders)
    )


# ------------------------------------------------------------------- image


def test_the_image_load_route_names_the_class_only(client, monkeypatch, tmp_path):
    """This file had ZERO image-route coverage while `image_runtime.py` grew
    from 587 to 732 lines in one sitting.

    The static grep below catches `{err}` on a line. It cannot catch a leak
    laundered through a LOCAL VARIABLE — `why = str(err)` on one line and
    `f"...{why}"` on the next — which is the exact shape the refusals in
    `_load_processor` and `_load_transformers` are built in. So this asserts
    at the SINK, on the bytes that come back.
    """
    checkpoint = tmp_path / "vit"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        '{"model_type": "vit", "architectures": ["ViTForImageClassification"]}',
        encoding="utf-8",
    )
    (checkpoint / "model.safetensors").write_bytes(
        b"\x08\x00\x00\x00\x00\x00\x00\x00{}"
    )

    def explode(*_a, **_k):
        raise RuntimeError(f"boom {WEIGHTS} {CERT}")

    for name in ("AutoModelForImageClassification", "AutoModel"):
        # Dotted string, NOT `setattr(transformers, name, ...)`. transformers
        # is a lazy module: until an attribute is first read it lives behind
        # `__getattr__` rather than in `__dict__`, so setting it on the module
        # object works only when some earlier import happened to materialise
        # it. That makes the patch depend on test ORDER, which is how this
        # very test first passed for the wrong reason.
        monkeypatch.setattr(
            f"transformers.{name}.from_pretrained", staticmethod(explode)
        )

    # A local path, so the machine guard fires first and correctly. Patched
    # out to reach the layer beneath it — that refusal has its own tests.
    monkeypatch.setattr("modelmri.server._not_from_this_machine", lambda *a, **k: None)
    resp = client.post("/api/image/load", json={"repo": str(checkpoint)})
    assert not leaked(resp.text), f"the image load route leaked {leaked(resp.text)}"
    # And it still says which KIND of failure, which is the actionable half.
    assert "RuntimeError" in resp.text


def test_a_preprocessor_that_will_not_import_names_the_package_not_the_message(
    tmp_path, monkeypatch
):
    """`_load_processor` scans an ImportError's prose for a known package name
    rather than relaying it. transformers writes "requires the Torchvision
    library but it was not found in your environment" — quoting that verbatim
    is how library text reaches a browser, and the package name alone is the
    whole actionable part."""
    from modelmri import image_runtime as ir

    class _Boom:
        @staticmethod
        def from_pretrained(*_a, **_k):
            raise ImportError(
                f"FastImageProcessor requires the Torchvision library but it "
                f"was not found. Looked in {CERT} and {WEIGHTS}"
            )

    for name in ir._PROCESSOR_CLASSES:
        monkeypatch.setattr(f"transformers.{name}", _Boom)

    _, why = ir._load_processor(tmp_path)
    assert not leaked(why), f"the preprocessor reason leaked {leaked(why)}"
    assert "torchvision" in why


def test_a_hub_failure_names_the_class_and_not_the_url(tmp_path, monkeypatch):
    """`snapshot_download` puts the full URL, the cache directory and an
    authentication paragraph into its message, and the cache directory is a
    path on this machine."""
    import huggingface_hub

    from modelmri import image_runtime as ir

    def explode(*_a, **_k):
        raise OSError(
            f"Repository Not Found for url https://huggingface.co/api/models/x. "
            f"Cache at {WEIGHTS}, certs at {CERT}"
        )

    monkeypatch.setattr(huggingface_hub, "snapshot_download", explode)
    with pytest.raises(Exception) as caught:
        ir._snapshot("owner/name", ["*.json"], local_ok=False)
    said = str(caught.value)
    assert not leaked(said), f"the hub refusal leaked {leaked(said)}"
    assert "owner/name" in said


def test_a_falsy_strerror_falls_back_to_the_class_and_not_the_path(monkeypatch):
    """The leak the static check could not see, measured rather than argued.

    Six sites wrote `{err.strerror or err}`. A single-argument `OSError` has
    `strerror = None`, so the fallback is what actually runs — and pyarrow
    raises exactly that shape:

        pq.read_table("/definitely/not/here.parquet")
        -> FileNotFoundError, strerror None, str(err) == the path

    `datasets.py` reads parquet, so two of the six were one bad path away from
    putting an absolute path into a 422. The regex above missed all six
    because ` or err` sits inside the braces.
    """
    from modelmri import datasets
    from modelmri.errors import BadRequest

    # Fails the way pyarrow's reader does: a SINGLE-ARGUMENT OSError, so
    # `strerror` is None and the `or` fallback is what actually runs. The
    # three-argument form stdlib `open` raises has a real `strerror` and
    # never reached the fallback, which is why this went unnoticed.
    def _single_arg(*_a, **_k):
        raise FileNotFoundError(f"Failed to open local file: {WEIGHTS}")

    monkeypatch.setattr(Path, "open", _single_arg)

    with pytest.raises(BadRequest) as caught:
        datasets.read_dataset(Path("cases.jsonl"))
    said = str(caught.value)
    assert not leaked(said), f"a dataset read leaked {leaked(said)}"
    # The class survives, because which KIND of failure it was is the
    # actionable half and a class name cannot carry a path.
    assert "FileNotFoundError" in said
