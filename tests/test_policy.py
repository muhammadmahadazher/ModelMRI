"""The policy sidecar, and the four things it must never quietly do.

The action expert lives in another process because lerobot's pins cannot share
an environment with ModelMRI's. That boundary is the feature, and it is also
what makes this testable: a real sidecar is a subprocess speaking JSON on
loopback, so a fake one is a `ThreadingHTTPServer` in the test.

Nothing here needs lerobot, a policy, or a GPU. What is being tested is the
contract and the refusals, which are the parts that decide whether a number
reaching a panel is trustworthy:

  * a mismatched contract is refused, not best-efforted,
  * a missing sidecar refuses with the command that fixes it,
  * two processes holding weights is a capacity refusal, not an OOM,
  * empty normalisation means "do not overlay", never "identity".
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from modelmri import policy

# ------------------------------------------------------- a sidecar that isn't


def _fake_sidecar(handler_cls):
    """Serve `handler_cls` on a free loopback port for the duration of a test."""
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, httpd.server_address[1]


def _responder(payload: dict, *, code: int = 200, kind: str = "application/json"):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(length)
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-ModelMRI-Contract", str(payload.get("contract", "")))
            self.end_headers()
            self.wfile.write(body)

    return Handler


# --------------------------------------------------- the contract is checked


def test_a_sidecar_speaking_another_contract_is_refused():
    """ROADMAP #49: "contract drift silently serving actions from a stale
    policy is the worst failure available here." An action chunk is a claim
    about what a robot would DO, and one read across a version boundary is a
    different policy's answer wearing this one's name."""
    httpd, port = _fake_sidecar(_responder({"contract": policy.CONTRACT + 1}))
    try:
        with pytest.raises(policy.SidecarError, match="different wire formats"):
            policy._post(port, "/act", {}, timeout=5)
    finally:
        httpd.shutdown()


def test_a_sidecar_that_states_no_contract_is_refused():
    """Absence is not agreement. A response with no version is not from a
    sidecar this can trust, whatever else it contains."""
    httpd, port = _fake_sidecar(_responder({"action_chunk": [[0.0, 1.0]]}))
    try:
        with pytest.raises(policy.SidecarError, match="did not state a contract"):
            policy._post(port, "/act", {}, timeout=5)
    finally:
        httpd.shutdown()


def test_a_matching_contract_is_accepted():
    httpd, port = _fake_sidecar(
        _responder({"contract": policy.CONTRACT, "action_chunk": [[0.5]]})
    )
    try:
        data, _ = policy._post(port, "/act", {}, timeout=5)
        assert data["action_chunk"] == [[0.5]]
    finally:
        httpd.shutdown()


def test_true_is_not_a_contract_version():
    """`isinstance(True, int)` is True in Python, so a bare int check lets a
    boolean through and then compares it as 1."""
    with pytest.raises(policy.SidecarError, match="did not state a contract"):
        policy.check_contract(True)


def test_the_client_declares_its_own_version_rather_than_importing_one():
    """The two halves live in different venvs, so the client CANNOT import the
    sidecar's constant — and that is the point. Two independent declarations
    compared on every exchange is what makes drift visible; one shared import
    would move both halves together and there would be nothing to check."""
    src = __import__("pathlib").Path(policy.__file__).read_text("utf-8")
    assert "CONTRACT = " in src
    assert "from modelmri_policy" not in src, (
        "the client imported the sidecar's package, which defeats the handshake"
    )


# ------------------------------------------------------ a missing sidecar


def test_no_sidecar_refuses_with_the_command_that_fixes_it(monkeypatch, tmp_path):
    """Not a crash, not an empty action, not a zero-filled chunk. A zero-filled
    action chunk looks exactly like a policy deciding to hold still."""
    monkeypatch.setattr(policy, "venv_dir", lambda: tmp_path / "nope")
    with pytest.raises(policy.SidecarError) as caught:
        policy.require_installed()
    said = str(caught.value)
    assert "modelmri policy install" in said
    assert "separate virtual environment" in said


def test_a_sidecar_that_is_not_answering_says_so_rather_than_hanging():
    """A closed port is a sentence, not a traceback."""
    httpd, port = _fake_sidecar(_responder({"contract": policy.CONTRACT}))
    httpd.shutdown()
    httpd.server_close()
    with pytest.raises(policy.SidecarError, match="not answering"):
        policy._post(port, "/act", {}, timeout=3)


# ------------------------------------------- two processes, one graphics card


def test_a_second_process_holding_weights_is_refused_before_it_starts(monkeypatch):
    """This feature is the only thing in the package that puts a model in a
    SECOND process. On the 8 GB card this targets, a policy beside a loaded
    LLM is the normal configuration, and an OOM halfway through a sweep is
    what `capacity.guard` exists to prevent."""
    from modelmri import capacity

    with pytest.raises(capacity.TooBig) as caught:
        policy.check_capacity(
            vram_gb=8.0,
            accel_name="RTX 4060",
            already_held_bytes=6_000_000_000,
            policy_bytes=3_500_000_000,
        )
    assert "policy sidecar" in str(caught.value)


def test_the_capacity_check_counts_what_is_already_loaded():
    """A policy alone may fit where a policy beside a 1.7B does not, so the
    already-held bytes are part of the question rather than a footnote."""
    from modelmri import capacity

    policy.check_capacity(
        vram_gb=80.0, accel_name="A100", already_held_bytes=0, policy_bytes=1_000
    )
    with pytest.raises(capacity.TooBig):
        policy.check_capacity(
            vram_gb=8.0,
            accel_name="RTX 4060",
            already_held_bytes=7_500_000_000,
            policy_bytes=3_500_000_000,
        )


# -------------------------------------------------- what the status promises


def test_a_status_with_no_sidecar_says_what_is_missing_not_just_false():
    """ "running: false" is a fact about this process. The panel needs the
    reason, because "install it" and "it crashed" lead to different actions."""
    status = policy.PolicyStatus(reason="Nothing has been installed yet.")
    said = status.means()
    assert "only where it looked" in said
    assert "Nothing has been installed yet." in said


def test_a_running_status_never_calls_an_action_ground_truth():
    """A policy's output on a frame is that policy's output. ROADMAP #50 is
    explicit that a recorded action is one human demonstration rather than
    ground truth, and the same care applies in the other direction."""
    status = policy.PolicyStatus(
        running=True,
        policy_repo="lerobot/smolvla_base",
        revision="abc123",
        device="cuda:0",
    )
    said = status.means()
    assert "not a demonstration and not" in said
    assert "ground truth" in said


def test_normalisation_travels_so_an_overlay_can_be_refused():
    """A policy's actions are normalised against ITS training statistics and a
    dataset's against its own. Overlaying two curves in different units is the
    plausible-wrong output ROADMAP #50 refuses, and refusing needs both sides
    to have stated their units."""
    status = policy.PolicyStatus(running=True, normalisation={"mean": [0.0, 1.0]})
    assert status.to_dict()["normalisation"] == {"mean": [0.0, 1.0]}
    assert policy.PolicyStatus(running=True).to_dict()["normalisation"] == {}
