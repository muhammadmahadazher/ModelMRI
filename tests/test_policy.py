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
import pathlib
import re
import sys
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

        def _answer(self):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-ModelMRI-Contract", str(payload.get("contract", "")))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self._answer()

        def do_GET(self):
            self._answer()

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
    src = pathlib.Path(policy.__file__).read_text("utf-8")
    assert "CONTRACT = " in src

    # The whole package, not just this module. `modelmri-policy` IS installed
    # in the dev environment — its input rules are tested here rather than
    # behind an `importorskip` that would go dark — so "it does not import
    # because it cannot" stopped being true and this became a real check
    # rather than a restatement of the venv layout.
    #
    # IMPORTS, not mentions. The module docstrings name
    # `modelmri_policy.contract` precisely to explain why the version is
    # declared twice, and a check that banned the words would ban the
    # explanation of itself.
    root = pathlib.Path(policy.__file__).parent
    offenders = [
        f"{path.relative_to(root)}:{n}"
        for path in sorted(root.rglob("*.py"))
        for n, line in enumerate(path.read_text("utf-8").splitlines(), 1)
        if re.match(r"\s*(?:from|import)\s+modelmri_policy\b", line)
        or 'import_module("modelmri_policy' in line
    ]
    assert not offenders, (
        f"these import or name the sidecar's package: {offenders}. Two "
        f"independent declarations of the contract version, compared on every "
        f"exchange, is what makes drift visible — a shared import would move "
        f"both halves together and there would be nothing left to check."
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


def test_a_status_never_reports_an_empty_normalisation_as_identity():
    """The word "identity" would be a claim. Empty means the policy did not
    publish its statistics, and the only honest thing to do with that is
    refuse the overlay."""
    said = policy.PolicyStatus(running=True, normalisation={}).to_dict()
    assert said["normalisation"] == {}
    assert "identity" not in said["means"].lower()


def test_normalisation_travels_so_an_overlay_can_be_refused():
    """A policy's actions are normalised against ITS training statistics and a
    dataset's against its own. Overlaying two curves in different units is the
    plausible-wrong output ROADMAP #50 refuses, and refusing needs both sides
    to have stated their units."""
    status = policy.PolicyStatus(running=True, normalisation={"mean": [0.0, 1.0]})
    assert status.to_dict()["normalisation"] == {"mean": [0.0, 1.0]}
    assert policy.PolicyStatus(running=True).to_dict()["normalisation"] == {}


# ------------------------------------------- installing, starting, stopping


def test_a_stale_port_file_is_a_crash_report_not_a_running_sidecar(
    monkeypatch, tmp_path
):
    """The file outlives the process that wrote it. A port read out of a file
    is a claim, and `status` decides by ASKING — otherwise a crashed sidecar
    stays "running" until somebody reboots."""
    recorded = tmp_path / "policy-port.json"
    recorded.write_text(json.dumps({"port": 1, "pid": 999999, "contract": 1}))
    monkeypatch.setattr(policy, "port_file", lambda: recorded)

    status = policy.status(timeout=1.0)
    assert not status.running
    assert "has exited" in status.reason
    # And the dead claim is gone, so the next caller is told "not running"
    # rather than being sent to the same dead port.
    assert not recorded.exists()


def test_a_port_file_that_is_not_a_port_is_no_sidecar(monkeypatch, tmp_path):
    """Corrupt and absent mean the same thing to every caller: nothing to
    attach to. 70000 is not a port; True is not a port either."""
    recorded = tmp_path / "policy-port.json"
    monkeypatch.setattr(policy, "port_file", lambda: recorded)

    for payload in (
        '{"port": 70000}',
        '{"port": true}',
        '{"port": "5900"}',
        "not json",
    ):
        recorded.write_text(payload)
        assert policy._read_port() == 0, payload


def test_status_with_nothing_installed_names_the_install_command(monkeypatch, tmp_path):
    monkeypatch.setattr(policy, "venv_dir", lambda: tmp_path / "nope")
    monkeypatch.setattr(policy, "port_file", lambda: tmp_path / "gone.json")
    status = policy.status()
    assert not status.running
    assert "modelmri policy install" in status.reason


def test_a_sidecar_up_with_no_policy_is_not_reported_as_running(monkeypatch, tmp_path):
    """ "the process exists" and "it can answer what the robot would do" are
    different states, and only the second is `running`."""
    httpd, port = _fake_sidecar(
        _responder({"contract": policy.CONTRACT, "ready": False})
    )
    try:
        state = _status_on(port, monkeypatch, tmp_path)
        assert not state.running
        assert state.port == port
        assert "no policy loaded" in state.reason
    finally:
        httpd.shutdown()


def test_a_ready_sidecar_reports_what_it_is_holding(monkeypatch, tmp_path):
    httpd, port = _fake_sidecar(
        _responder(
            {
                "contract": policy.CONTRACT,
                "ready": True,
                "policy_repo": "lerobot/smolvla_base",
                "revision": "deadbeef",
                "device": "cuda:0",
                "dtype": "bfloat16",
                "normalisation": {"mean": [0.0]},
            }
        )
    )
    try:
        state = _status_on(port, monkeypatch, tmp_path)
        assert state.running
        assert state.policy_repo == "lerobot/smolvla_base"
        assert state.revision == "deadbeef"
        assert state.normalisation == {"mean": [0.0]}
    finally:
        httpd.shutdown()


def test_a_status_route_speaking_the_wrong_contract_is_refused():
    """The version rides on EVERY exchange, `/status` included. A sidecar can
    be restarted — upgraded, even — under a live client, and a check that ran
    once at connect would not see it."""
    httpd, port = _fake_sidecar(
        _responder({"contract": policy.CONTRACT + 7, "ready": True})
    )
    try:
        with pytest.raises(policy.SidecarError, match="different wire formats"):
            policy._get(port, "/status", timeout=5)
    finally:
        httpd.shutdown()


def _status_on(port: int, monkeypatch, tmp_path):
    """Point `status` at a fake sidecar by writing the port file it reads."""
    handle = pathlib.Path(tmp_path) / "policy-port.json"
    handle.write_text(json.dumps({"port": port, "pid": 0, "contract": 1}))
    monkeypatch.setattr(policy, "port_file", lambda: handle)
    return policy.status(timeout=5.0)


def test_loading_a_policy_with_no_sidecar_refuses_rather_than_posting_nowhere(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(policy, "port_file", lambda: tmp_path / "absent.json")
    with pytest.raises(policy.SidecarError, match="nothing to load a policy into"):
        policy.load("lerobot/smolvla_base")


def test_stop_reports_false_when_this_process_started_nothing(monkeypatch, tmp_path):
    """It only kills a child of THIS process. Signalling a pid out of a file
    means signalling whatever inherited that pid after a crash."""
    monkeypatch.setattr(policy, "port_file", lambda: tmp_path / "absent.json")
    monkeypatch.setattr(policy, "_CHILD", None)
    assert policy.stop() is False


def test_the_requirement_prefers_the_checkout_over_the_release():
    """A contributor editing the sidecar wants THEIR copy in the venv; a user
    who ran `pip install modelmri` wants the published one. This checkout has
    `packages/modelmri-policy`, so this run must resolve to the local path."""
    want = policy.requirement()
    assert want.endswith("[policy]")
    local = policy.source_dir()
    if local is not None:
        assert str(local) in want
    else:
        assert want == "modelmri-policy[policy]"


def test_an_environment_without_the_sidecar_in_it_is_refused_by_the_handshake(
    tmp_path,
):
    """An install that FINISHES is not an install that WORKS, and the probe is
    what separates the two.

    Against a REAL empty environment rather than a stub. `--without-pip` makes
    one in about a third of a second, and it is the honest shape of the
    failure this catches: a venv that exists, has an interpreter, and cannot
    import what it was built for. Deliberately not the suite's own interpreter
    — whether that happens to have `modelmri_policy` on its path is a fact
    about somebody's machine, and a test that asserts it is testing the
    machine.
    """
    import subprocess

    bare = tmp_path / "bare"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(bare)], check=True
    )

    with pytest.raises(policy.SidecarError) as caught:
        policy.probe(policy.python_in(bare))
    said = str(caught.value)
    assert "does not have a working" in said
    assert "install --force" in said
    # The child's own words, not a swallowed exit code.
    assert "ModuleNotFoundError" in said or "No module named" in said


def test_a_broken_environments_refusal_is_not_re_wrapped(monkeypatch, tmp_path):
    """`probe` already names the environment, quotes what its interpreter said
    and gives the remedy. Re-wrapping it would either duplicate that or blur
    the specifics — and it would interpolate a caught exception's text, which
    `test_no_exception_leaks` exists to stop."""
    venv = tmp_path / "policy-venv"
    where = venv / ("Scripts" if sys.platform == "win32" else "bin")
    where.mkdir(parents=True)
    (where / ("python.exe" if sys.platform == "win32" else "python")).write_text("")
    monkeypatch.setattr(policy, "venv_dir", lambda: venv)
    monkeypatch.setattr(
        policy,
        "probe",
        lambda _exe: (_ for _ in ()).throw(policy.SidecarError("exact words.")),
    )
    with pytest.raises(policy.SidecarError) as caught:
        policy.install()
    assert str(caught.value) == "exact words."


def test_a_cancelled_install_says_the_environment_is_unusable(monkeypatch, tmp_path):
    """Half a torch is not a working environment, and "stopped" that leaves
    somebody thinking otherwise is the wrong message."""
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(policy.SidecarError, match="not usable"):
        policy._run(
            [__import__("sys").executable, "-c", "import time; time.sleep(30)"],
            cancel=cancel,
            timeout=30.0,
        )


def test_the_child_runner_returns_output_as_well_as_a_code():
    """The tail exists to be quoted in a refusal — pip puts the reason at the
    END — so it has to actually arrive."""
    code, tail = policy._run(
        [__import__("sys").executable, "-c", "print('hello from the child')"],
        timeout=60.0,
    )
    assert code == 0
    assert "hello from the child" in tail


# ------------------------------------- which Python can hold the policy at all


def test_the_venv_interpreter_is_asked_its_version_not_read_off_its_name():
    """`python3.12` on PATH is a name somebody chose, and a symlink pointing
    somewhere else is exactly how you end up with a venv one minor too old and
    a pip error about `requires-python` that reads as a broken install."""
    got = policy._version_of(pathlib.Path(sys.executable))
    assert got == sys.version_info[:2]


def test_something_that_is_not_an_interpreter_answers_none_rather_than_raising():
    """`None` and not an exception: "this candidate cannot answer" is a step in
    a search, not a failure of the search."""
    assert policy._version_of(pathlib.Path("definitely-not-a-python-xyz")) is None


def test_this_interpreter_is_preferred_when_it_is_new_enough():
    """Borrowing a different Python builds a second CUDA wheel set against a
    different ABI. It is a fallback, never a preference."""
    if sys.version_info < policy.POLICY_PYTHON_MIN:
        pytest.skip("this interpreter is below lerobot's floor, so the fallback runs")
    assert policy.interpreter_for_venv() == pathlib.Path(sys.executable)


def test_too_old_with_nothing_newer_refuses_naming_lerobots_floor(monkeypatch):
    """lerobot declares `requires-python >=3.12`; ModelMRI supports 3.10. On
    two of the four Pythons this project is tested against, the sidecar's venv
    cannot be built from `sys.executable` at all — and the refusal has to say
    that ModelMRI itself does not need to move, which is the entire point of
    the two being separate."""
    monkeypatch.setattr(sys, "version_info", (3, 10, 0, "final", 0))
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr("shutil.which", lambda _name: None)
    with pytest.raises(policy.SidecarError) as caught:
        policy.interpreter_for_venv()
    said = str(caught.value)
    assert "3.12 or newer" in said
    assert "lerobot's own floor" in said
    assert "ModelMRI itself can stay where it is" in said


def test_a_newer_interpreter_on_path_is_used_when_this_one_is_too_old(monkeypatch):
    """The fallback has to actually work, not just refuse politely. This run's
    own interpreter stands in for "the newer one found on PATH" — it is a real
    Python of a real version, which is what the search is looking for."""
    if sys.version_info < policy.POLICY_PYTHON_MIN:
        pytest.skip("needs an interpreter at or above the floor to stand in")
    monkeypatch.setattr(sys, "version_info", (3, 10, 0, "final", 0))
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        "shutil.which",
        lambda name: sys.executable if name.startswith("python3.") else None,
    )
    assert policy.interpreter_for_venv() == pathlib.Path(sys.executable)
