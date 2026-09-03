# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""`POST /api/vla/actions/compare` — the wire, not the arithmetic.

`tests/test_vla_actions.py` proves the maths, and it can, because `vla_actions`
does no I/O and holds no model. What it cannot prove is that the numbers reach
the browser: nothing in the suite called this route before this file existed,
and `tests/test_api_contract.py` cannot see it, because `_routes()` only matches
single-argument `fetch("/api/...")` calls and only issues GETs. Every POST route
in this app is invisible to it.

So the failure this file exists to catch is the cheap one: a block computed in
the module, added to `api.ts`, and never threaded through `_run_compare` — which
passes the entire rest of the suite while the panel renders nothing.

The other half is a regression. Chunk consistency is free measurement bought
with zero extra forward passes, and the price of it must be zero as well: the
predicted-versus-recorded section this route already returned has to come back
byte-identical, which is asserted here against the same comparison run WITHOUT
the chunks rather than against numbers copied from a previous run of the code
being tested.

The seams are `modelmri.policy.status` / `modelmri.policy.act` — the route does
`from . import policy as _policy` inside the function, so patching the module
attribute is what it will see — and `app.state.vla_reader`, which `_reader()`
returns unchanged when it is already set. That is the one place these routes
read `app.state` rather than a closure, so it is the one place a fake reader can
be injected.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from modelmri import policy as policy_module
from modelmri import vla_actions as va
from modelmri.policy import PolicyStatus
from modelmri.server import create_app

REPO = "lerobot/smolvla_base"
REVISION = "abc123"
# H = 3, D = 2. Step k of the chunk emitted at frame t claims frame t + k, so
# the first column is the identity — every chunk agrees with every other about
# WHERE the arm goes — and the second drifts by 0.25 per frame of hindsight.
# That makes the consistency numbers hand-computable: two chunks dt apart
# disagree by exactly 0.25 * dt, in dimension 1 alone.
HORIZON = 3


def chunk_at(t: int) -> list[list[float]]:
    return [[float(t + k), 10.0 + 0.25 * t] for k in range(HORIZON)]


class Ep:
    def __init__(self, index: int, length: int):
        self.index, self.length = index, length


class Frame:
    """The fields `_run_compare` reads off a `FrameSample`, and no others.

    `state` carries the frame index because the fake policy has no other way
    to know which frame it was asked about: the route hands it an image, a
    state, an instruction and a seed, and the image is a constant here.
    """

    def __init__(self, t: int):
        self.t = t
        self.image = "data:image/png;base64,AAAA"
        self.state = [float(t)]
        self.action = [0.0, 0.0]
        self.task = "push the T"


class FakeReader:
    repo_id = "lerobot/fake"

    def __init__(self, length: int = 4):
        self._eps = [Ep(0, length)]

    def episodes(self):
        return self._eps

    def frame(self, episode: int, t: int) -> Frame:
        return Frame(t)

    def action_names(self) -> list[str]:
        return ["x", "y"]

    def action_stats(self) -> dict:
        return {"action": {"mean": [0.0, 0.0], "std": [1.0, 1.0]}}


def fake_status() -> PolicyStatus:
    return PolicyStatus(
        running=True,
        policy_repo=REPO,
        revision=REVISION,
        cameras=["observation.images.top"],
        normalisation={"action": {"mean": [0.0, 0.0], "std": [1.0, 1.0]}},
        action_width=2,
        chunk_size=HORIZON,
        samples=True,
    )


def fake_act(*, frames, state, instruction, seed=None, **_kw) -> dict:
    return {"action_chunk": chunk_at(int(state[0])), "shape": [HORIZON, 2]}


@pytest.fixture
def client(monkeypatch, request) -> TestClient:
    length = getattr(request, "param", 4)
    monkeypatch.setattr(policy_module, "status", fake_status)
    monkeypatch.setattr(policy_module, "act", fake_act)
    app = create_app()
    app.state.vla_reader = FakeReader(length)
    return TestClient(app, raise_server_exceptions=False)


def test_the_consistency_block_reaches_the_wire(client):
    """The whole feature, at the only layer that can fail silently. A block
    computed in `vla_actions` and never passed through `_run_compare` still
    passes every module test and renders as an empty panel."""
    body = client.post("/api/vla/actions/compare", json={"episode": 0}).json()
    assert "chunk_consistency" in body, body.keys()
    block = body["chunk_consistency"]
    assert block["measurable"] is True
    # Four frames at a stride of 1 with H=3: dt=1 gives three pairs of two
    # shared timesteps each, dt=2 gives two pairs of one, and dt=3 reaches
    # nothing. Five pairs, eight shared timesteps.
    assert block["pairs"] == 5
    assert block["overlapping_steps"] == 8
    assert block["horizon"] == 3
    assert block["stride"] == 1
    # 0.25 per frame of hindsight: six readings of 0.25 and two of 0.5.
    assert block["median"] == 0.25
    assert [r["steps_ahead"] for r in block["by_steps_ahead"]] == [1, 2]
    assert block["by_steps_ahead"][0] == {
        "steps_ahead": 1,
        "pairs": 3,
        "overlapping_steps": 6,
        "median": 0.25,
        "p25": 0.25,
        "p75": 0.25,
    }
    assert block["by_steps_ahead"][1]["median"] == 0.5
    assert block["worst_pair"] == {
        "t_earlier": 0,
        "t_later": 2,
        "steps_ahead": 2,
        "step": 0,
        "distance": 0.5,
    }


def test_the_dimension_that_never_moved_reads_exactly_zero(client):
    """The joint names the dataset published, reused rather than re-derived,
    and the per-dimension independence the module proves — carried over the
    wire, because a block that arrives with its dimensions smeared together
    cannot say which joint the policy keeps changing its mind about."""
    body = client.post("/api/vla/actions/compare", json={"episode": 0}).json()
    assert body["chunk_consistency"]["by_dimension"] == [
        {"dimension": 0, "name": "x", "revision_bias": 0.0, "disagreement": 0.0},
        {"dimension": 1, "name": "y", "revision_bias": 0.3125, "disagreement": 0.25},
    ]


def test_the_predicted_versus_recorded_section_is_untouched(client):
    """The regression the whole workstream turns on. The reference is the same
    comparison run WITHOUT the chunks — not numbers copied out of a previous
    run of this code, which would agree with whatever the code became."""
    body = client.post("/api/vla/actions/compare", json={"episode": 0}).json()
    before = va.compare(
        frames=[(t, chunk_at(t)[0], [0.0, 0.0]) for t in range(4)],
        joint_names=["x", "y"],
        stride=1,
        total_frames=4,
        policy_repo=REPO,
        revision=REVISION,
        seed=None,
    )
    assert {k: v for k, v in body.items() if k != "chunk_consistency"} == {
        k: v for k, v in before.items() if k != "chunk_consistency"
    }
    # And the numbers themselves, so a change to BOTH sides is still caught.
    assert body["worst_frame"] == 3
    assert body["bias"] == [1.5, 10.375]
    assert body["frames_measured"] == 4
    assert body["frames_skipped"] == 0
    assert "NOT GROUND TRUTH" in body["means"]


@pytest.mark.parametrize("client", [20], indirect=True)
def test_a_stride_past_the_horizon_still_answers_200_with_the_error_section(client):
    """A refusal INSIDE a successful response. The chunks never overlap at a
    stride of 5 with a 3-step horizon, and the honest answer to that is the
    first-step comparison plus a sentence — not a 409 that throws away four
    forward passes of real measurement, and not a consistency of zero."""
    reply = client.post("/api/vla/actions/compare", json={"episode": 0, "stride": 5})
    assert reply.status_code == 200
    body = reply.json()
    assert body["frames_measured"] == 4
    assert body["worst_distance"] > 0
    block = body["chunk_consistency"]
    assert block["measurable"] is False
    assert block["median"] is None
    assert block["by_steps_ahead"] == []
    assert "NOT A CONSISTENCY OF ZERO" in block["means"]
    assert "5 frames apart" in block["means"]
    assert "is 3 steps" in block["means"]


def test_the_seed_and_the_revision_travel_into_the_new_sentence(client):
    """The receipt names the policy repo, the revision, the seed, the stride
    and the horizon. A strip drawn from this block is read on its own, and a
    number whose provenance is two paragraphs up is a number nobody checks."""
    body = client.post(
        "/api/vla/actions/compare", json={"episode": 0, "seed": 11}
    ).json()
    said = body["chunk_consistency"]["means"]
    assert REPO in said
    assert REVISION in said
    assert "seed 11" in said
    assert "stride of 1" in said
    assert "3-step horizon" in said
    assert "AGAINST ITSELF" in said
    assert "NO THRESHOLD IS APPLIED" in said


def test_a_policy_that_is_not_running_still_refuses_before_any_pass(
    client, monkeypatch
):
    """Unchanged behaviour, asserted because the new block sits inside the
    function the refusals guard. A 409 carrying the sidecar's own sentence is
    the answer here; a 500 would throw that sentence away."""
    monkeypatch.setattr(
        policy_module,
        "status",
        lambda: PolicyStatus(running=False, reason="no sidecar is running"),
    )
    reply = client.post("/api/vla/actions/compare", json={"episode": 0})
    assert reply.status_code == 409
    # The SIDECAR'S OWN sentence, not `str(err)` and not merely something
    # truthy: `err.sentence` is the whole of the house rule this route is an
    # instance of, and a truthiness assertion passes against any of the ways
    # of losing it.
    assert "no sidecar is running" in reply.json()["error"]


@pytest.mark.parametrize("declared", [None, 7])
def test_the_horizon_comes_from_the_chunk_returned_not_the_one_declared(
    client, monkeypatch, declared
):
    """`PolicyStatus.chunk_size` is not the horizon and cannot be used as one.
    The adapter reads it off `cfg.chunk_size`, a field lerobot's base config
    does not have: the diffusion and VQ-BeT families publish `None` there
    while returning real 32- and 5-step chunks, and even ACT disagrees with
    itself between `predict_action_chunk` and `select_action`. A block that
    read the declared number would go unmeasurable on exactly the policies
    whose chunks are longest.

    The fixture elsewhere in this file sets `chunk_size` equal to the length
    of the chunk the fake returns, so it cannot tell the two apart. These two
    parameters are the wrong number and no number at all."""
    monkeypatch.setattr(
        policy_module,
        "status",
        lambda: PolicyStatus(
            running=True,
            policy_repo=REPO,
            revision=REVISION,
            cameras=["observation.images.top"],
            normalisation={"action": {"mean": [0.0, 0.0], "std": [1.0, 1.0]}},
            action_width=2,
            chunk_size=declared,
            samples=True,
        ),
    )
    body = client.post("/api/vla/actions/compare", json={"episode": 0}).json()
    block = body["chunk_consistency"]
    assert block["measurable"] is True
    assert block["horizon"] == HORIZON
    assert block["horizon_min"] == HORIZON and block["horizon_max"] == HORIZON
    assert f"{HORIZON}-step horizon" in block["means"]
    assert block["overlapping_steps"] == 8
