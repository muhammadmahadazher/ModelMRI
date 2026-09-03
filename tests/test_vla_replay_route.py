# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""A robot finding somebody sent you has to be readable when it arrives.

`/api/vla/share` has written a validated robot section since the robot work
landed. `session._vla` has guarded it to `_patch`'s standard the whole time,
and `mcp_server` has advertised it in its `has` dict. Nothing served it back.

So the Share button in `VLACausal` produced a `.mri` whose recipient opened an
empty text session -- "1 tokens, 0 attention maps" -- with no mention of the
policy, the frame, or the occlusion map. Measured, before the fix: the words
"robot", "occlusion" and "episode" appeared nowhere on the page, and
`modelmri inspect` printed the same empty session.

That is the third section in this project to carry a writer with no reader.
The agent trace was the first (`/api/session/trace`) and the image run the
second (`/api/image/replay`), and all three were found the same way -- by
opening a real file rather than assuming a writer implies a reader. These
tests are the standing version of that check.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from modelmri.server import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def _provenance(**over) -> dict:
    out = {
        "policy": "lerobot/smolvla_base",
        "dataset": "lerobot/pusht",
        "camera": "observation.images.top",
        "revision": "abc123def456",
        "episode": 5,
        "timestep": 12,
    }
    out.update(over)
    return out


def _vla(**over) -> dict:
    out = {
        "provenance": _provenance(),
        "frame": "data:image/png;base64,AAAA",
        "frame_size": [96, 96],
        "attention": [[[0.1, 0.2], [0.3, 0.4]]],
        "occlusion": {
            "baseline": "episode_mean",
            "grid": [2, 2],
            "stride": 1,
            "attention_agreement": -0.12,
            "compared_layer": 0,
            "compared_head": 3,
            "blocks": [
                {
                    "row": 0,
                    "col": 0,
                    "shift": 0.4,
                    "control_max": 0.1,
                    "clears_control": True,
                    "control_draws": 8,
                },
                {
                    "row": 0,
                    "col": 1,
                    "shift": 0.1,
                    "control_max": None,
                    "clears_control": None,
                    "control_draws": 0,
                },
            ],
        },
    }
    out.update(over)
    return out


def _mri(**over) -> bytes:
    from modelmri import session

    args = dict(
        model_id="smolvla",
        device="cpu",
        dtype="float32",
        n_params=1,
        tokens=["a"],
        prompt="a",
        generation="",
        attention={},
        n_layers=1,
        n_heads=1,
        vla=_vla(),
    )
    args.update(over)
    return session.build(**args)


def _open(client: TestClient, blob: bytes):
    return client.post("/api/session/open", content=blob)


def test_no_session_is_a_state_and_not_an_error(client):
    """`available: false` rather than a 404. Most sessions carry no robot
    finding, and a panel that got an error for the ordinary case would render
    "this measurement is broken" on every text session."""
    r = client.get("/api/vla/replay")
    assert r.status_code == 200
    assert r.json() == {"available": False}


def test_a_text_only_session_carries_no_robot_finding(client):
    """The other half of the same rule: a file WITHOUT the section is not a
    file whose section failed to load."""
    assert _open(client, _mri(vla=None)).status_code == 200
    assert client.get("/api/vla/replay").json() == {"available": False}


def test_a_shared_robot_finding_comes_back_whole(client):
    """THE BUG THIS FILE EXISTS FOR. Before the route, this answered nothing
    at all and the finding was unreadable by its recipient."""
    assert _open(client, _mri()).status_code == 200
    d = client.get("/api/vla/replay").json()
    assert d["available"] is True

    # All five things `session._vla` refuses the section without, because a
    # heat map without them is -- in that validator's words -- a picture of
    # nothing in particular.
    p = d["provenance"]
    assert p["policy"] == "lerobot/smolvla_base"
    assert p["dataset"] == "lerobot/pusht"
    assert p["camera"] == "observation.images.top"
    assert p["episode"] == 5
    assert p["timestep"] == 12

    assert d["frame_size"] == [96, 96]
    assert d["frame_downsampled"] is False
    assert len(d["occlusion"]["blocks"]) == 2


def test_an_uncontrolled_block_keeps_its_null(client):
    """`clears_control: None` is "nobody ran a control", which is NOT "it
    failed to clear one" and NOT `False`. Occlusion is out of distribution --
    covering anything moves the action -- so a block with no control behind it
    is a number rather than a finding, and the difference has to survive the
    trip."""
    assert _open(client, _mri()).status_code == 200
    blocks = client.get("/api/vla/replay").json()["occlusion"]["blocks"]
    cleared, uncontrolled = blocks[0], blocks[1]
    assert cleared["clears_control"] is True
    assert cleared["control_max"] == 0.1
    assert uncontrolled["clears_control"] is None
    # NOT 0.0, which would read as "a random occlusion moved the action not at
    # all" -- a measurement nobody made.
    assert uncontrolled["control_max"] is None


def test_an_uncompared_attention_map_keeps_its_null(client):
    """`compared_layer: None` is "not compared", and a reader has to be able
    to tell that from "layer 0"."""
    occ = _vla()["occlusion"]
    del occ["compared_layer"]
    del occ["compared_head"]
    assert _open(client, _mri(vla=_vla(occlusion=occ))).status_code == 200
    got = client.get("/api/vla/replay").json()["occlusion"]
    assert got["compared_layer"] is None
    assert got["compared_head"] is None


def test_a_negative_agreement_survives_with_its_sign(client):
    """Attention is not a cause, and the policy looking hardest where covering
    changes least is a real and common result. Dropping the sign -- or the
    number -- would hide the finding the two measurements exist to expose."""
    assert _open(client, _mri()).status_code == 200
    assert client.get("/api/vla/replay").json()["occlusion"][
        "attention_agreement"
    ] == pytest.approx(-0.12)


def test_nothing_is_recomputed_on_the_way_out(client):
    """Every number in the section was measured on somebody else's machine,
    against a policy and a dataset this process does not have. It is served
    exactly as `session._vla` validated it."""
    from modelmri import session

    blob = _mri()
    assert _open(client, blob).status_code == 200
    served = client.get("/api/vla/replay").json()
    served.pop("available")
    assert served == session.parse(blob).vla


def test_opening_a_second_file_replaces_the_first_finding(client):
    """A finding left behind would be read under the next file's name, which
    is the worst thing this panel could do."""
    assert _open(client, _mri()).status_code == 200
    assert client.get("/api/vla/replay").json()["available"] is True
    assert _open(client, _mri(vla=None)).status_code == 200
    assert client.get("/api/vla/replay").json() == {"available": False}
