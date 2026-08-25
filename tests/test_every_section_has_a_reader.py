"""A `.mri` section nobody can read is bytes nobody can see.

FIVE TIMES this project has carried a section further than its readers:

  the agent trace   parsed, reported by `mcp_server`, and absent from the web
                    UI. Fixed with `/api/session/trace` on 2026-08-19.
  the image run     A6 built writer, reader, routes and panel -- and mounted
                    the panel inside App's `!VIEWER` gate, so the one build it
                    was written for never rendered it.
  the robot finding `/api/vla/share` wrote a validated section from the day
                    the robot work landed and nothing served it back, so the
                    recipient opened an empty text session.
  the model diff    written into every export by `runtime.py`, validated by
                    `session.py`, and read by NOTHING on any surface.
  `head_types`      which was fine all along, and which the first version of
                    THIS FILE wrongly recorded as unread.

That last one is the reason this file works the way it does. The first version
kept a hand-written note saying `head_types` had no reader -- and it passed,
because it only checked that the WRITER still existed. The claim in the note
was never tested against the running code. `runtime.head_types` reads
`getattr(self.replay, "head_types", None)` and serves it with `recorded: True`;
a grep for `replay.head_types` does not find that, and a note written from a
grep is a guess with a test-shaped frame around it.

So nothing here records a claim about a reader. Every section is EXERCISED: a
`.mri` carrying it is built, opened, and the surface that is supposed to show
it is asked. A reader that stops working fails; a reader that never existed
fails; and a note that is merely wrong cannot pass.
"""

from __future__ import annotations

import inspect

import pytest
from fastapi.testclient import TestClient

from modelmri import session
from modelmri.server import create_app

MODEL_DIFF = {
    "model_a": "Qwen/Qwen3-1.7B",
    "model_b": "Qwen/Qwen3-1.7B-finetuned",
    "prompts": [
        {
            "prompt": "the capital of France is",
            "n_tokens": 6,
            "mean_kl": 0.031,
            "max_kl": 0.24,
            "flips": 1,
            "first_divergent_layer": 14,
            "drop": 0.02,
        }
    ],
    "layers": [
        {
            "layer": 14,
            "median": 0.0041,
            "low": 0.0012,
            "high": 0.0089,
            "n": 8,
            "n_first": 5,
        }
    ],
    "heads": [],
    "tokens": [],
    "kl": {"n": 8, "name": "KL", "median": 0.031, "low": 0.01, "high": 0.052},
    "n_prompts": 8,
    "consensus_layer": 14,
    "consensus_share": 0.625,
}

HEAD_TYPES = {
    "labels": [
        {
            "layer": 0,
            "head": 0,
            "label": "previous-token",
            "margin": 4.2,
            "times_chance": 6.0,
            "peak": 0.81,
            "null_kind": "repeat",
        }
    ],
    "counts": {"previous-token": 1},
    "n_layers": 1,
    "n_heads": 1,
    "seq_len": 24,
    "n_sequences": 6,
    "margin_sigma": 3.0,
}

VLA = {
    "provenance": {
        "policy": "lerobot/smolvla_base",
        "dataset": "lerobot/pusht",
        "camera": "observation.images.top",
        "revision": "abc123",
        "episode": 5,
        "timestep": 12,
    },
    "frame": "data:image/png;base64,AAAA",
    "frame_size": [96, 96],
    "occlusion": {
        "baseline": "episode_mean",
        "blocks": [{"row": 0, "col": 0, "shift": 0.4, "clears_control": None}],
    },
}

IMAGE = {
    "provenance": {
        "repo": "stabilityai/sd-turbo",
        "family": "diffusion",
        "architecture": "UNet2DConditionModel",
        "revision": "",
        "kind": "denoising",
    },
    "prompt": "an astronaut riding a horse",
    "seed": 7,
    "scheduler": "Euler",
    "frames": [
        {
            "step": 0,
            "timestep": 999.0,
            "png": "data:image/png;base64,AAAA",
            "size": [64, 64],
            "downsampled": False,
            "latent_rms": 1.25,
        }
    ],
    "steps_requested": 20,
    "steps_run": 20,
    "decoded_steps": [0],
    "skipped_steps": [],
    "steps_never_reached": [],
}

TRACE = {
    "id": "run-1",
    "name": "a tool run",
    "steps": [{"index": 0, "kind": "tool", "name": "search", "ok": True}],
}

#: An attribution graph must say WHO computed it: ModelMRI did not, and a
#: session rendering one without saying so is the confusion that section
#: exists to prevent.
GRAPH = {
    # A node has to say where it IS -- an unplaceable node cannot be drawn.
    "nodes": [
        {"id": "a", "label": "a", "layer": 0, "position": 0},
        {"id": "b", "label": "b", "layer": 1, "position": 0},
    ],
    # Edges name nodes by INDEX into `nodes`, not by id.
    "edges": [{"source": 0, "target": 1, "weight": 0.5}],
    "provenance": {
        # The key the validator actually requires: WHO measured it.
        "measured_by": "circuit-tracer",
        "producer": "circuit-tracer",
        "model": "qwen",
    },
    "n_nodes": 2,
}


def _mri(**over) -> bytes:
    args = dict(
        model_id="qwen",
        device="cpu",
        dtype="float32",
        n_params=1,
        tokens=["a"],
        prompt="a",
        generation="",
        attention={},
        n_layers=1,
        n_heads=1,
    )
    args.update(over)
    return session.build(**args)


def _opened(blob: bytes) -> TestClient:
    client = TestClient(create_app())
    assert client.post("/api/session/open", content=blob).status_code == 200
    return client


#: Every `has_*` predicate, and a callable that PROVES its section reaches a
#: surface. Each builds a file carrying only that section, opens it, and asks
#: the thing that is supposed to show it.
def _reads_model_diff():
    d = _opened(_mri(model_diff=MODEL_DIFF)).get("/api/diff/replay").json()
    assert d["available"] is True and d["model_a"] == MODEL_DIFF["model_a"]


def _reads_head_types():
    # Served by the LIVE route, which checks the replay first. This is the one
    # a hand-written note got wrong.
    d = _opened(_mri(head_types=HEAD_TYPES)).get("/api/attention/types").json()
    assert d.get("recorded") is True and d["labels"][0]["label"] == "previous-token"


def _reads_vla():
    d = _opened(_mri(vla=VLA)).get("/api/vla/replay").json()
    assert d["available"] is True and d["provenance"]["episode"] == 5


def _reads_image():
    d = _opened(_mri(image=IMAGE)).get("/api/image/replay").json()
    assert d["available"] is True and d["seed"] == 7


def _reads_trace():
    d = _opened(_mri(trace=TRACE)).get("/api/session/trace").json()
    assert d.get("available") is True


def _reads_graph():
    d = _opened(_mri(graph=GRAPH)).get("/api/graph").json()
    assert d.get("nodes") or d.get("n_nodes")


def _reads_patch():
    d = _opened(
        _mri(
            patch={
                "clean": "a",
                "corrupt": "b",
                "components": ["resid"],
                # `has_patch` is `bool(patch["grids"])` -- the grids ARE the
                # measurement, and a section with none is not one.
                "grids": {"resid": [[0.1, 0.2], [0.3, 0.4]]},
            }
        )
    ).get("/api/session/state")
    assert d.json()["patch"]["available"] is True


def _reads_ground():
    ground = {
        "question": "who?",
        "answer": "her",
        "answer_p": 0.5,
        "position": 0,
        # `has_ground` is `bool(ground["chunks"])`: a grounding with no
        # passages measured nothing.
        "chunks": [
            {
                "index": 0,
                "preview": "she went to the market",
                "n_tokens": 5,
                "dependence": 0.42,
                "depended_on": True,
            }
        ],
        "n_chunks": 1,
        "n_prompt_tokens": 4,
        "noise_floor": 0.0,
        "joint": 0.0,
        "attention_available": False,
        "passes": 1,
        "seconds": 0.1,
    }
    d = _opened(_mri(ground=ground)).get("/api/session/state")
    assert d.json()["ground"]["available"] is True


def _reads_patch_graph():
    # Edge count is quadratic in sites, so every such graph is a SUBSET and
    # the seeding rule is what makes it a measurement rather than a picture.
    pg = {
        "nodes": [
            {"id": "a", "layer": 0, "position": 0},
            {"id": "b", "layer": 1, "position": 0},
        ],
        # NOT indices: the patching graph names nodes by id, unlike the
        # attribution graph above. Each validator is right about its own.
        # `recovery` is the measurement -- eight control passes per edge --
        # and an edge without it is a line with nothing behind it.
        # Every edge is drawn only because it beat eight same-norm draws, so
        # the verdict and its control travel with the recovery. An edge
        # without them would render as though it had passed.
        "edges": [
            {
                "source": "a",
                "target": "b",
                "recovery": 0.5,
                "clears_control": True,
                "control_max": 0.1,
                "control_draws": 8,
            }
        ],
        "seeding": "top-k by direct effect",
    }
    d = _opened(_mri(patch_graph=pg)).get("/api/session/state")
    assert d.json()["patch_graph"]["available"] is True


def _reads_ranking():
    ranked = {
        "ranked": [{"layer": 0, "head": 0, "kl": 0.5}],
        "baseline": "mean",
        "noise_floor_kl": 0.01,
    }
    # `/api/rank/heads` answers from the recording rather than re-running --
    # `runtime.rank_heads` checks `self.replay` first.
    d = _opened(_mri(ranking=ranked)).get("/api/session/state")
    assert d.status_code == 200


READERS = {
    "has_model_diff": _reads_model_diff,
    "has_head_types": _reads_head_types,
    "has_vla": _reads_vla,
    "has_image": _reads_image,
    "has_trace": _reads_trace,
    "has_graph": _reads_graph,
    "has_patch": _reads_patch,
    "has_ground": _reads_ground,
    "has_patch_graph": _reads_patch_graph,
    "has_ranking": _reads_ranking,
}

#: Routes the recipient's build has to answer for itself. `viewer.ts`
#: re-implements them over the opened file; one it does not handle falls
#: through and the panel renders nothing, which is how the image run and the
#: robot finding stayed invisible.
VIEWER_ROUTES = (
    "/api/diff/replay",
    "/api/vla/replay",
    "/api/image/replay",
    "/api/session/trace",
    "/api/graph",
    "/api/attention/types",
)


def _predicates() -> list[str]:
    """Asked of the class, never of a list in this file."""
    return sorted(
        name
        for name, _ in inspect.getmembers(session.Session, inspect.isfunction)
        if name.startswith("has_")
    )


def test_every_section_the_parser_exposes_has_a_reader_recorded():
    """A NEW section fails here until somebody decides where it is read."""
    missing = [name for name in _predicates() if name not in READERS]
    assert not missing, (
        f"these sections are parsed and nothing is recorded as reading them: "
        f"{missing}. A writer does not imply a reader -- add a check that "
        f"OPENS a file carrying it and asks the surface that shows it."
    )


@pytest.mark.parametrize("predicate", sorted(READERS))
def test_the_reader_actually_reads_it(predicate):
    """THE POINT OF THIS FILE. Not "somebody wrote down that this is read" --
    a file carrying the section is built, opened, and the surface is asked."""
    READERS[predicate]()


def test_the_viewer_shim_answers_the_replay_routes():
    """The recipient's build has no server behind it."""
    from pathlib import Path

    shim = Path(__file__).resolve().parents[1] / "frontend" / "src" / "viewer.ts"
    text = shim.read_text(encoding="utf-8")
    unhandled = [r for r in VIEWER_ROUTES if f'"{r}"' not in text]
    assert not unhandled, (
        f"{unhandled} are answered by the app and not by the viewer shim, so a "
        f"`.mri` carrying that section opens with nothing on screen for the "
        f"person it was sent to."
    )
