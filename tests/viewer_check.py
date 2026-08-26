"""The browser viewer must produce exactly the numbers the tool produces.

    python scripts/build_frontend.py --viewer
    python tests/viewer_check.py

`frontend/src/viewer.ts` re-implements `modelmri/session.py` in TypeScript:
gunzip, base64, uint8 dequantisation, rounding. Two implementations of one
format drift — and a viewer that renders a *slightly* different matrix than
the tool is worse than no viewer, because nothing on screen would say so.

So this parses the same file both ways and compares every cell. It serves the
bundled viewer over http (module scripts do not load from file://) and drives
it with Playwright, reading the numbers back out of the running page rather
than trusting the source to be equivalent.

Needs: playwright (uv run playwright install chromium), and a built viewer.
"""

from __future__ import annotations

import asyncio
import functools
import http.server
import socketserver
import sys
import threading
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIEWER = ROOT / "modelmri" / "static" / "viewer"
FIXTURE = ROOT / "tests" / "fixtures" / "parity.mri"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def build_fixture() -> bytes:
    """A session with structure worth disagreeing about: an attention sink,
    a causal mask, and values across the whole quantisation range."""
    sys.path.insert(0, str(ROOT))
    from modelmri import session

    n, layers, heads = 24, 4, 3
    matrices = {}
    for layer in range(layers):
        for head in range(heads):
            rows = []
            for r in range(n):
                raw = [
                    ((layer * 7 + head * 13 + r * 3 + c * 11) % 23) + 0.5
                    for c in range(r + 1)
                ]
                raw[0] += 40.0  # the sink every real head has
                raw += [0.0] * (n - r - 1)
                total = sum(raw)
                rows.append([v / total for v in raw])
            matrices[(layer, head)] = rows
    return session.build(
        model_id="parity/fixture",
        device="cpu",
        dtype="float32",
        n_params=1234,
        tokens=[f"t{i}" for i in range(n)],
        prompt="parity",
        generation="check",
        attention=matrices,
        n_layers=layers,
        n_heads=heads,
        note="every cell must match",
    )


def build_image_fixture() -> bytes:
    """A `.mri` carrying an image run, hostile in the four ways a stranger's
    can be — none of which this writer can produce.

    `session.build` validates through `session._image`, so the file is built
    legitimately and then EDITED: the point is a file that never went through
    this reader at all, which is what the viewer build is handed. It gets the
    raw section straight out of the gzip with only the provenance checked.
    """
    import base64
    import gzip
    import json

    sys.path.insert(0, str(ROOT))
    from modelmri import session

    png = (
        "data:image/png;base64,"
        + base64.b64encode(
            bytes.fromhex(
                "89504e470d0a1a0a0000000d494844520000000100000001080600000"
                "01f15c4890000000a49444154789c6360000002000100ffff0300000600"
                "0557bfabd40000000049454e44ae426082"
            )
        ).decode()
    )

    blob = session.build(
        model_id="stabilityai/sd-turbo",
        device="cpu",
        dtype="float32",
        n_params=None,
        tokens=[],
        prompt="",
        generation="",
        attention={},
        n_layers=0,
        n_heads=0,
        note="a file from a stranger",
        scope="one denoising run",
        image={
            "provenance": {
                "repo": "stabilityai/sd-turbo",
                "family": "diffusion",
                "architecture": "UNet2DConditionModel",
                "revision": "",
                "kind": "denoising",
            },
            "prompt": "an astronaut riding a horse",
            "seed": None,
            "scheduler": "Euler",
            "frames": [
                {
                    "step": 0,
                    "timestep": 999.0,
                    "png": png,
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
            "attention": {
                "tokens": ["an", "astronaut", "<pad>", "<pad>"],
                "steps": [
                    {
                        "step": 0,
                        "timestep": 999.0,
                        "per_token": [0.4, 0.3, 0.2, 0.1],
                        "blocks": 16,
                    }
                ],
                "padding_from": 2,
                "conditioning_width": 77,
                "columns_unlabelled": 0,
                "steps_requested": 20,
                "steps_measured": 1,
                "resolutions": [16],
                "means": "one step of twenty",
            },
            "means": "1 decoded frame of a 20-step run.",
        },
    )

    doc = json.loads(gzip.decompress(blob).decode("utf-8"))
    img = doc["image"]
    # 1. the boundary nobody measured -- absent, which is not 0
    img["attention"].pop("padding_from", None)
    # 2. the run length nobody stated -- absent, which is not a 0-step run
    img.pop("steps_requested", None)
    img.pop("steps_run", None)
    # 3. a frame that is a LINK. Opening the file must not tell whoever wrote
    #    it that you did.
    img["frames"].append(
        {
            "step": 1,
            "timestep": 500.0,
            "png": "https://beacon.invalid/1x1.png",
            "size": [64, 64],
            "downsampled": False,
            "latent_rms": None,
        }
    )
    # 4. arrays that are not there at all -- a TypeError with no error
    #    boundary above it is a white page where the recording used to be.
    for gone in ("skipped_steps", "steps_never_reached", "decoded_steps"):
        img.pop(gone, None)
    return gzip.compress(json.dumps(doc, separators=(",", ":")).encode(), 6)


def build_robot_fixture() -> bytes:
    """A `.mri` carrying a robot finding, hostile in the ways a stranger's can
    be — built legitimately and then EDITED, because the point is a file that
    never passed through `session._vla`.

    `/api/vla/share` wrote this section from the day the robot work landed and
    nothing served it back, so the recipient opened an empty text session. The
    reader was there the whole time; only the route and the panel were not.
    """
    import gzip
    import json

    sys.path.insert(0, str(ROOT))
    from modelmri import session

    blob = session.build(
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
        note="a robot finding from a stranger",
        vla={
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
                "grid": [2, 2],
                "attention_agreement": -0.12,
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
                    # A VERDICT WITH NOTHING BEHIND IT. The reader validates
                    # `clears_control` and the two control numbers
                    # independently -- a null `control_max` survives on
                    # purpose, because an uncontrolled block genuinely has
                    # none -- so this shape passes it. The panel rendered it
                    # as "0.0000 the best of 0 random occlusions managed": a
                    # measured-looking zero over an absence.
                    #
                    # The LARGEST shift, deliberately, so it is the block the
                    # answer slot speaks about.
                    {
                        "row": 1,
                        "col": 0,
                        "shift": 0.9,
                        "control_max": None,
                        "clears_control": True,
                        "control_draws": None,
                    },
                ],
            },
        },
    )

    doc = json.loads(gzip.decompress(blob).decode("utf-8"))
    vla = doc["vla"]
    # A FRAME THAT IS A LINK. `session._vla` refuses one in as many words;
    # the viewer shim serves the section raw, so only the panel stands between
    # a hostile file and the recipient announcing that they opened it.
    vla["frame"] = "https://beacon.invalid/frame.png"
    return gzip.compress(json.dumps(doc, separators=(",", ":")).encode(), 6)


def build_diff_fixture() -> bytes:
    """A `.mri` carrying a comparison of two models and head labels.

    Neither needs editing to be hostile: the point here is the MOUNT. A route
    test cannot see a panel that is never rendered, which is how the image run
    and the robot finding both stayed invisible in the build they were written
    for.
    """
    sys.path.insert(0, str(ROOT))
    from modelmri import session

    return session.build(
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
        note="a comparison somebody ran",
        model_diff={
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
            # A ROW THE READER ACCEPTS AND THE PANEL HAD TO GUESS AT.
            # `_rows` keeps `None` for any field, justified by a comment about
            # `first_divergent_layer` only -- so a spread with no median, no
            # bounds and no count is a legal row, and `?? 0` printed it as a
            # measured zero-width spread over zero prompts.
            "layers": [
                {"layer": 0, "median": None, "low": None, "high": None, "n": None}
            ],
            "heads": [],
            "tokens": [],
            # A TIGHT spread, so the panel must print the amount rather than
            # refuse to: low/high within half the median.
            "kl": {"n": 8, "name": "KL", "median": 0.030, "low": 0.028, "high": 0.032},
            "n_prompts": 8,
            # `null` is a RESULT -- the cosine never fell -- and the panel has
            # to say so rather than print a layer.
            "consensus_layer": None,
        },
        head_types={
            "labels": [
                {
                    "layer": 0,
                    "head": 0,
                    "label": "previous-token",
                    "margin": 4.2,
                    "null_kind": "repeat",
                }
            ],
            "counts": {"previous-token": 1},
            "n_layers": 1,
            "n_heads": 1,
            "seq_len": 24,
            "n_sequences": 6,
            "margin_sigma": 3.0,
        },
    )


def build_causal_fixture() -> bytes:
    """A `.mri` carrying all three CAUSAL sections at once — the patching
    grid, the graph walked back out of it, and the grounding result.

    Nothing here is edited after the build, and that is the point. This is
    what `session.build` writes today, through `session._patch`,
    `_patch_graph` and `_ground`, which are the same readers the app opens a
    forwarded file with. All three sections have been written since the day
    each landed, and the viewer's shim answered none of their three routes —
    so every one of them fell through to its last line, "install ModelMRI to
    point these instruments at your own models". That sentence is false about
    a file whose bytes already carry the measurement, and it is the one
    refusal a reader cannot act on: there is nothing to install their way out
    of.

    ONE file rather than three, because the three arrive together and are read
    together. The graph is seeded from the grid's own flagged sites, so a
    viewer that draws the graph and refuses the grid shows a conclusion with
    its evidence withheld — and a reader who gets neither cannot tell a
    measured circuit from a picture of one.
    """
    sys.path.insert(0, str(ROOT))
    from modelmri import session

    return session.build(
        model_id="Qwen/Qwen3-1.7B",
        device="cpu",
        dtype="float32",
        n_params=1,
        tokens=["a"],
        prompt="a",
        generation="",
        attention={},
        n_layers=2,
        n_heads=1,
        note="a causal session somebody sent",
        patch={
            # `resid` is the tab the panel opens on, so this is the grid the
            # recipient sees first. The negative cell is deliberate: recovery
            # is SIGNED, and a site that pushed the answer further from the
            # clean run must not paint like a site that did nothing.
            "grids": {
                "resid": [[0.02, 0.87], [0.41, -0.31]],
                "attn": [[0.00, 0.23], [0.09, 0.02]],
            },
            # WHAT MAKES THE GRID READABLE. Its columns are token positions,
            # and its numbers are a fraction of the gap between two answers.
            # The section carried neither, so the panel mounted and then said
            # the server had answered in a shape it did not know -- to a
            # reader with no server to restart.
            "components": ["resid", "attn"],
            "tokens": {
                "clean": ["The", " Eiffel"],
                "corrupt": ["The", " Colosseum"],
            },
            "answers": {
                "clean": {"text": " Paris", "p": 0.71},
                "corrupt": {"text": " Rome", "p": 0.66},
            },
            "sites": [
                {
                    "layer": 1,
                    "position": 1,
                    "component": "resid",
                    "recovery": 0.87,
                    "control_max": 0.11,
                    "control_draws": 8,
                    "shifted_position": 0.19,
                    "clears_control": True,
                    "clears_position": True,
                }
            ],
            # The component this architecture never exposed. `patch.py` keeps
            # the refusal that named it so two grids cannot arrive looking
            # like the whole answer, and this note is the only place the third
            # one is accounted for at all.
            "notes": [
                "mlp: this architecture exposes no separate mlp submodule, so "
                "that grid is absent rather than empty"
            ],
            "clean": "The Eiffel Tower is located in the city of",
            "corrupt": "The Colosseum is located in the city of",
        },
        patch_graph={
            "nodes": [
                {
                    "id": "resid:1:1",
                    "layer": 1,
                    "head": None,
                    "position": 1,
                    "role": "seed",
                    "depth": 0,
                },
                {
                    "id": "attn:0:1:h0",
                    "layer": 0,
                    "head": 0,
                    "position": 1,
                    "role": "sender",
                    "depth": 1,
                },
                {
                    "id": "mlp:0:0",
                    "layer": 0,
                    "head": None,
                    "position": 0,
                    "role": "sender",
                    "depth": 1,
                },
            ],
            "edges": [
                {
                    "source": "attn:0:1:h0",
                    "target": "resid:1:1",
                    "recovery": 0.62,
                    "control_max": 0.14,
                    "controls": [0.05, 0.09, 0.14, 0.02, 0.11, 0.07, 0.13, 0.06],
                    "control_draws": 8,
                    "clears_control": True,
                    "clears_position": True,
                },
                # TESTED AND FAILED, which is a finding rather than an absence
                # — `patch_graph` returns it marked rather than dropping it,
                # and `clears_position` is null because that second control was
                # never run for this edge. Null travels; the reader is three-
                # valued here on purpose, because coercing it to False would
                # turn "not run" into "run, and failed".
                {
                    "source": "mlp:0:0",
                    "target": "resid:1:1",
                    "recovery": 0.08,
                    "control_max": 0.21,
                    "controls": [0.03, 0.21, 0.10, 0.07, 0.19, 0.05, 0.12, 0.08],
                    "control_draws": 8,
                    "clears_control": False,
                    "clears_position": None,
                },
            ],
            # NOT a footnote, and `_patch_graph` refuses a graph without one.
            # Edge count is quadratic in sites, so every graph here is a
            # subset by construction; one whose rule for choosing edges has
            # been stripped is a picture rather than a measurement.
            "seeding": "the 1 residual site that beat its controls, "
            "expanded one level back",
            "means": "A patching graph, built from nothing but the model this "
            "file was recorded on — not a transcoder attribution graph.",
            # DELIBERATELY DIFFERENT from the patch section's pair above.
            # These were identical, which is exactly why the panel being
            # handed the PATCH section's prompts was invisible: a file can
            # carry a graph and a trace of two different prompts, and the
            # graph was shown above the pair it was not measured on.
            "clean": "The Louvre is located in the city of",
            "corrupt": "The Prado is located in the city of",
            "depth": 1,
            "n_scored": 137,
            "n_pruned": 12,
            # HOW MUCH OF THE NETWORK THIS GRAPH STOPPED LOOKING AT, and what
            # it cost. The reader dropped all three, so a recorded graph drew
            # no pruning chips -- one that pruned most of the network read as
            # the whole circuit -- and printed its two-minute run as "0s".
            "n_weak": 41,
            "n_untested": 9,
            "seconds": 119.4,
            "passes": 1096,
            "prune_threshold": 0.02,
            "prune_from": "the strongest edge at this level",
            # The receiver whose senders were never expanded. Dropping it
            # would turn "we stopped asking here" into "nothing wrote this".
            "frontier": ["mlp:0:0"],
        },
        ground={
            "question": "Question: which alloy carried the load?\nAnswer:",
            "answer": " steel",
            "answer_p": 0.7314,
            "position": 41,
            "chunks": [
                {
                    "index": 0,
                    "preview": "The 1968 deck was rebuilt in weathering "
                    "steel throughout.",
                    "n_tokens": 34,
                    "dependence": 0.4271,
                    "attention": None,
                    "depended_on": True,
                    "looked_not_used": None,
                },
                {
                    "index": 1,
                    "preview": "The original towers were wrought iron, "
                    "riveted on site.",
                    "n_tokens": 22,
                    "dependence": 0.0044,
                    "attention": None,
                    "depended_on": False,
                    "looked_not_used": None,
                },
                {
                    "index": 2,
                    "preview": "Unrelated: the ferry runs hourly at weekends.",
                    "n_tokens": 18,
                    "dependence": 0.0009,
                    "attention": None,
                    "depended_on": False,
                    "looked_not_used": None,
                },
            ],
            "n_chunks": 3,
            "n_prompt_tokens": 96,
            # A REAL floor, which is what lets `depended_on` above be a claim
            # rather than an assertion: `_ground` refuses a passage marked as
            # depended-on in a file that does not say what it cleared.
            "noise_floor": 0.0121,
            "joint": 0.5108,
            # THE UNMEASURED HALF, and the reason this fixture leaves every
            # `attention` null. `ground.measure` reports None for every
            # passage on a model whose attention implementation never built
            # the score matrix, and sets `looked_not_used` to None with it —
            # all of them or none of them, never a mix. A share of 0.0 there
            # would read as "nothing looked at this passage", which is a
            # finding, and nobody took that reading.
            "attention_share": None,
            "attention_available": False,
            "attention_note": "this model fuses attention and never returned "
            "a score matrix, so the looked-at half was not taken",
            "floor_degenerate": False,
            "ungrounded": False,
            "passes": 7,
            "seconds": 3.482,
        },
    )


def build_lens_fixture() -> bytes:
    """A `.mri` carrying a LOGIT LENS — the oldest section in the format, and
    the one nothing could display.

    `lens` has been a field on `Session` since the format existed,
    `session._lens` validates it, and `runtime.logit_lens` has an explicit
    replay branch whose docstring says "a recording that carries a lens can
    serve it". Two locks kept that promise from ever being kept: `LensPanel`
    is mounted from inside `FeaturesPanel`, which is `!replay` because it also
    holds live-model controls, and `viewer.ts` had no `/api/lens` handler at
    all — so the route fell through to "install ModelMRI to point these
    instruments at your own models", which is false about a file whose bytes
    already carry the trajectory.

    Written through `session.build`, unedited, so this is exactly what the
    tool writes today.

    LAYER 2 CARRIES NO ENTROPY, on purpose. `_lens` copies `entropy` only
    when the file has a finite one, so that row is what a real file produces
    — and the panel divided by it to size a bar and called `.toFixed` on it,
    which is a `NaN%` width followed by a TypeError with no error boundary
    above it. The reader's viewer went white. Layer 3 carries an entropy of
    exactly 0.0 beside it, which IS a reading — the model is certain there —
    so the two rows on one screen are the project's rule made visible:
    unmeasured is "—", and zero is 0.00.
    """
    sys.path.insert(0, str(ROOT))
    from modelmri import session

    return session.build(
        model_id="Qwen/Qwen3-1.7B",
        device="cpu",
        dtype="float32",
        n_params=1,
        tokens=["a"],
        prompt="The Eiffel Tower is located in the city of",
        generation=" Paris",
        attention={},
        n_layers=3,
        n_heads=1,
        note="a lens somebody sent",
        lens=[
            {
                "layer": 0,
                "tokens": [" the", " a"],
                "probs": [0.0413, 0.0209],
                "entropy": 3.2109,
                "kl_to_final": 7.4412,
            },
            {
                "layer": 1,
                "tokens": [" France", " Paris"],
                "probs": [0.1806, 0.1274],
                "entropy": 1.4,
                "kl_to_final": 2.115,
            },
            # THE ROW THE READER HAS TO SURVIVE. No `entropy` and no
            # `kl_to_final`: the sender's model reported neither for this
            # depth, and `_lens` carries the row rather than dropping it,
            # because a trajectory with a hole in it is still the trajectory.
            {
                "layer": 2,
                "tokens": [" Paris", " Lyon"],
                "probs": [0.6021, 0.0884],
            },
            {
                "layer": 3,
                "tokens": [" Paris", " France"],
                "probs": [0.9137, 0.0311],
                # A MEASURED ZERO. The last layer is the model's own answer,
                # so it disagrees with itself by nothing at all — and this
                # must print as 0.00 while the row above prints "—".
                "entropy": 0.0,
                "kl_to_final": 0.0,
            },
        ],
        lens_info={
            "final": " Paris",
            "settled_at": 2,
            "n_layers": 3,
            "reliability": {
                "note": "read on the sender's machine, not here",
            },
        },
    )


def build_ranking_fixture() -> bytes:
    """A `.mri` carrying a HEAD RANKING and the head labels beside it — the
    tool's headline measurement, and the one a recipient could not read.

    Both sections have been written by `session.build` and validated by
    `session._ranking` / `session._head_types` since each landed, and both
    have working replay branches: `runtime.ablate_heads` returns the recorded
    ranking with `recorded: True`, `runtime.head_types` does the same for the
    labels, and `viewer.ts` already answered `/api/attention/types` with a
    refusal written for recipients. Nothing could ask for either.
    `AttentionPanel` gated the Rank-heads button on `!replay` — true of
    MEASURING a ranking, false of SHOWING one already in the file — and the
    only caller of the labels sat inside `{ranked && …}`, so the labels were
    locked behind a button a recording could never press.

    Written through `session.build`, unedited, so this is exactly what the
    tool writes today.

    THE ATTENTION CUBE IS NOT DECORATION. `AttentionPanel` returns its
    "this session carries no attention maps" stub while `layers === 0`, so a
    file with a ranking and no slices never reaches the controls at all — and
    a ranking is measured against a generation whose attention you were
    looking at, which is what `runtime.share` captures.

    THREE ROWS, AND ONE OF THEM IS BELOW THE FLOOR. `noise_floor_kl` travels
    with the scores because anything at or below it is arithmetic rather than
    the model; a reader who gets the rows without it cannot tell the
    difference, and every row would read as a finding.

    `elapsed_s` IS A FLOAT ON PURPOSE. `ablate.py` writes
    `round(seconds, 2)`, and `session._ranking` copies `elapsed_s` only when
    it is an `int` — so this is what a real recorded ranking looks like on
    arrival: no duration at all. The panel has to SAY that rather than print
    "N forward passes · s".
    """
    sys.path.insert(0, str(ROOT))
    from modelmri import session

    n, layers, heads = 6, 4, 3
    matrices = {}
    for layer in range(layers):
        for head in range(heads):
            rows = []
            for r in range(n):
                raw = [
                    ((layer * 5 + head * 7 + r * 3 + c) % 11) + 0.5
                    for c in range(r + 1)
                ]
                raw[0] += 12.0  # the sink every real head has
                raw += [0.0] * (n - r - 1)
                total = sum(raw)
                rows.append([v / total for v in raw])
            matrices[(layer, head)] = rows

    return session.build(
        model_id="Qwen/Qwen3-1.7B",
        device="cpu",
        dtype="float32",
        n_params=1,
        tokens=[f"t{i}" for i in range(n)],
        prompt="The Eiffel Tower is located in the city of",
        generation=" Paris",
        attention=matrices,
        n_layers=layers,
        n_heads=heads,
        note="a ranking somebody sent",
        ranking={
            # Descending by KL, which is the order `ablate.rank_heads` writes
            # and the order the panel trusts: it opens `ranked[0]`, so a file
            # sorted any other way would take the reader to the wrong head.
            "ranked": [
                {
                    "layer": 1,
                    "head": 2,
                    "kl": 0.4173,
                    "p_top_before": 0.7104,
                    "p_top_after": 0.2210,
                    "flips_top": True,
                },
                {
                    "layer": 1,
                    "head": 0,
                    "kl": 0.0231,
                    "p_top_before": 0.7104,
                    "p_top_after": 0.6902,
                    "flips_top": False,
                },
                # AT OR BELOW THE FLOOR, which is a reading and not a gap: the
                # row must render as "below the noise floor" rather than as a
                # small finding.
                {
                    "layer": 1,
                    "head": 1,
                    "kl": 0.0004,
                    "p_top_before": 0.7104,
                    "p_top_after": 0.7101,
                    "flips_top": False,
                },
            ],
            # `_ranking` REFUSES a ranking without this: the three baselines
            # agree only weakly, so rows that do not name theirs cannot be
            # compared against anything.
            "baseline": "zero",
            "noise_floor_kl": 0.0008,
            "target_token": " Paris",
            "means": "Each row says only how far the answer moves when that "
            "one head is removed. They do not add up.",
            "position": 5,
            "layer": 1,
            "passes": 5,
            "elapsed_s": 4.21,
        },
        head_types={
            "labels": [
                {
                    "layer": 1,
                    "head": 2,
                    "label": "previous-token",
                    "margin": 4.2,
                    "times_chance": 3.1,
                    "peak": 0.44,
                    "null_kind": "repeat",
                },
                # NO TYPE DETECTED, which is the finding for most heads.
                # `_head_types` keeps `None` rather than coercing it to "" —
                # an unlabelled head must not look like one whose label went
                # missing — and the panel must draw no chip for it.
                {"layer": 1, "head": 0, "label": None},
            ],
            "counts": {"previous-token": 1},
            "n_layers": layers,
            "n_heads": heads,
            "seq_len": 24,
            "n_sequences": 6,
            "margin_sigma": 3.0,
            "means": "A behavioural label from random repeated tokens. It "
            "does NOT explain the KL beside it.",
        },
    )


async def ranking_side(port: int) -> dict:
    """Open that file in the real viewer and read what it served and drew.

    Same two halves as `lens_side`, because it is the same failure two
    sections along: a route with no button is data nobody sees, and a button
    with no route is a control that can only refuse.

    Both buttons are pressed, in the order a reader meets them: the labels
    live inside the ranking block, so until the ranking is on screen there is
    no labels button to find. What each press actually clicked is returned —
    a probe that quietly found no button reads exactly like a panel that drew
    nothing, and those are different failures.
    """
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        # `frontend/src` has no error boundary, so ONE TypeError anywhere
        # unmounts the whole page — the recipient's screen goes white with
        # nothing on it saying why. "The panel did not draw" and "the page
        # died drawing it" have to be distinguishable on the failure line.
        crashed: list[str] = []
        page.on(
            "pageerror",
            lambda err: crashed.append(str(err).splitlines()[0][:160]),
        )
        await page.goto(f"http://127.0.0.1:{port}/", wait_until="networkidle")
        got = await page.evaluate(
            """async () => {
              const blob = await (await fetch('./ranking.mri')).blob();
              const file = new File([blob], 'ranking.mri');
              const input = document.querySelector('input[type=file][accept=".mri"]');
              const dt = new DataTransfer(); dt.items.add(file);
              input.files = dt.files;
              input.dispatchEvent(new Event('change', {bubbles: true}));
              await new Promise(r => setTimeout(r, 1500));

              const state = await (await fetch('/api/session/state')).json();
              // The route exactly as `api.ts` calls it, query string and all.
              const r = await fetch(
                '/api/attention/ablate?layer=1&baseline=zero&scope=layer');
              // The BODY as text: the thing under test is whether the
              // recording came back or the refusal did, and both parse.
              const body = await r.text();
              let served = {};
              try { served = JSON.parse(body); } catch (e) { served = {}; }

              // The panel is re-queried inside the helper on every press:
              // React delegates events at the root, and a button detached by
              // an earlier render is a button whose click goes nowhere.
              const press = async (starts) => {
                const panel = document.querySelector('.panel.attn');
                if (!panel) return '';
                const button = [...panel.querySelectorAll('button')]
                  .find(b => (b.textContent || '').startsWith(starts));
                if (!button) return '';
                const label = button.textContent.trim();
                button.click();
                await new Promise(r => setTimeout(r, 900));
                return label;
              };
              const askedRanking = await press('Show the recorded ranking');
              const askedLabels = await press('Show the recorded head labels');
              await new Promise(r => setTimeout(r, 500));

              const panel = document.querySelector('.panel.attn');
              const list = panel ? panel.querySelector('.ranking-list') : null;
              return {
                available: {
                  ranking: state && state.ranking
                    ? state.ranking.available === true : false,
                  types: state && state.head_types
                    ? state.head_types.available === true : false,
                  target: state && state.ranking
                    ? state.ranking.target_token : undefined,
                  rows: state && state.ranking ? state.ranking.n_heads : -1,
                },
                status: r.status,
                body,
                // The replay branch's shape, not a shape invented here: the
                // recorded section spread whole with `recorded` last, exactly
                // as `runtime.ablate_heads` returns `{**recorded, ...}`.
                shape: {
                  ranked: Array.isArray(served.ranked) ? served.ranked.length : -1,
                  recorded: served.recorded === true,
                  baseline: served.baseline,
                  target_token: served.target_token,
                },
                mounted: !!panel,
                askedRanking,
                askedLabels,
                // THE COLOUR, read as RENDERED, because the text alone
                // cannot catch the way this breaks. The chip's class is the
                // label with everything but [a-z] stripped -- `t-` then
                // `previoustoken` -- and the four per-type rules in
                // styles.css were written WITHOUT that hyphen, so not one of
                // them could ever match. Every type drew in the inherited
                // ink: four colours that existed in the stylesheet and had
                // never once applied, with nothing on screen saying so and
                // the label still reading correctly the whole time.
                //
                // The row the chip sits in is the reference rather than a
                // colour written down here. A dead selector leaves the chip
                // at EXACTLY its parent's colour, and comparing the two says
                // "the rule matched" without pinning this to whatever
                // --sem-base resolves to in the current theme.
                chip: (() => {
                  const el = list ? list.querySelector('.headtype') : null;
                  if (!el) return null;
                  return {
                    cls: el.className,
                    color: getComputedStyle(el).color,
                    inherited: getComputedStyle(el.parentElement).color,
                  };
                })(),
                // The LIST on its own, not the whole panel: the head dropdown
                // carries the same KL, so "0.417 is somewhere in the panel"
                // and "the ranked list drew it" are different questions.
                list: list ? list.innerText : '',
                text: panel ? panel.innerText : '',
              };
            }"""
        )
        await browser.close()
        got["crashed"] = crashed
        return got


async def lens_side(port: int) -> dict:
    """Open that file in the real viewer and read what it served and drew.

    Same two halves as `causal_side`, because it is the same failure one
    section along: a route with no panel is data nobody sees, and a panel with
    no route is a heading over a refusal.

    The `pageerror` listener is the point of the third assertion. The row with
    no entropy used to throw inside `rows.map`, which unmounts the whole React
    tree — the reader's screen goes white with nothing on it saying why — so
    "the panel did not mount" and "the page died drawing it" have to be
    distinguishable on the failure line.
    """
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        crashed: list[str] = []
        page.on(
            "pageerror",
            lambda err: crashed.append(str(err).splitlines()[0][:160]),
        )
        await page.goto(f"http://127.0.0.1:{port}/", wait_until="networkidle")
        got = await page.evaluate(
            """async () => {
              const blob = await (await fetch('./lens.mri')).blob();
              const file = new File([blob], 'lens.mri');
              const input = document.querySelector('input[type=file][accept=".mri"]');
              const dt = new DataTransfer(); dt.items.add(file);
              input.files = dt.files;
              input.dispatchEvent(new Event('change', {bubbles: true}));
              await new Promise(r => setTimeout(r, 1500));

              const state = await (await fetch('/api/session/state')).json();
              // The route exactly as `api.ts` calls it, query string and all.
              const r = await fetch('/api/lens?top_k=4&kind=plain');
              // The BODY as text: the thing under test is whether the
              // recording came back or the refusal did, and both parse.
              const body = await r.text();
              let served = {};
              try { served = JSON.parse(body); } catch (e) { served = {}; }

              // Re-queried after the click for `causal_side`'s reason: React
              // delegates at the root, and a node detached by a re-render is
              // one whose click goes nowhere.
              let asked = '';
              const panel0 = document.querySelector('.panel.lensp');
              if (panel0) {
                const button = [...panel0.querySelectorAll('button')]
                  .find(b => (b.textContent || '').startsWith('Show the recorded'));
                if (button) {
                  asked = button.textContent.trim();
                  button.click();
                  await new Promise(r => setTimeout(r, 900));
                }
              }

              const panel = document.querySelector('.panel.lensp');
              // Per ROW, not as one blob of text. "the entropy column reads
              // 0.00 somewhere" and "THIS row reads 0.00" are different
              // questions, and only the second one is the bug.
              const rows = panel
                ? [...panel.querySelectorAll('.lens-row')]
                    .filter(el => !el.classList.contains('head'))
                    .map(el => ({
                      name: (el.querySelector('.l-name') || {}).innerText || '',
                      entropy: (el.querySelector('.lens-h') || {}).innerText || '',
                      lost: (el.querySelector('.lens-kl') || {}).innerText || '',
                    }))
                : [];
              // The raw probabilities, which the table carries as a tooltip so
              // a rounded percentage is never the only copy on screen.
              const titles = panel
                ? [...panel.querySelectorAll('.lens-tok')].map(el => el.title)
                : [];
              return {
                available: state && state.lens ? state.lens.available === true : false,
                status: r.status,
                body,
                // The replay branch's shape, not a shape invented here:
                // `layers` plus the recorded scalars spread beside it.
                shape: {
                  layers: Array.isArray(served.layers) ? served.layers.length : -1,
                  recorded: served.recorded === true,
                  final: served.final,
                  settled_at: served.settled_at,
                },
                mounted: !!panel,
                asked,
                rows,
                titles,
                text: panel ? panel.innerText : '',
              };
            }"""
        )
        await browser.close()
        got["crashed"] = crashed
        return got


async def diff_side(port: int) -> dict:
    """Open it in the real viewer and read what landed on screen."""
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(f"http://127.0.0.1:{port}/", wait_until="networkidle")
        got = await page.evaluate(
            """async () => {
              const blob = await (await fetch('./diff.mri')).blob();
              const file = new File([blob], 'diff.mri');
              const input = document.querySelector('input[type=file][accept=".mri"]');
              const dt = new DataTransfer(); dt.items.add(file);
              input.files = dt.files;
              input.dispatchEvent(new Event('change', {bubbles: true}));
              await new Promise(r => setTimeout(r, 1500));

              const served = await (await fetch('/api/diff/replay')).json();
              const types = await (await fetch('/api/attention/types')).json();
              const panel = [...document.querySelectorAll('.panel')]
                .find(p => p.innerText.includes('TWO MODELS COMPARED'));
              return {
                served: served.available === true,
                mounted: !!panel,
                text: panel ? panel.innerText : '',
                labelled: types.recorded === true,
              };
            }"""
        )
        await browser.close()
        return got


async def causal_side(port: int) -> dict:
    """Open that file in the real viewer and read what it served and drew.

    Three routes and three panels in one pass, because it is one failure.
    `viewerFetch` handled none of `/api/patch`, `/api/patch/graph` or
    `/api/ground`, so all three fell through to the shim's "install ModelMRI";
    and `Playground` returned the attention panel alone under `if (VIEWER)`,
    so the three panels that read a recorded causal result were never mounted
    in the ONE build that exists to show recordings. Either half alone leaves
    the reader with nothing: a route with no panel is data nobody sees, and a
    panel with no route is a heading over a refusal.

    None of these panels draws a recording until the reader asks for it —
    each offers "Show the recorded …" where the button that re-runs the
    measurement would be. So this clicks that button inside each panel rather
    than waiting for something that never arrives on its own, and returns what
    it clicked: a probe that quietly found no button reads exactly like a
    panel that rendered nothing, and those are different failures.
    """
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        # `frontend/src` has no error boundary, so ONE TypeError in one panel
        # unmounts the whole page. Without this the three mount checks below
        # would fail together with nothing on the line saying why.
        crashed: list[str] = []
        page.on(
            "pageerror",
            lambda err: crashed.append(str(err).splitlines()[0][:160]),
        )
        await page.goto(f"http://127.0.0.1:{port}/", wait_until="networkidle")
        got = await page.evaluate(
            """async () => {
              const post = async (path) => {
                const r = await fetch(path, {
                  method: 'POST',
                  headers: {'Content-Type': 'application/json'},
                  body: '{}',
                });
                // The BODY as text, not as JSON: the thing under test is
                // whether the recording came back or the refusal did, and
                // both parse.
                return {status: r.status, body: await r.text()};
              };
              const blob = await (await fetch('./causal.mri')).blob();
              const file = new File([blob], 'causal.mri');
              const input = document.querySelector('input[type=file][accept=".mri"]');
              const dt = new DataTransfer(); dt.items.add(file);
              input.files = dt.files;
              input.dispatchEvent(new Event('change', {bubbles: true}));
              await new Promise(r => setTimeout(r, 1500));

              const state = await (await fetch('/api/session/state')).json();
              const patch = await post('/api/patch');
              const graph = await post('/api/patch/graph');
              const ground = await post('/api/ground');

              // Re-queried inside the loop and never collected up front: a
              // click re-renders its panel, React delegates events at the
              // root, and a button detached by an earlier render is a button
              // whose click goes nowhere.
              const asked = [];
              for (const sel of ['.panel.patch', '.panel.pgraph', '.panel.ground']) {
                const panel = document.querySelector(sel);
                if (!panel) continue;
                const button = [...panel.querySelectorAll('button')]
                  .find(b => (b.textContent || '').startsWith('Show the recorded'));
                if (!button) continue;
                asked.push(sel + ' — ' + button.textContent.trim());
                button.click();
                await new Promise(r => setTimeout(r, 700));
              }
              await new Promise(r => setTimeout(r, 600));

              const drawn = (sel) => {
                const el = document.querySelector(sel);
                return el ? el.innerText : '';
              };
              return {
                available: {
                  patch: state && state.patch ? state.patch.available === true : false,
                  graph: state && state.patch_graph
                    ? state.patch_graph.available === true : false,
                  ground: state && state.ground
                    ? state.ground.available === true : false,
                },
                status: {
                  patch: patch.status,
                  graph: graph.status,
                  ground: ground.status,
                },
                served: {
                  patch: patch.body,
                  graph: graph.body,
                  ground: ground.body,
                },
                mounted: {
                  patch: !!document.querySelector('.panel.patch'),
                  graph: !!document.querySelector('.panel.pgraph'),
                  ground: !!document.querySelector('.panel.ground'),
                },
                // THE PROMPT BOXES, which `innerText` cannot see: the pair a
                // panel was prefilled with lives in an <input> value, not in
                // the rendered text.
                graphPrompts: [
                  ...document.querySelectorAll('.panel.pgraph input'),
                ].map((i) => i.value).join(' | '),
                text: {
                  patch: drawn('.panel.patch'),
                  graph: drawn('.panel.pgraph'),
                  ground: drawn('.panel.ground'),
                },
                asked,
              };
            }"""
        )
        await browser.close()
        got["crashed"] = crashed
        return got


async def robot_side(port: int) -> dict:
    """Open that file in the real viewer and read what landed on screen."""
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        fetched: list[str] = []
        page.on(
            "request",
            lambda r: fetched.append(r.url) if "beacon.invalid" in r.url else None,
        )
        await page.goto(f"http://127.0.0.1:{port}/", wait_until="networkidle")
        got = await page.evaluate(
            """async () => {
              const blob = await (await fetch('./robot.mri')).blob();
              const file = new File([blob], 'robot.mri');
              const input = document.querySelector('input[type=file][accept=".mri"]');
              const dt = new DataTransfer(); dt.items.add(file);
              input.files = dt.files;
              input.dispatchEvent(new Event('change', {bubbles: true}));
              await new Promise(r => setTimeout(r, 1500));

              const served = await (await fetch('/api/vla/replay')).json();
              const panel = [...document.querySelectorAll('.panel')]
                .find(p => p.innerText.includes('ROBOT FINDING'));
              return {
                served: served.available === true,
                mounted: !!panel,
                text: panel ? panel.innerText : '',
                linked: [...document.images]
                  .map(i => i.src)
                  .filter(s => !s.startsWith('data:')).length,
              };
            }"""
        )
        await browser.close()
        got["requested"] = fetched
        return got


async def image_side(port: int) -> dict:
    """Open that file in the real viewer and read what it put on screen.

    THIS EXISTS BECAUSE THE PANEL WAS NOT MOUNTED HERE AT ALL. `ImageRunReplay`
    sat inside App's `!VIEWER` gate under a comment explaining that it sits
    OUTSIDE it -- so the build A6 exists for, the recipient's, was the one
    build that never rendered a shared image run, while `/api/image/replay`
    answered `available: true` to nobody. No unit test could see that: the
    component was correct and nothing rendered it.
    """
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        fetched: list[str] = []
        page.on(
            "request",
            lambda r: fetched.append(r.url) if "beacon.invalid" in r.url else None,
        )
        await page.goto(f"http://127.0.0.1:{port}/", wait_until="networkidle")
        got = await page.evaluate(
            """async () => {
              const blob = await (await fetch('./image.mri')).blob();
              const file = new File([blob], 'image.mri');
              const input = document.querySelector('input[type=file][accept=".mri"]');
              const dt = new DataTransfer(); dt.items.add(file);
              input.files = dt.files;
              input.dispatchEvent(new Event('change', {bubbles: true}));
              await new Promise(r => setTimeout(r, 1500));

              const served = await (await fetch('/api/image/replay')).json();
              const panel = [...document.querySelectorAll('.panel')]
                .find(p => p.innerText.includes('IMAGE RUN'));
              return {
                served: served.available === true,
                mounted: !!panel,
                text: panel ? panel.innerText : '',
                cells: document.querySelectorAll('.irr-cell').length,
                dimmed: document.querySelectorAll('.irr-cell.pad').length,
                linked: [...document.images]
                  .map(i => i.src)
                  .filter(s => !s.startsWith('data:')).length,
              };
            }"""
        )
        await browser.close()
        got["requested"] = fetched
        return got


def python_side(data: bytes) -> dict:
    sys.path.insert(0, str(ROOT))
    from modelmri import session

    parsed = session.parse(data)
    total, cells, worst = 0.0, 0, 0.0
    for key in sorted(parsed.attention):
        layer, head = (int(x) for x in key.split(":"))
        for row in parsed.attention_slice(layer, head)["matrix"]:
            rs = 0.0
            for v in row:
                total += v
                cells += 1
                rs += v
            worst = max(worst, abs(rs - 1))
    return {
        "slices": len(parsed.attention),
        "cells": cells,
        "checksum": round(total, 6),
        "worst": round(worst, 5),
        "tokens": parsed.tokens,
    }


def serve(directory: Path, port: int) -> socketserver.TCPServer:
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(directory)
    )
    httpd = socketserver.TCPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


# `?f=` names a file the local server is serving. Each of these tries to make
# the page fetch somewhere else; a backslash defeated the first version of the
# guard, which pattern-matched instead of resolving.
HOSTILE = [
    "https://evil.example/x.mri",
    "//evil.example/x.mri",
    "\\\\evil.example\\x.mri",
    "\\/evil.example/x.mri",
    "http://127.0.0.1:9/x.mri",
    "/etc/passwd",
    "../../../../etc/passwd",
    "..%2f..%2fpyproject.toml",
    "javascript:alert(1)",
    "data:text/plain,x",
]


# What the page says about itself after a probe. The file input is rendered
# unconditionally by SessionBar, so it is the marker for "the viewer mounted";
# `.panel.replay` only exists once a session is open.
_STATE = """() => {
  if (document.querySelector('.panel.replay')) return 'opened';
  if (document.querySelector('input[type=file][accept=".mri"]')) return 'idle';
  return 'absent';
}"""


async def hostile_side(port: int) -> dict:
    """Load the viewer with each hostile `?f=` and watch what it requests.

    Returns escaped probes AND probes that never ran. The second list is the
    point: this is a security check, and a probe whose navigation failed looks
    exactly like a probe that loaded and was correctly blocked — no off-origin
    request, no `.panel.replay`. Without it, a dead browser or a server that
    never came up reports ten clean probes having tested nothing.
    """
    from playwright.async_api import Error as PlaywrightError
    from playwright.async_api import async_playwright

    origin = f"http://127.0.0.1:{port}/"
    escaped: list[str] = []
    vacuous: list[str] = []
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        seen: list[str] = []
        page.on("request", lambda r: seen.append(r.url))
        for probe in HOSTILE:
            seen.clear()
            url = f"{origin}?f={urllib.parse.quote(probe, safe='')}"
            failure = ""
            try:
                await page.goto(url, wait_until="networkidle")
            except PlaywrightError as err:
                # One class covers everything Playwright reports here: a
                # navigation the browser aborted, and the `networkidle`
                # timeout (TimeoutError subclasses Error). Neither is fatal to
                # the sweep — the state check below decides whether this probe
                # still tested anything — but neither is a pass on its own,
                # so the reason is kept rather than dropped.
                failure = str(err).splitlines()[0][:120]
            await page.wait_for_timeout(400)
            for requested in seen:
                # Anything off this origin, or reaching above the served
                # directory, means the guard let it through.
                if not requested.startswith(origin):
                    escaped.append(f"{probe} -> {requested}")
            try:
                state = await page.evaluate(_STATE)
            except PlaywrightError as err:
                state = "absent"
                failure = failure or str(err).splitlines()[0][:120]
            if state == "opened":
                escaped.append(f"{probe} -> opened a session")
            elif state == "absent":
                vacuous.append(
                    f"{probe} -> viewer never loaded: {failure or 'no page'}"
                )
        await browser.close()
    return {"escaped": escaped, "vacuous": vacuous, "probes": len(HOSTILE)}


async def browser_side(port: int) -> dict:
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(f"http://127.0.0.1:{port}/", wait_until="networkidle")
        result = await page.evaluate(
            """async () => {
              const blob = await (await fetch('./parity.mri')).blob();
              const file = new File([blob], 'parity.mri');
              const input = document.querySelector('input[type=file][accept=".mri"]');
              const dt = new DataTransfer(); dt.items.add(file);
              input.files = dt.files;
              input.dispatchEvent(new Event('change', {bubbles: true}));
              await new Promise(r => setTimeout(r, 1200));

              const meta = await (await fetch('/api/attention/meta')).json();
              if (!meta.available) return {error: 'the viewer did not open the file'};
              let total = 0, cells = 0, worst = 0, tokens = null;
              for (let l = 0; l < meta.n_layers; l++) {
                for (let h = 0; h < meta.n_heads; h++) {
                  const d = await (await fetch(`/api/attention?layer=${l}&head=${h}`)).json();
                  if (d.error) return {error: d.error};
                  tokens = d.tokens;
                  for (const row of d.matrix) {
                    let s = 0;
                    for (const v of row) { total += v; cells++; s += v; }
                    worst = Math.max(worst, Math.abs(s - 1));
                  }
                }
              }
              return {
                slices: meta.n_layers * meta.n_heads,
                cells,
                checksum: Number(total.toFixed(6)),
                worst: Number(worst.toFixed(5)),
                tokens,
              };
            }"""
        )
        await browser.close()
        return result


def main() -> int:
    if not (VIEWER / "index.html").is_file():
        print(
            f"no viewer build at {VIEWER}\n  python scripts/build_frontend.py --viewer",
            file=sys.stderr,
        )
        return 1

    data = build_fixture()
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_bytes(data)
    (VIEWER / "parity.mri").write_bytes(data)
    (VIEWER / "image.mri").write_bytes(build_image_fixture())
    (VIEWER / "robot.mri").write_bytes(build_robot_fixture())
    (VIEWER / "diff.mri").write_bytes(build_diff_fixture())
    (VIEWER / "causal.mri").write_bytes(build_causal_fixture())
    (VIEWER / "lens.mri").write_bytes(build_lens_fixture())
    (VIEWER / "ranking.mri").write_bytes(build_ranking_fixture())

    expected = python_side(data)
    port = 5921
    httpd = serve(VIEWER, port)
    try:
        got = asyncio.run(browser_side(port))
        hostile = asyncio.run(hostile_side(port))
        shared = asyncio.run(image_side(port))
        robot = asyncio.run(robot_side(port))
        diff = asyncio.run(diff_side(port))
        causal = asyncio.run(causal_side(port))
        lens = asyncio.run(lens_side(port))
        rank = asyncio.run(ranking_side(port))
    finally:
        httpd.shutdown()

    if got.get("error"):
        print(f"FAILED: {got['error']}", file=sys.stderr)
        return 1

    print(f"  fixture   {len(data) / 1024:.1f} KB, {expected['slices']} slices")
    cells_ok = True
    for key in ("slices", "cells", "checksum", "worst", "tokens"):
        same = got.get(key) == expected[key]
        cells_ok = cells_ok and same
        shown = key if key != "tokens" else "tokens"
        mark = "PASS" if same else "FAIL"
        detail = (
            f"{expected[key]}"
            if key != "tokens"
            else f"{len(expected[key])} identical"
            if same
            else f"python={expected[key][:4]} browser={str(got.get(key))[:60]}"
        )
        print(f"  [{mark}] {shown:9} {detail}")
        if not same and key != "tokens":
            print(f"         python={expected[key]}  browser={got.get(key)}")

    print()
    # A probe that did not run is not a probe that passed. Reported as its own
    # failure line rather than folded into "escaped", because the two mean
    # opposite things: one is a guard that leaked, the other is a guard nobody
    # tested.
    tested = hostile["probes"] - len(hostile["vacuous"])
    clean = not hostile["escaped"] and not hostile["vacuous"]
    ok = cells_ok and clean
    if hostile["escaped"]:
        print(f"  [FAIL] ?f=       escaped: {hostile['escaped'][:4]}")
    if hostile["vacuous"]:
        print(
            f"  [FAIL] ?f=       {len(hostile['vacuous'])} probe(s) tested "
            f"nothing: {hostile['vacuous'][:4]}"
        )
    if clean:
        print(
            f"  [PASS] ?f=       {tested} hostile values, all loaded the "
            f"viewer, none escaped the origin"
        )

    print()
    # A SHARED IMAGE RUN, IN THE BUILD IT WAS WRITTEN FOR. Each line is one
    # thing a file from a stranger could have made this page do.
    image_ok = True
    for label, passed, detail in (
        (
            "mounted",
            shared["served"] and shared["mounted"],
            "the panel is on the page"
            if shared["mounted"]
            else "the panel is NOT mounted in the viewer build"
            if shared["served"]
            else "the viewer did not serve the section at all",
        ),
        (
            "no beacon",
            not shared["requested"] and shared["linked"] == 0,
            f"{shared['linked']} linked <img>, "
            f"{len(shared['requested'])} request(s) off-origin",
        ),
        (
            "reported",
            "never fetches" in shared["text"],
            "the dropped frame is named, not silently missing",
        ),
        (
            "no padding",
            shared["cells"] > 0 and shared["dimmed"] == 0,
            f"{shared['dimmed']} of {shared['cells']} cells dimmed with no "
            f"measured boundary",
        ),
        (
            "unstated",
            "does not say how many steps" in shared["text"]
            and "of 0 step" not in shared["text"],
            "an unstated run length is SAID to be unstated, not printed as 0",
        ),
    ):
        mark = "PASS" if passed else "FAIL"
        print(f"  [{mark}] image     {label:10} — {detail}")
        image_ok = image_ok and passed
    ok = ok and image_ok

    print()
    # A SHARED ROBOT FINDING, in the build it was written for. `/api/vla/share`
    # wrote this section for months with no route and no panel behind it.
    robot_ok = True
    for label, passed, detail in (
        (
            "mounted",
            robot["served"] and robot["mounted"],
            "the panel is on the page"
            if robot["mounted"]
            else "the panel is NOT mounted — a shared robot finding is unreadable"
            if robot["served"]
            else "the viewer did not serve the section at all",
        ),
        (
            "no beacon",
            not robot["requested"] and robot["linked"] == 0,
            f"{robot['linked']} linked <img>, "
            f"{len(robot['requested'])} request(s) off-origin",
        ),
        (
            "control",
            "not yet a finding" in robot["text"],
            "an uncontrolled block is named as uncontrolled, not as a result",
        ),
        (
            "agreement",
            "-0.120" in robot["text"],
            "a negative attention/cause agreement keeps its sign",
        ),
        (
            "no numbers",
            "control not recorded" in robot["text"]
            and "not evidence on its own" in robot["text"],
            "a verdict with no control numbers behind it says so",
        ),
        (
            "no fake zero",
            "the best of 0 random occlusion" not in robot["text"],
            "a missing control is not printed as a control of zero over zero draws",
        ),
    ):
        mark = "PASS" if passed else "FAIL"
        print(f"  [{mark}] robot     {label:10} — {detail}")
        robot_ok = robot_ok and passed
    ok = ok and robot_ok

    print()
    # A SHARED MODEL COMPARISON, and the head labels beside it. `model_diff`
    # had no reader on any surface; `head_types` had one in the app and none
    # in THIS build, where the route fell through to "install ModelMRI" over a
    # file that carries the labels.
    diff_ok = True
    for label, passed, detail in (
        (
            "mounted",
            diff["served"] and diff["mounted"],
            "the panel is on the page"
            if diff["mounted"]
            else "the panel is NOT mounted — a shared comparison is unreadable"
            if diff["served"]
            else "the viewer did not serve the section at all",
        ),
        (
            "both names",
            "Qwen/Qwen3-1.7B" in diff["text"]
            and "Qwen3-1.7B-finetuned" in diff["text"],
            "a diff can ride in a file about a third model, so it names its own",
        ),
        (
            "no divergence",
            "anywhere in particular" in diff["text"],
            "`consensus_layer: null` reads as a result, not a missing field",
        ),
        (
            "labels",
            diff["labelled"],
            "head labels are served from the file, not refused with 'install ModelMRI'",
        ),
        (
            "no median",
            "median —" in diff["text"],
            "a layer row with no median reads as unmeasured, not as 0.00000",
        ),
        (
            "no fake count",
            "an unrecorded number of prompts" in diff["text"],
            "a row with no prompt count says so rather than claiming zero "
            "prompts were compared",
        ),
    ):
        mark = "PASS" if passed else "FAIL"
        print(f"  [{mark}] diff      {label:13} — {detail}")
        diff_ok = diff_ok and passed
    ok = ok and diff_ok

    print()
    # A SHARED CAUSAL RESULT: the patching grid, the graph walked back out of
    # it, and the grounding beside them. `session.build` has written all three
    # for as long as each has existed — and the viewer answered none of their
    # routes and mounted none of their panels, so in the one build that exists
    # to show recordings, the causal half of a file answered "install
    # ModelMRI" over bytes that already held the measurement.
    causal_ok = True
    refused = [
        name
        for name, body in causal["served"].items()
        if "install modelmri" in body.lower()
    ]
    absent = [name for name, up in causal["mounted"].items() if not up]
    codes = ", ".join(f"{n}={c}" for n, c in causal["status"].items())
    for label, passed, detail in (
        (
            "available",
            all(causal["available"].values()),
            "the session state names all three sections, so each panel can "
            "offer the recording instead of a button that can only refuse"
            if all(causal["available"].values())
            else "the state reports "
            + ", ".join(
                f"{n}={'yes' if flag else 'no'}"
                for n, flag in causal["available"].items()
            )
            + " — a panel that is not told the file carries the measurement "
            "cannot show it",
        ),
        (
            "answered",
            all(code == 200 for code in causal["status"].values()),
            f"all three routes served the recording ({codes})"
            if all(code == 200 for code in causal["status"].values())
            else f"a route the file has an answer for did not give it ({codes})",
        ),
        (
            "not refused",
            not refused,
            "no route told the reader to install the tool over a file that "
            "already carries the measurement"
            if not refused
            else f"{', '.join(refused)} answered with the shim's install-the-"
            "tool refusal, which is false about this file and the one thing "
            "the reader cannot act on",
        ),
        (
            "mounted",
            not absent,
            "all three panels are on the page"
            if not absent
            else f"{', '.join(absent)} never mounted"
            + (
                f"; the page threw {causal['crashed'][0]}"
                if causal["crashed"]
                else ", and nothing threw — this build simply does not render "
                "them, which no unit test can see"
            ),
        ),
        (
            "patch grid",
            "0.87" in causal["text"]["patch"],
            "the site that recovered 0.87 of the gap is drawn, so the grid "
            "rendered rather than merely mounting",
        ),
        (
            "graph",
            "137" in causal["text"]["graph"]
            and "expanded one level back" in causal["text"]["graph"],
            "the graph names the 137 senders it scored and the rule that "
            "chose its edges, which is what makes it a measurement rather "
            "than a picture of one",
        ),
        (
            "ground nats",
            "0.4271" in causal["text"]["ground"],
            "the passage the answer depended on carries its 0.4271 nats, so "
            "the finding is readable and not just the question it answered",
        ),
        (
            "own prompts",
            "Louvre" in causal["graphPrompts"],
            "the graph panel prefills with the pair the GRAPH was measured on, "
            "not the patch section's -- a file can carry a graph and a trace "
            "of two different prompts",
        ),
        (
            "pruning",
            "41" in causal["text"]["graph"] and "9" in causal["text"]["graph"],
            "the graph says how many senders it found too weak and how many "
            "it never tested, so a pruned circuit is not read as a whole one",
        ),
        (
            "cost",
            "119.4s" in causal["text"]["graph"],
            "the run's real duration is printed, not a measured-looking zero "
            "standing in for a number the file never carried",
        ),
        (
            "unmeasured",
            "not measured" in causal["text"]["ground"]
            and "0.0000" not in causal["text"]["ground"],
            "attention this model never produced reads as unmeasured, not as "
            "a share of zero — which would be a finding nobody took",
        ),
    ):
        mark = "PASS" if passed else "FAIL"
        print(f"  [{mark}] causal    {label:11} — {detail}")
        causal_ok = causal_ok and passed
    ok = ok and causal_ok

    print()
    # A SHARED LOGIT LENS. The oldest section in the format and the last one
    # with no reader: `session.build` writes it, `session._lens` validates it,
    # `runtime.logit_lens` serves it out of a replay — and the only panel that
    # draws one is mounted inside the features panel, which is `!replay`, so
    # nothing could ask. The route was not handled here at all.
    lens_ok = True
    entropy_at = {row["name"].strip(): row["entropy"].strip() for row in lens["rows"]}
    for label, passed, detail in (
        (
            "available",
            lens["available"],
            "the session state names the lens, so the panel can offer the "
            "recording instead of a button that can only refuse"
            if lens["available"]
            else "the state does not say the file carries a lens, so nothing "
            "downstream can mount a panel for it",
        ),
        (
            "answered",
            lens["status"] == 200,
            f"the route served the recording (status {lens['status']})",
        ),
        (
            "not refused",
            "install modelmri" not in lens["body"].lower(),
            "the route did not tell the reader to install the tool over a "
            "file that already carries the trajectory"
            if "install modelmri" not in lens["body"].lower()
            else "the shim's install-the-tool refusal came back for a lens "
            "sitting in the open file — the one refusal a reader cannot act on",
        ),
        (
            "runtime shape",
            lens["shape"]["recorded"]
            and lens["shape"]["layers"] == 4
            and lens["shape"]["final"] == " Paris"
            and lens["shape"]["settled_at"] == 2,
            "the answer is `runtime.logit_lens`'s replay shape exactly — the "
            "trajectory under `layers` with the recorded scalars spread "
            f"beside it ({lens['shape']})",
        ),
        (
            "mounted",
            lens["mounted"],
            "the panel is on the page"
            if lens["mounted"]
            else "the panel never mounted"
            + (
                f"; the page threw {lens['crashed'][0]}"
                if lens["crashed"]
                else ", and nothing threw — this build simply does not render "
                "it, which no unit test can see"
            ),
        ),
        (
            "asked",
            lens["asked"].startswith("Show the recorded"),
            f"the panel offers the recording rather than a live run "
            f"({lens['asked']!r})",
        ),
        (
            "probability",
            "p = 0.9137" in lens["titles"] and "91%" in lens["text"],
            "the last layer's 0.9137 is on screen, so the trajectory drew "
            "rather than merely mounting",
        ),
        (
            "unmeasured",
            entropy_at.get("L 02") == "—",
            "a layer whose entropy the file never carried reads as unmeasured"
            if entropy_at.get("L 02") == "—"
            else f"the row with no entropy reads {entropy_at.get('L 02')!r} — "
            "an entropy nobody measured must not be printed as one",
        ),
        (
            "measured zero",
            entropy_at.get("L 03") == "0.00",
            "an entropy that really is zero still prints as zero, so "
            '"unmeasured" did not swallow a reading with it',
        ),
        (
            "no white screen",
            not lens["crashed"],
            "nothing threw while drawing a trajectory with a hole in it"
            if not lens["crashed"]
            else f"the page threw {lens['crashed'][0]} — there is no error "
            "boundary above this panel, so that is the whole viewer gone white",
        ),
    ):
        mark = "PASS" if passed else "FAIL"
        print(f"  [{mark}] lens      {label:15} — {detail}")
        lens_ok = lens_ok and passed
    ok = ok and lens_ok

    print()
    # A SHARED HEAD RANKING, AND THE LABELS BEHIND IT. The tool's headline
    # measurement: `session.build` writes it, `session._ranking` validates it
    # row by row so it can be re-read, `runtime.ablate_heads` serves it out of
    # a replay — and `AttentionPanel` gated the only button that asks on
    # `!replay`, under a comment that is true of MEASURING a ranking and false
    # of SHOWING one already in the file. The labels were locked one level
    # deeper still: their only caller sits inside `{ranked && …}`, a block a
    # recording could never open.
    ranking_ok = True
    for label, passed, detail in (
        (
            "available",
            rank["available"]["ranking"] and rank["available"]["types"],
            f"the session state names both sections, and the ranking names "
            f"its target token ({rank['available']['target']!r}) over "
            f"{rank['available']['rows']} ranked rows"
            if rank["available"]["ranking"] and rank["available"]["types"]
            else f"the state reports ranking="
            f"{'yes' if rank['available']['ranking'] else 'no'}, head_types="
            f"{'yes' if rank['available']['types'] else 'no'} — a panel that "
            f"is not told the file carries the measurement cannot show it",
        ),
        (
            "answered",
            rank["status"] == 200,
            f"the route served the recording (status {rank['status']})",
        ),
        (
            "not refused",
            "install modelmri" not in rank["body"].lower(),
            "the route did not tell the reader to install the tool over a "
            "file that already carries the ranking"
            if "install modelmri" not in rank["body"].lower()
            else "the shim's install-the-tool refusal came back for a ranking "
            "sitting in the open file — the one refusal a reader cannot act on",
        ),
        (
            "runtime shape",
            rank["shape"]["recorded"]
            and rank["shape"]["ranked"] == 3
            and rank["shape"]["baseline"] == "zero"
            and rank["shape"]["target_token"] == " Paris",
            "the answer is `runtime.ablate_heads`'s replay shape exactly — "
            "the recorded section spread whole, with `recorded` beside it "
            f"({rank['shape']})",
        ),
        (
            "mounted",
            rank["mounted"],
            "the panel is on the page"
            if rank["mounted"]
            else "the attention panel never mounted"
            + (
                f"; the page threw {rank['crashed'][0]}"
                if rank["crashed"]
                else ", so there was nowhere for either control to appear"
            ),
        ),
        (
            "asked",
            rank["askedRanking"].startswith("Show the recorded"),
            f"the panel offers the recording rather than a live sweep "
            f"({rank['askedRanking']!r})",
        ),
        (
            "no live sweep",
            "Rank heads" not in rank["text"],
            "a recording is never offered the button that spends a forward "
            "pass per head — there is no model here to spend them on",
        ),
        (
            "ranking drawn",
            "0.417" in rank["list"],
            "the head that moved the answer by 0.417 nats is in the ranked "
            "list, so the ranking rendered rather than merely arriving",
        ),
        (
            "noise floor",
            "below the noise floor" in rank["list"],
            "the floor travelled with the scores, so the row at 0.0004 reads "
            "as arithmetic rather than as a small finding",
        ),
        (
            "no blank duration",
            "forward passes · s" not in rank["text"],
            "a ranking whose duration the file never carried says so, rather "
            "than printing a gap where the seconds go",
        ),
        (
            "labels asked",
            rank["askedLabels"].startswith("Show the recorded"),
            f"the labels are reachable from the recorded ranking "
            f"({rank['askedLabels']!r})"
            if rank["askedLabels"]
            else "no labels button was found inside the ranking block — the "
            "labels are still locked behind a button a recording cannot press",
        ),
        (
            # Case-folded, and that is not laziness. `.headtype` is
            # `text-transform: uppercase` in styles.css and `innerText`
            # reports text as RENDERED, so the chip reads "PREVIOUS-TOKEN" on
            # screen while the file says "previous-token". Asserting the file
            # spelling against rendered text would fail for a stylesheet
            # reason and send the next reader hunting through the labels
            # route, which is not where the answer would be.
            "label drawn",
            "previous-token" in rank["list"].lower(),
            "the label this head earned is on the row beside its KL, so the "
            "labels route was actually read rather than merely offered",
        ),
        (
            # The label being on screen does NOT mean the label is being
            # SHOWN as a type. `previous-token` renders identically whether
            # its colour rule matched or missed, which is how four dead
            # selectors survived every release that had this check in it:
            # the assertion above passed the entire time they were dead.
            "label coloured",
            bool(rank["chip"]) and rank["chip"]["color"] != rank["chip"]["inherited"],
            f"the chip draws in its own colour rather than the row's — "
            f"{rank['chip']['cls'].split()[-1]} is "
            f"{rank['chip']['color']}, against {rank['chip']['inherited']} "
            f"beside it"
            if rank["chip"] and rank["chip"]["color"] != rank["chip"]["inherited"]
            else f"the chip is {rank['chip']['color']}, EXACTLY the colour of "
            f"the row it sits in — {rank['chip']['cls']!r} matches no rule "
            f"in styles.css, so the per-type colours are dead code and all "
            f"four types read the same"
            if rank["chip"]
            else "the ranked list has no .headtype chip to read a colour off",
        ),
        (
            "no white screen",
            not rank["crashed"],
            "nothing threw while drawing a ranking and its labels"
            if not rank["crashed"]
            else f"the page threw {rank['crashed'][0]} — there is no error "
            "boundary above this panel, so that is the whole viewer gone white",
        ),
    ):
        mark = "PASS" if passed else "FAIL"
        print(f"  [{mark}] ranking   {label:17} — {detail}")
        ranking_ok = ranking_ok and passed
    ok = ok and ranking_ok

    print()
    # Two different failures, and the last line has to name the right one:
    # "THE VIEWER DISAGREES WITH THE TOOL" about a run where every cell
    # matched and the browser simply never started would send the next reader
    # looking for a quantisation bug that is not there.
    if ok:
        print("the viewer and the tool agree on every cell")
    elif not cells_ok:
        print("THE VIEWER DISAGREES WITH THE TOOL")
    elif not image_ok:
        print("THE VIEWER MISHANDLES A SHARED IMAGE RUN — see above")
    elif not robot_ok:
        print("THE VIEWER MISHANDLES A SHARED ROBOT FINDING — see above")
    elif not diff_ok:
        print("THE VIEWER MISHANDLES A SHARED MODEL COMPARISON — see above")
    elif not causal_ok:
        print("THE VIEWER MISHANDLES A SHARED CAUSAL RESULT — see above")
    elif not lens_ok:
        print("THE VIEWER MISHANDLES A SHARED LOGIT LENS — see above")
    elif not ranking_ok:
        print("THE VIEWER MISHANDLES A SHARED HEAD RANKING — see above")
    else:
        print("every cell matched, but the ?f= guard was not proven — see above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
