"""`.mri` — an analysis you can send to someone who has no GPU.

Everything ModelMRI shows you is currently ephemeral. You find the head that
moves the subject token, and the only way to show anyone is a screenshot,
which they cannot explore. Reproducing it themselves means downloading the
model, matching your prompt, and finding the same head.

A `.mri` file is the observation, not the model: the tokens, the attention
that was actually captured, the generation, and the run's settings. It opens
in any ModelMRI with nothing loaded, and every panel reads it exactly as it
reads a live model — because the runtime serves it through the same methods.

**Size is the design constraint.** A 24-layer, 14-head, 141-token attention
tensor is 6.7 million numbers; as JSON at four decimals that is tens of
megabytes for something meant to be attached to a message. Two decisions fix
it:

  * uint8 with a per-matrix scale. Attention rows sum to 1 and are dominated
    by near-zero entries, so a linear quantisation against each matrix's own
    maximum keeps the visible structure — the arcs are drawn from relative
    weight — while costing one byte per value instead of eight.
  * gzip. Attention is highly structured (the sink column, the causal
    triangle of zeros) and compresses hard.

The quantisation is lossy and the file says so in `precision`, because a
number that has silently lost precision is exactly the kind of thing this
project refuses to ship.
"""

from __future__ import annotations

import base64
import gzip
import json
import math
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from . import receipts as _receipts_mod
from .errors import BadRequest, Refusal

FORMAT = "modelmri-session"
FORMAT_VERSION = 1

# Bounds on untrusted input. A `.mri` is meant to be forwarded, so `parse`
# takes bytes a stranger sent — and every one of these was reachable.
#
# MAX_FILE is on the compressed bytes; MAX_INFLATED stops a gzip bomb, which
# the server's 64 MB body cap did not (64 MB of zeros inflates to ~69 GB).
# MAX_CELLS bounds n^2, because the expensive thing is per-slice cells rather
# than tokens. MAX_DIM keeps layer/head counts to something that is a shape.
MAX_FILE = 256 * 1024 * 1024
MAX_INFLATED = 512 * 1024 * 1024
MAX_CELLS = 24_000_000
MAX_DIM = 4096


def _inflate(data: bytes) -> bytes:
    """gunzip, refusing to keep going past MAX_INFLATED.

    `gzip.decompress` has no bound at all: it allocates whatever the stream
    tells it to. Decompressing incrementally and checking `eof` is the only
    way to tell "the file ended" from "the file is still going and we have
    stopped listening".
    """
    engine = zlib.decompressobj(31)  # 31 = gzip wrapper
    raw = engine.decompress(data, MAX_INFLATED)
    if not engine.eof:
        raise SessionError(
            f"this file expands to more than {MAX_INFLATED // 1024 // 1024} MB. "
            f"A session holds an observation, not a model — that is not one."
        )
    return raw


class SessionError(BadRequest):
    """The file is not a session we can open, and we say why.

    A `BadRequest`, so it answers 422 in its own words on all three routes
    that serve this module. It used to be a plain `ValueError`, which meant
    `/api/attention` and `/api/session/open` answered it through a
    transitional arm in server.py and `/api/session/export` — which never got
    that arm — answered it as a generic 500. The same sentence, written for
    the same reader, came back three different ways depending on which button
    was pressed. `BadRequest` is a `ValueError`, so every `except ValueError`
    that was catching this still catches it.

    422 and not 409 because every one of these is a fact about the bytes the
    caller sent: too big, truncated, not gzip, a format version from the
    future, a slice the file does not contain. The one check that is NOT about
    the file — an attention map full of nan, on the way OUT — raises `Refusal`
    instead, and says so where it is raised.
    """


MAX_PATCH_CELLS = 2_000_000

# One row per head. A whole-model sweep is n_layers x n_heads:
# 448 on Qwen3-1.7B, ~1,800 on a 70B. The bound is far above any real model
# and far below what would make a browser lay out a table for a minute.
MAX_RANKING_ROWS = 20_000

# The ranking's own strings -- baseline name, target token, corpus label. A
# label is a few words; anything longer is a payload heading for a browser.
MAX_RANKING_TEXT = 1_000

# An attribution graph section carries a PRUNED edge list, never a dense
# matrix -- `circuit.py` reduces before anything reaches here. The bound is on
# what a browser will lay out: 50,000 edges is already far past what anyone
# reads, and a file claiming millions is not a graph anyone measured.
MAX_GRAPH_EDGES = 50_000
MAX_GRAPH_NODES = 200_000

# Longest string this section may carry. A note or a prompt is a sentence; a
# 40 MB one is a payload, and it renders into somebody else's browser.
MAX_GRAPH_TEXT = 4_000

# A PATCHING graph, which is a different section from `graph` above and far
# smaller: `patch_graph.MAX_EDGES` is 400 and every edge cost eight control
# passes to earn. These bounds are generous over that and nowhere near
# `MAX_GRAPH_EDGES`, because a file claiming 50,000 patched edges is claiming
# 400,000 forward passes nobody sat through.
MAX_PATCH_GRAPH_EDGES = 4_000
MAX_PATCH_GRAPH_NODES = 4_000

# Draws behind one edge. `patch.CONTROL_DRAWS` is 8 and a caller may raise it;
# a file claiming hundreds per edge is claiming a run nobody made.
MAX_CONTROL_DRAWS = 128

# One row per passage. `ground.MAX_CHUNKS` is 24 and a caller may raise it, but
# a file claiming thousands of passages is not a document anybody read.
MAX_GROUND_CHUNKS = 2_000

# A passage PREVIEW, not a passage. `ground.py` already truncates to about 120
# characters; this is the outer bound on what a stranger's file may put on the
# page, in the same class as MAX_GRAPH_TEXT.
MAX_GROUND_TEXT = 4_000

# A diff carries one row per prompt, one per layer, one per head and one per
# prompt token. The head list is the big one -- n_layers x n_heads, 448 on a
# 1.7B -- and the token list is n_prompts x n_tokens.
MAX_DIFF_ROWS = 20_000

# A prompt travels in full, unlike a grounding document. The two are not the
# same kind of text: a grounded document is source material somebody attached,
# and a prompt set is what they ASKED -- the same thing this format already
# carries whole in `prompt`. Bounded anyway, because it reaches a browser.
MAX_DIFF_TEXT = 1_000

# A robot section carries ONE frame and a handful of grids. The grid bound is
# the tower's patch grid -- 32x32 on SmolVLA -- and anything claiming
# thousands is not a patch grid anybody measured.
MAX_VLA_GRID = 256

# The frame travels as a PNG data URL. THE CAP IS THE FIRST OF THIS SECTION'S
# TWO BLOCKING RULES: a frame silently shrunk under a causal map is a wrong
# picture, so the writer states the resolution it wrote and whether it
# downsampled, and the reader refuses a payload past this.
MAX_VLA_FRAME_BYTES = 4_000_000


def _graph(doc: dict) -> dict:
    """The attribution-graph section of an untrusted file, or nothing.

    Same standard as `_patch`, and one extra rule that is the whole reason
    #53 exists: a graph section MUST carry its provenance. This tool did not
    compute these attributions, and a `.mri` that renders them without saying
    so is precisely the confusion the feature was built to prevent -- so a
    graph without `provenance.measured_by` is refused rather than rendered
    under ModelMRI's own chrome.

    Absent is fine. Malformed is refused, not dropped: a damaged file shown as
    an intact one minus a section is the failure this module exists to avoid.
    """
    raw = doc.get("graph")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise SessionError("this session's graph section is not a set of fields")

    provenance = raw.get("provenance")
    # A NON-EMPTY STRING, not merely truthy. All four copies of this guard
    # were truthiness tests, and `"measured_by": true` passes every one of
    # them while React renders a boolean as nothing -- so a forwarded graph
    # rendered under ModelMRI's chrome with a BLANK disclaimer, which is the
    # exact outcome the guard exists to prevent. `" "` did it too.
    claim = provenance.get("measured_by") if isinstance(provenance, dict) else None
    if not isinstance(provenance, dict) or not (
        isinstance(claim, str) and claim.strip()
    ):
        raise SessionError(
            "this session carries an attribution graph with no provenance. A "
            "graph ModelMRI did not compute must say who did, so a section "
            "without it is refused rather than rendered as if it were ours."
        )

    nodes = raw.get("n_nodes")
    if not isinstance(nodes, int) or isinstance(nodes, bool) or nodes < 0:
        raise SessionError("this session's graph does not say how many nodes it has")
    if nodes > MAX_GRAPH_NODES:
        raise SessionError(
            f"this session's graph claims {nodes:,} nodes, above the "
            f"{MAX_GRAPH_NODES:,} this reads."
        )

    edges = raw.get("edges")
    if not isinstance(edges, list):
        raise SessionError("this session's graph edge list is missing or malformed")
    if len(edges) > MAX_GRAPH_EDGES:
        raise SessionError(
            f"this session's graph carries {len(edges):,} edges, above the "
            f"{MAX_GRAPH_EDGES:,} this reads. An attribution graph is pruned "
            "to its strongest edges before it travels."
        )

    clean: list[dict] = []
    for edge in edges:
        if not isinstance(edge, dict):
            raise SessionError("this session's graph has a malformed edge")
        source, target, weight = (
            edge.get("source"),
            edge.get("target"),
            edge.get("weight"),
        )
        for value in (source, target):
            if not isinstance(value, int) or isinstance(value, bool):
                raise SessionError("a graph edge names a node that is not an index")
            # Indices reach the viewer as array subscripts and as node ids.
            if not 0 <= value < nodes:
                raise SessionError(
                    f"a graph edge points at node {value}, outside the "
                    f"{nodes} this graph declares."
                )
        if not isinstance(weight, (int, float)) or isinstance(weight, bool):
            raise SessionError("a graph edge has a weight that is not a number")
        # `math.isfinite`, for the reason spelled out forty lines down and not
        # followed here: the identity trick is a correct NaN test and an
        # obscure one, `weight in (inf, -inf)` is the second half bolted on,
        # and CodeQL reads the first as a comparison of identical values. One
        # call says what it means and covers all three. The file had already
        # decided this; this line predated the decision and kept the old shape,
        # which is how one module ends up answering one question two ways.
        if not math.isfinite(weight):
            raise SessionError("a graph edge has a non-finite weight")
        clean.append(
            {"source": int(source), "target": int(target), "weight": float(weight)}
        )

    out = {
        "n_nodes": nodes,
        "edges": clean,
        "provenance": {
            str(k): v
            for k, v in provenance.items()
            if isinstance(v, (str, int, float, bool)) or v is None
        },
    }
    # These three used to be pass-through, and they are the three the panel
    # DEREFERENCES BY METHOD CALL: `summary.density.toExponential`,
    # `summary.max_abs_weight.toFixed`, `notes.map`. The section bounded the
    # one field the viewer only ever renders as text and left the dangerous
    # three unbounded, so a hostile `.mri` white-screened the zero-install
    # viewer -- the one reader that always has a stranger's file.
    prompt = raw.get("prompt")
    if isinstance(prompt, str):
        out["prompt"] = prompt[:MAX_GRAPH_TEXT]

    notes = raw.get("notes")
    if isinstance(notes, list):
        out["notes"] = [n[:MAX_GRAPH_TEXT] for n in notes if isinstance(n, str)][:64]

    summary = raw.get("summary")
    if isinstance(summary, dict):
        clean_summary: dict = {}
        for key, value in summary.items():
            if not isinstance(key, str):
                continue
            if isinstance(value, bool):
                clean_summary[key] = value
            elif isinstance(value, (int, float)):
                # Non-finite reaches `toFixed`/`toExponential` as NaN and
                # renders as "NaN"; worse, `json.dumps` writes a bare `NaN`
                # token that the viewer's own `JSON.parse` rejects outright.
                #
                # `math.isfinite`, not `value == value`: the identity trick is
                # a correct NaN test and an obscure one, it misses both
                # infinities, and CodeQL reads it as a comparison of identical
                # values. One call says what it means and covers all three.
                if math.isfinite(value):
                    clean_summary[key] = value
            elif isinstance(value, str):
                clean_summary[key] = value[:MAX_GRAPH_TEXT]
            elif isinstance(value, dict):
                clean_summary[key] = {
                    str(k): v
                    for k, v in value.items()
                    if isinstance(v, (str, int, float, bool))
                }
        out["summary"] = clean_summary
    return out


def _lens(doc: dict) -> tuple[list, dict]:
    """The logit-lens trajectory of an untrusted file, and its scalars.

    This section was carried in the format from the beginning and never
    validated, because nothing ever wrote it -- `export_session` did not pass
    it, so `lens` was always `[]` and the hole was invisible. It reaches the
    viewer as a table of tokens and a bar per probability, so it is held to the
    same standard as everything else a stranger can send.

    Absent is fine. Malformed is refused rather than dropped.
    """
    rows = doc.get("lens")
    if rows is None:
        return [], {}
    if not isinstance(rows, list):
        raise SessionError("this session's lens section is not a list of layers")
    if len(rows) > MAX_DIM:
        raise SessionError(
            f"this session's lens claims {len(rows):,} layers, above the "
            f"{MAX_DIM:,} this reads."
        )

    clean: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            raise SessionError("this session's lens has a layer that is not fields")
        layer = row.get("layer")
        if not isinstance(layer, int) or isinstance(layer, bool) or layer < 0:
            raise SessionError("this session's lens has a layer with no index")
        tokens = row.get("tokens")
        probs = row.get("probs")
        if not isinstance(tokens, list) or not isinstance(probs, list):
            raise SessionError(f"lens layer {layer} carries no predictions")
        if len(tokens) != len(probs):
            # A row the panel would zip together. Mismatched lengths render as
            # a token with somebody else's probability beside it.
            raise SessionError(
                f"lens layer {layer} has {len(tokens)} tokens and "
                f"{len(probs)} probabilities, which cannot be read together."
            )
        if len(tokens) > MAX_DIM:
            raise SessionError(f"lens layer {layer} carries too many predictions")
        keep: dict = {
            "layer": layer,
            "tokens": [str(t)[:MAX_RANKING_TEXT] for t in tokens],
            "probs": [],
        }
        for value in probs:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise SessionError(
                    f"lens layer {layer} has a probability that is not a number"
                )
            if not math.isfinite(value):
                raise SessionError(f"lens layer {layer} has a non-finite probability")
            keep["probs"].append(float(value))
        for name in ("entropy", "kl_to_final"):
            value = row.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if math.isfinite(value):
                    keep[name] = float(value)
        clean.append(keep)

    raw_info = doc.get("lens_info")
    info: dict = {}
    if isinstance(raw_info, dict):
        final = raw_info.get("final")
        if isinstance(final, str):
            info["final"] = final[:MAX_RANKING_TEXT]
        settled = raw_info.get("settled_at")
        # None is a real value here and NOT the same as absent: "the answer
        # never settles before the last layer" is a finding, and coercing it to
        # 0 would claim it settled immediately.
        if settled is None or (
            isinstance(settled, int) and not isinstance(settled, bool)
        ):
            info["settled_at"] = settled
        n_layers = raw_info.get("n_layers")
        if isinstance(n_layers, int) and not isinstance(n_layers, bool):
            info["n_layers"] = n_layers
        reliability = raw_info.get("reliability")
        if isinstance(reliability, dict):
            info["reliability"] = {
                k: v
                for k, v in reliability.items()
                if isinstance(k, str)
                and (
                    isinstance(v, (bool, int, float))
                    or (isinstance(v, str) and len(v) < MAX_GRAPH_TEXT)
                    or v is None
                )
            }
    return clean, info


def _head_types(doc: dict) -> dict:
    """The head-label section of an untrusted file, or nothing.

    Same standard as `_ranking`. These render as a chip beside a head in the
    list, so a label is bounded text and a score is a finite number before it
    reaches anybody's browser.

    A label without its gates is refused. The whole value of this section is
    that a name was earned against a measured null, and a row carrying the
    name and not the evidence would be exactly the bare assertion the feature
    exists to replace.
    """
    raw = doc.get("head_types")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise SessionError("this session's head-type section is not a set of fields")

    rows = raw.get("labels")
    if not isinstance(rows, list):
        raise SessionError("this session's head types carry no labels")
    if len(rows) > MAX_RANKING_ROWS:
        raise SessionError(
            f"this session claims {len(rows):,} labelled heads, above the "
            f"{MAX_RANKING_ROWS:,} this reads."
        )

    clean: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            raise SessionError("this session has a head-type row that is not fields")
        layer, head = row.get("layer"), row.get("head")
        if not all(
            isinstance(v, int) and not isinstance(v, bool) and 0 <= v < MAX_DIM
            for v in (layer, head)
        ):
            raise SessionError("a head-type row does not name a head")

        label = row.get("label")
        if label is not None and not isinstance(label, str):
            raise SessionError(f"the head-type label for L{layer}H{head} is not text")
        keep: dict = {
            "layer": layer,
            "head": head,
            # None survives as None. "No type detected" is the finding for most
            # heads, and coercing it to "" would make an unlabelled head look
            # like one whose label went missing.
            "label": label[:MAX_RANKING_TEXT] if isinstance(label, str) else None,
        }
        if keep["label"]:
            for name in ("margin", "times_chance", "peak"):
                value = row.get(name)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if math.isfinite(value):
                        keep[name] = float(value)
            kind = row.get("null_kind")
            keep["null_kind"] = kind if isinstance(kind, str) else ""
            if "margin" not in keep or not keep["null_kind"]:
                raise SessionError(
                    f"L{layer}H{head} is labelled {keep['label']!r} with no "
                    f"margin or no null named. A label that does not say what "
                    f"it cleared is the bare assertion this section exists to "
                    f"replace."
                )
        scores = row.get("scores")
        if isinstance(scores, dict):
            keep["scores"] = {
                str(k)[:MAX_RANKING_TEXT]: float(v)
                for k, v in scores.items()
                if isinstance(v, (int, float))
                and not isinstance(v, bool)
                and math.isfinite(v)
            }
        clean.append(keep)

    out: dict = {"labels": clean}
    for name in ("means",):
        value = raw.get(name)
        if isinstance(value, str):
            out[name] = value[:MAX_GRAPH_TEXT]
    for name in ("n_layers", "n_heads", "seq_len", "n_sequences", "seed"):
        value = raw.get(name)
        if isinstance(value, int) and not isinstance(value, bool):
            out[name] = value
    counts = raw.get("counts")
    if isinstance(counts, dict):
        out["counts"] = {
            str(k)[:MAX_RANKING_TEXT]: v
            for k, v in counts.items()
            if isinstance(v, int) and not isinstance(v, bool)
        }
    margin = raw.get("margin_sigma")
    if isinstance(margin, (int, float)) and not isinstance(margin, bool):
        if math.isfinite(margin):
            out["margin_sigma"] = float(margin)
    return out


def _ranking(doc: dict) -> dict:
    """The head-ranking section of an untrusted file, or nothing.

    Same standard as `_patch`: absent is fine, malformed is refused rather
    than dropped. The rows reach a table and a bar width in somebody else's
    browser, so a string where a KL belongs, or a claim of two million heads,
    stops here.

    This section exists so a finding can be CHECKED. Until it, a `.mri`
    recorded that a ranking had run and carried none of it, which meant
    `modelmri verify` could name the measurement and not re-run it -- the one
    number in the file that could not be audited was the headline one.
    """
    raw = doc.get("ranking")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise SessionError("this session's ranking section is not a set of fields")

    rows = raw.get("ranked")
    if not isinstance(rows, list):
        raise SessionError("this session's ranking carries no rows")
    if len(rows) > MAX_RANKING_ROWS:
        raise SessionError(
            f"this session's ranking claims {len(rows):,} heads, above the "
            f"{MAX_RANKING_ROWS:,} this reads."
        )

    clean_rows: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            raise SessionError("this session's ranking has a row that is not fields")
        layer, head = row.get("layer"), row.get("head")
        if not all(
            isinstance(v, int) and not isinstance(v, bool) and 0 <= v < MAX_DIM
            for v in (layer, head)
        ):
            raise SessionError(
                "this session's ranking has a row that does not name a head"
            )
        kl = row.get("kl")
        if not isinstance(kl, (int, float)) or isinstance(kl, bool):
            raise SessionError(
                f"the ranking row for layer {layer} head {head} has no score"
            )
        if not math.isfinite(kl):
            # Same rule the attention path keeps: a non-finite number renders
            # as a blank cell or an infinite bar and explains neither.
            raise SessionError(
                f"the ranking row for layer {layer} head {head} is not finite"
            )
        keep: dict = {"layer": layer, "head": head, "kl": float(kl)}
        # Optional, and copied only when they are the right shape. A resample
        # ranking carries min/max/draws; a zero-baseline one does not, and an
        # absent field must not become 0 -- that would state a measured spread
        # of nothing where none was taken.
        for name in ("kl_min", "kl_max", "p_top_before", "p_top_after"):
            value = row.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if math.isfinite(value):
                    keep[name] = float(value)
        draws = row.get("draws")
        if isinstance(draws, int) and not isinstance(draws, bool) and draws >= 0:
            keep["draws"] = draws
        if isinstance(row.get("flips_top"), bool):
            keep["flips_top"] = row["flips_top"]
        clean_rows.append(keep)

    out: dict = {"ranked": clean_rows}

    baseline = raw.get("baseline")
    # A non-empty string, and deliberately NOT checked against a list of known
    # baselines. `ablate.BASELINES` lives beside torch, and this module is the
    # one that must stay importable without it -- `modelmri inspect` and
    # `modelmri open` are fast precisely because they never touch torch, and
    # duplicating the tuple here would make a file written by a newer version
    # with a fourth baseline unreadable by this one. Requiring that the
    # ranking SAYS which baseline it used is the part that matters: `ablate.py`
    # measures the three agreeing only weakly, so a ranking that does not name
    # its baseline cannot be compared against one that does.
    if not (isinstance(baseline, str) and baseline.strip()):
        raise SessionError(
            "this session's ranking does not say which baseline produced it, "
            "and the baselines disagree — so the rows cannot be read."
        )
    out["baseline"] = baseline[:MAX_RANKING_TEXT]

    # The measured floor travels with the scores. Anything at or below it is
    # arithmetic rather than the model, and a reader without it cannot tell
    # the difference.
    floor = raw.get("noise_floor_kl")
    if isinstance(floor, (int, float)) and not isinstance(floor, bool):
        if math.isfinite(floor):
            out["noise_floor_kl"] = float(floor)

    for name in ("target_token", "corpus", "means"):
        value = raw.get(name)
        if isinstance(value, str):
            out[name] = value[:MAX_RANKING_TEXT]
    for name in ("position", "layer", "passes", "draws"):
        value = raw.get(name)
        if isinstance(value, int) and not isinstance(value, bool):
            out[name] = value
    return out


def _vla(doc: dict) -> dict:
    """The robot-findings section of an untrusted file, or nothing.

    Held to `_patch`'s standard, because a `.mri` is designed to arrive from a
    stranger and this section reaches a browser as an <img> src and two
    nested loops. Every grid is checked for rectangularity and finite values;
    a ragged one renders as a heat map with a torn edge and a NaN renders as
    a smooth blank one, and neither says anything is wrong.

    TWO RULES SPECIFIC TO THIS SECTION.

    THE FRAME MUST SAY ITS OWN SIZE, and whether it was downsampled to get
    here. A causal map is drawn over the frame; a frame silently shrunk to fit
    a byte budget puts every block in the wrong place, and the picture is
    wrong in a way that looks exactly like a finding. A section carrying an
    occlusion map and a frame with no stated resolution is refused.

    THE PROVENANCE IS NOT OPTIONAL. Which policy revision, which dataset,
    which episode, which timestep, which camera. Every other section here
    describes the model the file names; this one describes a policy AND a
    dataset AND one frame of it, and a heat map without those four is a
    picture of nothing in particular.
    """
    raw = doc.get("vla")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise SessionError("this session's robot section is not a set of fields")

    provenance = raw.get("provenance")
    if not isinstance(provenance, dict):
        raise SessionError(
            "this session's robot section carries no provenance. A heat map "
            "without the policy, dataset, episode, timestep and camera that "
            "produced it is a picture of nothing in particular."
        )
    keep_prov: dict = {}
    for name in ("policy", "dataset", "camera", "revision"):
        value = provenance.get(name)
        if not isinstance(value, str) or not value.strip():
            raise SessionError(
                f"this session's robot section does not say which {name} produced it."
            )
        keep_prov[name] = value[:MAX_RANKING_TEXT]
    for name in ("episode", "timestep"):
        value = provenance.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise SessionError(
                f"this session's robot section does not say which {name} produced it."
            )
        keep_prov[name] = value

    def _grid(value, what: str) -> list[list[float]]:
        if not isinstance(value, list) or not value:
            raise SessionError(f"this session's {what} is not a grid")
        if len(value) > MAX_VLA_GRID:
            raise SessionError(
                f"this session's {what} claims {len(value):,} rows, above the "
                f"{MAX_VLA_GRID:,} a patch grid can be."
            )
        width = None
        out: list[list[float]] = []
        for row in value:
            if not isinstance(row, list):
                raise SessionError(f"this session's {what} has a row that is not one")
            if width is None:
                width = len(row)
                if width == 0 or width > MAX_VLA_GRID:
                    raise SessionError(
                        f"this session's {what} has a row of {width} cells"
                    )
            elif len(row) != width:
                # A ragged grid renders as a heat map with a torn edge and
                # nothing on screen says the numbers are wrong.
                raise SessionError(
                    f"this session's {what} is ragged — one row has {len(row)} "
                    f"cells and another has {width}."
                )
            clean = []
            for cell in row:
                if (
                    not isinstance(cell, (int, float))
                    or isinstance(cell, bool)
                    or not math.isfinite(cell)
                ):
                    # A NaN quantises every cell to zero: a smooth, plausible,
                    # entirely blank map. `_quantise` records the same lesson.
                    raise SessionError(
                        f"this session's {what} has a cell that is not a finite "
                        f"number, which renders as a blank map rather than as "
                        f"an error."
                    )
                clean.append(float(cell))
            out.append(clean)
        return out

    out: dict = {"provenance": keep_prov}

    frame = raw.get("frame")
    if frame is not None:
        if not isinstance(frame, str):
            raise SessionError("this session's robot frame is not a data URL")
        if not frame.startswith("data:image/"):
            raise SessionError(
                "this session's robot frame is not an image data URL. A `.mri` "
                "never carries a path or a link — the frame travels inside it "
                "or not at all."
            )
        if len(frame) > MAX_VLA_FRAME_BYTES:
            raise SessionError(
                f"this session's robot frame is {len(frame):,} bytes, above "
                f"the {MAX_VLA_FRAME_BYTES:,} this reads."
            )
        out["frame"] = frame
        size = raw.get("frame_size")
        if (
            not isinstance(size, list)
            or len(size) != 2
            or not all(
                isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in size
            )
        ):
            raise SessionError(
                "this session's robot frame does not state its own resolution. "
                "A causal map is drawn over the frame, so a frame that has "
                "been shrunk without saying so puts every block in the wrong "
                "place — and the picture is wrong in a way that looks exactly "
                "like a finding."
            )
        out["frame_size"] = [int(size[0]), int(size[1])]
        # Stated, never inferred. `False` is the positive claim "this is the
        # resolution the policy saw"; a missing key would let a downsampled
        # frame pass as an original.
        out["frame_downsampled"] = bool(raw.get("frame_downsampled", False))
        if out["frame_downsampled"]:
            note = raw.get("frame_note")
            out["frame_note"] = (
                note[:MAX_RANKING_TEXT]
                if isinstance(note, str) and note.strip()
                else "downsampled to fit the file; the original resolution is "
                "not recorded"
            )

    attention = raw.get("attention")
    if attention is not None:
        if not isinstance(attention, list):
            raise SessionError("this session's robot attention is not a list of layers")
        if len(attention) > MAX_DIM:
            raise SessionError(
                f"this session's robot attention claims {len(attention):,} "
                f"layers, above the {MAX_DIM:,} this reads."
            )
        out["attention"] = [
            _grid(layer, f"robot attention layer {i}")
            for i, layer in enumerate(attention)
        ]

    occlusion = raw.get("occlusion")
    if isinstance(occlusion, dict):
        blocks = occlusion.get("blocks")
        if not isinstance(blocks, list):
            raise SessionError("this session's occlusion map carries no blocks")
        if len(blocks) > MAX_RANKING_ROWS:
            raise SessionError(
                f"this session's occlusion map claims {len(blocks):,} blocks, "
                f"above the {MAX_RANKING_ROWS:,} this reads."
            )
        clean_blocks = []
        for block in blocks:
            if not isinstance(block, dict):
                raise SessionError("an occlusion block is not fields")
            keep: dict = {}
            for name in ("row", "col", "control_draws"):
                value = block.get(name)
                if isinstance(value, int) and not isinstance(value, bool):
                    keep[name] = value
            # `shift` is NOT nullable and the other two are. Every block was
            # measured, so a missing shift is a broken row rather than an
            # honest unknown -- and folding it in with the nullable fields
            # meant `block.get("shift")` returned None, took the None branch,
            # and stored `shift: None`, so the guard below never fired. The
            # block then rendered as a blank cell in a map of real ones.
            shift = block.get("shift")
            if (
                not isinstance(shift, (int, float))
                or isinstance(shift, bool)
                or not math.isfinite(shift)
            ):
                raise SessionError(
                    "an occlusion block carries no shift, so it would render "
                    "as a blank cell in a map of measured ones."
                )
            keep["shift"] = float(shift)
            for name in ("control_max", "attention"):
                value = block.get(name)
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                ):
                    keep[name] = float(value)
                elif value is None:
                    # None survives. An uncontrolled block has no control_max,
                    # and 0.0 would read as "a random occlusion did nothing".
                    keep[name] = None
            clears = block.get("clears_control")
            keep["clears_control"] = None if clears is None else bool(clears)
            clean_blocks.append(keep)
        kept: dict = {"blocks": clean_blocks}
        for name in ("baseline", "means"):
            value = occlusion.get(name)
            if isinstance(value, str):
                kept[name] = value[:MAX_GRAPH_TEXT]
        if not kept.get("baseline"):
            raise SessionError(
                "this session's occlusion map does not say which fill baseline "
                "produced it. Occlusion is out of distribution and the two "
                "baselines do not agree, so a map without its fill is not "
                "reproducible."
            )
        for name in ("stride", "n_blocks", "n_controlled", "passes"):
            value = occlusion.get(name)
            if isinstance(value, int) and not isinstance(value, bool):
                kept[name] = value
        # Which attention map the agreement was measured against. Nullable
        # separately from the block above: an absent layer must stay None
        # rather than fall through to a missing key, because a reader that
        # finds no layer has to be able to tell "not compared" from "layer 0".
        for name in ("compared_layer", "compared_head"):
            value = occlusion.get(name)
            if isinstance(value, int) and not isinstance(value, bool):
                kept[name] = value
            else:
                kept[name] = None
        for name in ("scale", "attention_agreement"):
            value = occlusion.get(name)
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            ):
                kept[name] = float(value)
            elif value is None:
                kept[name] = None
        grid = occlusion.get("grid")
        if isinstance(grid, list) and len(grid) == 2:
            kept["grid"] = [int(g) for g in grid if isinstance(g, int)]
        # A map drawn over a frame needs the frame's resolution stated. See
        # the docstring: this is the first of the two blocking rules.
        if "frame" in out and "frame_size" not in out:
            raise SessionError(
                "this session carries an occlusion map over a frame whose "
                "resolution is not stated."
            )
        out["occlusion"] = kept

    return out


def _model_diff(doc: dict) -> dict:
    """The finetune-diff section of an untrusted file, or nothing.

    Same standard as every other additive section. Two rules are specific to
    this one:

    IT NAMES ITS OWN TWO MODELS, and they need not be the model the rest of
    the file describes. A `.mri` is one analysis of one model; a diff is a
    comparison of two others, and it can legitimately ride in a file about a
    third. `model_a` and `model_b` are therefore REQUIRED -- a diff section
    that does not say what it compared would be read as being about the file's
    own model, which is the one confusion this section can cause.

    A SPREAD WITHOUT ITS `n` IS REFUSED. The entire content of this section is
    that its numbers are distributions over a prompt set rather than single
    measurements, and a median arriving without the count it was taken over is
    exactly the single number the module exists to avoid printing.
    """
    raw = doc.get("model_diff")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise SessionError("this session's model-diff section is not a set of fields")

    model_a = raw.get("model_a")
    model_b = raw.get("model_b")
    if not (isinstance(model_a, str) and model_a.strip()):
        raise SessionError(
            "this session's model-diff does not say which model it compared "
            "FROM. A diff can ride in a file about a different model, so a "
            "section that does not name its own two sides would be read as "
            "being about this file's model."
        )
    if not (isinstance(model_b, str) and model_b.strip()):
        raise SessionError(
            "this session's model-diff does not say which model it compared TO."
        )

    def _spread(value, what: str) -> dict | None:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise SessionError(f"this session's {what} spread is not fields")
        n = value.get("n")
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise SessionError(
                f"the {what} spread carries no prompt count. A median without "
                f"the n it was taken over is the single number this section "
                f"exists to avoid printing."
            )
        out = {"n": n, "name": str(value.get("name") or what)[:MAX_DIFF_TEXT]}
        for key in ("median", "low", "high"):
            number = value.get(key)
            if (
                not isinstance(number, (int, float))
                or isinstance(number, bool)
                or not math.isfinite(number)
            ):
                raise SessionError(f"the {what} spread has no {key}")
            out[key] = float(number)
        n_nonzero = value.get("n_nonzero")
        if isinstance(n_nonzero, int) and not isinstance(n_nonzero, bool):
            out["n_nonzero"] = max(0, min(n_nonzero, n))
        return out

    def _rows(key: str, required: tuple[str, ...]) -> list[dict]:
        rows = raw.get(key)
        if rows is None:
            return []
        if not isinstance(rows, list):
            raise SessionError(f"this session's model-diff {key} is not a list")
        if len(rows) > MAX_DIFF_ROWS:
            raise SessionError(
                f"this session's model-diff claims {len(rows):,} {key}, above "
                f"the {MAX_DIFF_ROWS:,} this reads."
            )
        clean: list[dict] = []
        for row in rows:
            if not isinstance(row, dict):
                raise SessionError(f"a model-diff {key} row is not fields")
            keep: dict = {}
            for name in required:
                value = row.get(name)
                if isinstance(value, bool):
                    keep[name] = value
                elif isinstance(value, (int, float)) and math.isfinite(value):
                    keep[name] = value
                elif isinstance(value, str):
                    keep[name] = value[:MAX_DIFF_TEXT]
                elif value is None:
                    # None survives. `first_divergent_layer` is None when the
                    # cosine never falls, which is a result and not a gap.
                    keep[name] = None
                else:
                    raise SessionError(
                        f"a model-diff {key} row has a {name} this cannot read"
                    )
            clean.append(keep)
        return clean

    out: dict = {
        "model_a": model_a[:MAX_DIFF_TEXT],
        "model_b": model_b[:MAX_DIFF_TEXT],
        "prompts": _rows(
            "prompts",
            (
                "prompt",
                "n_tokens",
                "mean_kl",
                "max_kl",
                "flips",
                "first_divergent_layer",
                "drop",
            ),
        ),
        "layers": _rows("layers", ("layer", "median", "low", "high", "n", "n_first")),
        "heads": _rows(
            "heads",
            ("layer", "head", "median_a", "median_b", "shift", "n", "top_a", "top_b"),
        ),
        "tokens": _rows(
            "tokens",
            (
                "prompt_index",
                "index",
                "token",
                "kl_a",
                "kl_b",
                "shift",
                "newly_used",
                "newly_ignored",
            ),
        ),
    }
    for key, what in (("kl", "KL"), ("flips", "flips")):
        spread = _spread(raw.get(key), what)
        if spread is not None:
            out[key] = spread
    for name in ("means",):
        value = raw.get(name)
        if isinstance(value, str):
            out[name] = value[:MAX_GRAPH_TEXT]
    for name in ("n_prompts", "n_layers", "consensus_layer", "head_passes"):
        value = raw.get(name)
        if isinstance(value, int) and not isinstance(value, bool):
            out[name] = value
        elif value is None:
            # consensus_layer is None when nothing diverged. A result.
            out[name] = None
    for name in ("consensus_share", "seconds"):
        value = raw.get(name)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        ):
            out[name] = float(value)
    return out


def _ground(doc: dict) -> dict:
    """The grounding section of an untrusted file, or nothing.

    Same standard as `_patch` and `_head_types`: a `.mri` is meant to be
    forwarded, so this runs on bytes a stranger sent, and every passage
    preview reaches somebody's browser as text.

    Two rules specific to this section, and both are about the claim rather
    than the bytes:

    A passage carrying `depended_on` WITHOUT the floor it cleared is refused.
    The whole content of "this passage mattered" is that removing it moved the
    answer further than a pass that changed nothing, and a row that asserts
    the verdict while dropping the reference is the bare claim the feature
    exists to replace -- the same rule `_head_types` applies to a label
    without its margin.

    `attention` and `looked_not_used` survive as None. They are three-valued
    on the way out: a model whose attention implementation never built the
    score matrix reports None, and a zero noise floor makes the flag
    undecidable. Coercing either to 0.0 or False here would turn "not
    measured" into "measured, and fine" inside a file whose whole purpose is
    to travel away from the machine that could tell the difference.
    """
    raw = doc.get("ground")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise SessionError("this session's grounding section is not a set of fields")

    rows = raw.get("chunks")
    if not isinstance(rows, list):
        raise SessionError("this session's grounding carries no passages")
    if len(rows) > MAX_GROUND_CHUNKS:
        raise SessionError(
            f"this session claims {len(rows):,} passages, above the "
            f"{MAX_GROUND_CHUNKS:,} this reads."
        )

    floor = raw.get("noise_floor")
    has_floor = (
        isinstance(floor, (int, float))
        and not isinstance(floor, bool)
        and math.isfinite(floor)
    )

    clean: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            raise SessionError("this session has a grounding row that is not fields")
        index = row.get("index")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise SessionError("a grounding row does not name a passage")

        dependence = row.get("dependence")
        if (
            not isinstance(dependence, (int, float))
            or isinstance(dependence, bool)
            or not math.isfinite(dependence)
        ):
            raise SessionError(
                f"passage {index} carries no dependence score, so there is "
                f"nothing measured to render."
            )

        depended = bool(row.get("depended_on"))
        if depended and not has_floor:
            raise SessionError(
                f"passage {index} is marked as one the answer depended on, "
                f"with no noise floor in the file. A verdict that does not "
                f"say what it cleared is the bare claim this section exists "
                f"to replace."
            )

        preview = row.get("preview")
        attention = row.get("attention")
        looked = row.get("looked_not_used")
        keep: dict = {
            "index": index,
            "preview": (preview[:MAX_GROUND_TEXT] if isinstance(preview, str) else ""),
            "dependence": float(dependence),
            "depended_on": depended,
            # None survives. See the docstring: a fused-attention model never
            # produced a share, and 0.0 would read as "nothing looked here".
            "attention": (
                float(attention)
                if isinstance(attention, (int, float))
                and not isinstance(attention, bool)
                and math.isfinite(attention)
                else None
            ),
            # Three-valued on the way out and three-valued on the way in.
            "looked_not_used": None if looked is None else bool(looked),
        }
        n_tokens = row.get("n_tokens")
        if isinstance(n_tokens, int) and not isinstance(n_tokens, bool):
            keep["n_tokens"] = max(0, min(n_tokens, MAX_DIM))
        clean.append(keep)

    out: dict = {"chunks": clean}
    for name in ("question", "answer", "attention_note", "means"):
        value = raw.get(name)
        if isinstance(value, str):
            out[name] = value[:MAX_GROUND_TEXT]
    for name in (
        "answer_p",
        "noise_floor",
        "joint",
        "attention_share",
        "seconds",
    ):
        value = raw.get(name)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        ):
            out[name] = float(value)
    for name in ("n_chunks", "n_prompt_tokens", "position", "passes"):
        value = raw.get(name)
        if isinstance(value, int) and not isinstance(value, bool):
            out[name] = value
    for name in ("attention_available", "floor_degenerate", "ungrounded"):
        # Defaulted DELIBERATELY: a file that omits `attention_available`
        # predates the field, and the safe reading of an absent flag is the
        # one that claims least. `attention_available` defaults False so an
        # old file's blank attention column is not presented as measured.
        out[name] = bool(raw.get(name, False))
    return out


# The trace section's own bounds. `bundle.py` enforces these at WRITE time
# and this enforces them again at READ time, because a `.mri` is meant to be
# forwarded and the reader runs on bytes a stranger sent.
MAX_TRACE_STEPS = 500
MAX_TRACE_TEXT = 4_200  # the writer's clip plus its marker
MAX_TRACE_NAME = 200


def _trace(doc: dict) -> dict:
    """The agent-run section of an untrusted file, or nothing.

    Held to the same standard as `patch` and `vla`. The steps reach the
    viewer's timeline as loop bounds and their `started_ms`/`duration_ms` reach
    a pixel offset, so a string where a number belongs or a 400,000-step claim
    has to stop here rather than in whoever's browser opened the file.

    Absent is fine and common — most sessions carry no agent run. MALFORMED is
    not: it is refused rather than dropped, because a damaged file presented as
    an intact one without that section is the failure this module exists to
    avoid.
    """
    raw = doc.get("trace")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise SessionError("this session's agent-run section is not a set of fields")

    steps = raw.get("steps")
    if not isinstance(steps, list):
        raise SessionError("this session's agent run has no steps list")
    if len(steps) > MAX_TRACE_STEPS:
        raise SessionError(
            f"this session's agent run carries {len(steps)} steps and the "
            f"format holds {MAX_TRACE_STEPS}. A file this size is not one "
            f"somebody's browser can open."
        )

    kept = []
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            raise SessionError(f"step {i} of this session's agent run is not an object")
        kind = str(step.get("kind") or "")
        if not kind:
            raise SessionError(f"step {i} of this session's agent run has no kind")
        clean = {
            "id": str(step.get("id") or f"s{i}")[:120],
            "kind": kind[:40],
            "name": str(step.get("name") or "")[:MAX_TRACE_NAME],
            "input": str(step.get("input") or "")[:MAX_TRACE_TEXT],
            "output": str(step.get("output") or "")[:MAX_TRACE_TEXT],
            "error": bool(step.get("error")),
        }
        parent = step.get("parent_id")
        clean["parent_id"] = str(parent)[:120] if parent else None
        # Nullable, and kept nullable. `duration_ms` is None for a step
        # recorded bare, and 0 is a different claim — the same distinction
        # `traces.py` made the column nullable to express.
        for name in ("started_ms", "duration_ms", "tokens_in", "tokens_out"):
            value = step.get(name)
            clean[name] = (
                int(value)
                if isinstance(value, int) and not isinstance(value, bool)
                else None
            )
        if clean["started_ms"] is None:
            clean["started_ms"] = 0
        kept.append(clean)

    out: dict = {
        "id": str(raw.get("id") or "")[:120],
        "name": str(raw.get("name") or "")[:MAX_TRACE_NAME],
        "started_at": str(raw.get("started_at") or "")[:60],
        "steps": kept,
    }
    total = raw.get("n_steps_total")
    out["n_steps_total"] = (
        int(total)
        if isinstance(total, int) and not isinstance(total, bool)
        else len(kept)
    )
    dropped = raw.get("truncated")
    out["truncated"] = (
        int(dropped)
        if isinstance(dropped, int) and not isinstance(dropped, bool)
        else 0
    )

    ref = raw.get("step_ref")
    if ref:
        ref = str(ref)[:120]
        # The whole point of the section is "click the failing step and land
        # in its attention view". A ref naming a step that is not here would
        # open a bundle whose highlighted step does not exist.
        if not any(s["id"] == ref for s in kept):
            raise SessionError(
                "this session names a failing step that is not among the steps "
                "it carries, so there is nothing for the viewer to open."
            )
        out["step_ref"] = ref
    return out


def _patch(doc: dict) -> dict:
    """The patching section of an untrusted file, or nothing.

    Held to the same standard as `attention`: a `.mri` is meant to be
    forwarded, so this runs on bytes a stranger sent. The grids reach the
    viewer as nested loop bounds and their values reach a colour scale, so a
    ragged grid, a string where a number belongs, or a 40,000 x 40,000 claim
    are all things that have to stop here rather than in whoever's browser
    opened the file.

    Absent is fine and common -- most sessions have no patch trace. MALFORMED
    is not: it is refused rather than dropped, because a damaged file
    presented as an intact one without that section is the failure this
    module exists to avoid.
    """
    raw = doc.get("patch")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise SessionError("this session's patching section is not a set of fields")

    grids = raw.get("grids")
    if not isinstance(grids, dict):
        raise SessionError("this session's patching grids are missing or malformed")

    cells = 0
    clean: dict[str, list[list[float]]] = {}
    for name, grid in grids.items():
        if not isinstance(name, str) or not isinstance(grid, list):
            raise SessionError("this session's patching grids are malformed")
        width = None
        rows: list[list[float]] = []
        for row in grid:
            if not isinstance(row, list) or len(row) > MAX_DIM:
                raise SessionError(f"the {name!r} patching grid is not a grid")
            # Rectangular, because the viewer indexes it as one. A ragged grid
            # renders as a table with holes and no error.
            if width is None:
                width = len(row)
            elif len(row) != width:
                raise SessionError(
                    f"the {name!r} patching grid has rows of different lengths, "
                    "so the file is damaged"
                )
            out: list[float] = []
            for v in row:
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise SessionError(
                        f"the {name!r} patching grid holds something that is "
                        "not a number"
                    )
                # NaN and infinity survive JSON round-trips through most
                # writers and colour-scale to nothing visible.
                if not math.isfinite(v):
                    raise SessionError(
                        f"the {name!r} patching grid holds a value that is not "
                        "finite, so it cannot be drawn"
                    )
                out.append(float(v))
            rows.append(out)
            cells += len(out)
            if cells > MAX_PATCH_CELLS:
                raise SessionError(
                    f"this session's patching grids hold more than "
                    f"{MAX_PATCH_CELLS:,} cells — more than ModelMRI will render"
                )
        if len(rows) > MAX_DIM:
            raise SessionError(f"the {name!r} patching grid has too many layers")
        clean[name] = rows

    sites = raw.get("sites")
    notes = raw.get("notes")
    return {
        "grids": clean,
        "sites": sites if isinstance(sites, list) else [],
        "notes": [n for n in (notes or []) if isinstance(n, str)]
        if isinstance(notes, list)
        else [],
        "clean": raw.get("clean") if isinstance(raw.get("clean"), str) else "",
        "corrupt": raw.get("corrupt") if isinstance(raw.get("corrupt"), str) else "",
    }


def _is_index(v: Any) -> bool:
    """A non-negative whole number, and `True` is not one.

    `isinstance(True, int)` is True in Python, so a file carrying
    `{"layer": true}` passes a bare int check and then indexes as layer 1.
    """
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def _patch_graph(doc: dict) -> dict:
    """The PATCHING graph of an untrusted file, or nothing.

    A different section from `graph`, and deliberately not the same key.
    `graph` carries a transcoder attribution graph THIS TOOL DID NOT COMPUTE
    and is gated on provenance saying so. This one carries a graph built here,
    out of `patch.path_trace`, from nothing but the model that was loaded. Two
    different objects from two different measurements, and merging them into
    one key would make the panel's whole disclaimer unreadable.

    Three rules specific to this section, and all three are about the claim
    rather than the bytes:

    An edge with no VERDICT is refused. The section's guarantee is that every
    drawn edge was run against the eight same-norm draws behind it -- that is
    what makes the picture a measurement instead of a ranking -- so a
    hand-written file whose edges carry a score and no `clears_control` is
    refused rather than rendered as though everything in it had passed.
    `clears_control: false` is a real verdict and travels; `null` does not.

    A graph with no SEEDING SENTENCE is refused. Edge count is quadratic in
    sites, so every such graph is a subset by construction; one whose rule for
    choosing edges has been stripped is a picture, not a measurement, and
    ROADMAP #52 makes printing the rule with the graph the condition of the
    feature existing at all.

    An edge naming a node the file does not carry is refused. It reaches the
    viewer as a lookup, and a dangling one draws as an edge from nowhere.

    `control_max` survives as None -- but only on an edge that has no verdict
    either, which the first rule already refuses. Where a verdict exists the
    control must be a real number, because 0.0 means "random noise at this
    site recovered nothing", which is a finding, and "we did not draw" is not.
    """
    raw = doc.get("patch_graph")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise SessionError(
            "this session's patching graph is not a set of fields",
        )

    nodes = raw.get("nodes")
    edges = raw.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise SessionError("this session's patching graph carries no nodes or edges")
    if len(nodes) > MAX_PATCH_GRAPH_NODES or len(edges) > MAX_PATCH_GRAPH_EDGES:
        raise SessionError(
            f"this session claims {len(nodes):,} nodes and {len(edges):,} "
            f"patched edges, above the {MAX_PATCH_GRAPH_NODES:,} and "
            f"{MAX_PATCH_GRAPH_EDGES:,} this reads. Every edge in a patching "
            f"graph costs eight control passes to earn, so a file this size is "
            f"not one anybody measured."
        )

    seeding = raw.get("seeding")
    if not (isinstance(seeding, str) and seeding.strip()):
        raise SessionError(
            "this session's patching graph does not say how its edges were "
            "chosen. Edge count is quadratic in sites, so every such graph is "
            "a subset — and one whose seeding rule has been stripped is a "
            "picture rather than a measurement."
        )

    clean_nodes: list[dict] = []
    ids: set[str] = set()
    for node in nodes:
        if not isinstance(node, dict):
            raise SessionError("this session has a graph node that is not fields")
        nid = node.get("id")
        if not isinstance(nid, str) or not nid or len(nid) > MAX_GRAPH_TEXT:
            raise SessionError("a node in this session's patching graph has no name")
        layer = node.get("layer")
        position = node.get("position")
        if not _is_index(layer) or not _is_index(position):
            raise SessionError(
                f"node {nid!r} does not say which layer and position it is, so "
                f"it cannot be placed."
            )
        head = node.get("head")
        if head is not None and not _is_index(head):
            raise SessionError(f"node {nid!r} names a head that is not an index")
        ids.add(nid)
        clean_nodes.append(
            {
                "id": nid,
                "layer": int(layer),
                "head": None if head is None else int(head),
                "position": int(position),
                "role": node.get("role") if isinstance(node.get("role"), str) else "",
                "depth": int(node["depth"]) if _is_index(node.get("depth")) else 0,
            }
        )

    clean_edges: list[dict] = []
    for edge in edges:
        if not isinstance(edge, dict):
            raise SessionError("this session has a graph edge that is not fields")
        source, target = edge.get("source"), edge.get("target")
        if source not in ids or target not in ids:
            raise SessionError(
                "this session's patching graph has an edge between nodes it "
                "does not carry, so the file is damaged."
            )

        recovery = edge.get("recovery")
        if (
            not isinstance(recovery, (int, float))
            or isinstance(recovery, bool)
            or not math.isfinite(recovery)
        ):
            raise SessionError(
                f"the edge {source} → {target} carries no finite recovery, so "
                f"there is nothing measured to draw."
            )

        clears = edge.get("clears_control")
        if not isinstance(clears, bool):
            raise SessionError(
                f"the edge {source} → {target} carries no verdict against its "
                f"controls. Every edge in a patching graph is drawn only "
                f"because it was tested against eight same-norm draws, and one "
                f"without them would render as though it had passed."
            )
        control = edge.get("control_max")
        if (
            not isinstance(control, (int, float))
            or isinstance(control, bool)
            or not math.isfinite(control)
        ):
            raise SessionError(
                f"the edge {source} → {target} claims a verdict with no control "
                f"behind it. A verdict without its reference is the bare claim "
                f"this section exists to replace."
            )
        draws = edge.get("control_draws")
        if not _is_index(draws) or draws < 1:
            raise SessionError(
                f"the edge {source} → {target} does not say how many control "
                f"draws it was tested against, so its verdict cannot be read."
            )

        # The individual draws, which the panel opens behind an edge. Optional
        # -- the verdict rests on `control_max` and is complete without them --
        # but when they are here they must AGREE with it. Two numbers answering
        # "what did random noise recover at this site" differently is a defect
        # even when each is individually plausible, and the one the reader
        # clicks would be the one that is wrong.
        raw_draws = edge.get("controls")
        controls: list[float] = []
        if raw_draws is not None:
            if not isinstance(raw_draws, list) or len(raw_draws) > MAX_CONTROL_DRAWS:
                raise SessionError(
                    f"the edge {source} → {target} carries a control list that "
                    f"is not one, or is longer than the {MAX_CONTROL_DRAWS} "
                    f"draws this reads."
                )
            for c in raw_draws:
                if (
                    not isinstance(c, (int, float))
                    or isinstance(c, bool)
                    or not math.isfinite(c)
                ):
                    raise SessionError(
                        f"the edge {source} → {target} has a control draw that "
                        f"is not a finite number."
                    )
                controls.append(float(c))
            if controls and abs(max(controls) - float(control)) > 1e-6:
                raise SessionError(
                    f"the edge {source} → {target} says its strongest control "
                    f"was {float(control):g} and carries draws whose strongest "
                    f"is {max(controls):g}. The verdict rests on one of those "
                    f"and the panel shows the other."
                )

        position_clears = edge.get("clears_position")
        clean_edges.append(
            {
                "source": source,
                "target": target,
                "recovery": float(recovery),
                "control_max": float(control),
                "controls": controls,
                "control_draws": int(draws),
                "clears_control": clears,
                # Three-valued on the way out. The shifted-position control is
                # a separate pass and a file may legitimately carry an edge
                # that never got one; coercing None to False here would turn
                # "not run" into "run, and failed".
                "clears_position": position_clears
                if isinstance(position_clears, bool)
                else None,
                "tested": True,
            }
        )

    return {
        "nodes": clean_nodes,
        "edges": clean_edges,
        "seeding": seeding[:MAX_GRAPH_TEXT],
        "means": raw.get("means")[:MAX_GRAPH_TEXT]
        if isinstance(raw.get("means"), str)
        else "",
        "clean": raw.get("clean")[:MAX_GRAPH_TEXT]
        if isinstance(raw.get("clean"), str)
        else "",
        "corrupt": raw.get("corrupt")[:MAX_GRAPH_TEXT]
        if isinstance(raw.get("corrupt"), str)
        else "",
        "depth": int(raw["depth"]) if _is_index(raw.get("depth")) else 0,
        "n_scored": int(raw["n_scored"]) if _is_index(raw.get("n_scored")) else 0,
        "n_pruned": int(raw["n_pruned"]) if _is_index(raw.get("n_pruned")) else 0,
        "passes": int(raw["passes"]) if _is_index(raw.get("passes")) else 0,
        "prune_threshold": float(raw["prune_threshold"])
        if isinstance(raw.get("prune_threshold"), (int, float))
        and not isinstance(raw.get("prune_threshold"), bool)
        and math.isfinite(raw["prune_threshold"])
        else 0.0,
        "prune_from": raw.get("prune_from")[:MAX_GRAPH_TEXT]
        if isinstance(raw.get("prune_from"), str)
        else "",
        # Names the receivers whose senders were never expanded. Dropping it
        # would turn "we stopped asking here" into "nothing wrote this".
        "frontier": [
            f for f in (raw.get("frontier") or []) if isinstance(f, str) and f in ids
        ]
        if isinstance(raw.get("frontier"), list)
        else [],
    }


def _boundary(doc: dict, n_tokens: int) -> int:
    """Where the prompt ends, from an untrusted file.

    Additive field: sessions written before it exists carry nothing, and 0
    means "unknown" — the panel then rests on no token rather than claiming
    the whole sequence is prompt. Anything outside [0, n_tokens] is a
    malformed claim about the file's own token list, so it is discarded
    rather than believed.
    """
    raw = doc.get("n_prompt")
    if not isinstance(raw, int) or isinstance(raw, bool):
        return 0
    return raw if 0 <= raw <= n_tokens else 0


def _quantise(matrix: Any) -> tuple[str, float]:
    """[S,S] floats -> (base64 uint8, scale). value ~= byte * scale.

    Takes a torch tensor or a list of lists. The tensor path is not an
    optimisation for its own sake: a 141-token, 24x14 export is 6.7 million
    values, and quantising those one at a time in Python takes long enough
    that a user would assume the button was broken.
    """
    if hasattr(matrix, "clamp") and hasattr(matrix, "contiguous"):
        import torch

        # float64, not float32: the two paths must agree bit for bit. A value
        # of exactly 0.1 lands on 26.0 in double and 25.999998 in single, and
        # truncation turns that into two different bytes -- so the "fast" path
        # would quietly export a different matrix than the portable one.
        m = matrix.detach().to(torch.float64)
        # NaN loses every comparison, so `max()` returns nan, the scale
        # becomes nan, and every cell quantises to 0: a smooth, plausible,
        # entirely blank heat map with nothing on screen saying the numbers
        # were never there. Refuse instead.
        if m.numel() and not bool(torch.isfinite(m).all()):
            # A Refusal and not a SessionError, alone among the raises in this
            # file. Every other one is a complaint about bytes the caller sent
            # and answers 422; this one is on the way OUT, and the request that
            # triggered it — GET /api/session/export?layer=0&head=0 — is
            # perfectly well formed. There is no parameter to correct. What is
            # wrong is the state of the numbers, and the sentence says what to
            # do about that instead, so it is a 409.
            raise Refusal(
                "this attention map contains non-finite values (nan or inf), "
                "so there is nothing honest to export. That usually means the "
                "model produced nan during the forward pass — the custom-model "
                "panel reports which layer first goes non-finite."
            )
        peak = float(m.max()) if m.numel() else 0.0
        scale = (peak / 255.0) if peak > 0 else 1.0
        q = (m / scale + 0.5).clamp(0, 255).to(torch.uint8).contiguous()
        return base64.b64encode(q.numpy().tobytes()).decode("ascii"), scale

    peak = 0.0
    for row in matrix:
        for v in row:
            # `math.isfinite` rather than `v != v or v in (inf, -inf)`: same
            # answer, one call, and it does not read as a typo. The NaN half of
            # that idiom is correct but every reader has to stop and remember
            # why, and a static analyser flags it as comparing a value to
            # itself.
            if not math.isfinite(v):
                # The portable path's half of the check above, and a Refusal
                # for the same reason: the export cannot be taken, the request
                # asking for it was fine.
                raise Refusal(
                    "this attention map contains non-finite values (nan or "
                    "inf), so there is nothing honest to export."
                )
            if v > peak:
                peak = v
    scale = (peak / 255.0) if peak > 0 else 1.0
    flat = bytearray()
    for row in matrix:
        for v in row:
            q = int(v / scale + 0.5) if scale else 0
            flat.append(255 if q > 255 else (0 if q < 0 else q))
    return base64.b64encode(bytes(flat)).decode("ascii"), scale


def _dequantise(blob: str, scale: float, n: int) -> list[list[float]]:
    raw = base64.b64decode(blob)
    if len(raw) != n * n:
        raise SessionError(
            f"attention block is {len(raw)} bytes but the token count says "
            f"{n}x{n}={n * n} — the file is truncated or not a session"
        )
    return [[round(raw[r * n + c] * scale, 5) for c in range(n)] for r in range(n)]


@dataclass
class Session:
    """A recorded analysis. Read-only by construction."""

    meta: dict = field(default_factory=dict)
    tokens: list[str] = field(default_factory=list)
    generation: str = ""
    prompt: str = ""
    # (layer, head) -> {"q": base64, "scale": float}
    attention: dict[str, dict] = field(default_factory=dict)
    lens: list[dict] = field(default_factory=list)
    # `final`, `settled_at`, `n_layers` and the reliability block that come
    # with the trajectory. Additive and separate because `lens` is an existing
    # key with a declared list type, and scalars do not fit in a list.
    lens_info: dict = field(default_factory=dict)
    n_layers: int = 0
    n_heads: int = 0
    # Where the prompt ends. Additive: a file written before this carries 0,
    # which every reader must treat as "unknown" and not as "all prompt".
    n_prompt: int = 0
    # An activation-patching trace, when one was run. A `.mri` carried the
    # attention and the logit lens and nothing else, so the one result in this
    # tool that is CAUSAL rather than correlational -- "the answer is decided
    # at layer 15, position 4" -- was the one result you could not send to
    # anybody. Optional and additive: a file written before this has no
    # `patch` key, and an older reader ignores the key rather than failing on
    # it, which is why the format version does not move.
    patch: dict = field(default_factory=dict)
    # An attribution graph THIS TOOL DID NOT COMPUTE, read from a
    # circuit-tracer file. Optional and additive like `patch`, so a file
    # written before it has no `graph` key and an older reader ignores it
    # rather than failing -- which is why the format version does not move.
    #
    # Its `provenance` is not optional: see `_graph`.
    graph: dict = field(default_factory=dict)
    # A patching graph THIS TOOL DID compute, walked backwards from the sites
    # the node grid flagged. A SEPARATE KEY from `graph` above on purpose: that
    # one is somebody else's transcoder attribution graph and this one is ours,
    # from a different measurement, and a viewer that could not tell them apart
    # would make the disclaimer on both unreadable. Optional and additive, so
    # the format version does not move.
    #
    # Its `seeding` sentence and its per-edge verdicts are not optional: see
    # `_patch_graph`.
    patch_graph: dict = field(default_factory=dict)
    # What produced each number in this file: model, revision, dtype, device,
    # attention implementation, seed, tokenizer and prompt hashes. Optional and
    # additive like `patch` and `graph`, so the format version does not move
    # and an older reader ignores the key.
    #
    # This is what makes a finding auditable after it has left the machine
    # that took it. Every panel already printed its setup in prose for whoever
    # was looking at the screen at the time; none of that survived an export.
    receipts: list = field(default_factory=list)
    # The head ranking, when one was measured against this run: which head
    # moves the answer most, and by how many nats. Optional and additive like
    # `patch`. It carries `noise_floor_kl` with it, because a score without
    # the floor it was measured against cannot be told from arithmetic.
    ranking: dict = field(default_factory=dict)
    # Behavioural labels for each head, each with the margin and null it
    # cleared. Optional and additive. A label without its evidence is refused
    # rather than carried, because the evidence is the whole point.
    head_types: dict = field(default_factory=dict)
    # Whether the answer came from the attached document or from the weights:
    # a dependence score and an attention share per passage. Optional and
    # additive like `patch`.
    #
    # Worth carrying for the same reason `patch` is. The claim "the model
    # answered from its weights, not from the document you gave it" is one
    # somebody wants to SHOW a colleague, and until now it was one of the few
    # findings in this tool that could not leave the machine that took it.
    # A recording cannot re-run it -- masking a passage needs the model -- but
    # it can carry what was measured.
    ground: dict = field(default_factory=dict)
    # A finetune-vs-base comparison over a prompt set. Optional and additive.
    #
    # UNLIKE every other section here, this one is not necessarily about the
    # model the file describes: it names its own two sides, and a comparison
    # of two checkpoints can legitimately ride in a `.mri` taken on a third
    # model. `_model_diff` requires both names for exactly that reason.
    model_diff: dict = field(default_factory=dict)
    # Robot-policy findings: the camera frame, per-layer attention, and the
    # causal occlusion map with its control band — plus exactly which policy,
    # dataset, episode, timestep and camera produced them. Optional and
    # additive.
    #
    # There is no portable, no-account artifact for robot-policy internals
    # anywhere: Foxglove archived its open-source Studio, Rerun's `.rrd`
    # carries what the robot recorded rather than what the network computed,
    # and HF Spaces need an upload and an account.
    vla: dict = field(default_factory=dict)

    # The agent run this analysis belongs to: the timeline, and which step
    # failed. Optional and additive like `patch`.
    #
    # This is the half no hosted platform can ship. Every competitor's share
    # artefact is a link into their own trace UI, which dies when the account
    # lapses — Helicone went into maintenance mode in March 2026 and Langfuse
    # changed owners in January. A recipient opens this with nothing installed,
    # clicks the failing tool call, and lands in the attention view of the
    # generation that produced the bad argument, on a machine with no GPU.
    trace: dict = field(default_factory=dict)

    # -------------------------------------------------- the runtime's shape
    def attention_meta(self) -> dict:
        return {
            "available": bool(self.attention),
            "n_prompt": self.n_prompt,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_tokens": len(self.tokens),
            "replay": True,
        }

    def has_patch(self) -> bool:
        return bool(self.patch.get("grids"))

    def has_head_types(self) -> bool:
        return bool(self.head_types.get("labels"))

    def has_ranking(self) -> bool:
        return bool(self.ranking.get("ranked"))

    def has_ground(self) -> bool:
        return bool(self.ground.get("chunks"))

    def has_model_diff(self) -> bool:
        return bool(self.model_diff.get("prompts"))

    def has_vla(self) -> bool:
        return bool(self.vla.get("provenance"))

    def has_trace(self) -> bool:
        return bool(self.trace.get("steps"))

    def failing_step(self) -> dict | None:
        """The step this bundle was built around, when it names one.

        None means "no step was singled out", not "the step is missing":
        `_trace` refuses a `step_ref` naming a step the file does not carry,
        so a ref that survives parsing always resolves.
        """
        ref = self.trace.get("step_ref")
        if not ref:
            return None
        return next(
            (s for s in self.trace.get("steps", []) if s.get("id") == ref), None
        )

    def has_graph(self) -> bool:
        return bool(self.graph.get("edges") or self.graph.get("n_nodes"))

    def has_patch_graph(self) -> bool:
        return bool(self.patch_graph.get("edges"))

    def attention_slice(self, layer: int, head: int) -> dict:
        key = f"{layer}:{head}"
        block = self.attention.get(key)
        if block is None:
            have = sorted(self.attention)[:6]
            raise SessionError(
                f"this session does not contain layer {layer} head {head}. "
                f"It has {len(self.attention)} slices, e.g. {', '.join(have)}. "
                "A session stores what was captured, not every combination."
            )
        return {
            "layer": layer,
            "head": head,
            # So a replayed session rests on the same token a live one does.
            "n_prompt": self.n_prompt,
            "tokens": self.tokens,
            "matrix": _dequantise(block["q"], block["scale"], len(self.tokens)),
            "replay": True,
        }


def build(
    *,
    model_id: str | None,
    device: str | None,
    dtype: str | None,
    n_params: int | None,
    tokens: list[str],
    prompt: str,
    generation: str,
    attention: dict[tuple[int, int], list[list[float]]],
    n_layers: int,
    n_heads: int,
    n_prompt: int = 0,
    lens: list[dict] | None = None,
    lens_info: dict | None = None,
    note: str = "",
    scope: str = "",
    patch: dict | None = None,
    graph: dict | None = None,
    patch_graph: dict | None = None,
    ranking: dict | None = None,
    head_types: dict | None = None,
    ground: dict | None = None,
    model_diff: dict | None = None,
    vla: dict | None = None,
    trace: dict | None = None,
    step_ref: str = "",
    receipts: list | None = None,
) -> bytes:
    """Serialise one analysis into a gzipped `.mri`.

    When `trace` is given, the run and the analysis ship together and BOTH
    halves go through `bundle.prepare` first — the recorder's redaction runs
    at delivery, which is behind us by the time steps come out of the store,
    and a document that arrived by import or OTLP ingest never went through it
    at all. See `bundle.py`.
    """
    from . import __version__
    from . import bundle as bundle_mod

    # BEFORE anything is written. Redacting after the document is assembled
    # would mean the only difference between a safe file and an unsafe one is
    # whether a later step remembered to run — and the prompt and generation
    # go through it whether or not there is a trace, because a credential
    # pasted into a prompt is a credential either way.
    clean_trace, prompt, generation, _leaving = bundle_mod.prepare(
        trace, prompt=prompt, generation=generation, step_ref=step_ref
    )

    blocks: dict[str, dict] = {}
    for (layer, head), matrix in attention.items():
        q, scale = _quantise(matrix)
        blocks[f"{layer}:{head}"] = {"q": q, "scale": scale}

    doc = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "modelmri": __version__,
        "meta": {
            "model": model_id,
            "device": device,
            "dtype": dtype,
            "n_params": n_params,
            "note": note,
            # What was captured, spelled out. A session that quietly holds a
            # slice of the cube looks identical to one that holds all of it
            # until you click the head it does not have.
            "scope": scope,
            # Stated, not implied. The arcs are drawn from relative weight, so
            # this is lossless for what you see and lossy for what you'd
            # compute — anyone doing arithmetic on these numbers should know.
            "precision": "attention quantised to uint8 against each matrix's "
            "own maximum; about 0.4% of that maximum per step",
        },
        "prompt": prompt,
        "generation": generation,
        "tokens": tokens,
        "n_prompt": int(n_prompt or 0),
        "n_layers": n_layers,
        "n_heads": n_heads,
        "attention": blocks,
        "lens": lens or [],
    }
    # Only when there is one. An empty key would make every file claim a
    # patching section and every reader render an empty one.
    if patch and patch.get("grids"):
        doc["patch"] = patch
    # Same additive rule: written only when there is one, so a session without
    # a graph carries no empty section for a reader to render as one.
    #
    # But a graph WITHOUT provenance is refused rather than quietly dropped.
    # Dropping it would hand back a file the caller believes carries a graph
    # and which does not, silently -- and the reason it would be dropped is
    # the one thing this section exists to guarantee. `parse` refuses the
    # same shape; a writer that is laxer than the reader is a way to build
    # files nobody can open.
    if graph:
        _claim = (graph.get("provenance") or {}).get("measured_by")
        if not (isinstance(_claim, str) and _claim.strip()):
            raise SessionError(
                "an attribution graph needs provenance saying who computed "
                "it. ModelMRI did not, and a session that renders one without "
                "saying so is the confusion this section exists to prevent."
            )
        doc["graph"] = graph
    # Through the READER's validator like every additive section below, which
    # here means a graph whose seeding rule was left out, or one carrying an
    # edge with no verdict behind it, is refused at WRITE time. Both are the
    # section's whole guarantee, and a writer laxer than the reader would build
    # files this tool signs its name to and then cannot open.
    if patch_graph and patch_graph.get("edges"):
        doc["patch_graph"] = _patch_graph({"patch_graph": patch_graph})
    # Same additive rule. Written through the same validator the READER uses,
    # not straight from the caller: a writer laxer than the reader is how you
    # build files nobody can open, and `_graph` records that lesson two
    # sections above. This also means a receipt is bounded on the way out, so
    # a hostile `request` block cannot be smuggled into a file this tool signs
    # its own name to.
    # Additive like `patch`, and written through the READER's validator for
    # the same reason `graph` is: a writer laxer than the reader is how you
    # build files nobody can open.
    if head_types and head_types.get("labels"):
        doc["head_types"] = _head_types({"head_types": head_types})
    if lens_info and (lens or []):
        doc["lens_info"] = lens_info
    if ranking and ranking.get("ranked"):
        doc["ranking"] = _ranking({"ranking": ranking})
    # Through the READER's validator, like every additive section above it: a
    # writer laxer than the reader is how you build files nobody can open, and
    # it means a grounding row that asserts `depended_on` without a floor is
    # refused at WRITE time rather than reaching somebody else's viewer.
    if ground and ground.get("chunks"):
        doc["ground"] = _ground({"ground": ground})
    # Through the reader's validator like every additive section above it.
    if model_diff and model_diff.get("prompts"):
        doc["model_diff"] = _model_diff({"model_diff": model_diff})
    # Through the reader's validator like every additive section above it, so
    # a robot section missing its provenance is refused at WRITE time rather
    # than reaching somebody else's viewer.
    if vla and vla.get("provenance"):
        doc["vla"] = _vla({"vla": vla})
    # Through the reader's validator like every additive section above it, so
    # a bundle whose step_ref names a step it does not carry is refused at
    # WRITE time rather than opening as a dead link in somebody's viewer.
    if clean_trace and clean_trace.get("steps"):
        doc["trace"] = _trace({"trace": clean_trace})
    if receipts:
        doc["receipts"] = _receipts_mod.parse(receipts)
    # allow_nan=False. The default emits a bare `NaN`/`Infinity` token, which
    # is not JSON: Python reads it back, the viewer's `JSON.parse` does not.
    # `modelmri open` printed "forwardable" for a file it could not itself
    # reopen, because circuit.py deliberately KEEPS non-finite weights and
    # `topk` sorts NaN to the front of the strongest edges.
    try:
        body = json.dumps(doc, separators=(",", ":"), allow_nan=False)
    except ValueError as err:
        raise SessionError(
            "this session contains a non-finite number (nan or inf), which "
            "cannot be written as JSON. A file carrying one is readable by "
            "Python and rejected by every browser, so it is refused here "
            "rather than written as something that will not open."
        ) from err
    return gzip.compress(body.encode("utf-8"), 6)


def parse(data: bytes) -> Session:
    """Read a `.mri`, refusing anything that is not one, with the reason.

    Every bound below exists because this function takes bytes a stranger
    sent you. The whole premise of the format is that it travels.
    """
    if not data:
        raise SessionError("the file is empty")
    if len(data) > MAX_FILE:
        raise SessionError(
            f"this file is {len(data) / 1e6:,.0f} MB. A session is the "
            f"observation, not the model — a large one is tens of megabytes, "
            f"so this is almost certainly not one."
        )
    try:
        raw = _inflate(data) if data[:2] == b"\x1f\x8b" else data
    except (OSError, EOFError, zlib.error) as err:
        # NOT `{err}`. errors.py forbids interpolating a caught exception's
        # text into a published message, and this is the one function whose
        # docstring says it takes bytes a stranger sent you. zlib's own strings
        # are harmless C literals today, but OSError carries its `filename`
        # when set — the exact shape that leaked absolute paths to the browser
        # before, rebuilt one arm over. `from err` keeps the cause for the log.
        raise SessionError(
            "this file starts like a gzip but could not be decompressed — it "
            "is damaged, or it is not a .mri"
        ) from err

    try:
        doc: Any = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as err:
        # Deliberately not the exception class name. "JSONDecodeError" told a
        # user nothing except that something internal had gone wrong, which is
        # the wrong impression: the file is fine, it is just not one of ours.
        raise SessionError(
            "this file is not a ModelMRI session — a .mri is written by "
            "'Share this view' in the attention panel"
        ) from err

    if not isinstance(doc, dict) or doc.get("format") != FORMAT:
        raise SessionError(
            "this is not a ModelMRI session file (no 'modelmri-session' marker)"
        )
    version = doc.get("format_version")
    # Split, because the old single check interpolated the value BEFORE
    # establishing it was a number. Measured on a hand-made file: a
    # `format_version` of "<img src=x onerror=alert(1)>ATTACKER" came back
    # verbatim in the 422 body, and a dict came back as a Python repr. Not an
    # XSS — the body is JSON and React renders it as a text node — but it is
    # attacker-supplied content reflected by the one function documented as
    # taking bytes a stranger sent, and `n_layers`/`n_heads` and `_boundary`
    # in this same file already refuse a non-int without echoing it.
    if not isinstance(version, int) or isinstance(version, bool):
        raise SessionError(
            "this session does not say which format version it is, so it is "
            "damaged or it is not a .mri"
        )
    if version > FORMAT_VERSION:
        raise SessionError(
            f"this session is format version {version}, and this ModelMRI "
            f"reads up to {FORMAT_VERSION}. Upgrade with `pip install -U modelmri`."
        )

    tokens = doc.get("tokens") or []
    if not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens):
        raise SessionError("the session's token list is missing or malformed")

    # Cells, not tokens. Cost is n^2 per slice, so the token count is the
    # wrong thing to bound: a 31 KB file claiming 10,000 tokens asks for a
    # hundred million Python floats the moment a layer/head dial is clicked,
    # and the identical loop runs in the browser viewer for whoever you
    # forwarded it to.
    size = len(tokens)
    if size * size > MAX_CELLS:
        raise SessionError(
            f"this session claims {size:,} tokens, which is {size * size / 1e6:,.0f} "
            f"million attention cells per map — more than ModelMRI will render. "
            f"The file is either damaged or not a session."
        )

    # Validated before anything is built from it. `attention` is indexed by
    # string keys and iterated by the panels; a list or a dict with non-string
    # keys got past this and turned every later request into a 500.
    attention = doc.get("attention")
    if attention is None:
        attention = {}
    # `or {}` here turned a malformed value into an empty one: a file whose
    # attention was `[]` opened as a session with no maps rather than being
    # refused, which is a damaged file presented as an intact empty one.
    if not isinstance(attention, dict) or not all(
        isinstance(k, str) and isinstance(v, dict) for k, v in attention.items()
    ):
        raise SessionError("the session's attention index is missing or malformed")

    # And each block has the two fields it will be READ for. The check above
    # stops at "is a dict", so `{"0:0": {"layer": 0}}` parsed cleanly and then
    # `_dequantise(block["q"], block["scale"], ...)` raised KeyError on the
    # first request for that head -- a 500, from a file the reader was told
    # had opened. Exactly the failure the comment above describes, one level
    # further in. `verify.dequantise` reads the same two keys, so validating
    # here covers both.
    for key, block in attention.items():
        scale = block.get("scale")
        if not isinstance(block.get("q"), str) or (
            not isinstance(scale, (int, float)) or isinstance(scale, bool)
        ):
            raise SessionError(
                f"the session's attention block {key!r} has no stored matrix "
                f"and scale to read, so it cannot be drawn"
            )

    # These reach the UI as loop bounds. A float, a negative, or 1e20 is not
    # a shape — it is a hang or a crash in whatever renders it.
    counts = {}
    for key in ("n_layers", "n_heads"):
        value = doc.get(key) or 0
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 0 <= value <= MAX_DIM
        ):
            raise SessionError(f"the session's {key} is not a sensible number")
        counts[key] = value

    # `meta` is spread below, and `**` on anything that is not a mapping raises
    # a bare TypeError — measured, a `.mri` carrying `"meta": "hi"` gave
    # `TypeError: 'str' object is not a mapping`, which is not a BadRequest, so
    # it fell past the 409 and 422 arms to the generic 500. Every other
    # untrusted field in this function is type-checked; this one was spread.
    meta = doc.get("meta")
    if meta is not None and not isinstance(meta, dict):
        raise SessionError(
            "this session's metadata is not a set of fields, so the file is "
            "damaged or it is not a .mri"
        )

    # Once, not once per field: `_lens` validates every layer and every
    # probability, and calling it twice would do that work twice on a file a
    # stranger sent.
    lens_rows, lens_info = _lens(doc)
    return Session(
        meta={
            **(meta or {}),
            "created_at": doc.get("created_at"),
            "modelmri": doc.get("modelmri"),
        },
        tokens=tokens,
        prompt=doc.get("prompt") or "",
        generation=doc.get("generation") or "",
        attention=attention,
        lens=lens_rows,
        lens_info=lens_info,
        n_prompt=_boundary(doc, len(tokens)),
        n_layers=counts["n_layers"],
        n_heads=counts["n_heads"],
        patch=_patch(doc),
        trace=_trace(doc),
        graph=_graph(doc),
        patch_graph=_patch_graph(doc),
        ranking=_ranking(doc),
        head_types=_head_types(doc),
        ground=_ground(doc),
        model_diff=_model_diff(doc),
        vla=_vla(doc),
        # Validated in `receipts.parse` rather than here: the rules belong
        # beside the writer that produces them, and this module already has
        # more section validators than is comfortable.
        receipts=_receipts_mod.parse(doc.get("receipts")),
    )
