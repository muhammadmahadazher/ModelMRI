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

# One row per head. A whole-model sweep is n_layers x n_heads: 144 on gpt2,
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
        if weight != weight or weight in (float("inf"), float("-inf")):
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
    # measures the three agreeing only weakly (Spearman 0.34-0.47 on gpt2
    # layer 0), so a ranking that does not name its baseline cannot be
    # compared against one that does.
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

    def has_graph(self) -> bool:
        return bool(self.graph.get("edges") or self.graph.get("n_nodes"))

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
    ranking: dict | None = None,
    head_types: dict | None = None,
    receipts: list | None = None,
) -> bytes:
    """Serialise one analysis into a gzipped `.mri`."""
    from . import __version__

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
        graph=_graph(doc),
        ranking=_ranking(doc),
        head_types=_head_types(doc),
        # Validated in `receipts.parse` rather than here: the rules belong
        # beside the writer that produces them, and this module already has
        # more section validators than is comfortable.
        receipts=_receipts_mod.parse(doc.get("receipts")),
    )
