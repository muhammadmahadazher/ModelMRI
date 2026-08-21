"""A PATCHING graph: what wrote what, across a whole prompt.

`patch.trace` answers "does this site matter" one cell at a time, and
`patch.path_trace` answers "what wrote into this one receiver". Neither answers
the question people actually open a circuit view for -- what wrote the thing
that wrote the answer -- because that needs the second question asked again of
its own senders.

This walks it backwards. Seed from the sites the node grid already flagged,
run `path_trace` into each, take the senders that beat their controls, and ask
the same question of them. Nodes are (component, layer, position); edges are
the signed recovery one sender contributes to one receiver.

## It is a patching graph, and the name matters

circuit-tracer's attribution graphs are built from transcoders, which exist for
a handful of models and whose gemma-2-2b set does not fit 8 GB. This is a
different object built from a different measurement, out of nothing but the
model already loaded -- and it is called a patching graph everywhere it
appears, in the payload and on the panel, rather than borrowing the more famous
name for a thing it is not. `circuit.py` READS one of theirs; this computes one
of ours, and the two must never be presented as the same artefact.

## The seeding rule is part of the answer

Edge count is quadratic in sites. Anything drawable is therefore a subset, and
a graph whose edges were chosen by an undisclosed rule is a picture rather than
a measurement. So `seeding` and `prune` travel in the payload and are printed
with the graph: which receivers were expanded, how many senders each offered,
what the threshold was, and how many edges were never looked at.

## Every drawn edge was controlled; one that FAILED its control is still drawn

Two different rules that are easy to confuse. Every edge in the graph carries
the same eight same-norm draws the node grid uses, because an edge with a score
and no verdict has nothing behind it to click -- so a sender `path_trace` never
controlled is pruned and counted, never drawn. But an edge that WAS controlled
and lost is kept and marked `clears_control: false`: dropping that one silently
would turn "we tested this and it did not survive" into "we never saw this",
and those are different findings. The panel draws them differently; it does not
hide them.

Controlling every drawn edge also decouples the picture's SIZE from the
arithmetic. Pruning on recovery alone leaves the edge count set by whatever the
resolution happens to be on this pair, and that number is not a constant of the
dtype: MEASURED in bfloat16, Qwen3-1.7B reads 0.006231 on the Eiffel /
Colosseum pair, and the same figure on another model can sit orders of
magnitude away in the same number format -- because the resolution is one
representable step of the GAP between the two runs' answers, and the gap
differs per model and per pair. Bounding the graph by what was controlled instead makes its size
a function of `max_controlled`, which is a number this module chose and
reports. The resolution still cuts below it, because a recovery the arithmetic
cannot separate from zero is not a measurement -- it is just no longer what
decides how big the picture is.

## Depth is bounded and the bound is reported

Each level costs one `path_trace` per receiver, and each of those is
`n_senders + controls` forward passes. Two levels of a handful of receivers is
a real wait on a laptop, which is why `estimate` exists and why `depth`,
`max_receivers` and the passes actually spent are all in the payload.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .errors import BadRequest

# How deep the backward walk goes by default. 1 is `path_trace` with extra
# steps; 3 is a wait nobody sits through on the machines this targets.
DEFAULT_DEPTH = 2

# Receivers expanded per level. The seeds are the node grid's strongest sites,
# and beyond a handful the graph stops being readable before it stops being
# affordable.
DEFAULT_MAX_RECEIVERS = 4

# Above this the graph is refused rather than drawn. A picture with a thousand
# edges is not a finding, and the refusal names the number so the caller can
# raise the threshold or narrow the seeds deliberately.
MAX_EDGES = 400


class GraphError(BadRequest):
    """This graph cannot be built honestly, and the message says why."""


@dataclass
class Node:
    """One (component, layer, position) the graph mentions."""

    id: str
    layer: int
    # None for an MLP or a residual site; the head index otherwise.
    head: int | None
    position: int
    # "seed" for a site the node grid flagged, "sender" for one reached by
    # walking backwards from it.
    role: str = "sender"
    # How far from a seed. 0 for a seed itself.
    depth: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Edge:
    """One sender's signed contribution to one receiver."""

    source: str
    target: str
    # The SAME recovery fraction the node grid and `path_trace` report, so an
    # edge here and a cell there are on one scale.
    recovery: float
    # The strongest of `control_draws` same-norm random patches at this site.
    # None when this edge was scored but never controlled -- NOT 0.0, which
    # would read as "random noise here does nothing".
    control_max: float | None = None
    # EVERY draw, not just the strongest. ROADMAP #52 asks for the eight
    # controls to be clickable behind an edge, and the SPREAD is the finding:
    # a verdict quoted as "beat 0.28" reads differently once you can see that
    # seven of the eight were nowhere near it.
    controls: list[float] = field(default_factory=list)
    control_draws: int = 0
    # None when untested, for the same reason. False is a real verdict.
    clears_control: bool | None = None
    clears_position: bool | None = None

    @property
    def tested(self) -> bool:
        return self.clears_control is not None

    def to_dict(self) -> dict:
        out = asdict(self)
        out["tested"] = self.tested
        return out


@dataclass
class PatchGraph:
    """Everything the walk found, and everything it did not look at."""

    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    clean: str = ""
    corrupt: str = ""
    answer: str = ""
    depth: int = 0
    max_receivers: int = 0
    n_receivers_expanded: int = 0
    # Senders scored across every expansion, against the edges kept. The
    # difference is the prune, and it is reported rather than implied.
    n_scored: int = 0
    n_pruned: int = 0
    passes: int = 0
    seconds: float = 0.0
    # The recovery below which an edge was not kept, and where it came from.
    prune_threshold: float = 0.0
    prune_from: str = ""
    # Receivers that had senders left to expand when the depth ran out. NOT an
    # empty graph edge: the walk stopped, and saying where is the difference
    # between "nothing wrote this" and "we did not ask".
    frontier: list[str] = field(default_factory=list)

    @property
    def weak(self) -> list[Edge]:
        """Edges tested against their controls and beaten by them."""
        return [e for e in self.edges if e.clears_control is False]

    @property
    def untested(self) -> list[Edge]:
        return [e for e in self.edges if e.clears_control is None]

    def to_dict(self) -> dict:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "clean": self.clean,
            "corrupt": self.corrupt,
            "answer": self.answer,
            "depth": self.depth,
            "max_receivers": self.max_receivers,
            "n_receivers_expanded": self.n_receivers_expanded,
            "n_nodes": len(self.nodes),
            "n_edges": len(self.edges),
            "n_scored": self.n_scored,
            "n_pruned": self.n_pruned,
            "n_weak": len(self.weak),
            "n_untested": len(self.untested),
            "passes": self.passes,
            "seconds": self.seconds,
            "prune_threshold": self.prune_threshold,
            "prune_from": self.prune_from,
            "frontier": list(self.frontier),
            "seeding": self.seeding(),
            "means": self.means(),
        }

    def seeding(self) -> str:
        """Which edges were considered at all. Never omitted."""
        return (
            f"Seeded from the {self.n_receivers_expanded} strongest sites the "
            f"node grid flagged and walked back {self.depth} level(s), at most "
            f"{self.max_receivers} receivers per level. Every attention head "
            f"and MLP earlier than each receiver was scored into it — "
            f"{self.n_scored} senders in total — and {self.n_pruned} were "
            f"pruned: an edge is drawn only if it was CONTROLLED (the eight "
            f"same-norm draws run on the strongest few per receiver, so an "
            f"uncontrolled edge has nothing behind it to click) and only if "
            f"its recovery clears {self.prune_threshold:g} "
            f"({self.prune_from}). Edge count is quadratic in sites, so this "
            f"is a subset by construction: it is the strongest edges LOOKED "
            f"AT, not the strongest edges there are."
        )

    def means(self) -> str:
        parts = [
            f"A PATCHING graph, not a transcoder attribution graph. "
            f"{len(self.nodes)} nodes and {len(self.edges)} edges over "
            f"{self.clean!r} against {self.corrupt!r}, scored by the same "
            f"recovery fraction the node grid reports, so an edge here and a "
            f"cell there can be read together."
        ]
        if self.weak:
            parts.append(
                f"{len(self.weak)} edge(s) were tested against eight same-norm "
                f"draws and did NOT beat them. They are drawn differently "
                f"rather than dropped: 'we tested this and it did not survive' "
                f"and 'we never saw this' are different findings."
            )
        if self.n_pruned:
            parts.append(
                f"{self.n_pruned} scored sender(s) are NOT drawn: they were "
                f"never controlled, or their recovery does not clear what this "
                f"dtype can express. Absence from the picture is not a verdict "
                f"against them — it is the absence of one."
            )
        if self.frontier:
            parts.append(
                f"The walk stopped at {len(self.frontier)} receiver(s) with "
                f"senders still unexpanded ({', '.join(self.frontier[:4])}). "
                f"Nothing beyond them was asked about."
            )
        parts.append(
            "DIRECTION ONLY, AND NO Q/K/V SPLIT. An edge says a sender wrote "
            "what a receiver reads, inherited verbatim from `path_trace`'s "
            "scope — not which of the receiver's query, key or value it "
            "reached."
        )
        return " ".join(parts)


def node_id(layer: int, head: int | None, position: int) -> str:
    """The stable name for one (component, layer, position)."""
    where = f"L{layer}H{head}" if head is not None else f"L{layer} MLP"
    return f"{where}@{position}"


def estimate(
    n_layers: int,
    n_heads: int,
    *,
    depth: int = DEFAULT_DEPTH,
    max_receivers: int = DEFAULT_MAX_RECEIVERS,
    draws: int = 8,
    max_controlled: int = 12,
) -> dict:
    """What the walk will cost, before anybody waits for it.

    Every receiver costs one `path_trace`: one scoring pass per earlier
    component, plus `draws` controls and one shifted-position pass for each of
    the strongest `max_controlled`. Nobody should discover that by waiting,
    which is the same reason `patch.estimate` and `vla_sweep.estimate` exist.
    """
    if n_layers <= 0 or n_heads <= 0:
        raise GraphError(
            f"this model's config states {n_layers} layers and {n_heads} "
            f"attention heads, so the cost of a walk cannot be projected. "
            f"Running it blind is what this projection exists to prevent."
        )
    # Mean senders per receiver: a receiver at layer L sees L*(heads+1)
    # earlier components, and averaged over the stack that is about half.
    per_receiver = max(1, (n_layers * (n_heads + 1)) // 2)
    per_receiver += max_controlled * (draws + 1)
    # `build` seeds with `ranked[:max_receivers]` and then caps every level at
    # `next_frontier[:max_receivers]`, so it expands up to `max_receivers` per
    # level for `depth` levels. The old formula here was
    # `sum(min(m, m**level) ...)`, which makes level 0 a SINGLE receiver and
    # under-quoted every walk: at depth 2 with 2 receivers it projected 3 where
    # the walk expanded 4 (MEASURED on Qwen3-1.7B), and at depth 2 with 4 it
    # projected 5 against 8. A preflight that under-quotes is worse than no
    # preflight, which is this function's whole reason to exist.
    #
    # An UPPER bound, and it says so below: a walk stops early when the seeds
    # or the surviving senders run out, and erring high is the safe direction
    # for a number somebody decides to wait on.
    receivers = max_receivers * depth
    return {
        "receivers": receivers,
        "passes_per_receiver": per_receiver,
        "passes_total": receivers * per_receiver,
        "depth": depth,
        "max_receivers": max_receivers,
        # Stated, not implied. A walk that runs out of seeds or of senders that
        # beat their controls expands fewer, and a reader comparing this to the
        # `passes` a finished graph reports should know which way they differ.
        "bound": (
            f"an upper bound: at most {max_receivers} receivers per level for "
            f"{depth} level(s). A walk that runs out of seeds, or of senders "
            f"that beat their controls, expands fewer and costs less."
        ),
        # No seconds. A pass costs what it costs on THIS machine, and
        # `ablate.py` measured the same model between 12 and 71 ms/pass across
        # sessions on one card -- a figure in seconds would be fiction.
        "seconds": None,
        "seconds_from": "passes are portable; milliseconds per pass are not",
    }


def build(
    trace_fn,
    node_sites: list[dict],
    *,
    depth: int = DEFAULT_DEPTH,
    max_receivers: int = DEFAULT_MAX_RECEIVERS,
    max_edges: int = MAX_EDGES,
    clean: str = "",
    corrupt: str = "",
    answer: str = "",
) -> PatchGraph:
    """Walk backwards from the node grid's sites, one `path_trace` per receiver.

    `trace_fn(layer, position) -> dict` is `patch.path_trace` with the model
    already bound, so this module never touches a model and stays testable
    without one. `node_sites` are the grid's own rows: whatever it flagged is
    what gets expanded, rather than a threshold invented here.
    """
    if depth < 1:
        raise GraphError("a walk of zero levels has nothing to draw.")
    if not node_sites:
        raise GraphError(
            "the node grid flagged no sites, so there is nothing to walk back "
            "from. Run a patch trace first and pick a cell."
        )

    graph = PatchGraph(
        clean=clean,
        corrupt=corrupt,
        answer=answer,
        depth=depth,
        max_receivers=max_receivers,
    )

    # Seeds: the grid's strongest sites, its own ordering. Sites that cleared
    # their control come first, because the grid already has a verdict on them
    # and expanding one it rejected would spend a whole `path_trace` on a cell
    # the grid says is noise.
    ranked = sorted(
        node_sites,
        key=lambda s: (not s.get("clears_control"), -float(s.get("recovery") or 0.0)),
    )

    seen_nodes: dict[str, Node] = {}
    seen_edges: set[tuple[str, str]] = set()

    def remember(layer, head, position, *, role, level) -> str:
        key = node_id(int(layer), head, int(position))
        if key not in seen_nodes:
            seen_nodes[key] = Node(
                id=key,
                layer=int(layer),
                head=head,
                position=int(position),
                role=role,
                depth=level,
            )
        return key

    frontier: list[tuple[int, int, str]] = []
    # Same one-entry-per-receiver rule as the level loop below, and it bites
    # here first: the node grid scores THREE components at every cell, so its
    # flagged sites routinely name one (layer, position) more than once --
    # `resid` and `mlp` at the same site are two rows and one receiver. Taking
    # `ranked[:max_receivers]` off the raw list therefore spent a whole seed
    # slot, and a whole `path_trace`, re-answering a question already asked.
    seeded: set[tuple[int, int]] = set()
    for site in ranked:
        if len(frontier) >= max_receivers:
            break
        layer, position = int(site.get("layer", 0)), int(site.get("position", 0))
        if (layer, position) in seeded:
            continue
        seeded.add((layer, position))
        key = remember(layer, None, position, role="seed", level=0)
        frontier.append((layer, position, key))

    threshold = 0.0
    threshold_from = "every scored sender was kept"

    for level in range(depth):
        if not frontier:
            break
        next_frontier: list[tuple[int, int, str]] = []
        for layer, position, target in frontier:
            if layer <= 0:
                # Layer 0 has no earlier component; `path_trace` refuses it and
                # asking would spend a refusal per seed.
                continue
            traced = trace_fn(layer, position)
            graph.n_receivers_expanded += 1
            graph.passes += int(traced.get("passes") or 0)
            graph.seconds = round(graph.seconds + float(traced.get("seconds") or 0), 2)

            senders = list(traced.get("senders") or [])
            graph.n_scored += len(senders)

            # THE THRESHOLD, from the trace's own resolution rather than
            # invented here. Two senders closer than the dtype can express are
            # tied, so keeping edges below it would draw arithmetic.
            resolution = float(traced.get("recovery_resolution") or 0.0)
            if resolution > threshold:
                threshold = resolution
                threshold_from = (
                    "the dtype's own recovery resolution, below which two "
                    "senders are tied rather than ranked"
                )

            kept = 0
            for row in senders:
                recovery = float(row.get("recovery") or 0.0)
                # DRAWN MEANS CONTROLLED. `path_trace` runs its eight
                # same-norm draws on the strongest few per receiver, and an
                # edge without them has a score and no verdict -- there would
                # be nothing behind it to click. Keeping those would also hand
                # the graph's SIZE to the resolution, which is not a constant:
                # MEASURED in bfloat16, Qwen3-1.7B reads 0.006231 on
                # the reference pair, and the same figure on another model can
                # sit orders of magnitude away. See the module docstring.
                if row.get("clears_control") is None:
                    graph.n_pruned += 1
                    continue
                # And a controlled edge still has to clear what the dtype can
                # express, or the number behind it is arithmetic.
                #
                # ON THE MAGNITUDE, because recovery is SIGNED. A sender that
                # pushes the answer AWAY from the clean run is a finding, not a
                # weak one: `patch.trace` keeps exactly those (its docstring
                # records 5 of 132 sites moving it away, the worst by -0.157)
                # and it is the reason the metric is a signed fraction rather
                # than KL. `recovery <= resolution` dropped every one of them
                # and so conflated "too small for this dtype to express" with
                # "in the other direction" -- two different findings, and the
                # node grid one panel up distinguishes them by colour.
                #
                # Rare rather than impossible today: `path_trace` ranks by
                # SIGNED recovery and controls only the strongest
                # `max_controlled`, so the most negative senders are at the
                # bottom of its list and usually go uncontrolled -- MEASURED on
                # Qwen3-1.7B into L2@2, 4 of 34 senders were negative and none
                # of those were among the 12 controlled. That is an upstream
                # ranking choice, and relying on it to keep this arm unreachable
                # would make the sign here a lie that happens to hold.
                if abs(recovery) <= resolution:
                    graph.n_pruned += 1
                    continue
                if len(graph.edges) >= max_edges:
                    graph.n_pruned += 1
                    continue
                source = remember(
                    row.get("layer", 0),
                    row.get("head"),
                    position,
                    role="sender",
                    level=level + 1,
                )
                if (source, target) in seen_edges:
                    continue
                seen_edges.add((source, target))
                graph.edges.append(
                    Edge(
                        source=source,
                        target=target,
                        recovery=round(recovery, 6),
                        control_max=(
                            None
                            if row.get("control_max") is None
                            else float(row["control_max"])
                        ),
                        controls=[
                            float(c)
                            for c in (row.get("controls") or [])
                            if isinstance(c, (int, float)) and not isinstance(c, bool)
                        ],
                        control_draws=int(row.get("control_draws") or 0),
                        clears_control=row.get("clears_control"),
                        clears_position=row.get("clears_position"),
                    )
                )
                kept += 1
                # Only an edge that BEAT its control earns an expansion. One
                # that did not is still drawn -- see the class docstring -- but
                # spending a whole `path_trace` on it would walk the graph into
                # noise.
                if row.get("clears_control"):
                    if level + 1 < depth:
                        next_frontier.append(
                            (int(row.get("layer", 0)), position, source)
                        )
                    elif source not in graph.frontier:
                        # The depth ran out with this one still expandable. It
                        # is a place the walk STOPPED, and recording it is the
                        # difference between "nothing wrote this" and "we did
                        # not ask" -- which is the whole reason `frontier`
                        # exists.
                        graph.frontier.append(source)
            if kept == 0:
                graph.frontier.append(target)
        # The strongest few, so the next level does not fan out quadratically.
        next_frontier.sort(key=lambda item: -item[0])
        # ONE ENTRY PER RECEIVER. `path_trace(layer, position)` takes only
        # those two, so two queue entries with the same pair are the same
        # receiver and tracing both spends the whole pass budget twice for one
        # answer. It happens whenever two receivers share a sender, which is
        # the normal case rather than a corner: MEASURED on a two-seed walk
        # where both seeds were written by one MLP, `path_trace` was called for
        # (5, 3) twice -- several hundred forward passes on a real model, for a
        # result already in hand.
        #
        # It also made the seeding sentence's own arithmetic wrong. The second
        # visit re-scored the same senders into `n_scored` while `seen_edges`
        # silently dropped the repeated edges, so `n_scored - n_pruned` came to
        # 4 against 3 edges drawn -- a reader doing the subtraction the
        # sentence invites got a number the picture did not have.
        seen_receivers: set[tuple[int, int]] = set()
        frontier = []
        for item in next_frontier:
            key = (item[0], item[1])
            if key in seen_receivers:
                continue
            seen_receivers.add(key)
            frontier.append(item)
            if len(frontier) >= max_receivers:
                break

    # Anything still queued when the depth ran out is a place the walk stopped,
    # not a place with nothing in it.
    graph.frontier.extend(key for _, _, key in frontier if key not in graph.frontier)

    graph.nodes = list(seen_nodes.values())
    graph.prune_threshold = round(threshold, 6)
    graph.prune_from = threshold_from
    return graph
