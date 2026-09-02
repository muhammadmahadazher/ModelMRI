"""The package's import graph has no cycles, and this is what says so.

A deferred import — `from . import x` inside the function that needs it — is
how this package has always broken load-time cycles, and it works: nothing here
has ever failed to import. What it does not do is make the cycle go away. It
makes it not matter *yet*, which is a different claim, and the difference shows
up the day somebody moves one of those imports to the top of its file for
tidiness and the package stops importing at all.

CodeQL found the one that existed. `runtime` reaches for `model_diff`,
`gguf_load` and `behavdiff`; `model_diff` reaches for `behavdiff`; `behavdiff`
reaches for `gguf_load` — all forward. Then `gguf_load` reached back for
`runtime.move_to_device` and closed the loop, and `py/cyclic-import` was
reported on all seven of those imports. Seven notes, one edge.

`move_to_device` never needed anything from `runtime`, so it lives in
`device_move.py` now and the graph is a DAG. This test is the reader for that:
it walks every `modelmri` module, collects the intra-package imports whether
they sit at module level or inside a function, and fails with the cycle written
out if one comes back.

It is deliberately about the WHOLE package rather than those four modules. The
next cycle will not be in the ones already fixed.
"""

from __future__ import annotations

import ast
import pathlib

PACKAGE = pathlib.Path(__file__).resolve().parents[1] / "modelmri"


def _intra_package_imports(tree: ast.AST, me: str) -> set[str]:
    """Every `modelmri.x` this module imports, at any depth.

    Both spellings and both depths: `from . import x`, `from .x import y`,
    `import modelmri.x`, and the same three written inside a function body,
    which is the form that matters here because it is the one the cycle was
    made of.
    """
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level and node.module:
                # `from .x import y` / `from .x.y import z` -> "x"
                found.add(node.module.split(".")[0])
            elif node.level:
                # `from . import x, y`
                found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == "modelmri" and len(parts) > 1:
                    found.add(parts[1])
    found.discard(me)
    return found


def _graph() -> dict[str, set[str]]:
    modules = {p.stem: p for p in PACKAGE.glob("*.py") if p.stem != "__init__"}
    graph: dict[str, set[str]] = {}
    for name, path in modules.items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        graph[name] = {m for m in _intra_package_imports(tree, name) if m in modules}
    return graph


def _find_cycle(graph: dict[str, set[str]]) -> list[str] | None:
    """One cycle as a readable path, or None. Iterative DFS with a colour map —
    the package is ~100 modules and a recursive walk over a graph this shape is
    a stack overflow waiting for a bad day."""
    WHITE, GREY, BLACK = 0, 1, 2
    colour = dict.fromkeys(graph, WHITE)

    for root in sorted(graph):
        if colour[root] != WHITE:
            continue
        stack: list[tuple[str, list[str]]] = [(root, [root])]
        while stack:
            node, path = stack.pop()
            if colour[node] == GREY:
                colour[node] = BLACK
                continue
            if colour[node] == BLACK:
                continue
            colour[node] = GREY
            stack.append((node, path))
            for nxt in sorted(graph[node]):
                if colour[nxt] == GREY:
                    return path[path.index(nxt) :] + [nxt]
                if colour[nxt] == WHITE:
                    stack.append((nxt, path + [nxt]))
    return None


def test_no_module_in_this_package_imports_itself_in_a_circle():
    cycle = _find_cycle(_graph())
    assert cycle is None, (
        "these modules import each other in a circle:\n  "
        + " -> ".join(cycle or [])
        + "\n\nA deferred import inside a function hides this at load time and "
        "does not remove it. Break the edge that points backwards — the way "
        "`move_to_device` was moved out of `runtime` into `device_move` — "
        "rather than moving the import deeper into the function."
    )


def test_the_edge_that_used_to_close_the_loop_stays_broken():
    """The specific regression, named, so the general test above is not the
    only thing standing between this and a repeat.

    `gguf_load` needs `move_to_device`. Importing it from `runtime` is what
    made the graph cyclic, and it is an easy edit to make again — `runtime` is
    where that function lived for most of this project's life and it is still
    re-exported there, so `from .runtime import move_to_device` would work
    perfectly and quietly restore the cycle.
    """
    graph = _graph()
    assert "runtime" not in graph["gguf_load"], (
        "`gguf_load` imports `runtime` again. `move_to_device` is re-exported "
        "from `runtime` for its existing callers, but importing it from there "
        "is the edge that made this package's import graph cyclic — take it "
        "from `modelmri.device_move`, which is a leaf."
    )
    assert "device_move" in graph["gguf_load"]
    # And the leaf really is a leaf: it imports `progress` and nothing that
    # could ever point back at it.
    assert graph["device_move"] <= {"progress"}, graph["device_move"]
