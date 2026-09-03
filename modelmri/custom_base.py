# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""The two pieces of the custom-adapter surface that both halves of it need.

`custom.py` owns loading an adapter and `custom_ablate.py` owns ablating
through one, and the second is a consumer of the first — which is the right way
round. What made the graph cyclic was the one edge going the other way:
`custom.py`'s `CustomHandle.ablate()` reaching for `custom_ablate` to do the
work, while `custom_ablate` reached back for `AdapterError` and `leaf_modules`.

Both imports were deferred, so nothing ever broke at load time, and a cycle
that never bites is still a cycle: it bites the day somebody hoists one of
those imports to the top of its file. `tests/test_import_graph.py` found this
one — CodeQL did not report it, having flagged only the `runtime`/`gguf_load`
loop — and it is the same shape and takes the same fix.

Neither of these needs anything from `custom.py`. An exception class and a
six-line walk over `named_modules()` are exactly the kind of thing two
collaborating modules should share from underneath rather than sideways.

`custom.py` re-exports both, so `custom.AdapterError` and
`custom.leaf_modules` still resolve for `server.py`, the tests, and anything
else that has always spelled them that way.
"""

from __future__ import annotations

from .errors import BadRequest


class AdapterError(BadRequest):
    """Something about the file or its contents is wrong, and we say what.

    A `BadRequest`, and therefore still a ValueError, so every handler that
    caught it before catches it unchanged. The classification is the point:
    each of these is a fact about the path in the request or the file it names
    — not a Python module, no `load()`, a state_dict where a model was
    expected — which is 422 with the sentence, exactly what `/api/custom/load`
    and `/api/custom/run` answer today.

    What it is NOT is the exception raised *by* the adapter. That one is the
    user's own code failing, and server.py deliberately names its class rather
    than hiding it behind the generic 500 — see the note there.
    """


def leaf_modules(model) -> list[tuple[str, object]]:
    """Modules with no children, in declaration order, named as you named them."""
    out = []
    for name, mod in model.named_modules():
        if name and not list(mod.children()):
            out.append((name, mod))
    return out
