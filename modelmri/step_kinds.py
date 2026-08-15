"""The kinds a recorded step may be, and nothing else.

A leaf module on purpose. `traces.py` validates against this set when writing
a step and `trace_query.py` validates against it when parsing a `kind:` filter,
so whichever of those two owned it, the other had to import it — and that was
a genuine import cycle, papered over with a function-level import inside
`traces.search()`. Deferring an import to hide a cycle works right up until
somebody moves it back to the top of the file.

Nothing here imports anything, so nothing can cycle through it.
"""

from __future__ import annotations

# Written by `modelmri-record` and read by the agents panel. Adding one means
# adding it to the recorder too, or the writer will refuse its own output.
VALID_KINDS = {"llm_call", "tool_call", "subagent", "mcp_call", "user_turn", "error"}
