# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

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
# adding it to `modelmri_record.KINDS` too — the recorder cannot import this
# module (it is stdlib-only by contract, proved by a test that spawns a fresh
# interpreter), so the two lists are separate literals kept in agreement by
# `tests/test_step_kinds.py` rather than by an import.
#
# The last four are what a retrieval pipeline is made of. They were absent, and
# the absence was not neutral: `import_trace` refuses a kind it does not know
# and refuses the WHOLE document, so a RAG agent could not record itself at
# all. Every one of them had to be filed as `tool_call`, and a metric asking
# "how long did retrieval take" then had no way to find retrieval except by
# pattern-matching on step names somebody else chose.
#
#   retrieval  fetching candidate documents — a vector store, a search index,
#              a grep. The step that decides what the model is allowed to know.
#   embedding  text -> vector. Separate from `llm_call` because it reports
#              provider token counts under a different meaning and answers a
#              different question, and because it is usually the step a RAG
#              pipeline runs twice as often as anything else.
#   rerank     reordering candidates against the query. Its own kind rather
#              than a second `retrieval`, because the interesting measurement
#              is what it CHANGED, and that needs the two to be separable.
#   guardrail  a policy check on the way in or out. Not `error`: a guardrail
#              that fires did its job, and filing a working safety check under
#              the kind that means "this run broke" is a wrong answer that
#              looks like a right one.
VALID_KINDS = {
    "llm_call",
    "tool_call",
    "subagent",
    "mcp_call",
    "user_turn",
    "error",
    "retrieval",
    "embedding",
    "rerank",
    "guardrail",
}
