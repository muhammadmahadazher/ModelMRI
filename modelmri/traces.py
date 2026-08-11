"""Agent trace storage: SQLite-backed, stdlib only.

A Trace is a tree of Steps (llm_call / tool_call / subagent / user_turn /
error). Traces arrive as one JSON document (imported from a .mri bundle or
posted by modelmri-record) and are stored denormalized enough to render a
timeline without joins at query time.

SQLite on purpose: ModelMRI is pip-install local-first — the store must
ship embedded, zero-config. (A hosted/team edition would be the moment for
PostgreSQL, not the local tool.)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from pathlib import Path

from .errors import BadRequest

log = logging.getLogger("modelmri")

VALID_KINDS = {"llm_call", "tool_call", "subagent", "mcp_call", "user_turn", "error"}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trace (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  started_at TEXT NOT NULL,
  meta TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS step (
  id TEXT PRIMARY KEY,
  trace_id TEXT NOT NULL REFERENCES trace(id) ON DELETE CASCADE,
  parent_id TEXT,
  kind TEXT NOT NULL,
  name TEXT NOT NULL DEFAULT '',
  started_ms INTEGER NOT NULL,
  duration_ms INTEGER NOT NULL DEFAULT 0,
  input TEXT NOT NULL DEFAULT '',
  output TEXT NOT NULL DEFAULT '',
  tokens_in INTEGER,
  tokens_out INTEGER,
  error INTEGER NOT NULL DEFAULT 0,
  seq INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS step_trace ON step(trace_id, seq);
"""


class TraceStore:
    """One SQLite file; safe for the single-process server (per-call cursors)."""

    def __init__(self, path: str | Path) -> None:
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        # WAL needs a shared-memory file and byte-range locking, which SQLite
        # documents as unsupported on network filesystems. An NFS-mounted Linux
        # home, a macOS home on SMB, or MODELMRI_HOME on a mapped drive answers
        # SQLITE_IOERR here rather than declining the mode -- and this runs
        # unguarded inside `create_app`, so the exception escaped before the
        # server printed its URL. The whole tool became unusable over a feature
        # the reader may not be using. The rollback journal works everywhere.
        try:
            self._db.execute("PRAGMA journal_mode=WAL")
        except sqlite3.Error as err:
            # Not silent: a reader whose traces feel slow deserves to be able
            # to find out that WAL was declined and why. Recorded rather than
            # swallowed, and not raised, because the rollback journal is a
            # working database and losing the whole server over a journal mode
            # is the bug this guard exists to prevent.
            log.info(
                "WAL journal unavailable at %s (%s); using the rollback "
                "journal, which is slower and works on network filesystems",
                path,
                err,
            )
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.executescript(_SCHEMA)

    def import_trace(self, doc: dict) -> str:
        """Store one trace document; returns the trace id.

        Expected shape:
        {name, started_at, meta?, steps: [{id?, parent_id?, kind, name?,
         started_ms, duration_ms?, input?, output?, tokens_in?, tokens_out?,
         error?}, ...]}   (steps in chronological order)
        """
        # Schema validation of a document somebody posted: BadRequest, 422.
        # `/api/traces/import` takes a bare dict — there is no model between
        # the wire and here — so these lines are the entire contract, and a
        # hand-written or third-party document has nothing else to go on.
        steps = doc.get("steps", [])
        if not isinstance(steps, list) or not steps:
            raise BadRequest("trace document needs a non-empty 'steps' list")
        # Every check runs BEFORE the lock and the INSERTs below, and that
        # ordering is load-bearing: a raise partway through the insert loop
        # leaves this connection holding an open transaction with half a trace
        # in it, which the next commit from anywhere would then write.
        timings: list[tuple[int, int]] = []
        for i, s in enumerate(steps):
            # Measured: `steps: [1, 2]` used to reach `s.get` and die with
            # AttributeError, which the server can only answer as a 500 —
            # "something inside ModelMRI failed" about a document that is
            # simply the wrong shape.
            if not isinstance(s, dict):
                raise BadRequest(
                    f"step {i} is not an object with a 'kind' (got {type(s).__name__})"
                )
            if s.get("kind") not in VALID_KINDS:
                # `sorted`, because VALID_KINDS is a set and an unordered list
                # in an error message is a different sentence every run.
                raise BadRequest(
                    f"invalid step kind: {s.get('kind')!r} — "
                    f"use one of {', '.join(sorted(VALID_KINDS))}"
                )
            timings.append((_ms(s, "started_ms", i), _ms(s, "duration_ms", i)))

        trace_id = str(doc.get("id") or uuid.uuid4().hex[:12])
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO trace(id, name, started_at, meta) VALUES(?,?,?,?)",
                (
                    trace_id,
                    str(doc.get("name", "unnamed-trace")),
                    str(doc.get("started_at", "")),
                    json.dumps(doc.get("meta", {})),
                ),
            )
            self._db.execute("DELETE FROM step WHERE trace_id=?", (trace_id,))
            for seq, s in enumerate(steps):
                self._db.execute(
                    "INSERT INTO step(id, trace_id, parent_id, kind, name, started_ms,"
                    " duration_ms, input, output, tokens_in, tokens_out, error, seq)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        str(s.get("id") or f"{trace_id}-{seq}"),
                        trace_id,
                        s.get("parent_id"),
                        s["kind"],
                        str(s.get("name", "")),
                        timings[seq][0],
                        timings[seq][1],
                        _clip(s.get("input", "")),
                        _clip(s.get("output", "")),
                        s.get("tokens_in"),
                        s.get("tokens_out"),
                        1 if s.get("error") else 0,
                        seq,
                    ),
                )
            self._db.commit()
        return trace_id

    def list_traces(self) -> list[dict]:
        rows = self._db.execute(
            "SELECT t.id, t.name, t.started_at,"
            " (SELECT COUNT(*) FROM step s WHERE s.trace_id=t.id),"
            " (SELECT COALESCE(MAX(s.started_ms + s.duration_ms),0) FROM step s"
            "   WHERE s.trace_id=t.id),"
            " (SELECT COUNT(*) FROM step s WHERE s.trace_id=t.id AND s.error=1),"
            " t.meta"
            " FROM trace t ORDER BY t.started_at DESC"
        ).fetchall()
        return [
            {
                "id": r[0],
                "name": r[1],
                "started_at": r[2],
                "n_steps": r[3],
                "total_ms": r[4],
                "n_errors": r[5],
                # Scripted sample data must never be indistinguishable from a
                # run you actually recorded. examples/record_demo.py writes a
                # deliberately failing `git push` so the timeline has an error
                # to render, and in the list that looked exactly like your own
                # agent failing.
                "demo": bool((json.loads(r[6] or "{}") or {}).get("demo")),
            }
            for r in rows
        ]

    def delete(self, trace_id: str) -> bool:
        """Remove one trace and its steps. False when it was not there."""
        with self._lock:
            cur = self._db.execute("DELETE FROM trace WHERE id=?", (trace_id,))
            self._db.commit()
        return cur.rowcount > 0

    def clear(self, keep_demo: bool = False) -> int:
        """Remove every trace. Returns how many went.

        `keep_demo` exists so "clear my runs" does not also throw away the
        sample the docs tell people to look at.
        """
        with self._lock:
            if keep_demo:
                cur = self._db.execute(
                    "DELETE FROM trace WHERE COALESCE(json_extract(meta,'$.demo'), 0) = 0"
                )
            else:
                cur = self._db.execute("DELETE FROM trace")
            self._db.commit()
        return cur.rowcount

    def get_trace(self, trace_id: str) -> dict | None:
        t = self._db.execute(
            "SELECT id, name, started_at, meta FROM trace WHERE id=?", (trace_id,)
        ).fetchone()
        if t is None:
            return None
        rows = self._db.execute(
            "SELECT id, parent_id, kind, name, started_ms, duration_ms, input,"
            " output, tokens_in, tokens_out, error, seq"
            " FROM step WHERE trace_id=? ORDER BY seq",
            (trace_id,),
        ).fetchall()
        steps = [
            {
                "id": r[0],
                "parent_id": r[1],
                "kind": r[2],
                "name": r[3],
                "started_ms": r[4],
                "duration_ms": r[5],
                "input": r[6],
                "output": r[7],
                "tokens_in": r[8],
                "tokens_out": r[9],
                "error": bool(r[10]),
                "seq": r[11],
            }
            for r in rows
        ]
        return {
            "id": t[0],
            "name": t[1],
            "started_at": t[2],
            "meta": json.loads(t[3]),
            "steps": steps,
        }


def _ms(step: dict, field: str, index: int) -> int:
    """One millisecond field as an int, or a BadRequest that names it.

    Bare `int(...)` on a document somebody hand-wrote raises ValueError in
    Python's own words — "invalid literal for int() with base 10: 'soon'" —
    which names neither the field nor the step it was in, and `int(None)`
    raises TypeError, which the server could only answer as a 500. Same 422,
    a sentence the sender can act on.
    """
    raw = step.get(field, 0)
    try:
        return int(raw)
    except (TypeError, ValueError) as err:
        raise BadRequest(
            f"step {index}: {field} must be a whole number of milliseconds, got {raw!r}"
        ) from err


def _clip(value: object, limit: int = 20_000) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[:limit] + f"… [+{len(text) - limit}]"
