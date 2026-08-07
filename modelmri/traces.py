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
import sqlite3
import threading
import uuid
from pathlib import Path

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
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.executescript(_SCHEMA)

    def import_trace(self, doc: dict) -> str:
        """Store one trace document; returns the trace id.

        Expected shape:
        {name, started_at, meta?, steps: [{id?, parent_id?, kind, name?,
         started_ms, duration_ms?, input?, output?, tokens_in?, tokens_out?,
         error?}, ...]}   (steps in chronological order)
        """
        steps = doc.get("steps", [])
        if not isinstance(steps, list) or not steps:
            raise ValueError("trace document needs a non-empty 'steps' list")
        for s in steps:
            if s.get("kind") not in VALID_KINDS:
                raise ValueError(f"invalid step kind: {s.get('kind')!r}")

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
                        int(s.get("started_ms", 0)),
                        int(s.get("duration_ms", 0)),
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
            " (SELECT COUNT(*) FROM step s WHERE s.trace_id=t.id AND s.error=1)"
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
            }
            for r in rows
        ]

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


def _clip(value: object, limit: int = 20_000) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[:limit] + f"… [+{len(text) - limit}]"
