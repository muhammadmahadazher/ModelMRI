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
  seq INTEGER NOT NULL,
  meta TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS step_trace ON step(trace_id, seq);
"""

# Columns added after the table shipped. `CREATE TABLE IF NOT EXISTS` does
# nothing to a database that already has the table, so a store written by an
# earlier version keeps its old shape and every INSERT naming the new column
# fails — which would be an existing user's traces breaking on upgrade.
_MIGRATIONS = (
    ("step", "meta", "TEXT NOT NULL DEFAULT '{}'"),
)


def _loads(raw) -> dict:
    """Parse a stored JSON blob, treating damage as empty rather than fatal.

    A step whose `meta` cannot be read is a step that cannot be adopted, which
    is exactly what `{}` means here — the same outcome as a hosted-API call.
    Raising instead would take down the whole trace view over one bad row.
    """
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


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
        self._migrate()

    def _migrate(self) -> None:
        """Add columns this version needs to a store an older one created.

        Read from `PRAGMA table_info` rather than attempted-and-caught: an
        `ALTER TABLE` that fails for a reason other than "column exists" should
        surface, and catching OperationalError blindly would hide it.

        Takes the lock even though it only runs from `__init__`, where nothing
        else can hold a reference to the store yet. `test_traces_concurrency`
        asserts every method touching the connection serialises it, and its
        reasoning is that the original defect was an ABSENCE nobody noticed —
        "a new method added tomorrow is the same bug". This was that method,
        and the test caught it. An uncontended lock costs nothing; an
        invariant with a remembered exception is how the 0.10 data race
        happened.
        """
        with self._lock:
            for table, column, decl in _MIGRATIONS:
                existing = {
                    row[1] for row in self._db.execute(f"PRAGMA table_info({table})")
                }
                if not existing:  # table absent; the schema above handles it
                    continue
                if column not in existing:
                    self._db.execute(
                        f"ALTER TABLE {table} ADD COLUMN {column} {decl}"
                    )
                    log.info("added %s.%s to the trace store", table, column)
            self._db.commit()

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
                    " duration_ms, input, output, tokens_in, tokens_out, error, seq,"
                    " meta)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
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
                        # What the recorder captured about the model that
                        # produced this step, when a local one did: model id,
                        # input ids, dtype, device. Absent for a hosted-API
                        # call, and absence is the signal that the weights are
                        # not on this machine rather than a missing field.
                        json.dumps(s.get("meta") or {}),
                    ),
                )
            self._db.commit()
        return trace_id

    def list_traces(self) -> list[dict]:
        # UNDER THE LOCK, like every writer in this class.
        #
        # `__init__` opens ONE connection with check_same_thread=False and
        # shares it across threads, which Python's sqlite3 permits and does
        # not make safe: serialising access is the caller's job. Every writer
        # here did it; the two readers did not. So a request arriving while
        # anything else touched the database ran a second statement on the
        # same connection, the cursors interleaved, and `fetchall()` came back
        # with rows of the wrong width -- surfacing as
        # `IndexError: tuple index out of range` in the row mapping below, on
        # a SELECT whose column count is fixed and cannot vary.
        #
        # It showed up as an intermittent 500 from GET /api/traces on a cold
        # start, when the browser's first load races the store's own setup.
        # Before the agents panel was given a retry, one of those left the
        # panel empty for the rest of the session with no way back -- which is
        # indistinguishable from "you have not recorded anything".
        with self._lock:
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
        # Same rule as list_traces, and for the same reason: one connection
        # shared across threads has to be serialised by its owner. This one
        # runs TWO statements whose results are read together, so an
        # interleaving here can also pair one trace's header with another
        # trace's steps -- a wrong answer rather than a crash, which is worse.
        with self._lock:
            t = self._db.execute(
                "SELECT id, name, started_at, meta FROM trace WHERE id=?",
                (trace_id,),
            ).fetchone()
            if t is None:
                return None
            rows = self._db.execute(
                "SELECT id, parent_id, kind, name, started_ms, duration_ms,"
                " input, output, tokens_in, tokens_out, error, seq, meta"
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
                "meta": _loads(r[12]),
                # The one thing the panel needs without parsing meta itself:
                # can this step be opened in the mechanistic panels at all?
                "adoptable": bool(_loads(r[12]).get("input_ids")),
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
