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
import re
import sqlite3
import threading
import uuid
from pathlib import Path

from . import trace_query
from .errors import BadRequest

log = logging.getLogger("modelmri")

# Re-exported from the leaf module so this name keeps working for anything
# that already imports it from here. It LIVES in step_kinds.py because both
# this module and trace_query need it, and having them import each other made
# a genuine cycle: trace_query imported traces at module scope while traces
# imported trace_query inside `search()` to break it at runtime. Deferring an
# import to hide a cycle works until someone moves it back.
from .step_kinds import VALID_KINDS  # noqa: E402

# meta["source"] for a run this app performed itself, as opposed to one
# somebody's own instrumented program posted to /api/traces/import.
#
# A NEW KEY, NOT A REUSE OF meta["demo"]. `demo` means "sample data shipped
# with ModelMRI, not something you produced"; this means the exact opposite —
# you produced it, here, in this page. Collapsing the two would put your own
# generations behind the "Remove sample" button.
SOURCE_APP = "app"

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
  -- Nullable on purpose. `NOT NULL DEFAULT 0` made a step recorded bare
  -- indistinguishable from one that genuinely took no measurable time, which
  -- is the same shape as the `.get(name, 0.0)` that made 206 robot episodes
  -- show the same video. NULL means "not recorded" and renders as that.
  duration_ms INTEGER,
  input TEXT NOT NULL DEFAULT '',
  output TEXT NOT NULL DEFAULT '',
  tokens_in INTEGER,
  tokens_out INTEGER,
  -- Nullable for the same reason `duration_ms` is. Providers report these
  -- inconsistently: Anthropic returns cache counts only when a cache was in
  -- play, and reasoning tokens only from models that reason. `0` would claim
  -- the provider said zero. NULL says nobody asked or nobody answered, and
  -- the panel renders that as "not reported by provider".
  tokens_cache_read INTEGER,
  tokens_cache_write INTEGER,
  tokens_reasoning INTEGER,
  error INTEGER NOT NULL DEFAULT 0,
  seq INTEGER NOT NULL,
  meta TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS step_trace ON step(trace_id, seq);
-- Saved rubrics: a named set of exact predicates the reader wrote. The rules
-- live as JSON rather than as columns because they are a small document with
-- a validator of its own (`rubric.parse`), and normalising them into tables
-- would put a second, weaker validator in the schema.
CREATE TABLE IF NOT EXISTS rubric (
  name TEXT PRIMARY KEY,
  rules TEXT NOT NULL,
  saved_at TEXT NOT NULL
);
"""

# Columns added after the table shipped. `CREATE TABLE IF NOT EXISTS` does
# nothing to a database that already has the table, so a store written by an
# earlier version keeps its old shape and every INSERT naming the new column
# fails — which would be an existing user's traces breaking on upgrade.
#
# The token columns are added WITHOUT a default, so every row an older version
# wrote reads NULL — "this store predates the column, nobody asked" — rather
# than 0, which would assert the provider reported no cache use on calls that
# were never examined for it.
_MIGRATIONS = (
    ("step", "meta", "TEXT NOT NULL DEFAULT '{}'"),
    ("step", "tokens_cache_read", "INTEGER"),
    ("step", "tokens_cache_write", "INTEGER"),
    ("step", "tokens_reasoning", "INTEGER"),
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
        # RLock, not Lock. Every method that touches the connection takes
        # it — `test_traces_concurrency` asserts exactly that, and the reason
        # is that the 0.10 data race was an ABSENCE nobody noticed. Some of
        # those methods are helpers called by others that already hold it, and
        # with a plain Lock that self-deadlocks; the alternative is a list of
        # remembered exemptions, which is how the original bug survived review.
        # A reentrant lock serialises other threads identically.
        self._lock = threading.RLock()
        # `paths.data_dir()` deliberately does not create anything -- it answers
        # where a thing belongs, and `paths.ensure` creates at the moment of
        # writing. Opening this file IS that moment: sqlite creates the database
        # but not the directory holding it, and no caller was ensuring it. On a
        # machine with no legacy `~/.modelmri` -- which is every NEW user, and
        # anyone pointing MODELMRI_HOME at a fresh path -- `modelmri serve` died
        # on `unable to open database file` before printing its URL. It survived
        # this long because every machine it ran on had already been an older
        # version's machine.
        parent = Path(path).parent
        if str(parent):
            try:
                parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                # A read-only or otherwise unusable location. Left to sqlite to
                # report against the real path, which names the actual problem
                # better than a mkdir failure one level up would.
                pass
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
                    self._db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
                    log.info("added %s.%s to the trace store", table, column)
            self._relax_duration()
            self._db.commit()
        self._build_index()

    def _relax_duration(self) -> None:
        """Drop `NOT NULL` from step.duration_ms on a store that still has it.

        SQLite cannot relax a constraint with ALTER TABLE, so this is the
        documented rebuild: new table, copy, drop, rename. Runs once — after it
        the PRAGMA reports notnull=0 and the check below is false forever.

        Called with the lock already held.
        """
        with self._lock:
            columns = list(self._db.execute("PRAGMA table_info(step)"))
            if not columns:
                return
            duration = next((c for c in columns if c[1] == "duration_ms"), None)
            if duration is None or not duration[3]:  # c[3] is `notnull`
                return

            names = ", ".join(c[1] for c in columns)
            log.info("rebuilding the step table to make duration_ms nullable")
            # Foreign keys off for the swap: `step` references `trace`, and the
            # drop would otherwise be refused or cascade. Restored immediately —
            # __init__ turned them on and the rest of the class relies on it.
            self._db.execute("PRAGMA foreign_keys=OFF")
            try:
                self._db.executescript(
                    f"""
                    CREATE TABLE step_new AS SELECT {names} FROM step;
                    DROP TABLE step;
                    """
                )
                # `CREATE TABLE AS SELECT` keeps the data and loses every
                # constraint, which is exactly what is wanted here — the schema
                # below re-declares the real table and the copy fills it.
                self._db.executescript(_SCHEMA)
                self._db.execute(
                    f"INSERT INTO step({names}) SELECT {names} FROM step_new"
                )
                self._db.execute("DROP TABLE step_new")
                self._db.commit()
            finally:
                self._db.execute("PRAGMA foreign_keys=ON")

    def _build_index(self) -> None:
        """Create the FTS5 index, or record that this build has no FTS5.

        FTS5 is compiled into essentially every CPython SQLite, which is what
        makes full-text search over every trace a `pip install` rather than a
        ClickHouse container. "Essentially every" is not "every", so a build
        without it degrades to a substring scan and the response says which
        engine answered — the same degrade-and-say-so shape as the WAL guard
        above, rather than a feature that silently becomes a different feature.

        Not backfilled on every start: `INSERT INTO ... SELECT` runs only when
        the index is empty and the step table is not.
        """
        with self._lock:
            try:
                self._db.executescript(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS step_fts USING fts5(
                      input, output, name, content='step', content_rowid='rowid'
                    );
                    """
                )
                # `count(*) FROM step_fts` does NOT count index rows. This is
                # an external-content table, so an unqualified scan reads
                # through to `step` and returns the CONTENT count — which made
                # `indexed` equal `stored` on every store an earlier version
                # wrote, so `stored and not indexed` was false and the backfill
                # never ran. Once. Ever. The search box then answered
                # `engine: "fts5"` with an empty list for every trace the user
                # already had, and nothing in the payload dissented.
                #
                # `step_fts_docsize` is the shadow table that holds one row per
                # INDEXED document, so it is the count that means what this
                # check needs it to mean.
                indexed = self._db.execute(
                    "SELECT count(*) FROM step_fts_docsize"
                ).fetchone()[0]
                stored = self._db.execute("SELECT count(*) FROM step").fetchone()[0]
                if stored and not indexed:
                    # 'rebuild' rather than an INSERT..SELECT: it is FTS5's own
                    # idempotent resync from the content table, so running it
                    # against a partially-populated index cannot double-index.
                    self._db.execute("INSERT INTO step_fts(step_fts) VALUES('rebuild')")
                    log.info("backfilled the trace search index (%d steps)", stored)
                self._db.commit()
                self.fts = True
            except sqlite3.Error as err:
                log.info(
                    "SQLite full-text search unavailable (%s); trace search "
                    "will scan for substrings instead, which is slower and "
                    "matches inside words",
                    err,
                )
                self.fts = False

    def _retract_from_index(self, trace_id: str | None = None) -> None:
        """Tell FTS5 to forget rows that are about to be deleted.

        External-content FTS5 keeps its own copy of the terms and is not
        notified by a DELETE on the content table. The documented retraction is
        an insert of the special 'delete' command carrying the OLD values, so
        this reads them back out of `step` while they still exist.

        Called with the lock held. Never raises: an index that drifts is a
        search that returns a stale hit, and losing a whole import over that
        would be the worse trade.
        """
        with self._lock:
            if not getattr(self, "fts", False):
                return
            where = "WHERE trace_id=?" if trace_id else ""
            args = (trace_id,) if trace_id else ()
            try:
                self._db.execute(
                    "INSERT INTO step_fts(step_fts, rowid, input, output, name)"
                    f" SELECT 'delete', rowid, input, output, name FROM step {where}",
                    args,
                )
            except sqlite3.Error as err:
                log.info("could not retract trace search entries (%s)", err)

    def _publish_to_index(self, trace_id: str) -> None:
        """Add one trace's steps to the index. Lock held; never raises."""
        with self._lock:
            if not getattr(self, "fts", False):
                return
            try:
                self._db.execute(
                    "INSERT INTO step_fts(rowid, input, output, name)"
                    " SELECT rowid, input, output, name FROM step WHERE trace_id=?",
                    (trace_id,),
                )
            except sqlite3.Error as err:
                log.info("could not index trace %s for search (%s)", trace_id, err)

    def search(self, raw: str, limit: int = 100) -> dict:
        """Steps matching a query, newest first, with the engine named.

        Results are STEPS rather than runs, because the thing somebody is
        looking for is the tool call that failed, not the hour it happened in.
        """
        query = trace_query.parse(raw)
        if query.is_empty:
            return {
                "engine": "fts5" if self.fts else "substring-scan",
                "query": query.to_dict(),
                "results": [],
                "note": "type something to search",
            }

        clauses, params = trace_query.where(query)
        engine = "fts5" if (self.fts and query.text) else "substring-scan"

        if engine == "fts5":
            sql = (
                "SELECT s.id, s.trace_id, t.name, s.kind, s.name, s.started_ms,"
                " s.duration_ms, s.input, s.output, s.error, s.seq, t.started_at"
                " FROM step_fts f JOIN step s ON s.rowid = f.rowid"
                " JOIN trace t ON t.id = s.trace_id"
                " WHERE step_fts MATCH ?"
            )
            args: list = [_fts_match(query.text)]
        else:
            sql = (
                "SELECT s.id, s.trace_id, t.name, s.kind, s.name, s.started_ms,"
                " s.duration_ms, s.input, s.output, s.error, s.seq, t.started_at"
                " FROM step s JOIN trace t ON t.id = s.trace_id"
                " WHERE 1=1"
            )
            args = []
            if query.text:
                sql += " AND (s.input LIKE ? OR s.output LIKE ? OR s.name LIKE ?)"
                like = f"%{query.text}%"
                args += [like, like, like]

        if clauses:
            sql += f" AND {clauses}"
            args += params
        # `step.started_ms` is NOT a clock. The recorder writes it as
        # milliseconds since that trace's own start
        # (`int((time.monotonic() - t0) * 1000)`), so ordering by it ranked hits
        # by how deep into their run they happened. A step nine minutes into a
        # run from last month outranked one a second into today's, the
        # docstring promised "newest first", and the LIMIT then dropped the
        # newest matches — a full page of stale hits with today's run silently
        # absent, which is worse than an empty list because it looks complete.
        #
        # `trace.started_at` is the real clock, and was already joined here and
        # unused. `list_traces` has always ordered by it.
        sql += " ORDER BY t.started_at DESC, s.started_ms ASC LIMIT ?"
        args.append(int(limit))

        # UNDER THE LOCK, like every other reader in this class. The two that
        # were not are what produced short rows and intermittent 500s from
        # /api/traces in 0.10; a new query path is a fresh chance to repeat
        # exactly that bug.
        with self._lock:
            try:
                rows = self._db.execute(sql, args).fetchall()
            except sqlite3.Error as err:
                # A malformed FTS5 expression is the caller's query, not a
                # server fault — `foo(` and `NEAR/` both land here.
                raise BadRequest(
                    "that search could not be run as a full-text query. Try "
                    "plain words, or quote a phrase."
                ) from err

        results = []
        for r in rows:
            text_in, clipped_in = _unclip(r[7])
            text_out, clipped_out = _unclip(r[8])
            results.append(
                {
                    "step_id": r[0],
                    "trace_id": r[1],
                    "trace_name": r[2],
                    "kind": r[3],
                    "name": r[4],
                    "started_ms": r[5],
                    "duration_ms": r[6],
                    "input": text_in,
                    "output": text_out,
                    "truncated_by": clipped_in + clipped_out,
                    "error": bool(r[9]),
                    "seq": r[10],
                    # So a hit can say WHEN it happened, and the panel can
                    # re-sort. A result row carried no wall-clock at all.
                    "trace_started_at": r[11],
                }
            )

        return {
            "engine": engine,
            "query": query.to_dict(),
            "results": results,
            "note": (
                "Full-text matching is by whole word, so a multi-word query is "
                "a contiguous phrase."
                if engine == "fts5"
                else "Scanning for substrings — matches inside words, and gets "
                "slower as the store grows."
            ),
        }

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
            timings.append((_ms(s, "started_ms", i), _ms_or_none(s, "duration_ms", i)))

        trace_id = str(doc.get("id") or uuid.uuid4().hex[:12])
        with self._lock:
            # BEFORE the trace row is replaced. `INSERT OR REPLACE` on `trace`
            # deletes the old row, and with `foreign_keys=ON` that CASCADES to
            # `step` — so doing this after the REPLACE meant the retraction
            # read an already-empty table and every old term survived in the
            # index, bound to rowids SQLite then reused. Re-importing a trace
            # with changed text made a search for the NEW word return a step
            # whose stored input was the OLD one, and freed rowids leaked terms
            # onto later, unrelated traces.
            #
            # The FTS index is external-content over `step`, so it learns about
            # a delete only when told, and it can only be told while the values
            # are still there to read. `delete()` and `clear()` already do this
            # in the right order; this was the one site that did not.
            self._retract_from_index(trace_id)
            self._db.execute("DELETE FROM step WHERE trace_id=?", (trace_id,))
            self._db.execute(
                "INSERT OR REPLACE INTO trace(id, name, started_at, meta) VALUES(?,?,?,?)",
                (
                    trace_id,
                    str(doc.get("name", "unnamed-trace")),
                    str(doc.get("started_at", "")),
                    json.dumps(doc.get("meta", {})),
                ),
            )
            for seq, s in enumerate(steps):
                self._db.execute(
                    "INSERT INTO step(id, trace_id, parent_id, kind, name, started_ms,"
                    " duration_ms, input, output, tokens_in, tokens_out,"
                    " tokens_cache_read, tokens_cache_write, tokens_reasoning,"
                    " error, seq, meta)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
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
                        # `.get(...)` with no default, so a key the recorder
                        # never set stores NULL. A `0` here would be this
                        # module asserting the provider reported no cache use.
                        s.get("tokens_cache_read"),
                        s.get("tokens_cache_write"),
                        s.get("tokens_reasoning"),
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
            self._publish_to_index(trace_id)
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
                " (SELECT COALESCE(MAX(s.started_ms + COALESCE(s.duration_ms,0)),0) FROM step s"
                "   WHERE s.trace_id=t.id),"
                " (SELECT COUNT(*) FROM step s WHERE s.trace_id=t.id AND s.error=1),"
                " t.meta"
                " FROM trace t ORDER BY t.started_at DESC"
            ).fetchall()
        out = []
        for r in rows:
            meta = json.loads(r[6] or "{}") or {}
            out.append(
                {
                    "id": r[0],
                    "name": r[1],
                    "started_at": r[2],
                    "n_steps": r[3],
                    "total_ms": r[4],
                    "n_errors": r[5],
                    # Scripted sample data must never be indistinguishable
                    # from a run you actually recorded.
                    # examples/record_demo.py writes a deliberately failing
                    # `git push` so the timeline has an error to render, and
                    # in the list that looked exactly like your own agent
                    # failing.
                    "demo": bool(meta.get("demo")),
                    # Same argument, one step further out: a generation you
                    # ran in the playground and a run of your own agent code
                    # both belong here, and they are not the same thing. The
                    # panel labels the first so the list stays readable when
                    # it holds both. "" for every trace written before this
                    # existed, and for anything posted without the key.
                    "source": str(meta.get("source") or ""),
                }
            )
        return out

    # ------------------------------------------------------------ rubrics

    def save_rubric(self, name: str, rules: list) -> None:
        """Store a named rubric. Rules arrive already validated by `rubric`.

        Under the lock like every other writer here — see `list_traces` for
        what happened the two times a method on this class forgot.
        """
        from datetime import datetime, timezone

        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO rubric(name, rules, saved_at) VALUES(?,?,?)",
                (
                    str(name),
                    json.dumps([r.to_dict() for r in rules]),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            self._db.commit()

    def rubrics(self) -> list:
        with self._lock:
            rows = self._db.execute(
                "SELECT name, rules, saved_at FROM rubric ORDER BY name"
            ).fetchall()
        # `_loads` treats damage as empty rather than fatal, but a rubric whose
        # rules will not parse is one that cannot be applied — so it is listed
        # with an empty rule set rather than dropped, which would make a saved
        # rubric silently vanish.
        out = []
        for name, raw, saved in rows:
            try:
                rules = json.loads(raw)
            except ValueError:
                rules = []
            out.append(
                {
                    "name": name,
                    "rules": rules if isinstance(rules, list) else [],
                    "saved_at": saved,
                    "readable": bool(rules),
                }
            )
        return out

    def delete_rubric(self, name: str) -> bool:
        with self._lock:
            cur = self._db.execute("DELETE FROM rubric WHERE name=?", (str(name),))
            self._db.commit()
        return cur.rowcount > 0

    def all_traces_with_steps(self, limit: int = 500):
        """Every run and its steps, for scoring. Newest first, capped.

        The cap is reported by the caller rather than swallowed here: a rubric
        answered over 500 of 4,000 runs is a different claim from one answered
        over all of them, and `slowest_percent` in particular is a claim ABOUT
        the set it was measured on.
        """
        for summary in self.list_traces()[: max(1, int(limit))]:
            doc = self.get_trace(str(summary.get("id")))
            if doc is not None:
                yield summary, doc.get("steps") or []

    def delete(self, trace_id: str) -> bool:
        """Remove one trace and its steps. False when it was not there."""
        with self._lock:
            # ON DELETE CASCADE removes the steps, and the index does not see
            # a cascade any more than it sees a plain delete.
            self._retract_from_index(trace_id)
            cur = self._db.execute("DELETE FROM trace WHERE id=?", (trace_id,))
            self._db.commit()
        return cur.rowcount > 0

    def clear(self, keep_demo: bool = False) -> int:
        """Remove every trace. Returns how many went.

        `keep_demo` exists so "clear my runs" does not also throw away the
        sample the docs tell people to look at.
        """
        with self._lock:
            # Retract everything first: `keep_demo` decides which traces
            # survive, but the cheap correct move is to empty the index and
            # re-publish what remains, rather than reproducing the WHERE twice.
            self._retract_from_index()
            if keep_demo:
                cur = self._db.execute(
                    "DELETE FROM trace WHERE COALESCE(json_extract(meta,'$.demo'), 0) = 0"
                )
            else:
                cur = self._db.execute("DELETE FROM trace")
            self._reindex_all()
            self._db.commit()
        return cur.rowcount

    def _reindex_all(self) -> None:
        """Re-publish every surviving step. Lock held; never raises."""
        with self._lock:
            if not getattr(self, "fts", False):
                return
            try:
                self._db.execute(
                    "INSERT INTO step_fts(rowid, input, output, name)"
                    " SELECT rowid, input, output, name FROM step"
                )
            except sqlite3.Error as err:
                log.info("could not rebuild the trace search index (%s)", err)

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
                " input, output, tokens_in, tokens_out, error, seq, meta,"
                " tokens_cache_read, tokens_cache_write, tokens_reasoning"
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
                # The clip marker is lifted out of the payload rather than
                # rendered as characters the agent produced. A truncated tool
                # output that reads as a complete one is how you debug the
                # wrong thing for an hour.
                "input": _unclip(r[6])[0],
                "output": _unclip(r[7])[0],
                "truncated_in": _unclip(r[6])[1],
                "truncated_out": _unclip(r[7])[1],
                "tokens_in": r[8],
                "tokens_out": r[9],
                # None travels as None all the way to the panel, which draws
                # "not reported by provider". Coercing to 0 anywhere on this
                # path would make a provider that stayed silent look like one
                # that answered zero.
                "tokens_cache_read": r[13],
                "tokens_cache_write": r[14],
                "tokens_reasoning": r[15],
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


def record_generation(
    store: TraceStore,
    *,
    model: str,
    backend: str,
    prompt: str,
    output: str,
    started_at: str,
    duration_ms: int,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    failure: str | None = None,
) -> str | None:
    """File one generation made in the app as a trace. Returns the id, or None.

    THE PANEL WAS ONLY EVER HALF A FEATURE.

    Until this, the only way to put anything in the agents panel was to add
    `modelmri-record` to a program of your own. So the obvious thing — load a
    model here, generate, look at the panel that says RECORDED RUNS — filled
    nothing, on any model, ever. That is indistinguishable from broken, and it
    was reported as broken. A generation IS an llm_call; the store already
    holds exactly that shape.

    Backend-agnostic on purpose: everything that varies between HF and Ollama
    (the model id, the token counts, whether there are token counts at all)
    arrives as an argument. Nothing here knows which one ran.

    NEVER RAISES. The recorder's own contract is that recording cannot crash
    the host app, and here the host is somebody's generation — the reader
    watching tokens arrive must not lose them because a SQLite write failed.
    One log line and the generation stands.

    `failure` is a SENTENCE, NOT AN EXCEPTION, and the name says so because
    the difference is the whole of test_no_exception_leaks: what lands here is
    the same text the caller published to the browser — an authored Refusal,
    or `_INTERNAL` when the cause was internal. A trace is a document people
    export and attach to issues, so torch's `str(err)` (which names paths on
    this machine) must never reach it, and there is nothing here that could
    turn one into a string.
    """
    try:
        # Partial output is kept and the failure is appended to it. A stream
        # that died after 40 tokens produced those 40 tokens; throwing them
        # away would leave a trace that says a run failed without showing how
        # far it got, which is the question you open the timeline to answer.
        text = output
        if failure:
            text = (
                f"{output}\n\n[failed] {failure}" if output else f"[failed] {failure}"
            )
        return store.import_trace(
            {
                # The model id is the trace NAME because the panel groups runs
                # by name — so repeated generations on one model collapse into
                # one row with a count, which is what the grouping is for.
                # Also in meta, where a reader parsing the store finds it
                # without having to guess that the name means a model.
                "name": model or "generation",
                "started_at": started_at,
                "meta": {
                    "source": SOURCE_APP,
                    "model": model,
                    "backend": backend,
                },
                "steps": [
                    {
                        "kind": "llm_call",
                        "name": "generate",
                        "started_ms": 0,
                        "duration_ms": max(int(duration_ms), 0),
                        "input": prompt,
                        "output": text,
                        "tokens_in": tokens_in,
                        "tokens_out": tokens_out,
                        "error": bool(failure),
                    }
                ],
            }
        )
    except Exception:
        # Everything, deliberately — see the docstring. `Exception` rather
        # than `BaseException` so a Ctrl-C during a write still ends the
        # process instead of being logged as a failed recording.
        log.exception("could not record this generation as a trace")
        return None


def _ms(step: dict, field: str, index: int) -> int:
    """One millisecond field as an int, or a BadRequest that names it.

    Bare `int(...)` on a document somebody hand-wrote raises ValueError in
    Python's own words — "invalid literal for int() with base 10: 'soon'" —
    which names neither the field nor the step it was in, and `int(None)`
    raises TypeError, which the server could only answer as a 500. Same 422,
    a sentence the sender can act on.

    ABSENT IS NOT ZERO. This defaulted a missing field to 0, and its only
    caller is `started_ms` -- which `import_trace`'s documented shape lists
    without a `?`, unlike `duration_ms?` beside it, because it is required.
    A step that never recorded when it started was filed as having started at
    the very instant the trace did, indistinguishable from one that genuinely
    did. On the timeline every such step stacks at the left edge; in a search
    it sorts first; and `patterns.py` reads these offsets to find retry
    storms, so a handful of fabricated zeros is a burst of activity at t=0
    that nothing ever did.
    `_ms_or_none` is the helper for a field that may legitimately be missing,
    and it tests for absence before it gets here.
    """
    if field not in step or step.get(field) is None:
        raise BadRequest(
            f"step {index}: {field} is required and was not given. A step with "
            f"no {field} cannot be placed on the timeline, and filing it at 0 "
            f"would put it at the start of the run."
        )
    raw = step[field]
    try:
        return int(raw)
    except (TypeError, ValueError) as err:
        raise BadRequest(
            f"step {index}: {field} must be a whole number of milliseconds, got {raw!r}"
        ) from err


def _ms_or_none(step: dict, field: str, index: int) -> int | None:
    """Same, but an ABSENT field stays absent instead of becoming 0.

    `duration_ms INTEGER NOT NULL DEFAULT 0` made a step recorded bare
    indistinguishable from one that genuinely took no measurable time, which
    is the same class as the `.get(name, 0.0)` that made 206 robot episodes
    show one video. A missing duration is now None all the way to the screen,
    where it renders as "not recorded" rather than a zero-width bar.

    An explicit 0 is still 0 — the caller said so.
    """
    if step.get(field) is None:
        return None
    return _ms(step, field, index)


def _clip(value: object, limit: int = 20_000) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[:limit] + f"… [+{len(text) - limit}]"


# What `_clip` appends. Parsed back out on read so the UI can render it as a
# marker instead of showing it as characters the agent produced — a truncated
# tool output that reads as a complete one is how you debug the wrong thing
# for an hour.
_CLIPPED = re.compile(r"… \[\+(\d+)\]$")


def _fts_match(text: str) -> str:
    """Free text as an FTS5 phrase, with the operator syntax neutralised.

    FTS5 has its own query language — `AND`, `NEAR`, `*`, `^`, `:` — and a
    person pasting an error message into a search box is not writing one.
    Wrapping the whole thing in double quotes makes it a literal phrase, and
    doubling any embedded quote keeps that true for text that contains them.
    """
    return '"' + (text or "").replace('"', '""') + '"'


def _unclip(text: str) -> tuple[str, int]:
    """(text without the marker, characters not stored).

    Returns 0 when nothing was clipped. The stored text keeps its 20,000
    characters; only the marker is lifted out, because the marker is metadata
    about the payload rather than part of it.
    """
    match = _CLIPPED.search(text or "")
    if not match:
        return text, 0
    return text[: match.start()], int(match.group(1))
