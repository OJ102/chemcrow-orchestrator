"""Saving runs to disk: the LangGraph checkpoint database and our runs tables.

WHAT THIS FILE IS
-----------------
Before this file, runs were saved only in memory, so closing the program lost
them. This file keeps them in SQLite DATABASE FILES on disk, so a run can be:
  - looked up later by its run_id (Nyla's dashboard, via the web API),
  - RESUMED after a crash from the last step that finished,
  - picked up by the separate WORKER program (worker.py), which runs the
    queued runs the web API (api.py) has recorded,
  - AUDITED: every important event is recorded (item 6, D2's G4).

SQLite = a small database stored in ONE ordinary file. Python can use it
directly (the built-in `sqlite3` module); no database server is needed. Inside
the file are TABLES: grids with named columns and one row per record.

TWO FILES (decision, 2026-11-06; D2 describes a single store, we chose two):
  checkpoints.db - LangGraph's own auto-saves: the whole shared notebook
                   (OPVState) after EVERY step. Used to resume a run. Its
                   internal format belongs to LangGraph; we don't read it directly.
  runs.db        - OUR tables:
                     runs         one row per run, with D1's "Run" fields plus what
                                  the worker needs (claim, heartbeat, cancel request)
                     receipts     idempotency receipts for POST requests (D1)
                     workers      when each worker program was last seen
                     audit_events the audit log (D2 G4): who did what, when

Both files go in one folder (default "runs/", ignored by git). The folder is
configurable, so later it can point at the shared server Nyla can reach.

HOW THE API AND THE WORKER SHARE WORK (they never talk directly, only through runs.db):
  1. API: create a run row with status "queued"
  2. Worker: claim_next() changes ONE queued run to "running", only if it's still
     queued, so two workers can never both claim it
  3. Worker: heartbeat() every few seconds while running it
  4. API: request_cancel() sets a flag; the worker checks it between steps
  5. Worker: claim_stale() takes over a "running" run whose heartbeat stopped
     (its worker crashed) and resumes it

THE AUDIT LOG (decisions Q4/Q5, 2026-11-10): every change to a run is recorded
in audit_events IN THE SAME TRANSACTION as the change itself. So if the audit
record can't be saved, the change doesn't happen either (D2: "inability to
save required audit fails the write"). Every change method therefore asks WHO
is making it (`actor`, e.g. "api" or "worker:<id>"). Audit records never
contain SMILES text, tokens or secrets (D2), only short codes and counts.

WHAT IS STILL DEFERRED: D2's full leases with fencing tokens. The heartbeat
and the "claimed_by" check are a lighter version, enough for one worker
(decision, 2026-11-08).
"""

# hashlib: makes a short "fingerprint" (hash) of a request body, for idempotency.
import hashlib

# json: converts between Python data and JSON text.
import json

# sqlite3: Python's built-in SQLite library.
import sqlite3

# datetime/timedelta/timezone: timestamps and time differences. D1 wants "UTC
# ISO 8601" text, e.g. "2026-11-09T15:04:05+00:00" (UTC = the same clock
# everywhere, no time zones). Text in this exact format sorts in time order,
# so the database can compare timestamps as text.
from datetime import datetime, timedelta, timezone

# Path: a tidy way to work with file and folder paths.
from pathlib import Path
from typing import Optional

# uuid4: a unique ID for each audit event.
from uuid import uuid4

# The SQLite auto-save from the langgraph-checkpoint-sqlite package (installed 2026-11-06).
from langgraph.checkpoint.sqlite import SqliteSaver

# JsonPlusSerializer: how LangGraph turns the notebook into bytes and back.
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import BaseModel

from .state import RunConfig, RunStatus

# Default folder for both database files, relative to where the program runs.
DEFAULT_RUNS_DIR = Path("runs")
CHECKPOINTS_FILE = "checkpoints.db"  # LangGraph's auto-saves
RUNS_FILE = "runs.db"  # our tables

# D1: idempotency keys are kept for 24 hours.
RECEIPT_LIFETIME = timedelta(hours=24)

# THE ALLOW-LIST. When LangGraph reads a saved checkpoint, it rebuilds Python
# objects from it. Rebuilding ANY class named in a file would be unsafe (a
# tampered file could make the program create something harmful), so LangGraph
# only rebuilds classes on this list. Without it, LangGraph warns that loading
# our classes "will be blocked in a future version" (seen on 2026-11-06).
# Each entry is (module name, class name). If you add a new class to the state,
# add it here too, or saved runs that contain it won't load.
ALLOWED_STATE_CLASSES: list[tuple[str, str]] = [
    ("orchestrator.state", name) for name in (
        "RunConstraints", "RunConfig", "Policy", "Values", "Uncertainty", "Prediction",
        "ChemistryViolation", "ThresholdViolation", "Decision", "NearMiss", "Candidate",
    )
]


def _connect(path: Path) -> sqlite3.Connection:
    """Open (or create) a SQLite database file with the settings D2 asks for.

      foreign_keys=ON      - SQLite enforces links between tables (off by default):
                             e.g. an audit event must belong to a run that exists
      journal_mode=WAL     - "write-ahead log": readers don't block while a run is
                             writing, so the dashboard can read a run while it works
      busy_timeout=5000    - if the file is busy (e.g. the API and the worker write
                             at the same moment), wait up to 5,000 ms (5 s) instead
                             of failing immediately (D2's 5-second busy timeout)

    check_same_thread=False lets the connection be used from more than one
    thread (one at a time); the worker's steps may run on a helper thread.
    """
    path.parent.mkdir(parents=True, exist_ok=True)  # create the folder if it doesn't exist
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")  # PRAGMA = a SQLite setting
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def make_serializer() -> JsonPlusSerializer:
    """LangGraph's converter (state <-> bytes) with OUR allow-list applied.

    Used by BOTH savers: the disk one below and the in-memory one that
    build_graph uses by default, so neither warns about our classes.
    """
    return JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_STATE_CLASSES)


def make_checkpointer(runs_dir: Path = DEFAULT_RUNS_DIR) -> SqliteSaver:
    """Create LangGraph's SQLite auto-save, writing to runs_dir/checkpoints.db.

    Pass the result to build_graph(tools, checkpointer=...). After that, every
    step of every run is saved to disk under the run's thread_id (its run_id).
    """
    saver = SqliteSaver(_connect(Path(runs_dir) / CHECKPOINTS_FILE), serde=make_serializer())
    saver.setup()  # creates LangGraph's tables in the file if they don't exist yet
    return saver


def _iso(moment: datetime) -> str:
    """A moment as UTC ISO 8601 text, the timestamp format D1 requires."""
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds")


def _now() -> str:
    """The current time as UTC ISO 8601 text."""
    return _iso(datetime.now(timezone.utc))


def body_fingerprint(body: dict) -> str:
    """A short fingerprint of a request body, for idempotency (D1).

    Two bodies with the same content (even with keys in a different order) give
    the same fingerprint; any change gives a different one. sort_keys=True puts
    keys in a fixed order first. sha256 is a standard fingerprint ("hash") method.
    """
    text = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


class RunRecord(BaseModel):
    """One row of the runs table, following D1's "Run" shape plus worker fields.

    D1 field not here yet: prediction_batch_id (needs D2's batch records).
    """

    run_id: str  # the run's unique ID (also LangGraph's thread_id)
    status: RunStatus  # queued / running / completed / failed / stopped
    iteration: int  # rounds finished so far
    objective: str  # the goal in plain English
    config: RunConfig  # what the user chose (targets, limits, partner)
    policy_version: str  # which approved policy the run uses (policies.py)
    knowledge_snapshot_id: Optional[str] = None  # D1: Kyle's knowledge snapshot (pinned, not yet verifiable)
    model_version: Optional[str] = None  # D1: Arleen's model version (pinned, not yet verifiable)
    created_at: str  # when the run was created (UTC ISO 8601)
    updated_at: str  # when this row last changed (UTC ISO 8601)
    stop_reason: Optional[str] = None  # why it ended, once it has
    failure: Optional[str] = None  # the error message, only for failed runs (D1)
    # Why it failed, as a short code (decision Q7): DEPENDENCY_UNAVAILABLE,
    # TOOL_ERROR, ALL_PREDICTIONS_FAILED, UNKNOWN_POLICY or INTERNAL_ERROR.
    failure_code: Optional[str] = None
    cancel_requested: bool = False  # set by POST /runs/{id}/cancel; the worker checks it between steps
    claimed_by: Optional[str] = None  # which worker is running it (None while queued)
    heartbeat_at: Optional[str] = None  # when that worker last said "still working on it"


class RunStore:
    """Our tables in runs_dir/runs.db: runs, receipts, workers, audit_events.

    Example:
        store = RunStore()
        store.create("abc-123", "find a donor for Y6", config, "placeholder-0", actor="cli")
        store.update("abc-123", actor="cli", action="run_started", status="running")
        store.get("abc-123").status   ->  "running"
    """

    # The table definitions, written in SQL (the language databases understand).
    # TEXT/INTEGER are column types; PRIMARY KEY = unique ID for each row; NOT
    # NULL = must have a value; CHECK = a rule the database itself enforces.
    _CREATE_TABLES = [
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id         TEXT PRIMARY KEY,
            status         TEXT NOT NULL CHECK (status IN
                               ('queued', 'running', 'completed', 'failed', 'stopped')),
            iteration      INTEGER NOT NULL DEFAULT 0,
            objective      TEXT NOT NULL,
            config_json    TEXT NOT NULL,
            policy_version TEXT NOT NULL,
            created_at     TEXT NOT NULL,
            updated_at     TEXT NOT NULL,
            stop_reason    TEXT,
            failure        TEXT
        )
        """,
        # Idempotency receipts (D1): the first reply to a POST, remembered for 24 h
        # so a repeated request (same key) gets the SAME reply instead of acting twice.
        # The PRIMARY KEY is the three columns together: the same key may be
        # reused by a different caller or on a different endpoint.
        """
        CREATE TABLE IF NOT EXISTS receipts (
            caller_id     TEXT NOT NULL,
            endpoint      TEXT NOT NULL,
            key           TEXT NOT NULL,
            body_hash     TEXT NOT NULL,
            status_code   INTEGER NOT NULL,
            response_json TEXT NOT NULL,
            created_at    TEXT NOT NULL,
            PRIMARY KEY (caller_id, endpoint, key)
        )
        """,
        # Each worker program's "last seen" time, so the API can tell whether any
        # worker is running at all.
        """
        CREATE TABLE IF NOT EXISTS workers (
            worker_id    TEXT PRIMARY KEY,
            last_seen_at TEXT NOT NULL
        )
        """,
        # The AUDIT LOG (D2 G4's AuditEvent, plus `seq` and `detail`):
        #   seq          - a number that only ever goes up (AUTOINCREMENT): the
        #                  reliable ORDER of events. D2: timestamps alone don't order
        #                  events (two can share the same second).
        #   event_id     - the event's unique ID (D2)
        #   run_id       - the run it's about. REFERENCES runs = it must be a real run
        #                  (a "foreign key", enforced because foreign_keys is ON)
        #   request_id   - the web request that caused it, if any
        #   actor_id     - WHO did it: "api", "worker:<id>", "cli". CHECK: never empty (D2)
        #   action       - WHAT happened, e.g. "run_created", "tool_attempt"
        #   result_code  - HOW it went, e.g. "ok", "temporary_error", "TOOL_ERROR"
        #   phase        - WHERE: a step or tool name, e.g. "generator"
        #   attempt      - which attempt (for retried tool calls)
        #   detail       - short extra info: codes and counts only, NEVER SMILES (D2)
        """
        CREATE TABLE IF NOT EXISTS audit_events (
            seq         INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id    TEXT NOT NULL UNIQUE,
            run_id      TEXT REFERENCES runs (run_id),
            request_id  TEXT,
            actor_id    TEXT NOT NULL CHECK (actor_id <> ''),
            action      TEXT NOT NULL,
            result_code TEXT NOT NULL,
            phase       TEXT,
            attempt     INTEGER,
            detail      TEXT,
            created_at  TEXT NOT NULL
        )
        """,
    ]
    # Columns added to the runs table AFTER item 4 (2026-11-09 and 2026-11-10). A
    # runs.db made earlier doesn't have them, so __init__ adds any that are missing
    # (a "migration": updating an old database to the new layout without losing data).
    _ADDED_RUN_COLUMNS = {
        "knowledge_snapshot_id": "TEXT",
        "model_version": "TEXT",
        "cancel_requested": "INTEGER NOT NULL DEFAULT 0",
        "claimed_by": "TEXT",
        "heartbeat_at": "TEXT",
        "failure_code": "TEXT",
    }
    # The columns update() is allowed to change. Anything else is refused, so a
    # typo can't silently write into the wrong place. (Claiming, heartbeats and
    # cancelling have their own methods below.)
    _UPDATABLE = {"status", "iteration", "stop_reason", "failure", "failure_code"}

    def __init__(self, runs_dir: Path = DEFAULT_RUNS_DIR):
        # Remember the folder, so another thread can open its OWN connection to the
        # same file (worker.py's heartbeat thread does). One SQLite connection must
        # not be used by two threads at the same time.
        self.runs_dir = Path(runs_dir)
        self.conn = _connect(self.runs_dir / RUNS_FILE)
        # Rows come back as sqlite3.Row, which can be read by column name: row["status"].
        self.conn.row_factory = sqlite3.Row
        with self.conn:  # "with conn" = commit the changes, or undo them if an error happens
            for sql in self._CREATE_TABLES:
                self.conn.execute(sql)
            # PRAGMA table_info lists a table's columns; row["name"] is each column's name.
            existing = {row["name"] for row in self.conn.execute("PRAGMA table_info(runs)")}
            for column, definition in self._ADDED_RUN_COLUMNS.items():
                if column not in existing:
                    self.conn.execute(f"ALTER TABLE runs ADD COLUMN {column} {definition}")

    def close(self) -> None:
        """Close this store's database connection (the API does this after each request)."""
        self.conn.close()

    # ------------------------------------------------------------------------
    # The audit log
    # ------------------------------------------------------------------------

    def _audit(self, run_id: Optional[str], *, actor: str, action: str, result_code: str = "ok",
               request_id: Optional[str] = None, phase: Optional[str] = None,
               attempt: Optional[int] = None, detail: Optional[str] = None) -> None:
        """Write one audit event. Called INSIDE the caller's transaction (`with self.conn:`),
        so the event and the change it describes are saved together, or not at all.

        An empty actor is refused (by the database's CHECK rule): D2 says an audit
        event without an actor is rejected, and that also undoes the change.
        """
        self.conn.execute(
            "INSERT INTO audit_events (event_id, run_id, request_id, actor_id, action, result_code, "
            "phase, attempt, detail, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (str(uuid4()), run_id, request_id, actor, action, result_code, phase, attempt, detail, _now()))

    def record_event(self, run_id: str, *, actor: str, action: str, result_code: str = "ok",
                     request_id: Optional[str] = None, phase: Optional[str] = None,
                     attempt: Optional[int] = None, detail: Optional[str] = None) -> None:
        """Write an audit event on its own (for events that aren't a change to the runs
        table, e.g. a retried tool call or a finished round)."""
        with self.conn:
            self._audit(run_id, actor=actor, action=action, result_code=result_code,
                        request_id=request_id, phase=phase, attempt=attempt, detail=detail)

    def list_events(self, run_id: str, limit: int = 50, after_seq: int = 0) -> list[dict]:
        """A run's audit events in order, one page at a time.

        after_seq: only events after this position (the "cursor" from the previous
        page; 0 = from the start). Returns plain dictionaries, oldest first.
        """
        rows = self.conn.execute(
            "SELECT * FROM audit_events WHERE run_id = ? AND seq > ? ORDER BY seq LIMIT ?",
            (run_id, after_seq, limit)).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------------
    # Creating, changing and reading runs
    # ------------------------------------------------------------------------

    def create(self, run_id: str, objective: str, config: RunConfig, policy_version: str,
               knowledge_snapshot_id: Optional[str] = None, model_version: Optional[str] = None,
               *, actor: str, request_id: Optional[str] = None) -> RunRecord:
        """Add a new run with status "queued" (D1: recorded, not started yet). Audited."""
        with self.conn:
            self._insert_run(run_id, objective, config, policy_version,
                             knowledge_snapshot_id, model_version)
            self._audit(run_id, actor=actor, action="run_created", request_id=request_id)
        return self.get(run_id)

    def _insert_run(self, run_id, objective, config, policy_version,
                    knowledge_snapshot_id, model_version, created_at: Optional[str] = None) -> None:
        """The INSERT itself, shared by create() and create_with_receipt().

        created_at can be given so the API's reply and the database show exactly
        the same time; otherwise it's now.
        """
        now = created_at or _now()
        # The ? marks are placeholders that SQLite fills in safely. Never build
        # SQL by pasting values into the text: that allows "SQL injection".
        self.conn.execute(
            "INSERT INTO runs (run_id, status, iteration, objective, config_json, policy_version, "
            "knowledge_snapshot_id, model_version, created_at, updated_at) "
            "VALUES (?, 'queued', 0, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, objective, config.model_dump_json(), policy_version,
             knowledge_snapshot_id, model_version, now, now),
        )

    def update(self, run_id: str, *, actor: str, action: str, result_code: str = "ok",
               request_id: Optional[str] = None, detail: Optional[str] = None,
               **changes) -> RunRecord:
        """Change some columns of a run AND record why, in one transaction.

        Example: update(run_id, actor="worker:w1", action="run_started", status="running")
        Every change needs an `action` (what happened) and an `actor` (who), so
        every change to a run appears in the audit log.

        Also refreshes updated_at. Raises KeyError if the run doesn't exist, and
        ValueError if asked to change a column that isn't allowed.
        (**changes collects the remaining name=value arguments into a dictionary.)
        """
        bad = set(changes) - self._UPDATABLE  # names that aren't allowed
        if bad:
            raise ValueError(f"cannot update column(s): {sorted(bad)}")
        changes["updated_at"] = _now()
        # Builds e.g. "status = ?, updated_at = ?". Column names come only from our
        # allowed set above, never from outside, so this is safe.
        assignments = ", ".join(f"{column} = ?" for column in changes)
        with self.conn:
            cursor = self.conn.execute(f"UPDATE runs SET {assignments} WHERE run_id = ?",
                                       (*changes.values(), run_id))
            if cursor.rowcount == 0:  # rowcount = how many rows were changed
                raise KeyError(f"no run with run_id {run_id!r}")
            self._audit(run_id, actor=actor, action=action, result_code=result_code,
                        request_id=request_id, detail=detail)
        return self.get(run_id)

    def get(self, run_id: str) -> Optional[RunRecord]:
        """Look up one run, or None if there is no run with that ID."""
        row = self.conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return self._to_record(row) if row else None

    def list(self, status: Optional[RunStatus] = None, limit: int = 20) -> list[RunRecord]:
        """The most recent runs first, optionally only those with one status.

        For Nyla's run picker (GET /runs?status=&limit=).
        """
        if status is None:
            rows = self.conn.execute(
                "SELECT * FROM runs ORDER BY created_at DESC, rowid DESC LIMIT ?", (limit,)).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM runs WHERE status = ? ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (status, limit)).fetchall()
        return [self._to_record(r) for r in rows]

    # ------------------------------------------------------------------------
    # Idempotency receipts (D1: POST requests carry an Idempotency-Key header)
    # ------------------------------------------------------------------------

    def find_receipt(self, caller_id: str, endpoint: str, key: str) -> Optional[dict]:
        """The remembered reply for this key, or None if there's none (or it's over 24 h old).

        Returns {"body_hash": ..., "status_code": ..., "response": {...}}.
        """
        row = self.conn.execute(
            "SELECT * FROM receipts WHERE caller_id = ? AND endpoint = ? AND key = ?",
            (caller_id, endpoint, key)).fetchone()
        if row is None or row["created_at"] < _iso(datetime.now(timezone.utc) - RECEIPT_LIFETIME):
            return None  # never seen, or expired (D1: keys are kept for 24 hours)
        return {"body_hash": row["body_hash"], "status_code": row["status_code"],
                "response": json.loads(row["response_json"])}

    def _insert_receipt(self, caller_id, endpoint, key, body_hash, status_code, response) -> None:
        """Remove an expired receipt for this key (so it can be reused), then insert the new one.

        Called inside the caller's transaction. Raises sqlite3.IntegrityError if a
        current receipt for the key already exists.
        """
        expired = _iso(datetime.now(timezone.utc) - RECEIPT_LIFETIME)
        self.conn.execute(
            "DELETE FROM receipts WHERE caller_id = ? AND endpoint = ? AND key = ? AND created_at < ?",
            (caller_id, endpoint, key, expired))
        self.conn.execute(
            "INSERT INTO receipts (caller_id, endpoint, key, body_hash, status_code, "
            "response_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (caller_id, endpoint, key, body_hash, status_code, json.dumps(response), _now()))

    def create_with_receipt(self, run_id: str, objective: str, config: RunConfig,
                            policy_version: str, knowledge_snapshot_id: Optional[str],
                            model_version: Optional[str], *, caller_id: str, endpoint: str,
                            key: str, body_hash: str, status_code: int, response: dict,
                            actor: str, request_id: Optional[str] = None) -> bool:
        """Create a queued run, remember the reply for its idempotency key, and audit it, together.

        All happen in ONE transaction: either all are saved or none. If the same
        key arrives twice at the same moment, only one receipt insert can succeed
        (the receipt's PRIMARY KEY), so only one run is created.

        Returns True if this call created the run, False if the key was already
        used (then nothing is created; the caller should reply with the stored receipt).
        """
        try:
            with self.conn:
                self._insert_receipt(caller_id, endpoint, key, body_hash, status_code, response)
                self._insert_run(run_id, objective, config, policy_version,
                                 knowledge_snapshot_id, model_version,
                                 created_at=response.get("created_at"))
                self._audit(run_id, actor=actor, action="run_created", request_id=request_id)
        except sqlite3.IntegrityError:  # the key is already in use -> everything was undone
            return False
        return True

    def save_receipt(self, *, caller_id: str, endpoint: str, key: str, body_hash: str,
                     status_code: int, response: dict) -> bool:
        """Remember the reply to a POST that doesn't create a run (e.g. cancel).

        Returns False if a receipt for this key already exists (nothing is changed).
        """
        try:
            with self.conn:
                self._insert_receipt(caller_id, endpoint, key, body_hash, status_code, response)
        except sqlite3.IntegrityError:
            return False
        return True

    # ------------------------------------------------------------------------
    # The worker side: claiming, heartbeats, cancelling, recovery
    # ------------------------------------------------------------------------

    def claim_next(self, worker_id: str) -> Optional[RunRecord]:
        """Claim the OLDEST queued run for this worker, or return None if none is queued.

        One SQL statement does it all: change the run to "running" and record the
        worker, but ONLY IF it is still "queued" at that instant. Two workers can
        never both claim the same run, because only one change can match.
        (RETURNING gives back the run_id of the row that was changed.) Audited.
        """
        now = _now()
        with self.conn:
            row = self.conn.execute(
                "UPDATE runs SET status = 'running', claimed_by = ?, heartbeat_at = ?, updated_at = ? "
                "WHERE run_id = (SELECT run_id FROM runs WHERE status = 'queued' "
                "                ORDER BY created_at, rowid LIMIT 1) "
                "AND status = 'queued' RETURNING run_id",
                (worker_id, now, now)).fetchone()
            if row:
                self._audit(row["run_id"], actor=f"worker:{worker_id}", action="run_claimed")
        return self.get(row["run_id"]) if row else None

    def claim_stale(self, worker_id: str, stale_after: timedelta,
                    now: Optional[datetime] = None) -> Optional[RunRecord]:
        """Take over ONE "running" run whose worker has stopped sending heartbeats.

        A run is "stale" if its heartbeat is older than `stale_after` (or missing):
        its worker probably crashed. The new worker claims it (only if the
        heartbeat is still stale at that instant) and can then resume it.
        `now` exists so tests can pretend time has passed. Audited.
        """
        moment = now or datetime.now(timezone.utc)
        cutoff = _iso(moment - stale_after)  # heartbeats older than this are stale
        with self.conn:
            row = self.conn.execute(
                "UPDATE runs SET claimed_by = ?, heartbeat_at = ?, updated_at = ? "
                "WHERE run_id = (SELECT run_id FROM runs WHERE status = 'running' "
                "                AND (heartbeat_at IS NULL OR heartbeat_at < ?) "
                "                ORDER BY created_at, rowid LIMIT 1) "
                "AND status = 'running' AND (heartbeat_at IS NULL OR heartbeat_at < ?) "
                "RETURNING run_id",
                (worker_id, _iso(moment), _iso(moment), cutoff, cutoff)).fetchone()
            if row:
                self._audit(row["run_id"], actor=f"worker:{worker_id}", action="run_taken_over",
                            detail=f"no heartbeat for over {int(stale_after.total_seconds())} s")
        return self.get(row["run_id"]) if row else None

    def heartbeat(self, run_id: str, worker_id: str) -> bool:
        """The worker says "still working on this run". Returns False if it no longer owns it.

        Only the worker named in claimed_by can update the heartbeat. If another
        worker has taken the run over (e.g. after this one froze for a while),
        this returns False and this worker should stop working on it, so two
        workers never both keep going on one run. (Not audited: it happens every
        20 seconds and would drown out the events that matter.)
        """
        with self.conn:
            cursor = self.conn.execute(
                "UPDATE runs SET heartbeat_at = ? WHERE run_id = ? AND claimed_by = ?",
                (_now(), run_id, worker_id))
        return cursor.rowcount > 0

    def request_cancel(self, run_id: str, *, actor: str, request_id: Optional[str] = None) -> RunRecord:
        """Mark a run as "please stop" (POST /runs/{id}/cancel). The worker checks it between steps.

        A queued run that no worker has started yet is stopped straight away.
        Raises KeyError if the run doesn't exist. Whether the run has already
        finished (D2: 409 RUN_CLOSED) is checked by the API before calling this. Audited.
        """
        now = _now()
        with self.conn:
            cursor = self.conn.execute(
                "UPDATE runs SET cancel_requested = 1, updated_at = ? WHERE run_id = ?", (now, run_id))
            if cursor.rowcount == 0:
                raise KeyError(f"no run with run_id {run_id!r}")
            self._audit(run_id, actor=actor, action="cancel_requested", request_id=request_id)
            # Never started -> nothing to interrupt: end it now.
            stopped = self.conn.execute(
                "UPDATE runs SET status = 'stopped', stop_reason = 'cancelled', updated_at = ? "
                "WHERE run_id = ? AND status = 'queued'", (now, run_id))
            if stopped.rowcount:
                self._audit(run_id, actor=actor, action="run_cancelled", request_id=request_id,
                            detail="cancelled before any worker started it")
        return self.get(run_id)

    def is_cancel_requested(self, run_id: str) -> bool:
        """True if someone asked this run to stop."""
        row = self.conn.execute("SELECT cancel_requested FROM runs WHERE run_id = ?",
                                (run_id,)).fetchone()
        return bool(row and row["cancel_requested"])

    def worker_seen(self, worker_id: str) -> None:
        """Record that this worker program is alive right now."""
        with self.conn:
            self.conn.execute(
                "INSERT INTO workers (worker_id, last_seen_at) VALUES (?, ?) "
                "ON CONFLICT (worker_id) DO UPDATE SET last_seen_at = excluded.last_seen_at",
                (worker_id, _now()))

    def last_worker_seen(self) -> Optional[str]:
        """When ANY worker was last seen (UTC ISO 8601), or None if no worker ever ran."""
        row = self.conn.execute("SELECT MAX(last_seen_at) AS seen FROM workers").fetchone()
        return row["seen"] if row else None

    @staticmethod
    def _to_record(row: sqlite3.Row) -> RunRecord:
        """Turn one database row into a RunRecord (the config column is JSON text)."""
        data = dict(row)  # a sqlite3.Row can be turned into a normal dictionary
        data["config"] = RunConfig.model_validate(json.loads(data.pop("config_json")))
        return RunRecord(**data)  # ** spreads the dictionary out as name=value arguments
