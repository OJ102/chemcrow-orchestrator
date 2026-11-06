"""Saving runs to disk: the LangGraph checkpoint database and our runs table.

WHAT THIS FILE IS
-----------------
Before this file, runs were saved only in memory, so closing the program lost
them. This file keeps them in SQLite DATABASE FILES on disk, so a run can be:
  - looked up later by its run_id (Nyla's dashboard, via the web API in item 5),
  - RESUMED after a crash from the last step that finished.

SQLite = a small database stored in ONE ordinary file. Python can use it
directly (the built-in `sqlite3` module); no database server is needed. Inside
the file are TABLES: grids with named columns and one row per record.

TWO FILES (decision, 2026-11-06; D2 describes a single store, we chose two):
  checkpoints.db - LangGraph's own auto-saves: the whole shared notebook
                   (OPVState) after EVERY step. Used to resume a run. Its
                   internal format belongs to LangGraph; we don't read it directly.
  runs.db        - OUR table, one row per run, with D1's "Run" fields (status,
                   iteration, config, timestamps, ...). Quick to search and
                   list, which is what a dashboard needs.

Both files go in one folder (default "runs/", ignored by git). The folder is
configurable, so later it can point at the shared server Nyla can reach.

WHAT IS DEFERRED (decision Q4, answering peer review PR-03): D2 also describes
worker leases and fencing tokens (needed only when several workers share runs)
and idempotency receipts (needed for repeated web requests, item 5). Phase 1
has one worker, so they're not built here.
"""

# json: converts between Python data and JSON text (used for the `config` column).
import json

# sqlite3: Python's built-in SQLite library.
import sqlite3

# datetime/timezone: for timestamps. D1 wants "UTC ISO 8601" text, e.g.
# "2026-11-06T15:04:05+00:00" (UTC = the same clock everywhere, no time zones).
from datetime import datetime, timezone

# Path: a tidy way to work with file and folder paths.
from pathlib import Path
from typing import Optional

# The SQLite auto-save from the langgraph-checkpoint-sqlite package (installed 2026-11-06).
from langgraph.checkpoint.sqlite import SqliteSaver

# JsonPlusSerializer: how LangGraph turns the notebook into bytes and back.
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import BaseModel

from .state import RunConfig, RunStatus

# Default folder for both database files, relative to where the program runs.
DEFAULT_RUNS_DIR = Path("runs")
CHECKPOINTS_FILE = "checkpoints.db"  # LangGraph's auto-saves
RUNS_FILE = "runs.db"  # our runs table

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

      foreign_keys=ON      - SQLite enforces links between tables (off by default)
      journal_mode=WAL     - "write-ahead log": readers don't block while a run is
                             writing, so the dashboard can read a run while it works
      busy_timeout=5000    - if the file is busy, wait up to 5,000 ms (5 s) instead
                             of failing immediately (D2's 5-second busy timeout)

    check_same_thread=False lets the connection be used from more than one
    thread. The web server in item 5 will need that.
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


def _now() -> str:
    """The current time as UTC ISO 8601 text, the timestamp format D1 requires."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RunRecord(BaseModel):
    """One row of our runs table, following D1's "Run" shape.

    D1 fields not here yet: knowledge_snapshot_id and model_version (they arrive
    with the API and pinned versions in item 5) and prediction_batch_id (needs
    D2's batch records).
    """

    run_id: str  # the run's unique ID (also LangGraph's thread_id)
    status: RunStatus  # queued / running / completed / failed / stopped
    iteration: int  # rounds finished so far
    objective: str  # the goal in plain English
    config: RunConfig  # what the user chose (targets, limits, partner)
    policy_version: str  # which approved policy the run used (D1: pinned per run)
    created_at: str  # when the run was created (UTC ISO 8601)
    updated_at: str  # when this row last changed (UTC ISO 8601)
    stop_reason: Optional[str] = None  # why it ended, once it has
    failure: Optional[str] = None  # the error message, only for failed runs (D1)


class RunStore:
    """Our runs table in runs_dir/runs.db: create, update, look up and list runs.

    Example:
        store = RunStore()
        store.create("abc-123", "find a donor for Y6", config, "placeholder-0")
        store.update("abc-123", status="running")
        store.get("abc-123").status   ->  "running"
    """

    # The table definition, written in SQL (the language databases understand).
    # TEXT/INTEGER are column types; PRIMARY KEY = unique ID for each row; NOT
    # NULL = must have a value; CHECK = a rule the database itself enforces.
    _CREATE_TABLE = """
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
    """
    # The columns update() is allowed to change. Anything else is refused, so a
    # typo can't silently write into the wrong place.
    _UPDATABLE = {"status", "iteration", "stop_reason", "failure"}

    def __init__(self, runs_dir: Path = DEFAULT_RUNS_DIR):
        self.conn = _connect(Path(runs_dir) / RUNS_FILE)
        # Rows come back as sqlite3.Row, which can be read by column name: row["status"].
        self.conn.row_factory = sqlite3.Row
        with self.conn:  # "with conn" = commit the change, or undo it if an error happens
            self.conn.execute(self._CREATE_TABLE)

    def create(self, run_id: str, objective: str, config: RunConfig, policy_version: str) -> RunRecord:
        """Add a new run with status "queued" (D1: recorded, not started yet)."""
        now = _now()
        with self.conn:
            # The ? marks are placeholders that SQLite fills in safely. Never build
            # SQL by pasting values into the text: that allows "SQL injection".
            self.conn.execute(
                "INSERT INTO runs (run_id, status, iteration, objective, config_json, "
                "policy_version, created_at, updated_at) VALUES (?, 'queued', 0, ?, ?, ?, ?, ?)",
                (run_id, objective, config.model_dump_json(), policy_version, now, now),
            )
        return self.get(run_id)

    def update(self, run_id: str, **changes) -> RunRecord:
        """Change some columns of a run, e.g. update(run_id, status="running").

        Also refreshes updated_at. Raises KeyError if the run doesn't exist, and
        ValueError if asked to change a column that isn't allowed.
        (**changes collects the name=value arguments into a dictionary.)
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
        return self.get(run_id)

    def get(self, run_id: str) -> Optional[RunRecord]:
        """Look up one run, or None if there is no run with that ID."""
        row = self.conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return self._to_record(row) if row else None

    def list(self, status: Optional[RunStatus] = None, limit: int = 20) -> list[RunRecord]:
        """The most recent runs first, optionally only those with one status.

        For Nyla's run picker (her requested GET /runs?status=&limit=).
        """
        if status is None:
            rows = self.conn.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM runs WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (status, limit)).fetchall()
        return [self._to_record(r) for r in rows]

    @staticmethod
    def _to_record(row: sqlite3.Row) -> RunRecord:
        """Turn one database row into a RunRecord (the config column is JSON text)."""
        data = dict(row)  # a sqlite3.Row can be turned into a normal dictionary
        data["config"] = RunConfig.model_validate(json.loads(data.pop("config_json")))
        return RunRecord(**data)  # ** spreads the dictionary out as name=value arguments
