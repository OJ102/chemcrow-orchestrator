"""The WORKER program: picks queued runs from the database and runs them.

WHAT THIS FILE IS
-----------------
The web API (api.py) never runs a run itself: it only records it as "queued"
in runs.db and replies straight away (D1: 202 Accepted). This separate program
does the actual work (decision Q2, 2026-11-08):

    loop forever:
        1. say "I'm alive"                    (workers table, so the API can tell)
        2. take over a STALE run if there is  (a "running" run whose worker stopped
           one, otherwise claim the oldest       sending heartbeats, i.e. crashed)
           QUEUED run
        3. run it, sending a HEARTBEAT every  (so nobody mistakes it for a dead run)
           few seconds from a background thread
        4. nothing to do? wait a moment, then check again ("polling")

The API and the worker never talk directly. Everything goes through runs.db
(see storage.py for how claiming, heartbeats and cancelling work).

Start it with:   python -m orchestrator worker --db runs
(On the cluster, run it inside `tmux` so it keeps going after you disconnect.)

WHY A SEPARATE PROGRAM: a slow or broken run can never slow down or crash the
API, and later the heavy models (MatGPT, Chemprop) load only here, once.
"""

# os.getpid(): this program's process number; socket.gethostname(): the computer's
# name. Both go into the worker's ID, so you can tell workers apart.
import os
import socket

# threading: runs the heartbeat in the background while the run works.
import threading
from datetime import timedelta
from typing import Mapping, Optional
from uuid import uuid4

from .policies import POLICIES
from .runner import run_claimed
from .state import Policy
from .storage import RunRecord, RunStore

# Timing, from D2's lease design: a heartbeat every 20 s; a run whose heartbeat is
# older than 60 s is treated as abandoned (its worker crashed).
HEARTBEAT_EVERY = 20.0  # seconds between heartbeats
STALE_AFTER = timedelta(seconds=60)  # silence after which a running run counts as abandoned
POLL_EVERY = 2.0  # seconds to wait before checking again when there's nothing to do


class Heartbeat:
    """Sends "still working on run X" every few seconds from a background thread.

    Use it with `with`:
        with Heartbeat(store.runs_dir, run_id, worker_id) as beat:
            ... do the run ...
            if beat.lost: ...   # another worker took the run over

    It opens its OWN database connection (one connection must not be shared by
    two threads). If a heartbeat is refused, another worker has taken the run
    over, so `lost` becomes True and the run should stop.
    """

    def __init__(self, runs_dir, run_id: str, worker_id: str, every: float = HEARTBEAT_EVERY):
        self.runs_dir, self.run_id, self.worker_id, self.every = runs_dir, run_id, worker_id, every
        self.lost = False  # becomes True if another worker took the run over
        self._stop = threading.Event()  # an on/off signal threads can wait on
        self._thread = threading.Thread(target=self._beat, daemon=True)
        # daemon=True: the thread never keeps the program alive on its own.

    def _beat(self) -> None:
        """The background thread: heartbeat, wait `every` seconds, repeat until stopped."""
        store = RunStore(self.runs_dir)  # this thread's own connection
        # _stop.wait(t) waits up to t seconds; it returns True early if stop was signalled.
        while not self._stop.wait(self.every):
            if not store.heartbeat(self.run_id, self.worker_id):
                self.lost = True
                return

    def __enter__(self) -> "Heartbeat":
        """Called at the start of the `with` block: start the background thread."""
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        """Called at the end of the `with` block (even after an error): stop the thread."""
        self._stop.set()
        self._thread.join()  # wait for the thread to finish


class Worker:
    """Claims and runs queued runs, one at a time (D2 Phase 1: one worker).

    Parameters:
        graph           - from build_graph(..., checkpointer=make_checkpointer(runs_dir),
                          should_cancel=store.is_cancel_requested)
        store           - the runs table (storage.RunStore) for the same runs_dir
        policies        - version name -> Policy (default: the registry in policies.py)
        worker_id       - a name for this worker; default "<computer>-<process id>-<random>"
        heartbeat_every - seconds between heartbeats (D2: 20)
        stale_after     - silence after which another worker's run counts as abandoned (D2: 60 s)
    """

    def __init__(self, graph, store: RunStore, policies: Mapping[str, Policy] = POLICIES,
                 worker_id: Optional[str] = None, heartbeat_every: float = HEARTBEAT_EVERY,
                 stale_after: timedelta = STALE_AFTER):
        self.graph, self.store, self.policies = graph, store, policies
        # os.getpid() = this program's process number. uuid4().hex[:6] = 6 random characters.
        self.worker_id = worker_id or f"{socket.gethostname()}-{os.getpid()}-{uuid4().hex[:6]}"
        self.heartbeat_every, self.stale_after = heartbeat_every, stale_after

    def process_next_run(self) -> Optional[RunRecord]:
        """Do ONE unit of work: claim one run and run it to the end.

        Returns the run's final row, or None if there was nothing to do.
        Abandoned (stale) runs are recovered BEFORE new queued runs are started.
        Tests call this directly, so they never need a second program.
        """
        self.store.worker_seen(self.worker_id)
        record = (self.store.claim_stale(self.worker_id, self.stale_after)
                  or self.store.claim_next(self.worker_id))
        if record is None:
            return None
        policy = self.policies.get(record.policy_version)
        if policy is None:  # the API checks this too; this guards against a registry change
            return self.store.update(record.run_id, status="failed",
                                     failure=f"unknown policy_version {record.policy_version!r}")
        with Heartbeat(self.store.runs_dir, record.run_id, self.worker_id,
                       self.heartbeat_every) as beat:
            # still_owner: "has nobody taken this run over?" (lambda = a tiny function)
            return run_claimed(self.graph, self.store, record, policy,
                               still_owner=lambda: not beat.lost)

    def run_forever(self, poll_every: float = POLL_EVERY,
                    stop: Optional[threading.Event] = None) -> None:
        """Keep processing runs until `stop` is set (or the program is interrupted, Ctrl+C)."""
        stop = stop or threading.Event()
        # flush=True writes each line out immediately. Otherwise, when the output goes
        # to a log file, Python holds lines back and they can be lost if the worker is killed.
        print(f"worker {self.worker_id} started; checking for runs every {poll_every:g} s", flush=True)
        while not stop.is_set():
            record = self.process_next_run()
            if record is not None:
                print(f"run {record.run_id}: {record.status}"
                      + (f" ({record.stop_reason})" if record.stop_reason else "")
                      + (f" - {record.failure}" if record.failure and not record.stop_reason else ""),
                      flush=True)
            else:
                stop.wait(poll_every)  # nothing to do: wait, then check again
