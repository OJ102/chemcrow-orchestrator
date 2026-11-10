"""Starting, resuming and reading runs, and keeping the runs table up to date.

WHAT THIS FILE IS
-----------------
graph.py knows how to RUN the loop; storage.py knows how to SAVE things. This
file connects them:

  start_run(...)    - record a new run and run it straight away (the demo's --db mode)
  run_claimed(...)  - run a run the WORKER has claimed: from the start if
                      nothing was saved yet, otherwise resume from the last step
  resume_run(...)   - continue a run that crashed, from its last finished step
  saved_state(...)  - read a run's saved results without running anything
                      (the web API uses this for the shortlist and the report)

Each run's run_id is used as LangGraph's thread_id (its save-slot name), as
agreed with Nyla, so there's one ID per run.

Status flow in the runs table (D1 statuses):
  queued -> running -> completed | stopped | failed
  If the run hits an error, it's marked "failed" with a FAILURE CODE and the
  error message (decision Q7, 2026-11-10), so it never stays stuck on "running":
    DEPENDENCY_UNAVAILABLE  a tool kept failing TEMPORARILY; all 3 attempts used (retry.py)
    TOOL_ERROR              a tool failed PERMANENTLY (not retried)
    ALL_PREDICTIONS_FAILED  every prediction in a round failed
    UNKNOWN_POLICY          the run names a policy that isn't registered (worker.py)
    INTERNAL_ERROR          anything else: a bug in our own code
  D2 says a failed run should be replaced by a NEW run; the command-line
  --resume can still resume one, which is handy while developing.

Every change is written to the AUDIT LOG together with WHO made it (`actor`):
"cli" for the command line, "worker:<id>" for the worker.
"""

# Callable: the type of "something you can call like a function".
from typing import Callable

# uuid4(): a new random unique ID for each run.
from uuid import uuid4

from .retry import RetriesExhausted, ToolFailed
from .state import Policy, RunConfig
from .storage import RunRecord, RunStore

# Statuses that mean "this run is over and can't be continued".
# (failed is NOT here: a failed run can be resumed once the problem is fixed.)
FINISHED = {"completed", "stopped"}


def _thread(run_id: str) -> dict:
    """The LangGraph config that selects this run's save slot (thread_id = run_id)."""
    return {"configurable": {"thread_id": run_id}}


def saved_state(checkpointer, run_id: str) -> dict:
    """A run's latest saved state (the shared notebook), or {} if nothing is saved yet.

    Reads straight from the checkpoint database, without building or running a
    graph. `checkpointer.get(...)` returns the latest checkpoint, whose
    "channel_values" are the state's keys and values.
    """
    checkpoint = checkpointer.get(_thread(run_id))
    return dict(checkpoint["channel_values"]) if checkpoint else {}


def failure_of(error: BaseException) -> tuple[str, str]:
    """Turn an error into (failure code, message) for the runs table (decision Q7)."""
    if isinstance(error, RetriesExhausted):
        return "DEPENDENCY_UNAVAILABLE", str(error)
    if isinstance(error, ToolFailed):
        return "TOOL_ERROR", str(error)
    return "INTERNAL_ERROR", f"{type(error).__name__}: {error}"


def start_run(graph, store: RunStore, *, objective: str, config: RunConfig, policy: Policy,
              run_id: str | None = None, actor: str = "cli") -> RunRecord:
    """Record a new run, run it to the end, and return its final runs-table row.

    Parameters (the * means everything after it must be given by name):
        graph     - from build_graph(...). It must have a checkpointer that saves
                    to disk (storage.make_checkpointer) for resume to work later.
        store     - the runs table (storage.RunStore)
        objective - the goal in plain English
        config    - the run's settings (targets, partner, limits)
        policy    - the approved scientific limits
        run_id    - normally left out: a fresh random ID is made. Tests pass one.
        actor     - who is doing this, for the audit log

    Returns the run's row after it ends: status completed, stopped or failed.
    """
    run_id = run_id or str(uuid4())  # `a or b` = a, unless a is empty/None, then b
    store.create(run_id, objective, config, policy.policy_version, actor=actor)  # status: queued
    store.update(run_id, actor=actor, action="run_started", status="running")
    inputs = {"run_id": run_id, "objective": objective, "config": config, "policy": policy}
    return _drive(graph, store, run_id, inputs, actor)


def run_claimed(graph, store: RunStore, record: RunRecord, policy: Policy, *, actor: str,
                still_owner: Callable[[], bool] = lambda: True) -> RunRecord:
    """Run a run that a worker has just claimed (status "running"), to the end.

    - Nothing saved yet (a fresh claim)  -> start from the beginning, using the
      settings stored in the runs table.
    - Something saved (a stale run taken over from a crashed worker)
      -> RESUME from the last finished step; finished rounds aren't redone.

    still_owner: a function the worker provides that returns False if another
    worker has taken this run over (its heartbeat was refused). Then this worker
    must NOT write the final status, because the run belongs to someone else now.
    """
    if saved_state(graph.checkpointer, record.run_id):
        store.record_event(record.run_id, actor=actor, action="run_resumed")
        inputs = None  # None = "continue this save slot from its last checkpoint"
    else:
        store.record_event(record.run_id, actor=actor, action="run_started")
        inputs = {"run_id": record.run_id, "objective": record.objective,
                  "config": record.config, "policy": policy}
    return _drive(graph, store, record.run_id, inputs, actor, still_owner)


def resume_run(graph, store: RunStore, run_id: str, actor: str = "cli") -> RunRecord:
    """Continue a run from its last saved step, e.g. after a crash.

    Works for runs marked "failed" (an error was recorded) or still "running"
    (the program died before it could record anything). Refuses runs that
    already finished normally (D2: a closed run can't be continued).

    `graph` must use the SAME checkpoint database the run was saved in.
    """
    record = store.get(run_id)
    if record is None:
        raise KeyError(f"no run with run_id {run_id!r}")
    if record.status in FINISHED:
        raise ValueError(f"run {run_id!r} already finished with status {record.status!r}")
    # Clear the old error, and record that the run is being resumed.
    store.update(run_id, actor=actor, action="run_resumed", status="running",
                 failure=None, failure_code=None)
    # Passing None instead of inputs tells LangGraph: "don't start over, continue
    # this thread from its last checkpoint".
    return _drive(graph, store, run_id, None, actor)


def _drive(graph, store: RunStore, run_id: str, inputs, actor: str,
           still_owner: Callable[[], bool] = lambda: True) -> RunRecord:
    """Run (or continue) the graph, then write the outcome into the runs table and audit log.

    Any error is caught and recorded as status "failed" with a failure code and
    message (D1: `failure` is set only for failed runs), instead of crashing the
    caller. The state saved up to the error stays in the checkpoint database.

    If still_owner() says this worker lost the run to another worker, nothing is
    written, and the row is returned as it is.
    """
    try:
        final = graph.invoke(inputs, _thread(run_id))
    except Exception as error:  # deliberately broad: ANY error must be recorded, not lost
        if not still_owner():
            return store.get(run_id)
        code, message = failure_of(error)
        saved = saved_state(graph.checkpointer, run_id)  # what was saved before the error
        return store.update(run_id, actor=actor, action="run_failed", result_code=code,
                            status="failed", iteration=saved.get("iteration", 0),
                            failure=message, failure_code=code)
    if not still_owner():
        return store.get(run_id)
    status, reason = final["status"], final.get("stop_reason")
    # The audit action names the outcome: run_completed / run_stopped / run_failed,
    # or run_cancelled for a run stopped by a cancel request.
    action = "run_cancelled" if reason == "cancelled" else f"run_{status}"
    # A run can also END as failed without an error (every prediction failed).
    code = "ALL_PREDICTIONS_FAILED" if status == "failed" else None
    return store.update(run_id, actor=actor, action=action, result_code=code or "ok",
                        status=status, iteration=final["iteration"], stop_reason=reason,
                        failure=reason if status == "failed" else None, failure_code=code)
