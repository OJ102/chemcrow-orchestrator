"""Starting and resuming runs, and keeping the runs table up to date.

WHAT THIS FILE IS
-----------------
graph.py knows how to RUN the loop; storage.py knows how to SAVE things. This
file connects them, giving two simple functions that the web API (item 5)
will call:

  start_run(...)    - record a new run, run it, record how it ended
  resume_run(...)   - continue a run that crashed, from its last finished step

Each run's run_id is used as LangGraph's thread_id (its save-slot name), as
agreed with Nyla, so there's one ID per run.

Status flow in the runs table (D1 statuses):
  queued -> running -> completed | stopped | failed
  If the program throws an error, the run is marked "failed" with the error
  message, so it never stays stuck on "running". A failed or interrupted run
  can be resumed, which sets it back to "running".
  (Item 6 adds retries before giving up, and the audit log.)

Usage:
    from orchestrator.storage import make_checkpointer, RunStore
    graph = build_graph(tools, checkpointer=make_checkpointer())
    record = start_run(graph, RunStore(), objective="...", config=..., policy=...)
    record.status   ->  e.g. "completed"
"""

# uuid4(): a new random unique ID for each run.
from uuid import uuid4

from .state import Policy, RunConfig
from .storage import RunRecord, RunStore

# Statuses that mean "this run is over and can't be continued".
# (failed is NOT here: a failed run can be resumed once the problem is fixed.)
FINISHED = {"completed", "stopped"}


def _thread(run_id: str) -> dict:
    """The LangGraph config that selects this run's save slot (thread_id = run_id)."""
    return {"configurable": {"thread_id": run_id}}


def start_run(graph, store: RunStore, *, objective: str, config: RunConfig, policy: Policy,
              run_id: str | None = None) -> RunRecord:
    """Record a new run, run it to the end, and return its final runs-table row.

    Parameters (the * means everything after it must be given by name):
        graph     - from build_graph(...). It must have a checkpointer that saves
                    to disk (storage.make_checkpointer) for resume to work later.
        store     - the runs table (storage.RunStore)
        objective - the goal in plain English
        config    - the run's settings (targets, partner, limits)
        policy    - the approved scientific limits
        run_id    - normally left out: a fresh random ID is made. Tests pass one.

    Returns the run's row after it ends: status completed, stopped or failed.
    """
    run_id = run_id or str(uuid4())  # `a or b` = a, unless a is empty/None, then b
    store.create(run_id, objective, config, policy.policy_version)  # status: queued
    store.update(run_id, status="running")
    inputs = {"run_id": run_id, "objective": objective, "config": config, "policy": policy}
    return _drive(graph, store, run_id, inputs)


def resume_run(graph, store: RunStore, run_id: str) -> RunRecord:
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
    store.update(run_id, status="running", failure=None)  # clear the old error message
    # Passing None instead of inputs tells LangGraph: "don't start over, continue
    # this thread from its last checkpoint".
    return _drive(graph, store, run_id, None)


def _drive(graph, store: RunStore, run_id: str, inputs) -> RunRecord:
    """Run (or continue) the graph, then write the outcome into the runs table.

    Any error is caught and recorded as status "failed" with the error message
    (D1: `failure` is set only for failed runs), instead of crashing the caller,
    which will be the web API. The state saved up to the error stays in the
    checkpoint database, so resume_run can pick up from there.
    """
    try:
        final = graph.invoke(inputs, _thread(run_id))
    except Exception as error:  # deliberately broad: ANY error must be recorded, not lost
        saved = graph.get_state(_thread(run_id)).values  # what was saved before the error
        return store.update(run_id, status="failed", iteration=saved.get("iteration", 0),
                            failure=f"{type(error).__name__}: {error}")
    status = final["status"]
    return store.update(run_id, status=status, iteration=final["iteration"],
                        stop_reason=final.get("stop_reason"),
                        # A run can also END as failed (every prediction failed). The
                        # reason then also goes in `failure`, as D1 expects.
                        failure=final.get("stop_reason") if status == "failed" else None)
