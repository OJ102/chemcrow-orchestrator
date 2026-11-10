"""Tests for the worker program (worker.py) and cancelling between steps (graph.py).

Run from the repo root, in the langgraph venv:
    python -m pytest orchestrator/tests -v

The worker is normally a separate, never-ending program. Its core is one
function, process_next_run() ("claim one run and run it"), so these tests call
that directly. No second program is needed.
"""

import time

from orchestrator import Tools, build_graph
from orchestrator.gate import PlaceholderGate, PlaceholderRanker
from orchestrator.runner import run_claimed, saved_state
from orchestrator.storage import RunStore, make_checkpointer
from orchestrator.stubs import StubRetriever
from orchestrator.tests.fakes import (CONFIG, OPAQUE, POLICY, FakeClock, ListGenerator,
                                      TablePredictor)
from orchestrator.tests.test_storage import THREE_ROUNDS, CrashOnceGenerator
from orchestrator.worker import Heartbeat, Worker

# The policy registry for these tests: only the test policy (fakes.py).
TEST_POLICIES = {POLICY.policy_version: POLICY}


def make_worker(tmp_path, generator, default=OPAQUE):
    """A worker on tmp_path's database, using the given fake generator.

    Returns (store, worker). Heartbeats every 0.05 s so tests don't wait long.
    should_cancel=store.is_cancel_requested is exactly how the real worker
    command wires up cancelling.
    """
    store = RunStore(tmp_path)
    tools = Tools(retriever=StubRetriever(), generator=generator,
                  predictor=TablePredictor({}, default),
                  gate=PlaceholderGate(), ranker=PlaceholderRanker())
    graph = build_graph(tools, checkpointer=make_checkpointer(tmp_path), clock=FakeClock(),
                        should_cancel=store.is_cancel_requested)
    return store, Worker(graph, store, policies=TEST_POLICIES, worker_id="w-test",
                         heartbeat_every=0.05)


def queue(store: RunStore, run_id: str, policy_version: str = POLICY.policy_version):
    """Record a queued run, as the web API would."""
    return store.create(run_id, "test", CONFIG, policy_version)


# --- Basic work ------------------------------------------------------------------------

def test_nothing_queued_means_nothing_to_do(tmp_path):
    store, worker = make_worker(tmp_path, ListGenerator([]))
    assert worker.process_next_run() is None
    assert store.last_worker_seen() is not None  # it still said "I'm alive"


def test_worker_runs_a_queued_run_to_the_end(tmp_path):
    """OPAQUE molecules are always revised, so the run uses all 3 rounds and stops."""
    store, worker = make_worker(tmp_path, ListGenerator(THREE_ROUNDS))
    queue(store, "r1")
    record = worker.process_next_run()
    assert record.run_id == "r1" and record.claimed_by == "w-test"
    assert record.status == "stopped" and record.stop_reason == "iteration budget (3) exhausted"
    assert len(saved_state(worker.graph.checkpointer, "r1")["history"]) == 3


def test_oldest_queued_run_goes_first(tmp_path):
    store, worker = make_worker(tmp_path, ListGenerator([["CCO"], ["CCN"]]))
    queue(store, "first")
    queue(store, "second")
    assert worker.process_next_run().run_id == "first"
    assert worker.process_next_run().run_id == "second"


def test_unknown_policy_fails_the_run(tmp_path):
    store, worker = make_worker(tmp_path, ListGenerator([]))
    queue(store, "r1", policy_version="no-such-policy")
    record = worker.process_next_run()
    assert record.status == "failed" and "unknown policy_version" in record.failure


# --- Cancelling --------------------------------------------------------------------------

class CancelInRound(ListGenerator):
    """A fake generator that, during round `n`, does what a dashboard user might:
    asks for the run to be cancelled (as POST /runs/{id}/cancel would)."""

    def __init__(self, batches, store: RunStore, run_id: str, n: int):
        super().__init__(batches)
        self.store, self.run_id, self.n = store, run_id, n

    def generate_validate(self, packet):
        if packet.iteration == self.n:
            self.store.request_cancel(self.run_id)
        return super().generate_validate(packet)


def test_cancel_stops_the_run_before_its_next_step_and_keeps_earlier_work(tmp_path):
    """Cancel arrives during round 2. Round 2's remaining steps are skipped; the
    run ends stopped / "cancelled"; round 1's candidate is kept."""
    # The fake generator needs its own connection to the same database (like the API has).
    generator = CancelInRound(THREE_ROUNDS, RunStore(tmp_path), "r1", n=2)
    store, worker = make_worker(tmp_path, generator)
    queue(store, "r1")
    record = worker.process_next_run()
    assert record.status == "stopped" and record.stop_reason == "cancelled"
    saved = saved_state(worker.graph.checkpointer, "r1")
    assert [c.iteration for c in saved["history"]] == [1]  # round 1 kept; round 2 never judged


def test_cancelling_a_queued_run_stops_it_without_a_worker(tmp_path):
    store, worker = make_worker(tmp_path, ListGenerator([]))
    queue(store, "r1")
    record = store.request_cancel("r1")
    assert record.status == "stopped" and record.stop_reason == "cancelled"
    assert worker.process_next_run() is None  # nothing left to claim


# --- Crashed workers: stale runs, heartbeats ---------------------------------------------

def test_a_stale_run_is_taken_over_and_resumed(tmp_path):
    """Another worker claimed "r1", finished round 1, then DIED in round 2 (it never
    recorded anything). Its heartbeat is long out of date. Our worker takes the
    run over and resumes it: round 1 isn't redone."""
    store = RunStore(tmp_path)
    queue(store, "r1")
    store.claim_next("dead-worker")
    # The dead worker's attempt: crashes in round 2, after round 1 was saved.
    dead_tools = Tools(retriever=StubRetriever(),
                       generator=CrashOnceGenerator(THREE_ROUNDS, crash_in=2),
                       predictor=TablePredictor({}, OPAQUE),
                       gate=PlaceholderGate(), ranker=PlaceholderRanker())
    dead_graph = build_graph(dead_tools, checkpointer=make_checkpointer(tmp_path), clock=FakeClock())
    try:
        dead_graph.invoke({"run_id": "r1", "objective": "test", "config": CONFIG, "policy": POLICY},
                          {"configurable": {"thread_id": "r1"}})
    except RuntimeError:
        pass  # the "crash"; the dead worker never got to record it
    with store.conn:  # its last heartbeat was long ago
        store.conn.execute("UPDATE runs SET heartbeat_at = '2000-01-01T00:00:00+00:00' WHERE run_id = 'r1'")

    # Our worker only needs rounds 2 and 3.
    store, worker = make_worker(tmp_path, ListGenerator(THREE_ROUNDS[1:]))
    record = worker.process_next_run()
    assert record.run_id == "r1" and record.claimed_by == "w-test"
    assert record.status == "stopped"
    saved = saved_state(worker.graph.checkpointer, "r1")
    assert [c.iteration for c in saved["history"]] == [1, 2, 3]


def test_heartbeat_notices_when_another_worker_takes_the_run(tmp_path):
    store = RunStore(tmp_path)
    queue(store, "r1")
    store.claim_next("w1")
    with Heartbeat(tmp_path, "r1", "w1", every=0.02) as beat:
        with store.conn:  # someone else takes the run over
            store.conn.execute("UPDATE runs SET claimed_by = 'w2' WHERE run_id = 'r1'")
        deadline = time.monotonic() + 2  # wait at most 2 seconds
        while not beat.lost and time.monotonic() < deadline:
            time.sleep(0.01)
    assert beat.lost


def test_a_worker_that_lost_the_run_does_not_overwrite_its_status(tmp_path):
    """If still_owner() says the run now belongs to another worker, the result
    of this worker's attempt is NOT written to the runs table."""
    store, worker = make_worker(tmp_path, ListGenerator(THREE_ROUNDS))
    queue(store, "r1")
    record = store.claim_next("w-test")
    after = run_claimed(worker.graph, store, record, POLICY, still_owner=lambda: False)
    assert after.status == "running"  # untouched: not "stopped"
