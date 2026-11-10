"""Tests for item 6: the audit log, failure codes, retries inside a real run, the events
endpoint, and the operational log files.

Run from the repo root, in the langgraph venv:
    python -m pytest orchestrator/tests -v
"""

import logging
import sqlite3
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from orchestrator import Tools, build_graph
from orchestrator.api import create_app
from orchestrator.gate import PlaceholderGate, PlaceholderRanker
from orchestrator.interfaces import TemporaryToolError
from orchestrator.logs import setup_logging
from orchestrator.runner import saved_state
from orchestrator.storage import RunStore, make_checkpointer
from orchestrator.stubs import StubRetriever
from orchestrator.tests.fakes import FAILED, GOOD, PARTNER, FakeClock, ListGenerator
from orchestrator.worker import Worker
from orchestrator.tests.test_api import AUTH, RUNS, post, run_body
from orchestrator.tests.test_storage import THREE_ROUNDS
from orchestrator.tests.test_worker import TEST_POLICIES, make_worker, queue


class FlakyGenerator(ListGenerator):
    """A fake generator that raises TemporaryToolError the first `times` times it's called.

    Simulates e.g. the MatGPT server being briefly busy.
    """

    def __init__(self, batches, times: int):
        super().__init__(batches)
        self.failures_left = times

    def generate_validate(self, packet):
        if self.failures_left > 0:
            self.failures_left -= 1
            raise TemporaryToolError("generator busy")
        return super().generate_validate(packet)


def actions(store: RunStore, run_id: str) -> list[str]:
    """The run's audit log as a list of action names, in order."""
    return [e["action"] for e in store.list_events(run_id, limit=100)]


# --- The audit trail of a normal run --------------------------------------------------------

def test_a_run_leaves_a_complete_audit_trail(tmp_path):
    store, worker = make_worker(tmp_path, ListGenerator(THREE_ROUNDS))
    queue(store, "r1")
    worker.process_next_run()
    assert actions(store, "r1") == ["run_created", "run_claimed", "run_started",
                                    "iteration_finished", "iteration_finished", "iteration_finished",
                                    "run_stopped"]
    events = store.list_events("r1", limit=100)
    assert events[0]["actor_id"] == "api"  # queue() records the run as the API would
    assert {e["actor_id"] for e in events[1:]} == {"worker:w-test"}
    assert [e["seq"] for e in events] == sorted(e["seq"] for e in events)  # a reliable order
    assert events[3]["detail"].startswith("round 1: 1 candidates: 0 Accept, 1 Revise")


def test_the_audit_log_never_contains_smiles(tmp_path):
    """D2: no molecular text in the audit log. Check every field of every event
    against the partner's and every candidate's SMILES."""
    store, worker = make_worker(tmp_path, ListGenerator(THREE_ROUNDS))
    queue(store, "r1")
    worker.process_next_run()
    history = saved_state(worker.graph.checkpointer, "r1")["history"]
    molecules = {PARTNER} | {c.generated_smiles for c in history} | {c.canonical_smiles for c in history}
    text = " ".join(str(value) for event in store.list_events("r1", limit=100) for value in event.values())
    assert not any(m in text for m in molecules if m)


def test_an_audit_record_that_cannot_be_saved_undoes_the_change(tmp_path):
    """D2: if the audit can't be saved, the change fails too. An empty actor is refused
    by the database, so the status change is undone."""
    store = RunStore(tmp_path)
    queue(store, "r1")
    with pytest.raises(sqlite3.IntegrityError):
        store.update("r1", actor="", action="run_completed", status="completed")
    assert store.get("r1").status == "queued"  # unchanged
    assert actions(store, "r1") == ["run_created"]  # and no half-written event


# --- Retries inside a real run, and failure codes ------------------------------------------

def test_a_temporary_tool_error_is_retried_and_the_run_carries_on(tmp_path):
    store, worker = make_worker(tmp_path, FlakyGenerator(THREE_ROUNDS, times=1))
    queue(store, "r1")
    record = worker.process_next_run()
    assert record.status == "stopped" and record.failure is None  # the retry worked
    attempts = [(e["phase"], e["attempt"], e["result_code"], e["detail"])
                for e in store.list_events("r1", limit=100) if e["action"] == "tool_attempt"]
    assert attempts == [("generator", 1, "temporary_error", "TemporaryToolError"),
                        ("generator", 2, "succeeded_after_retry", None)]


def test_a_tool_that_stays_down_fails_the_run_with_dependency_unavailable(tmp_path):
    store, worker = make_worker(tmp_path, FlakyGenerator(THREE_ROUNDS, times=99))
    queue(store, "r1")
    record = worker.process_next_run()
    assert record.status == "failed" and record.failure_code == "DEPENDENCY_UNAVAILABLE"
    assert "after 3 attempts" in record.failure
    events = store.list_events("r1", limit=100)
    assert sum(e["action"] == "tool_attempt" for e in events) == 3  # D2: at most 3 attempts
    assert events[-1]["action"] == "run_failed" and events[-1]["result_code"] == "DEPENDENCY_UNAVAILABLE"


def test_every_prediction_failing_has_its_own_code(tmp_path):
    store, worker = make_worker(tmp_path, ListGenerator(THREE_ROUNDS), default=FAILED)
    queue(store, "r1")
    record = worker.process_next_run()
    assert record.status == "failed" and record.failure_code == "ALL_PREDICTIONS_FAILED"


class WrongCountPredictor:
    """Breaks the predictor contract (one prediction per candidate): returns none at all."""

    def predict(self, candidates):
        return []


def test_a_broken_contract_is_an_internal_error(tmp_path):
    """Not a tool error: the predictor answered, but wrongly. Our own check catches it."""
    store = RunStore(tmp_path)
    queue(store, "r1")
    tools = Tools(retriever=StubRetriever(), generator=ListGenerator(THREE_ROUNDS),
                  predictor=WrongCountPredictor(), gate=PlaceholderGate(), ranker=PlaceholderRanker())
    graph = build_graph(tools, checkpointer=make_checkpointer(tmp_path), clock=FakeClock())
    record = Worker(graph, store, policies=TEST_POLICIES, worker_id="w-test").process_next_run()
    assert record.status == "failed" and record.failure_code == "INTERNAL_ERROR"
    assert "predictor returned 0 predictions" in record.failure


# --- The events endpoint ----------------------------------------------------------------------

@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(tmp_path, "test-token", policies=TEST_POLICIES))


def test_events_endpoint_pages_through_the_audit_log(client, tmp_path):
    run_id = post(client, RUNS, run_body()).json()["run_id"]
    _, worker = make_worker(tmp_path, ListGenerator([["CCO", "CCN"]]), default=GOOD)
    worker.process_next_run()

    first = client.get(f"{RUNS}/{run_id}/events", params={"limit": 2}, headers=AUTH).json()
    assert [e["action"] for e in first["items"]] == ["run_created", "run_claimed"]
    assert first["items"][0]["request_id"]  # the API request that created it
    rest = client.get(f"{RUNS}/{run_id}/events", params={"cursor": first["next_cursor"]}, headers=AUTH).json()
    assert [e["action"] for e in rest["items"]] == ["run_started", "iteration_finished", "run_completed"]
    assert rest["next_cursor"] is None  # no more pages


def test_events_endpoint_errors(client):
    run_id = post(client, RUNS, run_body()).json()["run_id"]
    bad = client.get(f"{RUNS}/{run_id}/events", params={"cursor": "abc"}, headers=AUTH)
    assert bad.status_code == 422 and bad.json()["error"]["details"][0]["field"] == "cursor"
    assert client.get(f"{RUNS}/{uuid4()}/events", headers=AUTH).status_code == 404


def test_failure_code_appears_in_the_api(client, tmp_path):
    run_id = post(client, RUNS, run_body()).json()["run_id"]
    _, worker = make_worker(tmp_path, FlakyGenerator(THREE_ROUNDS, times=99))
    worker.process_next_run()
    failure = client.get(f"{RUNS}/{run_id}", headers=AUTH).json()["failure"]
    assert failure["code"] == "DEPENDENCY_UNAVAILABLE"


def test_cancel_through_the_api_is_audited_with_its_request_id(client):
    run_id = post(client, RUNS, run_body()).json()["run_id"]
    response = post(client, f"{RUNS}/{run_id}/cancel", {})
    events = client.get(f"{RUNS}/{run_id}/events", headers=AUTH).json()["items"]
    assert [e["action"] for e in events] == ["run_created", "cancel_requested", "run_cancelled"]
    assert events[1]["request_id"] == response.headers["X-Request-ID"]


# --- The operational log ------------------------------------------------------------------------

def test_operational_log_writes_to_a_rotating_file(tmp_path):
    path = setup_logging(tmp_path, "worker")
    ours = logging.getLogger("orchestrator")
    try:
        logging.getLogger("orchestrator.retry").warning("generator: temporary error (test)")
        for handler in ours.handlers:
            handler.flush()
        assert path == tmp_path / "logs" / "worker.log"
        assert "generator: temporary error (test)" in path.read_text(encoding="utf-8")
    finally:  # undo the setup, so the log file is closed and other tests aren't affected
        for handler in ours.handlers:
            handler.close()
        ours.handlers = []
        ours.propagate = True
