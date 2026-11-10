"""Tests for saving runs to disk (storage.py) and starting/resuming them (runner.py).

Run from the repo root, in the langgraph venv:
    python -m pytest orchestrator/tests -v

PYTEST FEATURES USED HERE (all built into pytest):
  tmp_path - when a test function has a parameter named `tmp_path`, pytest gives
             it a brand-new empty folder that's deleted later. Each test gets its
             own database files, so tests never affect each other or your real runs.
  caplog   - a parameter named `caplog` captures messages the program logs, so a
             test can check that a warning was (or wasn't) printed.
"""

import sqlite3

import pytest

from orchestrator import build_graph, Tools
from orchestrator.gate import PlaceholderGate, PlaceholderRanker
from orchestrator.runner import resume_run, start_run
from orchestrator.storage import ALLOWED_STATE_CLASSES, RunStore, make_checkpointer
from orchestrator.stubs import StubRetriever
from orchestrator.tests.fakes import (CONFIG, OPAQUE, POLICY, FakeClock, ListGenerator,
                                      TablePredictor)

# Three rounds of one molecule each. With OPAQUE (too dark) every round is
# revised, so a run uses all 3 rounds and ends "stopped" at the iteration budget.
THREE_ROUNDS = [["c1ccccc1"], ["Cc1ccccc1"], ["CCc1ccccc1"]]


class CrashOnceGenerator(ListGenerator):
    """A fake generator that CRASHES the first time it's asked for round `crash_in`.

    Simulates something going wrong mid-run (a tool outage, the machine going
    down). On the next attempt it works normally. It inherits everything else
    from ListGenerator (class X(Y) = "X is a Y, with these changes").
    """

    def __init__(self, batches, crash_in: int):
        super().__init__(batches)  # set up the ListGenerator part
        self.crash_in = crash_in  # which round to crash in
        self.crashed = False  # becomes True after the one crash

    def generate_validate(self, packet):
        if packet.iteration == self.crash_in and not self.crashed:
            self.crashed = True
            raise RuntimeError("simulated generator outage")
        return super().generate_validate(packet)  # behave like a normal ListGenerator


def make_tools(generator) -> Tools:
    """The test tools: the given generator, an all-OPAQUE predictor, placeholder gate/ranker."""
    return Tools(retriever=StubRetriever(), generator=generator,
                 predictor=TablePredictor({}, OPAQUE),
                 gate=PlaceholderGate(), ranker=PlaceholderRanker())


def start(tmp_path, generator, clock=None):
    """Build a disk-saving graph in tmp_path and start run "run-1". Returns (tools, record)."""
    tools = make_tools(generator)
    graph = build_graph(tools, checkpointer=make_checkpointer(tmp_path), clock=clock or FakeClock())
    record = start_run(graph, RunStore(tmp_path), objective="test", config=CONFIG,
                       policy=POLICY, run_id="run-1")
    return tools, record


# --- Saving and reloading ---------------------------------------------------------

def test_a_finished_run_survives_a_restart(tmp_path):
    """Run to the end, then build EVERYTHING again from the same folder (as if the
    program had restarted) and read the run back."""
    start(tmp_path, ListGenerator(THREE_ROUNDS))
    # "Restart": brand-new saver, graph and runs table, same files on disk.
    graph = build_graph(make_tools(ListGenerator([])), checkpointer=make_checkpointer(tmp_path))
    saved = graph.get_state({"configurable": {"thread_id": "run-1"}}).values
    assert saved["status"] == "stopped" and len(saved["history"]) == 3
    record = RunStore(tmp_path).get("run-1")
    assert record.status == "stopped" and record.iteration == 3
    assert record.stop_reason == "iteration budget (3) exhausted"


def test_reloading_needs_no_unregistered_types(tmp_path, caplog):
    """With our allow-list, reloading doesn't log LangGraph's "Deserializing
    unregistered type ... will be blocked" message."""
    start(tmp_path, ListGenerator(THREE_ROUNDS))
    graph = build_graph(make_tools(ListGenerator([])), checkpointer=make_checkpointer(tmp_path))
    graph.get_state({"configurable": {"thread_id": "run-1"}})  # forces a reload from disk
    assert "unregistered type" not in caplog.text
    # And the allow-list names every class in our state (a reminder to update it).
    assert ("orchestrator.state", "Candidate") in ALLOWED_STATE_CLASSES


# --- Crashes and resuming -----------------------------------------------------------

def test_a_crash_is_recorded_as_failed_then_resume_finishes_the_run(tmp_path):
    """Crash in round 2. The run is marked failed (with the error), keeps round 1's
    saved work, and resume_run continues from there to the normal end."""
    generator = CrashOnceGenerator(THREE_ROUNDS, crash_in=2)
    tools, record = start(tmp_path, generator)
    assert record.status == "failed"
    # A plain RuntimeError counts as PERMANENT (retry.py), so it isn't retried: TOOL_ERROR.
    assert record.failure_code == "TOOL_ERROR"
    assert record.failure == "generator failed: RuntimeError: simulated generator outage"
    assert record.iteration == 1  # round 1 had finished and was saved

    # "Restart" with a new saver and graph (same generator object, so it won't crash again).
    graph = build_graph(tools, checkpointer=make_checkpointer(tmp_path), clock=FakeClock())
    record = resume_run(graph, RunStore(tmp_path), "run-1")
    assert record.status == "stopped" and record.iteration == 3
    assert record.failure is None  # cleared when the run resumed
    saved = graph.get_state({"configurable": {"thread_id": "run-1"}}).values
    assert [c.iteration for c in saved["history"]] == [1, 2, 3]  # round 1 wasn't redone


def test_downtime_is_not_counted_against_the_time_budget(tmp_path):
    """Decision Q3: only working time counts. The clock jumps a million seconds
    (about 11 days) between the crash and the resume, but no work happened then,
    so the resumed run is NOT stopped for running out of time."""
    clock = FakeClock(step=0)  # steps take no time unless we move the clock
    generator = CrashOnceGenerator(THREE_ROUNDS, crash_in=2)
    tools, _ = start(tmp_path, generator, clock=clock)
    clock.now += 1_000_000  # "the machine was off for 11 days"
    graph = build_graph(tools, checkpointer=make_checkpointer(tmp_path), clock=clock)
    record = resume_run(graph, RunStore(tmp_path), "run-1")
    assert record.stop_reason == "iteration budget (3) exhausted"  # not "runtime budget"


def test_resume_refuses_finished_or_unknown_runs(tmp_path):
    tools, _ = start(tmp_path, ListGenerator(THREE_ROUNDS))
    graph = build_graph(tools, checkpointer=make_checkpointer(tmp_path))
    with pytest.raises(ValueError):  # finished runs can't be continued (D2)
        resume_run(graph, RunStore(tmp_path), "run-1")
    with pytest.raises(KeyError):
        resume_run(graph, RunStore(tmp_path), "no-such-run")


# --- The runs table ------------------------------------------------------------------

def test_runs_table_create_update_get_list(tmp_path):
    store = RunStore(tmp_path)
    store.create("a", "first", CONFIG, "test-policy", actor="test")
    store.create("b", "second", CONFIG, "test-policy", actor="test")
    assert store.get("a").status == "queued"  # D1: recorded, not started yet
    store.update("a", actor="test", action="run_completed", status="completed", iteration=2)
    assert store.get("a").status == "completed" and store.get("a").iteration == 2
    assert store.get("missing") is None
    assert {r.run_id for r in store.list()} == {"a", "b"}
    assert [r.run_id for r in store.list(status="completed")] == ["a"]
    assert store.get("b").config == CONFIG  # the settings survive the round trip


def test_runs_table_refuses_bad_changes(tmp_path):
    store = RunStore(tmp_path)
    store.create("a", "first", CONFIG, "test-policy", actor="test")
    with pytest.raises(ValueError):  # not an updatable column
        store.update("a", actor="test", action="edit", objective="changed")
    with pytest.raises(KeyError):  # no such run
        store.update("missing", actor="test", action="run_started", status="running")
    with pytest.raises(sqlite3.IntegrityError):  # the DATABASE refuses statuses not in D1
        store.update("a", actor="test", action="pause", status="paused")
