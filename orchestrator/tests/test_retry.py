"""Tests for the retry rules (retry.py): D2's 3 attempts, 1 s / 2 s waits, temporary vs permanent.

Run from the repo root, in the langgraph venv:
    python -m pytest orchestrator/tests -v

The tests never really wait: they pass a fake `sleep` that just writes down
how long it was asked to wait, and a fixed `rng` (random number) so the
jitter is predictable.
"""

import pytest

from orchestrator.interfaces import TemporaryToolError
from orchestrator.retry import RetriesExhausted, ToolFailed, call_with_retry


class Flaky:
    """A fake tool call that raises the given errors in order, then returns "ok".

    Example: Flaky([TemporaryToolError("busy")]) fails once, then succeeds.
    It counts how many times it was called.
    """

    def __init__(self, errors):
        self.errors = list(errors)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)  # pop(0) takes the first one off the list
        return "ok"


def run(call, rng=lambda: 0.0):
    """Call call_with_retry with a fake sleep and a reporting list. Returns (result or error, waits, reports)."""
    waits, reports = [], []
    try:
        result = call_with_retry("generator", call, sleep=waits.append, rng=rng,
                                 report=lambda tool, attempt, code, error: reports.append((attempt, code)))
    except (RetriesExhausted, ToolFailed) as error:
        result = error
    return result, waits, reports


def test_success_first_time_needs_no_retry_and_reports_nothing():
    result, waits, reports = run(Flaky([]))
    assert result == "ok" and waits == [] and reports == []


def test_temporary_errors_are_retried_with_1s_then_2s_waits():
    call = Flaky([TemporaryToolError("busy"), TemporaryToolError("busy")])
    result, waits, reports = run(call)
    assert result == "ok" and call.calls == 3
    assert waits == [1.0, 2.0]  # D2: 1 s, then 2 s (jitter is 0 here)
    assert reports == [(1, "temporary_error"), (2, "temporary_error"), (3, "succeeded_after_retry")]


def test_three_temporary_failures_give_up_with_retries_exhausted():
    call = Flaky([TemporaryToolError("down")] * 3)
    result, waits, _ = run(call)
    assert isinstance(result, RetriesExhausted) and result.attempts == 3
    assert call.calls == 3 and waits == [1.0, 2.0]  # no wait after the last attempt


def test_permanent_errors_are_never_retried():
    call = Flaky([ValueError("invalid input")])
    result, waits, reports = run(call)
    assert isinstance(result, ToolFailed) and isinstance(result.error, ValueError)
    assert call.calls == 1 and waits == []  # D2: permanent errors are never retried
    assert reports == [(1, "permanent_error")]


@pytest.mark.parametrize("error", [TimeoutError("slow"), ConnectionError("network")])
def test_timeouts_and_connection_errors_count_as_temporary(error):
    result, _, _ = run(Flaky([error]))
    assert result == "ok"  # retried, then succeeded


def test_jitter_adds_at_most_a_quarter_second():
    _, waits, _ = run(Flaky([TemporaryToolError("busy")]), rng=lambda: 0.999)
    assert 1.0 < waits[0] < 1.25
