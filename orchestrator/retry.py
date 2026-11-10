"""Retrying tool calls (item 6), following D2's "RetryHandler" rules.

WHAT THIS FILE IS
-----------------
Every call the orchestrator makes to a tool (retriever, generator, predictor,
gate, ranker) goes through call_with_retry(). The rules (D2, decisions Q1/Q2
of 2026-11-10):

  - at most 3 ATTEMPTS in total
  - between attempts, wait 1 s, then 2 s, each plus a random 0-0.25 s
    ("jitter", so that many callers retrying at once don't all hit a busy
    server at exactly the same moment)
  - retry ONLY TEMPORARY errors: TemporaryToolError (interfaces.py),
    TimeoutError and ConnectionError
  - any other error is PERMANENT: no retry; it fails straight away
  - every failed attempt is REPORTED (for the audit log), and so is a success
    that needed retries

What the caller sees when it doesn't work out:
  RetriesExhausted - a temporary error happened on all 3 attempts
                     -> the run fails with code DEPENDENCY_UNAVAILABLE
  ToolFailed       - a permanent error (no retry)
                     -> the run fails with code TOOL_ERROR

Example:
    predictions = call_with_retry("predictor", lambda: tools.predictor.predict(batch))
`lambda: ...` packs the call into a little function so call_with_retry can run
it again on each attempt.
"""

# logging: writes diagnostic messages to the operational log (see logs.py).
import logging

# random: the jitter. time.sleep: waiting between attempts.
import random
import time
from typing import Callable, Optional, TypeVar

from .interfaces import TemporaryToolError

# The rules from D2.
MAX_ATTEMPTS = 3  # attempts in total (the first try + 2 retries)
DELAYS = (1.0, 2.0)  # seconds to wait after attempt 1 and after attempt 2
JITTER = 0.25  # up to this many extra random seconds per wait

# The error types that mean "temporary, try again". A tuple, so `except` can
# catch any of them at once.
TEMPORARY_ERRORS = (TemporaryToolError, TimeoutError, ConnectionError)

log = logging.getLogger("orchestrator.retry")  # this file's channel in the operational log

# T = "whatever type the tool returns". Lets type hints say "call_with_retry
# returns the same type as the function it calls".
T = TypeVar("T")

# What the reporting callback receives: (tool name, attempt number, result code,
# the error or None). Result codes: "temporary_error", "permanent_error",
# "succeeded_after_retry".
AttemptReport = Callable[[str, int, str, Optional[BaseException]], None]


class RetriesExhausted(Exception):
    """A tool kept failing TEMPORARILY: all attempts used. The run fails (DEPENDENCY_UNAVAILABLE)."""

    def __init__(self, tool: str, attempts: int, last_error: BaseException):
        super().__init__(f"{tool} still failing after {attempts} attempts: "
                         f"{type(last_error).__name__}: {last_error}")
        self.tool, self.attempts, self.last_error = tool, attempts, last_error


class ToolFailed(Exception):
    """A tool failed PERMANENTLY (not retried). The run fails (TOOL_ERROR)."""

    def __init__(self, tool: str, error: BaseException):
        super().__init__(f"{tool} failed: {type(error).__name__}: {error}")
        self.tool, self.error = tool, error


def call_with_retry(tool: str, call: Callable[[], T], *,
                    report: Optional[AttemptReport] = None,
                    sleep: Callable[[float], None] = time.sleep,
                    rng: Callable[[], float] = random.random) -> T:
    """Run `call()`, retrying temporary errors by D2's rules. Returns what `call()` returns.

    Parameters:
        tool   - the tool's name, for messages and the audit log ("generator", ...)
        call   - a function with no arguments that does the actual tool call
        report - optional function told about every failed attempt (and a late
                 success); the graph uses it to write audit events
        sleep  - the waiting function. Tests pass one that doesn't really wait.
        rng    - random number source (0-1) for the jitter. Tests pass a fixed one.

    Raises RetriesExhausted or ToolFailed (see the top of this file).
    """
    def tell(attempt: int, result_code: str, error: Optional[BaseException]) -> None:
        if report is not None:
            report(tool, attempt, result_code, error)

    for attempt in range(1, MAX_ATTEMPTS + 1):  # attempt = 1, 2, 3
        try:
            result = call()
        except TEMPORARY_ERRORS as error:
            tell(attempt, "temporary_error", error)
            if attempt == MAX_ATTEMPTS:
                log.error("%s: temporary error on final attempt %d: %s", tool, attempt, type(error).__name__)
                raise RetriesExhausted(tool, attempt, error) from error
            wait = DELAYS[attempt - 1] + rng() * JITTER
            log.warning("%s: temporary error on attempt %d (%s); retrying in %.2f s",
                        tool, attempt, type(error).__name__, wait)
            sleep(wait)
            continue  # go round the loop: the next attempt
        except Exception as error:  # anything else is permanent
            tell(attempt, "permanent_error", error)
            log.error("%s: permanent error on attempt %d: %s", tool, attempt, type(error).__name__)
            raise ToolFailed(tool, error) from error
        if attempt > 1:
            tell(attempt, "succeeded_after_retry", None)
            log.info("%s: succeeded on attempt %d", tool, attempt)
        return result
