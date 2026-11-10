"""The web API (item 5): the "front door" Nyla's dashboard uses.

WHAT THIS FILE IS
-----------------
A web API is a set of web addresses that PROGRAMS send requests to. This one
follows D1's API contract (paths start with /api/v1, JSON in and out, D1's error
format), plus D2's cancel endpoint and two endpoints Nyla asked for:

  POST /api/v1/runs                       start a run -> 202 {run_id, status: "queued", created_at}
  GET  /api/v1/runs?status=&limit=        list runs, newest first            (Nyla)
  GET  /api/v1/runs/{run_id}              one run's status, settings, failure
  POST /api/v1/runs/{run_id}/cancel       ask a run to stop                  (D2)
  GET  /api/v1/runs/{run_id}/shortlist    the ranked results (completed runs only)
  GET  /api/v1/runs/{run_id}/report       everything about a run in one bundle (Nyla)
  GET  /api/v1/runs/{run_id}/events       the run's audit log, page by page  (D2 G4)
  GET  /api/v1/health                     is a worker running?               (our addition)

It never RUNS anything: POST /runs only records the run as "queued" and replies
at once (D1: 202 Accepted = "your order is in", not "it's done"). The separate
worker program (worker.py) picks it up. Both share the runs database.

Built with FastAPI (decision Q1): request and response shapes are pydantic
models (like the rest of our code), and FastAPI generates interactive
documentation at http://127.0.0.1:<port>/docs, where every endpoint can be tried in a browser.

SECURITY (decision Q4, Phase 1): the server only listens on 127.0.0.1 (this
machine), and every request must carry the shared API token:
    Authorization: Bearer <token>
Proper user accounts (D2's G1) are deferred.

Start it with:   python -m orchestrator serve --db runs --port 8100
"""

# logging: one line per request in runs/logs/api.log (see logs.py).
import logging

# secrets.compare_digest: compares two secrets safely (see require_token).
import secrets

# time.perf_counter: a precise stopwatch, to log how long each request took.
import time

# sqlite3: only to recognise "database unavailable" errors (-> 503).
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, Header, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
# HTTPBearer: FastAPI's standard way to read "Authorization: Bearer <token>". Declaring
# it this way also adds an "Authorize" button to the /docs page.
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from .policies import POLICIES
from .runner import FINISHED, saved_state
from .state import Policy, RunConfig, RunConstraints, RunStatus, TargetType
from .storage import RunRecord, RunStore, _iso, _now, body_fingerprint, make_checkpointer
from .worker import STALE_AFTER

API_PREFIX = "/api/v1"  # D1: all paths begin with /api/v1
MAX_BODY_BYTES = 1024 * 1024  # D1: bodies over 1 MiB are refused (413)
# Phase 1 has ONE shared token, so every caller is the same caller as far as
# idempotency receipts are concerned (D2 keys receipts by caller).
SHARED_CALLER = "shared-token"
# How the API appears in the audit log (actor_id). With proper user accounts
# (deferred), this would become the user who made the request.
API_ACTOR = "api"

log = logging.getLogger("orchestrator.api")  # this file's channel in the operational log


# ============================================================================
# ERRORS in D1's format:
#   {"error": {"code": ..., "message": ..., "details": [{"field": ..., "reason": ...}]},
#    "request_id": ...}
# ============================================================================

class ApiError(Exception):
    """Raise this anywhere in an endpoint to reply with a D1-format error.

    Example: raise ApiError(404, "NOT_FOUND", "No run with that ID")
    """

    def __init__(self, status: int, code: str, message: str, details: Optional[list] = None):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message
        self.details = details or []  # [] when no particular field is to blame (D1)


def _error_response(request: Request, status: int, code: str, message: str,
                    details: Optional[list] = None) -> JSONResponse:
    """Build a D1-format error reply. The request_id helps find the request in logs."""
    request_id = getattr(request.state, "request_id", str(uuid4()))
    return JSONResponse(status_code=status, headers={"X-Request-ID": request_id}, content={
        "error": {"code": code, "message": message, "details": details or []},
        "request_id": request_id,
    })


def _field_details(errors: list) -> list:
    """Turn pydantic/FastAPI validation errors into D1 details: [{field, reason}].

    Each error has a "loc" (where it happened, e.g. ("body", "constraints",
    "avt_min_fraction")) and a "msg". "body"/"query"/"path"/"header" are dropped
    from the start, so the field reads like D1's example: "constraints.avt_min_fraction".
    """
    details = []
    for err in errors:
        loc = [str(part) for part in err.get("loc", ()) if part not in ("body", "query", "path", "header")]
        details.append({"field": ".".join(loc), "reason": err.get("msg", "invalid")})
    return details


# ============================================================================
# REQUEST SHAPES
# ============================================================================

class CreateRunRequest(BaseModel):
    """The body of POST /runs: D1's fields plus OUR additions (marked).

    extra="forbid": unknown fields are refused with 422 (D1).
    """

    model_config = ConfigDict(extra="forbid")

    # --- D1 fields ---
    knowledge_snapshot_id: UUID  # Kyle's knowledge snapshot. Stored ("pinned"); can't be verified yet.
    model_version: str = Field(min_length=1, max_length=64)  # Arleen's model. Pinned; not verifiable yet.
    policy_version: str = Field(min_length=1, max_length=64)  # must be in policies.py, else 422
    constraints: RunConstraints  # the three minimum targets (ranges checked by RunConstraints)
    candidate_limit: int = Field(default=20, ge=1, le=100)
    max_iterations: int = Field(default=3, ge=1, le=10)
    seed: int = Field(default=0, ge=0, le=2147483647)

    # --- OUR additions (decision Q3, 2026-11-08; Shashank is telling the team) ---
    # A run can't happen without knowing WHAT to design against WHICH partner, so
    # these two are REQUIRED. By D1's own rule, adding a required request field is a
    # breaking change, so D1 should be updated to include them.
    target_type: TargetType  # design "donor"s or "acceptor"s
    fixed_partner_smiles: str = Field(min_length=1, max_length=4096)  # the other half of every pair
    # Optional extras, off by default:
    objective: str = Field(default="", max_length=2000)  # the goal in plain English
    target_count: Optional[int] = Field(default=None, ge=1)  # finish early after this many Accepts
    patience: Optional[int] = Field(default=None, ge=1)  # stop after this many rounds without progress


class CancelRequest(BaseModel):
    """The body of POST /runs/{id}/cancel: must be exactly {} (D2)."""

    model_config = ConfigDict(extra="forbid")


# ============================================================================
# THE APP
# ============================================================================

def create_app(runs_dir: Path, api_token: str,
               policies: Mapping[str, Policy] = POLICIES) -> FastAPI:
    """Build the web API for the runs stored in runs_dir.

    Parameters:
        runs_dir  - the folder with runs.db and checkpoints.db (the worker must use the same one)
        api_token - the shared secret every request must send as "Authorization: Bearer <token>"
        policies  - the policy registry (tests pass their own)

    Returns a FastAPI app. `serve` runs it with uvicorn; tests use FastAPI's TestClient.
    """
    if not api_token:
        raise ValueError("an API token is required")
    checkpointer = make_checkpointer(runs_dir)  # read-only use: shortlist, report, live iteration

    def get_store():
        """Give each request its OWN connection to runs.db, and close it afterwards.

        The server answers requests on several threads at once, and one SQLite
        connection must not be shared between threads. A function that `yield`s
        is FastAPI's way to do "set up, hand over, clean up".
        """
        store = RunStore(runs_dir)
        try:
            yield store
        finally:
            store.close()

    # auto_error=False: if the header is missing, give us None instead of FastAPI's
    # own error, so we can reply in D1's error format.
    bearer = HTTPBearer(auto_error=False)

    def require_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer)) -> None:
        """Refuse the request (401) unless it carries "Authorization: Bearer <the token>".

        secrets.compare_digest compares in constant time, so an attacker can't
        guess the token character by character from how long the check takes.
        """
        if credentials is None or not secrets.compare_digest(credentials.credentials.encode(),
                                                             api_token.encode()):
            raise ApiError(401, "UNAUTHENTICATED", "Missing or invalid API token")

    app = FastAPI(
        title="Transparent-OPV Orchestrator API",
        version="v1",
        description="D1/D2 run API for the transparent-OPV capstone. FAKE tools for now.",
        dependencies=[Depends(require_token)],  # every endpoint checks the token
    )

    # --- Checks that apply to every request ("middleware" runs around each request) ---
    @app.middleware("http")
    async def request_basics(request: Request, call_next):
        """Give each request an ID, and refuse oversized or non-JSON POSTs (D1: 413, 415)."""
        request.state.request_id = str(uuid4())
        if request.method == "POST":
            if int(request.headers.get("content-length") or 0) > MAX_BODY_BYTES:
                return _error_response(request, 413, "PAYLOAD_TOO_LARGE", "Body exceeds 1 MiB")
            if not request.headers.get("content-type", "").startswith("application/json"):
                return _error_response(request, 415, "UNSUPPORTED_MEDIA_TYPE",
                                       "Content-Type must be application/json")
        began = time.perf_counter()
        response = await call_next(request)  # let the endpoint handle it
        response.headers["X-Request-ID"] = request.state.request_id
        # Operational log: method, path, status, time, request id. Never bodies or tokens.
        log.info("%s %s -> %d (%.0f ms) request_id=%s", request.method, request.url.path,
                 response.status_code, (time.perf_counter() - began) * 1000, request.state.request_id)
        return response

    # --- Turn every kind of error into D1's error format ---
    @app.exception_handler(ApiError)
    async def on_api_error(request: Request, exc: ApiError):
        return _error_response(request, exc.status, exc.code, exc.message, exc.details)

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(request: Request, exc: RequestValidationError):
        """Bad input: 400 if the body isn't even valid JSON, otherwise 422 with the fields to blame."""
        if any(err.get("type") == "json_invalid" for err in exc.errors()):
            return _error_response(request, 400, "MALFORMED_JSON", "The request body cannot be parsed as JSON")
        return _error_response(request, 422, "VALIDATION_ERROR", "Invalid request",
                               _field_details(exc.errors()))

    @app.exception_handler(StarletteHTTPException)
    async def on_http_error(request: Request, exc: StarletteHTTPException):
        """Unknown addresses (404) and wrong methods (405), in D1's format."""
        code = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}.get(exc.status_code, "HTTP_ERROR")
        return _error_response(request, exc.status_code, code, str(exc.detail))

    @app.exception_handler(sqlite3.OperationalError)
    async def on_database_error(request: Request, exc: sqlite3.OperationalError):
        """The database is busy or unreachable: D1 says 503, safe to retry with the same key."""
        return _error_response(request, 503, "DEPENDENCY_UNAVAILABLE", "The run database is unavailable")

    @app.exception_handler(Exception)
    async def on_unexpected_error(request: Request, exc: Exception):
        """Anything else: 500, WITHOUT the internal error details (D1: no stack traces exposed).

        The full details go to the operational log instead (api.log), with the
        request_id, so whoever runs the server can find out what went wrong.
        """
        log.error("unexpected error, request_id=%s", getattr(request.state, "request_id", "?"),
                  exc_info=exc)
        return _error_response(request, 500, "INTERNAL_ERROR", "Unexpected server failure")

    # --- Helpers shared by the endpoints ---
    def check_key(idempotency_key: Optional[str]) -> str:
        """D1: every POST needs an Idempotency-Key header that is a UUID."""
        try:
            return str(UUID(idempotency_key or ""))
        except ValueError:
            raise ApiError(422, "VALIDATION_ERROR", "Invalid request", [{
                "field": "Idempotency-Key", "reason": "Required header; must be a UUID"}]) from None

    def replay(receipt: dict, body_hash: str) -> JSONResponse:
        """Answer a repeated POST from its receipt (D1 idempotency rules).

        Same key + same body -> the SAME reply as the first time (nothing new happens).
        Same key + different body -> 409 CONFLICT.
        """
        if receipt["body_hash"] != body_hash:
            raise ApiError(409, "CONFLICT", "Idempotency-Key was already used with a different body")
        headers = ({"Location": f"{API_PREFIX}/runs/{receipt['response']['run_id']}"}
                   if receipt["status_code"] == 202 and "created_at" in receipt["response"] else {})
        return JSONResponse(receipt["response"], status_code=receipt["status_code"], headers=headers)

    def find_run(store: RunStore, run_id: UUID) -> RunRecord:
        """The run with this ID, or a 404 error."""
        record = store.get(str(run_id))
        if record is None:
            raise ApiError(404, "NOT_FOUND", f"No run with run_id {run_id}")
        return record

    def run_view(record: RunRecord, live: bool = False) -> dict:
        """A run in D1's "Run" shape (plus our additions stop_reason and cancel_requested).

        live=True: for a RUNNING run, read the current iteration from its latest
        checkpoint (the runs table is only updated when the run ends).
        """
        iteration = record.iteration
        if live and record.status == "running":
            iteration = saved_state(checkpointer, record.run_id).get("iteration", iteration)
        cfg = record.config
        return {
            "run_id": record.run_id,
            "status": record.status,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
            "iteration": iteration,
            "config": {
                "knowledge_snapshot_id": record.knowledge_snapshot_id,
                "model_version": record.model_version,
                "policy_version": record.policy_version,
                "constraints": cfg.constraints.model_dump(),
                "candidate_limit": cfg.candidate_limit,
                "max_iterations": cfg.max_iterations,
                "seed": cfg.seed,
                # our additions:
                "target_type": cfg.target_type,
                "fixed_partner_smiles": cfg.fixed_partner_smiles,
                "target_count": cfg.target_count,
                "patience": cfg.patience,
            },
            "prediction_batch_id": None,  # D1 field; needs D2's batch records (not built yet)
            # D1: {code, message} for failed runs. The code says WHY (decision Q7):
            # DEPENDENCY_UNAVAILABLE, TOOL_ERROR, ALL_PREDICTIONS_FAILED, UNKNOWN_POLICY, INTERNAL_ERROR.
            "failure": ({"code": record.failure_code or "INTERNAL_ERROR", "message": record.failure}
                        if record.status == "failed" and record.failure else None),
            # our additions:
            "stop_reason": record.stop_reason,
            "cancel_requested": record.cancel_requested,
        }

    # ========================================================================
    # ENDPOINTS
    # ========================================================================

    @app.post(f"{API_PREFIX}/runs", status_code=202)
    def create_run(body: CreateRunRequest, request: Request, store: RunStore = Depends(get_store),
                   idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key")):
        """Record a new run as "queued" and reply at once (D1). The worker runs it later."""
        key = check_key(idempotency_key)
        endpoint = "POST /runs"
        body_hash = body_fingerprint(body.model_dump(mode="json"))
        receipt = store.find_receipt(SHARED_CALLER, endpoint, key)
        if receipt:  # seen this key before: same reply as last time, or 409
            return replay(receipt, body_hash)
        if body.policy_version not in policies:
            raise ApiError(422, "VALIDATION_ERROR", "Invalid request", [{
                "field": "policy_version", "reason": "Unknown or unregistered policy version"}])
        try:  # RunConfig also checks that the partner is a real molecule
            config = RunConfig(target_type=body.target_type,
                               fixed_partner_smiles=body.fixed_partner_smiles,
                               constraints=body.constraints, candidate_limit=body.candidate_limit,
                               max_iterations=body.max_iterations, seed=body.seed,
                               target_count=body.target_count, patience=body.patience)
        except ValidationError as error:
            raise ApiError(422, "VALIDATION_ERROR", "Invalid request",
                           _field_details(error.errors())) from None
        run_id = str(uuid4())
        response = {"run_id": run_id, "status": "queued", "created_at": _now()}
        created = store.create_with_receipt(
            run_id, body.objective, config, body.policy_version,
            str(body.knowledge_snapshot_id), body.model_version,
            caller_id=SHARED_CALLER, endpoint=endpoint, key=key, body_hash=body_hash,
            status_code=202, response=response,
            actor=API_ACTOR, request_id=request.state.request_id)  # audit: who and which request
        if not created:  # an identical request won the race a moment ago: reply as it did
            return replay(store.find_receipt(SHARED_CALLER, endpoint, key), body_hash)
        return JSONResponse(response, status_code=202,
                            headers={"Location": f"{API_PREFIX}/runs/{run_id}"})

    @app.get(f"{API_PREFIX}/runs")
    def list_runs(status: Optional[RunStatus] = None, limit: int = Query(default=20, ge=1, le=100),
                  store: RunStore = Depends(get_store)):
        """The most recent runs, optionally only one status (Nyla's run picker)."""
        return {"items": [run_view(r) for r in store.list(status=status, limit=limit)]}

    @app.get(f"{API_PREFIX}/runs/{{run_id}}")
    def get_run(run_id: UUID, store: RunStore = Depends(get_store)):
        """One run (D1). A malformed ID -> 422, an unknown one -> 404. Never advances the run."""
        return run_view(find_run(store, run_id), live=True)

    @app.post(f"{API_PREFIX}/runs/{{run_id}}/cancel", status_code=202)
    def cancel_run(run_id: UUID, body: CancelRequest, request: Request,
                   store: RunStore = Depends(get_store),
                   idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key")):
        """Ask a run to stop (D2). It ends "stopped", reason "cancelled", keeping work done so far.

        A finished (or failed) run -> 409 RUN_CLOSED. A queued run stops at once;
        a running one stops before its next step.
        """
        key = check_key(idempotency_key)
        endpoint = f"POST /runs/{run_id}/cancel"
        body_hash = body_fingerprint(body.model_dump(mode="json"))
        receipt = store.find_receipt(SHARED_CALLER, endpoint, key)
        if receipt:
            return replay(receipt, body_hash)
        record = find_run(store, run_id)
        if record.status in FINISHED or record.status == "failed":
            raise ApiError(409, "RUN_CLOSED", f"Run already ended with status {record.status!r}")
        record = store.request_cancel(record.run_id, actor=API_ACTOR,
                                      request_id=request.state.request_id)
        response = {"run_id": record.run_id, "status": record.status, "cancel_requested": True}
        store.save_receipt(caller_id=SHARED_CALLER, endpoint=endpoint, key=key, body_hash=body_hash,
                           status_code=202, response=response)
        return JSONResponse(response, status_code=202)

    @app.get(f"{API_PREFIX}/runs/{{run_id}}/shortlist")
    def get_shortlist(run_id: UUID, limit: int = Query(default=10, ge=1, le=100),
                      store: RunStore = Depends(get_store)):
        """The ranked accepted pairs (D1). Only completed runs have one: otherwise 409."""
        record = find_run(store, run_id)
        if record.status != "completed":
            raise ApiError(409, "RESULTS_NOT_READY", f"Run status is {record.status!r}, not 'completed'")
        ranked = saved_state(checkpointer, record.run_id).get("ranked", [])
        items = []
        for rank, cand in enumerate(ranked[:limit], start=1):  # enumerate(..., start=1) counts 1, 2, 3...
            p = cand.prediction
            items.append({
                "rank": rank, "candidate_id": cand.candidate_id,
                "donor_smiles": cand.donor_smiles, "acceptor_smiles": cand.acceptor_smiles,
                "values": p.values.model_dump(), "uncertainty": p.uncertainty.model_dump(),
                "in_domain": p.in_domain, "rationale": cand.decision.rationale,
                "risk_flags": cand.decision.risk_flags,
                "prediction_batch_id": None,  # D1 field; batch records not built yet
                "parent_candidate_id": cand.parent_candidate_id,  # our addition (lineage)
            })
        return {"run_id": record.run_id, "generated_at": record.updated_at,
                "model_version": record.model_version, "policy_version": record.policy_version,
                "items": items}

    @app.get(f"{API_PREFIX}/runs/{{run_id}}/report")
    def get_report(run_id: UUID, store: RunStore = Depends(get_store)):
        """EVERYTHING about a run in one bundle (Nyla's request, 2026-10-22).

        Every candidate with its iteration, parent, chemistry violations,
        prediction, gate decision and near-miss numbers; the shortlist; the
        set-aside (escalated) pile; the ranked near misses; failed predictions;
        how the run ended; and the pinned versions. Works for running runs too
        (a partial report). A queued run has nothing yet -> 409.
        """
        record = find_run(store, run_id)
        if record.status == "queued":
            raise ApiError(409, "RESULTS_NOT_READY", "Run has not started yet")
        saved = saved_state(checkpointer, record.run_id)

        def ids(key: str) -> list:
            """The candidate_ids in one of the saved lists."""
            return [c.candidate_id for c in saved.get(key, [])]

        return {
            "run": run_view(record, live=True),
            "pinned_versions": {"knowledge_snapshot_id": record.knowledge_snapshot_id,
                                "model_version": record.model_version,
                                "policy_version": record.policy_version},
            "stop_reason": record.stop_reason,
            "active_seconds": saved.get("active_seconds", 0.0),  # working time used (decision Q3)
            "shortlist_ids": ids("ranked"),  # accepted, best first
            "escalated_ids": ids("escalated"),  # set aside for DFT/lab review
            "near_miss_ids": ids("near_misses"),  # closest to passing first (formula NM-3)
            "failed_prediction_ids": ids("failed_predictions"),
            # every candidate ever made, in full (model_dump(mode="json") = as plain JSON data)
            "candidates": [c.model_dump(mode="json") for c in saved.get("history", [])],
            # The run's audit log is long, so it has its own (paged) endpoint:
            "events": f"{API_PREFIX}/runs/{record.run_id}/events",
            # Parts of Nyla's request that don't exist in the system yet:
            "not_yet_available": [
                "human reviews (approve/decline with reviewer, time, reason): needs D2 Handoff/Feedback records",
                "prediction_batch_id: needs D2 PredictionBatch records",
            ],
        }

    @app.get(f"{API_PREFIX}/runs/{{run_id}}/events")
    def list_events(run_id: UUID, limit: int = Query(default=50, ge=1, le=100),
                    cursor: Optional[str] = Query(default=None, max_length=256),
                    store: RunStore = Depends(get_store)):
        """A run's AUDIT LOG, oldest first, one page at a time (D2 G4).

        Each event: who (actor_id), what (action), how it went (result_code),
        where (phase), which attempt, and a short detail (never SMILES).
        Paging: if there may be more events, the reply has a `next_cursor`; pass
        it back as ?cursor=... to get the next page. An invalid cursor -> 422 (D2).
        """
        record = find_run(store, run_id)
        after = 0
        if cursor is not None:
            if not cursor.isdigit():  # our cursors are always the position number of the last event
                raise ApiError(422, "VALIDATION_ERROR", "Invalid request",
                               [{"field": "cursor", "reason": "Not a cursor returned by this endpoint"}])
            after = int(cursor)
        events = store.list_events(record.run_id, limit=limit, after_seq=after)
        # A full page might mean more events follow: give the position to continue from.
        next_cursor = str(events[-1]["seq"]) if len(events) == limit else None
        return {"items": events, "next_cursor": next_cursor}

    @app.get(f"{API_PREFIX}/health")
    def health(store: RunStore = Depends(get_store)):
        """Is the API up, and has a worker been seen recently? (our addition)

        worker_online is True if a worker was seen within the last 60 s. If it's
        False, queued runs won't start until a worker is running.
        """
        seen = store.last_worker_seen()
        cutoff = _iso(datetime.now(timezone.utc) - STALE_AFTER)
        return {"status": "ok", "worker_last_seen": seen,
                "worker_online": bool(seen and seen >= cutoff)}

    return app
