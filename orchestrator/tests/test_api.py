"""Tests for the web API (api.py), using FastAPI's built-in TEST CLIENT.

Run from the repo root, in the langgraph venv:
    python -m pytest orchestrator/tests -v

TestClient sends requests straight to the app inside the test, so no real
server is started and nothing listens on the network. Each request returns
a response with a .status_code (e.g. 202) and .json() (the reply's data).

The worker runs queued runs: the tests call worker.process_next_run() (see
test_worker.py) when they need a run to actually happen.
"""

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from orchestrator.api import create_app
from orchestrator.tests.fakes import GOOD, ListGenerator
from orchestrator.tests.test_worker import TEST_POLICIES, make_worker

TOKEN = "test-token"  # the shared secret for these tests
AUTH = {"Authorization": f"Bearer {TOKEN}"}  # the header every request must carry
RUNS = "/api/v1/runs"


@pytest.fixture
def client(tmp_path):
    """A test client for an API whose database lives in this test's own tmp_path.

    @pytest.fixture makes this available to any test that has a parameter named
    `client`, like tmp_path itself.
    """
    return TestClient(create_app(tmp_path, TOKEN, policies=TEST_POLICIES))


def run_body(**changes) -> dict:
    """A valid POST /runs body (D1 example values plus our two additions), with any changes."""
    body = {
        "knowledge_snapshot_id": "7f48d012-2c91-45e1-9804-c2f674be2390",
        "model_version": "chemprop-test",
        "policy_version": "test-policy",
        "constraints": {"avt_min_fraction": 0.4, "pce_min_percent": 2.0, "lue_min_percent": 0.8},
        "target_type": "donor",
        "fixed_partner_smiles": "c1ccsc1",
    }
    body.update(changes)
    return body


def post(client, path, body, key=None, **headers):
    """POST with the token and an Idempotency-Key (a fresh one unless given)."""
    return client.post(path, json=body,
                       headers={**AUTH, "Idempotency-Key": key or str(uuid4()), **headers})


def error_of(response) -> dict:
    """The "error" part of a D1-format error reply."""
    return response.json()["error"]


# --- The token -------------------------------------------------------------------------

def test_requests_without_the_right_token_are_refused(client):
    for headers in ({}, {"Authorization": "Bearer wrong"}):
        response = client.get(RUNS, headers=headers)
        assert response.status_code == 401 and error_of(response)["code"] == "UNAUTHENTICATED"
        assert response.json()["request_id"]  # every error carries a request_id (D1)


# --- Creating runs -------------------------------------------------------------------------

def test_create_run_replies_202_queued_with_location(client):
    response = post(client, RUNS, run_body())
    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "queued" and set(data) == {"run_id", "status", "created_at"}
    assert response.headers["Location"] == f"{RUNS}/{data['run_id']}"
    run = client.get(f"{RUNS}/{data['run_id']}", headers=AUTH).json()
    assert run["status"] == "queued" and run["config"]["fixed_partner_smiles"] == "c1ccsc1"
    assert run["config"]["model_version"] == "chemprop-test" and run["failure"] is None


def test_same_idempotency_key_and_body_gives_the_same_run(client):
    key = str(uuid4())
    first = post(client, RUNS, run_body(), key=key).json()
    again = post(client, RUNS, run_body(), key=key)
    assert again.status_code == 202 and again.json() == first  # the very same reply
    assert len(client.get(RUNS, headers=AUTH).json()["items"]) == 1  # only ONE run created


def test_same_idempotency_key_with_a_different_body_is_a_conflict(client):
    key = str(uuid4())
    post(client, RUNS, run_body(), key=key)
    response = post(client, RUNS, run_body(seed=7), key=key)
    assert response.status_code == 409 and error_of(response)["code"] == "CONFLICT"


def test_missing_idempotency_key_is_refused(client):
    response = client.post(RUNS, json=run_body(), headers=AUTH)
    assert response.status_code == 422
    assert error_of(response)["details"][0]["field"] == "Idempotency-Key"


# pytest.mark.parametrize runs the same test once per (changes, field) pair below.
@pytest.mark.parametrize("changes, field", [
    ({"constraints": {"avt_min_fraction": 1.5, "pce_min_percent": 2, "lue_min_percent": 1}},
     "constraints.avt_min_fraction"),                  # AVT must be 0-1
    ({"candidate_limit": 101}, "candidate_limit"),     # D1: 1-100
    ({"policy_version": "no-such-policy"}, "policy_version"),
    ({"fixed_partner_smiles": "not_a_smiles"}, "fixed_partner_smiles"),
    ({"knowledge_snapshot_id": "not-a-uuid"}, "knowledge_snapshot_id"),
    ({"surprise": 1}, "surprise"),                     # D1: unknown fields are rejected
])
def test_invalid_run_requests_are_422_with_the_field_to_blame(client, changes, field):
    response = post(client, RUNS, run_body(**changes))
    assert response.status_code == 422 and error_of(response)["code"] == "VALIDATION_ERROR"
    assert field in [d["field"] for d in error_of(response)["details"]]


def test_body_that_is_not_json_is_400(client):
    response = client.post(RUNS, content="{not json",
                           headers={**AUTH, "Idempotency-Key": str(uuid4()),
                                    "Content-Type": "application/json"})
    assert response.status_code == 400 and error_of(response)["code"] == "MALFORMED_JSON"


def test_wrong_content_type_is_415(client):
    response = client.post(RUNS, content="hello",
                           headers={**AUTH, "Idempotency-Key": str(uuid4()), "Content-Type": "text/plain"})
    assert response.status_code == 415 and error_of(response)["code"] == "UNSUPPORTED_MEDIA_TYPE"


# --- Reading runs ----------------------------------------------------------------------------

def test_unknown_run_is_404_and_malformed_id_is_422(client):
    assert client.get(f"{RUNS}/{uuid4()}", headers=AUTH).status_code == 404
    assert client.get(f"{RUNS}/not-a-uuid", headers=AUTH).status_code == 422


def test_list_runs_with_status_filter_and_limit(client):
    for _ in range(3):
        post(client, RUNS, run_body())
    assert len(client.get(RUNS, params={"limit": 2}, headers=AUTH).json()["items"]) == 2
    assert client.get(RUNS, params={"status": "completed"}, headers=AUTH).json()["items"] == []
    assert client.get(RUNS, params={"status": "paused"}, headers=AUTH).status_code == 422


def test_results_are_not_ready_while_queued(client):
    run_id = post(client, RUNS, run_body()).json()["run_id"]
    for path in ("shortlist", "report"):
        response = client.get(f"{RUNS}/{run_id}/{path}", headers=AUTH)
        assert response.status_code == 409 and error_of(response)["code"] == "RESULTS_NOT_READY"


# --- A whole run: API -> worker -> API ------------------------------------------------------

def test_full_lifecycle_shortlist_report_and_health(client, tmp_path):
    """Create a run through the API, let the worker run it, then read the results.
    Both molecules get GOOD predictions, so both are accepted in round 1."""
    assert client.get("/api/v1/health", headers=AUTH).json()["worker_online"] is False
    run_id = post(client, RUNS, run_body()).json()["run_id"]

    store, worker = make_worker(tmp_path, ListGenerator([["CCO", "CCN"]]), default=GOOD)
    assert worker.process_next_run().run_id == run_id

    run = client.get(f"{RUNS}/{run_id}", headers=AUTH).json()
    assert run["status"] == "completed" and run["iteration"] == 1

    shortlist = client.get(f"{RUNS}/{run_id}/shortlist", headers=AUTH).json()
    assert [item["rank"] for item in shortlist["items"]] == [1, 2]
    assert shortlist["items"][0]["acceptor_smiles"] == "c1ccsc1"  # the fixed partner

    report = client.get(f"{RUNS}/{run_id}/report", headers=AUTH).json()
    assert len(report["candidates"]) == 2 and len(report["shortlist_ids"]) == 2
    assert report["pinned_versions"]["policy_version"] == "test-policy"
    assert report["not_yet_available"]  # honest about what isn't built yet

    assert client.get("/api/v1/health", headers=AUTH).json()["worker_online"] is True


# --- Cancelling -------------------------------------------------------------------------------

def test_cancel_a_queued_run_then_cancel_again_is_run_closed(client):
    run_id = post(client, RUNS, run_body()).json()["run_id"]
    key = str(uuid4())
    response = post(client, f"{RUNS}/{run_id}/cancel", {}, key=key)
    assert response.status_code == 202 and response.json()["cancel_requested"] is True
    run = client.get(f"{RUNS}/{run_id}", headers=AUTH).json()
    assert run["status"] == "stopped" and run["stop_reason"] == "cancelled"
    # The same request again (same key): the same reply, not an error.
    assert post(client, f"{RUNS}/{run_id}/cancel", {}, key=key).json() == response.json()
    # A NEW cancel request for a run that has ended: 409 RUN_CLOSED (D2).
    again = post(client, f"{RUNS}/{run_id}/cancel", {})
    assert again.status_code == 409 and error_of(again)["code"] == "RUN_CLOSED"


def test_cancel_body_must_be_empty(client):
    run_id = post(client, RUNS, run_body()).json()["run_id"]
    assert post(client, f"{RUNS}/{run_id}/cancel", {"now": True}).status_code == 422
