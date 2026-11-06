"""Tests for the whole loop (graph.py), using the fully controlled fakes in fakes.py.

Run from the repo root, in the langgraph venv:
    python -m pytest orchestrator/tests -v

Each test prepares the molecules the fake generator will hand out each round
(`batches`) and which molecules get special predictions (`table`), runs the
graph once with `run(...)`, then checks the final state.
Reminder (fakes.py): GOOD -> Accept, OPAQUE -> Revise, UNSURE -> Escalate,
FAILED -> failed prediction.
"""

import pytest
from pydantic import ValidationError

from orchestrator import RunConfig
from orchestrator.tests.fakes import CONSTRAINTS, FAILED, OPAQUE, UNSURE, FakeClock, run


def ids(candidates):
    """The candidate_ids of a list of candidates, e.g. ["it1-c01", "it1-c02"]. Makes asserts short."""
    return [c.candidate_id for c in candidates]


# --- How a run ends -------------------------------------------------------------

def test_completes_when_nothing_is_left_to_revise():
    """Three GOOD molecules in round 1: all accepted, nothing to revise -> completed after 1 round."""
    _, result = run([["CCO", "CCN", "CCC"]])
    assert result["status"] == "completed"
    assert result["stop_reason"] == "no candidates left to revise"
    assert result["iteration"] == 1 and len(result["ranked"]) == 3


def test_revise_rounds_pass_parents_and_stop_at_the_iteration_budget():
    """Every molecule is too dark (OPAQUE), so each round's candidate is revised in the next.
    Checks: the parent reaches the generator WITH its machine-readable violation, the
    new candidate records its parent (lineage), and the run stops after max_iterations (3)."""
    gen, result = run([["c1ccccc1"], ["Cc1ccccc1"], ["CCc1ccccc1"]], default=OPAQUE)
    assert gen.packets[0].parents == []  # round 1: invent from scratch
    parent = gen.packets[1].parents[0]  # round 2 revises round 1's candidate
    assert parent.candidate_id == "it1-c01"
    assert parent.decision.violations[0].field == "avt_fraction"
    assert result["history"][1].parent_candidate_id == "it1-c01"  # lineage recorded
    assert result["status"] == "stopped"
    assert result["stop_reason"] == "iteration budget (3) exhausted"
    assert ids(result["near_misses"]) == ["it1-c01", "it2-c01", "it3-c01"]  # all kept (advisor)


def test_all_predictions_failing_marks_the_run_failed():
    _, result = run([["CCO"]], default=FAILED)
    assert result["status"] == "failed"


def test_runtime_budget_stops_the_run():
    """A fake clock where every step appears to take 1,000 s of WORK. Round 1 alone
    is 6 steps = 6,000 s, more than the 1,800 s budget. The run would otherwise
    revise again, but runs out of time after round 1."""
    _, result = run([["c1ccccc1"], ["Cc1ccccc1"]], default=OPAQUE, clock=FakeClock(step=1000))
    assert result["status"] == "stopped" and "runtime budget" in result["stop_reason"]
    assert result["iteration"] == 1
    # 7 timed steps: start, retrieve, generate, validate, predict, decide, finish.
    assert result["active_seconds"] == 7000


# --- Escalate = set aside -------------------------------------------------------

def test_escalated_candidates_are_set_aside_and_the_run_finishes():
    """CCN is too uncertain -> set aside (not shortlisted), and the run does NOT pause."""
    _, result = run([["CCO", "CCN"]], table={"CCN": UNSURE})
    assert "__interrupt__" not in result  # this key would appear if the graph had paused
    assert result["status"] == "completed"
    assert ids(result["escalated"]) == ["it1-c02"]
    assert ids(result["ranked"]) == ["it1-c01"]  # only CCO is on the shortlist


# --- Our validate safety net, and failed predictions ------------------------------

def test_validate_catches_invalid_and_duplicate_molecules():
    """"OCC" is ethanol again (same pair as "CCO"); "not_a_smiles" isn't a molecule.
    Both are logged with a ChemistryViolation and never reach the predictor."""
    _, result = run([["CCO", "OCC", "not_a_smiles"]])
    codes = {c.candidate_id: [v.check_code for v in c.chemistry_violations] for c in result["history"]}
    assert codes == {"it1-c01": [], "it1-c02": ["duplicate_pair"], "it1-c03": ["invalid_smiles"]}
    assert result["history"][1].prediction is None  # never predicted


def test_a_failed_prediction_is_logged_and_the_rest_continue():
    _, result = run([["CCO", "CCN"]], table={"CCN": FAILED})
    assert ids(result["failed_predictions"]) == ["it1-c02"]
    assert ids(result["ranked"]) == ["it1-c01"] and result["status"] == "completed"


# --- The optional extra stop rules (our addition) ---------------------------------

def test_optional_rules_are_off_by_default():
    config = RunConfig(target_type="donor", fixed_partner_smiles="C", constraints=CONSTRAINTS)
    assert config.target_count is None and config.patience is None


def test_target_count_finishes_early():
    """Benzene needs revising, but 1 accepted (CCO) already meets target_count=1."""
    _, result = run([["c1ccccc1", "CCO"]], table={"c1ccccc1": OPAQUE}, target_count=1)
    assert result["status"] == "completed" and "target count" in result["stop_reason"]


def test_patience_stops_when_there_is_no_near_miss_progress():
    """Round 2's near miss is exactly as far short as round 1's (formula NM-4: no
    progress). With patience=1, one round without progress stops the run."""
    _, result = run([["c1ccccc1"], ["Cc1ccccc1"]], default=OPAQUE, patience=1, max_iterations=5)
    assert result["status"] == "stopped"
    assert result["stop_reason"] == "no near-miss progress for 1 iterations"


# --- Run settings are checked up front ----------------------------------------------

def test_an_invalid_fixed_partner_is_rejected():
    """If the partner isn't a real molecule, every pair would be invalid, so refuse the run.
    pytest.raises(X) means "the code inside this block MUST raise error X"."""
    with pytest.raises(ValidationError):
        RunConfig(target_type="donor", fixed_partner_smiles="not_a_smiles", constraints=CONSTRAINTS)
