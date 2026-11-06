"""Tests for the PLACEHOLDER gate and ranker (gate.py), which follow D2's rules.

Run from the repo root, in the langgraph venv:
    python -m pytest orchestrator/tests -v

Each test builds one candidate with a chosen prediction, asks the gate for a
verdict, and checks it. Shared fakes (pred, candidate, GOOD, OPAQUE, ...) are
explained in fakes.py.
"""

from orchestrator.gate import PlaceholderGate, PlaceholderRanker
from orchestrator.state import ThresholdViolation
from orchestrator.tests.fakes import CONFIG, GOOD, OPAQUE, POLICY, UNSURE, candidate, pred


def verdict(prediction, iterations_remain=True):
    """Ask the placeholder gate about ONE candidate with this prediction; return its Decision.

    The gate takes a list and returns a list, so we wrap the one candidate in [...]
    and take the first (only) result with [0].
    """
    return PlaceholderGate().decide([candidate(prediction)], CONFIG, POLICY, iterations_remain)[0]


# --- Rule (5): Accept ------------------------------------------------------

def test_good_prediction_is_accepted():
    """Passes every minimum, confident, in domain -> Accept with no violations or flags."""
    d = verdict(GOOD)
    assert d.outcome == "Accept"
    assert d.violations == [] and d.risk_flags == []


def test_exactly_equal_to_a_minimum_passes():
    """D2: equality passes. AVT 0.4 = min, PCE 2.0 = min, LUE 0.4 x 2.0 = 0.8 = min."""
    assert verdict(pred(avt=0.4, pce=2.0)).outcome == "Accept"


# --- Rule (4): Revise / Stop for missed targets ------------------------------

def test_missed_target_gives_revise_with_machine_readable_violation():
    """Too dark -> Revise, and the violation says exactly which value and by how much
    (the advisor's "machine-readable feedback, not merely 'AVT failed'")."""
    d = verdict(OPAQUE)
    assert d.outcome == "Revise"
    assert d.violations == [ThresholdViolation(field="avt_fraction", actual=0.1,
                                               minimum=0.4, unit="fraction")]
    assert d.risk_flags == ["CONSTRAINT_VIOLATION"]


def test_missed_target_on_last_round_gives_stop():
    """D2: a miss is Revise only if rounds remain; otherwise this candidate's path Stops."""
    assert verdict(OPAQUE, iterations_remain=False).outcome == "Stop"


# --- Rule (3): Stop for low OPV-likeness -------------------------------------

def test_low_opv_likeness_gives_stop():
    """OPV-likeness 0.2 < policy minimum 0.5 -> doesn't look like a solar material -> Stop."""
    assert verdict(pred(opv=0.2)).outcome == "Stop"


# --- Rule (2): Escalate when the prediction can't be trusted ------------------

def test_high_uncertainty_is_escalated():
    """AVT uncertainty 0.2 > policy limit 0.05 -> Escalate (set aside for DFT/lab)."""
    d = verdict(UNSURE)
    assert d.outcome == "Escalate" and "HIGH_UNCERTAINTY" in d.risk_flags


def test_missing_uncertainty_is_escalated():
    """No uncertainty given (None) is NOT the same as 'perfectly sure' (D1) -> Escalate."""
    d = verdict(pred(s_avt=None))
    assert d.outcome == "Escalate" and "UNCERTAINTY_UNAVAILABLE" in d.risk_flags


def test_out_of_domain_is_escalated():
    """Molecule unlike the predictor's training data -> its numbers are guesses -> Escalate."""
    d = verdict(pred(in_domain=False))
    assert d.outcome == "Escalate" and "OUT_OF_DOMAIN" in d.risk_flags


def test_escalate_takes_priority_over_missed_targets():
    """Too dark AND too uncertain -> Escalate, not Revise (D2 checks trust first).
    The missed target is still recorded, for the human reviewer."""
    d = verdict(pred(avt=0.1, s_avt=0.2))
    assert d.outcome == "Escalate"
    assert d.violations[0].field == "avt_fraction"


# --- The ranker (Nyla's D2 rule A5) --------------------------------------------

def test_ranker_uses_pareto_on_avt_and_lue():
    """(AVT, LUE): X=(0.5, 4.0), Y=(0.8, 2.0), Z=(0.4, 3.5).
    X beats Z on both, so Z is last. X and Y trade off (front 1), ordered by LUE: X, Y."""
    # PCE is chosen so that LUE = AVT x PCE comes out as listed above.
    x = candidate(pred(avt=0.5, pce=8.0), "it1-c01", smiles="CCO")
    y = candidate(pred(avt=0.8, pce=2.5), "it1-c02", smiles="CCN")
    z = candidate(pred(avt=0.4, pce=8.75), "it1-c03", smiles="CCC")
    ranked = PlaceholderRanker().rank([z, y, x])
    assert [c.candidate_id for c in ranked] == ["it1-c01", "it1-c02", "it1-c03"]


def test_ranker_keeps_only_the_earliest_copy_of_a_repeated_pair():
    """The same donor>>acceptor pair accepted in rounds 1 and 2 appears once (round 1's copy)."""
    first = candidate(GOOD, "it1-c01", iteration=1)
    again = candidate(GOOD, "it2-c01", iteration=2)  # same molecule "CCO" -> same pair
    ranked = PlaceholderRanker().rank([again, first])
    assert [c.candidate_id for c in ranked] == ["it1-c01"]
