"""Tests for the near-miss formulas NM-1 to NM-4 (near_miss.py).

These use the same RUNNING EXAMPLE as near_miss.py's comments:
targets AVT >= 0.40, PCE >= 8%, LUE >= 3.0.
  Molecule A: AVT 0.38 +/- 0.04, PCE 9%   -> just misses AVT, borderline
  Molecule B: AVT 0.30 +/- 0.01, PCE 10%  -> clearly misses AVT
  Molecule C: AVT 0.39 +/- 0.01, PCE 7%   -> misses all three a little

pytest.approx(x) means "equal to x, allowing for tiny rounding errors".
Computers store decimals approximately, so (0.40 - 0.38) / 0.40 might come out
as 0.04999999999 instead of exactly 0.05.
"""

import math

import pytest

from orchestrator.gate import PlaceholderGate
from orchestrator.near_miss import (compute_near_miss, nm1_fraction_short, nm2_uncertainties_short,
                                    nm3_beats, nm4_made_progress, rank_near_misses)
from orchestrator.state import RunConstraints
from orchestrator.tests.fakes import CONFIG, POLICY, candidate, pred

# The running example's targets.
EXAMPLE = RunConstraints(avt_min_fraction=0.40, pce_min_percent=8.0, lue_min_percent=3.0)
A = pred(avt=0.38, s_avt=0.04, pce=9.0)  # LUE 3.42 (passes)
B = pred(avt=0.30, s_avt=0.01, pce=10.0)  # LUE 3.0 (exactly the minimum: passes)
C = pred(avt=0.39, s_avt=0.01, pce=7.0, s_pce=0.5)  # LUE 2.73 (misses)


# --- NM-1: fraction short -----------------------------------------------------

def test_nm1_fraction_short():
    assert nm1_fraction_short(0.38, 0.40) == pytest.approx(0.05)  # Molecule A: 5% short
    assert nm1_fraction_short(0.30, 0.40) == pytest.approx(0.25)  # Molecule B: 25% short
    assert nm1_fraction_short(0.50, 0.40) == 0.0  # passes -> 0
    assert nm1_fraction_short(0.40, 0.40) == 0.0  # exactly equal passes -> 0


# --- NM-2: uncertainties short, and borderline --------------------------------

def test_nm2_uncertainties_short():
    assert nm2_uncertainties_short(0.38, 0.40, 0.04) == pytest.approx(0.5)  # A: half an uncertainty
    assert nm2_uncertainties_short(0.30, 0.40, 0.01) == pytest.approx(10)  # B: ten uncertainties
    assert nm2_uncertainties_short(0.50, 0.40, 0.01) == 0.0  # passes -> 0
    assert nm2_uncertainties_short(0.38, 0.40, None) is None  # unknown uncertainty -> can't tell
    assert math.isinf(nm2_uncertainties_short(0.38, 0.40, 0.0))  # "perfectly sure" -> infinitely short


def test_borderline_flag():
    """Borderline = misses something, and every miss is within 1 uncertainty of passing."""
    assert compute_near_miss(candidate(A), EXAMPLE).borderline is True  # 0.5 uncertainties away
    assert compute_near_miss(candidate(B), EXAMPLE).borderline is False  # 10 uncertainties away
    assert compute_near_miss(candidate(pred(avt=0.38, s_avt=None, pce=9.0)), EXAMPLE).borderline is False
    assert compute_near_miss(candidate(pred(avt=0.5, pce=9.0)), EXAMPLE).borderline is False  # passes


# --- NM-3: Pareto comparison ----------------------------------------------------

def test_nm3_pareto_comparison():
    a = compute_near_miss(candidate(A), EXAMPLE)  # shortfall [0.05, 0, 0]
    b = compute_near_miss(candidate(B), EXAMPLE)  # shortfall [0.25, 0, 0]
    c = compute_near_miss(candidate(C), EXAMPLE)  # shortfall [0.025, 0.125, 0.09]
    assert nm3_beats(a, b) and not nm3_beats(b, a)  # A is closer on AVT, equal elsewhere
    assert not nm3_beats(a, c) and not nm3_beats(c, a)  # a trade-off: neither beats the other


# --- NM-4: progress, and the near-miss ranking ----------------------------------

def judged(prediction, candidate_id, iterations_remain=True):
    """A candidate after the decide step: gate verdict + near-miss numbers attached
    (exactly what graph.py's decide step does). Uses CONFIG's targets (AVT >= 0.4)."""
    cand = candidate(prediction, candidate_id)
    decision = PlaceholderGate().decide([cand], CONFIG, POLICY, iterations_remain)[0]
    return cand.model_copy(update={"decision": decision,
                                   "near_miss": compute_near_miss(cand, CONFIG.constraints)})


def test_nm4_progress():
    far = judged(pred(avt=0.30), "it1-c01")  # 25% short on AVT
    close = judged(pred(avt=0.38), "it2-c01")  # 5% short on AVT
    same_as_far = judged(pred(avt=0.30), "it2-c02")
    worse = judged(pred(avt=0.20), "it2-c03")  # 50% short
    accepted = judged(pred(), "it2-c04")
    assert nm4_made_progress([far], earlier=[]) is True  # first round with a near miss
    assert nm4_made_progress([close], earlier=[far]) is True  # closer than before
    assert nm4_made_progress([same_as_far], earlier=[far]) is False  # equal: not closer
    assert nm4_made_progress([worse], earlier=[far]) is False  # beaten by an earlier one
    assert nm4_made_progress([accepted], earlier=[far]) is True  # any Accept is progress


def test_rank_near_misses_closest_first_and_ignores_non_near_misses():
    far = judged(pred(avt=0.30), "it1-c01")
    close = judged(pred(avt=0.38), "it1-c02")
    accepted = judged(pred(), "it1-c03")  # not a near miss: it passed
    escalated = judged(pred(avt=0.1, s_avt=0.2), "it1-c04")  # not a near miss: untrusted numbers
    ranked = rank_near_misses([far, accepted, close, escalated])
    assert [c.candidate_id for c in ranked] == ["it1-c02", "it1-c01"]
