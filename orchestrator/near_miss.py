"""Near-miss formulas NM-1 to NM-4 (from the advisor's guidance, 2026-10-23).

WHAT THIS FILE IS
-----------------
A "near miss" is a candidate whose prediction is trustworthy but which missed
one or more minimum targets (AVT, PCE, LUE). The advisor asked us to:
  - KEEP near misses (not throw them away),
  - rank them by "uncertainty-aware normalized constraint violations and Pareto
    dominance, not a compensatory weighted score".
This file turns that sentence into four small formulas. Each has an ID you can
point to when asked:

  NM-1  fraction short       how far below the minimum, as a fraction of the minimum
  NM-2  uncertainties short  how far below, measured in units of the prediction's
                             uncertainty; within 1 uncertainty of passing = "borderline"
  NM-3  Pareto comparison    near miss X beats Y if X is no further short on EVERY
                             target and less short on at least one (no weights)
  NM-4  progress             a round "made progress" if it produced an Accept, or a
                             near miss that no earlier candidate beats (NM-3)

RUNNING EXAMPLE used in the comments (targets: AVT >= 0.40, PCE >= 8%, LUE >= 3.0):
  Molecule A: AVT 0.38 +/- 0.04, PCE 9%, LUE 3.42  -> just misses AVT, borderline
  Molecule B: AVT 0.30 +/- 0.01, PCE 10%, LUE 3.0  -> clearly misses AVT
  Molecule C: AVT 0.39 +/- 0.01, PCE 7% +/- 0.5, LUE 2.73 -> misses all three a little

Who uses these:
  - graph.py's decide step computes NM-1/NM-2 for every judged candidate and
    stores them on the candidate (Candidate.near_miss), so they appear in reports.
  - graph.py uses NM-4 for the optional `patience` stop rule.
  - graph.py's finish step uses NM-3 to sort the final near-miss list.
"""

from typing import Optional

from .state import Candidate, ConstrainedField, NearMiss, RunConstraints

# The NM-2 cut-off: a miss within this many uncertainties of passing is
# "borderline". Agreed with Shashank on 2026-10-26: 1 uncertainty (1 standard
# deviation). Change this one number to use a different cut-off.
BORDERLINE_SIGMAS = 1.0

# The three targets that have a minimum, in a fixed order (used for loops).
CONSTRAINED_FIELDS: tuple[ConstrainedField, ...] = ("avt_fraction", "pce_percent", "lue_percent")


def minimums(constraints: RunConstraints) -> dict[ConstrainedField, float]:
    """Look up the minimum for each target, keyed by the D1 value name.

    RunConstraints names its fields avt_min_fraction etc., while predictions use
    avt_fraction etc. This dictionary connects the two:
        {"avt_fraction": 0.40, "pce_percent": 8.0, "lue_percent": 3.0}
    """
    return {
        "avt_fraction": constraints.avt_min_fraction,
        "pce_percent": constraints.pce_min_percent,
        "lue_percent": constraints.lue_min_percent,
    }


# ============================================================================
# FORMULA NM-1: fraction short
# ============================================================================

def nm1_fraction_short(actual: float, minimum: float) -> float:
    """FORMULA NM-1: how far below the minimum, as a fraction of the minimum.

        fraction short = (minimum - actual) / minimum      if actual < minimum
                       = 0                                  otherwise (it passes)

    Example (Molecule A, AVT): (0.40 - 0.38) / 0.40 = 0.05, i.e. 5% short.
    Example (Molecule B, AVT): (0.40 - 0.30) / 0.40 = 0.25, i.e. 25% short.

    WHY divide by the minimum ("normalized"): it puts every target on the same
    scale. Being 0.02 short on AVT and 0.4 short on PCE are both "5% short",
    so they can be compared fairly without choosing weights.

    Exactly equal to the minimum passes (D2: "equality with a minimum passes").
    A minimum of 0 can't be missed (predictions are never negative), so the
    division below never divides by zero.
    """
    if actual >= minimum:
        return 0.0
    return (minimum - actual) / minimum


# ============================================================================
# FORMULA NM-2: uncertainties short (and the "borderline" cut-off)
# ============================================================================

def nm2_uncertainties_short(actual: float, minimum: float,
                            uncertainty: Optional[float]) -> Optional[float]:
    """FORMULA NM-2: how far below the minimum, in units of the prediction's uncertainty.

        uncertainties short = (minimum - actual) / uncertainty   if actual < minimum
                            = 0                                   otherwise (it passes)

    Example (Molecule A, AVT): (0.40 - 0.38) / 0.04 = 0.5  -> half an uncertainty away.
    Example (Molecule B, AVT): (0.40 - 0.30) / 0.01 = 10   -> ten uncertainties away.

    WHY ("uncertainty-aware"): A and B both "fail", but A's true value could easily
    be 0.41 (the prediction is only accurate to about +/- 0.04), while B is almost
    certainly below 0.40. This number tells those two situations apart.

    Special cases:
      - uncertainty is None (predictor didn't say how sure it is) -> None ("can't tell")
      - uncertainty is 0 (predictor claims to be perfectly sure) -> infinity
        ("certainly fails"), because dividing by zero isn't allowed
    """
    if actual >= minimum:
        return 0.0
    if uncertainty is None:
        return None
    if uncertainty == 0:
        return float("inf")
    return (minimum - actual) / uncertainty


def compute_near_miss(candidate: Candidate, constraints: RunConstraints) -> NearMiss:
    """Apply NM-1 and NM-2 to every target of one candidate, plus the borderline test.

    The candidate must have a successful prediction (status "ok").

    BORDERLINE (NM-2 cut-off): the candidate misses at least one target, AND every
    target it misses is within BORDERLINE_SIGMAS (1) uncertainty of passing.
      Molecule A: misses only AVT, 0.5 uncertainties away     -> borderline
      Molecule B: misses AVT, 10 uncertainties away           -> not borderline
      A candidate that passes everything                      -> not borderline (not a miss)
      A miss whose uncertainty is unknown (None)              -> not borderline (can't tell)
    Today this is a stored flag. The advisor wants borderline candidates escalated
    to an expert. That rule belongs in Arleen's gate, which can use this flag.
    """
    values = candidate.prediction.values  # the six predicted numbers
    uncertainty = candidate.prediction.uncertainty  # how unsure about each one
    mins = minimums(constraints)

    # getattr(obj, "avt_fraction") is the same as obj.avt_fraction, but lets us
    # choose the field name with a variable (we loop over the three names).
    shortfall = {f: nm1_fraction_short(getattr(values, f), mins[f]) for f in CONSTRAINED_FIELDS}
    sigmas = {f: nm2_uncertainties_short(getattr(values, f), mins[f], getattr(uncertainty, f))
              for f in CONSTRAINED_FIELDS}

    missed = [f for f in CONSTRAINED_FIELDS if shortfall[f] > 0]  # targets it failed
    borderline = bool(missed) and all(
        sigmas[f] is not None and sigmas[f] <= BORDERLINE_SIGMAS for f in missed)
    return NearMiss(shortfall=shortfall, sigmas_short=sigmas, borderline=borderline)


# ============================================================================
# FORMULA NM-3: Pareto comparison of near misses (no weights)
# ============================================================================

def nm3_beats(x: NearMiss, y: NearMiss) -> bool:
    """FORMULA NM-3: True if near miss x is closer to passing than y, in the Pareto sense.

    x beats y when x is NO further short (NM-1) on EVERY target, and LESS short on
    at least one. Example (NM-1 values as [AVT, PCE, LUE]):
        A = [0.05, 0,    0   ]
        B = [0.25, 0,    0   ]   -> A beats B (less short on AVT, equal elsewhere)
        C = [0.025, 0.125, 0.09] -> A vs C: A is further short on AVT, C is further
                                    short on PCE and LUE -> NEITHER beats the other
    WHY no weights (the advisor's "not a compensatory weighted score"): a weighted
    total like 0.6*AVT + 0.4*PCE lets being great at one thing hide being bad at
    another. Pareto only says "better" when it's better at everything that matters.
    The team's D2 (rationale R8) made the same choice for the shortlist.
    """
    no_worse = all(x.shortfall[f] <= y.shortfall[f] for f in CONSTRAINED_FIELDS)
    strictly_better = any(x.shortfall[f] < y.shortfall[f] for f in CONSTRAINED_FIELDS)
    return no_worse and strictly_better


def is_near_miss(candidate: Candidate) -> bool:
    """True if this candidate counts as a near miss: trustworthy prediction, missed a target.

    That means the gate gave it threshold violations and the outcome is Revise or
    Stop. Escalated candidates are NOT near misses here, because their numbers
    can't be trusted; they're in the set-aside pile instead. Candidates stopped for
    a reason other than a missed target (e.g. low OPV-likeness) have no threshold
    violations, so they're not near misses either.
    """
    return (candidate.decision is not None
            and candidate.decision.outcome in ("Revise", "Stop")
            and bool(candidate.decision.violations))


def rank_near_misses(candidates: list[Candidate]) -> list[Candidate]:
    """Sort near misses into Pareto "fronts" using NM-3, closest to passing first.

    Front 1 = near misses that no other near miss beats. Remove them; the
    unbeaten ones among the rest form front 2; and so on. Within a front, the
    order is by candidate_id, simply so the result is the same every run.
    (Deliberately NOT by a combined score, which would bring weights back in.)
    Non-near-misses in the input are ignored.
    """
    remaining = [c for c in candidates if is_near_miss(c)]  # not yet placed in a front
    ranked: list[Candidate] = []
    while remaining:
        front = [c for c in remaining
                 if not any(nm3_beats(o.near_miss, c.near_miss) for o in remaining if o is not c)]
        ranked += sorted(front, key=lambda c: c.candidate_id)
        remaining = [c for c in remaining if c not in front]
    return ranked


# ============================================================================
# FORMULA NM-4: did this round make progress? (used by the optional `patience`)
# ============================================================================

def nm4_made_progress(this_round: list[Candidate], earlier: list[Candidate]) -> bool:
    """FORMULA NM-4: True if this round got closer to the goal.

    A round made progress if EITHER
      (a) it produced at least one Accept, OR
      (b) it produced a near miss that no earlier near miss beats (NM-3) or
          equals, i.e. it got closer to passing in at least one way without
          being worse in every way.

    Failed candidates don't count by themselves (Shashank, 2026-10-15): only
    being CLOSER than before counts. Examples, with earlier near miss B = [0.25, 0, 0]:
        new A = [0.05, 0, 0]   -> nothing earlier beats or equals it -> progress
        new B' = [0.25, 0, 0]  -> equals B, no closer -> no progress
        new D = [0.30, 0, 0]   -> B beats it -> no progress
    The first round always makes progress if it has any Accept or near miss.

    Parameters:
        this_round - the candidates judged in this round
        earlier    - every candidate from earlier rounds (the run's history)
    """
    if any(c.decision is not None and c.decision.outcome == "Accept" for c in this_round):
        return True
    earlier_misses = [c.near_miss for c in earlier if is_near_miss(c)]
    for cand in this_round:
        if not is_near_miss(cand):
            continue
        beaten_or_equalled = any(nm3_beats(e, cand.near_miss) or e.shortfall == cand.near_miss.shortfall
                                 for e in earlier_misses)
        if not beaten_or_equalled:
            return True
    return False
