"""PLACEHOLDER Decision Gate and shortlist Ranker, following the D2 design document.

WHAT THIS FILE IS
-----------------
Two stand-ins for teammates' components, written to D2's rules so the loop
behaves like the real thing until they deliver:

  PlaceholderGate   -> stands in for Arleen's Transparent-OPV Decision Gate (D2 "AG")
  PlaceholderRanker -> stands in for Nyla's shortlist ranking            (D2 "A5")

When Arleen's and Nyla's real versions exist, they plug into the `gate` and
`ranker` slots of Tools (interfaces.py) instead, and this file can be deleted.
The orchestrator only CALLS the gate and ranker; it doesn't own them (D2).

WHY THE GATE IS "DETERMINISTIC" (NO AI)
---------------------------------------
Same input -> always the same verdict. These are plain if-statements, so runs
are reproducible and you can point at the exact rule behind every decision
(D2 rationale R3).

NOT IMPLEMENTED HERE (and why):
  - SA-score / synthesizability check: D2 escalates only if the policy REQUIRES
    it and it's unavailable. Our placeholder policy doesn't require it, because
    nobody has built that check yet.
  - The advisor's "escalate borderline candidates" and "diagnose the repair
    route": these are Arleen's gate decisions. We compute the borderline flag
    (near_miss.py, NM-2) so her gate can use it.
"""

from .near_miss import CONSTRAINED_FIELDS, minimums
from .state import (FIELD_UNITS, Candidate, Decision, Policy, RiskFlag, RunConfig,
                    ThresholdViolation)


class PlaceholderGate:
    """Stand-in for Arleen's Decision Gate. Implements the DecisionGate contract."""

    def decide(self, candidates: list[Candidate], config: RunConfig, policy: Policy,
               iterations_remain: bool) -> list[Decision]:
        """Give each candidate (all with successful predictions) one verdict.

        D2's rules, checked IN THIS ORDER (the first that applies wins):
          (1) failed prediction        -> no decision (the orchestrator filters these out first)
          (2) not trustworthy          -> Escalate   (out of domain, or an uncertainty
                                                      that is unknown or above the policy limit)
          (3) OPV-likeness too low     -> Stop       (below policy.opv_min)
          (4) misses a minimum target  -> Revise if rounds remain, otherwise Stop
          (5) otherwise                -> Accept
        Exactly equal to a minimum PASSES (D2).

        WHY "not trustworthy" comes BEFORE the targets: if the numbers can't be
        trusted, "AVT failed" might itself be a wrong guess. A human or DFT should
        look instead of us acting on a bad number.

        Returns one Decision per candidate, in the same order (D1).
        """
        return [self._decide_one(c, config, policy, iterations_remain) for c in candidates]

    def _decide_one(self, cand: Candidate, config: RunConfig, policy: Policy,
                    iterations_remain: bool) -> Decision:
        """Apply rules (2)-(5) to one candidate. (Leading underscore = only used inside this class.)"""
        p = cand.prediction  # the prediction (status is "ok" here)
        values, unc = p.values, p.uncertainty  # the six numbers and their uncertainties
        flags: list[RiskFlag] = []  # warning labels, collected as we go

        # Work out the missed targets FIRST, even if we end up escalating, so the
        # decision always records them (useful to the human reviewing it).
        violations = self._threshold_violations(cand, config)
        if violations:
            flags.append("CONSTRAINT_VIOLATION")

        # --- Rule (2): is the prediction trustworthy? ---
        trust_problems = []  # human-readable reasons, for the rationale sentence
        if p.in_domain is not True:  # False, or unknown (None): D2 escalates unknown trust too
            flags.append("OUT_OF_DOMAIN")
            trust_problems.append("pair is outside the model's training domain")
        # Every value listed in the policy must have a known uncertainty within its limit.
        # .items() gives (key, value) pairs of a dictionary.
        for name, limit in policy.uncertainty_max_by_target.items():
            sigma = getattr(unc, name)  # this value's uncertainty
            if sigma is None:
                flags.append("UNCERTAINTY_UNAVAILABLE")
                trust_problems.append(f"no uncertainty for {name}")
            elif sigma > limit:  # equal to the limit passes (D2)
                flags.append("HIGH_UNCERTAINTY")
                trust_problems.append(f"{name} uncertainty {sigma:.3g} > limit {limit:.3g}")
        # D1: risk flags are unique. dict.fromkeys keeps the first copy of each, in order.
        flags = list(dict.fromkeys(flags))
        if trust_problems:
            return Decision(candidate_id=cand.candidate_id, outcome="Escalate",
                            rationale="Escalated for DFT/lab review: " + "; ".join(trust_problems) + ".",
                            risk_flags=flags, violations=violations)

        # --- Rule (3): does it look like a solar-cell material at all? ---
        if values.opv_likeness < policy.opv_min:
            return Decision(candidate_id=cand.candidate_id, outcome="Stop",
                            rationale=(f"Stopped: OPV-likeness {values.opv_likeness:.2f} "
                                       f"< policy minimum {policy.opv_min:.2f}."),
                            risk_flags=flags, violations=violations)

        # --- Rule (4): missed a target ---
        if violations:
            missed = ", ".join(v.field for v in violations)  # e.g. "avt_fraction, lue_percent"
            if iterations_remain:
                return Decision(candidate_id=cand.candidate_id, outcome="Revise",
                                rationale=f"Revise: below minimum on {missed}.",
                                risk_flags=flags, violations=violations)
            return Decision(candidate_id=cand.candidate_id, outcome="Stop",
                            rationale=f"Stopped: below minimum on {missed} and no rounds remain.",
                            risk_flags=flags, violations=violations)

        # --- Rule (5): passed everything ---
        return Decision(candidate_id=cand.candidate_id, outcome="Accept",
                        rationale="Accepted: meets every minimum with trustworthy predictions.",
                        risk_flags=flags, violations=[])

    @staticmethod
    def _threshold_violations(cand: Candidate, config: RunConfig) -> list[ThresholdViolation]:
        """List every minimum target this candidate misses, as machine-readable records.

        Example: AVT 0.38 with minimum 0.40 ->
            ThresholdViolation(field="avt_fraction", actual=0.38, minimum=0.40, unit="fraction")
        Only "actual < minimum" is a violation, so equality passes.
        (@staticmethod: this function doesn't need `self`; it just lives in the class.)
        """
        mins = minimums(config.constraints)
        values = cand.prediction.values
        return [ThresholdViolation(field=f, actual=getattr(values, f), minimum=mins[f], unit=FIELD_UNITS[f])
                for f in CONSTRAINED_FIELDS if getattr(values, f) < mins[f]]


class PlaceholderRanker:
    """Stand-in for Nyla's shortlist ranking (D2 "A5"). Implements the Ranker contract."""

    def rank(self, accepted: list[Candidate]) -> list[Candidate]:
        """Sort accepted candidates best first, following D2 A5.

          1. Remove repeated pairs: keep the EARLIEST (lowest iteration, then
             lowest candidate_id) copy of each donor>>acceptor pair.
          2. Pareto fronts on the two goals, AVT and LUE (higher is better for both).
             PCE isn't a ranking goal: it's a hard floor the gate already enforced.
             "X dominates Y" = X is at least as high on both AVT and LUE, and
             strictly higher on at least one.
          3. Within each front, ties are broken by: higher LUE, then higher AVT,
             then higher PCE, then the pair text alphabetically (so the order is
             always the same).

        Example (AVT, LUE): X=(0.50, 4.0), Y=(0.80, 2.0), Z=(0.40, 3.5)
          X dominates Z (higher on both). X vs Y: neither dominates.
          Front 1 = [X (LUE 4.0), Y (LUE 2.0)], front 2 = [Z]. Result: X, Y, Z.
        """
        # Step 1: one copy per pair. Sort earliest-first, then keep the first seen.
        earliest_first = sorted(accepted, key=lambda c: (c.iteration, c.candidate_id))
        unique: dict[str, Candidate] = {}  # pair_key -> the candidate we keep
        for cand in earliest_first:
            unique.setdefault(cand.pair_key, cand)  # only stores it if the key isn't there yet
        remaining = list(unique.values())

        # Steps 2 and 3: peel off Pareto fronts, sorting each one by the tie-break rule.
        ranked: list[Candidate] = []
        while remaining:
            front = [a for a in remaining
                     if not any(self._dominates(b, a) for b in remaining if b is not a)]
            # Sorting by a tuple compares the first item, then the second if tied, etc.
            # Minus signs turn "higher first" into the default "lowest first" order.
            ranked += sorted(front, key=lambda c: (-c.prediction.values.lue_percent,
                                                   -c.prediction.values.avt_fraction,
                                                   -c.prediction.values.pce_percent,
                                                   c.pair_key))
            remaining = [c for c in remaining if c not in front]
        return ranked

    @staticmethod
    def _dominates(a: Candidate, b: Candidate) -> bool:
        """True if a is at least as good as b on AVT and LUE, and strictly better on one."""
        va, vb = a.prediction.values, b.prediction.values
        return (va.avt_fraction >= vb.avt_fraction and va.lue_percent >= vb.lue_percent
                and (va.avt_fraction > vb.avt_fraction or va.lue_percent > vb.lue_percent))
