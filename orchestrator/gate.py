"""Deterministic transparent-OPV decision gate and Pareto ranking.

Pure functions, no LLM: the gate must give the same verdict for the same
prediction every time so runs are reproducible and auditable.

Violations are "code: detail" strings. The code is the stable part a
generator can key on; the detail is for humans.
"""

from collections import Counter

from .state import Candidate, Constraints, Decision


def evaluate(candidate: Candidate, c: Constraints) -> tuple[Decision, list[str]]:
    """Return (decision, violations) for one candidate against constraints c.

    Checks run in priority order:
      1. invalid / unpredicted        -> "reject"
      2. any threshold failed         -> "revise" (with one violation per failure)
      3. passes but too uncertain     -> "escalate"
      4. otherwise                    -> "accept"
    """
    # validate() already attached a reason (invalid_smiles or duplicate); keep it.
    if not candidate.valid or candidate.prediction is None:
        return "reject", candidate.violations or ["invalid_smiles: RDKit could not parse"]

    p = candidate.prediction  # shorthand for the predicted properties
    violations = []  # collects one "code: detail" string per failed threshold
    if p.pce < c.min_pce:
        violations.append(f"low_pce: PCE {p.pce:.2f}% < {c.min_pce:.2f}%")
    if p.avt < c.min_avt:
        violations.append(f"low_avt: AVT {p.avt:.2f} < {c.min_avt:.2f}")
    if p.opv_prob < c.min_opv_prob:
        violations.append(f"low_opv_prob: OPV-likeness {p.opv_prob:.2f} < {c.min_opv_prob:.2f}")
    if violations:
        return "revise", violations

    # Passes every threshold but the predictor isn't confident: a human (or DFT)
    # decides, rather than silently accepting an out-of-domain extrapolation.
    if p.uncertainty > c.max_uncertainty:
        return "escalate", [f"high_uncertainty: {p.uncertainty:.2f} > {c.max_uncertainty:.2f}"]
    return "accept", []


def summarize_feedback(batch: list[Candidate]) -> list[str]:
    """Collapse a batch's violations into one line per violation code.

    e.g. ["low_avt: 5/8 candidates", "invalid_smiles: 1/8 candidates"]
    This is what the generator receives as `feedback` on the next iteration.
    """
    # Count only the code (text before the first ":"), most frequent first.
    codes = Counter(v.split(":", 1)[0] for cand in batch for v in cand.violations)
    return [f"{code}: {n}/{len(batch)} candidates" for code, n in codes.most_common()]


def pareto_rank(candidates: list[Candidate]) -> list[Candidate]:
    """Sort by Pareto front over (PCE, AVT), then by LUE within a front.

    Front 1 = every candidate that no other candidate beats on BOTH PCE and
    AVT. Remove it, and the non-dominated set of what remains is front 2, and
    so on. Within each front, higher LUE (PCE x AVT) comes first.
    """
    remaining = [c for c in candidates if c.prediction is not None]  # not yet placed in a front
    ranked: list[Candidate] = []  # the output, best front first
    while remaining:
        # Candidates that nothing else in `remaining` dominates.
        front = [
            a for a in remaining
            if not any(_dominates(b, a) for b in remaining if b is not a)
        ]
        ranked += sorted(front, key=lambda c: c.prediction.lue, reverse=True)
        remaining = [c for c in remaining if c not in front]
    return ranked


def _dominates(a: Candidate, b: Candidate) -> bool:
    """True if a is at least as good as b on PCE and AVT, and strictly better on one."""
    pa, pb = a.prediction, b.prediction
    return pa.pce >= pb.pce and pa.avt >= pb.avt and (pa.pce > pb.pce or pa.avt > pb.avt)
