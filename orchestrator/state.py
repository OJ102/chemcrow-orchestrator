"""Shared state for the transparent-OPV LangGraph orchestrator.

Everything the graph knows lives here: the constraint packet built from the
user's objective, every candidate ever proposed (with its predictions and
violations), the structured feedback that drives revision, and loop counters
for the stop conditions.
"""

import operator
from typing import Annotated, Literal, Optional, TypedDict

from pydantic import BaseModel, Field

# Which half of the donor:acceptor pair this run is designing. The other half
# is held fixed (see Constraints.fixed_partner_smiles).
TargetType = Literal["donor", "acceptor"]

# The four verdicts the decision gate (gate.py) can give a candidate:
#   accept   - passes every threshold and the predictor is confident
#   revise   - valid molecule but fails a threshold; its violations become feedback
#   escalate - passes every threshold but the predictor is unsure -> human review
#   reject   - invalid SMILES or a duplicate; never scored
Decision = Literal["accept", "revise", "escalate", "reject"]


class Constraints(BaseModel):
    """Constraint packet: hard thresholds plus loop budget.

    One run designs one side of a D:A pair against a fixed partner, the same
    convention CycleChemist's finetune.py uses (--target_type / --fixed_smiles).

    Pydantic BaseModel validates field types when an instance is created, so a
    typo like min_pce="abc" fails immediately instead of deep inside the graph.
    """

    target_type: TargetType  # design a "donor" or an "acceptor"
    fixed_partner_smiles: str  # the other molecule of the pair, held constant all run

    # Hard constraints checked by the decision gate.
    min_pce: float = 2.0  # %, mirrors finetune.py's pce_min_threshold
    min_avt: float = 0.4  # 0-1 average visible transmittance
    min_opv_prob: float = 0.5  # minimum "looks like an OPV molecule" score (0-1)
    max_uncertainty: float = 0.15  # above this -> escalate, not accept

    # Loop budget / stop conditions.
    batch_size: int = 8  # how many SMILES the generator proposes per iteration
    target_count: int = 3  # stop once this many candidates are accepted
    max_iterations: int = 5  # hard cap on generate->gate rounds
    patience: int = 2  # iterations without best-score improvement


class Prediction(BaseModel):
    """What the property predictor returns for one candidate paired with its partner."""

    pce: float  # predicted PCE of the pair, %
    avt: float  # predicted average visible transmittance, 0-1
    homo: float  # eV
    lumo: float  # eV
    opv_prob: float  # OPV-likeness, 0-1
    uncertainty: float  # predictor's own uncertainty estimate

    @property
    def lue(self) -> float:
        """Light utilization efficiency proxy: PCE x AVT.

        The single number used to tie-break the ranking and to measure whether
        an iteration "improved" (see graph.py's gate node).
        """
        return self.pce * self.avt


class Candidate(BaseModel):
    """One proposed molecule plus everything the graph learns about it.

    Fields are filled in progressively as the candidate moves through the
    nodes: generate -> validate -> predict -> gate. Nodes never mutate a
    Candidate in place; they make updated copies with model_copy(update=...).
    """

    smiles: str  # raw string exactly as the generator produced it
    iteration: int  # which loop round produced it (1-based)
    canonical_smiles: Optional[str] = None  # RDKit's standard spelling; set by validate
    valid: bool = False  # True once RDKit parsed it and it isn't a duplicate
    prediction: Optional[Prediction] = None  # set by predict (valid candidates only)
    decision: Optional[Decision] = None  # set by gate
    violations: list[str] = Field(default_factory=list)  # "code: detail" strings explaining failures

    @property
    def key(self) -> str:
        """Identity used for duplicate detection and for matching human approvals.

        Canonical SMILES when available, so "CCO" and "OCC" (both ethanol)
        compare equal; otherwise falls back to the raw string.
        """
        return self.canonical_smiles or self.smiles


class OPVState(TypedDict, total=False):
    """The LangGraph state: the dict passed from node to node.

    Each node returns only the keys it wants to change. By default a returned
    value REPLACES the old one. Keys wrapped in Annotated[..., operator.add]
    use a "reducer" instead: the returned list is APPENDED to the existing one.

    total=False means no key is required to be present, which lets the graph
    start with only "objective" and "constraints".
    """

    objective: str  # the user's goal in plain English (not yet read by any node)
    constraints: Constraints  # thresholds + loop budget for this run
    evidence: list[str]  # passages from the retriever (collected, not yet used)

    batch: list[Candidate]  # candidates of the current iteration
    history: Annotated[list[Candidate], operator.add]  # every candidate ever scored
    accepted: Annotated[list[Candidate], operator.add]  # every accepted candidate, across iterations
    escalated: list[Candidate]  # awaiting human review
    feedback: list[str]  # structured violations fed back to the generator

    iteration: int  # current round number (0 before the first generate)
    best_score: float  # best LUE seen so far in this run
    stale_iterations: int  # consecutive rounds with no LUE improvement (compared to patience)
    status: Literal["running", "done", "stopped"]  # done = target met, stopped = budget ran out
    stop_reason: str  # human-readable explanation of why the run ended
    ranked: list[Candidate]  # final Pareto-ranked accepted candidates (the output)
