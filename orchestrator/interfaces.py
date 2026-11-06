"""Tool interfaces the orchestrator depends on.

WHAT THIS FILE IS
-----------------
The orchestrator doesn't invent, predict, judge or rank molecules itself. It
CALLS five tools that teammates own (owners from the D2 design document):

  Slot        Contract            Owner                Real tool
  ---------   -----------------   ------------------   -----------------------------
  retriever   Retriever           Kyle   (C1)          search over the knowledge base
  generator   CandidateGenerator  Jaxon  (C3)          MatGPT + RDKit validation
  predictor   PropertyPredictor   Arleen (C4)          Chemprop
  gate        DecisionGate        Arleen (Gate)        the Transparent-OPV Decision Gate
  ranker      Ranker              Nyla   (C5)          shortlist Pareto ranking

This file writes down a CONTRACT for each one: "whatever you build must have a
method with exactly this name, these inputs, and this output". The graph
(graph.py) only ever uses these contracts.

WHY: today we plug in FAKE tools (stubs.py and the placeholders in gate.py) so
the loop runs. Tests plug in tiny fakes they control. Later the teammates' real
tools plug into the same slots, and graph.py doesn't change. Like a wall socket:
anything with the right plug shape works.

HOW: typing.Protocol is "duck typing with type hints". Any class with a method
of the right name and shape counts, without inheriting from the Protocol. The
`...` in each method body literally means "no code here": a Protocol only
describes the shape.
"""

# @dataclass: a shortcut that writes the setup code for a class that just stores values.
from dataclasses import dataclass

# Protocol: the "contract" tool described above.
from typing import Protocol

# BaseModel: for the two small data shapes defined in this file (they're
# specific to these contracts, so they live here rather than in state.py).
from pydantic import BaseModel

# Our own data shapes from state.py (the "." means "from this same folder").
from .state import (Candidate, Decision, Policy, Prediction, RunConfig, RunConstraints,
                    TargetType)


# ============================================================================
# DATA PASSED TO / RETURNED FROM THE GENERATOR
# ============================================================================

class ConstraintPacket(BaseModel):
    """Everything the generator needs for one round (D2 "ConstraintPacket").

    The orchestrator builds this every round and hands it to the generator.

    Difference from D2: D2 passes `parent_candidate_ids` (just IDs, because in
    the final system the generator can look candidates up in the shared
    database). We have no database yet, so we pass the full parent Candidates,
    which also carry their decisions and machine-readable violations. That is the
    start of the advisor's "structured repair instructions". The rest (what to
    preserve, which partner to change, which edits already failed) gets added
    once Jaxon says what his generator can use.
    """

    run_id: str  # which run this is for
    iteration: int  # which round this is (1, 2, 3...)
    target_type: TargetType  # invent "donor"s or "acceptor"s
    fixed_partner_smiles: str  # the other half of every pair
    constraints: RunConstraints  # the three minimum targets (so the generator knows the goal)
    # Candidates to revise this round: last round's "Revise" decisions.
    # EMPTY in round 1, which means "invent from scratch".
    parents: list[Candidate]
    candidate_limit: int  # maximum number of new candidates to return
    seed: int  # makes "random" choices repeatable


class GenerationResult(BaseModel):
    """What the generator returns (D2: generate_validate -> {valid, rejected, violations}).

    D2 puts validation INSIDE Jaxon's component, so the generator itself splits
    its output into two lists. (D2 also lists `violations` separately; we keep
    them on each rejected Candidate's chemistry_violations instead, so they can't
    get separated from the candidate they belong to.)
    """

    valid: list[Candidate]  # passed the generator's own checks
    rejected: list[Candidate]  # failed them; each has chemistry_violations saying why


# ============================================================================
# THE FIVE CONTRACTS
# ============================================================================

class Retriever(Protocol):
    """Kyle's evidence finder (C1): searches papers/patents in the knowledge base."""

    def retrieve(self, objective: str, config: RunConfig) -> list[str]:
        """Return text passages relevant to the run's goal.

        Inputs:  objective - the goal in plain English
                 config    - the run's settings (so it can search for relevant chemistry)
        Output:  a list of text passages
        """
        ...


class CandidateGenerator(Protocol):
    """Jaxon's molecule inventor + validator (C3): MatGPT, then RDKit checks."""

    def generate_validate(self, packet: ConstraintPacket) -> GenerationResult:
        """Invent up to packet.candidate_limit new candidates, then validate them.

        Round 1 (no parents): invent from scratch.
        Later rounds: revise the parents, each new Candidate's parent_candidate_id
        naming the parent it came from (lineage).
        Each returned Candidate must have a unique candidate_id and its
        generated_smiles filled in.

        If the generator itself is down, it should RAISE an error, not return
        empty lists (D2: "generator outage is RetryableDependency, not []").
        """
        ...


class PropertyPredictor(Protocol):
    """Arleen's Chemprop model (C4)."""

    def predict(self, candidates: list[Candidate]) -> list[Prediction]:
        """Predict the six D1 values for each candidate pair, as one batch.

        Input:   valid candidates (donor_smiles and acceptor_smiles filled in)
        Output:  EXACTLY one Prediction per input candidate, in the SAME order (D1).
                 A single bad candidate gets status="failed"; the others can still succeed.
        If the whole model is down, it should RAISE an error (D1: total outage = 503).
        """
        ...


class DecisionGate(Protocol):
    """Arleen's Transparent-OPV Decision Gate (D2 behaviour "AG")."""

    def decide(self, candidates: list[Candidate], config: RunConfig, policy: Policy,
               iterations_remain: bool) -> list[Decision]:
        """Give each candidate a verdict: Accept / Revise / Escalate / Stop.

        Inputs:  candidates        - only those with a SUCCESSFUL prediction (status "ok")
                 config            - the run's minimum targets
                 policy            - the approved limits (uncertainty, opv_min)
                 iterations_remain - True if another round is allowed. D2: a miss
                                     becomes "Revise" if rounds remain, otherwise "Stop".
        Output:  one Decision per candidate, in the same order (D1).
        """
        ...


class Ranker(Protocol):
    """Nyla's shortlist ranking (D2 behaviour "A5")."""

    def rank(self, accepted: list[Candidate]) -> list[Candidate]:
        """Return the accepted candidates sorted best first, with repeated pairs removed."""
        ...


@dataclass
class Tools:
    """A box holding one of each tool, so build_graph() takes one argument instead of five.

    Create it like:
        Tools(retriever=StubRetriever(), generator=StubGenerator(),
              predictor=StubPredictor(), gate=PlaceholderGate(), ranker=PlaceholderRanker())
    and use a tool like:  tools.gate.decide(...)

    To use a teammate's real tool, put it in that slot. Nothing in graph.py changes.
    """

    retriever: Retriever  # anything with a matching retrieve()
    generator: CandidateGenerator  # anything with a matching generate_validate()
    predictor: PropertyPredictor  # anything with a matching predict()
    gate: DecisionGate  # anything with a matching decide()
    ranker: Ranker  # anything with a matching rank()
