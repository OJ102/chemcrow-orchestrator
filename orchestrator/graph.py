"""LangGraph workflow for transparent-OPV candidate refinement.

    START -> retrieve -> generate -> validate -> predict -> gate --+
                            ^                                     |
                            +------------- revise ----------------+
                                                                  |
                         escalate (human review, interrupt) <-----+
                                                                  |
                                    rank (done) / stop  <---------+

The gate is deterministic (gate.py). The loop ends when enough candidates are
accepted, or on the iteration / no-improvement budget.

LangGraph vocabulary used below:
  node              - a function (state) -> dict of state updates
  edge              - fixed "after A, run B"
  conditional edge  - a function that inspects state and returns the next node's name
  checkpointer      - saves state after every node so a run can pause and resume
  interrupt()       - pauses the run and hands a payload to the caller; the caller
                      resumes with graph.invoke(Command(resume=...), config)
"""

from typing import Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from rdkit import Chem, RDLogger

from .gate import evaluate, pareto_rank, summarize_feedback
from .interfaces import Tools
from .state import Candidate, OPVState

RDLogger.DisableLog("rdApp.*")  # invalid SMILES are expected; we record them as violations


def build_graph(tools: Tools, checkpointer=None):
    """Build and compile the refinement graph around the given tools.

    The nodes are defined as inner functions so they can use `tools` without
    it being stored in the state. `checkpointer` defaults to InMemorySaver
    (state kept in RAM, lost when the process exits); a checkpointer is
    required for interrupt()/resume to work.
    """

    def retrieve(state: OPVState) -> dict:
        """First node: fetch evidence and initialise every loop counter."""
        return {
            "evidence": tools.retriever.retrieve(state["objective"], state["constraints"]),
            "iteration": 0,
            "best_score": 0.0,
            "stale_iterations": 0,
            "feedback": [],
            "escalated": [],
            "status": "running",
        }

    def generate(state: OPVState) -> dict:
        """Ask the generator for a new batch, conditioned on last round's feedback."""
        c = state["constraints"]
        iteration = state["iteration"] + 1  # this round's number
        smiles = tools.generator.generate(c, state["feedback"], c.batch_size)  # raw SMILES strings
        return {
            "batch": [Candidate(smiles=s, iteration=iteration) for s in smiles],
            "iteration": iteration,
        }

    def validate(state: OPVState) -> dict:
        """Parse each SMILES with RDKit, canonicalise it, and flag invalid/duplicates.

        Invalid and duplicate candidates stay in the batch (valid=False, with a
        violation) so they are still recorded in history and counted in feedback.
        """
        seen = {cand.key for cand in state.get("history", [])}  # keys already scored this run
        batch = []  # this batch with validation results attached
        for cand in state["batch"]:
            mol = Chem.MolFromSmiles(cand.smiles)  # None if RDKit can't parse it
            if mol is None:
                batch.append(cand.model_copy(update={
                    "violations": ["invalid_smiles: RDKit could not parse"]}))
                continue
            canonical = Chem.MolToSmiles(mol)  # RDKit's standard spelling of the molecule
            if canonical in seen:
                batch.append(cand.model_copy(update={
                    "canonical_smiles": canonical,
                    "violations": ["duplicate: already evaluated in this run"]}))
                continue
            seen.add(canonical)  # also catches duplicates within the same batch
            batch.append(cand.model_copy(update={"canonical_smiles": canonical, "valid": True}))
        return {"batch": batch}

    def predict(state: OPVState) -> dict:
        """Run the property predictor on valid candidates; invalid ones pass through unchanged."""
        c = state["constraints"]
        return {"batch": [
            cand.model_copy(update={"prediction": tools.predictor.predict(
                cand.canonical_smiles, c.fixed_partner_smiles, c.target_type)})
            if cand.valid else cand
            for cand in state["batch"]
        ]}

    def gate(state: OPVState) -> dict:
        """Apply the deterministic decision gate and update progress counters."""
        c = state["constraints"]
        decided = []  # the batch with decision + violations filled in
        for cand in state["batch"]:
            decision, violations = evaluate(cand, c)
            decided.append(cand.model_copy(update={"decision": decision, "violations": violations}))

        # Best LUE among every predicted candidate in this batch (including ones
        # that failed thresholds), and whether it beat the run's best so far.
        batch_best = max((d.prediction.lue for d in decided if d.prediction), default=0.0)
        improved = batch_best > state["best_score"]
        return {
            "batch": decided,
            "history": decided,  # appended (reducer), not replaced
            "accepted": [d for d in decided if d.decision == "accept"],  # appended (reducer)
            "escalated": [d for d in decided if d.decision == "escalate"],
            "feedback": summarize_feedback([d for d in decided if d.decision in ("revise", "reject")]),
            "best_score": max(batch_best, state["best_score"]),
            "stale_iterations": 0 if improved else state["stale_iterations"] + 1,
        }

    def escalate(state: OPVState) -> dict:
        """Pause for human review of confident-enough-but-uncertain candidates."""
        # Pauses the run until a human resumes it with
        # Command(resume={"approve": [<canonical smiles>, ...]}).
        # When resumed, interrupt() returns that resume value as `answer`.
        answer = interrupt({
            "question": "These candidates pass every threshold but the predictor is "
                        "uncertain. Approve which ones (e.g. after DFT review)?",
            "candidates": [cand.model_dump(mode="json") for cand in state["escalated"]],
        })
        approved = set(answer.get("approve", []))  # keys the human approved
        # Approved ones become accepted; unapproved ones are dropped (they remain in history).
        return {
            "accepted": [cand.model_copy(update={"decision": "accept"})
                         for cand in state["escalated"] if cand.key in approved],
            "escalated": [],
        }

    def rank(state: OPVState) -> dict:
        """Success ending: enough candidates accepted."""
        return {"ranked": pareto_rank(state.get("accepted", [])), "status": "done",
                "stop_reason": "target count of accepted candidates reached"}

    def stop(state: OPVState) -> dict:
        """Budget ending: out of iterations or patience. Still ranks what was accepted."""
        c = state["constraints"]
        reason = (f"no improvement for {c.patience} iterations"
                  if state["stale_iterations"] >= c.patience
                  else f"iteration budget ({c.max_iterations}) exhausted")
        return {"ranked": pareto_rank(state.get("accepted", [])), "status": "stopped",
                "stop_reason": reason}

    def next_step(state: OPVState) -> Literal["rank", "stop", "generate"]:
        """Routing: finish, stop, or loop again. The Literal lists the possible next nodes."""
        c = state["constraints"]
        if len(state.get("accepted", [])) >= c.target_count:
            return "rank"
        if state["iteration"] >= c.max_iterations or state["stale_iterations"] >= c.patience:
            return "stop"
        return "generate"  # revise: feedback is already in state for the generator

    def after_gate(state: OPVState) -> Literal["escalate", "rank", "stop", "generate"]:
        """Routing after the gate: human review first if anything was escalated."""
        return "escalate" if state["escalated"] else next_step(state)

    # --- Wire the graph -----------------------------------------------------
    g = StateGraph(OPVState)
    for name, fn in [("retrieve", retrieve), ("generate", generate), ("validate", validate),
                     ("predict", predict), ("gate", gate), ("escalate", escalate),
                     ("rank", rank), ("stop", stop)]:
        g.add_node(name, fn)

    # Fixed edges: the straight-line part of each iteration.
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "validate")
    g.add_edge("validate", "predict")
    g.add_edge("predict", "gate")
    # Conditional edges: the routing function's return value names the next node.
    g.add_conditional_edges("gate", after_gate)
    g.add_conditional_edges("escalate", next_step)
    g.add_edge("rank", END)
    g.add_edge("stop", END)

    return g.compile(checkpointer=checkpointer or InMemorySaver())
