"""Tests for the gate and the graph. Run from the repo root:

    python -m pytest orchestrator/tests -v

The graph tests swap in tiny fake tools (ListGenerator, TablePredictor) so each
test controls exactly which SMILES appear and what they score.
"""

from langgraph.types import Command

from orchestrator import Candidate, Constraints, Prediction, Tools, build_graph
from orchestrator.gate import evaluate, pareto_rank
from orchestrator.stubs import StubRetriever

# Reusable predictions. With default Constraints:
GOOD = Prediction(pce=8.0, avt=0.6, homo=-5.4, lumo=-3.6, opv_prob=0.9, uncertainty=0.05)  # -> accept
OPAQUE = GOOD.model_copy(update={"avt": 0.1})  # fails min_avt -> revise
UNSURE = GOOD.model_copy(update={"uncertainty": 0.5})  # passes but too uncertain -> escalate


class ListGenerator:
    """Yields one predetermined batch per call."""

    def __init__(self, batches):
        self.batches = iter(batches)  # each next() returns the following batch
        self.feedback_seen = []  # records the feedback passed on each call, for assertions

    def generate(self, constraints, feedback, n):
        self.feedback_seen.append(feedback)
        return next(self.batches)


class TablePredictor:
    """Looks up a prediction by SMILES; anything not in the table gets `default`."""

    def __init__(self, table, default=GOOD):
        self.table, self.default = table, default

    def predict(self, smiles, partner_smiles, target_type):
        return self.table.get(smiles, self.default)


def run(batches, table, default=GOOD, **constraint_kw):
    """Build a graph with the fake tools and run it once.

    Returns (graph, config, generator, result) so tests can inspect the
    feedback the generator saw, or resume the same run after an interrupt.
    """
    gen = ListGenerator(batches)
    graph = build_graph(Tools(StubRetriever(), gen, TablePredictor(table, default)))
    config = {"configurable": {"thread_id": "t"}}
    constraints = Constraints(target_type="donor", fixed_partner_smiles="c1ccsc1", **constraint_kw)
    result = graph.invoke({"objective": "test", "constraints": constraints}, config)
    return graph, config, gen, result


def test_gate_decisions():
    """Each of the four verdicts comes from the right kind of input."""
    c = Constraints(target_type="donor", fixed_partner_smiles="C")
    assert evaluate(Candidate(smiles="x", iteration=1), c)[0] == "reject"
    base = Candidate(smiles="C", iteration=1, valid=True)
    assert evaluate(base.model_copy(update={"prediction": GOOD}), c) == ("accept", [])
    decision, violations = evaluate(base.model_copy(update={"prediction": OPAQUE}), c)
    assert decision == "revise" and violations[0].startswith("low_avt:")
    assert evaluate(base.model_copy(update={"prediction": UNSURE}), c)[0] == "escalate"


def test_pareto_rank_puts_nondominated_first():
    """(3, 0.3) is beaten on both axes by (9, 0.4), so it must rank last."""
    # mk builds a candidate with the given PCE and AVT.
    mk = lambda pce, avt: Candidate(smiles=f"{pce}{avt}", iteration=1,
                                     prediction=GOOD.model_copy(update={"pce": pce, "avt": avt}))
    dominated, a, b = mk(3, 0.3), mk(9, 0.4), mk(4, 0.8)
    ranked = pareto_rank([dominated, a, b])
    assert ranked[-1] is dominated and set(map(id, ranked[:2])) == {id(a), id(b)}


def test_accepts_and_finishes():
    """Three good molecules in one batch meet target_count -> status "done"."""
    *_, result = run([["CCO", "CCN", "CCC"]], {}, target_count=3)
    assert result["status"] == "done"
    assert len(result["ranked"]) == 3


def test_revise_loop_passes_feedback_then_stops_on_budget():
    """Failures become feedback for the next round; the loop stops at max_iterations."""
    batches = [["c1ccccc1"], ["Cc1ccccc1"], ["CCc1ccccc1"]]
    _, _, gen, result = run(batches, {}, default=OPAQUE, max_iterations=3, patience=10)
    assert result["status"] == "stopped" and "iteration budget" in result["stop_reason"]
    assert gen.feedback_seen[0] == []
    assert gen.feedback_seen[1] == ["low_avt: 1/1 candidates"]


def test_invalid_and_duplicate_are_rejected():
    """OCC is ethanol, same as CCO, so it's a duplicate; garbage SMILES is invalid."""
    *_, result = run([["CCO", "OCC", "not_a_smiles"]], {}, target_count=5, max_iterations=1)
    decisions = [c.decision for c in result["history"]]
    assert decisions == ["accept", "reject", "reject"]
    assert result["history"][1].violations[0].startswith("duplicate:")


def test_escalation_interrupts_and_resumes():
    """An uncertain candidate pauses the run; approving it on resume accepts it."""
    graph, config, _, result = run([["CCO", "CCN"]], {"CCN": UNSURE}, target_count=2)
    assert "__interrupt__" in result
    pending = result["__interrupt__"][0].value["candidates"]
    assert [c["canonical_smiles"] for c in pending] == ["CCN"]

    result = graph.invoke(Command(resume={"approve": ["CCN"]}), config)
    assert result["status"] == "done"
    assert {c.canonical_smiles for c in result["ranked"]} == {"CCO", "CCN"}
