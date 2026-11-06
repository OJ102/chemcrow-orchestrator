"""Shared test helpers: tiny FAKE tools that do exactly what a test tells them.

WHY: the random fake tools in stubs.py are fine for a demo, but a test needs to
know EXACTLY which molecules appear and what numbers they get, so it can check
the graph did the right thing. These fakes are fully controlled by the test.

Used by test_gate.py, test_near_miss.py and test_graph.py. (This file's name
doesn't start with "test_", so pytest doesn't treat it as a test itself.)

MOLECULES USED IN THE TESTS (SMILES; see docs/primer.md section 2):
    "CCO" ethanol, "OCC" ethanol written backwards (a DUPLICATE of "CCO"),
    "CCN" ethylamine, "CCC" propane, "c1ccccc1" benzene, "Cc1ccccc1" toluene,
    "CCc1ccccc1" ethylbenzene, "c1ccsc1" thiophene (the dummy fixed partner).
These are simple molecules, NOT solar-cell materials. The tests only need valid
SMILES that RDKit can read.
"""

import time

from orchestrator import (Candidate, GenerationResult, Policy, Prediction, RunConfig,
                          RunConstraints, Tools, Uncertainty, Values, build_graph)
from orchestrator.gate import PlaceholderGate, PlaceholderRanker
from orchestrator.stubs import StubRetriever  # the retriever's output doesn't matter in tests

# The dummy fixed partner for every test run: thiophene.
PARTNER = "c1ccsc1"

# The minimum targets used by most tests: AVT >= 0.4, PCE >= 2%, LUE >= 0.8.
CONSTRAINTS = RunConstraints(avt_min_fraction=0.4, pce_min_percent=2.0, lue_min_percent=0.8)

# Test policy: OPV-likeness >= 0.5, and these maximum uncertainties.
POLICY = Policy(policy_version="test-policy", opv_min=0.5,
                uncertainty_max_by_target={"avt_fraction": 0.05, "pce_percent": 1.0,
                                           "lue_percent": 0.5})

# The rules most tests use: design donors against thiophene.
CONFIG = RunConfig(target_type="donor", fixed_partner_smiles=PARTNER, constraints=CONSTRAINTS)


def pred(avt=0.6, pce=8.0, opv=0.9, s_avt=0.01, s_pce=0.2, s_lue=0.1, in_domain=True) -> Prediction:
    """Build a successful Prediction. Every argument has a default, so a test only
    names what it wants to change, e.g. pred(avt=0.1) = "like GOOD, but nearly black".

    s_avt, s_pce, s_lue are the uncertainties (s for "sigma", the usual symbol for
    standard deviation). LUE is always computed as AVT x PCE, as D1 defines it.
    """
    return Prediction(
        status="ok",
        values=Values(opv_likeness=opv, donor_homo_eV=-5.4, acceptor_lumo_eV=-3.9,
                      pce_percent=pce, avt_fraction=avt, lue_percent=avt * pce),
        uncertainty=Uncertainty(opv_likeness=0.05, donor_homo_eV=0.1, acceptor_lumo_eV=0.1,
                                pce_percent=s_pce, avt_fraction=s_avt, lue_percent=s_lue),
        in_domain=in_domain,
    )


# Ready-made predictions. Compare with CONSTRAINTS and POLICY above:
GOOD = pred()  # passes everything, confident -> Accept
OPAQUE = pred(avt=0.1)  # AVT 0.1 < 0.4 (LUE = 0.1 x 8 = 0.8, exactly the minimum -> passes) -> Revise
UNSURE = pred(s_avt=0.2)  # AVT uncertainty 0.2 > limit 0.05 -> Escalate
FAILED = Prediction(status="failed", error="test: simulated failure")  # the model "crashed"


def candidate(prediction: Prediction, candidate_id: str = "it1-c01", iteration: int = 1,
              smiles: str = "CCO") -> Candidate:
    """A candidate that has already been validated and predicted, for testing the gate alone."""
    return Candidate(candidate_id=candidate_id, iteration=iteration, generated_smiles=smiles,
                     canonical_smiles=smiles, donor_smiles=smiles, acceptor_smiles=PARTNER,
                     valid=True, prediction=prediction)


class ListGenerator:
    """A fake generator that returns batches WE wrote in advance, one per round.

    Example: ListGenerator([["CCO"], ["CCN"]]) returns a candidate for "CCO" in
    round 1 and one for "CCN" in round 2. In rounds with parents, each new
    candidate is linked to a parent (lineage), like the real generator should do.
    It also remembers every ConstraintPacket it received, so a test can check
    what the orchestrator sent it.
    """

    def __init__(self, batches):
        self.batches = iter(batches)  # next(self.batches) gives the next prepared batch
        self.packets = []  # every packet received, in order

    def generate_validate(self, packet):
        self.packets.append(packet)
        # If a test runs more rounds than it prepared, next() raises StopIteration
        # and the test fails, which is how we'd notice an unexpected extra round.
        smiles_list = next(self.batches)
        cands = [Candidate(
            candidate_id=f"it{packet.iteration}-c{k + 1:02d}",
            iteration=packet.iteration,
            parent_candidate_id=(packet.parents[k % len(packet.parents)].candidate_id
                                 if packet.parents else None),
            generated_smiles=s,
        ) for k, s in enumerate(smiles_list)]  # enumerate gives (0, first), (1, second), ...
        return GenerationResult(valid=cands, rejected=[])


class TablePredictor:
    """A fake predictor that looks up each molecule's answer in a dictionary.

    Example: TablePredictor({"CCN": UNSURE}) answers UNSURE for "CCN" and
    `default` (GOOD unless told otherwise) for every other molecule. It looks up
    the canonical spelling of the generated molecule.
    """

    def __init__(self, table, default=GOOD):
        self.table, self.default = table, default

    def predict(self, candidates):
        # dict.get(key, fallback) = the value for key, or the fallback if it's missing.
        return [self.table.get(c.canonical_smiles, self.default) for c in candidates]


class FakeClock:
    """A pretend clock that tests control, so they never have to really wait.

    Every time it's read, it moves forward by `step` seconds. Each graph step
    reads the clock before and after itself (graph.py's `timed`), so with
    step=1000 every graph step appears to take 1,000 seconds of WORK.
    A test can also jump time forward without any work happening:
        clock.now += 1_000_000   # e.g. "the machine was off overnight"
    """

    def __init__(self, step: float = 0.0):
        self.now = 0.0  # the current pretend time, in seconds
        self.step = step  # how far each reading moves the clock

    def __call__(self) -> float:
        """Called like a function: clock(). __call__ makes the object callable."""
        self.now += self.step
        return self.now


def run(batches, table=None, default=GOOD, clock=None, **config_changes):
    """Build a graph with the fakes plus the placeholder gate and ranker, and run it once.

    Parameters:
        batches        - the molecules ListGenerator hands out, one list per round
        table          - molecules that get special predictions, e.g. {"CCN": UNSURE}
        default        - the prediction for every other molecule
        clock          - a fake clock for time-budget tests (None = the real clock)
        **config_changes - RunConfig fields to change, e.g. max_iterations=1.
                           (** gathers extra name=value arguments into a dictionary.)

    Returns (generator, result) so tests can inspect both the packets the
    generator received and the final state.
    """
    gen = ListGenerator(batches)
    tools = Tools(retriever=StubRetriever(), generator=gen,
                  predictor=TablePredictor(table or {}, default),
                  gate=PlaceholderGate(), ranker=PlaceholderRanker())
    graph = build_graph(tools, clock=clock or time.time)
    # model_copy(update=...) = CONFIG with the requested fields changed.
    config = CONFIG.model_copy(update=config_changes)
    result = graph.invoke(
        {"run_id": "test-run", "objective": "test", "config": config, "policy": POLICY},
        {"configurable": {"thread_id": "test-run"}},
    )
    return gen, result
