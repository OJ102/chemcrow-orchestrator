"""Demo run with stub tools: python -m orchestrator [--auto-approve]

`python -m orchestrator` runs this file because it is named __main__.py.
"""

import argparse
import uuid

from langgraph.types import Command

from . import Constraints, Tools, build_graph
from .stubs import StubGenerator, StubPredictor, StubRetriever

# SMILES of the Y6 non-fullerene acceptor: the fixed partner for this demo run
# (we design donors for it).
Y6 = ("CCCCCCCCCCCc1c(C=C2C(=O)c3cc(F)c(F)cc3C2=C(C#N)C#N)sc2c1sc1c3c4nsnc4c4c5sc6c(CCCCCCCCCCC)"
      "c(C=C7C(=O)c8cc(F)c(F)cc8C7=C(C#N)C#N)sc6c5n(CC(CC)CCCC)c4c3n(CC(CC)CCCC)c21")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto-approve", action="store_true",
                        help="approve every escalated candidate without prompting")
    args = parser.parse_args()

    graph = build_graph(Tools(StubRetriever(), StubGenerator(seed=0), StubPredictor()))
    # thread_id names this run's saved state in the checkpointer. A fresh UUID
    # guarantees a clean run (reusing one would append to its old history).
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    # Initial state: only these two keys; the retrieve node fills in the rest.
    inputs = {
        "objective": "Find a transparent donor for Y6 with AVT >= 0.4 and PCE >= 2%",
        "constraints": Constraints(target_type="donor", fixed_partner_smiles=Y6),
    }

    result = graph.invoke(inputs, config)
    # If the graph paused in the escalate node, the result contains
    # "__interrupt__". Ask the human, then resume; repeat until it finishes.
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value  # the dict passed to interrupt()
        print(payload["question"])
        approve = []  # canonical SMILES the user says yes to
        for cand in payload["candidates"]:
            smi = cand["canonical_smiles"]
            if args.auto_approve or input(f"  approve {smi}? [y/N] ").lower() == "y":
                approve.append(smi)
        result = graph.invoke(Command(resume={"approve": approve}), config)

    print(f"\nstatus: {result['status']} ({result['stop_reason']}) after "
          f"{result['iteration']} iterations, {len(result['history'])} candidates scored")
    for cand in result["ranked"]:
        p = cand.prediction
        print(f"  PCE {p.pce:5.2f}%  AVT {p.avt:.2f}  LUE {p.lue:5.2f}  {cand.canonical_smiles}")


if __name__ == "__main__":
    main()
