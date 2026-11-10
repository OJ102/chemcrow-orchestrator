"""Demo run with the fake tools: python -m orchestrator

WHAT THIS FILE IS
-----------------
A small demo program that runs the whole loop once with the FAKE tools
(stubs.py) and the PLACEHOLDER gate and ranker (gate.py), then prints the
results. Because the file is named __main__.py, running `python -m orchestrator`
(from the repo root) runs it.

    python -m orchestrator                         # run once, kept in memory only
    python -m orchestrator --db runs               # run once AND save it in the folder runs/
    python -m orchestrator --db runs --list        # list the runs saved in runs/
    python -m orchestrator --db runs --resume ID   # continue a saved run that crashed

The web API and the worker (item 5) are two separate programs, each started
with its own command, both pointing at the same folder:

    python -m orchestrator serve  --db runs --port 8100   # the web API (needs ORCHESTRATOR_API_TOKEN)
    python -m orchestrator worker --db runs               # runs the queued runs

Nothing pauses any more: uncertain candidates are SET ASIDE (escalated) and
listed at the end, as agreed on 2026-10-22 and specified in D2.

!!! All molecules and numbers are FAKE (see stubs.py). !!!
"""

# argparse: Python's standard tool for reading command-line options like --db.
import argparse

# os.environ: the environment variables (where the API token is read from).
import os

# sys.argv: the words typed after "python -m orchestrator".
import sys

# uuid: makes random unique IDs like "3f2a9c1e-...", used as the run's ID.
import uuid
from pathlib import Path

# "from ." = from this package's __init__.py (which re-exports these names).
from . import RunConfig, RunConstraints, Tools, build_graph
from .gate import PlaceholderGate, PlaceholderRanker
from .policies import PLACEHOLDER_POLICY
from .runner import resume_run, start_run
from .storage import RunStore, make_checkpointer
from .stubs import StubGenerator, StubPredictor, StubRetriever

# Y6: a famous, real acceptor molecule used in high-efficiency organic solar
# cells, written as SMILES. Two strings in parentheses with nothing between them
# are joined automatically, so this is one long string split over two lines.
# In this demo Y6 is the FIXED PARTNER: we search for donors that pair with it.
Y6 = ("CCCCCCCCCCCc1c(C=C2C(=O)c3cc(F)c(F)cc3C2=C(C#N)C#N)sc2c1sc1c3c4nsnc4c4c5sc6c(CCCCCCCCCCC)"
      "c(C=C7C(=O)c8cc(F)c(F)cc8C7=C(C#N)C#N)sc6c5n(CC(CC)CCCC)c4c3n(CC(CC)CCCC)c21")

# PLACEHOLDER scientific limits (nobody has approved real ones yet, peer review
# PR-06). Defined once in policies.py, which the web API also uses.
DEMO_POLICY = PLACEHOLDER_POLICY


def _fake_tools() -> Tools:
    """The FAKE tools (stubs.py) plus the placeholder gate and ranker (gate.py)."""
    return Tools(retriever=StubRetriever(), generator=StubGenerator(), predictor=StubPredictor(),
                 gate=PlaceholderGate(), ranker=PlaceholderRanker())


def serve_or_work(command: str, argv: list[str]) -> None:
    """The two long-running programs: `serve` (the web API) and `worker`.

    Both read the same --db folder: the API records runs there, the worker runs them.
    """
    parser = argparse.ArgumentParser(prog=f"python -m orchestrator {command}")
    parser.add_argument("--db", default="runs", metavar="FOLDER", help="the runs folder (default: runs)")
    if command == "serve":
        parser.add_argument("--port", type=int, default=8100, help="port to listen on (default: 8100)")
    else:
        parser.add_argument("--poll", type=float, default=2.0, help="seconds between checks for new runs")
    args = parser.parse_args(argv)

    if command == "serve":
        # The shared secret is read from an ENVIRONMENT VARIABLE, not typed on the
        # command line, so it doesn't end up in shell history or process lists.
        token = os.environ.get("ORCHESTRATOR_API_TOKEN")
        if not token:
            parser.error("set the ORCHESTRATOR_API_TOKEN environment variable to a secret first "
                         "(PowerShell: $env:ORCHESTRATOR_API_TOKEN = '...'; bash: export ORCHESTRATOR_API_TOKEN=...)")
        import uvicorn  # the web server program that runs a FastAPI app
        from .api import create_app
        print(f"API docs: http://127.0.0.1:{args.port}/docs  (only reachable from this machine)")
        # host="127.0.0.1": only programs on THIS machine can connect (decision Q4).
        uvicorn.run(create_app(Path(args.db), token), host="127.0.0.1", port=args.port)
    else:
        from .worker import Worker
        store = RunStore(args.db)
        # should_cancel lets a cancel request made through the API stop the run between steps.
        graph = build_graph(_fake_tools(), checkpointer=make_checkpointer(args.db),
                            should_cancel=store.is_cancel_requested)
        try:
            Worker(graph, store).run_forever(poll_every=args.poll)
        except KeyboardInterrupt:  # Ctrl+C
            print("worker stopped")


def main() -> None:
    """Read the command-line options, then run (or resume, or list) and print the results."""
    if len(sys.argv) > 1 and sys.argv[1] in ("serve", "worker"):
        return serve_or_work(sys.argv[1], sys.argv[2:])
    parser = argparse.ArgumentParser(description="Demo of the transparent-OPV orchestrator (FAKE tools).")
    parser.add_argument("--db", metavar="FOLDER",
                        help="save runs in this folder (creates checkpoints.db and runs.db)")
    parser.add_argument("--resume", metavar="RUN_ID", help="continue a saved run (needs --db)")
    parser.add_argument("--list", action="store_true", help="list saved runs (needs --db)")
    args = parser.parse_args()  # e.g. args.db = "runs", args.resume = None, args.list = False
    if (args.resume or args.list) and not args.db:
        parser.error("--resume and --list need --db FOLDER")  # prints the message and exits

    # Saving to disk (--db) or only in memory (no --db)?
    checkpointer = make_checkpointer(args.db) if args.db else None  # None = in memory
    store = RunStore(args.db) if args.db else None
    graph = build_graph(_fake_tools(), checkpointer=checkpointer)

    if args.list:
        for r in store.list():
            print(f"{r.run_id}  {r.status:<9}  iterations={r.iteration}  created={r.created_at}  "
                  f"{r.stop_reason or r.failure or ''}")
        return

    if args.resume:
        run_id = args.resume
        try:
            record = resume_run(graph, store, run_id)
        except (KeyError, ValueError) as error:  # unknown run, or one that already finished
            parser.error(f"cannot resume: {error}")  # a one-line message instead of a Python error dump
    else:
        run_id, record = _start_demo_run(graph, store)

    if record is not None:  # a saved run: show its row from the runs table
        print(f"saved in {args.db}: status {record.status}"
              + (f", failure: {record.failure}" if record.failure else ""))
    # The final state (the whole shared notebook), read back from the checkpoint.
    result = graph.get_state({"configurable": {"thread_id": run_id}}).values
    _print_results(run_id, result)


def _start_demo_run(graph, store):
    """Start the demo run. Returns (run_id, runs-table row, or None if not saving)."""
    # One ID per run, used both as the run's ID and as LangGraph's thread_id (its
    # save-slot name), as agreed with Nyla. A fresh random ID = a clean run.
    run_id = str(uuid.uuid4())
    objective = "Find a transparent donor for Y6"
    config = RunConfig(
        target_type="donor",  # design donors...
        fixed_partner_smiles=Y6,  # ...that pair with Y6
        # Demo targets: at least 50% see-through, 3% efficient, LUE 2.8. Chosen
        # (together with seed=1) because, with the FAKE tools, this run uses every
        # path: revisions over 3 rounds, near misses, invalid molecules, a repeated
        # pair, escalations and stops. They are not real scientific targets.
        constraints=RunConstraints(avt_min_fraction=0.50, pce_min_percent=3.0, lue_min_percent=2.8),
        candidate_limit=8,  # 8 candidates per round, so the printout stays short
        seed=1,
    )
    if store is not None:  # saving: runner records the run in the runs table too
        return run_id, start_run(graph, store, objective=objective, config=config,
                                 policy=DEMO_POLICY, run_id=run_id)
    graph.invoke({"run_id": run_id, "objective": objective, "config": config, "policy": DEMO_POLICY},
                 {"configurable": {"thread_id": run_id}})
    return run_id, None


def _print_results(run_id: str, result: dict) -> None:
    """Print the status line, shortlist, set-aside pile and top near misses."""
    print(f"run {run_id}")
    print(f"status: {result['status']} ({result['stop_reason']}) after {result['iteration']} "
          f"iterations, {len(result['history'])} candidates logged, "
          f"{len(result['failed_predictions'])} failed predictions")

    print("\nSHORTLIST (accepted, best first):")
    for cand in result["ranked"]:
        v = cand.prediction.values
        # :5.2f = at least 5 characters wide, 2 decimals, so the columns line up.
        print(f"  {cand.candidate_id}  AVT {v.avt_fraction:.2f}  LUE {v.lue_percent:5.2f}  "
              f"PCE {v.pce_percent:5.2f}%  parent={cand.parent_candidate_id}  {cand.canonical_smiles}")

    print("\nSET ASIDE for DFT/lab review (escalated):")
    for cand in result["escalated"]:
        print(f"  {cand.candidate_id}  flags={cand.decision.risk_flags}  {cand.canonical_smiles}")

    print("\nNEAR MISSES (closest first, formula NM-3; top 5):")
    for cand in result["near_misses"][:5]:
        nm = cand.near_miss
        # Show only the targets it missed, as "field -fraction short (uncertainties short)".
        missed = ", ".join(f"{f} -{s:.0%} ({nm.sigmas_short[f]:.1f} sd)"
                           for f, s in nm.shortfall.items() if s > 0)
        print(f"  {cand.candidate_id}  {cand.decision.outcome:<6}  borderline={nm.borderline}  {missed}")


# True only when this file is run directly (python -m orchestrator), not when imported.
if __name__ == "__main__":
    main()
