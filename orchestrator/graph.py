"""LangGraph workflow for transparent-OPV candidate refinement (D2 behaviour "A2").

WHAT THIS FILE IS
-----------------
The heart of the orchestrator: the FLOWCHART of steps, built with LangGraph.
(New to LangGraph? Read docs/primer.md section 5 first.)

    START -> start -> retrieve -> generate -> validate -> predict -> decide --+
                                     ^                                       |
                                     +------- next round (revise) ----------+
                                                                             |
                                          finish (completed/stopped/failed) <+ -> END

In words, each ROUND (iteration):
  1. generate - ask Jaxon's generator for new candidates. Round 1 invents from
                scratch; later rounds revise last round's "Revise" candidates (the parents).
  2. validate - OUR safety-net chemistry check (RDKit): is each molecule real, and
                is the pair new? (Decision A2: kept until Jaxon's validation exists.)
  3. predict  - ask Arleen's predictor for the six numbers of each valid pair
  4. decide   - ask Arleen's gate for a verdict on each: Accept / Revise / Escalate / Stop.
                Escalate = SET ASIDE for DFT/lab review; the run keeps going (D2, no pause).
  then the routing function picks: another round, or finish.

How a run ENDS (function `_next_move`, rules checked in order):
  failed    - every prediction in a round failed
  completed - the optional target_count was reached, or nothing is left to revise
  stopped   - out of time (policy budget, D2: 30 min of WORKING time), out of
              patience (optional, formula NM-4), or out of rounds while
              candidates still needed revising

Saving and resuming: pass a disk checkpointer (storage.make_checkpointer) and
the state is saved after every step under the run's thread_id. runner.py uses
that to resume a crashed run from its last finished step.

LangGraph vocabulary used below:
  node             - a function (state) -> dict of state updates
  edge             - a fixed arrow "after A, run B"
  conditional edge - a function that reads the state and returns the next node's name
  checkpointer     - saves the state after every node (a video-game auto-save)
"""

# time.time() gives "seconds since 1 Jan 1970". The difference between two
# readings = seconds elapsed. Used for the run's time budget.
import time

# Callable: the type of "something you can call like a function".
# Literal: lists the exact node names a routing function may return.
from typing import Callable, Literal

# InMemorySaver: a checkpointer that keeps its saves in memory (lost when the program exits).
from langgraph.checkpoint.memory import InMemorySaver

# StateGraph builds the flowchart; START/END mark its beginning and end.
from langgraph.graph import END, START, StateGraph

# RDKit: Chem reads SMILES. RDLogger controls RDKit's error printing.
from rdkit import Chem, RDLogger

from .interfaces import ConstraintPacket, Tools
from .near_miss import compute_near_miss, nm4_made_progress, rank_near_misses
from .retry import call_with_retry
from .state import Candidate, ChemistryViolation, OPVState
from .storage import make_serializer

# RDKit prints a scary message every time it can't read a SMILES. Broken molecules
# are EXPECTED here (we record them as ChemistryViolations), so silence those messages.
RDLogger.DisableLog("rdApp.*")


def build_graph(tools: Tools, checkpointer=None, clock: Callable[[], float] = time.time,
                should_cancel: Callable[[str], bool] | None = None,
                on_event: Callable[..., None] | None = None,
                sleep: Callable[[float], None] = time.sleep):
    """Build the flowchart around the given tools and return a runnable graph.

    Parameters:
        tools        - the five tools (fakes, test fakes, or teammates' real ones)
        checkpointer - where to auto-save. None (the default) = InMemorySaver
                       (lost when the program exits). For saving to disk, pass
                       storage.make_checkpointer(...).
        clock        - the function used to read the current time. Normally
                       time.time. Tests pass a fake clock so they can pretend 30
                       minutes have passed without actually waiting. Used only to
                       time each step (see `timed` below).
        should_cancel - optional function: given a run_id, returns True if someone
                       asked that run to stop (the worker passes
                       RunStore.is_cancel_requested). Checked before every step
                       (D2: "cancellation is observed at each checkpoint").
                       None = the run can't be cancelled (demo, most tests).
        on_event     - optional function that receives AUDIT EVENTS from inside the
                       run: on_event(run_id, action=..., result_code=..., phase=...,
                       attempt=..., detail=...). The worker passes
                       RunStore.record_event. Events: every failed/retried tool
                       attempt, and a summary of each finished round. None = not recorded.
        sleep        - the waiting function used between retries (retry.py).
                       Tests pass one that doesn't really wait.

    EVERY TOOL CALL is retried by D2's rules (retry.py, decisions Q1/Q2 of
    2026-11-10): up to 3 attempts for temporary errors, none for permanent ones.

    Returns a compiled graph. Run it with:
        graph.invoke({"run_id": ..., "objective": ..., "config": ..., "policy": ...},
                     {"configurable": {"thread_id": run_id}})

    WHY THE STEPS ARE DEFINED INSIDE THIS FUNCTION: each step needs `tools` and
    `clock`, but LangGraph calls a step with only the state. Defining the steps
    in here lets them "see" those variables (a Python feature called a closure).
    """

    # ======================================================================
    # NODES (the steps). Each one receives the current state and returns a
    # dictionary of ONLY the keys it changes.
    # ======================================================================

    def emit(run_id: str, **event) -> None:
        """Send one audit event to on_event, if one was given."""
        if on_event is not None:
            on_event(run_id, **event)

    def call_tool(state: OPVState, tool: str, call):
        """Call a tool through the retry rules, reporting failed/retried attempts as audit events.

        Only the error's TYPE goes into the audit detail, never its message, which
        might contain SMILES (D2: no molecular text in the audit log).
        """
        def report(tool_name, attempt, result_code, error):
            emit(state["run_id"], action="tool_attempt", result_code=result_code, phase=tool_name,
                 attempt=attempt, detail=type(error).__name__ if error else None)
        return call_with_retry(tool, call, report=report, sleep=sleep)

    def timed(step):
        """Wrap a step so it also reports how long it worked (for the time budget).

        The wrapped step runs as normal; then the seconds it took are added to its
        updates as "active_seconds", which the state ADDS to the running total
        (decision Q3: only working time counts). Example: a step that took 2.5 s
        returns {..its own updates.., "active_seconds": 2.5}.

        `timed` takes a function and returns a new function: a common Python
        pattern called a "wrapper" (or "decorator").
        """
        def run_and_time(state: OPVState) -> dict:
            began = clock()  # time just before the step
            updates = step(state)  # run the real step
            # {**updates, ...} = a copy of the step's updates plus one more key.
            return {**updates, "active_seconds": clock() - began}
        return run_and_time

    def cancellable(step):
        """Wrap a step so it's SKIPPED once the run has been asked to stop.

        Before the step runs, check (1) whether a cancel was already noticed, or
        (2) whether should_cancel says this run must stop. If so, do no work and
        just record {"cancel_seen": True}. Every remaining step is then skipped
        the same way until the routing sends the run to `finish`, which ends it
        as stopped / "cancelled". Steps that already finished keep their results.
        """
        def run_unless_cancelled(state: OPVState) -> dict:
            if state.get("cancel_seen") or (should_cancel and should_cancel(state["run_id"])):
                return {"cancel_seen": True}
            return step(state)
        return run_unless_cancelled

    def start(state: OPVState) -> dict:
        """FIRST STEP, once per run: set every counter to its starting value.

        (active_seconds needs no starting value: added-up numbers start at 0.)
        """
        return {
            "status": "running",
            "iteration": 0,  # no rounds done yet
            "stale_iterations": 0,  # no rounds without progress yet
            "parents": [],  # nothing to revise yet -> round 1 invents from scratch
        }

    def retrieve(state: OPVState) -> dict:
        """Fetch evidence from Kyle's knowledge base (stored for later use; nothing reads it yet)."""
        return {"evidence": call_tool(state, "retriever", lambda: tools.retriever.retrieve(
            state.get("objective", ""), state["config"]))}

    def generate(state: OPVState) -> dict:
        """Start a new round: build the ConstraintPacket and ask the generator for candidates.

        Reads:  config, iteration, parents, run_id
        Writes: batch (this round's candidates), iteration (+1)
        """
        cfg = state["config"]
        iteration = state["iteration"] + 1  # this round's number
        packet = ConstraintPacket(
            run_id=state["run_id"], iteration=iteration,
            target_type=cfg.target_type, fixed_partner_smiles=cfg.fixed_partner_smiles,
            constraints=cfg.constraints,
            parents=state["parents"],  # last round's Revise candidates (empty in round 1)
            candidate_limit=cfg.candidate_limit, seed=cfg.seed,
        )
        result = call_tool(state, "generator", lambda: tools.generator.generate_validate(packet))
        # Keep BOTH lists in the batch: rejected candidates are still logged in
        # history ("log every candidate"). They already carry chemistry_violations.
        return {"batch": result.valid + result.rejected, "iteration": iteration}

    def validate(state: OPVState) -> dict:
        """OUR SAFETY-NET chemistry check (decision A2; remove once Jaxon's validation exists).

        For each candidate the generator didn't already reject:
          1. Can RDKit read the molecule?        No -> ChemistryViolation "invalid_smiles"
          2. Build the donor+acceptor pair with the fixed partner, in canonical form.
          3. Was this exact pair seen before?    Yes -> ChemistryViolation "duplicate_pair"
             (D2 EC-C3-2: don't redo work on a repeated pair)
          4. Otherwise mark it valid.
        Invalid candidates stay in the batch (valid=False) so they're still logged.
        """
        cfg = state["config"]
        # Every pair already checked in earlier rounds. A "set" gives fast "is X in it?" checks.
        # state.get("history", []) = the history, or an empty list if there's none yet.
        seen = {c.pair_key for c in state.get("history", []) if c.valid}
        checked = []  # this round's candidates with the check results attached
        for cand in state["batch"]:
            if cand.chemistry_violations:  # the generator already rejected it; keep its reasons
                checked.append(cand)
                continue
            mol = Chem.MolFromSmiles(cand.generated_smiles)  # None = RDKit couldn't read it
            if mol is None:
                checked.append(cand.model_copy(update={"chemistry_violations": [ChemistryViolation(
                    check_code="invalid_smiles", severity="error",
                    detail="RDKit could not parse the generated SMILES")]}))
                continue
            canonical = Chem.MolToSmiles(mol)  # the molecule's ONE official spelling
            # Put the new molecule on its side of the pair; the partner fills the other side.
            # (fixed_partner_smiles is already canonical: RunConfig's validator did that.)
            if cfg.target_type == "donor":
                donor, acceptor = canonical, cfg.fixed_partner_smiles
            else:
                donor, acceptor = cfg.fixed_partner_smiles, canonical
            cand = cand.model_copy(update={"canonical_smiles": canonical,
                                           "donor_smiles": donor, "acceptor_smiles": acceptor})
            if cand.pair_key in seen:
                checked.append(cand.model_copy(update={"chemistry_violations": [ChemistryViolation(
                    check_code="duplicate_pair", severity="error",
                    detail="this donor>>acceptor pair was already evaluated in this run")]}))
                continue
            seen.add(cand.pair_key)  # also catches a repeat later in this SAME batch
            checked.append(cand.model_copy(update={"valid": True}))
        return {"batch": checked}

    def predict(state: OPVState) -> dict:
        """Ask the predictor for the numbers of every VALID candidate, as one batch.

        Invalid candidates are skipped: D2 says invalid chemistry never reaches the predictor.
        """
        valid = [c for c in state["batch"] if c.valid]
        predictions = call_tool(state, "predictor", lambda: tools.predictor.predict(valid)) if valid else []
        if len(predictions) != len(valid):  # the contract says one per candidate (D1)
            raise RuntimeError(f"predictor returned {len(predictions)} predictions "
                               f"for {len(valid)} candidates")
        # zip pairs up the two lists item by item: (valid[0], predictions[0]), ...
        # A dictionary from candidate_id to its prediction, for easy lookup below.
        by_id = {c.candidate_id: p for c, p in zip(valid, predictions)}
        return {"batch": [c.model_copy(update={"prediction": by_id[c.candidate_id]})
                          if c.candidate_id in by_id else c
                          for c in state["batch"]]}

    def decide(state: OPVState) -> dict:
        """Ask the gate for verdicts, then sort the results into the right piles.

        Reads:  batch, config, policy, iteration, history, stale_iterations
        Writes: batch, history (+append), accepted (+append), escalated (+append),
                failed_predictions (+append), parents (replaced), stale_iterations
        """
        cfg, policy = state["config"], state["policy"]
        # Only SUCCESSFUL predictions go to the gate (D1/D2). Failed ones are logged instead.
        ok = [c for c in state["batch"] if c.prediction is not None and c.prediction.status == "ok"]
        failed = [c for c in state["batch"] if c.prediction is not None and c.prediction.status == "failed"]

        # D2: a missed target becomes "Revise" only if another round is allowed.
        iterations_remain = state["iteration"] < cfg.max_iterations
        decisions = (call_tool(state, "gate", lambda: tools.gate.decide(ok, cfg, policy, iterations_remain))
                     if ok else [])
        # Attach each decision, plus the near-miss numbers (formulas NM-1, NM-2), to its candidate.
        judged = {c.candidate_id: c.model_copy(update={
                      "decision": d, "near_miss": compute_near_miss(c, cfg.constraints)})
                  for c, d in zip(ok, decisions)}
        batch = [judged.get(c.candidate_id, c) for c in state["batch"]]  # judged version if any

        def with_outcome(outcome: str) -> list[Candidate]:
            """This round's candidates whose verdict is `outcome`."""
            return [c for c in batch if c.decision is not None and c.decision.outcome == outcome]

        # Formula NM-4 (near_miss.py): did this round get closer to the goal?
        progress = nm4_made_progress(batch, state.get("history", []))

        # Audit: a one-line summary of the round, COUNTS ONLY (no SMILES, D2).
        counts = {o: len(with_outcome(o)) for o in ("Accept", "Revise", "Escalate", "Stop")}
        emit(state["run_id"], action="iteration_finished", phase="decide",
             detail=(f"round {state['iteration']}: {len(batch)} candidates: "
                     + ", ".join(f"{n} {o}" for o, n in counts.items())
                     + f", {sum(1 for c in batch if not c.valid)} invalid, "
                     + f"{len(failed)} failed predictions"))
        updates = {
            "batch": batch,
            "history": batch,  # APPENDED: log every candidate, whatever happened to it
            "accepted": with_outcome("Accept"),  # APPENDED
            "escalated": with_outcome("Escalate"),  # APPENDED to the set-aside pile
            "failed_predictions": failed,  # APPENDED
            "stale_iterations": 0 if progress else state["stale_iterations"] + 1,
        }
        # Next round revises THESE (D2 A2 step 6). Exception: if NO candidate was even
        # valid this round, keep the old parents so the next round can retry them
        # (D2 EC-C3-1: zero valid candidates -> try again while budget remains).
        if any(c.valid for c in batch):
            updates["parents"] = with_outcome("Revise")
        return updates

    def finish(state: OPVState) -> dict:
        """LAST STEP: set the final status, and rank the results.

        Both rankings are computed whatever the status, so nothing is lost even
        for a stopped or failed run (D1 only SHOWS the shortlist for completed runs).
        """
        _, status, reason = _next_move(state)
        history = state.get("history", [])
        return {
            "status": status,
            "stop_reason": reason,
            "ranked": call_tool(state, "ranker",  # Nyla's shortlist ranking
                                lambda: tools.ranker.rank(state.get("accepted", []))),
            "near_misses": rank_near_misses(history),  # formula NM-3, closest first
        }

    # ======================================================================
    # ROUTING: decides "another round" or "finish". It doesn't change the
    # state; it READS it and returns the next node's name.
    # ======================================================================

    def _next_move(state: OPVState) -> tuple[str, str, str]:
        """Decide what happens after a round. Returns (next_node, final_status, reason).

        final_status and reason only matter when next_node is "finish". Rules, in order:
          0. a cancel request was noticed                          -> stopped ("cancelled")
          1. valid candidates existed but EVERY prediction failed  -> failed
          2. optional target_count reached                         -> completed
          3. WORKING-time budget used up (policy.max_runtime_seconds) -> stopped
          4. optional patience used up (NM-4 rounds w/o progress)  -> stopped
          5. last round reached:
               candidates still needed revising, or none were valid -> stopped
               otherwise                                            -> completed
          6. no valid candidates this round                        -> another round (retry)
          7. nothing left to revise                                -> completed
          8. otherwise                                             -> another round
        Kept in one function so the router and `finish` always agree on the reason.
        """
        cfg, policy = state["config"], state["policy"]
        batch = state.get("batch", [])
        valid = [c for c in batch if c.valid]
        any_ok = any(c.prediction is not None and c.prediction.status == "ok" for c in valid)
        # "Would have been revised": stopped by the gate for missed targets on the last round.
        unrevised = [c for c in batch if c.decision is not None and c.decision.outcome == "Stop"
                     and c.decision.violations]

        if state.get("cancel_seen"):
            return "finish", "stopped", "cancelled"
        if valid and not any_ok:
            return "finish", "failed", f"every prediction failed in iteration {state['iteration']}"
        if cfg.target_count is not None and len(state.get("accepted", [])) >= cfg.target_count:
            return "finish", "completed", f"target count of {cfg.target_count} accepted candidates reached"
        if state.get("active_seconds", 0.0) >= policy.max_runtime_seconds:
            return "finish", "stopped", f"runtime budget ({policy.max_runtime_seconds} s) exhausted"
        if cfg.patience is not None and state["stale_iterations"] >= cfg.patience:
            return "finish", "stopped", f"no near-miss progress for {cfg.patience} iterations"
        if state["iteration"] >= cfg.max_iterations:
            if unrevised or not valid:
                return "finish", "stopped", f"iteration budget ({cfg.max_iterations}) exhausted"
            return "finish", "completed", "no candidates left to revise"
        if not valid:
            return "generate", "", ""
        if not state.get("parents"):
            return "finish", "completed", "no candidates left to revise"
        return "generate", "", ""

    def route_after_decide(state: OPVState) -> Literal["generate", "finish"]:
        """The conditional edge after `decide`: just the node-name part of _next_move."""
        return _next_move(state)[0]

    # ======================================================================
    # WIRING: put the steps into a flowchart and connect them with arrows.
    # ======================================================================
    g = StateGraph(OPVState)  # an empty flowchart whose notebook has the OPVState shape
    # The middle steps can be skipped by a cancel. `start` (which sets up the
    # counters) and `finish` (which records how the run ended) always run.
    steps = [("start", start)] + [(name, cancellable(fn)) for name, fn in [
        ("retrieve", retrieve), ("generate", generate), ("validate", validate),
        ("predict", predict), ("decide", decide)]] + [("finish", finish)]
    for name, fn in steps:
        g.add_node(name, timed(fn))  # register each step (wrapped so it's timed) under a name

    # Fixed arrows: the straight-line part.
    g.add_edge(START, "start")
    g.add_edge("start", "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "validate")
    g.add_edge("validate", "predict")
    g.add_edge("predict", "decide")
    # Conditional arrow: after decide, go wherever route_after_decide says.
    g.add_conditional_edges("decide", route_after_decide)
    g.add_edge("finish", END)

    # compile() checks the flowchart is complete and makes it runnable.
    # `checkpointer or InMemorySaver(...)` = the given checkpointer, or a new in-memory
    # one. The in-memory one also gets our allow-list (storage.make_serializer), so
    # reading a run back never warns about our classes.
    return g.compile(checkpointer=checkpointer or InMemorySaver(serde=make_serializer()))
