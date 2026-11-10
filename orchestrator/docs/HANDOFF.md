# Session Handoff: Orchestrator Project Context

**Written 2026-11-10** at the end of a long working session, so a new session
(human or Claude) can continue without the old chat. Read this first, then
[progress.md](progress.md) for the full dated log.

> **To start a new Claude session:** say *"Read `orchestrator/docs/HANDOFF.md`
> and continue from 'Next steps'."*

---

## 1. Who and what

- **Shashank Sathiyanarayanan** is **CS Student 2, the LangGraph Orchestration
  Lead**, on the OMID USA–sponsored capstone *AI-Assisted Discovery and Synthesis
  of Transparent Organic Photovoltaic (OPV) Cells* (University of Cincinnati,
  graduating 2027). Advisor/sponsor: **Dr. Mohsen Rezayat**.
- **Goal:** find donor–acceptor molecule pairs for see-through solar cells that
  maximise AVT (transparency) and LUE (= AVT × PCE) while protecting PCE
  (efficiency), and hand the finalists to DFT/lab review.
- **Shashank's components (per team design doc D2):** C2 the orchestrator
  (including the run web API), G1 authentication, G3 configuration/version
  pinning, G4 logging and audit.

### The team
| # | Name | Role | Connects via |
|---|---|---|---|
| S1 | Kyle Russell | Data & knowledge pipeline (OPV2D, spectra, RAG corpus) | `Retriever` slot |
| S2 | **Shashank** | LangGraph orchestration | owns `orchestrator/` |
| S3 | Jaxon Polo | Candidate generation & validation (MatGPT, RDKit) | `CandidateGenerator` slot |
| S4 | Arleen Monteiro | Chemprop prediction **and the Decision Gate** | `PropertyPredictor` + `DecisionGate` slots |
| S5 | Nyla Spencer | Dashboard, reports, **shortlist ranking**, feedback | `Ranker` slot; calls the web API |

Other people: **Om Jadhav** (sponsor contact; CycleChemist/MatGPT; owns the
GitHub repo `OJ102/chemcrow-orchestrator`), **Dr. Hendsbee** (Brilliant
Matters; expert for borderline candidates).

---

## 2. How Claude should work with Shashank (important)

- **Shashank is a beginner**: no chemistry background, new to LangGraph. Explain
  every term and Python construct in plain language.
- **Teach while building:** before writing code, explain the plan **file by
  file**, ask design questions with a recommendation for each, wait for
  answers, then build. Check in after the first file of a step.
- **Comment everything:** every variable, function and class says what it is
  and why it's needed, written for a beginner. Formulas get IDs (e.g. NM-1)
  that can be cited.
- **Keep the docs updated** with every change: `orchestrator/README.md`,
  `docs/primer.md`, `docs/code-guide.md`, `docs/spec-alignment.md`,
  `docs/progress.md` (dated log). **Don't create a separate design-questions
  doc**: questions go in chat, decisions go in progress.md.
- **Ask before installing packages.** Commit only when asked, on the branch
  `shashank/langgraph-orchestration`, with the attribution line.
- **Never type passwords.** Shashank does any password entry (e.g. cluster login).
- Use they/them for teammates.

---

## 3. The repository

`C:\Users\abyss\Documents\Capstone\chemcrow-orchestrator` (a fork of ChemCrow).

- **Two virtual environments, never mix them:**
  - `.venv-chemcrow`: the old ChemCrow code (LangChain 0.0.275)
  - `.venv-langgraph`: the orchestrator (Python 3.10, LangGraph 1.2,
    langgraph-checkpoint-sqlite, FastAPI, uvicorn, RDKit, pytest, httpx).
    Run things with `.\.venv-langgraph\Scripts\python.exe ...`. VS Code tends to
    auto-activate the chemcrow venv: check with `python -c "import sys; print(sys.prefix)"`.
- **Git:** branch `shashank/langgraph-orchestration`. Commits on top of `main`:
  `1552297` baseline → `1d5a981` spec reshape + saving to disk → `b9b3532` web
  API + worker → `b315559` retries + audit log → `6276a65` handoff + SynSpace
  results → run guide. Shashank now **has push access** to
  `OJ102/chemcrow-orchestrator` (2026-11-10).
- **Step-by-step run guide** (tests, demo, API, SynSpace, reading the output):
  [running.md](running.md). The SynSpace experiment lives in
  `experiments/check_synspace.py` (venv `.venv-synspace`).

### `orchestrator/` package
```
state.py       D1 data shapes: RunConfig, Policy, Prediction, Decision, Candidate, OPVState
interfaces.py  the 5 tool slots + ConstraintPacket + TemporaryToolError
near_miss.py   advisor's near-miss formulas NM-1..NM-4
gate.py        PLACEHOLDER gate (Arleen's) and ranker (Nyla's), D2 rules
stubs.py       FAKE retriever/generator/predictor (all numbers are fake)
graph.py       the LangGraph loop: start→retrieve→generate→validate→predict→decide→(generate|finish)
storage.py     SQLite: checkpoints.db + runs.db (runs, receipts, workers, audit_events)
runner.py      start_run / run_claimed / resume_run / saved_state; failure codes
policies.py    policy registry (only "placeholder-0")
worker.py      the worker program: claims queued runs, heartbeats, recovers crashed runs
api.py         FastAPI web API (D1/D2 endpoints + Nyla's list/report + events + health)
retry.py       D2 retry rules for every tool call
logs.py        rotating operational logs (runs/logs/)
__main__.py    demo + `serve` and `worker` commands
tests/         82 tests (fakes.py + test_gate/near_miss/graph/storage/worker/api/retry/audit)
docs/          running (how to run + read output), primer, code-guide, spec-alignment, chemcrow-tools, progress, HANDOFF (this file)
```

### How to run
```powershell
.\.venv-langgraph\Scripts\python.exe -m pytest orchestrator/tests -v     # expect 82 passed
.\.venv-langgraph\Scripts\python.exe -m orchestrator                     # demo (fake tools)
$env:ORCHESTRATOR_API_TOKEN = "secret"; .\.venv-langgraph\Scripts\python.exe -m orchestrator serve --db runs --port 8100
.\.venv-langgraph\Scripts\python.exe -m orchestrator worker --db runs    # second terminal
# then open http://127.0.0.1:8100/docs, click Authorize, paste the token
```

---

## 4. Source documents (outside the repo, in `C:\Users\abyss\Downloads`)

- **D0** `Design Document.docx`, **D1** `D1_Detailed_Design.docx` (= "Detailed
  System Design - Part 1": data model + `/api/v1` contract), **D2**
  `D2_Detailed_Design.docx` (orchestrator, gate, outputs, G1–G4, peer review
  PR-01..07), `Assignment_4_Consolidated_DRAFT.pdf` (all of them).
- Sponsor PDF `OMID_Sponsored_CAPSTONE_Transparent_OPV_2027 (1).pdf` (has the
  student role table and "Phase 1 = Chemprop only").
- Kyle's data: `Active_Database.csv` (OPV2D, 39,998 device rows) and
  `classified_opv_database.csv`. Repo: github.com/sunyrain/OPV2D.
- **Gotcha:** the sandboxed shell can't see Downloads; the Read tool can. To
  extract .docx/.pdf text, run Python/pdftotext with the sandbox disabled
  (read-only) and write the output to the scratchpad.

---

## 5. Decisions already made (summary; details and dates in progress.md)

**Architecture**
- The orchestrator follows D1/D2. Deviations are documented in spec-alignment.md.
- **Fixed-partner approach:** each run designs one side (donor or acceptor)
  against a fixed partner molecule.
- **Escalate = set aside** for DFT/lab review; the run never pauses.
- **Gate and ranker are tool slots** (Arleen's and Nyla's); ours are placeholders.
- **Our RDKit `validate` step stays as a safety net** until Jaxon's validation exists.
- `target_count` and `patience` are optional extras, **off by default**.
- **MCP dropped** (advisor, 2026-11-06): "MCP may not buy us much"; focus on
  orchestration, the best tools, and new polymer recommendations.

**Advisor's near-miss guidance (2026-10-23)**
- Keep near misses. Rank them by uncertainty-aware normalized violations plus
  Pareto (no weighted scores). Escalate borderline ones (within 1 uncertainty).
  Never retrain on predictions. Send structured repair instructions, with
  machine-readable gate output.
- Implemented as formulas NM-1..NM-4 in `near_miss.py`; the borderline cut-off
  is `BORDERLINE_SIGMAS = 1.0`.

**Items 4–6 (all done)**
- Two SQLite files (checkpoints + runs); the time budget counts **working
  time only**; leases/fencing deferred (PR-03).
- A **separate worker program** with heartbeats (20 s) and stale takeover (60 s);
  FastAPI on 127.0.0.1 with a shared bearer token; Idempotency-Key per D1.
- Cancel → `stopped` with reason `cancelled`.
- `POST /runs` **requires** `target_type` and `fixed_partner_smiles` (a breaking
  change versus D1; Shashank is telling the team).
- **Retries:** 3 attempts (1 s, 2 s + jitter) for temporary errors only.
- **Failure codes:** DEPENDENCY_UNAVAILABLE, TOOL_ERROR, ALL_PREDICTIONS_FAILED,
  UNKNOWN_POLICY, INTERNAL_ERROR.
- **Audit log:** written in the same transaction as each change; no SMILES.
- **Rotating log files.** Deadlines stored but not enforced yet.

---

## 6. Open questions / waiting on others

- **Jaxon:** how MatGPT uses parents and structured repair instructions;
  partner conditioning; where it runs (Python 3.9); diversity/similarity measure.
- **Arleen:** confirmed every output, AVT included, has an ensemble uncertainty
  and an in-domain flag. Still open: real AVT training data (OPV2D has **no AVT
  column**), and how to call her model and gate.
- **Team:** (1) add `target_type` and `fixed_partner_smiles` to D1's `POST /runs`;
  (2) should a run that hits its iteration limit with accepted molecules count
  as `completed` (D1: only completed runs have a shortlist)?; (3) a
  stereochemistry rule for SMILES matching (the data's Y6 has `/` `\`, ours
  doesn't); (4) peer review PR-06: nobody has approved the gate thresholds.
- **Kyle's data review:** de-duplicate **pairs**, not molecules (27,361 valid
  pair rows, 5,797 unique pairs); his "invalid" counts were all empty cells.

---

## 7. Next steps (in Shashank's order)

1. ~~SynSpace / ModifyMol testing~~ **Done 2026-11-10** (in a throwaway
   `.venv-synspace`). Result: zero variants for Y6, ITIC, IT-4F, L8-BO and D18;
   it only edits side chains at ester, amide, imide or ether handles, sometimes
   breaks the core ring, and its REOS filter rejects normal OPV alkyl chains.
   **Not suitable** as the near-miss repair tool. Details are in progress.md.
   To do: tell Jaxon; tell Kyle that Y6 and L8-BO are labelled `donor` in
   `classified_opv_database.csv`.
2. **Re-implement useful ChemCrow screens** (next) with structured results: controlled-
   chemical similarity (a hard stop), molecular similarity (for the bounded,
   diverse near-miss archive), patent check, and AiZynthFinder retrosynthesis
   for finalists. See [chemcrow-tools.md](chemcrow-tools.md). Coordinate with
   Jaxon (validation is C3) and Nyla (finalists are C5).
3. **Local LLM on the GPU cluster** for the ChemCrow tool-calling eval:
   - Cluster `cs-gpu-00.digitfut.uc.edu`: 4× A100 80 GB, no sudo, shared
     with another capstone team (GPUs 0/1 theirs, 2/4 ours). The root disk is
     97% full, so use `/home2/sathiysk`.
   - Model: the 27b Qwen. **Pending decision: vLLM vs Ollama** (vLLM recommended).
   - Then sequential tool calling plus an eval of tool choice, with the MCP
     expert's advice: a small tool subset per use case, fixed sequences in
     code, and capped LLM retries.
4. Later: connect the real tools (Jaxon, Arleen, Kyle) to the slots, the
   bounded diverse near-miss archive, full structured repair instructions,
   an LLM step for objective → constraints, and D2 Handoff/Feedback records
   for human reviews.
