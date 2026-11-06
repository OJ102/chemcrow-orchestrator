# Progress Log

A dated record of what we're doing on the orchestrator, why, and what's
next. Newest entries go at the top of the **Log** section.

## Team

| # | Name | Role | Connects to the orchestrator via |
|---|---|---|---|
| S1 | Kyle Russell | Data & Knowledge Pipeline: OPV2D, spectral data, RAG corpus, experiment data model | `Retriever` |
| S2 | Shashank Sathiyanarayanan | LangGraph Orchestration: workflow graph, state machine, tool routing, checkpoints | owns this package |
| S3 | Jaxon Polo | Candidate Generation & Validation: MatGPT wrapper, RDKit filters, diversity, novelty, refinement logic | `CandidateGenerator` |
| S4 | Arleen Monteiro | Chemprop & Scoring: property prediction, uncertainty, AVT/LUE ranking, Pareto scoring | `PropertyPredictor` |
| S5 | Nyla Spencer | Evaluation, Dashboard & Feedback: benchmarking, UI, reports, lab-feedback ingestion, final demo | starts runs / shows results |

## Where things stand (2026-10-26)

| Area | Status |
|---|---|
| Loop follows D1 shapes and D2 behaviour (step 1) | ✅ Done |
| Save runs to disk + resume after a crash (item 4) | ✅ Done (2026-11-06), 35 tests pass |
| Gate and ranker | ✅ Placeholders following D2 (Arleen's/Nyla's real ones plug into their slots) |
| Escalate = set aside (no pause) | ✅ Done |
| Near misses kept and ranked (NM-1..NM-4) | ✅ Done |
| Chemistry check | 🟡 Our `validate` safety net (RDKit validity + repeated pairs) until Jaxon's C3 validation exists |
| Web API (`POST /runs`, `GET /runs/{id}`, cancel) + pinned versions (item 5) | ⬜ Next |
| Retries, failure handling, audit log (item 6) | ⬜ |
| SynSpace / ModifyMol testing | ⬜ After items 5–6 |
| MCP server | ✖ Dropped (advisor, 2026-11-06) |
| Bounded, diverse near-miss archive | ⬜ Waiting on Jaxon's similarity measure |
| Structured repair instructions | 🟡 Parents carry machine-readable violations; the rest waits on Jaxon |
| Real retriever / generator / predictor | ⬜ Fakes (Kyle, Jaxon, Arleen) |
| LLM anywhere in the graph (objective → constraints) | ⬜ Later (end goal of Q4) |

## Project context: Chemprop vs CycleChemist

From the sponsor PDF ("CAPSTONE Simplified Architecture" slide): Phase 1
uses **one Chemprop model** (Student 4) as the single prediction path. It
replaces CycleChemist's specialist predictors (OPVC, MOE², P³) and makes
Graphormer and DimeNet++ unnecessary. **MatGPT stays** as the generator
(Student 3). Spectral/DFT models, toxicity and retrosynthesis remain separate
validation tools, mostly for final candidates. Chemprop only covers a
property if it is trained with labels for it (e.g. AVT/absorption).

For the orchestrator: `PropertyPredictor.predict(candidates)` (a batch of
donor+acceptor pairs) maps onto Chemprop's multicomponent (pair) mode.
Uncertainty comes from Chemprop's uncertainty methods.

**Unresolved:** does MatGPT's RL reward (`finetune.py`, currently P³ × OPVC)
also switch to Chemprop? If not, the generator optimises against different
predictions than our gate judges by. Ask Om / Dr. Rezayat.

## Known issues (in our own code)

The five issues found on 2026-10-13 have all been handled. Q1–Q5 below record how:
1. ~~"Improvement" counts failed molecules~~ → replaced by formula NM-4 (near-miss progress), used only by the optional `patience`.
2. ~~Unapproved escalations vanish~~ → escalate no longer pauses. Escalated candidates are kept in the `escalated` pile for DFT/lab handoff.
3. Reusing a `thread_id` mixes runs → `run_id` = `thread_id`, and every run gets a fresh UUID. A defensive check is optional, for the API step.
4. `evidence` is collected but never used → still true; deliberately deferred (Q4).
5. ~~The stub generator ignores feedback~~ → it now revises the parents with local edits. The real approach depends on Jaxon's answer.

## Open design questions

- **Revision with MatGPT** (waiting on Jaxon): how the real generator uses parents and structured repair instructions.
- **Chemprop outputs** (waiting on Arleen): all six D1 values with per-value uncertainty? Especially AVT.
- **Where the tools run** (waiting on Jaxon/Arleen): GPU server vs local; HTTP vs Python import.
- **`target_type` / `fixed_partner_smiles` in `POST /runs`** (team): D1 has no field for them.
- **Where an LLM fits**: objective → constraints is the agreed end goal.

## Decisions on the known issues (2026-10-15)

| # | Issue | Decision | Status |
|---|---|---|---|
| Q1 | What counts as progress / near misses | Molecules that fail the rules must not count as a "new best". 2026-10-23, **advisor (Dr. Rezayat): keep near misses**. See "Advisor guidance on near misses" below. | Decided; design next |
| Q2 | Declined escalations vanish | Declines come back as a **report with the reason** they were declined, so that reason can drive improvement (feedback). 2026-10-22: the reason is **both** a code from a fixed list (machine-readable, can become feedback) **and** an optional free-text note (for humans). Per Nyla, the reason is required. The list of reason codes is still to be defined (check C3). | To implement |
| Q3 | Reusing a thread_id mixes runs | Nyla (2026-10-22): use the **D1 `run_id` (random UUIDv4) as the LangGraph `thread_id`**, so there is one ID per run. Every run starts through `POST /runs`, which creates a fresh run_id, so accidental reuse shouldn't happen. A defensive check is optional. | Mostly resolved |
| Q4 | Evidence unused | Deferred: finish the graph first. **End goal: use evidence to turn the user's objective into constraints** (PDF step 1). | Later |
| Q5 | Generator ignores feedback | Shashank will ask **Jaxon Polo (S3)** (MatGPT owner) how the real generator can take feedback. | Waiting on S3 |

## Future architecture: MCP (Model Context Protocol)

**Decision (2026-10-15, Shashank):** MCP is not needed for the capstone graph,
but **will be needed later**. ChemCrow alone has 19 tools (see
`chemcrow/agents/tools.py`), and MatGPT, Chemprop, AiZynthFinder and others
will join them. At that size, an LLM agent choosing among many tools
needs a standard tool interface.

Extra reason it fits this repo: the tools live in **incompatible Python
environments**. ChemCrow's tools need the old LangChain (`.venv-chemcrow`), and
AiZynthFinder needs its own env. An MCP server runs *inside* each tool's own
environment and talks to the orchestrator over a connection, so nothing
has to be imported across venvs.

**What this means now:** keep tool access behind `interfaces.py`. Later,
an MCP-backed implementation can be plugged into the same slots without
rewriting the graph.

## Advisor guidance on near misses (Dr. Rezayat, 2026-10-23)

A "near miss" is a candidate that narrowly failed one or more thresholds.

| # | Guidance | Plain meaning | Likely owner | In D2 already? |
|---|---|---|---|---|
| A1 | Log every candidate | Nothing is ever thrown away from the record | C2 (Shashank) + G4 audit | Yes (history, audit) |
| A2 | Keep a **bounded, chemically diverse active near-miss archive** | A size-limited "working set" of the most promising near misses to revise from, without many near-identical molecules in it | C2 state (Shashank); diversity measure from C3 (Jaxon) | No |
| A3 | Rank near misses by **uncertainty-aware normalized constraint violations + Pareto dominance**, not a weighted score | Measure *how far* each one is from passing (as a fraction of the threshold, and relative to the prediction's uncertainty), then compare using Pareto ("better on every count"), never by adding up weighted scores | Gate/ranking (Arleen/Nyla) | Partly (R8 rejects weighted scores) |
| A4 | **Diagnose the repair route**: the molecule, its partner, or the device/process conditions | Work out *what* should change to fix the failure | Gate (Arleen) + ChemE input | No |
| A5 | **Escalate borderline candidates** to higher-fidelity evaluation (e.g. human expert Dr. Hendsbee, Brilliant Matters) | If a molecule is so close to the line that the uncertainty could flip pass/fail, ask an expert instead of guessing | Gate policy (Arleen) | Partly (escalate exists, but not for borderline fails) |
| A6 | **Never retrain predictors on their own predictions** | Only real measurements (DFT/lab) may become training data. Chemprop scores are not ground truth; training on them reinforces the model's errors and invites "reward hacking" | C1 ingest (Kyle) + C4 (Arleen) | Yes (NFR07, Feedback validation) |
| A7 | Send MatGPT **structured repair instructions**: preserve strengths, change one partner at a time, prefer local chemically meaningful edits, avoid repeating failed edits | The revision request says *what to keep*, *what to change*, *how small the change is*, and *what was already tried* | C2 builds the packet (Shashank); C3 consumes it (Jaxon) | No (ConstraintPacket has parents only) |
| A8 | The gate must return **machine-readable feedback**, not just "AVT failed" | Structured data (field, actual, minimum, shortfall, uncertainty, suggested repair route) instead of a sentence | Gate (Arleen) | Partly (D1 Violation has field/actual/minimum/unit) |

These go **beyond D2**, so the team must agree on spec updates (Arleen's gate
output, Jaxon's input packet, Kyle's ingest rule). Adding *optional response
fields* is not on D1's breaking-change list; adding enum values is.

## Team specs

D0, D1 and D2 were shared on 2026-10-22. "C3" means Component 3 (Generate
and Validate), not a document. **The full spec-vs-code comparison is in
[spec-alignment.md](spec-alignment.md).** The current code differs from the
spec in many places and must be reshaped before real tools are connected.

## Log

### 2026-11-06 (later): Item 4 done: runs saved to disk and resumable
- **`storage.py`** (new): `make_checkpointer` (LangGraph's SQLite auto-save,
  `runs/checkpoints.db`, with D2's settings: foreign keys, WAL, 5 s busy
  timeout) and `RunStore` (our `runs` table in `runs/runs.db`, D1's Run
  fields; create/update/get/list; the database refuses statuses outside D1).
  An **allow-list** of our classes (`ALLOWED_STATE_CLASSES`) stops LangGraph's
  "unregistered type ... will be blocked" message; the in-memory saver uses it too.
- **`runner.py`** (new): `start_run` (queued → running → final status) and
  `resume_run` (continues from the last checkpoint; refuses finished or unknown
  runs). Any error marks the run `failed` with the message.
- **`graph.py`**: every step is timed; `active_seconds` (added up) replaces
  `started_at`, so only working time counts (decision Q3).
- **`__main__.py`**: `--db FOLDER`, `--list`, `--resume RUN_ID`.
- **Tests**: 35 pass (7 new in `test_storage.py`). Verified the allow-list test
  can detect the message (it appears without the allow-list). The demo gives
  identical results in memory and on disk.
- `requirements.txt` gained `langgraph-checkpoint-sqlite>=3.1`; installing it
  upgraded `langgraph-checkpoint` 4.2.0 → 4.3.1. `.gitignore` ignores `runs/`
  and `*.db` files.
- Docs: README (new commands), primer (databases, try/except, wrappers,
  inheritance, pytest fixtures), code guide (storage.py, runner.py, timing),
  spec-alignment table.

### 2026-11-06: MCP dropped; item 4 (save runs to disk) decisions
- **MCP is not being pursued.** Advisor: "it seems that MCP may not buy us much.
  Therefore, let's focus on orchestration and finding the best tools and, of
  course, new polymer recommendations and results." (This follows the MCP
  expert's answer 2.)
- **Item 4 decisions (Shashank):**
  - Q1 location: runs should eventually live on the shared server so Nyla can
    reach them. **For now, local**, with the path configurable; delete if it
    takes too much space.
  - Q2: **two database files**, one for LangGraph checkpoints and one for our
    runs table. D1/D2 aren't perfect: D2 describes a single store, and we're
    deliberately deviating.
  - Q3: the time budget counts **only time the run is actually working**;
    downtime after a crash doesn't count.
  - Q4: Phase 1 = one worker, a checkpoint after every step, a runs table.
    Leases and fencing tokens are deferred; idempotency receipts go with item 5
    (the API). This answers peer review PR-03.
- Installed `langgraph-checkpoint-sqlite` into `.venv-langgraph`.

### 2026-11-05 (later): MCP expert's answers (questions 1–5, 7, 8)
1. **One server or several:** depends on use cases. If some tasks only need
   certain tools, split into several servers instead of one.
2. **MCP vs web API:** with a well-documented web API, MCP adds little. One MCP
   tool that knows every endpoint and decides what to call would make results
   depend on the LLM's choice, so it may not be a good place for MCP.
3. **Choosing among 20+ tools:** show the model only a small relevant subset.
4. **Sequential calling:** the LLM should know the sequence; specify the order
   and what to do on specific errors. A fixed order can be set up in code.
   Show results before the LLM's next choice.
5. **Eval plan** (expected tool sequences, fake tools, scores): sound.
7. **Errors/retries:** if the LLM can tell an error is transient, it retries;
   if changing the parameters would fix it, it does that; otherwise the
   error is passed up (bubbles up).
8. **Structured results:** tools return JSON, so data isn't a problem.
- Unanswered: 6 (slow tools), 9 (security on a shared machine), 10 (vLLM vs
  Ollama with MCP).
- Takeaways to discuss: a deterministic orchestrator calls a documented web
  API (no MCP needed); an LLM tool-calling layer is only for cases where the
  model really chooses, using a small, use-case-specific tool subset; fixed
  sequences belong in code, not in an LLM. Retries need two layers: D2's
  deterministic transport retries (max 3, 1 s/2 s) in code, plus LLM-level
  recovery (fix the parameters) with a cap and every attempt logged. Answer 2 is
  in tension with the advisor's view that MCP will be needed; to discuss.

### 2026-11-05: Cluster shared with another capstone team
- The advisor asked whether two CS students from another capstone team may
  also use `cs-gpu-00`; Shashank agreed. The vLLM processes on GPUs 0 and 1
  are probably theirs.
- To agree with them: which GPUs each team uses (proposal: theirs 0 and 1,
  ours 2 and 4), GPU memory limits, ports, disk use (home folders, not the
  nearly full root disk), and a way to contact each other.

### 2026-11-03 (later): More cluster facts; model chosen
- **Model: the 27b Qwen model** (same family as ChemCrow's previous setup).
- **GPUs:** believed to be the team's. But other vLLM processes were running on
  GPUs 0 and 1, so check whose they are before using those two.
- Python 3.12.3 (fine for vLLM). User is only in group `sathiysk`, so **no
  Docker access**; Singularity is the container option.
- **A shared Ollama server (v0.40.12) already runs on port 11434.** Pulling models
  through it would store them wherever that service keeps models, likely on
  the nearly full root disk. So if we use Ollama, run our **own** Ollama on a
  different port with models stored in home (as the previous setup did).
- Pending: Shashank to choose vLLM vs Ollama.

### 2026-11-03 (later): Cluster facts collected
`cs-gpu-00`: Ubuntu 24.04, 128 CPU cores, 503 GB RAM, **4x A100-SXM4-80GB**
(plus a small "DGX Display" card listed as GPU 3), driver 580 / CUDA 13.0.
- **Shared machine, no job scheduler** (no Slurm). Other users are already
  running vLLM: GPU 0 is ~36 GB used at 100% load, GPU 1 is ~51 GB used.
  **GPUs 2 and 4 were free.**
- **No sudo.** Installed: `ollama` (system-wide), `docker`, `singularity`,
  `python3`, CUDA `nvcc`. No `conda`, no `apptainer`.
- **Disk:** the root disk `/` (incl. `/tmp`) is **97% full (55 GB free)**, so
  don't download models there. Home `/home2/sathiysk` is network storage with
  184 TB free, so models go there.
- **Internet:** Hugging Face reachable (HTTP 200).
- Serving options: Ollama (installed; what ChemCrow and D1 use) vs vLLM
  (faster for many requests, structured tool calling, already used by others
  on this machine; installable without sudo via a Python venv or Singularity).

### 2026-11-03: GPU cluster access granted
- Shashank can log in to **`cs-gpu-00.digitfut.uc.edu`** with his UC 6+2
  username/password. Next: log in and collect the setup facts (GPUs, admin
  rights, scheduler, disk, internet, available tools) before choosing how to
  serve the model.

### 2026-11-02: Tool-calling plan and order agreed
- **The eval is for tool calls:** did the local model choose the right tool?
  This replaces the earlier idea of grading chemistry answers. Sequential tool
  calling is to be explored.
- **Order (Shashank):** (1) set up the 4 A100 GPUs, (2) serve the local model,
  (3) set up the MCP server, (4) build sequential tool calling and its evals.
- An MCP expert call at Siemens is being arranged by the advisor.
- Waiting on: cluster access and setup details.

### 2026-10-29 (later): New workstream: ChemCrow local-model inference + evaluation
- Requested: optimize ChemCrow's inference on the local model, and set up an
  evaluation (eval) set. This is ChemCrow-side work (`chemcrow/`, `.venv-chemcrow`),
  not the orchestrator.
- Findings: ChemCrow talks to Ollama through its OpenAI-compatible endpoint
  (`ChatOpenAI`, langchain 0.0.275); configured for `qwen3.8:27b` on a second
  Ollama server (port 11435, per git history). **This laptop has no Ollama
  and only a 4 GB RTX A1000**, so benchmarks must run on the model's server.
  Slowness has two sources: per-call speed (model size/quantization, GPU fit,
  context size, keep-alive, Qwen3 "thinking") and number/length of calls
  (full prompt with 19 tool descriptions resent each step, up to 40 steps,
  tools that call the LLM internally). Only existing agent test is skipped.
- Eval plan to agree: fixed questions with code-checked answers (no LLM
  grading, per the ChemCrow paper), repeated runs, metrics = accuracy, time,
  steps, tool/format errors; baseline first, then one change at a time.
- Waiting on: where the model runs and access to it; eval scope; offline-only
  first version?; ownership.

### 2026-10-29: Arleen confirms uncertainty; ChemCrow tools surveyed
- **Arleen:** every output, AVT included, will come with an **uncertainty
  value (the spread across an ensemble)** and an **in-domain flag**. This
  matches our `Uncertainty` (one per value) and `Prediction.in_domain`, so the
  gate's Escalate path and the near-miss formula NM-2 will have real inputs.
  (AVT training data is still the open issue from the 2026-10-27 entry.)
- **Advisor request (urgent):** evaluate ChemCrow's tools for our project.
  Written up in [chemcrow-tools.md](chemcrow-tools.md): all 19 tools rated
  with where each would fit. Highest value: control-chemical similarity,
  molecular similarity (for the diverse archive), patent check, and
  retrosynthesis for finalists. PubChem-based tools can't judge invented
  molecules. Found bugs, including broken local retrosynthesis and a
  retrosynthesis prompt that invents a solvent step. Tools can't be imported
  into `.venv-langgraph`: re-implement the simple ones; use MCP or HTTP for heavy ones.
- Also checked the **original repo and paper** (Nature Machine Intelligence
  2024). Our fork's tool list matches the original. The paper has tools the code
  lacks; the important one is **ModifyMol** (SynSpace, MIT, public): local,
  synthesizable analogs of a molecule, a candidate way to revise near misses.
  The paper's chromophore case study (target 369 nm, measured 336 nm) and its
  finding that GPT-4 can't judge chemistry support our deterministic gate and
  DFT/lab handoff. Added as section 6 of chemcrow-tools.md.

### 2026-10-27: Kyle's OPV2D data (Active_Database.csv) reviewed
Source: https://github.com/sunyrain/OPV2D. Files received: `Active_Database.csv`
(39,998 device rows, 37 columns) and `classified_opv_database.csv` (4,023
molecules with a donor/acceptor role).
- **Kyle's counts verified** (2,478 unique donors / 1,842 unique acceptors;
  978 / 1,677 non-canonical). But: all 6,206 / 7,570 "invalid or empty"
  entries are **empty cells** (0 unparseable); his script's "removed
  duplicates" message prints the number **kept**; and de-duplicating per
  column drops the pairs. Pairs are what matter: **27,361 rows with a valid
  pair, 5,797 unique ordered pairs**, 27,293 of them with a PCE value.
- **No AVT / transmittance column.** This confirms D1 and IA02: Chemprop can't
  learn AVT from this data alone. A blocker for the transparency goal (Arleen, Kyle).
- **Measured** HOMO/LUMO (donor and acceptor) and device/process columns
  (solvent, additive, D:A ratio, thickness, annealing, ETL/HTL). These are
  relevant to the advisor's "repair via device/process conditions" and are
  legitimate training labels (measured, not predicted).
- **Stereochemistry mismatch:** the data's Y6 (5,088 rows) includes `/` `\`
  double-bond geometry; our demo's Y6 doesn't. Same molecule, different text,
  so exact-text matching treats them as different. Team should agree on a
  stereo rule (keep vs strip; InChIKey per D1).
- 97 molecules appear in both roles in the data; the classified file assigns
  one role each (94 donor, 3 acceptor); 200 of 4,223 molecules are unclassified.

### 2026-10-26 (later): Step 1 done, the code now follows D1/D2
- **`state.py`** rewritten to D1's shapes: `RunConstraints`, `RunConfig`
  (with the optional `target_count`/`patience` off by default, and a check
  that the fixed partner is a real molecule), `Policy`, `Values`/`Uncertainty`/
  `Prediction`, `ChemistryViolation` vs `ThresholdViolation`, `Decision`,
  `NearMiss`, `Candidate` (pairs + lineage), and `OPVState` (set-aside pile,
  parents, failed predictions, near-miss output).
- **`interfaces.py`**: five tool slots (the gate and ranker are new),
  `ConstraintPacket`, `GenerationResult`.
- **`near_miss.py`** (new): the advisor's formulas, labelled **NM-1**
  (fraction short), **NM-2** (uncertainties short + borderline within 1),
  **NM-3** (Pareto, no weights), **NM-4** (progress, used by `patience`).
- **`gate.py`**: placeholder gate and ranker following D2's rules
  (Arleen's/Nyla's real versions replace them).
- **`stubs.py`**: fake tools in the new shapes. Revisions are local edits
  that attach a small group at an atom with a free hydrogen (gluing text onto
  the end produced impossible molecules in early testing).
- **`graph.py`**: start → retrieve → generate → validate → predict → decide →
  (generate | finish). Escalate no longer pauses. End conditions are in
  `_next_move`.
- **Tests**: 28 passing across `test_gate.py`, `test_near_miss.py` and
  `test_graph.py` (shared fakes in `fakes.py`).
- **Docs**: README, primer (new terms + Python features), code guide (rewritten),
  and the spec-alignment status table updated.
- Next: step 2, saving runs to disk (SQLite checkpointer), unless Jaxon's or
  Arleen's answers change priorities.

### 2026-10-26: Step 1 plan agreed; D1/D2 re-checked
- **Plan for step 1 ("reshape to the spec")** explained file by file and
  agreed: D1 data shapes (pairs, six named values with per-value uncertainty,
  Accept/Revise/Escalate/Stop, run statuses); D2 loop (escalate = set aside,
  revise from parents, iteration/runtime stops); placeholder gate and ranker
  following D2's rules; gate and ranker become tool slots (Arleen/Nyla).
- **Fixed-partner approach kept**: each generated molecule is paired with the
  fixed partner. This fits the advisor's "modify one partner at a time".
- **`target_count` and `patience` kept as optional settings, off by default.**
- **Validation (A2):** keep our own `validate` step as a safety net until
  Jaxon's C3 validation exists, then remove it.
- **Patience "improvement" follows the advisor's near-miss rule**: normalized
  shortfall per constraint, Pareto comparison (no weights), uncertainty-aware
  borderline flag. Exact formula proposed; waiting for Shashank's confirmation.
- **Re-checked D1/D2** (new copies `D1_Detailed_Design.docx`,
  `D2_Detailed_Design.docx`): D1 is identical to "Detailed System Design -
  Part 1". D2 sections 1–5 are unchanged; section 6 now contains the real peer
  review (PR-01..PR-07). Relevant to us: **PR-03**, which says to document
  which orchestrator machinery is Phase 1 vs deferred.
- Interpretations not stated in the spec (going ahead unless told otherwise):
  a round where every prediction failed sets the run status to `failed`
  (alternative: record it and end as `completed` with nothing to revise,
  since D2 reserves `failed` for infrastructure exhaustion); `stopped` runs
  still store a ranked list; the placeholder policy doesn't require the
  SA-score check. Open for the team: D1's `POST /runs` has no field for
  `target_type` / `fixed_partner_smiles`.

### 2026-10-23: Waiting on Jaxon and Arleen
- Sent updated questions (revised after reading D1/D2 and the advisor's
  email) to Jaxon (weights/runtime, partner conditioning, structured repair
  instructions, diversity measure, RL reward) and Arleen (six D1 values with
  per-value uncertainty, the advisor's gate requirements, how to call the
  model and gate, training only on measured data).
- Paused until they reply: archive size, who diagnoses the repair route, and
  whether to fold the archive and repair packet into step 1 of the reshaping plan.

### 2026-10-22: Read the team design docs (D0/D1/D2)
- **Escalate = set aside** (Shashank agreed). This matches D2: escalated
  candidates are excluded from the shortlist, shown separately, and handed to
  DFT/lab via Handoff records. Lab/DFT results return as Feedback, which Kyle
  publishes into a new knowledge snapshot, so a set-aside molecule can pass in a
  future run once the model has learned from that evidence.
- **Ownership per D2:** Shashank owns C2 (the run API `POST /runs`,
  `GET /runs/{id}`, cancel; leases, checkpoints, retries), G1 auth, G3
  config/version pinning and G4 audit log. The **Decision Gate is Arleen's**
  and **shortlist ranking is Nyla's**, so our `gate.py` is a placeholder for
  their work.
- **The spec has no target_count or patience stop.** A run repeats only
  "Revise" candidates (as parents of the next round) until none remain or the
  iteration/runtime budget runs out. This bears on the pending advisor email (Q1).
- Wrote `docs/spec-alignment.md` with the component map, the run behaviour,
  the gate rules, the escalate/handoff flow, and a table of every difference
  between spec and code.

### 2026-10-22: Nyla's (S5) answers
What we learned:
- **The dashboard talks to a web API, never to the graph directly:**
  `POST /runs` starts a run, `GET /runs/{run_id}` is polled for status, and
  the shortlist is shown when the run completes. She also needs
  `GET /runs?owner=&status=&limit=` to list old runs (not yet in D1/D2).
- **IDs:** use the D1 `run_id` UUID as our `thread_id`. This resolves Q3.
- **Conflict: escalate.** Our code *pauses* the run (`interrupt`). In
  D2, escalate does **not** pause: escalated candidates are excluded from
  acceptance and handed off to DFT/lab, and the run status list has no
  "paused" value. A real pause needs a new `waiting_for_review` status and
  `POST /runs/{run_id}/reviews` taking `{candidate_id, approve, reason}`.
  D1 treats a new status as a **breaking change**, so this needs team
  agreement. Nyla supports a **required** decline reason.
- **Report contents she needs** (ideally bundled as `GET /runs/{run_id}/report`):
  ranked list with values, uncertainty, rationale and risk flags; every
  candidate with its iteration and **parent**; C3 rejection reasons and
  **failed predictions**; every gate decision with violations; every human
  review with **reviewer, time and reason**; the stop reason (completed,
  iteration limit, **runtime limit, cancelled, failed**); and **pinned
  model/policy/snapshot versions**.
- Gaps vs our current code: no parent tracking, no rationale/risk flags,
  predictor errors aren't caught, reviews don't record reviewer/time/reason,
  only 2 of the 5 stop reasons exist, and no version pinning.

### 2026-10-15 (later): Beginner-level comments and primer
- Rewrote every comment and docstring in the package for a reader with no
  chemistry background who is new to LangGraph. Every variable, function,
  class and test line is now explained. Verified no logic changed: the code
  structure with comments stripped is identical to commit `1552297`, 6/6
  tests pass, and the demo output is unchanged.
- Added `docs/primer.md` (chemistry words incl. PCE/AVT/LUE, Python features,
  LangGraph, pytest) and `docs/design-questions.md` (the five known issues
  explained with examples and options). Shashank later removed
  design-questions.md: design questions are now discussed in chat, and the
  decisions are recorded in this file.

### 2026-10-15: Docs created; chemistry check in design
- Created `orchestrator/README.md` (setup and run instructions for the
  langgraph venv), `docs/code-guide.md` (file-by-file walkthrough with
  snippets), and this progress log.
- Started designing a **chemistry check** for the `validate` node. Today
  `validate` only checks that RDKit can parse the SMILES and that it isn't a
  duplicate. Open questions on scope (basic sanity vs OPV structural
  plausibility vs HOMO/LUMO alignment), where it lives, and hard vs soft
  failure. No code written yet.

### 2026-10-14: Commented the code; first commit
- Added explanatory comments and docstrings to every file (variables,
  functions, classes). No behaviour change: 6/6 tests pass and the demo
  output is identical.
- Committed as `1552297` on `shashank/langgraph-orchestration`.
- Pushing failed: `origin` is `OJ102/chemcrow-orchestrator` and the account
  doesn't have push access yet. Fix: become a collaborator, or fork and open a PR.

### 2026-10-13: Reviewed the generated baseline
- A previous Claude session (VS Code) generated the `orchestrator/` package:
  state models, tool interfaces, stubs, deterministic gate, LangGraph
  workflow, demo, and tests.
- Verified in `.venv-langgraph`: 6/6 tests pass; the demo finishes with 3
  accepted (fake) candidates.
- Compared it with the sponsor PDF's 6-step refinement loop and found the
  gaps listed in "Where things stand".
