# Spec Alignment: What the Team Design Docs Require vs What Our Code Does

The team wrote three design documents that the orchestrator must follow.
This page explains, in plain language, what they say about **our** part, and
lists every place where the current code differs. Read
[primer.md](primer.md) first if any word is unfamiliar.

> Source documents (in Shashank's Downloads folder, 2026-10-22):
>
> | Name used in the docs | File | What it covers |
> |---|---|---|
> | **D0** | `Design Document.docx` | The big picture: the 6 components, who owns what, the main design decisions |
> | **D1** | `Detailed System Design - Part 1.docx` | Data models + the **web API contract** (`/api/v1/...`), plus details for C1, C3, C4 |
> | **D2** | `D2_Detailed_Design.docx` | Finishes the orchestrator (C2), the Decision Gate, outputs (C5), and the support services (G1–G4) |
> | (all of the above) | `Assignment_4_Consolidated_DRAFT.pdf` | D0 + D1 + D2 + peer-review notes in one PDF |
> | (review) | `OPV_Peer_Review_Assessment.docx` | A classmate's critique of the peer-review notes |
>
> **"C3" is not a typo.** The docs number the components **C1–C5**, so when
> Nyla wrote "C3 rejection reasons" she meant *Component 3 (Generate and
> Validate, Jaxon's part)*.

---

## 1. The components and who owns them

| ID | Component | Owner | One-line job |
|---|---|---|---|
| C1 | Knowledge Sources | Kyle | Stores the data and research text (in versioned "snapshots") |
| **C2** | **OPV Agent Orchestrator** | **Shashank** | **Runs the loop, keeps state, checkpoints, decides when the run ends** |
| C3 | Generate and Validate | Jaxon | MatGPT proposes molecules; RDKit checks them; records rejection reasons and **lineage** (which molecule each came from) |
| C4 | Chemprop Property Prediction | Arleen | Predicts the numbers + uncertainty + "in domain?" for each pair |
| Gate | Transparent-OPV Decision Gate | **Arleen** | Gives each candidate Accept / Revise / Escalate / Stop |
| C5 | Outputs and Feedback | Nyla | Ranked shortlist, hand-offs to DFT/lab, and feedback coming back |
| **G1** | **Authentication** | **Shashank** | Checks who is calling the API and what they're allowed to do |
| G2 | Notifications | Nyla | In-app notices |
| **G3** | **Configuration** | **Shashank** | Validates and **pins** (freezes) the model/policy versions for each run |
| **G4** | **Logging and audit** | **Shashank** | Records every important event (the `/runs/{id}/events` log) |

**Important surprises:**
- The **Decision Gate belongs to Arleen**, and the **shortlist ranking belongs
  to Nyla**. Our `gate.py` (the `evaluate` and `pareto_rank` functions) was
  written by the previous Claude session as a placeholder. Under the spec, the
  orchestrator *calls* the gate and ranking instead of owning them.
- **The API server is yours.** D2 says the orchestrator (C2) handles
  `POST /runs`, `GET /runs/{run_id}` and `POST /runs/{run_id}/cancel`. G1, G3
  and G4 (authentication, configuration, audit log) are also yours. That's
  the answer to "who builds the API server".

---

## 2. How a run works according to D2 (behaviour "A2")

In plain steps:

1. **Someone calls `POST /runs`** (Nyla's dashboard). We check the request,
   look up and freeze ("pin") all the versions and settings, save the run, and
   immediately answer **202 Accepted** with a `run_id`. 202 means "your order
   is in", not "it's done".
2. **A worker picks up the run.** It holds a 60-second "lease" (a claim that
   says "I'm working on this run") and renews it every 20 seconds. If the
   worker crashes, another one can safely take over.
3. **Retrieve** evidence from Kyle's pinned snapshot. **Generate and validate**
   up to `candidate_limit` distinct pairs (Jaxon). **Save a checkpoint after each phase.**
4. **Predict** only the validated pairs (Arleen's Chemprop) and save the batch.
5. **Evaluate** the saved batch with the Decision Gate (Arleen) and save the decisions.
6. **Repeat only for candidates that got "Revise"**, while iteration and
   runtime budgets remain. Each new round is built from those candidates
   (they become the **parents**). **Never lower the thresholds.**
7. **Save the ranked output and the final status together.**

### How a run ends (statuses from D1)
`queued` → `running` → one of:
- `completed`: the shortlist is saved. **An empty shortlist is still "completed"**
  if every evaluation succeeded and nothing passed.
- `stopped`: ran out of revision attempts / iterations / runtime.
- `failed`: something broke (e.g. a tool stayed unavailable after retries).
- (cancellation via `POST /runs/{id}/cancel`)

There is **no "paused" status**. D1 says adding one is a breaking change.

### Budgets and retries (D2 proposals)
- `max_iterations`: 1–10, default **3**. `candidate_limit`: 1–100, default **20**.
- Whole run: **30 minutes**. Generation step: 120 s. Prediction step: 120 s.
- Retry a failing tool **at most 3 times** (waits of 1 s and 2 s, plus a little randomness).
  Never retry an invalid request.

---

## 3. The Decision Gate rules (Arleen's, but we must handle the results)

Checked **in this order**:

1. Prediction failed → no decision; the orchestrator logs it.
2. `in_domain` is false, or any required uncertainty is missing or above its
   limit → **Escalate**.
3. Synthesizability check unavailable → **Escalate**. SA score too high, or
   OPV-likeness below `opv_min` → **Stop** (this candidate only).
4. Any value below its minimum (AVT, PCE, LUE) → **Revise** if iterations
   remain, otherwise **Stop**.
5. Otherwise → **Accept**.

Exactly equal to a minimum **passes**.

**Outcomes:** `Accept`, `Revise`, `Escalate`, `Stop`. There is **no "reject"**.
Broken molecules are caught earlier by C3 (Jaxon) as **Violation** records and
never reach prediction or the gate.

---

## 4. Escalate = set aside (decided 2026-10-22)

D2: escalated candidates are **excluded from automatic acceptance** and shown
**separately** from the shortlist. A human reviewer can create a **Handoff** to
DFT or the lab (`POST /runs/{id}/handoffs`).

Handoff lifecycle: `requested → acknowledged → results_received → closed`.
Results come back as **Feedback** (measurements, reviewer, rejection reason).
Kyle approves valid feedback into a **new knowledge snapshot**. Old runs never
change.

**How a set-aside molecule can eventually pass:** it's recorded with its
lineage, so nothing is lost. If DFT/lab confirms it, that measurement enters
the knowledge base. Once Arleen's model is retrained on the new snapshot, it
knows that region of chemistry better: lower uncertainty, "in domain". In a
**future** run, that molecule (or a relative of it) can then pass the gate
automatically. Simply re-generating it with the *same* model would give the
same uncertain prediction, so what changes is the *evidence*, not the attempt.

---

## 5. Differences between the spec and our code

Status as of **2026-10-26**, after step 1 ("reshape to the spec"). ✅ = matches
the spec; 🟡 = partly; ⬜ = not yet (a later step).

| Area | Spec (D1/D2) | Our code now | Status |
|---|---|---|---|
| Outcomes | Accept / Revise / Escalate / Stop | Same (`state.Outcome`) | ✅ |
| Escalate | Set aside, no pause, handoff later | Set-aside pile `escalated`; no pause. Handoff records come with C5/API work | ✅ (handoff ⬜) |
| What a candidate is | A pair: `candidate_id`, `donor_smiles`, `acceptor_smiles` | Same; the generated molecule is paired with the fixed partner | ✅ |
| Predicted values | Six D1 values, per-value uncertainty (nullable), `in_domain`, `status ok/failed` | Same (`Values`, `Uncertainty`, `Prediction`) | ✅ |
| Run constraints | `avt_min_fraction`, `pce_min_percent`, `lue_min_percent` | Same (`RunConstraints`). OPV-likeness and uncertainty limits are in `Policy` | ✅ |
| Budget | `candidate_limit` (20), `max_iterations` (3), `seed`, 30-min runtime | Same. `target_count` and `patience` kept as **optional, off by default** (our addition) | ✅ |
| When to stop | Revise loop ends when no Revise remain, or budgets run out | `graph._next_move` | ✅ |
| Revision | Revise candidates become parents (`parent_candidate_ids`) | Parents passed in the `ConstraintPacket` (full candidates, not just IDs); lineage recorded | ✅ (IDs-only form needs the database) |
| Gate rules | D2 "AG" order: trust → OPV-likeness → targets → Accept | `PlaceholderGate`, until Arleen's gate exists | ✅ placeholder |
| Ranking | Pareto on AVT and LUE, PCE floor, D2 tie order, owned by Nyla | `PlaceholderRanker`, until Nyla's exists | ✅ placeholder |
| Validation | Inside C3 (Jaxon) | Our `validate` step as a safety net until Jaxon's exists (decision A2) | 🟡 by design |
| Statuses | queued / running / completed / failed / stopped | Same list. `queued` is set by the API later | ✅ |
| Run ID | D1 `run_id` UUID = LangGraph `thread_id` | Demo uses one UUID for both | ✅ |
| Saving state | SQLite store, checkpoint per phase, leases, idempotency receipts, audit events | **Phase 1 done (2026-11-06):** SQLite checkpoint after every step + a `runs` table with D1's Run fields; resume after a crash. Two files, not D2's single store (team decision). Leases/fencing **deferred** (one worker; PR-03); receipts → item 5; audit events → item 6 | ✅ Phase 1 |
| Time budget | 30-minute run budget | Counts **working time only**; downtime between a crash and a resume isn't counted (decision Q3) | ✅ |
| Tool failures | Retry 3×, then `failed`; per-candidate prediction failures logged | Per-candidate failures logged; all-failed → `failed`; any error → run recorded `failed` with the message (resumable). **No retries yet** | 🟡 item 6 |
| API | `POST /runs`, `GET /runs/{id}`, cancel; plus Nyla's list + report endpoints | None yet; `runner.start_run`/`resume_run` and `RunStore.list` are ready for it | ⬜ item 5 |
| Pinned versions | `knowledge_snapshot_id`, `model_version`, `policy_version` per run (G3) | `policy_version` stored in the runs table | 🟡 item 5 |
| Advisor: near misses kept and ranked | (beyond D2) | `near_miss.py` NM-1..NM-4, `near_misses` output | ✅ |
| Advisor: bounded, diverse archive | (beyond D2) | Not yet; needs Jaxon's similarity measure | ⬜ |
| Advisor: structured repair instructions | (beyond D2) | Parents carry machine-readable violations; preserve/route/tried-edits fields wait for Jaxon | 🟡 |
| Advisor: escalate borderline, diagnose repair route | (beyond D2, Arleen's gate) | Borderline flag computed (NM-2) for her gate to use | 🟡 |

## 6. Things the spec leaves open or that conflict

- **No "no-improvement" stop.** The sponsor PDF mentions stopping on
  "no-improvement limits", but D1/D2 only define iteration, runtime,
  cancellation and failure stops. This affects the advisor question about
  progress/near misses.
- **Near-miss molecules are kept**: D2 revises the "Revise" candidates
  themselves (as parents). Weighting thresholds was rejected in rationale R8
  ("arbitrary weights can hide useful candidates"). **The advisor (2026-10-23)
  added requirements beyond D2**: a bounded, diverse near-miss archive,
  uncertainty-aware ranking of near misses, repair-route diagnosis, escalation
  of borderline cases, structured repair instructions, and never training on
  predictions. See the "Advisor guidance" table in [progress.md](progress.md).
- **Nyla's extra endpoints** (`GET /runs?owner=&status=&limit=`,
  `GET /runs/{id}/report`) are not in D1/D2 yet.
- **D0's owner column is garbled** in the extracted text, but the D0
  responsibility table and D2 agree on the owners above.
- **Peer-review notes action IA01**: "Connect RDKit validator and add
  invalid-SMILES rejection evidence", owners **Jaxon + Shashank**. This is the
  "chemistry check".
