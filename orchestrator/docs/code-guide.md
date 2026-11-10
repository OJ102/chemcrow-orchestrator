# Code Guide

A file-by-file walkthrough of the `orchestrator` package: what each piece
does, why it exists, and how the pieces connect. Snippets are taken from the
code. If the code and this guide disagree, the code wins, and this guide
should be fixed.

New to the words (SMILES, AVT, LUE, Pareto, reducer, ...)? Read
[primer.md](primer.md) first. For how the code maps onto the team's design
documents, see [spec-alignment.md](spec-alignment.md).

Contents:
1. [The big picture](#1-the-big-picture)
2. [state.py: the data shapes](#2-statepy-the-data-shapes)
3. [interfaces.py: the five tool slots](#3-interfacespy-the-five-tool-slots)
4. [near_miss.py: formulas NM-1 to NM-4](#4-near_misspy-formulas-nm-1-to-nm-4)
5. [gate.py: placeholder gate and ranker](#5-gatepy-placeholder-gate-and-ranker)
6. [stubs.py: fake tools](#6-stubspy-fake-tools)
7. [graph.py: the loop](#7-graphpy-the-loop)
8. [storage.py: saving runs to disk](#8-storagepy-saving-runs-to-disk)
9. [runner.py: starting and resuming runs](#9-runnerpy-starting-and-resuming-runs)
10. [policies.py: approved policy versions](#10-policiespy-approved-policy-versions)
11. [worker.py: the worker program](#11-workerpy-the-worker-program)
12. [api.py: the web API](#12-apipy-the-web-api)
13. [\_\_init\_\_.py and \_\_main\_\_.py](#13-__init__py-and-__main__py)
14. [The tests](#14-the-tests)
15. [One run, step by step](#15-one-run-step-by-step)

---

## 1. The big picture

The orchestrator is a **manager**. It invents nothing, predicts nothing and
judges nothing itself. It calls five tools in the right order, keeps records,
and decides when the run ends.

| Tool slot | Owner (D2) | Today | Contract |
|---|---|---|---|
| retriever | Kyle (C1) | `StubRetriever` (fake) | `retrieve(objective, config)` |
| generator | Jaxon (C3) | `StubGenerator` (fake) | `generate_validate(packet)` |
| predictor | Arleen (C4) | `StubPredictor` (fake) | `predict(candidates)` |
| gate | Arleen (Gate) | `PlaceholderGate` (D2 rules) | `decide(candidates, config, policy, iterations_remain)` |
| ranker | Nyla (C5) | `PlaceholderRanker` (D2 rules) | `rank(accepted)` |

```
START → start → retrieve → generate → validate → predict → decide ─┬─ (Revise candidates left) → generate
                                                                   └─ (done / limit reached)   → finish → END
```

---

## 2. `state.py`: the data shapes

Every class names the document it comes from. Anything **not** from D1/D2 is
marked *(our addition)* in the code.

### Allowed-value lists
```python
Outcome   = Literal["Accept", "Revise", "Escalate", "Stop"]          # the gate's verdicts (D1)
RunStatus = Literal["queued", "running", "completed", "failed", "stopped"]  # (D1)
RiskFlag  = Literal["OUT_OF_DOMAIN", "HIGH_UNCERTAINTY", "UNCERTAINTY_UNAVAILABLE",
                    "CONSTRAINT_VIOLATION"]                          # (D1)
```
There's **no "reject"**. Broken molecules are caught by validation and get a
`ChemistryViolation` instead; they never reach the gate.

### What the user chooses: `RunConstraints` and `RunConfig`
```python
class RunConstraints(BaseModel):
    model_config = ConfigDict(extra="forbid")
    avt_min_fraction: float = Field(ge=0, le=1)
    pce_min_percent: float = Field(ge=0, le=100)
    lue_min_percent: float = Field(ge=0, le=100)
```
`Field(ge=..., le=...)` enforces D1's ranges, and `extra="forbid"` rejects
unknown fields (D1: "Extra request fields are rejected").

`RunConfig` adds `target_type` and `fixed_partner_smiles` *(our addition:
D1's POST /runs has no field for these yet; open team question)*, D1's
`candidate_limit` (default 20), `max_iterations` (default 3) and `seed`, and
our optional `target_count` and `patience` (both `None` = off by default).

The partner is checked as soon as a `RunConfig` is created:
```python
@field_validator("fixed_partner_smiles")
@classmethod
def _partner_must_be_a_real_molecule(cls, smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("fixed_partner_smiles is not a valid molecule ...")
    return Chem.MolToSmiles(mol)       # stored in canonical form
```
If the partner were broken, every pair would be invalid, so the run is refused up front.

### The team's approved limits: `Policy`
```python
class Policy(BaseModel):
    policy_version: str
    opv_min: float                                   # below -> Stop
    uncertainty_max_by_target: dict[ValueName, float]  # above, or unknown -> Escalate
    max_runtime_seconds: int = 1800                  # D2's 30-minute budget
```
Users choose *targets*. *Safety limits* are approved separately (D2
PolicyVersion). Real values haven't been approved yet (peer review PR-06), so
the demo and tests use placeholders.

### What the predictor returns: `Values`, `Uncertainty`, `Prediction`
```python
class Values(BaseModel):          # D1's six numbers, with D1's ranges
    opv_likeness: float = Field(ge=0, le=1)
    donor_homo_eV: float
    acceptor_lumo_eV: float
    pce_percent: float = Field(ge=0, le=100)
    avt_fraction: float = Field(ge=0, le=1)
    lue_percent: float = Field(ge=0, le=100)   # = avt_fraction x pce_percent
```
`Uncertainty` has the same six names, each `Optional[float]`. `None` means
"unknown", which is **not** the same as 0. `Prediction` is either
`status="ok"` (values, uncertainty and `in_domain` filled in) or
`status="failed"` (an `error` message instead).

### The two kinds of violation
| Class | Meaning | Made by | Example |
|---|---|---|---|
| `ChemistryViolation` | "not a usable molecule" | validation (before prediction) | `check_code="invalid_smiles"` |
| `ThresholdViolation` | "misses a target" | the gate (after prediction) | `field="avt_fraction", actual=0.38, minimum=0.40, unit="fraction"` |

`ThresholdViolation` is the machine-readable feedback the advisor asked for.

### `Decision` and `NearMiss`
`Decision` = `candidate_id`, `outcome`, `rationale` (a sentence for humans),
`risk_flags`, `violations`. `NearMiss` stores the results of formulas NM-1
(`shortfall`) and NM-2 (`sigmas_short`, `borderline`). See section 4.

### `Candidate`: one pair moving through the loop
```python
class Candidate(BaseModel):
    candidate_id: str                    # e.g. "it2-c05"; must match D1's pattern
    iteration: int
    parent_candidate_id: Optional[str]   # LINEAGE: which candidate it was revised from
    generated_smiles: str                # raw generator output
    canonical_smiles / donor_smiles / acceptor_smiles   # set by validate
    valid: bool
    chemistry_violations: list[ChemistryViolation]
    prediction / decision / near_miss    # set by predict and decide

    @property
    def pair_key(self) -> str:           # "donor>>acceptor", used to spot repeats
```
Steps never change a Candidate in place. They make updated copies with
`model_copy(update={...})`.

### `OPVState`: the shared notebook
| Key | Reducer | Meaning |
|---|---|---|
| `run_id`, `objective`, `config`, `policy` | replace | set by whoever starts the run |
| `evidence` | replace | from the retriever (not used yet) |
| `batch` | replace | this round's candidates |
| `history` | **append** | every candidate ever made ("log every candidate") |
| `accepted` | **append** | every Accept |
| `escalated` | **append** | the **set-aside pile** for DFT/lab review |
| `failed_predictions` | **append** | predictions that failed |
| `parents` | replace | this round's Revise candidates: next round improves these |
| `iteration`, `stale_iterations` | replace | progress counters |
| `active_seconds` | **add up** | seconds of actual work, for the time budget (each step adds its own duration) |
| `status`, `stop_reason` | replace | how the run ended |
| `ranked` | replace | **final output**: the shortlist, best first |
| `near_misses` | replace | **final output**: near misses, closest first (NM-3) |

---

## 3. `interfaces.py`: the five tool slots

```python
class CandidateGenerator(Protocol):
    def generate_validate(self, packet: ConstraintPacket) -> GenerationResult: ...
class PropertyPredictor(Protocol):
    def predict(self, candidates: list[Candidate]) -> list[Prediction]: ...   # one per candidate, same order
class DecisionGate(Protocol):
    def decide(self, candidates, config, policy, iterations_remain) -> list[Decision]: ...
class Ranker(Protocol):
    def rank(self, accepted: list[Candidate]) -> list[Candidate]: ...
```

**`ConstraintPacket`** is what the generator gets each round (D2): run ID,
iteration, target type, partner, constraints, **parents**, candidate limit and
seed. *Difference from D2:* D2 passes only the parents' IDs (the real generator
can look them up in the shared database). We pass the full parent candidates,
which also carry their machine-readable violations. That's the first piece of
the advisor's "structured repair instructions".

**`GenerationResult`** = `valid` + `rejected` lists, because D2 puts
validation inside Jaxon's component.

**`Tools`** bundles all five. Swapping a fake for a real tool means putting it
in that slot. `graph.py` doesn't change.

---

## 4. `near_miss.py`: formulas NM-1 to NM-4

From the advisor's guidance (2026-10-23), confirmed by Shashank (2026-10-26).
Running example: targets AVT ≥ 0.40, PCE ≥ 8%, LUE ≥ 3.0; Molecule A has
AVT 0.38 ± 0.04; Molecule B has AVT 0.30 ± 0.01.

| ID | Function | Formula | Example |
|---|---|---|---|
| **NM-1** | `nm1_fraction_short` | `(minimum − actual) / minimum`, or 0 if it passes | A: (0.40−0.38)/0.40 = **0.05** |
| **NM-2** | `nm2_uncertainties_short` | `(minimum − actual) / uncertainty`, or 0 if it passes; `None` if the uncertainty is unknown | A: 0.02/0.04 = **0.5**; B: 0.10/0.01 = **10** |
| NM-2 cut-off | `compute_near_miss` | **borderline** = misses something, and every miss is ≤ `BORDERLINE_SIGMAS` (1) uncertainty away | A borderline, B not |
| **NM-3** | `nm3_beats` | X beats Y if X is no further short (NM-1) on every target and less short on one. **No weights.** | A beats B |
| **NM-4** | `nm4_made_progress` | a round made progress if it has an Accept, or a near miss that no earlier near miss beats or equals | used by `patience` |

```python
def nm1_fraction_short(actual: float, minimum: float) -> float:
    if actual >= minimum:
        return 0.0
    return (minimum - actual) / minimum
```

**What counts as a near miss** (`is_near_miss`): the gate gave it threshold
violations and the outcome is Revise or Stop. Escalated candidates are not
near misses (their numbers can't be trusted). `rank_near_misses` sorts near
misses into Pareto fronts using NM-3; within a front, by candidate ID (not by
a combined score, which would bring weights back).

To change the borderline cut-off, change the one number `BORDERLINE_SIGMAS`.

---

## 5. `gate.py`: placeholder gate and ranker

Stand-ins for **Arleen's** gate and **Nyla's** ranking, written to D2's rules.
When their real versions exist, they plug into the `gate`/`ranker` slots and
this file can be deleted.

### `PlaceholderGate.decide` (D2 "AG"): rules in order
1. failed prediction → no decision (the graph filters these out first)
2. not trustworthy → **Escalate** (`in_domain` not True, or a policy-listed uncertainty unknown or above its limit)
3. OPV-likeness < `policy.opv_min` → **Stop**
4. misses a minimum → **Revise** if rounds remain, otherwise **Stop**
5. otherwise → **Accept**

Exactly equal to a minimum passes. Missed targets are recorded even when the
outcome is Escalate, so a human reviewer can see them.

```python
violations = self._threshold_violations(cand, config)   # recorded first, whatever the outcome
...
if trust_problems:
    return Decision(..., outcome="Escalate", risk_flags=flags, violations=violations)
```

**Not implemented here:** the SA-score check (the placeholder policy doesn't
require it); the advisor's borderline escalation and repair-route diagnosis
(Arleen's gate decisions; the borderline flag is ready for her in NM-2).

### `PlaceholderRanker.rank` (D2 "A5")
1. Keep the earliest copy of each repeated pair.
2. Pareto fronts on **AVT and LUE** (PCE is a hard floor the gate already enforced).
3. Within a front: higher LUE, then AVT, then PCE, then the pair text.

```python
ranked += sorted(front, key=lambda c: (-c.prediction.values.lue_percent,
                                       -c.prediction.values.avt_fraction,
                                       -c.prediction.values.pce_percent,
                                       c.pair_key))
```
Minus signs turn "highest first" into Python's default "lowest first" sort.

---

## 6. `stubs.py`: fake tools

> **All molecules and numbers are fake.** They exist only so every path runs.

- **`StubGenerator`**: round 1 attaches a random tail to a random seed
  molecule. Later rounds revise each parent with a **local edit**: `_attach`
  picks an atom that has a free hydrogen and attaches a small group (`F`, `C`,
  `OC`, `C#N`, ...) there. Each new candidate records its parent. It does
  **not** validate (Jaxon's validation isn't built yet), so our own `validate`
  step is the safety net. One seed is broken on purpose to exercise that.
- **`StubPredictor`**: more aromatic rings → higher PCE but lower AVT (the
  real trade-off). Bigger molecules are more uncertain and sometimes "out of
  domain". About 1 in 20 molecules returns a simulated failure.

---

## 7. `graph.py`: the loop

`build_graph(tools, checkpointer=None, clock=time.time)` defines the steps as
inner functions (so they can use `tools` and `clock`), wires them, and
compiles. `clock` is replaceable so tests can pretend time has passed.
`checkpointer` is where the state is auto-saved: in memory by default, or on
disk with `storage.make_checkpointer(...)` (section 8). Both use our allow-list.

**Every step is timed** (decision Q3, 2026-11-06: only working time counts):
```python
def timed(step):
    def run_and_time(state):
        began = clock()
        updates = step(state)
        return {**updates, "active_seconds": clock() - began}   # added to the running total
    return run_and_time
...
g.add_node(name, timed(fn))
```
Time *between* steps, such as a machine being off overnight after a crash, is
never added, so a resumed run isn't stopped for downtime.

**The middle steps can be cancelled** (D2: "cancellation is observed at each
checkpoint"). `build_graph(..., should_cancel=...)` takes a function that answers
"has someone asked run X to stop?" (the worker passes `RunStore.is_cancel_requested`).
`retrieve`, `generate`, `validate`, `predict` and `decide` are wrapped by
`cancellable`: once a cancel is seen, they do nothing except record
`cancel_seen: True`, and `_next_move` rule 0 sends the run to `finish` as
`stopped` / `"cancelled"`. `start` and `finish` always run. Work finished
before the cancel is kept.

### The steps (nodes)
| Node | Reads | Writes | What it does |
|---|---|---|---|
| `start` | — | status, iteration, stale_iterations, parents | sets the starting values |
| `retrieve` | objective, config | evidence | calls the retriever |
| `generate` | config, iteration, parents | batch, iteration | builds the `ConstraintPacket`, calls the generator |
| `validate` | batch, history | batch | **our safety net**: RDKit check, builds the pair, rejects repeated pairs |
| `predict` | batch | batch | predicts the valid candidates as one batch |
| `decide` | batch, config, policy | history, accepted, escalated, failed_predictions, parents, stale_iterations | calls the gate, attaches NM-1/NM-2, sorts the piles, applies NM-4 |
| `finish` | everything | status, stop_reason, ranked, near_misses | final status + both rankings |

`validate` in brief:
```python
mol = Chem.MolFromSmiles(cand.generated_smiles)      # None -> "invalid_smiles"
canonical = Chem.MolToSmiles(mol)
donor, acceptor = (canonical, partner) if target_type == "donor" else (partner, canonical)
if cand.pair_key in seen:                            # -> "duplicate_pair"
```

`decide` in brief:
```python
ok = [c for c in batch if c.prediction is not None and c.prediction.status == "ok"]
decisions = tools.gate.decide(ok, cfg, policy, iterations_remain)
# attach decision + compute_near_miss(...) to each, then:
"accepted":  with_outcome("Accept"),     # appended
"escalated": with_outcome("Escalate"),   # appended: the set-aside pile
"parents":   with_outcome("Revise"),     # next round revises these
```
If **no** candidate was valid in a round, `parents` is left unchanged, so the
next round retries them (D2 EC-C3-1).

### How a run ends: `_next_move` (rules in order)
| # | Condition | Result |
|---|---|---|
| 0 | a cancel request was noticed | **stopped** (`cancelled`) |
| 1 | valid candidates existed but every prediction failed | **failed** |
| 2 | optional `target_count` reached | **completed** |
| 3 | working-time budget used up (`active_seconds` ≥ `policy.max_runtime_seconds`) | **stopped** |
| 4 | optional `patience` used up (NM-4) | **stopped** |
| 5 | last round reached: candidates still needed revising, or none were valid | **stopped** |
| 5b | last round reached, nothing needed revising | **completed** |
| 6 | no valid candidates this round | another round (retry) |
| 7 | nothing left to revise | **completed** |
| 8 | otherwise | another round |

The router (`route_after_decide`) and `finish` both call `_next_move`, so they
always agree on the reason. Both rankings are computed whatever the status, so
nothing is lost.

---

## 8. `storage.py`: saving runs to disk

Two SQLite database files in one folder (default `runs/`, ignored by git;
decision Q2, 2026-11-06: two files, unlike D2's single store):

| File | Made by | Holds | Used for |
|---|---|---|---|
| `checkpoints.db` | `make_checkpointer(runs_dir)` | LangGraph's auto-save of the whole state after **every step** | resuming a run; reading a run's full results |
| `runs.db` | `RunStore(runs_dir)` | our `runs` table: one row per run, D1's Run fields | quick status lookups and listing (the dashboard, item 5) |

```python
def make_checkpointer(runs_dir=DEFAULT_RUNS_DIR) -> SqliteSaver:
    saver = SqliteSaver(_connect(Path(runs_dir) / CHECKPOINTS_FILE), serde=make_serializer())
    saver.setup()
    return saver
```

- **`make_serializer()`** applies the **allow-list** (`ALLOWED_STATE_CLASSES`):
  the only classes LangGraph may rebuild when reading a checkpoint. Without it,
  LangGraph logs that loading our classes "will be blocked in a future version".
  **If you add a class to the state, add it to this list.**
- **`_connect`** applies D2's SQLite settings: foreign keys on, WAL mode (readers
  aren't blocked by a writing run), and a 5-second busy timeout.
- **`RunStore`**: `create` (status `queued`), `update(run_id, status=..., ...)`
  (only `status`, `iteration`, `stop_reason` and `failure` may change),
  `get(run_id)`, and `list(status=None, limit=20)`, newest first. The table's
  `CHECK` rule means the database itself refuses any status outside D1's five.
- Values are always passed to SQL through `?` placeholders, never pasted into
  the SQL text (that prevents "SQL injection").

**Added for item 5 (2026-11-09):** `runs.db` has three tables:
`runs` (plus the columns `knowledge_snapshot_id`, `model_version`, `cancel_requested`,
`claimed_by`, `heartbeat_at`; an older `runs.db` gets them added automatically, a
"migration"), `receipts` (idempotency, 24 h) and `workers` (last seen).

| Method | Used by | What it does |
|---|---|---|
| `claim_next(worker)` | worker | Claims the oldest queued run in ONE SQL statement that only succeeds if it's still queued |
| `claim_stale(worker, after)` | worker | Takes over a running run whose heartbeat is older than `after` (its worker crashed) |
| `heartbeat(run, worker)` | worker | "Still working on it"; returns False if another worker has taken the run |
| `request_cancel(run)` | API | Sets `cancel_requested`; a still-queued run is stopped at once |
| `create_with_receipt(...)` | API | Creates the run AND its idempotency receipt in one transaction |
| `find_receipt` / `save_receipt` | API | Look up / store the first reply to a POST |
| `worker_seen` / `last_worker_seen` | worker / API | "I'm alive" / "when was a worker last alive?" |

**Still deferred:** D2's full leases with fencing tokens. The heartbeat and
`claimed_by` checks are a lighter version, enough for one worker.

---

## 9. `runner.py`: starting and resuming runs

```python
record = start_run(graph, store, objective=..., config=..., policy=...)   # new run
record = resume_run(graph, store, run_id)                                 # continue a crashed run
```
- **`start_run`**: makes a run_id (a UUID; also the thread_id), records the run
  as `queued` then `running`, runs the graph, and records the final status,
  iteration and reason.
- **`resume_run`**: refuses unknown runs and runs that already finished
  (`completed`/`stopped`). Otherwise it sets the status back to `running` and
  calls `graph.invoke(None, config)`. `None` means "continue from the last
  checkpoint", so finished rounds aren't redone.
- **`run_claimed(graph, store, record, policy, still_owner)`**: what the worker
  calls. If nothing is saved for the run yet, it starts from the beginning using
  the settings stored in the runs table; if something is saved (a stale run
  taken over), it resumes. If `still_owner()` says another worker took the run,
  the result is **not** written.
- **`saved_state(checkpointer, run_id)`**: a run's latest saved state, read
  straight from the checkpoint database without running anything (used by the
  API for the shortlist, the report and live iteration).
- **`_drive`**: if the graph throws **any** error, the run is recorded as
  `failed` with `"ErrorType: message"` in `failure` and the last saved
  iteration, instead of crashing the caller. Retries before giving up come in item 6.

---

## 10. `policies.py`: approved policy versions

A run stores only the **name** of its policy (`policy_version`, e.g.
`"placeholder-0"`). `POLICIES` maps names to `Policy` objects. D2: policies never
change once registered, so new limits get a new name. The API refuses an
unknown name (422), and the worker looks the policy up again when it runs.
Only `PLACEHOLDER_POLICY` exists (no real limits approved yet: PR-06).

---

## 11. `worker.py`: the worker program

Decision Q2 (2026-11-08): runs execute in a separate program, not inside the API.

```python
def process_next_run(self):
    self.store.worker_seen(self.worker_id)                                # "I'm alive"
    record = (self.store.claim_stale(self.worker_id, self.stale_after)    # crashed runs first
              or self.store.claim_next(self.worker_id))                   # then the oldest queued run
    ...
    with Heartbeat(self.store.runs_dir, record.run_id, self.worker_id, self.heartbeat_every) as beat:
        return run_claimed(self.graph, self.store, record, policy, still_owner=lambda: not beat.lost)
```
- **`Heartbeat`**: a background **thread** that calls `store.heartbeat(...)`
  every 20 s (D2) on its **own** database connection. If a heartbeat is refused,
  `lost` becomes True.
- **`run_forever(poll_every)`**: repeats `process_next_run`, waiting `poll_every`
  seconds when there's nothing to do. Prints one line per finished run.
- Timings: heartbeat every 20 s, stale after 60 s (D2's lease numbers), poll every 2 s.

---

## 12. `api.py`: the web API

`create_app(runs_dir, api_token, policies)` builds a FastAPI app. `serve` runs it
with uvicorn on **127.0.0.1 only**.

| Endpoint | Source | Notes |
|---|---|---|
| `POST /api/v1/runs` | D1 | → `202 {run_id, status: "queued", created_at}` + `Location` header. Needs `Idempotency-Key` |
| `GET /api/v1/runs?status=&limit=` | Nyla | newest first; `limit` 1–100 |
| `GET /api/v1/runs/{run_id}` | D1 | D1's Run shape + `stop_reason`, `cancel_requested`; live iteration while running |
| `POST /api/v1/runs/{run_id}/cancel` | D2 | body `{}`; → `202`; ended run → `409 RUN_CLOSED` |
| `GET /api/v1/runs/{run_id}/shortlist?limit=` | D1 | completed runs only, else `409 RESULTS_NOT_READY` |
| `GET /api/v1/runs/{run_id}/report` | Nyla | every candidate in full + id lists + pinned versions + `not_yet_available` |
| `GET /api/v1/health` | ours | `worker_online` = a worker seen in the last 60 s |

**`CreateRunRequest`**: D1's fields plus our additions: `target_type` and
`fixed_partner_smiles` (**required**: a run can't happen without them; by D1's
own rule this is a breaking change, so D1 must be updated), and optional
`objective`, `target_count` and `patience`.

**Errors** are always D1's format (`{"error": {code, message, details}, "request_id"}`):
`ApiError` for our own checks; FastAPI's validation errors become `422` (or
`400 MALFORMED_JSON` when the body isn't JSON); oversized bodies are `413` and
non-JSON content types `415` (middleware); a busy database is `503`; anything
unexpected is `500` without internal details.

**Security**: `HTTPBearer` reads `Authorization: Bearer <token>`, compared with
`secrets.compare_digest` (constant time). The token comes from the
`ORCHESTRATOR_API_TOKEN` environment variable. Declaring it with `HTTPBearer`
also gives the `/docs` page its **Authorize** button.

**Idempotency**: the receipt is looked up first. Same key + same body → the
stored reply; different body → `409 CONFLICT`. A new run and its receipt are
saved in one transaction, so even two identical requests at the same instant
create one run.

Each request gets its **own** `RunStore` connection (`get_store`), closed
afterwards, because the server answers requests on several threads.

---

## 13. `__init__.py` and `__main__.py`

`__init__.py` re-exports the public names (`build_graph`, `Tools`, `RunConfig`,
`Policy`, ...), so callers can write `from orchestrator import ...`.

`__main__.py` is the demo `python -m orchestrator` runs. It designs donors for
**Y6** with the fake tools, the placeholder gate and ranker, and a placeholder
policy (`DEMO_POLICY`). The targets (AVT ≥ 0.5, PCE ≥ 3%, LUE ≥ 2.8, seed 1)
were chosen because, with the fake tools, that run uses every path. It prints
the status, the shortlist with each candidate's parent, the set-aside pile,
and the top near misses.

Options: `--db FOLDER` saves the run (through `runner.start_run`);
`--db FOLDER --list` lists saved runs; `--db FOLDER --resume RUN_ID` continues
one. Asking to resume an unknown or finished run prints a one-line error.

`python -m orchestrator serve --db FOLDER --port 8100` starts the API (it refuses
to start unless `ORCHESTRATOR_API_TOKEN` is set). `python -m orchestrator worker
--db FOLDER` starts the worker with the fake tools and cancellation wired up.

---

## 14. The tests

Run: `python -m pytest orchestrator/tests -v` (from the repo root, langgraph venv). 63 tests.

**`fakes.py`** (shared helpers, not tests):
```python
GOOD   = pred()             # -> Accept
OPAQUE = pred(avt=0.1)      # -> Revise (too dark)
UNSURE = pred(s_avt=0.2)    # -> Escalate (too uncertain)
FAILED = Prediction(status="failed", ...)
class ListGenerator:   # hands out batches the test wrote; records every ConstraintPacket
class TablePredictor:  # looks each molecule's answer up in a dictionary
class FakeClock:       # pretend time: each reading moves it forward by `step`; tests can jump it
def run(batches, table=None, default=GOOD, clock=None, **config_changes): ...
```

| File | What it proves |
|---|---|
| `test_gate.py` | each gate rule (Accept, equality passes, Revise with a machine-readable violation, Stop on the last round, Stop for low OPV-likeness, Escalate for high, missing or out-of-domain uncertainty, Escalate beats Revise); the ranker uses AVT/LUE and removes repeated pairs |
| `test_near_miss.py` | NM-1 and NM-2 numbers from the running example; the borderline flag; NM-3 including a trade-off; NM-4 (first round, closer, equal, worse, Accept); near-miss ranking ignores Accepts and Escalates |
| `test_graph.py` | completes when nothing is left to revise; parents with violations reach the generator and lineage is recorded; stops at the iteration budget; all-failed → failed; time budget (fake clock); escalate sets aside without pausing; the validate safety net catches invalid and repeated molecules; a failed prediction is logged; the optional rules are off by default; `target_count` and `patience` work; an invalid partner is rejected |
| `test_storage.py` | a finished run survives a restart (brand-new saver, graph and store read it back); reloading logs no "unregistered type" message; a crash mid-run is recorded as `failed` with the error and resume finishes it without redoing round 1; **downtime doesn't count** against the time budget; resume refuses finished and unknown runs; the runs table creates, updates, gets and lists runs and refuses bad columns, unknown runs and invalid statuses |
| `test_worker.py` | nothing queued → nothing done; a queued run is claimed and run to the end; oldest first; unknown policy → failed; **cancel during round 2** stops before the next step and keeps round 1; cancelling a queued run needs no worker; a **stale run is taken over and resumed** without redoing round 1; the heartbeat notices a takeover; a worker that lost the run doesn't overwrite its status |
| `test_api.py` | token required (401); create → 202 + Location; same key + same body → same run (only one created); same key + different body → 409; missing key → 422; six invalid bodies → 422 naming the field; not JSON → 400; wrong content type → 415; unknown run → 404, malformed ID → 422; list with filters; results not ready while queued → 409; **full lifecycle** API → worker → shortlist, report, health; cancel → stopped/cancelled, repeat with same key → same reply, new cancel → 409 RUN_CLOSED; cancel body must be `{}` |

---

## 15. One run, step by step

Using `test_revise_rounds_pass_parents_and_stop_at_the_iteration_budget`
(batches `["c1ccccc1"]`, `["Cc1ccccc1"]`, `["CCc1ccccc1"]`; every prediction
is OPAQUE, i.e. AVT 0.1 < 0.4; `max_iterations` 3):

1. `start`: status running, iteration 0, no parents.
2. `generate` (round 1): the packet has **no parents**, and the generator returns `it1-c01` (benzene).
3. `validate`: it's valid, so the pair becomes `c1ccccc1>>c1ccsc1`.
4. `predict`: OPAQUE.
5. `decide`: the gate says **Revise** with violation `avt_fraction 0.1 < 0.4`. NM-1 = 0.75 short. `parents` = [it1-c01].
6. `_next_move`: rounds remain and there's something to revise → `generate`.
7. Round 2: the packet's parents = [it1-c01] with its violation. The new candidate `it2-c01` records `parent_candidate_id="it1-c01"`. Again Revise.
8. Round 3: `iterations_remain` is now False, so the gate says **Stop** (missed a target, no rounds left).
9. `_next_move` rule 5: last round, and a candidate still needed revising → **stopped**, "iteration budget (3) exhausted".
10. `finish`: `ranked` = [] (nothing accepted), `near_misses` = all three, kept as the advisor asked.
