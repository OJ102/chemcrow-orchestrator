# Running everything, and reading the output

A hands-on guide: what to type, what you'll see, and how to explain each line
to someone else (teammates, Dr. Rezayat, a demo audience). Every example
output below was captured from a real run on 2026-11-10.

> **Remember: every molecule score here is FAKE.** The retriever, generator
> and predictor are stand-ins (`stubs.py`), and the gate and ranker are
> placeholders. The *mechanics* are real; the *chemistry numbers* aren't.
> Say this first whenever you show the output to anyone.

Contents
1. [Before you start](#1-before-you-start)
2. [The tests](#2-the-tests)
3. [The demo run](#3-the-demo-run)
4. [Saving, listing and resuming runs](#4-saving-listing-and-resuming-runs)
5. [The web API and the worker](#5-the-web-api-and-the-worker)
6. [The SynSpace (ModifyMol) check](#6-the-synspace-modifymol-check)
7. [Cheat sheet: words you'll see in the output](#7-cheat-sheet-words-youll-see-in-the-output)

---

## 1. Before you start

Open PowerShell **in the repo root** (`chemcrow-orchestrator\`, the folder
that contains `orchestrator\`). Every command below assumes that.

There are **three** virtual environments (separate sets of installed
packages). Use the right one for each job:

| Venv | Used for | Why separate |
|---|---|---|
| `.venv-langgraph` | the orchestrator: tests, demo, API, worker | the main project |
| `.venv-synspace` | only the SynSpace check (section 6) | throwaway; keeps SynSpace's extra packages away from the orchestrator |
| `.venv-chemcrow` | the old ChemCrow code | old LangChain clashes with LangGraph |

The commands below call each venv's Python by its full path
(`.\.venv-langgraph\Scripts\python.exe`), so they work no matter which venv
VS Code activated for you.

---

## 2. The tests

**What they are:** 82 small automatic programs that each check one promise
the code makes, e.g. "a cancelled run ends as `stopped`" or "the API rejects
a request with no token". They use fake tools, take a few seconds, and
need no internet.

```powershell
.\.venv-langgraph\Scripts\python.exe -m pytest orchestrator/tests -q
```

Output:

```
........................................................................ [ 87%]
..........                                                               [100%]
82 passed in 3.63s
```

**How to read it:**
- Each `.` is one test that passed. An `F` is a test that **failed** (the code
  broke a promise); an `E` is a test that crashed while setting up. The
  `[ 87%]` is just progress.
- The last line is the verdict. Anything other than `82 passed` means
  something changed. pytest then prints, for each failure, the test's name, the
  line that failed, and what it expected versus what it got.
- Use `-v` instead of `-q` to see every test's name with PASSED/FAILED. Use
  `-k audit` to run only tests whose name contains "audit".

**How to explain it:** *"These are automated checks of the orchestration
logic: the loop, saving and resuming, the API, retries and the audit
trail. All 82 pass. They use fake chemistry tools, so they test the plumbing,
not the science."*

---

## 3. The demo run

**What it is:** one complete search, in memory, with the fake tools. Goal:
*find a transparent donor molecule that pairs with Y6* (Y6 is a famous real
acceptor, used here as the fixed partner). 8 candidates per round, at most 3
rounds. Targets: AVT ≥ 0.50, PCE ≥ 3.0 %, LUE ≥ 2.8 %.

```powershell
.\.venv-langgraph\Scripts\python.exe -m orchestrator
```

Output (molecules shortened here):

```
run c88086e8-b5f5-40b1-b691-d055ac4497ae
status: stopped (iteration budget (3) exhausted) after 3 iterations, 24 candidates logged, 0 failed predictions

SHORTLIST (accepted, best first):
  it3-c02  AVT 0.79  LUE  3.42  PCE  4.35%  parent=it2-c02  COc1c(C)ccc(-c2ccc(C#N)s2)c1OC
  it1-c03  AVT 0.67  LUE  3.29  PCE  4.88%  parent=None  Fc1cccc2c1[nH]c1ccccc12
  ...

SET ASIDE for DFT/lab review (escalated):
  it3-c01  flags=['HIGH_UNCERTAINTY']  CCCCCCc1csc(...)c1
  it3-c05  flags=['OUT_OF_DOMAIN', 'HIGH_UNCERTAINTY']  CCCCCCc1ccc(...)s1

NEAR MISSES (closest first, formula NM-3; top 5):
  it1-c04  Revise  borderline=True  lue_percent -2% (0.2 sd)
  it2-c02  Revise  borderline=False  lue_percent -9% (1.1 sd)
  ...
```

**Line by line:**

| Output | Meaning |
|---|---|
| `run c88086e8-...` | The run's unique ID (random each time). The same ID names its save slot and its API record. |
| `status: stopped` | How the run ended: `completed` (finished normally), `stopped` (hit a limit or was cancelled), `failed` (an error; see §5). |
| `(iteration budget (3) exhausted)` | The **stop reason**: it used all 3 rounds while still having near misses it could have kept revising. |
| `after 3 iterations` | Rounds of generate → validate → predict → decide. |
| `24 candidates logged` | Every molecule tried, across all rounds (8 × 3). Nothing is thrown away; it's all in the history. |
| `0 failed predictions` | Candidates the predictor couldn't score (none here). |

**SHORTLIST**: the molecules the gate **accepted**, ranked best first:

| Column | Meaning |
|---|---|
| `it3-c02` | Candidate ID: **i**teration **3**, **c**andidate **02**. |
| `AVT 0.79` | Average Visible Transmittance: 79 % of visible light passes through. Higher = more see-through. Target ≥ 0.50. |
| `PCE 4.35%` | Power Conversion Efficiency: 4.35 % of sunlight energy becomes electricity. Target ≥ 3.0 %. |
| `LUE 3.42` | Light Utilization Efficiency = AVT × PCE (in %). The single "transparent *and* useful" score. Target ≥ 2.8. |
| `parent=it2-c02` | It was made by **revising** near miss `it2-c02` from round 2. `None` = a brand-new molecule from round 1. Follow the parents to see how a molecule improved over rounds. |
| `COc1c(...)` | The donor molecule as SMILES (a text spelling of a molecule). |

**SET ASIDE (escalated)**: molecules whose prediction **can't be trusted**,
so instead of guessing, the run parks them for a human expert/DFT/lab check
and carries on:
- `HIGH_UNCERTAINTY`: the predictor's error bar is too wide to tell pass from fail.
- `OUT_OF_DOMAIN`: the molecule is unlike anything the model was trained on, so
  even a confident-looking prediction is unreliable.

**NEAR MISSES**: trustworthy predictions that **just** missed a target,
closest first (formula NM-3 in `near_miss.py`):

| Column | Meaning |
|---|---|
| `Revise` / `Stop` | `Revise` = sent back to the generator to be improved next round. `Stop` = it missed on the last round, so its path ends. |
| `lue_percent -2%` | Its LUE is 2 % **short** of the LUE target (only missed targets are shown). |
| `(0.2 sd)` | That shortfall measured in the prediction's own uncertainty (standard deviations). 0.2 sd = well within the error bar. |
| `borderline=True` | Short by **less than 1 sd** (`BORDERLINE_SIGMAS`): given the uncertainty it might actually pass, so it's a strong candidate for a closer look. `False` = clearly short. |

**How to explain it:** *"The loop proposed 24 donor molecules for Y6 over
3 rounds. Seven passed every target and are ranked on the shortlist; some
of them were made by improving earlier near misses (see the parent
column). Two had untrustworthy predictions, so they were set aside for expert
review instead of guessed at. The near-miss list shows the ones that fell
just short, measured against their own uncertainty. All numbers are fake;
the point is the decision logic."*

---

## 4. Saving, listing and resuming runs

```powershell
.\.venv-langgraph\Scripts\python.exe -m orchestrator --db runs                 # run once and SAVE it in runs\
.\.venv-langgraph\Scripts\python.exe -m orchestrator --db runs --list          # list saved runs
.\.venv-langgraph\Scripts\python.exe -m orchestrator --db runs --resume RUN_ID # continue a run that crashed
```

- `--db runs` prints the same output as §3, plus `saved in runs: status stopped`.
  The folder `runs\` gets `checkpoints.db` (LangGraph's auto-save after every
  step) and `runs.db` (one row per run, plus the audit trail).
- `--list` prints one line per run: `ID  status  iterations=N  created=...  reason`.
- `--resume` on a run that already finished prints a one-line refusal. That's
  expected; only crashed (half-finished) runs can be resumed.
- Delete the `runs\` folder to wipe all saved runs. Git ignores it.

**How to explain it:** *"Every step is checkpointed to disk, so if the machine
or program dies mid-run, we resume from the last step instead of starting over."*

---

## 5. The web API and the worker

**What they are:** two programs that run **at the same time**, sharing the
`runs\` folder.
- The **API** (web server) is the front desk. It *records* run requests and answers
  questions about runs. Nyla's dashboard will talk to it.
- The **worker** is the back office. It *picks up* queued runs and runs them.
  It also sends a heartbeat every 20 s so the API knows it's alive.

### 5.1 Start them (two terminals)

**Terminal 1, the API.** Pick any secret token; callers must send it:

```powershell
$env:ORCHESTRATOR_API_TOKEN = "choose-a-long-random-secret"
.\.venv-langgraph\Scripts\python.exe -m orchestrator serve --db runs --port 8100
```

It prints `API docs: http://127.0.0.1:8100/docs` and keeps running.

**Terminal 2, the worker:**

```powershell
.\.venv-langgraph\Scripts\python.exe -m orchestrator worker --db runs
```

Stop either with **Ctrl+C**.

### 5.2 Use it from the browser (easiest)

1. Open **http://127.0.0.1:8100/docs**. This is an auto-generated page listing
   every endpoint, with a "Try it out" button on each.
2. Click **Authorize** (top right), paste your token (without "Bearer"), click
   Authorize, then Close.
3. **Check the worker is alive:** `GET /api/v1/health` → Try it out → Execute.
4. **Start a run:** `POST /api/v1/runs` → Try it out. Put any text (e.g.
   `test-1`) in **Idempotency-Key**, and paste this body:

```json
{
  "knowledge_snapshot_id": "00000000-0000-0000-0000-000000000001",
  "model_version": "fake-0",
  "policy_version": "placeholder-0",
  "constraints": {"avt_min_fraction": 0.5, "pce_min_percent": 3.0, "lue_min_percent": 2.8},
  "candidate_limit": 8,
  "max_iterations": 3,
  "seed": 1,
  "target_type": "donor",
  "fixed_partner_smiles": "CCCCCCCCCCCc1c(C=C2C(=O)c3cc(F)c(F)cc3C2=C(C#N)C#N)sc2c1sc1c3c4nsnc4c4c5sc6c(CCCCCCCCCCC)c(C=C7C(=O)c8cc(F)c(F)cc8C7=C(C#N)C#N)sc6c5n(CC(CC)CCCC)c4c3n(CC(CC)CCCC)c21",
  "objective": "Find a transparent donor for Y6",
  "target_count": 3
}
```

5. Copy the `run_id` from the reply, and use it in `GET /runs/{run_id}`,
   `/shortlist`, `/report` and `/events`.

> **Why `"target_count": 3`?** D1 says only **completed** runs have a
> shortlist. With `target_count` the run finishes as `completed` once 3
> molecules are accepted. Without it, this demo run ends as `stopped`
> (iteration budget used up), and `/shortlist` answers `RESULTS_NOT_READY`.
> Whether that should count as completed is an open team question.

### 5.3 Reading each reply

**`GET /health`**
```json
{"status": "ok", "worker_last_seen": "2026-11-10T15:43:37+00:00", "worker_online": true}
```
`worker_online: false` means no worker heartbeat in the last 60 s: runs will
sit in `queued` until you start one. (Times are UTC, hence `+00:00`.)

**`POST /runs`** → HTTP **202 Accepted** (not 200: "I've recorded it, the work happens later")
```json
{"run_id": "6d443ac8-...", "status": "queued", "created_at": "2026-11-10T15:44:00+00:00"}
```
The `Location` header points to the new run. Sending the **same
Idempotency-Key** again returns the same run instead of creating a duplicate
(protection against double-clicks and network retries).

**`GET /runs/{id}`**, the run's status card:
```json
{"run_id": "6d443ac8-...", "status": "completed", "iteration": 1,
 "config": { ...exactly what you sent... },
 "failure": null, "stop_reason": "target count of 3 accepted candidates reached",
 "cancel_requested": false}
```
- `status` moves `queued` → `running` → `completed` / `stopped` / `failed`.
- `failure` is `null` unless the run failed; then it has a `code`:
  `DEPENDENCY_UNAVAILABLE` (a tool kept failing temporarily, all 3 retries
  used), `TOOL_ERROR` (a tool failed permanently), `ALL_PREDICTIONS_FAILED`,
  `UNKNOWN_POLICY`, `INTERNAL_ERROR` (our bug: check `runs\logs\worker.log`).

**`GET /runs`** is the same cards for many runs, newest first (`?status=completed&limit=5` to filter).

**`GET /runs/{id}/shortlist`**, the ranked results (one item shown):
```json
{"rank": 1, "candidate_id": "it1-c03",
 "donor_smiles": "Fc1cccc2c1[nH]c1ccccc12", "acceptor_smiles": "<Y6>",
 "values":      {"pce_percent": 4.88, "avt_fraction": 0.674, "lue_percent": 3.29, "donor_homo_eV": -5.15, ...},
 "uncertainty": {"pce_percent": 0.3,  "avt_fraction": 0.02,  "lue_percent": 0.22, ...},
 "in_domain": true,
 "rationale": "Accepted: meets every minimum with trustworthy predictions.",
 "risk_flags": [], "parent_candidate_id": null}
```
- `values` ± `uncertainty`: read PCE as "4.88 % ± 0.3 %" (one standard deviation).
- `donor_homo_eV` / `acceptor_lumo_eV`: energy levels in electron-volts that
  must line up for a donor and acceptor to work together.
- `in_domain`: whether the model has seen similar molecules (trust).
- `rationale`: the gate's plain-English reason; `risk_flags` its warnings.

**`GET /runs/{id}/report`** is everything in one bundle, for Nyla's dashboard:
`run`, `pinned_versions` (exact data/model/policy versions used, for
reproducibility), `stop_reason`, `active_seconds` (working time only),
the ID lists `shortlist_ids` / `escalated_ids` / `near_miss_ids` /
`failed_prediction_ids`, every `candidates` entry, the `events`, and
`not_yet_available` (parts of D1 we can't fill in yet).

**`GET /runs/{id}/events`** is the **audit trail**: who did what, in order:
```json
{"seq": 1, "actor_id": "api",                  "action": "run_created",  "result_code": "ok"}
{"seq": 2, "actor_id": "worker:Abyss-25828-…", "action": "run_claimed",  "result_code": "ok"}
{"seq": 3, "actor_id": "worker:Abyss-25828-…", "action": "run_started",  "result_code": "ok"}
{"seq": 4, "actor_id": "worker:Abyss-25828-…", "action": "iteration_finished", "phase": "decide",
 "detail": "round 1: 8 candidates: 3 Accept, 2 Revise, 0 Escalate, 1 Stop, 2 invalid, 0 failed predictions"}
```
`next_cursor: "4"` means there may be more: call again with `?cursor=4`.
Retried tool calls show up here too (with `attempt`). No molecules are written
to the audit trail on purpose.

**Errors** always have the same shape (D1):
```json
{"error": {"code": "UNAUTHENTICATED", "message": "Missing or invalid API token", "details": []},
 "request_id": "a33f85e8-..."}
```
Common codes: `UNAUTHENTICATED` (no or wrong token: re-do Authorize),
`VALIDATION_ERROR` (bad body: `details` says which field), `NOT_FOUND`,
`RESULTS_NOT_READY`. Quote the `request_id` when reporting a problem; it's
in the log files too (`runs\logs\api.log`, `runs\logs\worker.log`).

**How to explain it:** *"The dashboard submits a run and gets a ticket back
straight away (202). A separate worker process does the work, so the API
stays responsive and a crash doesn't lose anything. You can poll the run's
status, fetch the ranked shortlist with uncertainties and the gate's reasons,
or pull the full audit trail of who did what."*

---

## 6. The SynSpace (ModifyMol) check

**What it is:** a one-off experiment, not part of the orchestrator. It
tests whether SynSpace, the library behind ChemCrow's "ModifyMol" tool, can
make small, realistic variants of our molecules, so we could use it to
repair near misses. Script: [`experiments/check_synspace.py`](../../experiments/check_synspace.py).
Conclusion (2026-11-10): **not suitable**; see [progress.md](progress.md).

### 6.1 One-time setup (already done on this PC)

```powershell
py -3.10 -m venv .venv-synspace
.\.venv-synspace\Scripts\python.exe -m pip install synspace
```

### 6.2 Run it

It needs Kyle's `classified_opv_database.csv` (columns `molecule_name`,
`canonical_smiles`, `role`). Pass its path:

```powershell
.\.venv-synspace\Scripts\python.exe experiments\check_synspace.py C:\Users\abyss\Downloads\classified_opv_database.csv
```

It takes about 2 minutes (12 molecules × 2 runs × ~5 s). Full results are saved
to `experiments\output\synspace_results.json` (git-ignored).

### 6.3 Reading the output

```
Testing 12 molecules, time limit 180s per call

Y6             don  101 atoms filter_on  ok          4.2s variants=  0 valid=  0 sim(min/med/max)=None/None/None
Y6             don  101 atoms filter_off ok          4.2s variants=  0 valid=  0 sim(min/med/max)=None/None/None
                   original REOS verdict: ('Glaxo', 'I1 Aliphatic methylene chains 7 or more long')

PNEHD          acc   68 atoms filter_on  ok          6.1s variants=  0 valid=  0 sim(min/med/max)=None/None/None
PNEHD          acc   68 atoms filter_off ok          5.7s variants=249 valid=249 sim(min/med/max)=0.52/0.57/0.93
                   original REOS verdict: ('Glaxo', 'I1 Aliphatic methylene chains 7 or more long')
```

| Column | Meaning |
|---|---|
| `Y6` / `don` / `101 atoms` | Molecule name, role from Kyle's file (`don`or / `acc`eptor), size in heavy (non-hydrogen) atoms. |
| `filter_on` / `filter_off` | Whether SynSpace's **REOS** filter was used. REOS is a *drug-chemistry* rule list that throws out molecules with "undesirable" groups. |
| `ok` / `crash` / `timeout` | Did the call finish? (`timeout` = longer than 180 s.) |
| `4.2s` | How long it took. |
| `variants=249` | How many **new** molecules it proposed (the original isn't counted; 249 is near SynSpace's cap of 250). |
| `valid=249` | How many of those RDKit can read back as real molecules. |
| `sim(min/med/max)` | How similar the variants are to the original (Tanimoto similarity: 1.0 = identical, 0 = nothing in common). 0.5–0.9 = "close relatives". `None` = no variants. |
| `original REOS verdict` | What REOS says about the **starting** molecule. `('ok','ok')` = passes. `'Aliphatic methylene chains 7 or more long'` = rejected for a long carbon chain, which nearly every OPV molecule has, so REOS must be off for us. |

The JSON file also lists, per molecule, the **reactions** used (e.g.
`retro-Mitsunobu imide;Mitsunobu imide`) and 5 example variants. Comparing an
example's SMILES with the original shows *what* changed (in our test: only side
chains, never the light-absorbing core).

**How to explain it:** *"We tested ChemCrow's molecule-modification
library on 12 real OPV molecules from Kyle's data. For the most important
acceptors and donors (Y6, ITIC, IT-4F, L8-BO, D18) it produced nothing, because
its reaction rules are built for drug molecules. When it did produce variants,
it only swapped side chains, sometimes broke the core ring, and its default
filter rejects normal OPV side chains. So it's not a good near-miss repair
tool for us; repair needs OPV-specific edits, which fits Jaxon's MatGPT."*

---

## 7. Cheat sheet: words you'll see in the output

| Word | Plain meaning |
|---|---|
| Donor / acceptor | The two molecules in a solar-cell layer: the donor gives up electrons, the acceptor takes them. A run designs one side against a **fixed partner**. |
| SMILES | A one-line text spelling of a molecule, e.g. `c1ccccc1` = benzene. |
| AVT | Average Visible Transmittance (0–1): how see-through. |
| PCE | Power Conversion Efficiency (%): how much sunlight becomes electricity. |
| LUE | Light Utilization Efficiency = AVT × PCE: the combined score. |
| Uncertainty / sd | The predictor's error bar (one standard deviation). |
| In domain | The model has seen similar molecules, so its prediction is more trustworthy. |
| Accept / Revise / Escalate / Stop | Gate verdicts: shortlist / improve next round / set aside for experts / end this path. |
| Near miss / borderline | Missed a target, but trustworthy; borderline = missed by less than 1 sd. |
| Iteration / round | One pass of generate → validate → predict → decide. |
| Checkpoint | Auto-save after every step, so a crashed run can resume. |
| Idempotency-Key | A label for a request so repeating it doesn't create a duplicate. |
| Audit trail | Permanent record of who did what to each run. |
| Tanimoto similarity | 0–1 score of how alike two molecules are (1 = identical). |
| REOS | A drug-industry filter for "undesirable" chemical groups (wrong for OPV). |
