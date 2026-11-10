# Transparent-OPV Orchestrator (LangGraph)

The orchestration engine (component **C2**, owner Shashank Sathiyanarayanan) for
the OMID-sponsored transparent organic photovoltaic (OPV) capstone. It runs an
iterative **generate → validate → predict → decide** loop that searches for one
half of a donor:acceptor pair against a fixed partner molecule, revises the
near misses, sets aside uncertain candidates for DFT/lab review, and returns a
ranked shortlist. It follows the team design documents D1 and D2.

> **Status (2026-11-10):** the loop follows the D1 data shapes and D2 run
> behaviour; runs are **saved to disk and resumable**; and there's a **web API**
> (D1/D2 endpoints plus Nyla's list and report) with a separate **worker
> program** that runs queued runs. Tool calls are **retried** by D2's rules,
> failures get **specific codes**, and every run leaves an **audit trail**. The
> retriever, generator and predictor are **fakes that produce made-up numbers**,
> and the gate and ranker are **placeholders** written to D2's rules. Items 4–6
> are done. SynSpace (ChemCrow's ModifyMol) was tested and found unsuitable for
> OPV near-miss repair; next is re-implementing useful ChemCrow screens. See
> [docs/progress.md](docs/progress.md).

## Documentation

| Doc | What it covers |
|---|---|
| This README | What the package is, how to set up and run it |
| [docs/HANDOFF.md](docs/HANDOFF.md) | **Continuing in a new session?** Project state, decisions, open questions and next steps on one page |
| [docs/running.md](docs/running.md) | **Step by step:** run the tests, demo, API + worker and the SynSpace check, and how to read and explain every line of output |
| [docs/primer.md](docs/primer.md) | **Start here if you're new:** chemistry words, Python features, LangGraph, and tests explained from zero |
| [docs/spec-alignment.md](docs/spec-alignment.md) | What the team design docs (D0/D1/D2) require from the orchestrator, and where the code stands |
| [docs/chemcrow-tools.md](docs/chemcrow-tools.md) | ChemCrow's 19 tools: what each does, whether it fits, and where it would plug into the loop |
| [docs/code-guide.md](docs/code-guide.md) | File-by-file walkthrough of the code, with snippets |
| [docs/progress.md](docs/progress.md) | Dated log of what we've done, decisions, open questions |

## The loop at a glance

```
START → start → retrieve → generate → validate → predict → decide ─┬─ (Revise candidates left) → generate
                                                                   └─ (done / limit reached)   → finish → END
```

Each candidate gets one verdict from the gate:

| Verdict | Meaning | What happens |
|---|---|---|
| **Accept** | Passes every target, trustworthy prediction | Goes to the shortlist |
| **Revise** | Trustworthy, but misses a target (a *near miss*) | Next round revises it (it becomes a *parent*) |
| **Escalate** | Prediction can't be trusted | **Set aside** for DFT/lab review; the run continues |
| **Stop** | Not an OPV-like molecule, or missed a target on the last round | Its path ends |

## Package layout

```
orchestrator/
├── __init__.py        # re-exports the public API (build_graph, Tools, RunConfig, ...)
├── __main__.py        # demo: `python -m orchestrator`
├── state.py           # data shapes (D1): RunConfig, Policy, Prediction, Decision, Candidate, OPVState
├── interfaces.py      # contracts for the 5 tool slots + ConstraintPacket
├── near_miss.py       # the advisor's near-miss formulas NM-1 .. NM-4
├── gate.py            # PLACEHOLDER gate (Arleen's) and ranker (Nyla's), following D2
├── stubs.py           # FAKE retriever, generator and predictor
├── graph.py           # the LangGraph workflow (nodes, routing, end conditions)
├── storage.py         # saving to disk: checkpoint database + runs/receipts/workers tables (SQLite)
├── runner.py          # start/resume/run-claimed runs, read saved results, keep the runs table up to date
├── policies.py        # registry of approved policy versions (only a placeholder so far)
├── worker.py          # the WORKER program: claims queued runs, heartbeats, recovers crashed runs
├── api.py             # the WEB API (FastAPI): D1/D2 endpoints + Nyla's list and report
├── retry.py           # D2's retry rules for every tool call (3 attempts, 1 s / 2 s waits)
├── logs.py            # the operational log: rotating files in runs/logs/
├── requirements.txt   # dependencies for the langgraph venv
├── docs/              # primer, spec alignment, code guide, progress log
└── tests/
    ├── fakes.py           # controllable fake tools shared by the tests
    ├── test_gate.py       # gate and ranker rules
    ├── test_near_miss.py  # formulas NM-1 .. NM-4
    ├── test_graph.py      # the whole loop
    ├── test_storage.py    # saving, reloading after a restart, crash + resume, runs table
    ├── test_worker.py     # claiming, cancelling, stale-run takeover, heartbeats
    ├── test_api.py        # every endpoint, error format, token, idempotency
    ├── test_retry.py      # the retry rules
    └── test_audit.py      # audit trail, failure codes, events endpoint, log files
```

## Setup and running (Windows / PowerShell)

The repo has **two virtual environments**. This package must run in
`.venv-langgraph`. The other one, `.venv-chemcrow`, has the old LangChain that
the original ChemCrow code needs, and it conflicts with LangGraph 1.x.

All commands below are run from the **repo root** (`chemcrow-orchestrator/`),
not from inside `orchestrator/`, because Python must find the `orchestrator`
folder as a package.

### 1. Activate the langgraph venv

```powershell
.\.venv-langgraph\Scripts\Activate.ps1
```

Your prompt should now start with `(.venv-langgraph)`. Confirm which Python is
really active:

```powershell
python -c "import sys; print(sys.prefix)"
# expected: ...\chemcrow-orchestrator\.venv-langgraph
```

> VS Code may auto-activate a different venv in new terminals. If the prompt
> says `(.venv-chemcrow)`, just run the activate command above. It switches over.

### 2. Install dependencies (first time only, or after requirements change)

```powershell
python -m pip install -r orchestrator/requirements.txt
```

### 3. Run the tests

```powershell
python -m pytest orchestrator/tests -v
```

Expected: `82 passed`.

### 4. Run the demo

```powershell
python -m orchestrator
```

It prints the final status, the shortlist (with each candidate's parent), the
set-aside (escalated) candidates, and the top near misses. Expected beginning
(all numbers are FAKE):

```
status: stopped (iteration budget (3) exhausted) after 3 iterations, 24 candidates logged, 0 failed predictions

SHORTLIST (accepted, best first):
  it3-c02  AVT 0.79  LUE  3.42  PCE  4.35%  parent=it2-c02  COc1c(C)ccc(-c2ccc(C#N)s2)c1OC
  ...
```

### 5. Save runs to disk, list them, resume them

```powershell
python -m orchestrator --db runs                 # run once AND save it in the folder runs/
python -m orchestrator --db runs --list          # list saved runs: id, status, iterations, reason
python -m orchestrator --db runs --resume RUN_ID # continue a saved run that crashed
```

`runs/` holds two database files, `checkpoints.db` (LangGraph's auto-save
after every step) and `runs.db` (one row per run). The folder is ignored by git.
Delete it to remove all saved runs. A run that finished normally can't be
resumed; you'll get a one-line message saying so.

### 6. Run the web API and the worker

These are **two programs** that run at the same time, in two terminals, both
pointing at the same runs folder. The API records runs; the worker runs them.

**Terminal 1, the API.** First set the shared secret token (pick your own
value; anyone calling the API must send it):

```powershell
$env:ORCHESTRATOR_API_TOKEN = "choose-a-long-random-secret"
python -m orchestrator serve --db runs --port 8100
```

**Terminal 2, the worker:**

```powershell
python -m orchestrator worker --db runs
```

Then open **http://127.0.0.1:8100/docs** in a browser. It lists every
endpoint and lets you try them. Click **Authorize** (top right) and paste your
token first (just the token; the page adds "Bearer" itself). The server only accepts connections from the same
machine. Stop either program with **Ctrl+C**.

**On the cluster** (Linux), run each one inside `tmux` so it keeps running
after you disconnect:

```bash
tmux new -s orch-api      # then: export ORCHESTRATOR_API_TOKEN=...; python -m orchestrator serve --db ~/runs
                          # detach with Ctrl+B then D; come back with: tmux attach -t orch-api
tmux new -s orch-worker   # then: python -m orchestrator worker --db ~/runs
```

| Endpoint | What it does |
|---|---|
| `POST /api/v1/runs` | Start a run → `202 {run_id, status: "queued"}` (needs an `Idempotency-Key` header) |
| `GET /api/v1/runs?status=&limit=` | List runs, newest first |
| `GET /api/v1/runs/{run_id}` | One run's status, settings, failure |
| `POST /api/v1/runs/{run_id}/cancel` | Stop a run (body `{}`; needs an `Idempotency-Key`) |
| `GET /api/v1/runs/{run_id}/shortlist` | Ranked results (completed runs only) |
| `GET /api/v1/runs/{run_id}/report` | Everything about a run in one bundle |
| `GET /api/v1/runs/{run_id}/events` | The run's audit trail: who did what, page by page (`?limit=&cursor=`) |
| `GET /api/v1/health` | Is a worker running? |

**Log files:** both programs write their messages to `runs/logs/api.log` and
`runs/logs/worker.log` (5 MB each, keeping the last 3, so they never fill the
disk), as well as to the screen.

**When a run fails**, `GET /runs/{id}` shows why in `failure.code`:
`DEPENDENCY_UNAVAILABLE` (a tool kept failing temporarily, all 3 attempts used),
`TOOL_ERROR` (a tool failed permanently), `ALL_PREDICTIONS_FAILED`,
`UNKNOWN_POLICY`, or `INTERNAL_ERROR` (a bug in our code: check the log files).

### 7. The SynSpace (ModifyMol) experiment

A one-off check, outside this package, in `experiments/check_synspace.py`. It
uses its own throwaway venv, `.venv-synspace`. See
[docs/running.md §6](docs/running.md#6-the-synspace-modifymol-check) for how to run it and read its output.

### Without activating anything

You can always call the venv's interpreter by its full path. This works no
matter which venv the terminal has active:

```powershell
.\.venv-langgraph\Scripts\python.exe -m pytest orchestrator/tests -v
.\.venv-langgraph\Scripts\python.exe -m orchestrator
```
