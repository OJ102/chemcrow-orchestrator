# Transparent-OPV Orchestrator (LangGraph)

The orchestration engine (component **C2**, owner Shashank Sathiyanarayanan) for
the OMID-sponsored transparent organic photovoltaic (OPV) capstone. It runs an
iterative **generate → validate → predict → decide** loop that searches for one
half of a donor:acceptor pair against a fixed partner molecule, revises the
near misses, sets aside uncertain candidates for DFT/lab review, and returns a
ranked shortlist. It follows the team design documents D1 and D2.

> **Status (2026-11-06):** the loop follows the D1 data shapes and D2 run
> behaviour, and runs can be **saved to disk and resumed after a crash**. The
> retriever, generator and predictor are **fakes that produce made-up
> numbers**, and the gate and ranker are **placeholders** written to D2's rules.
> Next: the web API (item 5), then retries, failure handling and the audit log
> (item 6). See [docs/progress.md](docs/progress.md).

## Documentation

| Doc | What it covers |
|---|---|
| This README | What the package is, how to set up and run it |
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
├── storage.py         # saving to disk: checkpoint database + our runs table (SQLite)
├── runner.py          # start_run / resume_run: run the graph and keep the runs table up to date
├── requirements.txt   # dependencies for the langgraph venv
├── docs/              # primer, spec alignment, code guide, progress log
└── tests/
    ├── fakes.py           # controllable fake tools shared by the tests
    ├── test_gate.py       # gate and ranker rules
    ├── test_near_miss.py  # formulas NM-1 .. NM-4
    ├── test_graph.py      # the whole loop
    └── test_storage.py    # saving, reloading after a restart, crash + resume, runs table
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

Expected: `35 passed`.

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

### Without activating anything

You can always call the venv's interpreter by its full path. This works no
matter which venv the terminal has active:

```powershell
.\.venv-langgraph\Scripts\python.exe -m pytest orchestrator/tests -v
.\.venv-langgraph\Scripts\python.exe -m orchestrator
```
