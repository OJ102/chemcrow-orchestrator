# Primer: The Basics You Need Before Reading the Code

This assumes you know **no chemistry** and are **new to Python's advanced
features and to LangGraph**. Read it once, top to bottom. The code comments
point back here for the longer explanations.

Contents:
1. [What the project is trying to do](#1-what-the-project-is-trying-to-do)
2. [Chemistry words](#2-chemistry-words)
3. [Python features used in the code](#3-python-features-used-in-the-code)
4. [Saving to disk: databases in plain language](#4-saving-to-disk-databases-in-plain-language)
5. [LangGraph in plain language](#5-langgraph-in-plain-language)
6. [How tests work (pytest)](#6-how-tests-work-pytest)
7. [Web APIs and the worker in plain language](#7-web-apis-and-the-worker-in-plain-language)

---

## 1. What the project is trying to do

Solar panels on a farm block sunlight the plants need. The sponsor wants
**see-through solar panels** made from carbon-based ("organic") materials.
Light that plants use passes through, and the rest becomes electricity.

These panels need **two kinds of molecules mixed together**: a *donor* and an
*acceptor* (explained below). Nobody knows which molecules work best, and
making each one in a lab is slow and expensive. So the plan is:

1. A program **invents** candidate molecules (MatGPT, Student 3).
2. Another program **predicts** how good each one would be (Chemprop, Student 4).
3. **Our code (the orchestrator)** runs this as a loop: invent → check →
   predict → decide → invent better ones → … until we have a short list of
   good candidates worth making in a real lab.

Think of the orchestrator as a **manager**. It doesn't invent or predict
anything itself. It calls the tools in the right order, keeps records, and
decides when to stop.

---

## 2. Chemistry words

You don't need to understand the chemistry deeply. You only need to know
what each word *means* as a piece of data.

| Word | Plain meaning | In the code |
|---|---|---|
| **Molecule** | A group of atoms stuck together, e.g. water = 2 hydrogen + 1 oxygen. | — |
| **SMILES** | A way to write a molecule as a line of text, so computers can store it. `C` = methane (one carbon atom), `CCO` = ethanol (alcohol), `c1ccccc1` = benzene (a ring of 6 carbons). Lowercase letters mean the atom is part of a special flat ring ("aromatic"). Numbers mark where a ring opens and closes. | Every molecule is a `str` like `"CCO"`. |
| **Valid / invalid SMILES** | Like a sentence with a grammar error, some strings aren't real molecules, e.g. `"not_a_smiles"`. The generator sometimes produces broken ones. | `valid` field, `invalid_smiles` violation |
| **Canonical SMILES** | One molecule can be written many ways. `CCO` and `OCC` are both ethanol, read from opposite ends. "Canonical" means the one official spelling, so we can tell two strings describe the same molecule. | `canonical_smiles` field |
| **RDKit** | A free Python library that understands chemistry: it reads SMILES, checks validity, counts rings, and so on. | `from rdkit import Chem` |
| **Donor (D) / Acceptor (A)** | The two molecule types in the solar cell. When light hits, the donor gives away an electron and the acceptor takes it. That moving electron is the electricity. You need *both*, and the *pair* matters, like two puzzle pieces that must fit. | `target_type = "donor"` or `"acceptor"` |
| **Fixed partner** | We design one piece at a time. Each run keeps one molecule fixed (e.g. a well-known acceptor called **Y6**) and searches for the best partner for it. | `fixed_partner_smiles` |
| **Pair** | One donor + one acceptor together. The team's documents treat the *pair* as the unit everything is about. "Ordered" means donor first: swapping them is a different pair. | `Candidate.donor_smiles`, `acceptor_smiles`, `pair_key` (`"donor>>acceptor"`) |
| **PCE** (Power Conversion Efficiency) | What percentage of sunlight energy becomes electricity. Higher is better. Normal silicon panels are ~20%. These organic ones are lower. | `pce_percent` (`8.0` means 8%) |
| **AVT** (Average Visible Transmittance) | How see-through the material is, from 0 (black, blocks all visible light) to 1 (perfectly clear glass). Higher means more transparent. | `avt_fraction` (`0.6` = 60% of visible light passes through) |
| **The trade-off** | Absorbing more light usually makes *more electricity* (higher PCE) but makes the panel *darker* (lower AVT). We want both, and that's the hard part. | — |
| **LUE** (Light Utilization Efficiency) | One number that combines both: **LUE = AVT × PCE**. A panel at 8% PCE and 0.6 AVT has LUE = 4.8. A panel at 15% PCE that is nearly black (AVT 0.05) has LUE 0.75, so it's worse for our purpose even though it makes more power. **LUE rewards being efficient AND transparent at the same time.** | `lue_percent` |
| **HOMO / LUMO** | Two energy levels of a molecule, measured in **eV** (electron-volts, a tiny unit of energy). For the donor→acceptor electron hand-off to work, the two molecules' levels must be arranged correctly, like water only flowing downhill. The numbers are usually negative, e.g. HOMO = -5.4 eV. D1 predicts the donor's HOMO and the acceptor's LUMO. | `donor_homo_eV`, `acceptor_lumo_eV` |
| **OPV-likeness** | A score from 0 to 1 for "does this look like a real solar-cell molecule?". Some invented molecules are valid chemistry but nothing like a solar material. | `opv_likeness` |
| **Uncertainty / standard deviation** | How unsure the predictor is about each number. It's given as one *standard deviation*: roughly, the true value is usually within ± that amount. AVT 0.38 ± 0.04 means "probably between 0.34 and 0.42". "Unknown" (None) is **not** the same as 0 ("perfectly sure"). | `Uncertainty` (one per value) |
| **In domain** | Whether a molecule resembles what the predictor was trained on. If not ("out of domain"), its numbers are guesses. | `Prediction.in_domain` |
| **DFT** | A slow but much more accurate physics simulation. It's what you run on a promising-but-uncertain molecule to double-check it. | the Escalate verdict sends candidates here |
| **Near miss** | A candidate with trustworthy numbers that *just* missed a target. The advisor said to keep these and improve them. | `near_miss.py`, `Candidate.near_miss` |
| **Borderline** | A near miss so close that its uncertainty could flip it to passing (within 1 standard deviation). | `NearMiss.borderline` (formula NM-2) |
| **Lineage / parent** | Which earlier candidate a new one was revised *from*: its family tree. | `Candidate.parent_candidate_id` |
| **Pareto / dominates** | A fair way to compare things with several goals without adding them up. X *dominates* Y if X is at least as good on every goal and better on at least one. The *Pareto front* is everything nothing dominates. | `gate.py` ranker, `near_miss.py` NM-3 |

---

## 3. Python features used in the code

### Imports
```python
from rdkit import Chem        # load the "Chem" part of the RDKit library so we can use it
from .state import Candidate  # the "." means "from this same package (folder)"
```

### Keyword arguments
```python
RunConstraints(avt_min_fraction=0.4, pce_min_percent=2.0, lue_min_percent=0.8)
```
This **creates a `RunConstraints` object**. The `name=value` pairs are *keyword
arguments*: you name which field each value goes into. In bigger classes,
fields you don't mention keep their default values (e.g. `RunConfig`'s
`candidate_limit` stays `20`).

### Classes and objects
A **class** is a blueprint. An **object** (or *instance*) is one thing built
from that blueprint. `Candidate` is the blueprint for "a pair we're
considering". `Candidate(candidate_id="it1-c01", iteration=1, generated_smiles="CCO")`
is one specific candidate. Access its data with a dot: `cand.generated_smiles`.

### Type hints
```python
def nm1_fraction_short(actual: float, minimum: float) -> float:
```
`actual: float` means "this parameter should be a decimal number", and
`-> float` means "this function returns one". Python doesn't enforce them at
runtime. They are documentation, and editors use them for checking and
autocomplete. `list[str]` = a list of strings. `dict[str, float]` = a
dictionary from text to numbers. `Optional[X]` = an `X` or `None`.
`tuple[str, str, str]` = exactly three strings grouped together.

### pydantic `BaseModel` and `Field`
A class that *does* check its values. `RunConstraints(avt_min_fraction=1.5, ...)`
raises an error immediately because of `Field(ge=0, le=1)` ("greater than or
equal to 0, less than or equal to 1"). It also gives you `model_copy`:
```python
new = cand.model_copy(update={"valid": True})   # a copy of cand with valid changed; cand itself is untouched
```
`ConfigDict(extra="forbid")` makes unknown fields an error (D1 rejects them).

### Validators: `@field_validator`
```python
@field_validator("fixed_partner_smiles")
@classmethod
def _partner_must_be_a_real_molecule(cls, smiles): ...
```
A function pydantic runs automatically whenever that field is set. If it
raises `ValueError`, the object is rejected. If it returns a value, that value
is stored (we return the canonical spelling).

### `Literal`
```python
Outcome = Literal["Accept", "Revise", "Escalate", "Stop"]
```
A type that only allows those exact strings. `"Acept"` (a typo) gets rejected.

### `@property`
```python
@property
def pair_key(self) -> str:
    return f"{self.donor_smiles}>>{self.acceptor_smiles}"
```
Lets you write `cand.pair_key` (no parentheses) as if it were a stored field,
but it's calculated fresh every time.

### `@dataclass` and `@staticmethod`
`@dataclass` writes the boring `__init__` for you: `Tools(retriever=..., ...)`
just stores the five tools. `@staticmethod` marks a function that lives in a
class but doesn't need `self` (it uses only its own arguments).

### List and dictionary comprehensions
```python
[c for c in batch if c.valid]                       # list of the valid candidates
{c.candidate_id: p for c, p in zip(valid, preds)}   # dictionary: id -> prediction
```
Read the first as "make a list of every `c` in `batch` **where** `c.valid` is
true". It's a one-line loop that builds a list. The `{key: value for ...}`
form builds a dictionary the same way.

### `zip`, `enumerate`, `getattr`
- `zip(a, b)` pairs up two lists item by item: `(a[0], b[0]), (a[1], b[1]), ...`
- `enumerate(xs)` gives `(0, xs[0]), (1, xs[1]), ...`, i.e. each item with its position
- `getattr(values, "avt_fraction")` is the same as `values.avt_fraction`, but the
  field name can come from a variable, so one loop can handle all three targets

### f-strings
```python
f"PCE {v.pce_percent:.2f}%"   # -> "PCE 8.00%"
```
`f"..."` lets you put variables in `{}`. `:.2f` means "2 decimal places".
`:.0%` shows a fraction as a percentage (`0.05` → `5%`).

### Tuples and unpacking
```python
gen, result = run(...)            # run returns 2 things; this splits them into 2 variables
_, status, reason = _next_move(s) # keep the last two, throw away ("_") the first
```

### `lambda`
```python
key=lambda c: c.candidate_id
```
A tiny function without a name. The same as `def key(c): return c.candidate_id`.
Often used to tell `sorted()` what to sort by.

### Regular expressions
```python
CANDIDATE_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
```
A pattern text must match. This one means "1 to 64 characters, each a letter,
digit, `_` or `-`, and nothing else". It's D1's rule for candidate IDs.

### `assert` and `pytest.raises`
```python
assert result["status"] == "completed"
with pytest.raises(ValidationError):
    RunConfig(..., fixed_partner_smiles="not_a_smiles", ...)
```
`assert` means "this must be true; if it isn't, the test fails".
`pytest.raises(X)` means "the code inside MUST raise error X".

### Protocol (duck typing)
"If it walks like a duck and quacks like a duck, it's a duck." A `Protocol`
describes the *methods* something must have. Any object with those methods
is accepted, with no inheritance required.

### UUID
A random ID like `5d290209-71d7-423e-b9b1-e986731f3cbf`, so unlikely to repeat
that it's treated as unique. Each run gets one (`uuid.uuid4()`). It's the run's
ID and also LangGraph's `thread_id`.

### `try` / `except`: handling errors
```python
try:
    final = graph.invoke(inputs, config)
except Exception as error:
    store.update(run_id, status="failed", failure=str(error))
```
Run the `try` block. If anything inside raises an error, jump to `except`
instead of crashing, with the error in the variable `error`. `runner.py`
uses this so a crash is **recorded** as a failed run rather than lost.

### Functions that wrap functions ("wrappers" / "decorators")
```python
def timed(step):
    def run_and_time(state):
        began = clock()
        updates = step(state)
        return {**updates, "active_seconds": clock() - began}
    return run_and_time
```
`timed` takes a function and returns a *new* function that does the same thing
plus something extra, here measuring how long it took. `@field_validator` and
`@property` are the same idea, written with `@`.

### Inheritance: `class X(Y)`
```python
class CrashOnceGenerator(ListGenerator): ...
```
"A `CrashOnceGenerator` is a `ListGenerator`, with these changes". It gets
all of `ListGenerator`'s methods. `super().method(...)` calls the parent's
version from inside the child.

### `**` with dictionaries and arguments
- `def update(self, run_id, **changes)`: collects any `name=value` arguments into a dictionary called `changes`
- `RunRecord(**data)`: the reverse; spreads a dictionary out as `name=value` arguments
- `{**updates, "active_seconds": 2.5}`: a copy of the dictionary with one more key

### `__call__`
A class with a `__call__` method makes objects that can be called like
functions: `clock()` on a `FakeClock` object runs its `__call__`.

---

## 4. Saving to disk: databases in plain language

| Word | Plain meaning |
|---|---|
| **Database** | An organised store of records that a program can search and update safely |
| **SQLite** | A small database kept in **one ordinary file** (e.g. `runs.db`). Python has it built in, so no server is needed |
| **Table** | A grid inside the database: named columns, one row per record. Our `runs` table has one row per run |
| **SQL** | The language for talking to databases: `SELECT * FROM runs WHERE run_id = ?` means "get the row for this run" |
| **`?` placeholders** | Values are passed separately and filled in safely by SQLite. Never paste values into the SQL text: that allows "SQL injection" attacks |
| **Primary key** | The column that uniquely identifies each row (`run_id`) |
| **CHECK constraint** | A rule the database enforces itself. Ours: `status` must be one of D1's five values, so `"paused"` is refused |
| **Transaction** (`with conn:`) | Changes are saved all together, or, if an error happens, none of them are |
| **WAL mode** | A setting that lets readers (the dashboard) read while a run is writing |
| **Checkpoint** | LangGraph's auto-save of the whole state after a step |
| **Resume** | Continue a run from its last checkpoint instead of starting over: `graph.invoke(None, config)` |
| **Allow-list** | The list of our classes LangGraph may rebuild when reading a checkpoint. Anything else is refused, for safety |

---

## 5. LangGraph in plain language

**LangGraph** is a library for building programs as a **flowchart** that
remembers things.

| LangGraph word | Plain meaning | Analogy |
|---|---|---|
| **State** | A dictionary of everything the program knows so far. It's passed from step to step. | A shared notebook that every worker reads and writes in |
| **Node** | One step: a Python function that reads the state and returns *only the changes* it wants to make. | A worker who reads the notebook and writes a few lines |
| **Edge** | An arrow: "after step A, do step B". | The order workers take turns |
| **Conditional edge** | An arrow that picks the next step *based on the state*. | "If the notebook says X, hand it to worker Y, else to Z" |
| **Reducer** | The rule for *how* a node's change is written into the state. Default = **overwrite**. `operator.add` = **append to the list**. | Erasing a line and rewriting it vs adding a new line at the bottom |
| **START / END** | Special markers for where the flowchart begins and ends. | |
| **compile()** | Turns your flowchart description into a runnable program. | |
| **invoke(input, config)** | Runs the program from START until it reaches END. `invoke(None, config)` means "continue this save slot from where it stopped" (resume). | |
| **Checkpointer** | Saves the state after every step. `InMemorySaver` keeps it in memory (lost when the program exits); `SqliteSaver` writes it to a file on disk. | A video-game auto-save |
| **thread_id** | The name of one save slot. Same `thread_id` = continue that same saved game. We use the run's ID. | Save slot "Game 1" |
| **get_state(config)** | Reads the latest saved state of a save slot. | Loading a saved game to look at it |

LangGraph can also *pause* a run to ask a human (`interrupt()`). We used to do
that for uncertain molecules, but the team spec (D2) says to **set them aside
and keep going** instead, so the current code never pauses.

Our flowchart:
```
START → start → retrieve → generate → validate → predict → decide ─┬─ (Revise candidates left) → generate
                                                                   └─ (done / limit reached)   → finish → END
```

---

## 6. How tests work (pytest)

A **test** is a small function that runs part of our code with known inputs
and **asserts** the output is what we expect. If someone later breaks the
code, a test fails and tells them immediately.

- **pytest** is the program that finds and runs tests. It looks in files
  named `test_*.py` for functions named `test_*`.
- `python -m pytest orchestrator/tests -v` runs them all. `-v` (verbose) lists
  each test by name with PASSED or FAILED.
- Our tests replace the real tools with **fakes** we fully control (e.g. "when
  asked to predict `CCN`, always answer *uncertain*"). That way we know
  exactly what *should* happen and can check that the graph does it. This is
  the main reason the code uses interfaces: fakes slot in easily.
- **Fixtures**: if a test function has a parameter with a special name, pytest
  supplies it. `tmp_path` = a brand-new empty folder for that test only (our
  storage tests put their database files there). `caplog` = captures logged
  messages, so a test can check a warning was or wasn't printed.
- **A fake clock** (`FakeClock` in `fakes.py`) lets a test pretend hours have
  passed without waiting.

---

## 7. Web APIs and the worker in plain language

| Word | Plain meaning |
|---|---|
| **Web API** | Web addresses that *programs* (like Nyla's dashboard) send requests to, instead of people visiting pages |
| **Endpoint** | One of those addresses plus a method, e.g. `GET /api/v1/runs/{run_id}` |
| **`GET` / `POST`** | `GET` = "give me information" (never changes anything); `POST` = "do something / create something" |
| **JSON** | The text format for the data sent and received, e.g. `{"status": "queued"}` |
| **Status code** | A number in every reply. `200` OK, `202` accepted (started, not finished), `400` not valid JSON, `401` wrong/missing token, `404` not found, `409` conflict (e.g. the run already ended), `415` not JSON, `422` invalid values, `500` our bug, `503` database busy/unavailable |
| **Header** | Extra labelled information sent with a request, e.g. `Authorization: Bearer <token>`, `Idempotency-Key: <uuid>` |
| **Token** | A shared secret proving the caller is allowed in. We read it from the environment variable `ORCHESTRATOR_API_TOKEN` so it's never typed on the command line |
| **`127.0.0.1`** | "This machine". A server listening only there can't be reached from other computers |
| **Idempotency key** | A random ID the caller sends with a `POST`. If the same request arrives twice (a network hiccup, a double-click), the API recognises the key and returns the first reply instead of acting twice |
| **Polling** | Asking "is it done yet?" every few seconds. The dashboard polls `GET /runs/{id}`; the worker polls the database for queued runs |
| **FastAPI** | The Python library our API is built with. It reads pydantic models for requests and replies, and makes the interactive docs page at `/docs` |
| **uvicorn** | The program that actually runs a FastAPI app as a server |
| **Middleware** | Code that runs around *every* request, e.g. giving it an ID and refusing oversized bodies |
| **Worker** | The separate program that runs the queued runs (the API only records them) |
| **Claiming** | A worker marking a queued run as "mine". One database operation that succeeds only if the run is still queued, so two workers can't both take it |
| **Heartbeat** | The worker writing "still working on it" every few seconds. A run whose heartbeat has gone quiet (60 s) belonged to a crashed worker and can be taken over |
| **Thread** | A second line of work inside one program. The heartbeat runs in its own thread while the run works. One database connection must not be shared by two threads at once, so each thread opens its own |
| **`tmux`** | A Linux tool that keeps a terminal session running after you disconnect from the cluster |

The flow, end to end:
```
dashboard ──POST /runs──> API ──"queued" row──> runs.db <──claims, heartbeats── worker ──runs──> checkpoints.db
dashboard <──GET /runs/{id}, /shortlist, /report── API <──reads──────────────── runs.db + checkpoints.db
```
