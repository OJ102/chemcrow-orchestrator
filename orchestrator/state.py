"""Shared state for the transparent-OPV LangGraph orchestrator.

WHAT THIS FILE IS
-----------------
This file defines the *shapes of data* the whole program uses. It contains no
logic, only "blueprints" (classes) describing what information we store.

The shapes follow the team's design documents EXACTLY where they define them,
so that Arleen's gate, Nyla's dashboard and Jaxon's generator all see the same
field names:
  - D1 = "Detailed System Design - Part 1" (data model + web API contract)
  - D2 = "D2 Detailed Design" (orchestrator, gate, outputs)
Every class below says which document section it comes from. Anything that is
NOT from D1/D2 is marked "(our addition)".

The classes, from smallest to biggest:
  RunConstraints    : the three minimum targets a user chooses       (D1)
  RunConfig         : everything the user chooses for one run        (D1 + our additions)
  Policy            : scientific limits approved by the team         (D2 PolicyVersion)
  Values            : the six predicted numbers for one pair         (D1)
  Uncertainty       : how unsure the predictor is about each number  (D1)
  Prediction        : Values + Uncertainty + "in domain?" + status   (D1)
  ChemistryViolation: "this isn't a valid molecule"                  (D1 C3 Violation)
  ThresholdViolation: "this molecule misses a target"                (D1 API Violation)
  Decision          : the gate's verdict on one candidate            (D1)
  NearMiss          : how close a failed candidate came to passing   (our addition, advisor's request)
  Candidate         : one donor+acceptor pair and everything we learn about it
  OPVState          : the shared "notebook" LangGraph passes between steps

If a word like SMILES, PCE, AVT or LUE is unfamiliar, see docs/primer.md
section 2. For Python features (classes, type hints, pydantic) see section 3.
"""

# --- Imports: pulling in tools other people wrote ---------------------------

# `operator.add` is the function version of "+". For lists, [1] + [2] == [1, 2],
# so LangGraph can use it as a reducer meaning "append" (see OPVState below).
import operator

# Type-hint helpers from Python's `typing` module:
#   Annotated - attach extra info to a type (we attach a reducer to a list)
#   Literal   - a type that only allows specific exact values
#   Optional  - "this type, or None" (None = Python's "nothing here / unknown")
#   TypedDict - a dictionary whose keys and value types are declared up front
from typing import Annotated, Literal, Optional, TypedDict

# pydantic: data classes that CHECK their values when created.
#   BaseModel  - inherit from this to get checking + helpers like model_copy
#   ConfigDict - settings for a model (we use it to reject unknown fields)
#   Field      - lets us add rules to a field, e.g. Field(ge=0, le=1) means
#                "must be >= 0 and <= 1". Breaking a rule raises an error immediately.
from pydantic import BaseModel, ConfigDict, Field, field_validator

# RDKit's chemistry reader, used once below to check the fixed partner is a real molecule.
from rdkit import Chem


# ============================================================================
# ALLOWED-VALUE LISTS (type aliases)
# Literal[...] means only these exact strings are allowed. A typo like
# "Acept" is rejected. The spellings match D1 exactly, including capital letters.
# ============================================================================

# Which half of the donor:acceptor pair this run is designing. The other half is
# the fixed partner. (our addition: D1's POST /runs has no field for this yet,
# which is an open team question.)
TargetType = Literal["donor", "acceptor"]

# The gate's four verdicts (D1 "Decision.outcome"):
#   "Accept"   - passes every rule and the prediction is trustworthy -> shortlist candidate
#   "Revise"   - trustworthy prediction but misses a target -> improved in the next round
#   "Escalate" - can't trust the prediction (unsure / unfamiliar molecule) -> SET ASIDE
#                for DFT/lab review. The run keeps going (D2: no pause).
#   "Stop"     - this candidate's path ends (e.g. doesn't look like an OPV molecule,
#                or it would need revising but no rounds are left). Other candidates continue.
# There is NO "reject": broken molecules are caught earlier, by validation, and
# never reach the gate (they get ChemistryViolations instead).
Outcome = Literal["Accept", "Revise", "Escalate", "Stop"]

# The life of a run (D1 "Run.status"):
#   queued    - recorded, not started yet (used later by the web API)
#   running   - the loop is working
#   completed - finished normally; the shortlist is saved (it may be empty)
#   failed    - something broke and the run can't continue
#   stopped   - ended by a limit (rounds, time, or the optional patience)
RunStatus = Literal["queued", "running", "completed", "failed", "stopped"]

# Warning labels the gate can attach to a decision (D1 "RiskFlag"):
#   OUT_OF_DOMAIN           - molecule is unlike anything the predictor learned from
#   HIGH_UNCERTAINTY        - predictor is too unsure about at least one number
#   UNCERTAINTY_UNAVAILABLE - predictor couldn't say how unsure it is
#   CONSTRAINT_VIOLATION    - misses at least one minimum target
RiskFlag = Literal["OUT_OF_DOMAIN", "HIGH_UNCERTAINTY", "UNCERTAINTY_UNAVAILABLE",
                   "CONSTRAINT_VIOLATION"]

# The names of the six predicted numbers (D1 "Values"). Used as dictionary keys
# when something is "per value", e.g. the uncertainty limit for each one.
ValueName = Literal["opv_likeness", "donor_homo_eV", "acceptor_lumo_eV",
                    "pce_percent", "avt_fraction", "lue_percent"]

# The three values that have a user-chosen MINIMUM (D1 "Violation.field").
ConstrainedField = Literal["avt_fraction", "pce_percent", "lue_percent"]

# Units for the constrained values (D1 "Violation.unit"):
#   "fraction"          - 0 to 1, e.g. AVT 0.70 = 70% of light passes through
#   "percentage_points" - e.g. PCE 8.0 = 8%
Unit = Literal["fraction", "percentage_points"]

# Which unit each constrained field uses. A dictionary: key -> value.
FIELD_UNITS: dict[str, str] = {
    "avt_fraction": "fraction",
    "pce_percent": "percentage_points",
    "lue_percent": "percentage_points",
}

# D1 rule for candidate IDs: 1-64 characters, only letters, digits, "_" and "-".
# This is a "regular expression": a pattern that text must match.
#   ^ and $        = start and end of the text (the WHOLE id must match)
#   [A-Za-z0-9_-]  = one allowed character
#   {1,64}         = between 1 and 64 of them
CANDIDATE_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"


# ============================================================================
# WHAT THE USER CHOOSES FOR A RUN
# ============================================================================

class RunConstraints(BaseModel):
    """The three minimum targets for a run (D1 POST /runs "constraints").

    Every molecule must reach ALL three to be accepted. Example:
        RunConstraints(avt_min_fraction=0.40, pce_min_percent=8.0, lue_min_percent=3.0)
    means "at least 40% see-through, at least 8% efficient, and LUE at least 3.0".

    Field(ge=..., le=...) enforces D1's allowed ranges ("ge" = greater or equal,
    "le" = less or equal). A value outside the range is an error.
    """

    # extra="forbid": unknown fields are an error. D1: "Extra request fields are rejected."
    model_config = ConfigDict(extra="forbid")

    avt_min_fraction: float = Field(ge=0, le=1)  # minimum see-through-ness, 0-1
    pce_min_percent: float = Field(ge=0, le=100)  # minimum efficiency, in % (8.0 = 8%)
    lue_min_percent: float = Field(ge=0, le=100)  # minimum LUE (= AVT x PCE), in percentage points


class RunConfig(BaseModel):
    """Everything the person starting a run chooses (D1 POST /runs, plus our additions).

    Example (what the tests and demo do):
        RunConfig(target_type="donor", fixed_partner_smiles="c1ccsc1",
                  constraints=RunConstraints(avt_min_fraction=0.4,
                                             pce_min_percent=2.0,
                                             lue_min_percent=0.8))
    Every field not mentioned keeps its default (the value after "default=").

    NOT here yet (they arrive with the web API step): knowledge_snapshot_id,
    model_version and policy_version, the "pinned versions" from D1/G3.
    """

    model_config = ConfigDict(extra="forbid")  # reject unknown fields, as D1 requires

    # --- (our addition) what to design, against what ---
    target_type: TargetType  # design "donor"s or "acceptor"s
    fixed_partner_smiles: str = Field(min_length=1, max_length=4096)  # the other half of every pair.
    #   (4096 = D1's maximum SMILES length.) Open team question: D1 has no field for this.

    # --- D1 fields ---
    constraints: RunConstraints  # the three minimum targets (above)
    candidate_limit: int = Field(default=20, ge=1, le=100)  # max pairs per round (D1 default 20)
    max_iterations: int = Field(default=3, ge=1, le=10)  # max rounds (D1 default 3)
    seed: int = Field(default=0, ge=0, le=2147483647)  # number that makes "random" choices repeatable

    # --- (our addition) optional extra stop rules, OFF by default (None = off) ---
    target_count: Optional[int] = Field(default=None, ge=1)  # finish early once this many are accepted
    patience: Optional[int] = Field(default=None, ge=1)  # stop after this many rounds in a row with
    #   no near-miss progress (progress = formula NM-4 in near_miss.py)

    @field_validator("fixed_partner_smiles")
    @classmethod
    def _partner_must_be_a_real_molecule(cls, smiles: str) -> str:
        """Reject the run up front if the fixed partner isn't valid chemistry.

        If the partner were broken, EVERY pair in the run would be invalid, so
        it's better to refuse at the start (like D1's API refusing a bad request
        with 422). Returns the partner in RDKit's official (canonical) spelling,
        so every pair in the run uses exactly the same text for it.

        @field_validator("fixed_partner_smiles") tells pydantic to run this
        whenever that field is set. @classmethod is required by pydantic for
        validators (the function belongs to the class, not to one object).
        Raising ValueError makes pydantic reject the RunConfig with a clear error.
        """
        mol = Chem.MolFromSmiles(smiles)  # None if RDKit can't read it
        if mol is None:
            raise ValueError("fixed_partner_smiles is not a valid molecule (RDKit could not parse it)")
        return Chem.MolToSmiles(mol)


class Policy(BaseModel):
    """Scientific limits approved by the team (D2 "PolicyVersion", the parts we use so far).

    Different from RunConfig: users choose targets, but SAFETY limits like "how
    much uncertainty is acceptable" are approved separately and can't be changed
    per run (D2). Peer review PR-06 notes that nobody has approved real values
    yet, so the demo and tests use placeholder values.
    """

    model_config = ConfigDict(extra="forbid")

    policy_version: str = Field(min_length=1, max_length=64)  # name of this policy, e.g. "placeholder-0"
    opv_min: float = Field(ge=0, le=1)  # minimum OPV-likeness. Below it -> Stop (D2 gate rule).
    # Maximum acceptable uncertainty (one standard deviation) for each value that
    # MUST be trustworthy. A dictionary like {"avt_fraction": 0.05, "pce_percent": 1.0}.
    # Any value listed here whose uncertainty is unknown or bigger -> Escalate (D2).
    uncertainty_max_by_target: dict[ValueName, float]
    max_runtime_seconds: int = Field(default=1800, gt=0)  # whole-run time budget. D2 proposes 30 min.


# ============================================================================
# WHAT THE PREDICTOR (Arleen's Chemprop, C4) RETURNS
# ============================================================================

class Values(BaseModel):
    """The six predicted numbers for one donor+acceptor pair (D1 "Values").

    The Field(...) ranges are D1's. D2 says out-of-range numbers count as a FAILED
    prediction, so creating a Values with e.g. avt_fraction=1.5 raises an error,
    and the code that calls the predictor turns that into status="failed".
    """

    opv_likeness: float = Field(ge=0, le=1)  # 0-1: how much this looks like a real solar-cell material
    donor_homo_eV: float  # the DONOR's HOMO energy level, in eV (usually negative, e.g. -5.4)
    acceptor_lumo_eV: float  # the ACCEPTOR's LUMO energy level, in eV (e.g. -3.9)
    pce_percent: float = Field(ge=0, le=100)  # efficiency in %: 8.0 means 8%
    avt_fraction: float = Field(ge=0, le=1)  # see-through-ness, 0-1: 0.70 means 70%
    lue_percent: float = Field(ge=0, le=100)  # LUE = avt_fraction x pce_percent (0.70 x 8.0 = 5.6)


class Uncertainty(BaseModel):
    """How unsure the predictor is about EACH of the six numbers (D1 "Uncertainty").

    Each is "one standard deviation" in that value's own unit. Roughly: the true
    value is usually within +/- this amount of the prediction. Example:
    avt_fraction=0.04 with a predicted AVT of 0.38 means "probably 0.34 to 0.42".

    None means "the predictor doesn't know how unsure it is". D1: that is NOT the
    same as 0 (perfectly sure), and the gate treats it as untrustworthy.
    """

    opv_likeness: Optional[float] = Field(default=None, ge=0)
    donor_homo_eV: Optional[float] = Field(default=None, ge=0)
    acceptor_lumo_eV: Optional[float] = Field(default=None, ge=0)
    pce_percent: Optional[float] = Field(default=None, ge=0)
    avt_fraction: Optional[float] = Field(default=None, ge=0)
    lue_percent: Optional[float] = Field(default=None, ge=0)


class Prediction(BaseModel):
    """The predictor's answer for one candidate pair (D1 "Prediction").

    Two shapes are possible:
      status="ok"     -> values, uncertainty and in_domain are filled in; error is None
      status="failed" -> values, uncertainty and in_domain are None; error says why
    A failed prediction never reaches the gate (D1/D2). The orchestrator logs it.
    """

    status: Literal["ok", "failed"]
    values: Optional[Values] = None
    uncertainty: Optional[Uncertainty] = None
    # True if the pair resembles the predictor's training data (its "domain").
    # False means "this is unfamiliar territory, my numbers are guesses".
    in_domain: Optional[bool] = None
    error: Optional[str] = None  # why it failed (D1 has {code, message}; we keep the message)


# ============================================================================
# THE TWO KINDS OF "VIOLATION" (they are different things!)
# ============================================================================

class ChemistryViolation(BaseModel):
    """'This is not a usable molecule' (D1 C3 "Violation": check_code, severity, detail).

    Produced by VALIDATION (Jaxon's C3, and our own safety-net validate step)
    BEFORE prediction. A candidate with one of these never reaches the predictor.
    Examples:
        ChemistryViolation(check_code="invalid_smiles", severity="error",
                           detail="RDKit could not parse the SMILES")
        ChemistryViolation(check_code="duplicate_pair", severity="error",
                           detail="pair already evaluated in this run")
    """

    check_code: str  # short stable name a program can count, e.g. "invalid_smiles"
    severity: Literal["error", "warning"]  # D1 names the field but not its values (our choice)
    detail: str  # human-readable explanation


class ThresholdViolation(BaseModel):
    """'This molecule misses a target' (D1 API "Violation": field, actual, minimum, unit).

    Produced by the GATE after prediction. Machine-readable, as the advisor asked:
    instead of the sentence "AVT failed", it says exactly which value, by how much:
        ThresholdViolation(field="avt_fraction", actual=0.38, minimum=0.40, unit="fraction")
    """

    field: ConstrainedField  # which value missed its minimum
    actual: float  # the predicted value
    minimum: float  # the minimum it needed to reach
    unit: Unit  # "fraction" or "percentage_points"


class Decision(BaseModel):
    """The gate's verdict on one candidate (D1 "Decision")."""

    candidate_id: str  # which candidate this is about
    outcome: Outcome  # Accept / Revise / Escalate / Stop
    rationale: str = Field(min_length=1, max_length=2000)  # a sentence explaining why, for humans
    risk_flags: list[RiskFlag] = Field(default_factory=list)  # warning labels; may be empty
    violations: list[ThresholdViolation] = Field(default_factory=list)  # missed targets; empty if none


# ============================================================================
# NEAR MISS (our addition, from the advisor's guidance on 2026-10-23)
# ============================================================================

class NearMiss(BaseModel):
    """How close a candidate came to passing. Computed by near_miss.py.

    The formulas are labelled NM-1 ... NM-4 in near_miss.py. This class only
    STORES their results, so they appear in the run's records (Nyla's report).

    Example for Molecule A (AVT 0.38 +/- 0.04, minimum 0.40; PCE and LUE pass):
        shortfall    = {"avt_fraction": 0.05, "pce_percent": 0.0, "lue_percent": 0.0}
        sigmas_short = {"avt_fraction": 0.5,  "pce_percent": 0.0, "lue_percent": 0.0}
        borderline   = True   (only half an uncertainty away from passing)
    """

    # Formula NM-1: fraction short for each constrained value (0 = passes that rule).
    shortfall: dict[ConstrainedField, float]
    # Formula NM-2: how many uncertainties short for each value (0 = passes;
    # None = the predictor gave no uncertainty for that value).
    sigmas_short: dict[ConstrainedField, Optional[float]]
    # Formula NM-2's cut-off: True if every miss is within 1 uncertainty of passing,
    # i.e. "it could really pass, the prediction is just too rough to tell".
    borderline: bool


# ============================================================================
# THE CANDIDATE: one donor+acceptor pair moving through the loop
# ============================================================================

class Candidate(BaseModel):
    """One proposed donor+acceptor pair, plus everything we learn about it.

    The generator invents ONE molecule (the target_type side). It is paired with
    the run's fixed partner to make the pair (our fixed-partner approach, which
    fits the advisor's "modify one partner at a time").

    Fields get filled in step by step:
        generator -> candidate_id, iteration, parent_candidate_id, generated_smiles
        validate  -> canonical_smiles, donor_smiles, acceptor_smiles, valid
                     (or chemistry_violations if it's broken / a repeat)
        predict   -> prediction
        decide    -> decision, near_miss
    Steps never change a Candidate in place. They make an updated copy with
    model_copy(update={...}), which avoids one step accidentally changing data
    another step is using.
    """

    candidate_id: str = Field(pattern=CANDIDATE_ID_PATTERN)  # unique name within the run, e.g. "it2-c05"
    iteration: int = Field(ge=1)  # which round produced it (1, 2, 3...)
    # LINEAGE (D1 C3): the candidate this one was revised FROM. None in round 1
    # (invented from scratch). Lets anyone trace a molecule's family tree.
    parent_candidate_id: Optional[str] = None
    generated_smiles: str  # the new molecule exactly as the generator wrote it (may be broken)
    canonical_smiles: Optional[str] = None  # RDKit's one official spelling of that molecule
    donor_smiles: Optional[str] = None  # the pair's donor (canonical); set by validate
    acceptor_smiles: Optional[str] = None  # the pair's acceptor (canonical); set by validate
    valid: bool = False  # True once validation passed. Only valid candidates are predicted.
    chemistry_violations: list[ChemistryViolation] = Field(default_factory=list)  # why invalid, if so
    #   (default_factory=list gives each Candidate its OWN new empty list, never a shared one)
    prediction: Optional[Prediction] = None  # set by predict (valid candidates only)
    decision: Optional[Decision] = None  # set by the gate (successful predictions only)
    near_miss: Optional[NearMiss] = None  # set for candidates the gate judged (formulas NM-1, NM-2)

    @property
    def pair_key(self) -> str:
        """The pair's identity, used to spot repeats (D2: "canonical ordered pairs").

        "Ordered" means donor first, then acceptor. Swapping them is a DIFFERENT
        pair (D1). Before validation there is no canonical pair yet, so it falls
        back to the raw generated text.
        """
        if self.donor_smiles and self.acceptor_smiles:
            return f"{self.donor_smiles}>>{self.acceptor_smiles}"
        return self.generated_smiles


# ============================================================================
# THE GRAPH'S SHARED NOTEBOOK
# ============================================================================

class OPVState(TypedDict, total=False):
    """The LangGraph state: the shared "notebook" every step reads and writes.

    It's a plain dictionary, e.g. state["iteration"].

    HOW STEPS UPDATE IT: each step returns a small dictionary of ONLY the keys it
    changes, e.g. {"iteration": 2}. LangGraph merges that in using a "reducer":
      - normal keys are OVERWRITTEN (the new value replaces the old one)
      - keys declared Annotated[list[...], operator.add] are APPENDED
        (the returned list is added onto the end of the existing one)

    total=False: no key is required. A run starts with only run_id, objective,
    config and policy; the "start" step fills in the rest.
    """

    # --- Set by whoever starts the run ---
    run_id: str  # the run's unique ID. Also used as LangGraph's thread_id (save-slot name),
    #              as agreed with Nyla, so there's one ID per run.
    objective: str  # the goal in plain English (not read by any step yet; future LLM use)
    config: RunConfig  # what the user chose (targets, limits, partner)
    policy: Policy  # the approved scientific limits

    # --- Set by the retrieve step ---
    evidence: list[str]  # text passages from Kyle's knowledge base (collected, not used yet)

    # --- The candidates ---
    batch: list[Candidate]  # THIS round's candidates only (overwritten each round)
    history: Annotated[list[Candidate], operator.add]  # EVERY candidate ever made this run (appends).
    #   "Log every candidate" (advisor). Also used to spot repeated pairs.
    accepted: Annotated[list[Candidate], operator.add]  # every Accept, across all rounds
    escalated: Annotated[list[Candidate], operator.add]  # the SET-ASIDE pile: every Escalate, across
    #   all rounds. Shown separately from the shortlist and handed to DFT/lab review (D2).
    failed_predictions: Annotated[list[Candidate], operator.add]  # predictions that failed (logged, D2)
    parents: list[Candidate]  # this round's Revise candidates: next round improves THESE (D2 A2 step 6)

    # --- Progress counters ---
    iteration: int  # current round number: 0 before the first round, then 1, 2, 3...
    # Seconds the run has spent ACTUALLY WORKING (decision Q3, 2026-11-06). Every step
    # times itself and adds its duration here (operator.add on numbers = "add up").
    # Time between steps, e.g. overnight after a crash before a resume, is never counted.
    # Compared with Policy.max_runtime_seconds for the time budget. Starts at 0.
    active_seconds: Annotated[float, operator.add]
    stale_iterations: int  # rounds in a row with no near-miss progress (formula NM-4); for patience
    status: RunStatus  # queued / running / completed / failed / stopped
    stop_reason: str  # human-readable sentence explaining why the run ended
    ranked: list[Candidate]  # FINAL OUTPUT: accepted candidates, best first (from the ranker)
    near_misses: list[Candidate]  # FINAL OUTPUT: candidates that missed a target, closest first
    #   (formula NM-3). The advisor: "keep the near misses". Later this becomes the bounded,
    #   diverse active archive, once Jaxon provides a similarity measure.
