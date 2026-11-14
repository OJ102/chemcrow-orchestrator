"""opv_tools.py: four REAL tools the local model can call, over Kyle's OPV molecule list.

DATA: ~/llm/data/classified_opv_database.csv (4,023 rows: molecule_name, canonical_smiles,
role = donor/acceptor). Some names appear more than once; we use the first valid entry.

DESIGN CHOICE FOR A SMALL MODEL (Om: "optimize well, the model isn't very smart"):
    Every tool takes and returns molecule NAMES ("Y6"), never SMILES. OPV SMILES are
    100+ characters long; a small model copying them from one step to the next would
    make typos. The tools look the structure up themselves.
    Names are matched forgivingly ("y-6", "Y 6" -> "Y6"). When a name is unknown, the
    error lists the closest names, so the model can fix its call on the next step.

Each tool is a plain Python function that returns a dict (sent to the model as JSON).
TOOL_SCHEMAS describes them to the model in OpenAI's "function" format.
"""

import difflib  # finds "closest" strings, for did-you-mean suggestions
import re  # regular expressions, used to normalise names
from pathlib import Path

import pandas as pd  # reads the CSV
from rdkit import Chem, DataStructs, RDLogger  # chemistry toolkit
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")  # hide RDKit's warnings about odd molecules

DATA_FILE = Path.home() / "llm" / "data" / "classified_opv_database.csv"

# Morgan fingerprint = a list of on/off bits, one per small chemical pattern in the
# molecule. Two molecules sharing many bits are structurally similar.
_FP_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _norm(name: str) -> str:
    """Normalise a name for matching: lower case, no spaces, hyphens or underscores."""
    return re.sub(r"[\s\-_]+", "", str(name)).lower()


def _load():
    """Read the CSV once and prepare one entry per unique name (first valid structure)."""
    df = pd.read_csv(DATA_FILE)
    entries = {}  # normalised name -> dict with name, role, mol, fingerprint, count
    for row in df.itertuples():
        mol = Chem.MolFromSmiles(str(row.canonical_smiles))
        if mol is None:
            continue  # skip rows RDKit can't read
        key = _norm(row.molecule_name)
        if key in entries:
            entries[key]["entries"] += 1  # duplicate name: count it, keep the first
            continue
        entries[key] = {"name": str(row.molecule_name).strip(), "role": row.role,
                        "mol": mol, "fp": _FP_GEN.GetFingerprint(mol), "entries": 1}
    return entries


MOLECULES = _load()  # loaded when the file is imported (takes ~2 s)


class ToolError(Exception):
    """A problem the MODEL can fix (bad name, bad argument). Sent back to it with a hint."""


def _get(name: str) -> dict:
    """Find a molecule by name, or raise ToolError with the closest names as suggestions."""
    entry = MOLECULES.get(_norm(name))
    if entry is None:
        close = difflib.get_close_matches(_norm(name), MOLECULES.keys(), n=5, cutoff=0.6)
        raise ToolError(f"No molecule named '{name}'. Closest names: "
                        f"{[MOLECULES[k]['name'] for k in close] or 'none'}")
    return entry


# ---------------------------------------------------------------------------
# The four tools
# ---------------------------------------------------------------------------

def find_molecule(name: str) -> dict:
    """Look a molecule up: its exact name, role (donor/acceptor) and size."""
    m = _get(name)
    return {"name": m["name"], "role": m["role"], "heavy_atoms": m["mol"].GetNumHeavyAtoms(),
            "entries_with_this_name": m["entries"]}


def most_similar(name: str, role: str = "any", k: int = 3) -> dict:
    """The k molecules most similar in structure to `name` (Tanimoto similarity, 0..1)."""
    m = _get(name)
    if role not in ("donor", "acceptor", "any"):
        raise ToolError("role must be 'donor', 'acceptor' or 'any'")
    k = max(1, min(int(k), 10))  # keep answers short: 1..10
    pool = [e for e in MOLECULES.values()
            if e is not m and (role == "any" or e["role"] == role)]
    sims = DataStructs.BulkTanimotoSimilarity(m["fp"], [e["fp"] for e in pool])
    best = sorted(zip(sims, pool), key=lambda pair: -pair[0])[:k]
    return {"query": m["name"], "role_filter": role,
            "results": [{"name": e["name"], "role": e["role"], "similarity": round(s, 3)}
                        for s, e in best]}


def properties(name: str) -> dict:
    """Basic computed properties of one molecule (from its structure, via RDKit)."""
    m = _get(name)
    mol = m["mol"]
    count = lambda symbol: sum(1 for a in mol.GetAtoms() if a.GetSymbol() == symbol)
    return {"name": m["name"], "role": m["role"],
            "formula": rdMolDescriptors.CalcMolFormula(mol),
            "molecular_weight": round(Descriptors.MolWt(mol), 1),
            "heavy_atoms": mol.GetNumHeavyAtoms(),
            "rings": rdMolDescriptors.CalcNumRings(mol),
            "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
            "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "atom_counts": {s: count(s) for s in ("S", "N", "O", "F", "Cl")}}


def compare(name_a: str, name_b: str) -> dict:
    """Compare two molecules: structural similarity and molecular weights."""
    a, b = _get(name_a), _get(name_b)
    mw_a, mw_b = Descriptors.MolWt(a["mol"]), Descriptors.MolWt(b["mol"])
    return {"a": a["name"], "b": b["name"], "roles": [a["role"], b["role"]],
            "similarity": round(DataStructs.TanimotoSimilarity(a["fp"], b["fp"]), 3),
            "molecular_weights": [round(mw_a, 1), round(mw_b, 1)],
            "weight_difference": round(abs(mw_a - mw_b), 1)}


# Name -> Python function, so the agent can run whatever tool the model names.
TOOLS = {"find_molecule": find_molecule, "most_similar": most_similar,
         "properties": properties, "compare": compare}

_NAME = {"type": "string", "description": "molecule name, e.g. 'Y6' (never SMILES)"}

# What the model sees: each tool's name, purpose and arguments (OpenAI "function" format).
# Short, plain descriptions; small models do better with fewer, clearer words.
TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "find_molecule",
        "description": "Look up a molecule in the OPV database by name. Returns its exact name, role (donor or acceptor) and size.",
        "parameters": {"type": "object", "properties": {"name": _NAME}, "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "most_similar",
        "description": "Find the molecules most similar in structure to a given molecule. Optionally only donors or only acceptors.",
        "parameters": {"type": "object", "properties": {
            "name": _NAME,
            "role": {"type": "string", "enum": ["donor", "acceptor", "any"], "description": "which molecules to search"},
            "k": {"type": "integer", "minimum": 1, "maximum": 10, "description": "how many results"}},
            "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "properties",
        "description": "Computed properties of one molecule: formula, molecular weight, heavy atoms, rings, aromatic rings, rotatable bonds, and counts of S, N, O, F, Cl atoms.",
        "parameters": {"type": "object", "properties": {"name": _NAME}, "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "compare",
        "description": "Compare two molecules: structural similarity (0 to 1) and their molecular weights.",
        "parameters": {"type": "object", "properties": {"name_a": _NAME, "name_b": _NAME},
                       "required": ["name_a", "name_b"]}}},
]
