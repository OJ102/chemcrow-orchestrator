"""
check_synspace.py: a one-off experiment (NOT part of the orchestrator package).

(Named check_... rather than ..._test.py so pytest doesn't mistake it for a
test file: it needs the separate .venv-synspace, which pytest doesn't use.)
Results of the 2026-11-10 run are in orchestrator/docs/progress.md; how to run
it and read its output is in orchestrator/docs/running.md.

Question we are answering:
    Can SynSpace (the library behind ChemCrow's "ModifyMol" tool) make small,
    synthesizable variants of REAL organic-solar-cell molecules? Those molecules
    are much bigger than the drug molecules SynSpace was designed for.
    If it can, it is a candidate way to "repair" near misses (molecules that
    barely fail the Decision Gate) by nudging them slightly.

How it works:
    1. Read Kyle's classified_opv_database.csv (columns: molecule_name,
       canonical_smiles, role).
    2. Pick a mixed sample: some famous molecules by name, plus small/medium/
       large molecules of each role (donor, acceptor) by atom count.
    3. For each molecule, call synspace.chemical_space() twice:
         - filter=True  (SynSpace's default: drops variants that trip REOS,
                         a drug-chemistry "undesirable groups" filter)
         - filter=False (keeps everything)
       Each call runs in its own child process with a time limit, so one slow
       molecule can't freeze the whole test.
    4. Record: did it run, how long, how many variants, how many RDKit accepts,
       how similar they are to the original, which reactions were used.
    5. Print a table and save full results to
       experiments/output/synspace_results.json (git-ignored).

Run from the repo root with the throwaway venv (NOT .venv-langgraph):
    .\\.venv-synspace\\Scripts\\python.exe experiments\\check_synspace.py C:\\path\\to\\classified_opv_database.csv
"""

import json  # to save the results file
import multiprocessing as mp  # to run each test in a separate process (time limit)
import statistics  # median of similarity scores
import sys  # to read the CSV path from the command line
import time  # to measure how long each call takes
from collections import Counter  # to count which reactions were used
from pathlib import Path  # file paths that work on Windows and Linux

# ----------------------------------------------------------------------------
# Settings (all the knobs of this experiment live here)
# ----------------------------------------------------------------------------

# Famous OPV molecules to include if Kyle's file has them (matched by name).
# Y6 / L8-BO / ITIC / IT-4F: non-fullerene acceptors; PC71BM: fullerene
# acceptor; PM6 / D18 / PTB7-Th / P3HT: polymer donors.
FAMOUS_NAMES = ["Y6", "L8-BO", "ITIC", "IT-4F", "PC71BM", "PM6", "D18", "PTB7-Th", "P3HT"]

# How many famous molecules to keep at most (so the sample stays ~10-12).
MAX_FAMOUS = 6

# Size percentiles to sample per role: 10th = small, 50th = typical, 90th = large.
SIZE_PERCENTILES = [0.10, 0.50, 0.90]

# Time limit for ONE chemical_space() call, in seconds. If it takes longer we
# record "timeout"; a near-miss repair tool that takes minutes is not usable.
TIME_LIMIT_S = 180

# Fixed random seed: SynSpace picks building blocks at random, so a seed makes
# re-running this script give the same numbers.
SEED = 0


# ----------------------------------------------------------------------------
# The part that runs inside the child process
# ----------------------------------------------------------------------------

def _run_one(smiles, use_filter, queue):
    """Run SynSpace on one molecule and put a result dict on `queue`.

    Runs in a CHILD process, so the parent can kill it if it exceeds the time
    limit. Imports happen here because each child starts fresh on Windows.
    """
    import numpy as np  # SynSpace uses numpy's random generator
    from rdkit import Chem, RDLogger  # RDKit: the chemistry toolkit
    from synspace import chemical_space  # the function ChemCrow's ModifyMol wraps

    RDLogger.DisableLog("rdApp.*")  # silence RDKit's warnings in the output
    np.random.seed(SEED)  # make random building-block choices repeatable

    started = time.perf_counter()  # stopwatch start
    try:
        # mols  = list of RDKit molecules (the first one is usually the original)
        # props = one dict per molecule: reaction name, reaction SMILES, similarity
        mols, props = chemical_space(smiles, filter=use_filter)
    except Exception as exc:  # any crash inside SynSpace
        queue.put({"status": "crash", "error": f"{type(exc).__name__}: {exc}",
                   "seconds": round(time.perf_counter() - started, 2)})
        return
    seconds = round(time.perf_counter() - started, 2)  # stopwatch stop

    # The original molecule written the standard ("canonical") way, so we can
    # recognise it in the output and not count it as a variant.
    original = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

    variants = []  # one small dict per real variant
    for mol, prop in zip(mols, props):
        smi = Chem.MolToSmiles(mol)  # variant as text
        if smi == original:
            continue  # skip the input molecule itself
        # "Valid" = RDKit can read the variant's SMILES back in without error.
        valid = Chem.MolFromSmiles(smi) is not None
        variants.append({
            "smiles": smi,
            "valid": valid,
            "similarity": float(prop.get("similarity", 0.0)),  # Tanimoto 0..1 vs original
            "reaction": prop.get("rxn-name", ""),  # which reaction made it
            "heavy_atoms": mol.GetNumHeavyAtoms(),  # size of the variant
        })

    queue.put({"status": "ok", "seconds": seconds, "variants": variants})


def run_with_time_limit(smiles, use_filter):
    """Start _run_one in a child process; kill it after TIME_LIMIT_S seconds."""
    queue = mp.Queue()  # the child sends its result back through this
    proc = mp.Process(target=_run_one, args=(smiles, use_filter, queue))
    proc.start()
    try:
        return queue.get(timeout=TIME_LIMIT_S)  # wait for the result
    except Exception:  # queue.Empty => no result within the time limit
        proc.terminate()  # stop the stuck child
        return {"status": "timeout", "seconds": TIME_LIMIT_S}
    finally:
        proc.join(timeout=5)  # tidy up the child process


# ----------------------------------------------------------------------------
# Picking the sample
# ----------------------------------------------------------------------------

def pick_sample(csv_path):
    """Return a list of {name, role, smiles, heavy_atoms, why} to test."""
    import pandas as pd  # table handling
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    df = pd.read_csv(csv_path)  # columns: molecule_name, canonical_smiles, role

    # Count heavy atoms (all atoms except hydrogen) = a simple size measure.
    # Rows RDKit can't read get None and are dropped.
    def heavy(smi):
        mol = Chem.MolFromSmiles(str(smi))
        return mol.GetNumHeavyAtoms() if mol is not None else None

    df["heavy_atoms"] = df["canonical_smiles"].map(heavy)
    df = df.dropna(subset=["heavy_atoms"])

    chosen = {}  # SMILES -> row dict (a dict, so the same molecule isn't picked twice)

    # 1) Famous molecules, in the order listed, up to MAX_FAMOUS.
    for name in FAMOUS_NAMES:
        hits = df[df["molecule_name"].str.strip().str.upper() == name.upper()]
        if len(hits) and len([r for r in chosen.values() if r["why"] == "famous"]) < MAX_FAMOUS:
            row = hits.iloc[0]
            chosen[row["canonical_smiles"]] = {"why": "famous", **row.to_dict()}

    # 2) Small / typical / large molecule of each role, by heavy-atom percentile.
    for role in ["donor", "acceptor"]:
        sub = df[df["role"] == role].sort_values("heavy_atoms").reset_index(drop=True)
        for pct in SIZE_PERCENTILES:
            row = sub.iloc[int(pct * (len(sub) - 1))]
            chosen.setdefault(row["canonical_smiles"],
                              {"why": f"{role} p{int(pct * 100)} size", **row.to_dict()})

    return [{"name": r["molecule_name"], "role": r["role"], "smiles": smi,
             "heavy_atoms": int(r["heavy_atoms"]), "why": r["why"]}
            for smi, r in chosen.items()]


# ----------------------------------------------------------------------------
# Main: run everything, print a table, save JSON
# ----------------------------------------------------------------------------

def summarise(result):
    """Turn one run's raw result into the few numbers we print."""
    if result["status"] != "ok":
        return {"status": result["status"], "seconds": result["seconds"],
                "n": 0, "valid": 0, "sim_min": None, "sim_med": None, "sim_max": None,
                "top_reactions": [], "error": result.get("error")}
    vs = result["variants"]
    sims = [v["similarity"] for v in vs]
    return {
        "status": "ok",
        "seconds": result["seconds"],
        "n": len(vs),  # number of variants (original excluded)
        "valid": sum(v["valid"] for v in vs),  # how many RDKit accepts
        "sim_min": min(sims) if sims else None,
        "sim_med": round(statistics.median(sims), 2) if sims else None,
        "sim_max": max(sims) if sims else None,
        "top_reactions": Counter(v["reaction"] for v in vs).most_common(3),
    }


def main():
    if len(sys.argv) != 2:  # no CSV path given: explain instead of crashing
        sys.exit("usage: python experiments/check_synspace.py PATH\\TO\\classified_opv_database.csv")
    csv_path = Path(sys.argv[1])  # Kyle's CSV, passed on the command line
    # Results go in experiments/output/ (git-ignored), created if missing.
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "synspace_results.json"

    sample = pick_sample(csv_path)
    print(f"Testing {len(sample)} molecules, time limit {TIME_LIMIT_S}s per call\n")

    # Is the ORIGINAL molecule itself rejected by the REOS drug filter?
    # If yes, the filter is likely to reject most of its variants too.
    from rdkit import Chem
    from synspace.utils import reos

    all_results = []
    for m in sample:
        reos_verdict = reos.process_mol(Chem.MolFromSmiles(m["smiles"]))
        entry = {**m, "original_reos": list(reos_verdict), "runs": {}}
        for use_filter in (True, False):
            label = "filter_on" if use_filter else "filter_off"
            raw = run_with_time_limit(m["smiles"], use_filter)
            entry["runs"][label] = {"summary": summarise(raw),
                                    "examples": raw.get("variants", [])[:5]}
            s = entry["runs"][label]["summary"]
            print(f"{m['name'][:14]:14} {m['role'][:3]} {m['heavy_atoms']:4d} atoms "
                  f"{label:10} {s['status']:7} {s['seconds']:7.1f}s "
                  f"variants={s['n']:3d} valid={s['valid']:3d} "
                  f"sim(min/med/max)={s['sim_min']}/{s['sim_med']}/{s['sim_max']}")
        print(f"{'':14}     original REOS verdict: {reos_verdict}\n")
        all_results.append(entry)

    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"Full results saved to {out_path}")


if __name__ == "__main__":  # needed on Windows when using multiprocessing
    main()
