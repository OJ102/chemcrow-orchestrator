"""FAKE tool implementations so the graph runs end to end today.

WHAT THIS FILE IS
-----------------
Stand-ins for three teammates' tools, in the shapes the contracts in
interfaces.py require:

  StubRetriever -> stands in for Kyle's retriever      (C1)
  StubGenerator -> stands in for Jaxon's generator     (C3, MatGPT)
  StubPredictor -> stands in for Arleen's predictor    (C4, Chemprop)

(The gate and ranker stand-ins are in gate.py, because they follow D2's real rules.)

!!! THE MOLECULES AND NUMBERS THESE PRODUCE ARE FAKE !!!
They come from made-up formulas based on simple molecule counts (like "how many
rings"), chosen ONLY so every path through the graph gets used: Accept, Revise,
Escalate, Stop, invalid molecules, repeated pairs and failed predictions. They
say nothing about real solar cells.
"""

# math: square roots (for the fake LUE uncertainty). random: random choices.
import math
import random

# zlib.crc32 turns any text into a number; the SAME text always gives the SAME
# number. Used to make fake values that are repeatable per molecule.
import zlib

# RDKit: Chem reads SMILES; rdMolDescriptors counts things about a molecule.
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from .interfaces import ConstraintPacket, GenerationResult
from .state import Candidate, Prediction, RunConfig, Uncertainty, Values

# _SEEDS: hand-picked molecule "starting pieces" (SMILES) that resemble real
# solar-cell building blocks (rings containing sulfur, "s", are very common in
# this field). Round 1 glues one of these to one of the _TAILS.
# The leading underscore means "only used inside this file".
_SEEDS = [
    "c1ccc(s1)-c1ccc(s1)-c1ccc(s1)CCCCCC",  # three sulfur rings in a row + a carbon chain
    "CCCCCCc1cc(-c2ccc(-c3cccs3)c3nsnc23)sc1",
    "c1ccc2c(c1)[nH]c1ccccc12",  # carbazole, a common donor piece
    "O=C1c2ccccc2C(=O)N1CCCCCCCC",
    "c1cc2cc3ccsc3cc2s1",
    "CCCCCCCCc1ccc(-c2ccc3c(c2)C(=O)N(C)C3=O)s1",
    "c1ccc(cc1)-c1ccc(s1)C#N",
    "C1=CC(=O)C=CC1=O",
    "c1ccc(s1)-c1nsn",  # BROKEN ON PURPOSE: ring "1" is opened again but never closed,
    #                     so RDKit can't read it. Exercises our validate safety net.
]

# _TAILS: small pieces stuck onto a seed for variety ("" = add nothing).
# "CCCCCC" = a 6-carbon chain; "C#N", "F", "OC" = small chemical groups; "c1cccs1" = a sulfur ring.
_TAILS = ["", "CCCCCC", "C#N", "F", "OC", "c1cccs1"]

# _EDITS: SMALL changes used to revise a parent molecule in later rounds. Adding
# one small group is a "local" change, a crude imitation of the advisor's "prefer
# local, chemically meaningful changes". The real version is Jaxon's job.
_EDITS = ["F", "C", "OC", "C#N", "CC", "Cl"]


def _attach(smiles: str, group: str, rng: random.Random) -> str:
    """Make a LOCAL edit: attach a small `group` (e.g. "F") to one atom of `smiles`.

    Why not just glue the text together ("CCO" + "F")? Text glued onto the end
    bonds to whatever atom happens to be written last. If that atom is already
    "full" (e.g. a fluorine, or the N of a C#N group) the result is impossible
    chemistry. Instead we pick an atom that still has a free hydrogen and
    replace that hydrogen with the group.

    Returns the new molecule as SMILES. If anything goes wrong (e.g. the input is
    already broken), falls back to plain text gluing, which may give an invalid
    molecule. That's fine: our validate step will catch and record it.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not group:
        return smiles + group
    # Atoms with at least one hydrogen attached: places where a group can go.
    sites = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetTotalNumHs() > 0]
    if not sites:
        return smiles + group
    site = rng.choice(sites)
    # CombineMols puts both molecules side by side (not yet connected); RWMol makes
    # it editable. The group's first atom gets index mol.GetNumAtoms() in the combination.
    combined = Chem.RWMol(Chem.CombineMols(mol, Chem.MolFromSmiles(group)))
    combined.AddBond(site, mol.GetNumAtoms(), Chem.BondType.SINGLE)  # connect them
    try:
        Chem.SanitizeMol(combined)  # RDKit's "does this obey chemistry rules?" check
    except Exception:  # any rule broken -> use the fallback
        return smiles + group
    return Chem.MolToSmiles(combined)


class StubRetriever:
    """Fake evidence finder: returns one placeholder sentence instead of research text."""

    def retrieve(self, objective: str, config: RunConfig) -> list[str]:
        # {objective!r} inserts the objective with quotes around it.
        return [f"[stub evidence] no knowledge base connected yet; objective was: {objective!r}"]


class StubGenerator:
    """Fake molecule inventor. Implements the CandidateGenerator contract.

    Round 1 (no parents): attach a random tail to a random seed.
    Later rounds: take each parent molecule and attach one small group from
    _EDITS (a local change, see _attach), recording the parent's ID as lineage.

    It does NOT validate (everything goes in `valid`, `rejected` is empty). That
    mirrors reality today: Jaxon's validation isn't built yet, so our own validate
    step in graph.py is the safety net that catches broken molecules (decision A2).
    """

    def generate_validate(self, packet: ConstraintPacket) -> GenerationResult:
        # A random source seeded from the run's seed AND the round number, so every
        # round differs but re-running the same run gives the same molecules.
        # (Seeding from the packet, rather than storing a generator inside this
        # object, means a resumed run continues exactly the same way.)
        rng = random.Random(packet.seed * 1000 + packet.iteration)
        candidates = []
        for k in range(packet.candidate_limit):  # k = 0, 1, 2, ... candidate_limit-1
            if packet.parents:
                # Spread the revisions evenly over the parents: k % len gives
                # 0,1,2,0,1,2,... ("%" = remainder after division).
                parent = packet.parents[k % len(packet.parents)]
                smiles = _attach(parent.canonical_smiles, rng.choice(_EDITS), rng)  # local edit
                parent_id = parent.candidate_id
            else:
                smiles = _attach(rng.choice(_SEEDS), rng.choice(_TAILS), rng)
                parent_id = None
            candidates.append(Candidate(
                # e.g. "it2-c05" = round 2, candidate 5. :02d pads with a zero to 2 digits.
                candidate_id=f"it{packet.iteration}-c{k + 1:02d}",
                iteration=packet.iteration,
                parent_candidate_id=parent_id,
                generated_smiles=smiles,
            ))
        return GenerationResult(valid=candidates, rejected=[])


class StubPredictor:
    """Fake predictor. Implements the PropertyPredictor contract.

    The fake "rules" mimic the real-world trade-off (docs/primer.md section 2):
      - more aromatic rings -> absorbs more light -> higher PCE but lower AVT (darker)
      - bigger molecules (> 25 heavy atoms) -> more uncertain, sometimes "out of
        domain" -> exercises the Escalate path
      - about 1 in 20 molecules -> a simulated model failure -> exercises the
        failed-prediction path
    """

    def predict(self, candidates: list[Candidate]) -> list[Prediction]:
        """One Prediction per candidate, in the same order (the contract requires this)."""
        return [self._predict_one(c) for c in candidates]

    def _predict_one(self, cand: Candidate) -> Prediction:
        # We describe the NEW molecule only (the fixed partner is the same in every
        # pair, so it wouldn't change the fake numbers anyway).
        mol = Chem.MolFromSmiles(cand.canonical_smiles)  # always valid here: validate ran first
        rings = rdMolDescriptors.CalcNumAromaticRings(mol)  # number of flat "aromatic" rings
        heavy = mol.GetNumHeavyAtoms()  # number of non-hydrogen atoms (a rough size measure)
        # jitter: a repeatable pseudo-random number from 0 to 0.999 for this molecule.
        # crc32 -> big number; "% 1000" keeps the last 3 digits; "/ 1000" scales to 0-1.
        jitter = (zlib.crc32(cand.canonical_smiles.encode()) % 1000) / 1000
        big = heavy > 25  # True/False; Python treats True as 1 and False as 0 in arithmetic

        if jitter < 0.05:  # ~5% of molecules: pretend the model crashed on this one
            return Prediction(status="failed", error="stub: simulated per-candidate model failure")

        pce = min(1.2 * rings + 2 * jitter, 15.0)  # more rings -> more efficient; capped at 15%
        avt = max(0.0, min(1.0, 0.85 - 0.08 * rings + 0.1 * jitter))  # more rings -> darker; kept 0-1
        sigma_avt = 0.02 + 0.06 * jitter * big  # small molecules are confident (0.02)
        sigma_pce = 0.3 + 1.2 * jitter * big
        return Prediction(
            status="ok",
            values=Values(
                opv_likeness=min(1.0, 0.2 + 0.15 * rings),  # more rings -> looks more like an OPV molecule
                donor_homo_eV=-5.0 - 0.05 * rings,  # fake energy levels (not checked by the gate)
                acceptor_lumo_eV=-3.9 - 0.02 * rings,
                pce_percent=pce,
                avt_fraction=avt,
                lue_percent=avt * pce,  # LUE = AVT x PCE (D1 definition)
            ),
            uncertainty=Uncertainty(
                opv_likeness=0.05,
                donor_homo_eV=0.1,
                acceptor_lumo_eV=0.1,
                pce_percent=sigma_pce,
                avt_fraction=sigma_avt,
                # Stand-in for the "joint" uncertainty of a product (D2 wants the real
                # model to estimate this properly): the standard error-propagation rule
                # for multiplying two uncertain numbers.
                lue_percent=math.sqrt((pce * sigma_avt) ** 2 + (avt * sigma_pce) ** 2),
            ),
            in_domain=not (big and jitter > 0.8),  # some big molecules count as "unfamiliar"
        )
