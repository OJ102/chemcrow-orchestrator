"""Placeholder tool implementations so the graph runs end to end today.

THE NUMBERS THESE PRODUCE ARE FAKE. They are deterministic functions of simple
RDKit descriptors, chosen only so the graph exercises every branch (accept,
revise, escalate, reject). Replace each stub with the real tool from the
owning student; see interfaces.py.
"""

import random
import zlib

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from .state import Constraints, Prediction, TargetType

# Small set of OPV-flavoured building blocks (thiophenes, BT, fused cores),
# plus one deliberately broken string to exercise the validation node.
_SEEDS = [
    "c1ccc(s1)-c1ccc(s1)-c1ccc(s1)CCCCCC",
    "CCCCCCc1cc(-c2ccc(-c3cccs3)c3nsnc23)sc1",
    "c1ccc2c(c1)[nH]c1ccccc12",
    "O=C1c2ccccc2C(=O)N1CCCCCCCC",
    "c1cc2cc3ccsc3cc2s1",
    "CCCCCCCCc1ccc(-c2ccc3c(c2)C(=O)N(C)C3=O)s1",
    "c1ccc(cc1)-c1ccc(s1)C#N",
    "C1=CC(=O)C=CC1=O",
    "c1ccc(s1)-c1nsn",  # invalid on purpose
]
# Side groups appended to a seed ("" = no tail). Combining seeds x tails gives
# variety, and occasional repeats so duplicate detection gets exercised too.
_TAILS = ["", "CCCCCC", "C#N", "F", "OC", "c1cccs1"]


class StubRetriever:
    """Fake Retriever: returns one placeholder sentence instead of real evidence."""

    def retrieve(self, objective: str, constraints: Constraints) -> list[str]:
        return [f"[stub evidence] no corpus connected yet; objective was: {objective!r}"]


class StubGenerator:
    """Fake CandidateGenerator: random seed+tail combinations.

    NOTE: ignores `feedback` entirely, so the revise loop has no real effect
    until a real generator is plugged in.
    """

    def __init__(self, seed: int = 0):
        # Private random generator seeded for reproducibility: same seed ->
        # same sequence of molecules on every run.
        self._rng = random.Random(seed)

    def generate(self, constraints: Constraints, feedback: list[str], n: int) -> list[str]:
        return [self._rng.choice(_SEEDS) + self._rng.choice(_TAILS) for _ in range(n)]


class StubPredictor:
    """Fake PropertyPredictor: repeatable numbers from simple RDKit descriptors.

    Shape of the fake: more aromatic rings -> higher PCE but lower AVT (more
    absorbing). Uncertainty only rises for large molecules, which is what
    triggers the escalate branch.
    """

    def predict(self, smiles: str, partner_smiles: str, target_type: TargetType) -> Prediction:
        mol = Chem.MolFromSmiles(smiles)  # parsed molecule (only called on valid SMILES)
        rings = rdMolDescriptors.CalcNumAromaticRings(mol)  # number of aromatic rings
        heavy = mol.GetNumHeavyAtoms()  # number of non-hydrogen atoms
        jitter = (zlib.crc32(smiles.encode()) % 1000) / 1000  # stable per SMILES
        return Prediction(
            pce=min(1.2 * rings + 2 * jitter, 15.0),
            avt=max(0.0, 0.85 - 0.08 * rings + 0.1 * jitter),
            homo=-5.0 - 0.05 * rings,
            lumo=-3.5 - 0.05 * rings,
            opv_prob=min(1.0, 0.2 + 0.15 * rings),
            # (heavy > 25) is a bool, i.e. 0 or 1: small molecules always get 0.05.
            uncertainty=0.05 + 0.2 * jitter * (heavy > 25),
        )
