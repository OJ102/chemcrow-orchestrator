"""Local 2D structure rendering for the Streamlit UI.

Uses RDKit directly -- already a hard dependency of chemcrow -- rather than
`chemcrow/frontend/utils.py::cdk()`, which depends on an external web
service and isn't appropriate for a self-contained local UI.
"""

from typing import Optional, Tuple

from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw

from chemcrow.utils import is_smiles


def smiles_to_image(smiles: str, size: Tuple[int, int] = (350, 300)) -> Optional[Image.Image]:
    """Render a SMILES string to a 2D structure image.

    Returns None for empty/invalid input -- caller is expected to show
    `st.error` rather than crash, mirroring `tools/rdkit.py`'s
    validate-then-act pattern.
    """
    if not smiles or not is_smiles(smiles):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=size)
