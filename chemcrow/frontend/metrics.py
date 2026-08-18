"""Registry of metrics computed for the molecule shown in the Streamlit UI.

Add a new metric by writing a `fn(mol: Chem.Mol) -> Any` and appending
`("MetricName", fn)` to `METRICS` -- `compute_metrics()` and the UI pick it
up automatically, no other code changes needed.
"""

from typing import Any, Callable, Dict, List, Tuple

from rdkit import Chem

MetricFn = Callable[[Chem.Mol], Any]


def _pce_stub(mol: Chem.Mol) -> Any:
    """Power conversion efficiency -- placeholder.

    TODO: plug in a real PCE predictor (e.g. a trained property-prediction
    model). Returning None renders as "N/A" in the UI until then.
    """
    return None


METRICS: List[Tuple[str, MetricFn]] = [
    ("PCE", _pce_stub),
    # Add more metrics here, e.g.: ("LogP", _logp), ("MolWt", _mol_wt), ...
]


def compute_metrics(mol: Chem.Mol) -> Dict[str, Any]:
    """Run every registered metric against `mol`, returning {name: value}.

    A metric that raises or returns None is reported as "N/A" rather than
    breaking the whole panel.
    """
    results: Dict[str, Any] = {}
    for name, fn in METRICS:
        try:
            value = fn(mol)
        except Exception:
            value = None
        results[name] = "N/A" if value is None else value
    return results
