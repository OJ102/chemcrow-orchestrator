"""Tool interfaces the orchestrator depends on.

Each protocol is owned by a different capstone role; the graph only ever talks
to these signatures, so real implementations can replace the stubs in
stubs.py without touching graph code.

- Retriever          -> Student 1 (data / RAG pipeline)
- CandidateGenerator -> Student 3 (MatGPT wrapper)
- PropertyPredictor  -> Student 4 (Chemprop + scoring)

typing.Protocol is "duck typing with type hints": any class that has a method
with the matching name and signature counts as that protocol. It does NOT
need to inherit from it. E.g. stubs.StubGenerator never mentions
CandidateGenerator, yet satisfies it.
"""

from dataclasses import dataclass
from typing import Protocol

from .state import Constraints, Prediction, TargetType


class Retriever(Protocol):
    def retrieve(self, objective: str, constraints: Constraints) -> list[str]:
        """Return evidence passages relevant to the objective."""
        ...


class CandidateGenerator(Protocol):
    def generate(self, constraints: Constraints, feedback: list[str], n: int) -> list[str]:
        """Propose n SMILES for constraints.target_type, conditioned on the
        fixed partner and on feedback from previously rejected candidates."""
        ...


class PropertyPredictor(Protocol):
    def predict(self, smiles: str, partner_smiles: str, target_type: TargetType) -> Prediction:
        """Predict pair properties for the candidate paired with its partner."""
        ...


@dataclass
class Tools:
    """Bundle of the three tools, passed to build_graph() as one object.

    Swapping a stub for a real implementation means constructing Tools with a
    different object in that slot; graph.py is unaffected.
    """

    retriever: Retriever
    generator: CandidateGenerator
    predictor: PropertyPredictor
