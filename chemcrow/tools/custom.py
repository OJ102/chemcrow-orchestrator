"""Scaffold for a new, user-defined chemistry tool.

This is an extension point, not a finished tool: `CustomTool._run()` is a
placeholder. Fill in its body (and rename the class/tool) once the tool's
actual behavior is decided. Follow this same shape for any additional tools:
one `BaseTool` subclass per tool, exported here, registered in
`chemcrow/agents/tools.py::make_tools()`.
"""

from langchain.tools import BaseTool

from chemcrow.utils import is_smiles


class CustomToolHelper:
    """Optional helper class for shared state/config (API clients, caches,
    loaded models, etc.), mirroring `safety.py`'s `MoleculeSafety` pattern.
    Drop this if the tool doesn't need one -- have `CustomTool._run()` do
    its work directly instead.
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key


class CustomTool(BaseTool):
    name = "CustomTool"
    # LLM-facing description -- the agent picks tools by reading this, so be
    # precise about expected input format and what the tool returns.
    description = (
        "TODO: describe what this tool does, its expected input "
        "(e.g. 'a SMILES string'), and what it returns."
    )
    # BaseTool is a pydantic model -- non-default attributes must be
    # declared as typed fields here (see safety.py's `mol_safety` for the
    # same pattern) or assigning them in __init__ raises ValueError.
    helper: CustomToolHelper = None

    def __init__(self, api_key: str = None):
        super().__init__()
        self.helper = CustomToolHelper(api_key)

    def _run(self, query: str) -> str:
        # TODO: replace with real logic. Validate SMILES input the same way
        # the rest of the codebase does, e.g.:
        #   if not is_smiles(query):
        #       return "Invalid SMILES string"
        return "CustomTool is a scaffold -- implement _run() before use."

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()
