"""chemcrow_tools.py: ChemCrow's own tools, made callable by the local model.

Runs in ~/llm/chemcrow-venv (Python 3.10 + ChemCrow's old LangChain 0.0.275).

WHAT THIS FILE DOES
    ChemCrow's tools are LangChain "BaseTool" classes. Each takes ONE text input
    (usually SMILES) and returns a sentence. This file:
      1. creates the real ChemCrow tool objects (their code does the actual work),
      2. describes each one to the model in the native tool-calling format, and
      3. wraps each call with three helpers for a small model:
           * NAME RESOLUTION: the model passes a molecule NAME ("aspirin", "Y6") or a
             SMILES. Our code turns names into SMILES (Kyle's OPV list first, then
             PubChem), so the model never has to copy 100+-character SMILES.
           * FORMAT CONVERSION IN CODE: tools that need a CAS number (ExplosiveCheck,
             SafetySummary) get one automatically (via ChemCrow's Mol2CAS), instead
             of the model having to chain that step itself.
           * A TIME LIMIT (TOOL_TIMEOUT_S) so a stuck network call can't freeze the agent.

WHICH TOOLS (12 of ChemCrow's + our 4 OPV tools; see chemcrow-tools.md for all 19)
    Left OUT on purpose:
      python_repl       lets the model run any code: a security risk
      CustomTool        an empty placeholder
      LiteratureSearch  needs an OpenAI API key
      GetMoleculePrice, WebSearch, ReactionPredict, ReactionRetrosynthesis
                        need API keys we don't have (ChemSpace, SerpAPI, IBM RXN)

TOOLSETS (small groups, so the small model chooses among a few tools, not 16)
    lookup, analysis, safety, patents, opv. See TOOLSETS at the bottom.

ChemCrow is MIT-licensed (Bran, Cox, Schilter, Baldassari, White & Schwaller).
"""

import concurrent.futures  # to run a tool with a time limit
import functools  # partial(): a function with some arguments pre-filled
import re
from pathlib import Path

from rdkit import Chem

import opv_tools  # our 4 OPV tools and Kyle's molecule list (same folder)
from chemcrow.tools import (ControlChemCheck, ExplosiveCheck, FuncGroups, MolSimilarity,
                            PatentCheck, Query2CAS, Query2SMILES, SafetySummary,
                            SimilarControlChemCheck, SMILES2Name, SMILES2Weight)
from chemcrow.utils import is_cas, pubchem_query2smiles

TOOL_TIMEOUT_S = 90  # seconds one tool call may take
# SafetySummary asks the LLM to summarise ~11 PubChem sections one after another,
# then once more for the final summary, so it needs minutes, not seconds.
SLOW_TOOL_TIMEOUT_S = {"SafetySummary": 360}
MODEL = "qwen3.8-27b"
API_KEY = (Path.home() / "llm" / ".api_key").read_text().strip()


class ToolError(Exception):
    """A problem the model can fix (unknown molecule, bad input). Sent back with a hint."""


# ---------------------------------------------------------------------------
# Helpers: turn what the model wrote into what ChemCrow's tools expect
# ---------------------------------------------------------------------------

def to_smiles(molecule: str) -> str:
    """Name or SMILES -> SMILES. Order: Kyle's OPV list, then valid SMILES, then PubChem."""
    text = str(molecule).strip()
    entry = opv_tools.MOLECULES.get(opv_tools._norm(text))  # 1. an OPV name from Kyle's list
    if entry is not None:
        return Chem.MolToSmiles(entry["mol"])
    if " " not in text and Chem.MolFromSmiles(text) is not None:  # 2. already a SMILES
        return text
    try:  # 3. a common chemical name, looked up in PubChem by ChemCrow's own helper
        smiles = pubchem_query2smiles(text)
        # ChemCrow's helper RETURNS an error sentence (instead of raising) when the
        # lookup fails, so check that the answer really is a molecule.
        if Chem.MolFromSmiles(smiles) is not None:
            return smiles
    except Exception:
        pass
    raise ToolError(f"Couldn't find a structure for '{text}' (not in the OPV list or "
                        f"PubChem). Check the spelling, or give the SMILES.")


def to_cas(molecule: str) -> str:
    """Name/SMILES/CAS -> CAS number (via ChemCrow's Mol2CAS), for tools that need CAS."""
    text = str(molecule).strip()
    if is_cas(text):
        return text
    answer = _TOOL_OBJECTS["Mol2CAS"]._run(text)
    match = re.search(r"\d{2,7}-\d{2}-\d", answer)
    if not match:  # invented/OPV molecules usually have no CAS number
        raise ToolError(f"No CAS number found for '{text}' ({answer.strip()[:120]}). "
                        f"This tool only works for molecules listed in PubChem.")
    return match.group(0)


def _local_llm():
    """A LangChain chat model pointing at OUR vLLM server (SafetySummary writes text with it)."""
    from langchain.chat_models import ChatOpenAI  # ChemCrow's old LangChain
    return ChatOpenAI(model_name=MODEL, temperature=0.7, request_timeout=120,
                      openai_api_key=API_KEY, openai_api_base="http://127.0.0.1:8200/v1",
                      # extra fields passed through to vLLM: no "thinking" for summaries
                      model_kwargs={"chat_template_kwargs": {"enable_thinking": False}})


# The real ChemCrow tool objects. Creating the safety tools downloads a small
# dataset (ClinTox) once per start, so the first start takes a few seconds.
_TOOL_OBJECTS = {
    "Name2SMILES": Query2SMILES(),
    "Mol2CAS": Query2CAS(),
    "SMILES2Name": SMILES2Name(),
    "PatentCheck": PatentCheck(),
    "MolSimilarity": MolSimilarity(),
    "SMILES2Weight": SMILES2Weight(),
    "FunctionalGroups": FuncGroups(),
    "ExplosiveCheck": ExplosiveCheck(),
    "ControlChemCheck": ControlChemCheck(),
    "SimilarityToControlChem": SimilarControlChemCheck(),
    "SafetySummary": SafetySummary(llm=_local_llm()),
}


def _wikipedia(query: str) -> str:
    """LangChain's Wikipedia tool (one of ChemCrow's), created on first use."""
    from langchain.agents import load_tools
    return load_tools(["wikipedia"])[0].run(query)


# ---------------------------------------------------------------------------
# What runs for each tool name: (arguments from the model) -> ChemCrow's answer
# ---------------------------------------------------------------------------

_RUNNERS = {
    "Name2SMILES":      lambda name: _TOOL_OBJECTS["Name2SMILES"]._run(name),
    "Mol2CAS":          lambda molecule: _TOOL_OBJECTS["Mol2CAS"]._run(molecule),
    "SMILES2Name":      lambda molecule: _TOOL_OBJECTS["SMILES2Name"]._run(to_smiles(molecule)),
    "PatentCheck":      lambda molecule: _TOOL_OBJECTS["PatentCheck"]._run(to_smiles(molecule)),
    "MolSimilarity":    lambda molecule_a, molecule_b: _TOOL_OBJECTS["MolSimilarity"]._run(
                            f"{to_smiles(molecule_a)}.{to_smiles(molecule_b)}"),
    "SMILES2Weight":    lambda molecule: _TOOL_OBJECTS["SMILES2Weight"]._run(to_smiles(molecule)),
    "FunctionalGroups": lambda molecule: _TOOL_OBJECTS["FunctionalGroups"]._run(to_smiles(molecule)),
    "ExplosiveCheck":   lambda molecule: _TOOL_OBJECTS["ExplosiveCheck"]._run(to_cas(molecule)),
    "ControlChemCheck": lambda molecule: _TOOL_OBJECTS["ControlChemCheck"]._run(to_smiles(molecule)),
    "SimilarityToControlChem": lambda molecule: _TOOL_OBJECTS["SimilarityToControlChem"]._run(to_smiles(molecule)),
    "SafetySummary":    lambda molecule: _TOOL_OBJECTS["SafetySummary"]._run(to_cas(molecule)),
    "Wikipedia":        lambda query: _wikipedia(query),
}

_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=4)  # runs tools with a time limit


def _run_chemcrow(tool, **kwargs) -> dict:
    """Run one ChemCrow tool with its time limit; return its answer as {"result": text}."""
    limit = SLOW_TOOL_TIMEOUT_S.get(tool, TOOL_TIMEOUT_S)
    future = _POOL.submit(_RUNNERS[tool], **kwargs)
    try:
        text = future.result(timeout=limit)
    except concurrent.futures.TimeoutError:
        raise ToolError(f"{tool} took longer than {limit}s (network?). Try another tool.")
    return {"result": str(text).strip()[:3000]}  # cap very long answers


# Every tool the agent can run: ChemCrow's (wrapped) + our OPV tools.
# functools.partial "pre-fills" the tool name. (A lambda with a default argument
# called `name` broke Name2SMILES, whose own argument is also called `name`.)
TOOLS = {tool: functools.partial(_run_chemcrow, tool) for tool in _RUNNERS}
TOOLS.update(opv_tools.TOOLS)

# ---------------------------------------------------------------------------
# Descriptions the model sees. ChemCrow's own one-line descriptions, rewritten to
# say "name or SMILES" where our name resolution applies.
# ---------------------------------------------------------------------------

_MOL = {"type": "string", "description": "molecule name (e.g. 'aspirin', 'Y6') or SMILES"}


def _schema(name, description, params):
    """One tool description in OpenAI's function format; every listed argument is required."""
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": params, "required": list(params)}}}


SCHEMAS = {s["function"]["name"]: s for s in [
    _schema("Name2SMILES", "Input a molecule name, returns its SMILES (from PubChem).",
            {"name": {"type": "string", "description": "molecule name"}}),
    _schema("Mol2CAS", "Input a molecule (name or SMILES), returns its CAS registry number.", {"molecule": _MOL}),
    _schema("SMILES2Name", "Input a molecule, returns its common name (from PubChem).", {"molecule": _MOL}),
    _schema("PatentCheck", "Returns whether a molecule is patented.", {"molecule": _MOL}),
    _schema("MolSimilarity", "Returns the Tanimoto similarity (0 to 1) of two molecules.",
            {"molecule_a": _MOL, "molecule_b": _MOL}),
    _schema("SMILES2Weight", "Returns the molecular weight of a molecule.", {"molecule": _MOL}),
    _schema("FunctionalGroups", "Returns the list of functional groups in a molecule.", {"molecule": _MOL}),
    _schema("ExplosiveCheck", "Returns whether a molecule is classified as explosive (PubChem GHS data).", {"molecule": _MOL}),
    _schema("ControlChemCheck", "Returns whether a molecule is a controlled chemical (or highly similar to one).", {"molecule": _MOL}),
    _schema("SimilarityToControlChem", "Returns the highest similarity of a molecule to known controlled chemicals.", {"molecule": _MOL}),
    _schema("SafetySummary", "Returns a summary of safety information (operator safety, GHS, environmental risks). Only for molecules in PubChem.", {"molecule": _MOL}),
    _schema("Wikipedia", "Searches Wikipedia and returns a short summary. For general background questions.",
            {"query": {"type": "string", "description": "what to search for"}}),
]}
SCHEMAS.update({s["function"]["name"]: s for s in opv_tools.TOOL_SCHEMAS})

# ---------------------------------------------------------------------------
# Toolsets: small groups of tools for one kind of task
# ---------------------------------------------------------------------------

TOOLSETS = {
    "lookup":   {"description": "identify a molecule: its SMILES, name, CAS number, or background information",
                 "tools": ["Name2SMILES", "SMILES2Name", "Mol2CAS", "Wikipedia"]},
    "analysis": {"description": "structure and properties: molecular weight, functional groups, similarity of two molecules",
                 "tools": ["SMILES2Weight", "FunctionalGroups", "MolSimilarity"]},
    "safety":   {"description": "safety: controlled chemicals, explosives, hazards and safety summaries",
                 "tools": ["ControlChemCheck", "SimilarityToControlChem", "ExplosiveCheck", "SafetySummary"]},
    "patents":  {"description": "intellectual property: is a molecule patented",
                 "tools": ["PatentCheck"]},
    "opv":      {"description": "the team's OPV donor/acceptor database: look up OPV molecules, find similar donors or acceptors, compare them",
                 "tools": ["find_molecule", "most_similar", "properties", "compare"]},
}
