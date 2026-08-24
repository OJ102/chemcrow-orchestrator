"""Streamlit UI for non-technical users.

Paste a SMILES string -> see its 2D structure and a metrics panel. Ask the
ChemCrow agent to do something with the molecule -> the structure panel
updates live as soon as a molecule-transforming tool produces a new SMILES,
not just at the end of the full agent run.

Run with: streamlit run chemcrow/frontend/app.py
"""

import re
from typing import Optional

import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from rdkit import Chem

from chemcrow.agents import ChemCrow
from chemcrow.agents.chemcrow import CHEMCROW_MODEL
from chemcrow.frontend.depict import smiles_to_image
from chemcrow.frontend.metrics import compute_metrics
from chemcrow.utils import is_smiles

DEFAULT_SMILES = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin

# Tools whose output is a single product SMILES -- safe to auto-display.
# Lookup tools (e.g. Name2SMILES) are deliberately excluded: asking "what's
# the SMILES for benzene" shouldn't hijack the displayed molecule.
# ReactionRetrosynthesis is also excluded -- it returns a free-text recipe,
# not a SMILES.
CANVAS_DRIVING_TOOLS = {"ReactionPredict"}

# Best-effort scan of the agent's final-answer text for a SMILES string.
# Needed because there's no reaction tool wired up right now (ReactionPredict
# needs a local Docker server on :8051 -- see chemcrow/tools/reactions.py)
# that would drive CANVAS_DRIVING_TOOLS above; without a real tool to call,
# the agent just states the modified molecule in prose, e.g. "...the
# resulting SMILES is `CCO`." Highest-confidence patterns are tried first;
# a plain token scan is the last resort.
_SMILES_TOKEN = r"[A-Za-z0-9@+\-\[\]\(\)=#\\/%.]{3,}"
_LABELED_SMILES_RE = re.compile(
    rf"SMILES\s*(?:string)?\s*(?:is|:|=)\s*[`\"']?({_SMILES_TOKEN})[`\"']?",
    re.IGNORECASE,
)
_BACKTICK_RE = re.compile(r"`([^`]+)`")
_TOKEN_RE = re.compile(_SMILES_TOKEN)


def extract_smiles_from_text(text: str) -> Optional[str]:
    """Find a SMILES string in free text, or None.

    Not foolproof (a stray 3+ char token could coincidentally parse as a
    single-atom SMILES) -- prefers text explicitly labeled "SMILES: ..." or
    backtick-quoted, and only falls back to scanning all tokens (keeping the
    last valid one, since the final answer usually states the result last)
    if neither of those matches.
    """
    for pattern in (_LABELED_SMILES_RE, _BACKTICK_RE):
        for match in pattern.finditer(text):
            candidate = match.group(1).strip().strip(".,;")
            if is_smiles(candidate):
                return candidate

    last_valid = None
    for match in _TOKEN_RE.finditer(text):
        candidate = match.group(0).strip(".,;")
        if is_smiles(candidate):
            last_valid = candidate
    return last_valid


class MoleculeCanvasCallback(BaseCallbackHandler):
    """Keeps the structure panel in sync with molecules the agent produces.

    Writes straight into pre-created placeholders instead of calling
    `st.rerun()` -- a rerun mid-callback would abort `agent.run()` partway
    through (`RerunException` unwinds the whole script).
    """

    def __init__(self, img_slot, smiles_slot, metrics_slot):
        self.img_slot = img_slot
        self.smiles_slot = smiles_slot
        self.metrics_slot = metrics_slot
        self._last_tool_name: Optional[str] = None

    def on_tool_start(self, serialized, input_str, **kwargs):
        self._last_tool_name = serialized.get("name")

    def on_tool_end(self, output, **kwargs):
        if self._last_tool_name not in CANVAS_DRIVING_TOOLS:
            return
        candidate = output.strip() if isinstance(output, str) else ""
        if not is_smiles(candidate):
            return
        st.session_state.smiles = candidate
        render_molecule(candidate, self.img_slot, self.smiles_slot, self.metrics_slot)


def render_molecule(smiles: str, img_slot, smiles_slot, metrics_slot) -> None:
    """Draw one molecule's image, SMILES text, and metrics into given slots."""
    image = smiles_to_image(smiles)
    if image is None:
        img_slot.error("Invalid SMILES string.")
        smiles_slot.empty()
        metrics_slot.empty()
        return

    img_slot.image(image, width="stretch")
    smiles_slot.code(smiles, language=None)

    mol = Chem.MolFromSmiles(smiles)
    metrics = compute_metrics(mol)
    with metrics_slot.container():
        cols = st.columns(len(metrics)) if metrics else []
        for col, (name, value) in zip(cols, metrics.items()):
            col.metric(name, value)


def main() -> None:
    st.set_page_config(page_title="ChemCrow", page_icon="\U0001F9EA", layout="wide")
    st.title("ChemCrow")
    st.caption(f"Local agent model: {CHEMCROW_MODEL} (via Ollama)")

    if "smiles" not in st.session_state:
        st.session_state.smiles = DEFAULT_SMILES
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # list[(role, text)]
    if "agent" not in st.session_state:
        with st.spinner(f"Loading {CHEMCROW_MODEL}..."):
            st.session_state.agent = ChemCrow(verbose=False)

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Molecule")
        smiles_input = st.text_input("SMILES string", value=st.session_state.smiles)
        if smiles_input != st.session_state.smiles:
            if is_smiles(smiles_input):
                st.session_state.smiles = smiles_input
            elif smiles_input:
                st.error("Invalid SMILES string.")

        img_slot = st.empty()
        smiles_slot = st.empty()
        metrics_slot = st.empty()
        render_molecule(st.session_state.smiles, img_slot, smiles_slot, metrics_slot)

    with right:
        st.subheader("Ask ChemCrow")
        for role, text in st.session_state.chat_history:
            with st.chat_message(role):
                st.write(text)

        prompt = st.chat_input("e.g. What is the molecular weight of this molecule?")
        if prompt:
            st.session_state.chat_history.append(("user", prompt))
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                status = st.status("Thinking...", expanded=True)
                callback = MoleculeCanvasCallback(img_slot, smiles_slot, metrics_slot)
                full_prompt = (
                    f"The current molecule under discussion has SMILES "
                    f"'{st.session_state.smiles}'. {prompt}"
                )
                try:
                    answer = st.session_state.agent.run(full_prompt, callbacks=[callback])
                    new_smiles = extract_smiles_from_text(answer)
                    if new_smiles and new_smiles != st.session_state.smiles:
                        st.session_state.smiles = new_smiles
                        render_molecule(new_smiles, img_slot, smiles_slot, metrics_slot)
                except Exception as exc:  # keep the UI alive on agent errors
                    answer = f"Error running agent: {exc}"
                status.update(label="Done", state="complete", expanded=False)
                st.write(answer)

            st.session_state.chat_history.append(("assistant", answer))


if __name__ == "__main__":
    main()
