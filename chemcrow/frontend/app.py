"""Streamlit UI for non-technical users.

Paste a SMILES string -> see its 2D structure and a metrics panel. Ask the
ChemCrow agent to do something with the molecule -> the structure panel
updates live as soon as a molecule-transforming tool produces a new SMILES,
not just at the end of the full agent run.

Run with: streamlit run chemcrow/frontend/app.py
"""

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
                except Exception as exc:  # keep the UI alive on agent errors
                    answer = f"Error running agent: {exc}"
                status.update(label="Done", state="complete", expanded=False)
                st.write(answer)

            st.session_state.chat_history.append(("assistant", answer))


if __name__ == "__main__":
    main()
