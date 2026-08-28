[![DOI:10.48550/arXiv.2304.05376](https://zenodo.org/badge/DOI/10.48550/arXiv.2304.05376.svg)](https://doi.org/10.48550/arXiv.2304.05376)
[![DOI](https://zenodo.org/badge/649361700.svg)](https://zenodo.org/doi/10.5281/zenodo.10884638)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/chemcrow_dark_bold.png" width='100%'>
  <source media="(prefers-color-scheme: light)" srcset="assets/chemcrow_light_bold.png" width='100%'>
  <img alt="ChemCrow logo" src="/assets/" width="100%">
</picture>

<br></br>

# chemcrow-orchestrator

A local-first, self-hosted fork of **[ChemCrow](https://github.com/ur-whitelab/chemcrow-public)** — the
open-source LLM chemistry agent from Bran, Cox, Schilter, Baldassari, White & Schwaller — re-engineered to run
entirely offline against a local model, with a live molecule-visualization UI and a scaffolded integration point
for a generative organic-photovoltaic (OPV) molecule model. Built as the foundation for an accelerated
materials-discovery pipeline where an LLM agent proposes, evaluates, and iterates on candidate molecules directly.

## Engineering work in this fork

- **Removed the OpenAI dependency entirely.** Re-pointed the agent's `ChatOpenAI` client at a local Ollama
  server's OpenAI-compatible endpoint instead of pulling in a separate `ChatOllama` client — a deliberately
  minimal integration that required zero new dependencies and kept the (old, tightly pinned) `langchain` version
  the rest of the codebase depends on completely untouched. The agent now runs fully offline, with no API key,
  on a locally hosted model.
- **Solved a real UI-sync problem, not just a display.** The agent can change the active molecule two different
  ways — by calling a tool that returns a product SMILES, or simply by stating the new SMILES in its final
  answer text. Built a live-updating Streamlit interface that catches both: a tool-output callback for the
  former, and a dedicated SMILES-extraction pass over the agent's answer for the latter — so the 2D structure
  panel (rendered locally with RDKit) always reflects what the agent actually did, without a page reload and
  without interrupting the agent mid-run.
- **Diagnosed and fixed three silent, install-breaking gaps in the upstream package** that a plain `pip install`
  does not surface until runtime: a required dependency left commented out in `setup.py`, an undeclared
  `setuptools` version ceiling, and a model-selection code path with no local-model branch. Fixed all three and
  verified a clean install end-to-end.
- **Designed a portable local-config layer** for model/endpoint selection: a gitignored config file with a
  documented template, safe public fallback defaults, and environment-variable overrides — so machine-specific
  setup (which model, which port) never has to touch version control or the codebase itself.
- **Verified with real, non-trivial checks** — not just "it imports": end-to-end agent runs against the live
  local model, Streamlit's script-testing harness driving the actual UI through multiple interaction paths, and
  targeted unit tests on the SMILES-sync logic's allow-list and validation behavior.

## What ChemCrow does

ChemCrow uses an LLM as a reasoning engine (a ReAct-style agent, via Langchain) that answers chemistry questions
by calling out to a set of tools — RDKit for structure and property calculations, PubChem and Chem-Space for
lookups, paper-qa for literature search, RXN4Chem for reaction prediction and retrosynthesis, and more — rather
than trying to know chemistry from its own weights alone. All of that agent design and tooling is the original
authors' work; see [Credit and citation](#credit-and-citation) below.

## What this fork adds, concretely

- **Local reasoning model via Ollama, not OpenAI.** `chemcrow/agents/chemcrow.py` points `ChatOpenAI` at an
  Ollama server's OpenAI-compatible `/v1` endpoint instead of `api.openai.com`. Which model and which Ollama
  server (host/port) is read from `chemcrow/agents/ollama_local_config.py`, a gitignored file (template at
  `ollama_local_config.py.example`) — machine-specific setup, not something that belongs in version control. It
  falls back to a public-safe default (`qwen3.5:35b` on `localhost:11434`) if that file isn't present.
- **A Streamlit UI for non-technical users** (`chemcrow/frontend/app.py`, run with
  `streamlit run chemcrow/frontend/app.py`): paste a SMILES string, see its 2D structure and a metrics panel next
  to a chat pane for talking to the agent. The structure panel stays live-synced to whatever molecule the agent
  is currently discussing.
- **A scaffolded extension point for a new tool** (`chemcrow/tools/custom.py`, registered in
  `chemcrow/agents/tools.py::make_tools()`) — currently a placeholder, built out for the use case below.

## Intended use: accelerated OPV molecule discovery (placeholder)

The motivating use case for this fork is pairing the ChemCrow agent with a generative model for organic
photovoltaic (OPV) molecules — the agent would use the generator as just another tool: propose candidate
structures, evaluate them with the existing chemistry tools (safety checks, similarity, property lookups, and
so on), and iterate, rather than a human manually driving a separate molecule-generation script. The goal is to
shorten the loop between "propose a candidate" and "know whether it's worth synthesizing" for OPV material
discovery.

This integration does not exist yet. `chemcrow/tools/custom.py` (`CustomTool`) is a scaffold — it's wired into
the agent's tool list so the plumbing (registration, description, input/output shape) is in place, but `_run()`
is a placeholder that returns a fixed string. Likewise, the Streamlit UI's metrics panel
(`chemcrow/frontend/metrics.py`) already has a slot for PCE (power conversion efficiency) — the metric an OPV
generator/predictor would report — but it currently always returns `None`/"N/A". Both are meant to be filled in
once the actual OPV generator is wired up as the real implementation behind `CustomTool` and `_pce_stub()`.

## Installation

This fork targets Python 3.10 (the pinned dependency versions — `openai==0.27.8`, `langchain==0.0.275`, etc. —
are unlikely to build cleanly on newer Python).

```bash
git clone https://github.com/OJ102/chemcrow-orchestrator.git
cd chemcrow-orchestrator
pip install -e .
# three upstream install gaps this fork's setup.py fixes:
#   paper-scraper is a git dependency (needed at import time, was left commented out)
#   setuptools<81 (chemcrow/tools/safety.py imports pkg_resources, dropped in newer setuptools)
#   (see "Engineering work" above for the third: the local-model code path itself)
```

You'll also need [Ollama](https://ollama.com) installed and serving a model locally:

```bash
ollama serve
ollama pull qwen3.5:35b   # or whichever model you configure below
```

Optionally, copy `chemcrow/agents/ollama_local_config.py.example` to `ollama_local_config.py` in the same
directory and set `CHEMCROW_MODEL` / `OLLAMA_BASE_URL` for your machine. Without it, the defaults above apply.

## Usage

In a Python session:
```python
from chemcrow.agents import ChemCrow

chem_model = ChemCrow(temp=0.1, streaming=False)  # uses your local Ollama model by default
chem_model.run("What is the molecular weight of tylenol?")
```

Or launch the UI:
```bash
streamlit run chemcrow/frontend/app.py
```

To use OpenAI instead of a local model, pass a `gpt-*` model name and an API key:
```python
chem_model = ChemCrow(model="gpt-4-0613", openai_api_key="your-openai-api-key", temp=0.1, streaming=False)
```

You can optionally use Serp API for the web search tool:
```bash
export SERP_API_KEY=your-serpapi-api-key
```

## Self-hosting of some tools

By default, ChemCrow relies on the RXN4Chem API for retrosynthetic planning and reaction product prediction. This can however be slow and depends on you having an API key.

Optionally, you can also self host these tools by running some pre-made docker images.

Run

```bash
docker run --gpus all -d -p 8051:5000 doncamilom/rxnpred:latest
docker run --gpus all -d -p 8052:5000 doncamilom/retrosynthesis:latest
```

Now ChemCrow can be used like this:

```python
from chemcrow.agents import ChemCrow

chem_model = ChemCrow(temp=0.1, streaming=False, local_rxn=True)
chem_model.run("What is the product of the reaction between styrene and dibromine?")
```

## Note

This package does not contain all the tools described in the [ChemCrow paper](https://arxiv.org/abs/2304.05376) because
of API usage restrictions, and this fork changes the reasoning model besides. Results will not match that paper.

## Credit and citation

The agent architecture, prompts, and chemistry tools this fork builds on are the work of the original ChemCrow
authors. Please cite their paper for the underlying method:

Bran, Andres M., et al. "ChemCrow: Augmenting large-language models with chemistry tools." arXiv preprint arXiv:2304.05376 (2023).

```bibtex
@article{bran2023chemcrow,
      title={ChemCrow: Augmenting large-language models with chemistry tools},
      author={Andres M Bran and Sam Cox and Oliver Schilter and Carlo Baldassari and Andrew D White and Philippe Schwaller},
      year={2023},
      eprint={2304.05376},
      archivePrefix={arXiv},
      primaryClass={physics.chem-ph},
      publisher={arXiv}
}
```

Original repository: [ur-whitelab/chemcrow-public](https://github.com/ur-whitelab/chemcrow-public) (MIT licensed — see [`LICENSE`](LICENSE)). All released experiments from the original authors: [ChemCrow runs](https://github.com/ur-whitelab/chemcrow-runs).
