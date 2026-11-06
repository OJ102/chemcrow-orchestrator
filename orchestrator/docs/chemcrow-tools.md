# ChemCrow Tools: What They Do and Whether They Fit Our Orchestrator

Requested by the advisor (2026-10-29): look at ChemCrow's tools, check which we
can use, and how they would fit into the orchestrator. This survey is based on
reading the code in this repo (`chemcrow/tools/*.py`, `chemcrow/agents/tools.py`,
`chemcrow/utils.py`, `chemcrow/docker/`).

---

## 1. The short answer

- ChemCrow has **up to 19 tools** (`chemcrow/agents/tools.py::make_tools`).
  Several need API keys, so a given setup may load fewer.
- **About half are useful to us**, mostly as cheap **screening checks**
  (safety, patents, similarity, structural features) and as
  **finalist checks** (synthesizability).
- **Many don't fit**, because they look molecules up **in PubChem by name or
  CAS number**. Our molecules are *newly invented*, so PubChem has no entry
  for them and those tools can't answer.
- **They can't be imported directly** into the orchestrator. They're built
  for the old LangChain in `.venv-chemcrow` and return **sentences meant for an
  LLM**, not structured data. Each one we use needs a small adapter, or the MCP
  route we agreed on for the future.

---

## 2. Every tool, one by one

How to read the "Fit" column:
- **High** = clearly useful, we should plan for it
- **Medium** = useful in a narrower role
- **Low** = little value for invented OPV molecules
- **No** = don't use

| # | Tool (name the agent sees) | What it does (plain words) | Needs | Fit | Why |
|---|---|---|---|---|---|
| 1 | `python_repl` (LangChain) | Lets the LLM run any Python code | — | **No** | A security risk with no role in a deterministic loop |
| 2 | `wikipedia` (LangChain) | Searches Wikipedia | internet | Low | General background only; Kyle's knowledge base covers evidence |
| 3 | `Name2SMILES` | Name → SMILES (PubChem, then ChemSpace) | internet (ChemSpace key optional) | **Medium** | Turning a name a user types ("Y6") into a molecule. Useful for the future "objective → constraints" step. Won't know many OPV polymer names |
| 4 | `Mol2CAS` | Molecule → CAS registry number (PubChem) | internet | Low | Invented molecules have no CAS number |
| 5 | `SMILES2Name` | SMILES → common name (PubChem) | internet | Low | Same reason; nice for labelling *known* molecules in reports |
| 6 | `PatentCheck` | Is this molecule in a patent database? (SureChEMBL via `molbloom`) | `molbloom` (downloads its catalogue once) | **High** | Works on *any* SMILES, offline after the download. The sponsor's architecture lists "patent checks". A good **risk flag** for shortlisted candidates |
| 7 | `MolSimilarity` | How similar are two molecules? (Tanimoto on Morgan fingerprints) | RDKit only | **High** | This is the measure the advisor's **"chemically diverse" archive** needs, and it supports novelty checks against Kyle's data. It's ~5 lines of RDKit, so re-implement it rather than wrap it |
| 8 | `SMILES2Weight` | Molecular weight | RDKit only | Medium | A cheap size sanity check in validation |
| 9 | `FunctionalGroups` | Lists chemical groups present (nitro, nitrile, ketone, ...) using ~45 patterns | RDKit only | **Medium** | Could feed the advisor's **repair-route diagnosis** and **structured repair instructions** ("keep the nitrile groups"), and flag problem groups |
| 10 | `ExplosiveCheck` | Is it classed as explosive? (PubChem safety data, by CAS) | internet | Low | Needs a CAS number, so it can't judge invented molecules |
| 11 | `ControlChemCheck` | Is it on a list of controlled (chemical-weapon) substances? Falls back to similarity | local list + internet | **Medium** | Responsible-use safety screen; the similarity part works on any SMILES (see #12) |
| 12 | `SimilarityToControlChem` | Highest similarity to any controlled chemical (local list, `chemcrow/data/chem_wep_smi.csv`) | RDKit + local CSV | **High** | Cheap, offline, works on any SMILES. A sensible **safety screen before anything is handed to a lab** |
| 13 | `SafetySummary` | LLM-written safety summary from PubChem data | internet + LLM | Low | Needs CAS/PubChem; invented molecules have no data |
| 14 | `LiteratureSearch` | Finds papers (Semantic Scholar) and answers a question from them (paper-qa) | internet + LLM + **OpenAI embeddings key** | Medium | An evidence source, but it **overlaps Kyle's retriever (C1)** and needs OpenAI, which conflicts with running locally |
| 15 | `CustomTool` | Empty scaffold, placeholder for the OPV generator | — | **No** | Our orchestrator's `interfaces.py` slots replace this |
| 16 | `GetMoleculePrice` | Cheapest purchase price (ChemSpace) | ChemSpace API key | Low | Invented molecules can't be bought; maybe for building blocks later |
| 17 | `WebSearch` | Google search (SerpAPI) | SerpAPI key | Low | Unreliable evidence; Kyle's curated knowledge base is better |
| 18 | `ReactionPredict` | Predicts a reaction's product (IBM RXN, or a local Docker service) | IBM key or Docker | Low | Forward reaction prediction isn't part of our loop |
| 19 | `ReactionRetrosynthesis` | Plans how to *make* the molecule (IBM RXN, or local AiZynthFinder in Docker) | IBM key or Docker + LLM | **High (for finalists)** | The sponsor's deliverables include "evidence of synthesizability", and D1 plans AiZynthFinder for finalists. **Too slow for every candidate** (seconds to minutes each), so finalists only |

---

## 3. Where the useful ones would fit in our loop

```
START → start → retrieve → generate → validate → predict → decide ─┬─→ generate (revise)
                   ▲                      ▲          ▲              │
                   │                      │          │              └─→ finish ─→ finalist screening
          (A) Name2SMILES          (B) cheap screens  │                              (D) patent, retrosynthesis
              literature             safety similarity (C) gate/diagnosis
                                     size, groups        functional groups
```

| Where | Which tools | What they'd do | Who owns that part (D2) |
|---|---|---|---|
| **(A) Run setup / evidence** | `Name2SMILES`, (`LiteratureSearch`) | Resolve molecule names in the user's objective; extra evidence | Shashank (C2), with Kyle (C1) for evidence |
| **(B) Cheap screens, every candidate** (in or right after `validate`) | `SimilarityToControlChem`, `MolSimilarity`, `SMILES2Weight`, `FunctionalGroups` | Safety screen; diversity/novelty for the near-miss archive; size sanity; structural features | Jaxon (C3 owns validation, diversity, novelty), Shashank for the archive |
| **(C) Gate / repair diagnosis** | `FunctionalGroups` | Input to "repair the molecule, its partner, or the process?" and to the structured repair instructions | Arleen (gate) |
| **(D) Finalist screening** (after the shortlist, before handoff to DFT/lab) | `PatentCheck`, `ReactionRetrosynthesis` (AiZynthFinder) | IP risk flag; a synthesis route as evidence of synthesizability | Nyla (C5 shortlist/handoff) |

**How the results should come back:** as **structured data**, like our other
records (the advisor asked for machine-readable feedback), not ChemCrow's
sentences. For example:
```python
ScreenResult(check_code="control_chem_similarity", passed=True, score=0.21, detail="max similarity 0.21 < 0.35")
```
D1 fixes the list of risk flags, and adding values to it is a breaking change.
So screen results would go in a **separate field**, or in the rationale text,
not as new risk flags.

---

## 4. How to connect them (the technical problem)

ChemCrow's tools are LangChain `BaseTool` classes for the **old LangChain**
in `.venv-chemcrow`. The orchestrator runs in `.venv-langgraph`, so it
**can't import them**. Options:

| Option | What it means | Best for |
|---|---|---|
| **Re-implement** | Copy the few lines of logic (ChemCrow is MIT-licensed; keep the credit) into a small module in our venv | The simple RDKit/`molbloom` tools: #6, #7, #8, #9, #12. Each is just a few lines |
| **MCP server** | Run a small server inside `.venv-chemcrow` that offers the tools; the orchestrator calls it over a connection (the plan agreed on 2026-10-15) | The heavy/LLM/API tools: #3, #14, #19, and any future agent that picks tools itself |
| **Local web service** | Call ChemCrow's existing Docker services over HTTP | Retrosynthesis (#19, AiZynthFinder) |

Note: the team's D1 "build vs reuse" table says ChemCrow is a **design
reference only** ("we copy its tool-wrapper design rather than its code").
The advisor's request reopens that, so the team should update D1's decision.

---

## 5. Problems found in the ChemCrow code (worth knowing before reusing it)

1. **Local retrosynthesis is broken.** `reactions.py` sends `{"smiles": ...}`,
   but the Docker app (`docker/aizynthfinder/app.py`) reads `"target"`, so it
   would plan a route for an empty input.
2. **The retrosynthesis summary invents a step.** Its LLM prompt says
   *"add 15mL of DCM as a solvent in the first step"* whatever the route is.
   Don't present those summaries as real procedures.
3. **The safety tools hang without internet.** `MoleculeSafety` downloads a
   dataset when it starts and retries forever if offline (affects #10, #13).
4. **`LiteratureSearch` needs an OpenAI key** for embeddings, which goes against
   this fork's "runs fully locally" goal.
5. **Overconfident wording.** `SimilarityToControlChem` says "This substance is
   safe" whenever similarity is low. Low similarity to a weapons list isn't
   proof of safety.
6. **Everything returns sentences**, not data, so each tool needs an adapter
   before our code can use its result.

---

## 6. What the original repo and the paper add (checked 2026-10-29)

**Sources:** the original repo
[ur-whitelab/chemcrow-public](https://github.com/ur-whitelab/chemcrow-public)
(MIT) and the paper: Bran, Cox, Schilter, Baldassari, White & Schwaller,
"Augmenting large language models with chemistry tools", *Nature Machine
Intelligence* (2024), DOI 10.1038/s42256-024-00832-8; preprint
[arXiv:2304.05376](https://arxiv.org/abs/2304.05376) (v5, Oct 2023).

**Our fork's tool list matches the original.** `make_tools()` loads the same
tools under the same conditions; our fork only adds the `CustomTool`
scaffold. The original README says the package *"does not contain all the
tools described in the ChemCrow paper because of API usage restrictions."*

### Tools in the paper that are NOT in the code

| Paper tool | What it does | Backed by | Available to us? | Relevance |
|---|---|---|---|---|
| **ModifyMol** | Takes a molecule and returns a **modified version** by building its "local chemical space" (close analogs reachable by known reactions) | [SynSpace](https://github.com/whitead/synspace) (MIT, `pip install synspace`): 50 robust reactions (Hartenfeller et al.) run forwards and in reverse, plus purchasable building blocks (Mcule) | **Yes**, the library is public (marked "early-stage, in flux") | **High.** This is a working example of the advisor's "local, chemically meaningful changes", and each analog comes **with a synthesis route**. A candidate mechanism for revising near misses (Jaxon's area). Caveat: built for small drug-like molecules, so it needs testing on large OPV molecules |
| **Human** | The agent asks the user a question or for permission | LangChain | Yes | The same idea as our "set aside for human/DFT review" and D2's human-approved handoff |
| **NameRXN** | Names/classifies a reaction | NextMove (proprietary) | No | Low |
| **ReactionExecute** | Plans a synthesis and **runs it on a robot**, after asking permission | IBM RoboRXN (proprietary) | No | Out of scope (no robot; D2 hands off to people) |
| Name2SMILES's third fallback | Name → structure offline | **OPSIN** (open source) | Yes | Small: resolves systematic (IUPAC) names offline; our fork only tries PubChem/ChemSpace |

The paper's other tools (WebSearch, LitSearch, Python REPL, SMILES2Price,
Name2CAS, Similarity, PatentCheck, FuncGroups, SMILES2Weight, the three safety
tools, ReactionPredict, ReactionPlanner) correspond to the ones in section 2.
Note the paper describes **PatentCheck** as mainly a purchasability check (a
molbloom bloom filter); our fork's version uses the SureChEMBL patent catalogue.

### Lessons from the paper that support our design

1. **The closest case study to our project: discovering a new chromophore**
   (a light-absorbing molecule, like OPV materials). ChemCrow trained a
   Random Forest model on a chromophore library, using its Python tool, to hit a
   target absorption of **369 nm**. The molecule it proposed was made and
   measured at **336 nm**: close, but 33 nm off. **A prediction is not a
   result.** This supports D2's handoff to DFT/lab and the advisor's "never
   treat Chemprop scores as ground truth".
2. **An LLM can't be trusted to judge chemistry.** The authors found GPT-4,
   used as an evaluator, could not tell clearly wrong answers from correct
   ones, and favoured fluent-sounding text. Only expert chemists could. This
   supports our **deterministic, rule-based gate** (D2 rationale R3) and expert
   review of borderline candidates (the advisor's Dr. Hendsbee suggestion).
3. **Agents are not reproducible.** Over five runs of one task, ChemCrow
   misdescribed a molecule in two. This supports our seeded, deterministic
   loop and D2's pinned versions.
4. **Tools limit quality.** "Tools cannot fully rectify LLM's flawed
   reasoning", and results are capped by the tools. So the real value is in
   Jaxon's generator and Arleen's predictor, not in adding more tools.
5. **Safety stops are hard stops.** In the paper, a controlled-chemical match
   **stops execution**. If we add that screen, a match should end a
   candidate's path and never reach a lab handoff, not just add a warning.

## 7. Recommended next steps (for discussion)

1. **Start with the cheap, offline screens** (#7 similarity, #12 control-chem
   similarity, #6 patent check, plus #8/#9). Re-implemented, structured
   results, tested. Jaxon should agree, since validation, diversity and
   novelty are his (C3).
2. **Use #7 similarity for the advisor's bounded, diverse near-miss archive**,
   once Jaxon confirms the fingerprint settings and threshold.
3. **Retrosynthesis (#19) for finalists only**, via AiZynthFinder, after
   fixing bug 1, with Nyla (C5 handoff).
4. **Defer the PubChem-based tools.** They can't judge invented molecules.
5. **Update D1's build-vs-reuse entry for ChemCrow** to reflect the advisor's request.
6. **Evaluate SynSpace (the paper's ModifyMol) with Jaxon** as a way to make
   local, synthesizable revisions of near misses. First test it on a few
   molecules from Kyle's OPV2D data, since OPV molecules are far larger than
   the drug-like molecules it was built for.
