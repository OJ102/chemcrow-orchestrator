# Local model on the GPU cluster (vLLM + Qwen3.8-27B)

Scripts for serving our local language model on `cs-gpu-00.digitfut.uc.edu`, for
the ChemCrow tool-calling evaluation. Set up and first tested 2026-11-14.

| File | What it does |
|---|---|
| [setup_vllm.sh](setup_vllm.sh) | One-time setup: Python venv, vLLM, and the model download (all in `~/llm`) |
| [serve_vllm.sh](serve_vllm.sh) | Starts the model server on **GPU 2** (every setting is explained in comments) |
| [smoke_test.py](smoke_test.py) | Quick "is it alive and can it call tools?" check |
| [opv_tools.py](opv_tools.py) | 4 **real** tools over Kyle's 4,023 OPV molecules (look up, most similar, properties, compare) |
| [agent.py](agent.py) | **Multi-step tool calling**: each tool result is fed back so the model can pick the next call |

The copies that actually run live on the cluster in `~/llm/` (= `/home2/sathiysk/llm`).
After editing a script here, copy it over again (see "Updating a script" below).

## What's on the cluster

```
~/llm/
├── venv/          the SERVER's Python 3.12 environment: vLLM 0.33.0, transformers 5.20, torch 2.14 (15 GB).
│                  Install nothing else here: other packages can upgrade numpy past what vLLM accepts.
├── agent-venv/    the AGENT's environment: openai, rdkit, pandas
├── data/          classified_opv_database.csv (Kyle's molecule list)
├── opv_tools.py, agent.py
├── hf/            the downloaded model Qwen/Qwen3.8-27B, BF16 (70 GB)
├── .api_key       the server's password (random; only readable by sathiysk)
├── logs/          setup.log, vllm.log
├── tmp/, uv-cache/  temporary files (kept off the 95%-full root disk)
└── setup_vllm.sh, serve_vllm.sh, smoke_test.py
```

## Logging in

The laptop has an SSH key, so `ssh uc-gpu` logs in without a password. The
`uc-gpu` alias is in `C:\Users\abyss\.ssh\config`. To revoke the key, delete
its line (`sathiysk-laptop-cs-gpu-00`) from `~/.ssh/authorized_keys` on the cluster.

## Everyday commands (from the laptop's PowerShell)

The server runs inside **tmux**, a program that keeps a terminal session alive on
the cluster after you disconnect.

```powershell
# Is it running? (lists tmux sessions; "vllm" = the server)
ssh uc-gpu 'tmux ls'

# Start the server (takes ~15 minutes the first time, while it compiles;
# ready when the log says "Application startup complete")
ssh uc-gpu 'tmux new-session -d -s vllm "bash ~/llm/serve_vllm.sh 2>&1 | tee ~/llm/logs/vllm.log"'

# Stop the server (frees GPU 2)
ssh uc-gpu 'tmux kill-session -t vllm'

# Watch the server's screen live (leave again with Ctrl+B, then D)
ssh -t uc-gpu 'tmux attach -t vllm'

# Run the smoke test
ssh uc-gpu '~/llm/venv/bin/python ~/llm/smoke_test.py'

# Who is using which GPU?
ssh uc-gpu nvidia-smi
```

**Using the model from the laptop.** The server only accepts connections from
the cluster itself. To reach it from the laptop, open an **SSH tunnel**: a
private pipe that makes the laptop's port 8200 lead to the cluster's port 8200.
Keep this running in its own terminal:

```powershell
ssh -N -L 8200:127.0.0.1:8200 uc-gpu
```

Then any program on the laptop can use `http://127.0.0.1:8200/v1` with model
name `qwen3.8-27b`. The API key is the content of `~/llm/.api_key` on the
cluster (`ssh uc-gpu cat ~/llm/.api_key`). Don't commit it to git.

**Updating a script** after editing it here (strips Windows line endings):

```bash
tr -d '\r' < cluster/serve_vllm.sh | ssh uc-gpu 'cat > ~/llm/serve_vllm.sh'
```

## First results (2026-11-14)

Server: GPU 2 only, BF16, 32,768-token limit, text only. Memory after loading:
**15.4 GB left for the working cache = room for ~220,000 tokens**, i.e. about
6–7 full-length conversations at the same time.

Smoke test (question: *"What is the molecular weight of benzene, SMILES c1ccccc1?"*;
tools offered: `molecular_weight` and `patent_check`):

| Test | Time | Result |
|---|---|---|
| 1. plain question, thinking off | 1.8 s | correct one-sentence answer |
| 2. tool choice, thinking off, `auto` | 1.2 s | ✅ `molecular_weight({"smiles": "c1ccccc1"})` |
| 3. tool choice, thinking off, `required` (format forced) | 3.1 s | ✅ same |
| 4. tool choice, thinking on, `auto` | 2.5 s | ✅ same (130 characters of thinking first) |

**How to read it:**
- **Speed:** about **20–27 tokens per second** for one request. A token is about
  ¾ of a word. That's normal for a 27b model on one A100, and the GPU can serve
  several requests at once, so an eval with many cases runs in parallel.
- **`auto`** = the model decides by itself whether to call a tool. **`required`** =
  vLLM forces it to produce a valid call to one of the listed tools. The first
  `required` call is slower (3.1 s) because vLLM builds the format rules once;
  later ones reuse them.
- **Thinking on** = the model writes hidden reasoning before answering. Slower,
  sometimes more accurate. Whether it helps our small model is a question for the
  real eval.
- One easy question proves the plumbing works, **not** that the model is good at
  tool choice. That's what the eval (next step) measures.

## Multi-step tool calling (agent.py)

**What it does:** the model gets a question and 4 real tools. Whenever it asks for
a tool, *our code* runs it and sends the result back, and the model uses that
result to choose its next call. This repeats until it answers. Our code stays in
charge: at most **8 model turns** and **3 tool errors** per question, then the
model must answer with what it has. Errors (e.g. an unknown name) go back to the
model with suggestions, so it can fix its call.

**Optimized for a small model** (Om's guidance):
- Tools pass short **names** (`"Y6"`), never 100+-character SMILES the model could mistype.
- Names match forgivingly (`y-6` = `Y6`), and unknown names come back with the
  5 closest names.
- 4 tools only, with short descriptions; 5 short numbered rules in the system prompt.
- A fixed random seed, so the same question gives the same run (repeatable demos).

**Run it** (the server must be running):

```powershell
ssh uc-gpu 'cd ~/llm && ~/llm/agent-venv/bin/python agent.py'                    # 4 demo questions
ssh uc-gpu 'cd ~/llm && ~/llm/agent-venv/bin/python agent.py --think'            # model reasons first
ssh uc-gpu 'cd ~/llm && ~/llm/agent-venv/bin/python agent.py -q "Is ITIC more similar to Y6 or to IT-4F?"'
```

> **Before presenting:** run it once first. The first start after a while can
> take a minute or more, because Python loads RDKit and pandas from the network
> drive. The model itself answers each question in about 3–15 s.

**Demo results (2026-11-14, thinking off)**: every answer is correct, from real data:

```
QUESTION: Find the 3 acceptors most similar to Y6, then tell me which of those 3 has the highest molecular weight.
  step 1  -> most_similar({"name": "Y6", "role": "acceptor", "k": 3})
           <- EH-HD4F 0.975, SY2 0.939, BT-L4F 0.938
  step 2  -> properties({"name": "EH-HD4F"})   <- molecular_weight 1564.2
  step 2  -> properties({"name": "SY2"})       <- molecular_weight 1484.9
  step 2  -> properties({"name": "BT-L4F"})    <- molecular_weight 1552.1
  ANSWER: ... EH-HD4F has the highest molecular weight at 1564.2 g/mol (vs. BT-L4F at 1552.1 and SY2 at 1484.9).
  [3 model calls, 4 tool calls, 0 errors, 8.8s]

QUESTION: How many aromatic rings does IT-4FF have? I may have misspelled the name; if so, use the closest match.
  step 1  -> find_molecule({"name": "IT-4FF"})
           <- ERROR No molecule named 'IT-4FF'. Closest names: ['IT-4F', 'IT6-4F', ...]
  step 2  -> properties({"name": "IT-4F"})      <- aromatic_rings 11
  ANSWER: There's no molecule named "IT-4FF"; the closest match is IT-4F (an acceptor), which has 11 aromatic rings.
  [3 model calls, 2 tool calls, 1 errors, 4.3s]
```

**How to explain it:**
- **Step 2 uses step 1's output:** the model only knew the names `EH-HD4F`,
  `SY2`, `BT-L4F` because the first tool returned them. That's sequential tool
  calling. Several calls with the same step number were requested together, in one
  model turn (the model asked for all three properties at once).
- **Error recovery:** the tool's error is fed back like any result. The model read
  the suggestions and retried with the right name. Without "use the closest
  match" in the question, it instead refuses to guess and asks which one you
  meant. That's the safer behavior for research.
- **Thinking on vs off:** both got every question right. Thinking took about
  2× longer (9–17 s vs 4–9 s per question), so **off is the better default**
  here. The real eval will test this on many more questions.
- Every run is saved as a full transcript in `~/llm/logs/agent/` (JSON).

## Settings to remember

- Qwen's recommended sampling (from the model card): thinking off:
  `temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5`; thinking on:
  `temperature 1.0, top_p 0.95, top_k 20`. Turn thinking off per request with
  `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`.
- GPUs: ours are **2 and 4** (the other capstone team has 0 and 1). The server
  names its GPU by unique ID, because CUDA's numbering can differ from nvidia-smi's.
- Port **8200** (not vLLM's usual 8000), host `127.0.0.1` only, API key required,
  because the machine is shared.
