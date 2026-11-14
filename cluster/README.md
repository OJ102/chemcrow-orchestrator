# Local model on the GPU cluster (vLLM + Qwen3.8-27B)

Scripts for serving our local language model on `cs-gpu-00.digitfut.uc.edu`, for
the ChemCrow tool-calling evaluation. Set up and first tested 2026-11-14.

| File | What it does |
|---|---|
| [setup_vllm.sh](setup_vllm.sh) | One-time setup: Python venv, vLLM, and the model download (all in `~/llm`) |
| [serve_vllm.sh](serve_vllm.sh) | Starts the model server on **GPU 2** (every setting is explained in comments) |
| [smoke_test.py](smoke_test.py) | Quick "is it alive and can it call tools?" check |

The copies that actually run live on the cluster in `~/llm/` (= `/home2/sathiysk/llm`).
After editing a script here, copy it over again (see "Updating a script" below).

## What's on the cluster

```
~/llm/
├── venv/          Python 3.12 environment: vLLM 0.33.0, transformers 5.23, torch 2.14 (15 GB)
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

## Settings to remember

- Qwen's recommended sampling (from the model card): thinking off:
  `temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5`; thinking on:
  `temperature 1.0, top_p 0.95, top_k 20`. Turn thinking off per request with
  `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`.
- GPUs: ours are **2 and 4** (the other capstone team has 0 and 1). The server
  names its GPU by unique ID, because CUDA's numbering can differ from nvidia-smi's.
- Port **8200** (not vLLM's usual 8000), host `127.0.0.1` only, API key required,
  because the machine is shared.
