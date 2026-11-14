#!/usr/bin/env bash
# serve_vllm.sh: start the local Qwen3.8-27B model server (vLLM) on cs-gpu-00.
#
# WHAT IT IS: vLLM loads the model onto ONE of our GPUs and answers requests in
# the same format as OpenAI's API (POST /v1/chat/completions), so ChemCrow,
# LangChain or the openai Python package can talk to it unchanged.
#
# HOW TO RUN (from the laptop; runs inside tmux so it survives disconnects):
#   ssh uc-gpu 'tmux new-session -d -s vllm "bash ~/llm/serve_vllm.sh 2>&1 | tee ~/llm/logs/vllm.log"'
# Stop it:      ssh uc-gpu 'tmux kill-session -t vllm'
# Watch it:     ssh uc-gpu -t 'tmux attach -t vllm'   (detach again: Ctrl+B then D)
# Ready when the log says "Application startup complete" (loading takes a few minutes).

set -euo pipefail

LLM=~/llm
export HF_HOME=$LLM/hf    # where setup_vllm.sh downloaded the model
export TMPDIR=$LLM/tmp    # keep temp files off the nearly full root disk
source "$LLM/venv/bin/activate"

# --- Which GPU -----------------------------------------------------------------
# Our GPUs are nvidia-smi numbers 2 and 4 (the other team uses 0 and 1). CUDA can
# number GPUs differently from nvidia-smi (the small "DGX Display" card is #3),
# so we name the GPU by its unique ID instead of its number. This is GPU 2.
# GPU 4 (GPU-190d8606-67c9-a6c4-ca56-9d53abf6f785) stays free for a second copy
# of the model or for teammates (e.g. Chemprop training).
export CUDA_VISIBLE_DEVICES=GPU-51f63d6b-e221-2a59-66a8-1a5a1550e95a

# --- Password for the server ---------------------------------------------------
# The machine is shared with other users, who could otherwise call our server.
# The key is a random string made once and kept in a file only we can read.
KEY_FILE=$LLM/.api_key
if [ ! -s "$KEY_FILE" ]; then
  (umask 077; python -c "import secrets; print(secrets.token_urlsafe(32))" > "$KEY_FILE")
fi
export VLLM_API_KEY=$(cat "$KEY_FILE")  # vLLM requires "Authorization: Bearer <key>"

PORT=8200  # not vLLM's usual 8000, to avoid clashing with the other team

# --- Start ---------------------------------------------------------------------
# Flags, one by one:
#   --served-model-name     the short name clients use in "model": "qwen3.8-27b"
#   --host 127.0.0.1        only programs ON this machine can connect (laptop uses an SSH tunnel)
#   --max-model-len 32768   longest conversation (prompt + answer) in tokens. Native is 262K, but
#                           we don't need it, and a smaller limit leaves memory for more requests at once
#   --gpu-memory-utilization 0.90  use up to 90% of the GPU's 80 GB (model ~56 GB, rest = cache)
#   --language-model-only   skip the model's image/video part: we only send text, saves memory
#   --enable-auto-tool-choice + --tool-call-parser qwen3_coder
#                           turn the model's tool-call text into proper OpenAI "tool_calls"
#                           (the parser named by the official vLLM recipe for this model)
#   --reasoning-parser qwen3  put the model's "thinking" in a separate field, not in the answer
exec vllm serve Qwen/Qwen3.8-27B \
  --served-model-name qwen3.8-27b \
  --host 127.0.0.1 --port "$PORT" \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --language-model-only \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3
