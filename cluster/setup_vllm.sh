#!/usr/bin/env bash
# setup_vllm.sh: one-time setup of vLLM + the Qwen3.8-27B model on cs-gpu-00.
#
# WHAT IT DOES (all inside ~/llm, i.e. /home2/sathiysk/llm, the 128 TB home disk;
# never the nearly full root disk):
#   1. makes a Python virtual environment ~/llm/venv (Python 3.12)
#   2. installs uv (a fast pip replacement), then vLLM >= 0.17 and transformers >= 5.8
#      (versions required by the official vLLM recipe for Qwen3.8-27B)
#   3. downloads the model Qwen/Qwen3.8-27B (BF16, ~56 GB) into ~/llm/hf
#
# HOW TO RUN (from the laptop; keeps running after you disconnect):
#   ssh uc-gpu 'mkdir -p ~/llm/logs && nohup bash ~/llm/setup_vllm.sh > ~/llm/logs/setup.log 2>&1 &'
# Safe to re-run: finished steps are skipped or quickly re-checked.

set -euo pipefail  # stop at the first error, and treat unset variables as errors

LLM=~/llm                          # everything for the local model lives here
export HF_HOME=$LLM/hf             # Hugging Face downloads (the model) go here
export TMPDIR=$LLM/tmp             # temp files: NOT /tmp, which is on the full root disk
export UV_CACHE_DIR=$LLM/uv-cache  # uv's download cache
MODEL=Qwen/Qwen3.8-27B             # the model, as named on Hugging Face

mkdir -p "$LLM" "$HF_HOME" "$TMPDIR" "$UV_CACHE_DIR" "$LLM/logs"
echo "[$(date)] setup started on $(hostname)"

# 1. Virtual environment (a private folder of Python packages).
if [ ! -x "$LLM/venv/bin/python" ]; then
  python3 -m venv "$LLM/venv"
fi
source "$LLM/venv/bin/activate"

# 2. Install uv, then vLLM and transformers with it. "--torch-backend auto" picks the
#    PyTorch build that matches this machine's CUDA driver.
pip install -q -U pip uv
uv pip install -U "vllm>=0.17.0" --torch-backend auto
# transformers: >= 5.8 for this model (recipe), but within the range vLLM 0.33 accepts
# (an unpinned upgrade gave 5.23, which vLLM 0.33 doesn't support; fixed 2026-11-14).
uv pip install "transformers>=5.8.0,<5.21"
uv pip check  # prints a warning if any package versions disagree
# NOTE: don't install other packages (rdkit, pandas, ...) into this venv: they can
# upgrade numpy past what vLLM accepts. The agent has its own venv, ~/llm/agent-venv.
python -c "import vllm, transformers, torch; print('vllm', vllm.__version__, '| transformers', transformers.__version__, '| torch', torch.__version__, '| cuda ok:', torch.cuda.is_available())"

# 3. Download the model (resumes if interrupted). hf = Hugging Face's command-line tool.
echo "[$(date)] downloading $MODEL"
hf download "$MODEL"
du -sh "$HF_HOME"
echo "[$(date)] SETUP-DONE"
