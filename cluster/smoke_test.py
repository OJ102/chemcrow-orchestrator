"""smoke_test.py: first check that the local model server works and can call tools.

Runs ON the cluster, with the vLLM venv (it already has the `openai` package):
    ssh uc-gpu '~/llm/venv/bin/python ~/llm/smoke_test.py'

It is NOT the real evaluation, just a quick "is it alive and sane?" check:
  1. plain question, thinking OFF        -> does it answer, how fast?
  2. tool choice, thinking OFF, "auto"   -> does it pick the right tool by itself?
  3. tool choice, thinking OFF, "required" (vLLM FORCES a valid tool call: the output
     is constrained so it can only produce one of the listed tools with valid arguments.
     This is the main help for a small model, per Om)
  4. tool choice, thinking ON, "auto"    -> same, but the model reasons first (slower)
"""

import json  # to print tool arguments neatly
import time  # to time each request
from pathlib import Path  # to read the API key file

from openai import OpenAI  # the standard client; vLLM speaks the same API

# The server from serve_vllm.sh: same machine, port 8200, key from ~/llm/.api_key.
client = OpenAI(base_url="http://127.0.0.1:8200/v1",
                api_key=(Path.home() / "llm" / ".api_key").read_text().strip())
MODEL = "qwen3.8-27b"  # the --served-model-name

# Two small, fake tools in the OpenAI "function" format. The model only sees
# these descriptions; nothing actually runs. We just check which one it picks.
TOOLS = [
    {"type": "function", "function": {
        "name": "molecular_weight",
        "description": "Return the molecular weight (g/mol) of a molecule given as SMILES.",
        "parameters": {"type": "object",
                       "properties": {"smiles": {"type": "string", "description": "molecule as SMILES"}},
                       "required": ["smiles"]}}},
    {"type": "function", "function": {
        "name": "patent_check",
        "description": "Check whether a molecule (SMILES) is already patented.",
        "parameters": {"type": "object",
                       "properties": {"smiles": {"type": "string", "description": "molecule as SMILES"}},
                       "required": ["smiles"]}}},
]
QUESTION = "What is the molecular weight of benzene, SMILES c1ccccc1?"  # right answer: molecular_weight


def ask(label, messages, thinking, **extra):
    """Send one request, print the timing, the answer or tool call, and the reasoning length."""
    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=MODEL, messages=messages, max_tokens=2048,
        # Qwen's recommended settings for each mode (from the model card).
        temperature=0.7 if not thinking else 1.0, top_p=0.8 if not thinking else 0.95,
        extra_body={"top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": thinking}},
        **extra)
    secs = time.perf_counter() - start
    msg = resp.choices[0].message
    reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
    out_tokens = resp.usage.completion_tokens
    print(f"\n=== {label}  ({secs:.1f}s, {out_tokens} tokens, {out_tokens / secs:.0f} tokens/s)")
    if reasoning:
        print(f"thinking: {len(reasoning)} characters")
    if msg.tool_calls:
        for call in msg.tool_calls:
            print(f"TOOL CALL: {call.function.name}({json.dumps(json.loads(call.function.arguments))})")
    else:
        print("answer:", (msg.content or "").strip()[:300])


print("models served:", [m.id for m in client.models.list().data])
ask("1. plain question, thinking off",
    [{"role": "user", "content": "In one sentence: what is an organic solar cell?"}], thinking=False)
user = [{"role": "user", "content": QUESTION}]
ask("2. tool choice, thinking off, auto", user, thinking=False, tools=TOOLS, tool_choice="auto")
ask("3. tool choice, thinking off, REQUIRED (forced format)", user, thinking=False, tools=TOOLS, tool_choice="required")
ask("4. tool choice, thinking ON, auto", user, thinking=True, tools=TOOLS, tool_choice="auto")
