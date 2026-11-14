"""agent.py: multi-step (sequential) tool calling with the local model.

THE LOOP
    1. Send the question + the tool list (opv_tools.py) to the model.
    2. If the model asks for tool calls, OUR CODE runs them and sends each result
       back as a "tool" message. The model sees the results and decides the next call,
       so the output of one tool feeds into the next.
    3. Repeat until the model answers in plain text.

SAFETY CAPS (the MCP expert's advice: our code stays in charge)
    MAX_STEPS = 8    model turns per question; then the model must answer with what it has
    MAX_ERRORS = 3   tool errors per question (bad names/arguments); each error is sent
                     back with a hint so the model can fix its call, but not forever
    Every model turn and tool call is printed and saved to ~/llm/logs/agent/.

RUN (on the cluster, with the agent's own venv; the server must be running):
    ~/llm/agent-venv/bin/python ~/llm/agent.py                  # the built-in demo questions
    ~/llm/agent-venv/bin/python ~/llm/agent.py -q "your question"
    ~/llm/agent-venv/bin/python ~/llm/agent.py --think          # model reasons before each step
"""

import argparse  # command-line options
import json  # tool arguments/results are JSON
import time  # timing
from datetime import datetime
from pathlib import Path

from openai import OpenAI  # vLLM speaks OpenAI's API

from opv_tools import TOOL_SCHEMAS, TOOLS, ToolError

MAX_STEPS = 8
MAX_ERRORS = 3
MODEL = "qwen3.8-27b"  # --served-model-name in serve_vllm.sh
LOG_DIR = Path.home() / "llm" / "logs" / "agent"

client = OpenAI(base_url="http://127.0.0.1:8200/v1",
                api_key=(Path.home() / "llm" / ".api_key").read_text().strip())

# Short, explicit rules. Small models follow short numbered rules better than long prose.
SYSTEM_PROMPT = """You help researchers working on organic solar cells (OPV).
You have tools over the team's database of 4,023 OPV donor and acceptor molecules.
Rules:
1. Use a tool for every fact about a molecule. Never guess numbers.
2. Refer to molecules by NAME exactly as the user or a tool wrote it. Never write SMILES.
3. You may call tools several times. Use the results of earlier calls to choose the next call.
4. If a tool returns an error, read it and call again with corrected arguments.
5. When you have everything you need, answer in 1-3 sentences and include the numbers you used."""

# Demo questions. Each needs SEVERAL steps, where later steps use earlier results.
DEMO_QUESTIONS = [
    # step 1 finds names, step 2+ uses those names
    "Find the 3 acceptors most similar to Y6, then tell me which of those 3 has the highest molecular weight.",
    # two independent lookups, then a comparison
    "Is PTB7-Th structurally more similar to D18 or to PBDB-T?",
    # properties of two molecules, then reasoning on the numbers
    "Which has more fluorine atoms, ITIC or IT-4F, and what are their molecular weights?",
    # misspelled name: the tool's error suggests the right one, the model must recover.
    # ('IT-4FF' isn't fixed by the forgiving name matching; its closest name is IT-4F.)
    # (Without "use the closest match" the model rightly refuses to guess and asks.)
    "How many aromatic rings does IT-4FF have? I may have misspelled the name; if so, use the closest match.",
]


def short(obj, limit=220):
    """Compact one-line JSON for printing (long results are cut with '...')."""
    text = json.dumps(obj, separators=(",", ":"))
    return text if len(text) <= limit else text[:limit] + "..."


def run_tool(name, raw_args):
    """Run one tool call. Returns (result dict, is_error). Never raises."""
    if name not in TOOLS:
        return {"error": f"Unknown tool '{name}'. Available: {list(TOOLS)}"}, True
    try:
        args = json.loads(raw_args or "{}")  # the model sends arguments as JSON text
        return TOOLS[name](**args), False
    except ToolError as err:  # a problem the model can fix (e.g. unknown name)
        return {"error": str(err)}, True
    except (json.JSONDecodeError, TypeError, ValueError) as err:  # malformed arguments
        return {"error": f"Bad arguments for {name}: {err}"}, True


def ask(question, think=False):
    """Answer one question with the tool loop. Prints each step; returns a summary dict."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}]
    log = {"question": question, "think": think, "steps": []}  # saved to a JSON file
    errors = model_calls = tool_calls = 0
    started = time.perf_counter()
    print(f"\nQUESTION: {question}")

    for step in range(1, MAX_STEPS + 1):
        last_chance = step == MAX_STEPS or errors >= MAX_ERRORS  # must answer now
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOL_SCHEMAS,
            # "none" on the last chance = no more tools, the model must answer
            tool_choice="none" if last_chance else "auto",
            max_tokens=2048, seed=0,  # seed: same question -> same run (repeatable demos)
            # Qwen's recommended settings for each mode (model card)
            temperature=1.0 if think else 0.7, top_p=0.95 if think else 0.8,
            presence_penalty=0.0 if think else 1.5,
            extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": think}})
        model_calls += 1
        msg = resp.choices[0].message
        secs = time.perf_counter() - t0

        if not msg.tool_calls:  # plain text = the final answer
            answer = (msg.content or "").strip()
            print(f"  ANSWER ({secs:.1f}s): {answer}")
            log["answer"] = answer
            break

        # Keep the model's tool-call turn in the conversation, then add each result.
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [c.model_dump() for c in msg.tool_calls]})
        for call in msg.tool_calls:
            result, is_error = run_tool(call.function.name, call.function.arguments)
            tool_calls += 1
            errors += is_error
            print(f"  step {step}  -> {call.function.name}({call.function.arguments})   [{secs:.1f}s]")
            print(f"           <- {'ERROR ' if is_error else ''}{short(result)}")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
            log["steps"].append({"step": step, "tool": call.function.name,
                                 "arguments": call.function.arguments, "result": result,
                                 "error": is_error, "model_seconds": round(secs, 2)})
    else:
        log["answer"] = None  # loop ran out without a plain answer (shouldn't happen)
        print("  (no answer within the step limit)")

    total = time.perf_counter() - started
    summary = {"model_calls": model_calls, "tool_calls": tool_calls, "errors": errors,
               "seconds": round(total, 1)}
    log["summary"] = summary
    print(f"  [{model_calls} model calls, {tool_calls} tool calls, {errors} errors, {total:.1f}s]")
    return log


def main():
    parser = argparse.ArgumentParser(description="Multi-step tool calling with the local model.")
    parser.add_argument("-q", "--question", help="ask this instead of the demo questions")
    parser.add_argument("--think", action="store_true", help="let the model reason before each step")
    args = parser.parse_args()

    questions = [args.question] if args.question else DEMO_QUESTIONS
    logs = [ask(q, think=args.think) for q in questions]

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = LOG_DIR / f"run-{datetime.now():%Y%m%d-%H%M%S}{'-think' if args.think else ''}.json"
    out.write_text(json.dumps(logs, indent=2))
    print(f"\nfull transcript saved to {out}")


if __name__ == "__main__":
    main()
