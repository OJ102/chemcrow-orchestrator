"""agent.py: multi-step (sequential) tool calling with the local model.

THE LOOP
    1. Send the question + a list of tools to the model.
    2. If the model asks for tool calls, OUR CODE runs them and sends each result
       back as a "tool" message. The model sees the results and decides the next call,
       so the output of one tool feeds into the next.
    3. Repeat until the model answers in plain text.

TWO TOOL COLLECTIONS (--tools)
    opv       our 4 tools over Kyle's OPV molecules (opv_tools.py). Runs in ~/llm/agent-venv.
    chemcrow  12 of ChemCrow's own tools + the 4 OPV tools = 16 (chemcrow_tools.py).
              Needs ChemCrow's old LangChain, so it runs in ~/llm/chemcrow-venv.

TOOLSETS (--toolset, chemcrow only). Small models choose worse among many tools, so:
    all    the model sees all 16 tools at once (the baseline)
    auto   ROUTER first: one short model call, forced to answer only with toolset
           names (e.g. ["safety"]), then the model sees just those few tools
    NAME   a fixed toolset chosen by our code, e.g. --toolset safety  (or lookup,analysis)
    --compare runs every demo question with "all" AND "auto" and prints a table.

SAFETY CAPS (the MCP expert's advice: our code stays in charge)
    MAX_STEPS = 8 model turns per question; MAX_ERRORS = 3 tool errors per question.
    Errors go back to the model with a hint so it can fix its call.

RUN (on the cluster; the vLLM server must be running):
    ~/llm/agent-venv/bin/python ~/llm/agent.py                                   # OPV demo
    ~/llm/chemcrow-venv/bin/python ~/llm/agent.py --tools chemcrow --toolset auto
    ~/llm/chemcrow-venv/bin/python ~/llm/agent.py --tools chemcrow --compare
    ... -q "your question"   ask your own question;   --think   reason before each step
"""

import argparse  # command-line options
import json  # the server speaks JSON
import time  # timing
import urllib.request  # Python's built-in web client (works in every venv)
from datetime import datetime
from pathlib import Path

MAX_STEPS = 8
MAX_ERRORS = 3
MODEL = "qwen3.8-27b"  # --served-model-name in serve_vllm.sh
URL = "http://127.0.0.1:8200/v1/chat/completions"
API_KEY = (Path.home() / "llm" / ".api_key").read_text().strip()
LOG_DIR = Path.home() / "llm" / "logs" / "agent"

# Short, explicit rules. Small models follow short numbered rules better than long prose.
SYSTEM_PROMPT = """You help researchers working on organic solar cells (OPV) and chemistry.
Rules:
1. Use a tool for every fact about a molecule. Never guess numbers.
2. Refer to molecules by NAME (as the user or a tool wrote it). Only use SMILES if no name exists.
3. You may call tools several times. Use the results of earlier calls to choose the next call.
4. If a tool returns an error, read it and call again with corrected arguments, or try another tool.
5. When you have everything you need, answer in 1-3 sentences and include the results you used."""

# Demo questions. "expect" = tools a correct run should use (checked automatically).
OPV_DEMO = [
    {"q": "Find the 3 acceptors most similar to Y6, then tell me which of those 3 has the highest molecular weight.",
     "expect": ["most_similar", "properties"]},
    {"q": "Is PTB7-Th structurally more similar to D18 or to PBDB-T?", "expect": ["compare"]},
    {"q": "Which has more fluorine atoms, ITIC or IT-4F, and what are their molecular weights?",
     "expect": ["properties"]},
    # Misspelled name: the tool's error suggests the right one; the model must recover.
    # (Without "use the closest match" the model rightly refuses to guess and asks.)
    {"q": "How many aromatic rings does IT-4FF have? I may have misspelled the name; if so, use the closest match.",
     "expect": ["properties"]},
]
CHEMCROW_DEMO = [
    {"q": "What is the molecular weight of caffeine, and which functional groups does it have?",
     "expect": ["SMILES2Weight", "FunctionalGroups"]},
    {"q": "Is nitroglycerin explosive? Also check whether it is a controlled chemical.",
     "expect": ["ExplosiveCheck", "ControlChemCheck"]},
    {"q": "Find the CAS number of aspirin, then check whether aspirin is patented.",
     "expect": ["Mol2CAS", "PatentCheck"]},
    {"q": "How similar are the OPV acceptors Y6 and ITIC, and is Y6 patented?",
     "expect": ["PatentCheck"]},  # similarity may come from MolSimilarity or compare: both fine
    {"q": "Give me the SMILES of paracetamol, then list its functional groups.",
     "expect": ["Name2SMILES", "FunctionalGroups"]},
]


def chat(messages, tools=None, tool_choice="auto", think=False):
    """One request to the vLLM server (OpenAI chat-completions format). Returns the message dict."""
    body = {"model": MODEL, "messages": messages, "max_tokens": 2048, "seed": 0,
            # Qwen's recommended settings for each mode (model card)
            "temperature": 1.0 if think else 0.7, "top_p": 0.95 if think else 0.8,
            "presence_penalty": 0.0 if think else 1.5, "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": think}}
    if tools:
        body["tools"], body["tool_choice"] = tools, tool_choice
    request = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)["choices"][0]["message"]


def short(obj, limit=220):
    """Compact one-line JSON for printing (long results are cut with '...')."""
    text = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + "..."


def route(question, toolsets):
    """ROUTER: ask the model which toolsets the question needs (forced, constrained answer)."""
    names = list(toolsets)
    router_tool = {"type": "function", "function": {
        "name": "choose_toolsets",
        "description": "Choose the groups of tools needed to answer the question.",
        "parameters": {"type": "object", "properties": {"toolsets": {
            "type": "array", "minItems": 1, "maxItems": 3,
            "items": {"type": "string", "enum": names}}}, "required": ["toolsets"]}}}
    menu = "\n".join(f"- {n}: {toolsets[n]['description']}" for n in names)
    msg = chat([{"role": "user", "content": f"Question: {question}\n\nTool groups:\n{menu}\n\n"
                 "Which groups are needed? Pick every group the question needs."}],
               tools=[router_tool],
               # Forcing this one function: vLLM only lets the model write valid group names.
               tool_choice={"type": "function", "function": {"name": "choose_toolsets"}})
    chosen = json.loads(msg["tool_calls"][0]["function"]["arguments"])["toolsets"]
    return [c for c in dict.fromkeys(chosen) if c in toolsets]  # unique, valid, in order


def run_tool(registry, error_cls, name, raw_args):
    """Run one tool call. Returns (result dict, is_error). Never raises."""
    if name not in registry:
        return {"error": f"Unknown tool '{name}'. Available: {list(registry)}"}, True
    try:
        args = json.loads(raw_args or "{}")  # the model sends arguments as JSON text
        return registry[name](**args), False
    except error_cls as err:  # a problem the model can fix (e.g. unknown name)
        return {"error": str(err)}, True
    except (json.JSONDecodeError, TypeError, ValueError) as err:  # malformed arguments
        return {"error": f"Bad arguments for {name}: {err}"}, True
    except Exception as err:  # a bug or outage inside the tool itself
        return {"error": f"{name} failed: {type(err).__name__}: {err}"}, True


def ask(question, kit, toolset="all", think=False, expect=()):
    """Answer one question with the tool loop. Prints each step; returns a log dict."""
    print(f"\nQUESTION: {question}")
    started = time.perf_counter()
    model_calls = tool_calls = errors = 0

    # Which tools does the model see?
    if toolset == "auto":
        chosen = route(question, kit["toolsets"])
        model_calls += 1
        names = [t for ts in chosen for t in kit["toolsets"][ts]["tools"]]
        print(f"  router -> {chosen}  ({len(set(names))} tools)  [{time.perf_counter() - started:.1f}s]")
    elif toolset == "all":
        names = list(kit["schemas"])
    else:
        names = [t for ts in toolset.split(",") for t in kit["toolsets"][ts]["tools"]]
    tools = [kit["schemas"][n] for n in dict.fromkeys(names)]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    log = {"question": question, "toolset": toolset, "think": think, "tools_shown": len(tools), "steps": []}
    for step in range(1, MAX_STEPS + 1):
        last_chance = step == MAX_STEPS or errors >= MAX_ERRORS  # must answer now
        t0 = time.perf_counter()
        # "none" on the last chance = no more tools, the model must answer
        msg = chat(messages, tools=tools, tool_choice="none" if last_chance else "auto", think=think)
        model_calls += 1
        secs = time.perf_counter() - t0
        calls = msg.get("tool_calls") or []

        if not calls:  # plain text = the final answer
            log["answer"] = (msg.get("content") or "").strip()
            print(f"  ANSWER ({secs:.1f}s): {log['answer']}")
            break

        # Keep the model's tool-call turn in the conversation, then add each result.
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
        for call in calls:
            fn = call["function"]
            result, is_error = run_tool(kit["tools"], kit["error"], fn["name"], fn["arguments"])
            tool_calls += 1
            errors += is_error
            print(f"  step {step}  -> {fn['name']}({fn['arguments']})   [{secs:.1f}s]")
            print(f"           <- {'ERROR ' if is_error else ''}{short(result)}")
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)})
            log["steps"].append({"step": step, "tool": fn["name"], "arguments": fn["arguments"],
                                 "result": result, "error": is_error})
    else:
        log["answer"] = None

    used = {s["tool"] for s in log["steps"] if not s["error"]}
    missing = [t for t in expect if t not in used]
    total = time.perf_counter() - started
    log["summary"] = {"model_calls": model_calls, "tool_calls": tool_calls, "errors": errors,
                      "seconds": round(total, 1), "expected_tools_used": not missing, "missing": missing}
    check = "" if not expect else ("  expected tools: OK" if not missing else f"  expected tools: MISSING {missing}")
    print(f"  [{model_calls} model calls, {tool_calls} tool calls, {errors} errors, {total:.1f}s]{check}")
    return log


def load_kit(which):
    """Import the chosen tool collection (ChemCrow's needs ChemCrow's venv)."""
    if which == "opv":
        import opv_tools as m
        return {"tools": m.TOOLS, "schemas": {s["function"]["name"]: s for s in m.TOOL_SCHEMAS},
                "toolsets": {"opv": {"description": "OPV database", "tools": list(m.TOOLS)}},
                "error": m.ToolError, "demo": OPV_DEMO}
    import chemcrow_tools as m
    return {"tools": m.TOOLS, "schemas": m.SCHEMAS, "toolsets": m.TOOLSETS,
            "error": (m.ToolError, __import__("opv_tools").ToolError), "demo": CHEMCROW_DEMO}


def main():
    parser = argparse.ArgumentParser(description="Multi-step tool calling with the local model.")
    parser.add_argument("--tools", choices=["opv", "chemcrow"], default="opv", help="which tool collection")
    parser.add_argument("--toolset", default="all",
                        help="'all', 'auto' (router), or toolset name(s) like 'safety' or 'lookup,analysis'")
    parser.add_argument("--compare", action="store_true", help="run demo questions with 'all' and 'auto'")
    parser.add_argument("-q", "--question", help="ask this instead of the demo questions")
    parser.add_argument("--think", action="store_true", help="let the model reason before each step")
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="load the tools ONCE, then type questions one after another (best for demos)")
    args = parser.parse_args()

    print(f"loading tools ({args.tools}) ... (ChemCrow's libraries can take a few minutes to load)")
    kit = load_kit(args.tools)
    print(f"{len(kit['schemas'])} tools ready: {', '.join(kit['schemas'])}")

    if args.interactive:  # demo mode: no reload between questions
        print(f"\nType a question and press Enter (toolset = {args.toolset}). Empty line or Ctrl+C to quit.")
        while True:
            try:
                question = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not question:
                break
            try:
                ask(question, kit, toolset=args.toolset, think=args.think)
            except Exception as err:  # e.g. the server stopped: report it and keep going
                print(f"  (failed: {type(err).__name__}: {err})")
        return
    demo = [{"q": args.question, "expect": []}] if args.question else kit["demo"]
    modes = ["all", "auto"] if args.compare else [args.toolset]

    logs = []
    for mode in modes:
        print(f"\n################ toolset = {mode}")
        logs += [ask(d["q"], kit, toolset=mode, think=args.think, expect=d["expect"]) for d in demo]

    if args.compare:  # side-by-side table
        print("\nCOMPARISON  (tools shown | expected tools used | errors | model calls | seconds)")
        half = len(logs) // 2
        for a, b in zip(logs[:half], logs[half:]):
            sa, sb = a["summary"], b["summary"]
            fmt = lambda log, s: (f"{log['tools_shown']:2d} tools  {'OK  ' if s['expected_tools_used'] else 'MISS'}  "
                                  f"{s['errors']} err  {s['model_calls']} calls  {s['seconds']:5.1f}s")
            print(f"  {a['question'][:58]:58} | all: {fmt(a, sa)} | auto: {fmt(b, sb)}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = LOG_DIR / f"run-{datetime.now():%Y%m%d-%H%M%S}-{args.tools}-{'compare' if args.compare else args.toolset}.json"
    out.write_text(json.dumps(logs, indent=2))
    print(f"\nfull transcript saved to {out}")


if __name__ == "__main__":
    main()
