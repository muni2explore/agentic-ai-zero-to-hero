import requests
import json
import math
import re
import os
import time
import hashlib
from datetime import datetime
from typing import Callable
import logging
from llm import call_llm


LOG_DIR = "logs"
DEFAULT_LOG_FILE = os.path.join(LOG_DIR, "agent_06.log")

def setup_logging(log_file: str = DEFAULT_LOG_FILE) -> logging.Logger:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("agent_06_feedback_loop")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


logger = setup_logging()

# ─────────────────────────────────────────────
# EXECUTION TRACE
# Records every step the agent took — tool called,
# args used, result, duration, success/fail
# This becomes your debugging + observability layer
# ─────────────────────────────────────────────

class ExecutionTrace:
    def __init__(self, goal: str):
        self.goal       = goal
        self.steps: list[dict] = []
        self.start_time = time.time()

    def record(self, step: int, thought: str, tool: str,
               args: dict, result: str, success: bool, duration: float):
        self.steps.append({
            "step":     step,
            "thought":  thought,
            "tool":     tool,
            "args":     args,
            "result":   result,
            "success":  success,
            "duration": round(duration, 3),
        })

    def print_summary(self):
        total = round(time.time() - self.start_time, 2)
        success_count = sum(1 for s in self.steps if s["success"])
        fail_count    = len(self.steps) - success_count

        print(f"\n{'─'*60}")
        print(f"📋 Execution Trace — '{self.goal[:50]}'")
        print(f"{'─'*60}")
        for s in self.steps:
            icon = "✅" if s["success"] else "❌"
            print(f"  {icon} Step {s['step']}: {s['tool']}({json.dumps(s['args'])[:50]})")
            print(f"       Result: {str(s['result'])[:80]}")
            print(f"       Time:   {s['duration']}s")
        print(f"{'─'*60}")
        print(f"  Total steps: {len(self.steps)} | "
              f"Success: {success_count} | "
              f"Failed: {fail_count} | "
              f"Wall time: {total}s")


# ─────────────────────────────────────────────
# TOOL REGISTRY (same pattern, leaner version)
# ─────────────────────────────────────────────

class ToolRegistry:
    def __init__(self):
        self.tools: dict[str, dict] = {}

    def register(self, schema: dict, fn: Callable, category: str = "general"):
        self.tools[schema["name"]] = {
            "schema": schema, "fn": fn, "category": category
        }

    def execute(self, name: str, args: dict) -> tuple[str, bool]:
        if name not in self.tools:
            return f"Unknown tool '{name}'. Available: {list(self.tools.keys())}", False
        try:
            result = self.tools[name]["fn"](**args)
            return str(result), True
        except Exception as e:
            return f"Tool error: {e}", False

    def prompt_text(self) -> str:
        lines = []
        cats: dict[str, list] = {}
        for name, t in self.tools.items():
            cats.setdefault(t["category"], []).append((name, t))
        for cat, tools in cats.items():
            lines.append(f"[{cat.upper()}]")
            for name, t in tools:
                s = t["schema"]
                props = s["parameters"]["properties"]
                req   = s["parameters"].get("required", [])
                lines.append(f"  {name}: {s['description']}")
                for p, info in props.items():
                    r = "*" if p in req else "?"
                    enum_hint = f" choices={info['enum']}" if "enum" in info else ""
                    lines.append(f"    {p}{r}: {info['description']}{enum_hint}")
            lines.append("")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────

# Math
def calculator_fn(expression: str) -> str:
    allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
    allowed.update({"abs": abs, "round": round})
    return str(eval(expression.strip(), {"__builtins__": {}}, allowed))

def percentage_fn(operation: str, value: float,
                  total: float = None, percent: float = None) -> str:
    if operation == "what_percent" and total is not None:
        return f"{round((value / total) * 100, 2)}%"
    elif operation == "percent_of" and percent is not None:
        return f"{round(value * percent / 100, 4)}"
    elif operation == "percent_change" and total is not None:
        change = ((total - value) / value) * 100
        sign   = "increase" if change > 0 else "decrease"
        return f"{round(abs(change), 2)}% {sign}"
    return "Invalid args."

def unit_convert_fn(value: float, from_unit: str, to_unit: str) -> str:
    table = {
        ("km","miles"): 0.621371, ("miles","km"): 1.60934,
        ("kg","lbs"):   2.20462,  ("lbs","kg"):   0.453592,
        ("gb","mb"):    1024,     ("mb","gb"):    1/1024,
        ("gb","tb"):    1/1024,   ("tb","gb"):    1024,
    }
    fu, tu = from_unit.lower(), to_unit.lower()
    if (fu, tu) == ("c", "f"):
        return f"{round((value*9/5)+32, 2)} f"
    if (fu, tu) == ("f", "c"):
        return f"{round((value-32)*5/9, 2)} c"
    if (fu, tu) in table:
        return f"{round(value * table[(fu,tu)], 4)} {to_unit}"
    return f"Unsupported: {from_unit} → {to_unit}"

# File
def file_write_fn(filename: str, content: str, mode: str = "write") -> str:
    safe = os.path.basename(filename)
    with open(safe, "a" if mode == "append" else "w") as f:
        f.write(content)
    return f"Wrote {len(content)} chars to '{safe}'"

def file_read_fn(filename: str) -> str:
    safe = os.path.basename(filename)
    if not os.path.exists(safe):
        return f"Error: '{safe}' not found"
    with open(safe) as f:
        return f.read()

def file_list_fn(extension: str = "") -> str:
    files = [f for f in os.listdir(".") if f.endswith(extension)]
    return "\n".join(sorted(files)) if files else "No files found"

# Text
def text_analyze_fn(text: str, operation: str) -> str:
    words     = text.strip().split()
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    results = {
        "word_count":     str(len(words)),
        "char_count":     str(len(text)),
        "sentence_count": str(len(sentences)),
        "longest_word":   max(words, key=len),
        "avg_word_len":   str(round(sum(len(w) for w in words) / len(words), 2)),
        "all": "\n".join([
            f"words={len(words)}",
            f"chars={len(text)}",
            f"sentences={len(sentences)}",
            f"longest={max(words, key=len)}",
            f"avg_word_len={round(sum(len(w) for w in words)/len(words), 2)}",
        ])
    }
    return results.get(operation, f"Unknown: {operation}")

def text_transform_fn(text: str, operation: str) -> str:
    ops = {
        "uppercase":  text.upper(),
        "lowercase":  text.lower(),
        "title_case": text.title(),
        "slug":       re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-'),
        "hash_md5":   hashlib.md5(text.encode()).hexdigest(),
        "reverse":    text[::-1],
    }
    return ops.get(operation, f"Unknown: {operation}")

def datetime_fn(info_type: str) -> str:
    now = datetime.now()
    return {
        "full":        now.strftime("%Y-%m-%d %H:%M:%S, %A"),
        "date":        now.strftime("%Y-%m-%d"),
        "time":        now.strftime("%H:%M:%S"),
        "day":         now.strftime("%A"),
        "week_number": str(now.isocalendar()[1]),
        "day_of_year": str(now.timetuple().tm_yday),
        "timestamp":   str(int(now.timestamp())),
    }.get(info_type, now.strftime("%Y-%m-%d %H:%M:%S"))

# Report builder — combines earlier tool results into a document
def report_builder_fn(title: str, sections: str, filename: str) -> str:
    """sections is a JSON string: [{"heading": "...", "content": "..."}]"""
    try:
        section_list = json.loads(sections)
    except json.JSONDecodeError:
        return "Error: sections must be valid JSON array"

    now    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    border = "=" * 50
    lines  = [
        border,
        f"  {title}",
        f"  Generated: {now}",
        border, ""
    ]
    for sec in section_list:
        lines.append(f"## {sec.get('heading', 'Section')}")
        lines.append(sec.get("content", ""))
        lines.append("")

    report = "\n".join(lines)
    safe   = os.path.basename(filename)
    with open(safe, "w") as f:
        f.write(report)
    return f"Report '{safe}' written ({len(report)} chars, {len(section_list)} sections)"


# ─────────────────────────────────────────────
# BUILD REGISTRY
# ─────────────────────────────────────────────

def build_registry() -> ToolRegistry:
    r = ToolRegistry()

    r.register({"name":"calculator","description":"Evaluate math expressions.",
        "parameters":{"type":"object","properties":{
            "expression":{"type":"string","description":"Python math expression"}},
        "required":["expression"]}}, calculator_fn, "math")

    r.register({"name":"percentage","description":"Calculate percentage relationships.",
        "parameters":{"type":"object","properties":{
            "operation":{"type":"string","description":"what_percent | percent_of | percent_change",
                         "enum":["what_percent","percent_of","percent_change"]},
            "value":    {"type":"number","description":"Base value"},
            "total":    {"type":"number","description":"Total (for what_percent / percent_change)"},
            "percent":  {"type":"number","description":"Percent to apply (for percent_of)"}},
        "required":["operation","value"]}}, percentage_fn, "math")

    r.register({"name":"unit_convert","description":"Convert between km/miles, kg/lbs, c/f, gb/mb/tb.",
        "parameters":{"type":"object","properties":{
            "value":     {"type":"number","description":"Value to convert"},
            "from_unit": {"type":"string","description":"Source unit"},
            "to_unit":   {"type":"string","description":"Target unit"}},
        "required":["value","from_unit","to_unit"]}}, unit_convert_fn, "math")

    r.register({"name":"file_write","description":"Write or append content to a file.",
        "parameters":{"type":"object","properties":{
            "filename":{"type":"string","description":"Target filename"},
            "content": {"type":"string","description":"Text to write"},
            "mode":    {"type":"string","description":"write or append","enum":["write","append"]}},
        "required":["filename","content"]}}, file_write_fn, "file")

    r.register({"name":"file_read","description":"Read a file's contents.",
        "parameters":{"type":"object","properties":{
            "filename":{"type":"string","description":"File to read"}},
        "required":["filename"]}}, file_read_fn, "file")

    r.register({"name":"file_list","description":"List files in current directory.",
        "parameters":{"type":"object","properties":{
            "extension":{"type":"string","description":"Filter by extension e.g. .txt or .py"}},
        "required":[]}}, file_list_fn, "file")

    r.register({"name":"text_analyze","description":"Analyze text statistics.",
        "parameters":{"type":"object","properties":{
            "text":     {"type":"string","description":"Text to analyze"},
            "operation":{"type":"string",
                         "description":"word_count|char_count|sentence_count|longest_word|avg_word_len|all",
                         "enum":["word_count","char_count","sentence_count",
                                 "longest_word","avg_word_len","all"]}},
        "required":["text","operation"]}}, text_analyze_fn, "text")

    r.register({"name":"text_transform","description":"Transform text: case, slug, hash, reverse.",
        "parameters":{"type":"object","properties":{
            "text":     {"type":"string","description":"Text to transform"},
            "operation":{"type":"string",
                         "description":"uppercase|lowercase|title_case|slug|hash_md5|reverse",
                         "enum":["uppercase","lowercase","title_case","slug","hash_md5","reverse"]}},
        "required":["text","operation"]}}, text_transform_fn, "text")

    r.register({"name":"datetime","description":"Get current date/time information.",
        "parameters":{"type":"object","properties":{
            "info_type":{"type":"string",
                         "description":"full|date|time|day|week_number|day_of_year|timestamp",
                         "enum":["full","date","time","day","week_number","day_of_year","timestamp"]}},
        "required":["info_type"]}}, datetime_fn, "text")

    r.register({"name":"report_builder",
        "description":"Compile multiple results into a formatted report file.",
        "parameters":{"type":"object","properties":{
            "title":    {"type":"string","description":"Report title"},
            "sections": {"type":"string",
                         "description":'JSON array of sections: [{"heading":"...","content":"..."}]'},
            "filename": {"type":"string","description":"Output filename e.g. report.txt"}},
        "required":["title","sections","filename"]}}, report_builder_fn, "file")

    return r


# ─────────────────────────────────────────────
# SYSTEM PROMPT
# Key addition: explicitly teach chaining + partial failure behaviour
# ─────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """You are a methodical agent that completes multi-step tasks using tools.

TOOLS:
{tools}

CHAINING RULES:
- Break complex goals into ordered steps.
- **IMPORTANT**: You must issue ONLY ONE tool call at a time.
- After issuing a tool call, stop and wait for the "Observation" feedback.
- Once you have the observation, use it to decide your next step or provide the Final Answer.
- Do not output multiple tool calls or a Final Answer in the same message.

RESPONSE FORMAT — ONLY valid JSON, one of these two:

Tool call:
{{"thought": "what I need to do and why", "tool": "name", "args": {{"key": "value"}}}}

Final answer:
{{"thought": "all steps complete", "final_answer": "complete summary of everything done"}}
"""

def build_system_prompt(registry: ToolRegistry) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(tools=registry.prompt_text())


# ─────────────────────────────────────────────
# AGENT LOOP WITH FULL FEEDBACK LOOP
# ─────────────────────────────────────────────

def parse_llm(raw: str) -> dict:
    cleaned = re.sub(r'^```(?:json)?\s*', '', raw.strip())
    cleaned = re.sub(r'\s*```$', '', cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except:
                pass
    return {"parse_error": True, "raw": raw}


def run_agent(goal: str, registry: ToolRegistry, max_steps: int = 15) -> str:
    print(f"\n{'='*60}")
    print(f"🎯 Goal: {goal}")
    print('='*60)

    trace    = ExecutionTrace(goal)
    messages = [
        {"role": "system", "content": build_system_prompt(registry)},
        {"role": "user",   "content": goal}
    ]
    consecutive_failures = 0

    for step in range(1, max_steps + 1):
        logger.info("Messages: %s", messages)
        # ① THINK
        raw = call_llm(messages)
        logger.info("RAW LLM Response: %s", raw)
        parsed = parse_llm(raw)

        logger.info("parsed: %s", parsed)

        # Handle parse failure
        if parsed.get("parse_error"):
            consecutive_failures += 1
            print(f"\n⚠️  Step {step}: parse error (attempt {consecutive_failures})")
            if consecutive_failures >= 3:
                print("❌ Too many parse failures — stopping")
                break
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content":
                'Respond with ONLY valid JSON: {"thought":"...","tool":"...","args":{...}}'})
            continue

        consecutive_failures = 0
        thought   = parsed.get("thought", "")
        print(f"\n┌ Step {step}")
        print(f"│ 💭 {thought}")

        # ② FINAL ANSWER?
        if "final_answer" in parsed:
            print(f"└ ✅ {parsed['final_answer']}")
            trace.print_summary()
            return parsed["final_answer"]

        tool_name = parsed.get("tool")
        tool_args = parsed.get("args", {})

        if not tool_name:
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": "Provide a tool call or final_answer."})
            continue

        print(f"│ 🔧 {tool_name}({json.dumps(tool_args)[:80]})")

        # ③ EXECUTE TOOL + MEASURE TIME
        t0                    = time.time()
        observation, success  = registry.execute(tool_name, tool_args)
        duration              = time.time() - t0

        trace.record(step, thought, tool_name, tool_args,
                     observation, success, duration)

        if success:
            print(f"│ 👁️  {str(observation)[:120]}")
            print(f"│ ⏱️  {round(duration,3)}s")
            print(f"└────────────────────────────────────")

            # ④ FEED SUCCESS BACK — agent sees result and plans next step
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content":
                f"Observation (step {step}): {observation}\n\n"
                f"Continue to the next step toward the goal."})

        else:
            print(f"│ ❌ Failed: {observation}")
            print(f"└────────────────────────────────────")

            consecutive_failures += 1

            # ⑤ FEED FAILURE BACK — agent must adapt
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content":
                f"Tool '{tool_name}' failed: {observation}\n\n"
                f"Try a different tool or different arguments to continue."})

            if consecutive_failures >= 3:
                print("❌ 3 consecutive failures — stopping")
                break

    trace.print_summary()
    return "Agent stopped before completing."


# ─────────────────────────────────────────────
# MAIN — 4 pipeline scenarios
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("🤖 Phase 2, Step 2.3 — Tool Result Feedback Loop\n")

    registry = build_registry()

    # ── Pipeline 1 ─────────────────────────────
    # 3 chained tools: analyze → transform → write
    run_agent(
        "Take this text: 'Learning agentic AI by building real systems on Proxmox with Ollama.' "
        "Do the following in order: "
        "1) Get all text stats. "
        "2) Convert it to a URL slug. "
        "3) Write both results to a file called 'pipeline1.txt'.",
        registry
    )

    # ── Pipeline 2 ─────────────────────────────
    # Math chain: multiple dependent calculations
    run_agent(
        "I have a server with 128GB RAM. "
        "1) Convert 128GB to MB. "
        "2) I want to allocate 35% of total RAM to containers — how many MB is that? "
        "3) If each container gets 512MB, how many containers can I run? "
        "4) Write a summary of these calculations to 'server_plan.txt'.",
        registry
    )

    # ── Pipeline 3 ─────────────────────────────
    # Report generation: gathers data from multiple tools then compiles
    run_agent(
        "Create a system report file called 'daily_report.txt'. "
        "The report should include: "
        "1) Current date and time. "
        "2) What day of the year it is. "
        "3) The MD5 hash of today's date string. "
        "4) Compile all three findings into the report using the report_builder tool.",
        registry
    )

    # ── Pipeline 4 ─────────────────────────────
    # Recovery scenario: intentional bad step in the middle
    run_agent(
        "Do these steps: "
        "1) Calculate 2 to the power of 10. "
        "2) Read a file called 'nonexistent_file.txt'. "
        "3) Even if step 2 fails, calculate what 15% of the result from step 1 is. "
        "4) Write the final calculation result to 'recovery_test.txt'.",
        registry
    )