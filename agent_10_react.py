import json
import re
import math
import time
import os
import hashlib
from datetime import datetime
import logging
from typing import Callable, Optional
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
# TOOL REGISTRY (from Phase 2 — lean version)
# ─────────────────────────────────────────────

class Tool:
    def __init__(self, name: str, description: str,
                 params: dict, fn: Callable, category: str = "general"):
        self.name        = name
        self.description = description
        self.params      = params   # JSON schema properties
        self.required    = params.get("required", [])
        self.fn          = fn
        self.category    = category

    def validate(self, args: dict) -> tuple[bool, str]:
        for field in self.required:
            if field not in args:
                return False, f"Missing required: '{field}'"
        return True, ""

    def run(self, args: dict) -> tuple[str, bool]:
        valid, err = self.validate(args)
        if not valid:
            return err, False
        try:
            return str(self.fn(**args)), True
        except Exception as e:
            return f"Error: {e}", False

    def schema_text(self) -> str:
        props = self.params.get("properties", {})
        lines = [f"{self.name}: {self.description}"]
        for p, info in props.items():
            req      = "*" if p in self.required else "?"
            enum_str = f" [{'/'.join(info['enum'])}]" if "enum" in info else ""
            lines.append(f"  {p}{req}: {info.get('description','')}{enum_str}")
        return "\n".join(lines)


class ToolRegistry:
    def __init__(self):
        self.tools: dict[str, Tool] = {}

    def add(self, tool: Tool):
        self.tools[tool.name] = tool

    def run(self, name: str, args: dict) -> tuple[str, bool]:
        if name not in self.tools:
            similar = [t for t in self.tools if name.lower() in t.lower()]
            hint    = f" Did you mean: {similar}?" if similar else ""
            return f"Unknown tool '{name}'.{hint} Available: {list(self.tools.keys())}", False
        return self.tools[name].run(args)

    def prompt_block(self) -> str:
        return "\n\n".join(t.schema_text() for t in self.tools.values())


# ─────────────────────────────────────────────
# REACT TRACE
# Records every Thought/Action/Observation step
# Gives you full replay + debugging
# ─────────────────────────────────────────────

class ReActTrace:
    def __init__(self, goal: str):
        self.goal       = goal
        self.steps: list[dict] = []
        self.started    = time.time()

    def record(self, step_type: str, content: str,
               tool: str = None, args: dict = None,
               success: bool = True, duration: float = 0.0):
        self.steps.append({
            "type":     step_type,   # THOUGHT / ACTION / OBSERVATION / ANSWER
            "content":  content,
            "tool":     tool,
            "args":     args,
            "success":  success,
            "duration": round(duration, 3),
            "time":     datetime.now().strftime("%H:%M:%S"),
        })

    def print(self):
        elapsed = round(time.time() - self.started, 2)
        print(f"\n{'─'*60}")
        print(f"📋 ReAct Trace — '{self.goal[:50]}'")
        print(f"{'─'*60}")
        for i, s in enumerate(self.steps, 1):
            icon = {"THOUGHT":"💭","ACTION":"🔧","OBSERVATION":"👁️",
                    "ANSWER":"✅","ERROR":"❌"}.get(s["type"], "•")
            print(f"\n  {icon} Step {i}: {s['type']}  [{s['time']}]")
            if s["tool"]:
                print(f"     Tool: {s['tool']}({json.dumps(s['args'])[:60]})")
            print(f"     {s['content'][:120]}")
            if s["duration"]:
                print(f"     ⏱  {s['duration']}s")
        print(f"\n  Total steps: {len(self.steps)} | Wall: {elapsed}s")


# ─────────────────────────────────────────────
# REACT SYSTEM PROMPT
# The exact format the agent must follow.
# Strict format = reliable parsing.
# ─────────────────────────────────────────────

def build_react_prompt(registry: ToolRegistry) -> str:
    return f"""You are a reasoning agent using the ReAct framework.

AVAILABLE TOOLS:
{registry.prompt_block()}

STRICT OUTPUT FORMAT — follow exactly:

To reason:
Thought: <your reasoning about what to do next>

To use a tool:
Thought: <why you need this tool>
Action: <tool_name>
Args: {{"param": "value"}}

To give the final answer:
Thought: <final reasoning>
Answer: <your complete answer>

RULES:
- Always start with Thought
- ONE action per response
- Never skip the Args line when using a tool
- Use Answer only when the task is fully complete
- If a tool fails, think about why and try differently
- Never fabricate observations — only use actual tool results
"""


# ─────────────────────────────────────────────
# REACT PARSER
# Extracts Thought / Action / Args / Answer
# Handles model quirks and partial outputs
# ─────────────────────────────────────────────

def parse_react(text: str) -> dict:
    result = {
        "thought":    None,
        "action":     None,
        "args":       {},
        "answer":     None,
        "parse_ok":   True,
        "raw":        text,
    }

    # Thought
    m = re.search(r"Thought:\s*(.+?)(?=\nAction:|\nAnswer:|$)",
                  text, re.DOTALL | re.IGNORECASE)
    if m:
        result["thought"] = m.group(1).strip()

    # Answer
    m = re.search(r"Answer:\s*(.+?)$", text, re.DOTALL | re.IGNORECASE)
    if m:
        result["answer"] = m.group(1).strip()
        return result

    # Action
    m = re.search(r"Action:\s*(\w+)", text, re.IGNORECASE)
    if m:
        result["action"] = m.group(1).strip()

    # Args — try JSON block first, then inline
    m = re.search(r"Args:\s*(\{.*?\})", text, re.DOTALL | re.IGNORECASE)
    if m:
        try:
            result["args"] = json.loads(m.group(1))
        except json.JSONDecodeError:
            # Try to extract key-value pairs manually
            kv = re.findall(r'"(\w+)":\s*"([^"]*)"', m.group(1))
            result["args"] = dict(kv) if kv else {}

    if not result["thought"] and not result["action"]:
        result["parse_ok"] = False

    return result


# ─────────────────────────────────────────────
# REACT AGENT
# The core loop: Think → Act → Observe → repeat
# ─────────────────────────────────────────────

def run_react(goal: str, registry: ToolRegistry,
              max_steps: int = 12,
              verbose: bool = True) -> tuple[str, ReActTrace]:
    trace    = ReActTrace(goal)
    messages = [
        {"role": "system", "content": build_react_prompt(registry)},
        {"role": "user",   "content": f"Goal: {goal}"},
    ]

    if verbose:
        print(f"\n{'='*60}")
        print(f"🎯 Goal: {goal}")
        print('='*60)

    consecutive_errors = 0

    for step in range(1, max_steps + 1):

        # ① THINK
        t0  = time.time()
        raw = call_llm(messages)
        llm_time = time.time() - t0

        parsed = parse_react(raw)

        if not parsed["parse_ok"]:
            consecutive_errors += 1
            if verbose:
                print(f"\n⚠️  Step {step}: parse failed ({consecutive_errors}/3)")
            if consecutive_errors >= 3:
                break
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content":
                "Follow the format strictly:\n"
                "Thought: <reasoning>\n"
                "Action: <tool_name>\n"
                'Args: {"param": "value"}'})
            continue

        consecutive_errors = 0
        thought = parsed["thought"] or ""

        if verbose:
            print(f"\n┌ Step {step}")
            print(f"│ 💭 {thought}")

        trace.record("THOUGHT", thought, duration=llm_time)

        # ② FINAL ANSWER?
        if parsed["answer"]:
            if verbose:
                print(f"└ ✅ {parsed['answer']}")
            trace.record("ANSWER", parsed["answer"])
            return parsed["answer"], trace

        # ③ ACT
        action = parsed["action"]
        args   = parsed["args"]

        if not action:
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user",
                             "content": "Provide an Action or Answer."})
            continue

        if verbose:
            print(f"│ 🔧 {action}({json.dumps(args)[:60]})")

        t0                   = time.time()
        observation, success = registry.run(action, args)
        tool_time            = time.time() - t0

        if verbose:
            icon = "👁️" if success else "❌"
            print(f"│ {icon} {str(observation)[:100]}")
            print(f"│ ⏱  llm={round(llm_time,2)}s tool={round(tool_time,3)}s")
            print(f"└────────────────────────────────────")

        trace.record(
            "ACTION" if success else "ERROR",
            observation, tool=action, args=args,
            success=success, duration=tool_time
        )
        trace.record("OBSERVATION", observation, success=success)

        # ④ FEED BACK
        messages.append({"role": "assistant", "content": raw})

        if success:
            messages.append({"role": "user",
                "content": f"Observation: {observation}\n\nContinue."})
        else:
            messages.append({"role": "user",
                "content": f"Tool failed: {observation}\n"
                           f"Try a different approach."})

    final = "Max steps reached without answer."
    trace.record("ERROR", final)
    return final, trace


# ─────────────────────────────────────────────
# TOOLS — full set for testing
# ─────────────────────────────────────────────

def build_tools() -> ToolRegistry:
    r = ToolRegistry()

    # Math
    r.add(Tool("calculator", "Evaluate math expressions",
        {"properties": {"expression": {"type": "string",
            "description": "Python math expression e.g. '2**10' or 'math.sqrt(144)'"}},
         "required": ["expression"]},
        lambda expression: str(eval(expression, {"__builtins__": {}},
            {k: v for k, v in math.__dict__.items() if not k.startswith("_")})),
        "math"))

    r.add(Tool("unit_convert", "Convert between units: km/miles, kg/lbs, gb/mb, c/f",
        {"properties": {
            "value":     {"type": "number", "description": "Value to convert"},
            "from_unit": {"type": "string",  "description": "e.g. km, miles, kg, c, gb"},
            "to_unit":   {"type": "string",  "description": "e.g. miles, km, lbs, f, mb"}},
         "required": ["value", "from_unit", "to_unit"]},
        lambda value, from_unit, to_unit: _unit_convert(value, from_unit, to_unit),
        "math"))

    # File
    r.add(Tool("file_write", "Write content to a file",
        {"properties": {
            "filename": {"type": "string", "description": "filename e.g. notes.txt"},
            "content":  {"type": "string", "description": "text to write"},
            "mode":     {"type": "string", "description": "write or append",
                         "enum": ["write", "append"]}},
         "required": ["filename", "content"]},
        lambda filename, content, mode="write":
            _file_write(filename, content, mode), "file"))

    r.add(Tool("file_read", "Read a file's contents",
        {"properties": {
            "filename": {"type": "string", "description": "file to read"}},
         "required": ["filename"]},
        lambda filename: _file_read(filename), "file"))

    r.add(Tool("file_list", "List files in current directory",
        {"properties": {
            "extension": {"type": "string",
                          "description": "filter by extension e.g. .py or .txt"}},
         "required": []},
        lambda extension="": _file_list(extension), "file"))

    # Text
    r.add(Tool("text_analyze", "Analyze text statistics",
        {"properties": {
            "text":      {"type": "string", "description": "text to analyze"},
            "operation": {"type": "string",
                          "description": "word_count|char_count|sentence_count|all",
                          "enum": ["word_count","char_count","sentence_count","all"]}},
         "required": ["text", "operation"]},
        lambda text, operation: _text_analyze(text, operation), "text"))

    r.add(Tool("text_transform", "Transform text",
        {"properties": {
            "text":      {"type": "string", "description": "text to transform"},
            "operation": {"type": "string",
                          "description": "uppercase|lowercase|title_case|slug|reverse",
                          "enum": ["uppercase","lowercase","title_case","slug","reverse"]}},
         "required": ["text", "operation"]},
        lambda text, operation: _text_transform(text, operation), "text"))

    r.add(Tool("datetime", "Get current date and time info",
        {"properties": {
            "info_type": {"type": "string",
                          "description": "full|date|time|day|timestamp",
                          "enum": ["full","date","time","day","timestamp"]}},
         "required": ["info_type"]},
        lambda info_type: _datetime(info_type), "util"))

    return r


# Tool implementations
def _unit_convert(value: float, from_unit: str, to_unit: str) -> str:
    table = {
        ("km","miles"):0.621371, ("miles","km"):1.60934,
        ("kg","lbs"):2.20462,   ("lbs","kg"):0.453592,
        ("gb","mb"):1024,        ("mb","gb"):1/1024,
        ("gb","tb"):1/1024,      ("tb","gb"):1024,
    }
    f, t = from_unit.lower(), to_unit.lower()
    if (f,t) == ("c","f"): return f"{round((value*9/5)+32,2)} f"
    if (f,t) == ("f","c"): return f"{round((value-32)*5/9,2)} c"
    if (f,t) in table:     return f"{round(value*table[(f,t)],4)} {to_unit}"
    return f"Unsupported: {from_unit} → {to_unit}"

def _file_write(filename: str, content: str, mode: str = "write") -> str:
    safe = os.path.basename(filename)
    with open(safe, "a" if mode == "append" else "w") as f:
        f.write(content)
    return f"Wrote {len(content)} chars to '{safe}'"

def _file_read(filename: str) -> str:
    safe = os.path.basename(filename)
    if not os.path.exists(safe): return f"File '{safe}' not found"
    return open(safe).read()

def _file_list(extension: str = "") -> str:
    files = [f for f in os.listdir(".") if f.endswith(extension)]
    return "\n".join(sorted(files)) if files else "No files found"

def _text_analyze(text: str, operation: str) -> str:
    words = text.strip().split()
    sents = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    ops   = {
        "word_count":     str(len(words)),
        "char_count":     str(len(text)),
        "sentence_count": str(len(sents)),
        "all":            f"words={len(words)} chars={len(text)} sentences={len(sents)}",
    }
    return ops.get(operation, f"Unknown: {operation}")

def _text_transform(text: str, operation: str) -> str:
    return {
        "uppercase":  text.upper(),
        "lowercase":  text.lower(),
        "title_case": text.title(),
        "slug":       re.sub(r'[^a-z0-9]+','-',text.lower()).strip('-'),
        "reverse":    text[::-1],
    }.get(operation, f"Unknown: {operation}")

def _datetime(info_type: str) -> str:
    now = datetime.now()
    return {
        "full":      now.strftime("%Y-%m-%d %H:%M:%S, %A"),
        "date":      now.strftime("%Y-%m-%d"),
        "time":      now.strftime("%H:%M:%S"),
        "day":       now.strftime("%A"),
        "timestamp": str(int(now.timestamp())),
    }.get(info_type, now.strftime("%Y-%m-%d %H:%M:%S"))


# ─────────────────────────────────────────────
# MAIN — 5 progressively complex goals
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("🤖 Phase 4, Step 4.1 — ReAct Loop (Production Grade)\n")

    registry = build_tools()

    goals = [
        # 1. Single tool — baseline
        "What is the square root of 1764?",

        # 2. Two-tool chain with dependency
        "Convert 5.5 GB to MB, then calculate what 20% of that is.",

        # 3. Multi-tool pipeline with file output
        "Get today's date. Calculate how many days are left in 2025 "
        "(assume 2025 has 365 days, today is day {day_of_year}). "
        "Write the result to 'days_left.txt'.",

        # 4. Text pipeline
        "Take the text 'Building Agentic AI Systems on Proxmox with Ollama'. "
        "Get all text stats. Convert it to a slug. "
        "Write both results to 'text_report.txt'.",

        # 5. Recovery — file that doesn't exist
        "Read the file 'config.yaml'. If it doesn't exist, "
        "create it with content 'model: llama3.2\nport: 11434\nhost: localhost'. "
        "Then read it back and confirm the content.",
    ]

    traces = []
    for goal in goals:
        answer, trace = run_react(goal, registry, max_steps=10)
        traces.append(trace)

    # Print all traces
    print("\n\n" + "="*60)
    print("📊 ALL TRACES")
    print("="*60)
    for trace in traces:
        trace.print()