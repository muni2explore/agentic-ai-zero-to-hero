import requests
import json
import math
import re
import os
import time
import hashlib
import logging
from datetime import datetime
from typing import Any, Callable
from llm import call_llm

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2"
LOG_DIR = "logs"
DEFAULT_LOG_FILE = os.path.join(LOG_DIR, "agent_05.log")


def setup_logging(log_file: str = DEFAULT_LOG_FILE) -> logging.Logger:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("agent_05_tool_definitions")
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
# ARGUMENT VALIDATOR
# Validates args against JSON schema BEFORE running tool
# Catches type mismatches, missing required args early
# ─────────────────────────────────────────────

class ArgValidator:
    TYPE_MAP = {
        "string":  str,
        "number":  (int, float),
        "integer": int,
        "boolean": bool,
        "array":   list,
        "object":  dict,
    }

    @classmethod
    def validate(cls, args: dict, schema: dict) -> tuple[bool, str]:
        """
        Returns (is_valid, error_message).
        error_message is empty string if valid.
        """
        properties = schema["parameters"].get("properties", {})
        required   = schema["parameters"].get("required", [])

        # Check required fields exist
        for field in required:
            if field not in args:
                return False, f"Missing required argument: '{field}'"

        # Check types
        for field, value in args.items():
            if field not in properties:
                continue  # ignore extra args
            expected_type = properties[field].get("type")
            if expected_type and expected_type in cls.TYPE_MAP:
                if not isinstance(value, cls.TYPE_MAP[expected_type]):
                    return False, (
                        f"Argument '{field}' should be {expected_type}, "
                        f"got {type(value).__name__}: {repr(value)}"
                    )

            # Check enum if present
            enum_values = properties[field].get("enum")
            if enum_values and value not in enum_values:
                return False, (
                    f"Argument '{field}' must be one of {enum_values}, got '{value}'"
                )

        return True, ""


# ─────────────────────────────────────────────
# ENHANCED TOOL REGISTRY
# Supports categories, validation, retry, usage tracking
# ─────────────────────────────────────────────

class ToolRegistry:
    def __init__(self):
        self.tools: dict[str, dict] = {}
        self.usage_count: dict[str, int] = {}
        self.error_count: dict[str, int] = {}

    def register(self, schema: dict, fn: Callable, category: str = "general"):
        name = schema["name"]
        self.tools[name] = {
            "schema":   schema,
            "fn":       fn,
            "category": category,
        }
        self.usage_count[name] = 0
        self.error_count[name] = 0

    def execute(self, name: str, args: dict, retry: int = 2) -> tuple[str, bool]:
        """
        Execute tool with validation + retry.
        Returns (result, success).
        """
        if name not in self.tools:
            available = list(self.tools.keys())
            logger.warning("Unknown tool requested: %s", name)
            return f"Unknown tool '{name}'. Available tools: {available}", False

        tool = self.tools[name]

        # Validate args first
        valid, error = ArgValidator.validate(args, tool["schema"])
        if not valid:
            logger.warning("Validation failed for tool %s: %s", name, error)
            return f"Argument error for '{name}': {error}", False

        # Execute with retry
        last_error = ""
        for attempt in range(1, retry + 1):
            try:
                result = tool["fn"](**args)
                self.usage_count[name] += 1
                logger.info("Tool executed: %s args=%s result=%s", name, args, result)
                return str(result), True
            except Exception as e:
                last_error = str(e)
                self.error_count[name] += 1
                logger.exception("Tool '%s' failed on attempt %s: %s", name, attempt, e)
                if attempt < retry:
                    time.sleep(0.5)

        logger.error("Tool '%s' failed after %s attempts: %s", name, retry, last_error)
        return f"Tool '{name}' failed after {retry} attempts: {last_error}", False

    def get_prompt_text(self, category_filter: str = None) -> str:
        """Build tool descriptions for the system prompt."""
        lines = []
        # Group by category
        categories: dict[str, list] = {}
        for name, tool in self.tools.items():
            cat = tool["category"]
            if category_filter and cat != category_filter:
                continue
            categories.setdefault(cat, []).append((name, tool))

        for cat, tools in categories.items():
            lines.append(f"[{cat.upper()} TOOLS]")
            for name, tool in tools:
                s = tool["schema"]
                params = s["parameters"]["properties"]
                required = s["parameters"].get("required", [])
                param_strs = []
                for p, info in params.items():
                    req = "*" if p in required else "?"
                    enum_hint = ""
                    if "enum" in info:
                        enum_hint = f" options={info['enum']}"
                    param_strs.append(
                        f"{p}{req}({info['type']}{enum_hint}): {info['description']}"
                    )
                lines.append(f"  {name}: {s['description']}")
                for ps in param_strs:
                    lines.append(f"    └─ {ps}")
            lines.append("")
        return "\n".join(lines)

    def print_stats(self):
        print("\n📊 Tool Usage Stats:")
        print(f"  {'Tool':<25} {'Used':>6} {'Errors':>8}")
        print(f"  {'-'*25} {'-'*6} {'-'*8}")
        for name in self.tools:
            print(f"  {name:<25} {self.usage_count[name]:>6} {self.error_count[name]:>8}")


# ─────────────────────────────────────────────
# TOOLS — 10 tools across 3 categories
# ─────────────────────────────────────────────

# ── MATH TOOLS ──────────────────────────────

def calculator_fn(expression: str) -> str:
    allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
    allowed.update({"abs": abs, "round": round})
    result = eval(expression.strip(), {"__builtins__": {}}, allowed)
    return str(result)

def unit_converter_fn(value: float, from_unit: str, to_unit: str) -> str:
    conversions = {
        ("km",  "miles"): 0.621371, ("miles","km"):     1.60934,
        ("kg",  "lbs"):   2.20462,  ("lbs",  "kg"):     0.453592,
        ("c",   "f"):     None,     ("f",    "c"):       None,
        ("mb",  "gb"):    1/1024,   ("gb",   "mb"):      1024,
        ("gb",  "tb"):    1/1024,   ("tb",   "gb"):      1024,
        ("mb",  "tb"):    1/1048576,
    }
    fu, tu = from_unit.lower(), to_unit.lower()
    if (fu, tu) == ("c", "f"):
        result = (value * 9/5) + 32
    elif (fu, tu) == ("f", "c"):
        result = (value - 32) * 5/9
    elif (fu, tu) in conversions:
        result = value * conversions[(fu, tu)]
    else:
        return f"Conversion from {from_unit} to {to_unit} not supported."
    return f"{value} {from_unit} = {round(result, 4)} {to_unit}"

def percentage_fn(operation: str, value: float, total: float = None, percent: float = None) -> str:
    if operation == "what_percent" and total:
        return f"{value} is {round((value/total)*100, 2)}% of {total}"
    elif operation == "percent_of" and percent is not None:
        return f"{percent}% of {value} = {round(value * percent / 100, 4)}"
    elif operation == "percent_change" and total:
        change = ((total - value) / value) * 100
        direction = "increase" if change > 0 else "decrease"
        return f"Change from {value} to {total} is a {round(abs(change), 2)}% {direction}"
    return "Invalid operation. Use: what_percent, percent_of, percent_change"


# ── FILE TOOLS ──────────────────────────────

def file_write_fn(filename: str, content: str, mode: str = "write") -> str:
    safe = os.path.basename(filename)
    file_mode = "a" if mode == "append" else "w"
    with open(safe, file_mode) as f:
        f.write(content)
    action = "Appended to" if mode == "append" else "Wrote"
    return f"{action} '{safe}' ({len(content)} chars)"

def file_read_fn(filename: str) -> str:
    safe = os.path.basename(filename)
    if not os.path.exists(safe):
        return f"File '{safe}' not found"
    with open(safe) as f:
        content = f.read()
    return f"[{safe}] ({len(content)} chars):\n{content}"

def file_list_fn(extension_filter: str = "") -> str:
    files = os.listdir(".")
    if extension_filter:
        files = [f for f in files if f.endswith(extension_filter)]
    if not files:
        return f"No files found with filter '{extension_filter}'"
    return "Files:\n" + "\n".join(f"  - {f}" for f in sorted(files))


# ── TEXT TOOLS ──────────────────────────────

def text_analyze_fn(text: str, operation: str) -> str:
    words = text.strip().split()
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    ops = {
        "word_count":     f"{len(words)} words",
        "char_count":     f"{len(text)} chars ({len(text.replace(' ',''))} without spaces)",
        "sentence_count": f"{len(sentences)} sentences",
        "longest_word":   f"'{max(words, key=len)}' ({len(max(words, key=len))} chars)",
        "avg_word_len":   f"{round(sum(len(w) for w in words)/len(words), 2)} chars avg",
        "all": "\n".join([
            f"Words: {len(words)}",
            f"Characters: {len(text)}",
            f"Sentences: {len(sentences)}",
            f"Longest word: '{max(words, key=len)}'",
            f"Avg word length: {round(sum(len(w) for w in words)/len(words), 2)}",
        ])
    }
    return ops.get(operation, f"Unknown operation. Use: {list(ops.keys())}")

def text_transform_fn(text: str, operation: str) -> str:
    ops = {
        "uppercase":    text.upper(),
        "lowercase":    text.lower(),
        "title_case":   text.title(),
        "reverse":      text[::-1],
        "word_reverse": " ".join(text.split()[::-1]),
        "hash_md5":     hashlib.md5(text.encode()).hexdigest(),
        "slug":         re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-'),
    }
    return ops.get(operation, f"Unknown operation. Use: {list(ops.keys())}")

def datetime_fn(info_type: str) -> str:
    now = datetime.now()
    types = {
        "full":        now.strftime("%Y-%m-%d %H:%M:%S, %A"),
        "date":        now.strftime("%Y-%m-%d"),
        "time":        now.strftime("%H:%M:%S"),
        "day":         now.strftime("%A"),
        "week_number": str(now.isocalendar()[1]),
        "day_of_year": str(now.timetuple().tm_yday),
        "timestamp":   str(int(now.timestamp())),
    }
    return types.get(info_type, types["full"])


# ─────────────────────────────────────────────
# REGISTER ALL TOOLS
# ─────────────────────────────────────────────

def build_registry() -> ToolRegistry:
    r = ToolRegistry()

    # Math
    r.register({
        "name": "calculator",
        "description": "Evaluate any mathematical expression including trigonometry and logarithms.",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string", "description": "Python math expression e.g. '2 ** 10' or 'math.sqrt(144)'"}
        }, "required": ["expression"]}
    }, calculator_fn, category="math")

    r.register({
        "name": "unit_converter",
        "description": "Convert between units: km/miles, kg/lbs, c/f (celsius/fahrenheit), mb/gb/tb.",
        "parameters": {"type": "object", "properties": {
            "value":     {"type": "number",  "description": "The numeric value to convert"},
            "from_unit": {"type": "string",  "description": "Source unit: km, miles, kg, lbs, c, f, mb, gb, tb"},
            "to_unit":   {"type": "string",  "description": "Target unit: km, miles, kg, lbs, c, f, mb, gb, tb"},
        }, "required": ["value", "from_unit", "to_unit"]}
    }, unit_converter_fn, category="math")

    r.register({
        "name": "percentage",
        "description": "Calculate percentages: what percent X is of Y, X% of Y, or percent change.",
        "parameters": {"type": "object", "properties": {
            "operation": {"type": "string",  "description": "One of: what_percent, percent_of, percent_change",
                          "enum": ["what_percent", "percent_of", "percent_change"]},
            "value":     {"type": "number",  "description": "The base value or original amount"},
            "total":     {"type": "number",  "description": "The total or new value (for what_percent or percent_change)"},
            "percent":   {"type": "number",  "description": "The percentage to apply (for percent_of)"},
        }, "required": ["operation", "value"]}
    }, percentage_fn, category="math")

    # File
    r.register({
        "name": "file_write",
        "description": "Write or append text content to a file.",
        "parameters": {"type": "object", "properties": {
            "filename": {"type": "string", "description": "Filename to write, e.g. 'notes.txt'"},
            "content":  {"type": "string", "description": "Text content to write"},
            "mode":     {"type": "string", "description": "write (overwrite) or append",
                         "enum": ["write", "append"]},
        }, "required": ["filename", "content"]}
    }, file_write_fn, category="file")

    r.register({
        "name": "file_read",
        "description": "Read the contents of a file.",
        "parameters": {"type": "object", "properties": {
            "filename": {"type": "string", "description": "Filename to read"},
        }, "required": ["filename"]}
    }, file_read_fn, category="file")

    r.register({
        "name": "file_list",
        "description": "List files in the current directory, optionally filtered by extension.",
        "parameters": {"type": "object", "properties": {
            "extension_filter": {"type": "string",
                                 "description": "File extension to filter by, e.g. '.txt' or '.py'. Empty string for all files."},
        }, "required": []}
    }, file_list_fn, category="file")

    # Text
    r.register({
        "name": "text_analyze",
        "description": "Analyze text: count words, chars, sentences, find longest word, or get all stats.",
        "parameters": {"type": "object", "properties": {
            "text":      {"type": "string", "description": "Text to analyze"},
            "operation": {"type": "string",
                          "description": "One of: word_count, char_count, sentence_count, longest_word, avg_word_len, all",
                          "enum": ["word_count", "char_count", "sentence_count", "longest_word", "avg_word_len", "all"]},
        }, "required": ["text", "operation"]}
    }, text_analyze_fn, category="text")

    r.register({
        "name": "text_transform",
        "description": "Transform text: change case, reverse, create URL slug, or generate MD5 hash.",
        "parameters": {"type": "object", "properties": {
            "text":      {"type": "string", "description": "Text to transform"},
            "operation": {"type": "string",
                          "description": "One of: uppercase, lowercase, title_case, reverse, word_reverse, hash_md5, slug",
                          "enum": ["uppercase", "lowercase", "title_case", "reverse", "word_reverse", "hash_md5", "slug"]},
        }, "required": ["text", "operation"]}
    }, text_transform_fn, category="text")

    r.register({
        "name": "datetime",
        "description": "Get current date/time info: full datetime, date, time, day of week, week number, day of year, unix timestamp.",
        "parameters": {"type": "object", "properties": {
            "info_type": {"type": "string",
                          "description": "One of: full, date, time, day, week_number, day_of_year, timestamp",
                          "enum": ["full", "date", "time", "day", "week_number", "day_of_year", "timestamp"]},
        }, "required": ["info_type"]}
    }, datetime_fn, category="text")

    return r


# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────

def build_system_prompt(registry: ToolRegistry) -> str:
    return f"""You are a precise agent with access to tools organized in categories.

{registry.get_prompt_text()}
RESPONSE FORMAT — always respond with ONLY valid JSON, one of these two shapes:

To call a tool:
{{"thought": "why I'm using this tool", "tool": "tool_name", "args": {{"param": "value"}}}}

When done:
{{"thought": "I have everything I need", "final_answer": "complete answer here"}}

IMPORTANT:
- Pick the most specific tool for the job (use percentage tool for % questions, not calculator)
- Parameters marked * are required, ? are optional
- For enum parameters, use ONLY the listed values exactly
- Never fabricate results — always use tool observations
- ONLY output ONE JSON object. Do NOT output multiple lines of JSON.
- Never wrap your JSON in `<think>` tags or include reasoning outside the JSON.
- For multi-step tasks: Output ONE tool call, wait for the observation, then decide the next step.
"""


# ─────────────────────────────────────────────
# AGENT LOOP
# ─────────────────────────────────────────────

def parse_response(raw: str) -> dict:
    cleaned = re.sub(r'^```(?:json)?\s*', '', raw.strip())
    cleaned = re.sub(r'\s*```$', '', cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
    return {"parse_error": True, "raw": raw}


def run_agent(goal: str, registry: ToolRegistry, max_steps: int = 10):
    print(f"\n{'='*60}")
    print(f"🎯 {goal}")
    print('='*60)

    messages = [
        {"role": "system", "content": build_system_prompt(registry)},
        {"role": "user",   "content": goal}
    ]

    logger.info("Starting agent run: %s", goal)
    logger.info("Initial messages: %s", messages)

    for step in range(1, max_steps + 1):
        raw = call_llm(messages)

        parsed = parse_response(raw)

        if parsed.get("parse_error"):
            print(f"\n⚠️  Step {step}: JSON parse failed")
            print(f"   Raw: {parsed['raw'][:150]}...")
            logger.warning("JSON parse failed at step %s: %s", step, parsed['raw'])
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content":
                'Invalid JSON. Respond with exactly: {"thought": "...", "tool": "...", "args": {...}}'})
            continue

        thought = parsed.get("thought", "")
        print(f"\n┌ Step {step}")
        print(f"│ 💭 {thought}")
        logger.info("Step %s thought: %s", step, thought)

        if "final_answer" in parsed:
            print(f"└ ✅ {parsed['final_answer']}")
            return parsed["final_answer"]

        tool_name = parsed.get("tool")
        tool_args = parsed.get("args", {})

        if not tool_name:
            logger.warning("No tool name returned at step %s. Raw response: %s", step, raw)
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": "Provide a tool call or final_answer."})
            continue

        print(f"│ 🔧 {tool_name}({json.dumps(tool_args)})")
        logger.info("Calling tool %s with args %s", tool_name, tool_args)

        observation, success = registry.execute(tool_name, tool_args)

        if not success:
            print(f"│ ❌ Tool failed: {observation}")
            logger.warning("Tool %s failed at step %s: %s", tool_name, step, observation)
            # Tell agent the tool failed so it can try differently
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content":
                f"Tool failed: {observation}\nTry a different tool or different arguments."})
            continue

        print(f"│ 👁️  {observation}")
        logger.info("Tool %s observation: %s", tool_name, observation)
        print(f"└────────────────────────────────")

        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    return "Max steps reached."


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("🤖 Phase 2, Step 2.2 — Structured Tool Definitions\n")

    registry = build_registry()

    # Test 1: Tool selection — should pick percentage tool, not calculator
    run_agent(
        "What percentage is 347 out of 1250?",
        registry
    )

    # Test 2: Unit conversion chain
    run_agent(
        "I have a 2TB hard disk. How many GB is that? And how many MB?",
        registry
    )

    # Test 3: Validation — deliberately wrong enum to test error handling
    run_agent(
        "Transform the text 'Hello World Muni' into a URL-friendly slug",
        registry
    )

    # Test 4: Multi-category — uses math + file + text tools
    run_agent(
        "Analyze this sentence and get all stats: 'Building agentic AI systems from scratch is the best way to learn deeply.' Then write the stats to a file called 'analysis.txt'.",
        registry
    )

    # Test 5: Wrong tool test — agent must recover
    run_agent(
        "List all .py files and tell me how many there are",
        registry
    )

    # Print usage stats
    registry.print_stats()