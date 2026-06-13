import requests
import json
import math
import re
import os
from datetime import datetime
from dotenv import load_dotenv
from llm import call_llm

load_dotenv()

OLLAMA_URL = os.environ.get('OLLAMA_URL')
MODEL = os.environ.get('MODEL') 

# ─────────────────────────────────────────────
# TOOL REGISTRY
# Each tool has: schema (what the LLM sees) + fn (what Python runs)
# ─────────────────────────────────────────────

class ToolRegistry:
    def __init__(self):
        self.tools = {}  # name → {"schema": ..., "fn": ...}

    def register(self, schema: dict, fn):
        """Register a tool with its JSON schema and Python function."""
        name = schema["name"]
        self.tools[name] = {"schema": schema, "fn": fn}
        print(f"  ✅ Registered tool: {name}")

    def get_schemas(self) -> list[dict]:
        """Return all schemas — this goes into the system prompt."""
        return [t["schema"] for t in self.tools.values()]

    def execute(self, name: str, args: dict) -> str:
        """Execute a tool by name with given args."""
        if name not in self.tools:
            return f"Error: Unknown tool '{name}'. Available: {list(self.tools.keys())}"
        try:
            return str(self.tools[name]["fn"](**args))
        except TypeError as e:
            return f"Error: Wrong arguments for '{name}': {e}"
        except Exception as e:
            return f"Error running '{name}': {e}"

    def schemas_as_text(self) -> str:
        """Format schemas as readable text for the system prompt."""
        lines = []
        for name, tool in self.tools.items():
            s = tool["schema"]
            lines.append(f"Tool: {s['name']}")
            lines.append(f"  Description: {s['description']}")
            lines.append(f"  Parameters:")
            for param, info in s["parameters"]["properties"].items():
                required = param in s["parameters"].get("required", [])
                req_str = "(required)" if required else "(optional)"
                lines.append(f"    - {param} {req_str}: {info['description']} [{info['type']}]")
            lines.append("")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# TOOL DEFINITIONS
# schema = what the LLM reads to understand the tool
# fn     = actual Python function that runs
# ─────────────────────────────────────────────

# --- Tool 1: Calculator ---
calculator_schema = {
    "name": "calculator",
    "description": "Evaluate mathematical expressions. Use for any arithmetic, percentages, or formula calculations.",
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "A valid Python math expression. Example: '24 * 60' or 'round(100 * 0.175, 2)'"
            }
        },
        "required": ["expression"]
    }
}
def calculator_fn(expression: str) -> str:
    allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
    allowed.update({"abs": abs, "round": round})
    try:
        result = eval(expression.strip(), {"__builtins__": {}}, allowed)
        return f"{result}"
    except Exception as e:
        return f"Math error: {e}"


# --- Tool 2: File Writer ---
file_writer_schema = {
    "name": "file_writer",
    "description": "Write text content to a file on disk. Use when asked to save, create, or write a file.",
    "parameters": {
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "Name of the file to write, e.g. 'report.txt' or 'notes.md'"
            },
            "content": {
                "type": "string",
                "description": "The text content to write into the file"
            }
        },
        "required": ["filename", "content"]
    }
}
def file_writer_fn(filename: str, content: str) -> str:
    safe_name = os.path.basename(filename)  # prevent path traversal
    with open(safe_name, "w") as f:
        f.write(content)
    return f"Successfully wrote {len(content)} characters to '{safe_name}'"


# --- Tool 3: File Reader ---
file_reader_schema = {
    "name": "file_reader",
    "description": "Read the contents of a file from disk.",
    "parameters": {
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "Name of the file to read"
            }
        },
        "required": ["filename"]
    }
}
def file_reader_fn(filename: str) -> str:
    safe_name = os.path.basename(filename)
    if not os.path.exists(safe_name):
        return f"Error: File '{safe_name}' does not exist"
    with open(safe_name, "r") as f:
        content = f.read()
    return f"Contents of '{safe_name}':\n{content}"


# --- Tool 4: DateTime ---
datetime_schema = {
    "name": "get_datetime",
    "description": "Get the current date, time, day of week, or other time-related information.",
    "parameters": {
        "type": "object",
        "properties": {
            "format": {
                "type": "string",
                "description": "What to return: 'full', 'date', 'time', 'day', 'week_number'",
            }
        },
        "required": ["format"]
    }
}
def datetime_fn(format: str = "full") -> str:
    now = datetime.now()
    formats = {
        "full":        now.strftime("%Y-%m-%d %H:%M:%S, %A, week %W of %Y"),
        "date":        now.strftime("%Y-%m-%d"),
        "time":        now.strftime("%H:%M:%S"),
        "day":         now.strftime("%A"),
        "week_number": str(now.isocalendar()[1]),
    }
    return formats.get(format, formats["full"])


# --- Tool 5: Text Analyzer ---
text_analyzer_schema = {
    "name": "text_analyzer",
    "description": "Analyze a piece of text: count words, characters, sentences, or find the longest word.",
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to analyze"
            },
            "analysis_type": {
                "type": "string",
                "description": "Type of analysis: 'word_count', 'char_count', 'sentence_count', 'longest_word', 'all'"
            }
        },
        "required": ["text", "analysis_type"]
    }
}
def text_analyzer_fn(text: str, analysis_type: str) -> str:
    words = text.strip().split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    results = {
        "word_count":      f"Word count: {len(words)}",
        "char_count":      f"Character count: {len(text)} (without spaces: {len(text.replace(' ',''))})",
        "sentence_count":  f"Sentence count: {len(sentences)}",
        "longest_word":    f"Longest word: '{max(words, key=len)}' ({len(max(words, key=len))} chars)",
    }
    if analysis_type == "all":
        return "\n".join(results.values())
    return results.get(analysis_type, f"Unknown analysis type: {analysis_type}")


# ─────────────────────────────────────────────
# SYSTEM PROMPT BUILDER
# Dynamically built from tool registry
# ─────────────────────────────────────────────

def build_system_prompt(registry: ToolRegistry) -> str:
    return f"""You are a helpful agent with access to tools.

AVAILABLE TOOLS:
{registry.schemas_as_text()}
RESPONSE FORMAT:
When you need to use a tool, respond with ONLY this JSON (no other text):
{{
  "thought": "your reasoning about what to do",
  "tool": "tool_name",
  "args": {{ "param1": "value1" }}
}}

When you have the final answer (no more tools needed), respond with ONLY this JSON:
{{
  "thought": "your final reasoning",
  "final_answer": "your complete answer to the user"
}}

RULES:
- Always respond with valid JSON — nothing else
- Always start with a Thought
- Only call ONE tool per response
- STOP generating text immediately after the Action.
- Do NOT generate an "Observation" yourself. 
- Wait for the user to provide the "Observation" in the next message.
- Never make up tool results — always use the actual Observation
- When you have enough information, give the Final Answer
- Use tools when you need real data or calculations
- Never guess or make up results — always use tools
- You may call tools multiple times to complete a task
"""


# ─────────────────────────────────────────────
# JSON PARSER — robust, handles model quirks
# ─────────────────────────────────────────────

def parse_response(text: str) -> dict:
    """Parse JSON from LLM output, handling common model quirks."""
    # Strip markdown code fences if present
    cleaned = text.strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try extracting JSON object from messy output
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {"error": "Could not parse JSON", "raw": text}


# ─────────────────────────────────────────────
# AGENT LOOP
# ─────────────────────────────────────────────

def run_agent(goal: str, registry: ToolRegistry, max_steps: int = 8):
    print(f"\n{'='*60}")
    print(f"🎯 Goal: {goal}")
    print('='*60)

    messages = [
        {"role": "system", "content": build_system_prompt(registry)},
        {"role": "user",   "content": goal}
    ]

    for step in range(1, max_steps + 1):
        print(f"\n┌── Step {step} ───────────────────────────────")

        # ① THINK
        raw = call_llm(messages)

        parsed = parse_response(raw)

        if "error" in parsed:
            print(f"│ ⚠️  Parse error: {parsed['error']}")
            print(f"│ Raw: {parsed.get('raw','')[:200]}")
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": 'Respond with valid JSON only: {"thought": "...", "tool": "...", "args": {...}} or {"thought": "...", "final_answer": "..."}'})
            continue

        thought = parsed.get("thought", "")
        print(f"│ 💭 Thought: {thought}")

        # ② DONE?
        if "final_answer" in parsed:
            print(f"└── ✅ Final Answer: {parsed['final_answer']}")
            return parsed["final_answer"]

        # ③ ACT
        tool_name = parsed.get("tool")
        tool_args = parsed.get("args", {})

        if not tool_name:
            print("│ ⚠️  No tool specified and no final_answer")
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": "Please continue with a tool call or final_answer."})
            continue

        print(f"│ 🔧 Tool: {tool_name}")
        print(f"│ 📥 Args: {json.dumps(tool_args)}")

        # ④ OBSERVE
        observation = registry.execute(tool_name, tool_args)
        print(f"│ 👁️  Observation: {observation}")
        print(f"└────────────────────────────────────────────")

        # ⑤ FEED BACK
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": f"Observation: {observation}\n\nContinue to the next step."})

    return "Max steps reached."


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("🤖 Phase 2, Step 2.1 — Structured Tool Calling\n")

    # Build registry
    registry = ToolRegistry()
    print("Registering tools...")
    registry.register(calculator_schema,     calculator_fn)
    registry.register(file_writer_schema,    file_writer_fn)
    registry.register(file_reader_schema,    file_reader_fn)
    registry.register(datetime_schema,       datetime_fn)
    registry.register(text_analyzer_schema,  text_analyzer_fn)

    # --- Test 1: Single tool ---
    run_agent(
        "What is the area of a circle with radius 7? Use pi = 3.14159",
        registry
    )

    # --- Test 2: Multi-tool chain ---
    run_agent(
        "What day of the week is it today? Also calculate how many hours are in the current week number.",
        registry
    )

    # --- Test 3: File write then read ---
    run_agent(
        "Write a short summary about what an AI agent is to a file called 'agent_summary.txt', then read it back to confirm it was saved.",
        registry
    )

    # --- Test 4: Multi-step reasoning ---
    run_agent(
        "Analyze this text and tell me the word count, then calculate how many words per sentence on average: 'AI agents are systems that perceive their environment. They reason about what actions to take. Then they execute those actions and observe the results.'",
        registry
    )