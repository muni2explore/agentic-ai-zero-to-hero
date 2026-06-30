import json
import re
import math
import time
import os
from datetime import datetime
from typing import Callable, Optional
from dataclasses import dataclass, field
import logging
from llm import call_llm

LOG_DIR = "logs"
DEFAULT_LOG_FILE = os.path.join(LOG_DIR, "agent_11.log")

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
# PLAN DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class SubTask:
    id:           int
    description:  str              # what to do
    tool:         Optional[str]    # which tool (None = LLM reasoning only)
    args_template: dict            # args — may reference prior results
    depends_on:   list[int]        # which step ids must complete first
    status:       str = "pending"  # pending | running | done | failed | skipped
    result:       str = ""
    error:        str = ""
    started_at:   float = 0.0
    ended_at:     float = 0.0

    @property
    def duration(self) -> float:
        if self.started_at and self.ended_at:
            return round(self.ended_at - self.started_at, 3)
        return 0.0


@dataclass
class Plan:
    goal:       str
    subtasks:   list[SubTask] = field(default_factory=list)
    created_at: float         = field(default_factory=time.time)
    strategy:   str           = ""   # LLM's overall approach description

    def add(self, subtask: SubTask):
        self.subtasks.append(subtask)

    def get(self, task_id: int) -> Optional[SubTask]:
        return next((t for t in self.subtasks if t.id == task_id), None)

    def ready_tasks(self) -> list[SubTask]:
        """Tasks whose dependencies are all done."""
        done_ids = {t.id for t in self.subtasks if t.status == "done"}
        return [
            t for t in self.subtasks
            if t.status == "pending"
            and all(dep in done_ids for dep in t.depends_on)
        ]

    def is_complete(self) -> bool:
        return all(t.status in ("done", "skipped", "failed")
                   for t in self.subtasks)

    def failed_tasks(self) -> list[SubTask]:
        return [t for t in self.subtasks if t.status == "failed"]

    def results_so_far(self) -> dict:
        """Map of task_id → result for completed tasks."""
        return {
            t.id: t.result
            for t in self.subtasks
            if t.status == "done"
        }

    def print(self):
        print(f"\n{'─'*60}")
        print(f"📋 Plan: '{self.goal[:50]}'")
        if self.strategy:
            print(f"   Strategy: {self.strategy[:100]}")
        print(f"{'─'*60}")
        icons = {
            "pending":"⬜", "running":"🔄",
            "done":"✅",    "failed":"❌",
            "skipped":"⏭️",
        }
        for t in self.subtasks:
            icon = icons.get(t.status, "•")
            deps = f" (needs: {t.depends_on})" if t.depends_on else ""
            tool = f" [{t.tool}]" if t.tool else " [llm]"
            print(f"  {icon} Step {t.id}{tool}{deps}: {t.description}")
            if t.result:
                print(f"       → {t.result[:80]}")
            if t.error:
                print(f"       ✗ {t.error[:80]}")
            if t.duration:
                print(f"       ⏱  {t.duration}s")


# ─────────────────────────────────────────────
# PLANNER
# Turns a goal into a structured Plan using LLM
# ─────────────────────────────────────────────

PLANNER_PROMPT = """You are a task planner. Break down the user's goal into clear sequential steps.

Available tools:
{tools}

Return ONLY a JSON object — no markdown, no explanation:
{{
  "strategy": "one sentence describing your overall approach",
  "steps": [
    {{
      "id": 1,
      "description": "what this step does",
      "tool": "tool_name or null if pure reasoning",
      "args": {{"param": "value or {{{{step_N_result}}}} to reference prior step"}},
      "depends_on": []
    }},
    {{
      "id": 2,
      "description": "...",
      "tool": "...",
      "args": {{}},
      "depends_on": [1]
    }}
  ]
}}

PLANNING RULES:
- Use {{{{step_N_result}}}} in args to reference the output of step N
- Only add depends_on when the step genuinely needs a prior result
- Use null for tool when the step is reasoning/summarising without a tool call
- Keep steps atomic — one tool call per step
- Maximum 8 steps — combine trivial steps
"""

def create_plan(goal: str, registry, verbose: bool = True) -> Plan:
    """Ask LLM to decompose goal into a Plan."""

    tools_text = registry.prompt_block()
    prompt = PLANNER_PROMPT.format(tools=tools_text)

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user",   "content": f"Goal: {goal}"},
    ]

    if verbose:
        print("\n🧠 Planning...")

    raw = call_llm(messages)

    logger.info("RAW Text %s", raw)

    cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip()).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to extract JSON object from messy output
        m = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
            except Exception:
                data = {"strategy": "fallback", "steps": []}
        else:
            data = {"strategy": "fallback", "steps": []}

    plan = Plan(goal=goal, strategy=data.get("strategy", ""))

    logger.info("plan Text %s", plan)

    for s in data.get("steps", []):
        plan.add(SubTask(
            id            = s.get("id", len(plan.subtasks) + 1),
            description   = s.get("description", ""),
            tool          = s.get("tool"),
            args_template = s.get("args", {}),
            depends_on    = s.get("depends_on", []),
        ))

    if verbose:
        plan.print()

    return plan


# ─────────────────────────────────────────────
# ARG RESOLVER
# Replaces {{step_N_result}} placeholders
# with actual results from completed steps
# ─────────────────────────────────────────────

def resolve_args(args_template: dict, results: dict) -> dict:
    """
    Replace {{step_N_result}} placeholders with actual values.
    Works on nested dicts and string values.
    """
    def resolve_value(val):
        if not isinstance(val, str):
            return val
        # Find all {{step_N_result}} patterns
        placeholders = re.findall(r'\{\{step_(\d+)_result\}\}', val)
        for step_id in placeholders:
            step_id_int = int(step_id)
            replacement = results.get(step_id_int, f"[step {step_id} result unavailable]")
            val = val.replace(f"{{{{step_{step_id}_result}}}}", str(replacement))
        return val

    resolved = {}
    for key, value in args_template.items():
        if isinstance(value, dict):
            resolved[key] = resolve_args(value, results)
        elif isinstance(value, list):
            resolved[key] = [resolve_value(v) for v in value]
        else:
            resolved[key] = resolve_value(value)
    return resolved


# ─────────────────────────────────────────────
# LLM REASONING STEP
# For steps where tool=null — pure LLM reasoning
# using all results so far as context
# ─────────────────────────────────────────────

def run_llm_step(subtask: SubTask, results: dict) -> str:
    context = "\n".join(
        f"Step {sid} result: {res}"
        for sid, res in results.items()
    )
    messages = [
        {"role": "system", "content":
            "You are a helpful assistant completing one step of a larger task. "
            "Use the provided step results as context. Be concise."},
        {"role": "user", "content":
            f"Previous results:\n{context}\n\n"
            f"Your task: {subtask.description}\n\n"
            f"Provide a concise result for this step only."}
    ]
    return call_llm(messages)


# ─────────────────────────────────────────────
# EXECUTOR
# Runs a Plan step by step, resolving dependencies
# and handling partial failures
# ─────────────────────────────────────────────

class PlanExecutor:
    def __init__(self, registry, verbose: bool = True):
        self.registry = registry
        self.verbose  = verbose

    def execute(self, plan: Plan) -> Plan:
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"⚙️  Executing plan: '{plan.goal[:50]}'")
            print('='*60)

        max_iterations = len(plan.subtasks) * 2  # safety cap
        iterations     = 0

        while not plan.is_complete() and iterations < max_iterations:
            iterations += 1
            ready = plan.ready_tasks()

            if not ready:
                # Check for deadlock — tasks pending but none ready
                pending = [t for t in plan.subtasks if t.status == "pending"]
                if pending:
                    if self.verbose:
                        print(f"\n⚠️  Deadlock: {len(pending)} tasks pending "
                              f"but none ready. Forcing first pending task.")
                    pending[0].depends_on = []  # break deadlock
                continue

            # Execute ready tasks (could parallelise here in future)
            for task in ready:
                self._run_task(task, plan.results_so_far())

        if self.verbose:
            plan.print()

        return plan

    def _run_task(self, task: SubTask, results: dict):
        task.status     = "running"
        task.started_at = time.time()

        if self.verbose:
            print(f"\n┌ Step {task.id}: {task.description}")

        try:
            if task.tool is None:
                # Pure LLM reasoning step
                result = run_llm_step(task, results)
                task.result  = result
                task.status  = "done"
                task.ended_at = time.time()
                if self.verbose:
                    print(f"│ 🧠 [llm] {result[:100]}")
                    print(f"└ ✅ done ({task.duration}s)")

            else:
                # Tool step — resolve args then run
                resolved_args = resolve_args(task.args_template, results)

                if self.verbose:
                    print(f"│ 🔧 {task.tool}({json.dumps(resolved_args)[:80]})")

                observation, success = self.registry.run(
                    task.tool, resolved_args
                )
                task.ended_at = time.time()

                if success:
                    task.result = observation
                    task.status = "done"
                    if self.verbose:
                        print(f"│ 👁️  {observation[:100]}")
                        print(f"└ ✅ done ({task.duration}s)")
                else:
                    task.error  = observation
                    task.status = "failed"
                    if self.verbose:
                        print(f"│ ❌ {observation[:100]}")
                        print(f"└ failed — continuing with remaining steps")

        except Exception as e:
            task.error    = str(e)
            task.status   = "failed"
            task.ended_at = time.time()
            if self.verbose:
                print(f"└ ❌ Exception: {e}")


# ─────────────────────────────────────────────
# SYNTHESIZER
# Takes the completed plan and produces
# a coherent final answer from all results
# ─────────────────────────────────────────────

def synthesize_answer(plan: Plan) -> str:
    """Ask LLM to produce a final answer from all step results."""

    steps_summary = "\n".join(
        f"Step {t.id} ({t.description}): "
        + (t.result[:200] if t.status == "done" else f"FAILED: {t.error[:100]}")
        for t in plan.subtasks
    )

    messages = [
        {"role": "system", "content":
            "You are summarising the results of a completed multi-step task. "
            "Be concise. State what was accomplished and the key results. "
            "If any steps failed, mention it briefly."},
        {"role": "user", "content":
            f"Original goal: {plan.goal}\n\n"
            f"Step results:\n{steps_summary}\n\n"
            f"Provide a final answer summarising what was accomplished."}
    ]

    return call_llm(messages)


# ─────────────────────────────────────────────
# PLAN-AND-EXECUTE AGENT
# Full pipeline: plan → review → execute → synthesize
# ─────────────────────────────────────────────

def run_plan_and_execute(
    goal:           str,
    registry,
    require_approval: bool = False,
    verbose:          bool = True,
) -> tuple[str, Plan]:
    """
    Full Plan-and-Execute pipeline.

    require_approval: if True, shows plan and waits for user
                      confirmation before executing.
    """

    # ① PLAN
    plan = create_plan(goal, registry, verbose)

    if not plan.subtasks:
        return "Planner produced no steps.", plan

    # ② OPTIONAL HUMAN REVIEW
    if require_approval:
        print(f"\n{'─'*60}")
        print("👤 Review the plan above.")
        print("   Press Enter to execute, or type 'edit N description' "
              "to change step N, or 'skip N' to skip a step.")
        print("   Type 'abort' to cancel.")
        print(f"{'─'*60}")

        while True:
            cmd = input("Command (Enter=run): ").strip().lower()

            if not cmd:
                break

            if cmd == "abort":
                return "Plan aborted by user.", plan

            if cmd.startswith("skip "):
                try:
                    n = int(cmd.split()[1])
                    task = plan.get(n)
                    if task:
                        task.status = "skipped"
                        print(f"  ⏭️  Step {n} marked as skipped")
                except (IndexError, ValueError):
                    print("  Usage: skip <step_number>")

            elif cmd.startswith("edit "):
                parts = cmd.split(None, 2)
                if len(parts) == 3:
                    try:
                        n = int(parts[1])
                        task = plan.get(n)
                        if task:
                            task.description = parts[2]
                            print(f"  ✏️  Step {n} updated")
                    except ValueError:
                        print("  Usage: edit <step_number> <new description>")

            plan.print()

    # ③ EXECUTE
    executor = PlanExecutor(registry, verbose)
    plan     = executor.execute(plan)

    # ④ SYNTHESIZE
    if verbose:
        print(f"\n{'─'*60}")
        print("📝 Synthesising final answer...")

    answer = synthesize_answer(plan)

    if verbose:
        print(f"\n✅ Final Answer:\n{answer}")

    return answer, plan


# ─────────────────────────────────────────────
# TOOLS (same as 4.1)
# ─────────────────────────────────────────────

# (paste Tool, ToolRegistry classes from agent_10_react.py,
#  or import them — shown inline here for self-containment)

class Tool:
    def __init__(self, name, description, params, fn, category="general"):
        self.name        = name
        self.description = description
        self.params      = params
        self.required    = params.get("required", [])
        self.fn          = fn
        self.category    = category

    def validate(self, args):
        for f in self.required:
            if f not in args:
                return False, f"Missing required: '{f}'"
        return True, ""

    def run(self, args):
        valid, err = self.validate(args)
        if not valid:
            return err, False
        try:
            return str(self.fn(**args)), True
        except Exception as e:
            return f"Error: {e}", False

    def schema_text(self):
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
            return f"Unknown tool '{name}'. Available: {list(self.tools.keys())}", False
        return self.tools[name].run(args)

    def prompt_block(self) -> str:
        return "\n\n".join(t.schema_text() for t in self.tools.values())


def build_tools() -> ToolRegistry:
    r = ToolRegistry()

    r.add(Tool("calculator", "Evaluate math expressions",
        {"properties": {"expression": {"type": "string",
            "description": "Python math expression"}},
         "required": ["expression"]},
        lambda expression: str(eval(
            expression, {"__builtins__": {}},
            {k: v for k, v in math.__dict__.items()
             if not k.startswith("_")})),
        "math"))

    r.add(Tool("unit_convert", "Convert between units: km/miles, kg/lbs, gb/mb, c/f",
        {"properties": {
            "value":     {"type": "number", "description": "Value to convert"},
            "from_unit": {"type": "string",  "description": "Source unit"},
            "to_unit":   {"type": "string",  "description": "Target unit"}},
         "required": ["value", "from_unit", "to_unit"]},
        lambda value, from_unit, to_unit: _unit_convert(
            float(value), from_unit, to_unit),
        "math"))

    r.add(Tool("file_write", "Write content to a file",
        {"properties": {
            "filename": {"type": "string", "description": "filename"},
            "content":  {"type": "string", "description": "text to write"},
            "mode":     {"type": "string", "description": "write or append",
                         "enum": ["write", "append"]}},
         "required": ["filename", "content"]},
        lambda filename, content, mode="write":
            _file_write(filename, content, mode),
        "file"))

    r.add(Tool("file_read", "Read a file's contents",
        {"properties": {
            "filename": {"type": "string", "description": "file to read"}},
         "required": ["filename"]},
        lambda filename: _file_read(filename), "file"))

    r.add(Tool("file_list", "List files in current directory",
        {"properties": {
            "extension": {"type": "string",
                          "description": "filter e.g. .py or .txt"}},
         "required": []},
        lambda extension="": _file_list(extension), "file"))

    r.add(Tool("text_analyze", "Analyze text statistics",
        {"properties": {
            "text":      {"type": "string", "description": "text to analyze"},
            "operation": {"type": "string",
                          "description": "word_count|char_count|sentence_count|all",
                          "enum": ["word_count","char_count",
                                   "sentence_count","all"]}},
         "required": ["text", "operation"]},
        lambda text, operation: _text_analyze(text, operation), "text"))

    r.add(Tool("text_transform", "Transform text: case, slug, reverse",
        {"properties": {
            "text":      {"type": "string", "description": "text to transform"},
            "operation": {"type": "string",
                          "description":
                            "uppercase|lowercase|title_case|slug|reverse",
                          "enum": ["uppercase","lowercase",
                                   "title_case","slug","reverse"]}},
         "required": ["text", "operation"]},
        lambda text, operation: _text_transform(text, operation), "text"))

    r.add(Tool("datetime", "Get current date/time",
        {"properties": {
            "info_type": {"type": "string",
                          "description": "full|date|time|day|timestamp",
                          "enum": ["full","date","time","day","timestamp"]}},
         "required": ["info_type"]},
        lambda info_type: _datetime(info_type), "util"))

    return r


def _unit_convert(value, from_unit, to_unit):
    table = {
        ("km","miles"):0.621371, ("miles","km"):1.60934,
        ("kg","lbs"):2.20462,   ("lbs","kg"):0.453592,
        ("gb","mb"):1024,       ("mb","gb"):1/1024,
    }
    f, t = from_unit.lower(), to_unit.lower()
    if (f,t) == ("c","f"): return f"{round((value*9/5)+32,2)} f"
    if (f,t) == ("f","c"): return f"{round((value-32)*5/9,2)} c"
    if (f,t) in table:     return f"{round(value*table[(f,t)],4)} {to_unit}"
    return f"Unsupported: {from_unit} → {to_unit}"

def _file_write(filename, content, mode="write"):
    safe = os.path.basename(filename)
    with open(safe, "a" if mode == "append" else "w") as f:
        f.write(content)
    return f"Wrote {len(content)} chars to '{safe}'"

def _file_read(filename):
    safe = os.path.basename(filename)
    if not os.path.exists(safe): return f"File '{safe}' not found"
    return open(safe).read()

def _file_list(extension=""):
    files = [f for f in os.listdir(".") if f.endswith(extension)]
    return "\n".join(sorted(files)) if files else "No files found"

def _text_analyze(text, operation):
    words = text.strip().split()
    sents = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    return {
        "word_count":     str(len(words)),
        "char_count":     str(len(text)),
        "sentence_count": str(len(sents)),
        "all": f"words={len(words)} chars={len(text)} sentences={len(sents)}",
    }.get(operation, f"Unknown: {operation}")

def _text_transform(text, operation):
    return {
        "uppercase":  text.upper(),
        "lowercase":  text.lower(),
        "title_case": text.title(),
        "slug":  re.sub(r'[^a-z0-9]+','-',text.lower()).strip('-'),
        "reverse":    text[::-1],
    }.get(operation, f"Unknown: {operation}")

def _datetime(info_type):
    now = datetime.now()
    return {
        "full":      now.strftime("%Y-%m-%d %H:%M:%S, %A"),
        "date":      now.strftime("%Y-%m-%d"),
        "time":      now.strftime("%H:%M:%S"),
        "day":       now.strftime("%A"),
        "timestamp": str(int(now.timestamp())),
    }.get(info_type, str(now))


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    print("🤖 Phase 4, Step 4.2 — Task Decomposition\n")

    registry = build_tools()

    # ── Auto mode: run all goals automatically
    if "--auto" in sys.argv:
        goals = [
            # 1. Linear pipeline — each step feeds the next
            "Get the current date. Calculate how many seconds "
            "are in the current week number multiplied by 7. "
            "Write the result to 'week_seconds.txt'.",

            # 2. Branching — two independent calculations merged
            "I have a server with 64GB RAM and a 2TB disk. "
            "Convert both to MB. Then calculate the RAM-to-disk "
            "ratio. Write a summary to 'server_specs.txt'.",

            # 3. Text pipeline with dependency chain
            "Take this text: 'Agentic AI systems are transforming "
            "how we build software on modern infrastructure.' "
            "Get all text stats. Make it a URL slug. "
            "Write both outputs to 'text_pipeline.txt'.",

            # 4. Multi-domain — math + file + text in one plan
            "Calculate the area of a circle with radius 12 (use pi=3.14159). "
            "Convert the result from a number to title case text. "
            "Write 'Area report: <result>' to 'area_report.txt'. "
            "Then read the file back and count its words.",
        ]
        for goal in goals:
            answer, plan = run_plan_and_execute(
                goal, registry,
                require_approval=False,
                verbose=True,
            )
            print(f"\n{'='*60}\n")

    # ── Interactive mode: human reviews plan before execution
    else:
        print("Interactive mode — you review the plan before it runs.")
        print("Try: 'Build a report about today: get date, "
              "count words in a sample text, write both to report.txt'\n")

        goal = input("Enter your goal: ").strip()
        if goal:
            run_plan_and_execute(
                goal, registry,
                require_approval=True,
                verbose=True,
            )