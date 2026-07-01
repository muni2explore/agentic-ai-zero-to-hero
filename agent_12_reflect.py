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
DEFAULT_LOG_FILE = os.path.join(LOG_DIR, "agent_12.log")

def setup_logging(log_file: str = DEFAULT_LOG_FILE) -> logging.Logger:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("agent_12_reflect")
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
# ATTEMPT — one attempt at a task
# ─────────────────────────────────────────────

@dataclass
class Attempt:
    number:     int
    output:     str           # what the agent produced
    score:      float = 0.0   # 0–10 quality score
    critique:   str   = ""    # what's wrong / missing
    strengths:  str   = ""    # what's good
    passed:     bool  = False # did it meet the threshold?
    duration:   float = 0.0

    def print(self):
        icon = "✅" if self.passed else "🔄"
        print(f"\n  {icon} Attempt {self.number}  "
              f"[score={self.score}/10 | {self.duration}s]")
        print(f"     Output:    {self.output[:120]}...")
        if self.strengths:
            print(f"     ✓ Good:    {self.strengths[:100]}")
        if self.critique:
            print(f"     ✗ Issues:  {self.critique[:100]}")


@dataclass
class ReflexionTrace:
    goal:        str
    attempts:    list[Attempt] = field(default_factory=list)
    final:       str           = ""
    total_time:  float         = 0.0

    def print_summary(self):
        print(f"\n{'═'*60}")
        print(f"📊 Reflexion Summary — '{self.goal[:50]}'")
        print(f"{'═'*60}")
        print(f"  Attempts:   {len(self.attempts)}")
        if self.attempts:
            scores = [a.score for a in self.attempts]
            print(f"  Scores:     {' → '.join(str(s) for s in scores)}")
            print(f"  Improvement:{round(scores[-1] - scores[0], 1):+.1f} points")
        print(f"  Total time: {round(self.total_time, 2)}s")
        print(f"\n  Final output:\n  {self.final[:300]}")


# ─────────────────────────────────────────────
# CRITIC
# Evaluates any output against a goal.
# Returns structured score + feedback.
# ─────────────────────────────────────────────

CRITIC_PROMPT = """You are a strict quality critic evaluating an AI agent's output.

Evaluate the output against the original goal using these criteria:
- Completeness: does it fully address all parts of the goal?
- Accuracy:     is the information correct and precise?
- Clarity:      is it well-structured and easy to understand?
- Relevance:    does it stay on topic without unnecessary filler?

Respond with ONLY valid JSON — no markdown:
{{
  "score": <0.0 to 10.0>,
  "passed": <true if score >= {threshold}>,
  "strengths": "what is done well (1-2 sentences)",
  "critique": "specific gaps or errors (1-3 sentences)",
  "suggestions": ["specific improvement 1", "specific improvement 2"]
}}

Be strict. 10 = perfect, 7 = good enough, 5 = mediocre, 3 = poor, 1 = wrong.
"""

def critique_output(
    goal:      str,
    output:    str,
    context:   str = "",
    threshold: float = 7.0,
) -> dict:
    """
    Ask LLM to critique output against goal.
    Returns parsed critique dict.
    """
    prompt = CRITIC_PROMPT.format(threshold=threshold)

    user_content = f"GOAL:\n{goal}\n\n"
    if context:
        user_content += f"CONTEXT:\n{context}\n\n"
    user_content += f"OUTPUT TO EVALUATE:\n{output}"

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user",   "content": user_content},
    ]

    raw = call_llm(messages)

    cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip()).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    # Fallback if parsing fails
    return {
        "score": 5.0, "passed": False,
        "strengths": "",
        "critique": "Could not parse critique. Treat as partial.",
        "suggestions": ["Try again with more detail"],
    }


# ─────────────────────────────────────────────
# IMPROVER
# Takes the previous output + critique and
# produces an improved version
# ─────────────────────────────────────────────

IMPROVER_PROMPT = """You are improving your previous output based on specific feedback.

Your task:
1. Read the original goal carefully
2. Read your previous attempt and the critique
3. Address every issue mentioned in the critique
4. Keep what was already good
5. Produce an improved output that directly addresses the goal

Be thorough. Do not repeat the same mistakes.
"""

def improve_output(
    goal:        str,
    previous:    str,
    critique:    str,
    suggestions: list[str],
    context:     str = "",
    attempt_num: int = 2,
) -> str:
    """Ask LLM to improve its previous output using the critique."""

    suggestions_text = "\n".join(f"  - {s}" for s in suggestions)

    user_content = (
        f"ORIGINAL GOAL:\n{goal}\n\n"
        f"YOUR PREVIOUS ATTEMPT (attempt {attempt_num - 1}):\n{previous}\n\n"
        f"CRITIQUE:\n{critique}\n\n"
        f"SUGGESTIONS TO IMPLEMENT:\n{suggestions_text}\n\n"
    )
    if context:
        user_content += f"ADDITIONAL CONTEXT:\n{context}\n\n"

    user_content += "Produce an improved version that addresses all critique points:"

    messages = [
        {"role": "system", "content": IMPROVER_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    return call_llm(messages)


# ─────────────────────────────────────────────
# REFLEXION LOOP
# The core self-improvement cycle
# ─────────────────────────────────────────────

def reflexion_loop(
    goal:          str,
    initial_fn:    Callable[[], str],   # function that produces first attempt
    context:       str   = "",          # extra context for critic
    threshold:     float = 7.5,         # score needed to pass
    max_attempts:  int   = 4,           # max retries
    verbose:       bool  = True,
) -> tuple[str, ReflexionTrace]:
    """
    Run the reflexion loop:
    1. Produce initial output via initial_fn()
    2. Critique it
    3. If passes threshold → done
    4. Else improve using critique → go to 2
    """
    trace      = ReflexionTrace(goal=goal)
    start_time = time.time()

    if verbose:
        print(f"\n{'═'*60}")
        print(f"🪞 Reflexion Loop: '{goal[:55]}'")
        print(f"   threshold={threshold}/10 | max_attempts={max_attempts}")
        print('═'*60)

    current_output = None

    for attempt_num in range(1, max_attempts + 1):
        t0 = time.time()

        if attempt_num == 1:
            if verbose:
                print(f"\n🖊️  Attempt {attempt_num}: generating initial output...")
            current_output = initial_fn()
        else:
            if verbose:
                print(f"\n🔄 Attempt {attempt_num}: improving based on critique...")
            last = trace.attempts[-1]
            current_output = improve_output(
                goal        = goal,
                previous    = last.output,
                critique    = last.critique,
                suggestions = last_critique.get("suggestions", []),
                context     = context,
                attempt_num = attempt_num,
            )

        duration = round(time.time() - t0, 2)

        # Critique the output
        if verbose:
            print(f"   🔍 Critiquing...")

        last_critique = critique_output(goal, current_output, context, threshold)

        score    = float(last_critique.get("score", 5.0))
        passed   = bool(last_critique.get("passed", False)) or score >= threshold
        critique = last_critique.get("critique", "")
        strength = last_critique.get("strengths", "")

        attempt = Attempt(
            number    = attempt_num,
            output    = current_output,
            score     = score,
            critique  = critique,
            strengths = strength,
            passed    = passed,
            duration  = duration,
        )
        trace.attempts.append(attempt)

        if verbose:
            attempt.print()

        if passed:
            if verbose:
                print(f"\n✅ Passed threshold ({score}/10 ≥ {threshold})")
            break

        if attempt_num < max_attempts:
            if verbose:
                print(f"   📝 Will improve: {critique[:80]}")
        else:
            if verbose:
                print(f"\n⚠️  Max attempts reached. Using best output.")
            # Use the highest-scoring attempt as final
            best = max(trace.attempts, key=lambda a: a.score)
            current_output = best.output

    trace.final      = current_output
    trace.total_time = round(time.time() - start_time, 2)

    if verbose:
        trace.print_summary()

    return current_output, trace


# ─────────────────────────────────────────────
# TOOL-USING REFLEXION
# Reflexion where the agent also has tools.
# The initial output comes from a ReAct-style run.
# The critic then evaluates BOTH the reasoning
# and the tool outputs.
# ─────────────────────────────────────────────

class Tool:
    def __init__(self, name, description, params, fn):
        self.name     = name
        self.description = description
        self.params   = params
        self.required = params.get("required", [])
        self.fn       = fn

    def run(self, args):
        for f in self.required:
            if f not in args:
                return f"Missing required: '{f}'", False
        try:
            return str(self.fn(**args)), True
        except Exception as e:
            return f"Error: {e}", False

    def schema_text(self):
        props = self.params.get("properties", {})
        lines = [f"{self.name}: {self.description}"]
        for p, info in props.items():
            req = "*" if p in self.required else "?"
            lines.append(f"  {p}{req}: {info.get('description','')}")
        return "\n".join(lines)


class ToolRegistry:
    def __init__(self):
        self.tools: dict[str, Tool] = {}

    def add(self, t: Tool):
        self.tools[t.name] = t

    def run(self, name, args):
        if name not in self.tools:
            return f"Unknown tool '{name}'", False
        return self.tools[name].run(args)

    def prompt_block(self):
        return "\n\n".join(t.schema_text() for t in self.tools.values())


def react_with_tools(
    goal:     str,
    registry: ToolRegistry,
    context:  str = "",
    max_steps: int = 8,
) -> str:
    """
    Run a mini ReAct loop and return the final answer string.
    Used as the initial_fn inside reflexion_loop.
    """
    system = f"""You are a reasoning agent with tools.

TOOLS:
{registry.prompt_block()}

FORMAT:
Thought: <reasoning>
Action: <tool_name>
Args: {{"param": "value"}}

OR when done:
Thought: <final reasoning>
Answer: <complete answer>

{"CONTEXT: " + context if context else ""}
"""
    messages   = [
        {"role": "system", "content": system},
        {"role": "user",   "content": f"Goal: {goal}"},
    ]
    last_answer = "No answer produced."

    for _ in range(max_steps):
        raw = call_llm(messages)

        # Answer?
        m = re.search(r"Answer:\s*(.+?)$", raw, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()

        # Action?
        action_m = re.search(r"Action:\s*(\w+)", raw, re.IGNORECASE)
        args_m   = re.search(r"Args:\s*(\{.*?\})", raw,
                              re.DOTALL | re.IGNORECASE)

        if action_m:
            action = action_m.group(1)
            args   = {}
            if args_m:
                try:
                    args = json.loads(args_m.group(1))
                except Exception:
                    pass
            obs, _ = registry.run(action, args)
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user",
                             "content": f"Observation: {obs}\nContinue."})
        else:
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user",
                             "content": "Provide Action+Args or Answer."})

    return last_answer


# ─────────────────────────────────────────────
# SELF-CORRECTING CODE GENERATOR
# A specialised reflexion loop for code.
# Critic checks: correctness, edge cases, style, docs.
# ─────────────────────────────────────────────

CODE_CRITIC_PROMPT = """You are a senior Python code reviewer.

Evaluate the code against the specification using:
- Correctness:   does it do what was asked?
- Edge cases:    does it handle None, empty input, errors?
- Code quality:  readable, well-named, not overly complex?
- Documentation: docstring present and accurate?
- Completeness:  no TODOs, no missing pieces?

Respond ONLY with valid JSON:
{{
  "score": <0.0 to 10.0>,
  "passed": <true if score >= {threshold}>,
  "strengths": "what is done well",
  "critique": "specific code issues",
  "suggestions": ["fix 1", "fix 2", "fix 3"]
}}
"""

CODE_GEN_PROMPT = """You are an expert Python developer.
Write clean, well-documented Python code.
Include:
- A docstring explaining what the function does
- Type hints on all parameters and return value
- Input validation and error handling
- At least 2 usage examples in the docstring
Return ONLY the Python code — no explanation, no markdown fences.
"""

def generate_code(spec: str) -> str:
    messages = [
        {"role": "system", "content": CODE_GEN_PROMPT},
        {"role": "user",   "content": f"Write Python code for:\n{spec}"},
    ]
    raw = call_llm(messages)
    # Strip markdown fences if present
    cleaned = re.sub(r'^```(?:python)?\s*|\s*```$', '', raw.strip()).strip()
    return cleaned


def improve_code(spec: str, previous: str,
                 critique: str, suggestions: list[str]) -> str:
    sug_text = "\n".join(f"  - {s}" for s in suggestions)
    messages = [
        {"role": "system",  "content": CODE_GEN_PROMPT},
        {"role": "user",    "content":
            f"SPECIFICATION:\n{spec}\n\n"
            f"PREVIOUS CODE:\n{previous}\n\n"
            f"CRITIQUE:\n{critique}\n\n"
            f"REQUIRED IMPROVEMENTS:\n{sug_text}\n\n"
            f"Write an improved version addressing all issues:"},
    ]
    raw = call_llm(messages)
    cleaned = re.sub(r'^```(?:python)?\s*|\s*```$', '', raw.strip()).strip()
    return cleaned


def code_reflexion(
    spec:         str,
    threshold:    float = 7.5,
    max_attempts: int   = 3,
    verbose:      bool  = True,
) -> tuple[str, ReflexionTrace]:
    """Reflexion loop specialised for code generation."""

    # Custom critic that uses the code-specific prompt
    def code_critique(goal, output, context, thresh):
        prompt = CODE_CRITIC_PROMPT.format(threshold=thresh)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user",   "content":
                f"SPECIFICATION:\n{goal}\n\nCODE TO REVIEW:\n{output}"},
        ]
        raw = call_llm(messages)
        cleaned = re.sub(r'^```(?:json)?\s*|\s*```$',
                         '', raw.strip()).strip()
        try:
            return json.loads(cleaned)
        except Exception:
            return {"score": 5.0, "passed": False,
                    "strengths": "", "critique": "Parse error",
                    "suggestions": ["Retry"]}

    trace      = ReflexionTrace(goal=spec)
    start_time = time.time()
    last_critique_data = {}

    if verbose:
        print(f"\n{'═'*60}")
        print(f"💻 Code Reflexion: '{spec[:55]}'")
        print(f"   threshold={threshold}/10 | max_attempts={max_attempts}")
        print('═'*60)

    current_code = None

    for attempt_num in range(1, max_attempts + 1):
        t0 = time.time()

        if attempt_num == 1:
            if verbose:
                print(f"\n🖊️  Attempt {attempt_num}: generating code...")
            current_code = generate_code(spec)
        else:
            if verbose:
                print(f"\n🔄 Attempt {attempt_num}: improving code...")
            last = trace.attempts[-1]
            current_code = improve_code(
                spec        = spec,
                previous    = last.output,
                critique    = last.critique,
                suggestions = last_critique_data.get("suggestions", []),
            )

        duration = round(time.time() - t0, 2)

        if verbose:
            print(f"   🔍 Code review in progress...")

        last_critique_data = code_critique(
            spec, current_code, "", threshold
        )

        score    = float(last_critique_data.get("score", 5.0))
        passed   = score >= threshold
        critique = last_critique_data.get("critique", "")
        strength = last_critique_data.get("strengths", "")

        attempt = Attempt(
            number    = attempt_num,
            output    = current_code,
            score     = score,
            critique  = critique,
            strengths = strength,
            passed    = passed,
            duration  = duration,
        )
        trace.attempts.append(attempt)

        if verbose:
            attempt.print()
            # Show code snippet
            lines = current_code.split('\n')
            preview = '\n'.join(
                f"    {i+1:>2}│ {l}" for i, l in enumerate(lines[:12])
            )
            if len(lines) > 12:
                preview += f"\n    ...({len(lines)} total lines)"
            print(f"\n   Code preview:\n{preview}")

        if passed:
            if verbose:
                print(f"\n✅ Code passed review ({score}/10)")
            break

        if attempt_num == max_attempts:
            if verbose:
                print(f"\n⚠️  Max attempts. Using best version.")
            best = max(trace.attempts, key=lambda a: a.score)
            current_code = best.output

    trace.final      = current_code
    trace.total_time = round(time.time() - start_time, 2)

    if verbose:
        trace.print_summary()

    return current_code, trace


# ─────────────────────────────────────────────
# BUILD TOOLS for tool-using demo
# ─────────────────────────────────────────────

def build_tools() -> ToolRegistry:
    r = ToolRegistry()

    r.add(Tool("calculator", "Evaluate math expressions",
        {"properties": {"expression": {"type":"string",
            "description":"Python math expression"}},
         "required": ["expression"]},
        lambda expression: str(eval(
            expression, {"__builtins__": {}},
            {k: v for k, v in math.__dict__.items()
             if not k.startswith("_")}))))

    r.add(Tool("file_write", "Write content to a file",
        {"properties": {
            "filename": {"type":"string","description":"filename"},
            "content":  {"type":"string","description":"text to write"}},
         "required": ["filename","content"]},
        lambda filename, content: _file_write(filename, content)))

    r.add(Tool("file_read", "Read a file's contents",
        {"properties": {
            "filename": {"type":"string","description":"file to read"}},
         "required": ["filename"]},
        lambda filename: _file_read(filename)))

    r.add(Tool("datetime", "Get current date/time",
        {"properties": {
            "info_type": {"type":"string",
                          "description":"full|date|time|day",
                          "enum":["full","date","time","day"]}},
         "required": ["info_type"]},
        lambda info_type: _datetime(info_type)))

    return r


def _file_write(filename, content):
    safe = os.path.basename(filename)
    with open(safe, "w") as f: f.write(content)
    return f"Wrote {len(content)} chars to '{safe}'"

def _file_read(filename):
    safe = os.path.basename(filename)
    if not os.path.exists(safe): return f"File '{safe}' not found"
    return open(safe).read()

def _datetime(info_type):
    now = datetime.now()
    return {
        "full":  now.strftime("%Y-%m-%d %H:%M:%S, %A"),
        "date":  now.strftime("%Y-%m-%d"),
        "time":  now.strftime("%H:%M:%S"),
        "day":   now.strftime("%A"),
    }.get(info_type, str(now))


# ─────────────────────────────────────────────
# MAIN — 4 demonstrations
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("🪞 Phase 4, Step 4.3 — Self-Reflection & Correction\n")

    # ── Demo 1: Pure text reflexion ──────────
    # Agent writes an explanation, critic scores it,
    # agent improves until threshold met
    print("\n" + "█"*60)
    print("DEMO 1 — Text quality reflexion")
    print("█"*60)

    reflexion_loop(
        goal = (
            "Explain the ReAct agent pattern to a developer who knows "
            "Python but has never built an AI agent. Include: what ReAct "
            "stands for, the Think-Act-Observe loop with a concrete example, "
            "why it's better than a simple chatbot, and one code snippet."
        ),
        initial_fn = lambda: call_llm([
            {"role": "system", "content": "You are a technical writer. Be thorough and precise."},
            {"role": "user", "content": (
                "Explain the ReAct agent pattern to a developer who knows Python "
                "but has never built an AI agent. Include: what ReAct stands for, "
                "the Think-Act-Observe loop with a concrete example, why it's "
                "better than a simple chatbot, and one code snippet."
            )}
        ]),
        threshold    = 7.5,
        max_attempts = 3,
    )

    # ── Demo 2: Tool-using reflexion ─────────
    # Agent uses tools to answer, critic checks
    # if the answer used tools correctly and is complete
    print("\n\n" + "█"*60)
    print("DEMO 2 — Tool-using reflexion")
    print("█"*60)

    registry = build_tools()

    reflexion_loop(
        goal = (
            "What is today's date? Calculate how many hours are in "
            "365 days. Then write a summary file 'reflexion_demo.txt' "
            "containing both answers with clear labels."
        ),
        initial_fn = lambda: react_with_tools(
            goal     = "What is today's date? Calculate how many hours "
                       "are in 365 days. Then write a summary file "
                       "'reflexion_demo.txt' containing both answers.",
            registry = registry,
        ),
        context = "The agent has tools: calculator, file_write, file_read, datetime.",
        threshold    = 7.0,
        max_attempts = 3,
    )

    # ── Demo 3: Code generation reflexion ────
    # Generates Python code, reviews it,
    # improves until senior-dev quality
    print("\n\n" + "█"*60)
    print("DEMO 3 — Code generation reflexion")
    print("█"*60)

    final_code, trace = code_reflexion(
        spec = (
            "A function called `parse_duration` that takes a string like "
            "'2h 30m', '45s', '1h', '3h 15m 20s' and returns the total "
            "number of seconds as an integer. Handle invalid input by "
            "raising ValueError with a helpful message."
        ),
        threshold    = 7.5,
        max_attempts = 3,
    )

    # Save the final generated code
    with open("generated_parse_duration.py", "w") as f:
        f.write(final_code)
    print(f"\n💾 Final code saved to 'generated_parse_duration.py'")

    # ── Demo 4: Self-consistency check ───────
    # Run same goal 3 times with temperature > 0,
    # critic picks the best answer
    print("\n\n" + "█"*60)
    print("DEMO 4 — Self-consistency (best of 3)")
    print("█"*60)

    goal = (
        "Explain in exactly 3 bullet points why vector databases "
        "like ChromaDB are essential for AI agent memory systems. "
        "Each bullet must be one sentence, concrete, and practical."
    )

    print(f"\n🎯 Goal: {goal}\n")
    print("Generating 3 independent answers and picking the best...\n")

    candidates = []
    for i in range(3):
        print(f"  Generating candidate {i+1}...")
        raw = call_llm([
                {"role": "system",
                 "content": "You are a concise technical writer."},
                {"role": "user", "content": goal}
            ])
        candidates.append(raw)

    # Score all 3
    print("\n  Scoring all candidates...\n")
    scored = []
    for i, candidate in enumerate(candidates):
        result = critique_output(goal, candidate, threshold=7.0)
        score  = float(result.get("score", 5.0))
        scored.append((score, candidate, result))
        print(f"  Candidate {i+1}: score={score}/10 — "
              f"{result.get('critique','')[:60]}")

    # Pick best
    best_score, best_answer, best_critique = max(scored, key=lambda x: x[0])
    print(f"\n  🏆 Best candidate: score={best_score}/10")
    print(f"\n  Answer:\n{best_answer}")
    print(f"\n  Strengths: {best_critique.get('strengths','')[:100]}")