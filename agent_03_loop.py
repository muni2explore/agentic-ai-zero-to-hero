import math
import os
import requests
import re
import math
from datetime import datetime
import time
from dotenv import load_dotenv

load_dotenv()


OLLAMA_URL = os.environ.get('OLLAMA_URL')
MODEL = os.environ.get('MODEL') 

# ─────────────────────────────────────────────
# TOOLS
# These are the "hands" of the agent.
# Each tool takes a string input, returns a string output.
# ─────────────────────────────────────────────

def tool_calculator(expression: str) -> str:
    """Evaluate a math expression safely."""
    try:
        # Allow only safe math operations
        allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        allowed.update({"abs": abs, "round": round})
        result = eval(expression.strip(), {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def tool_get_time(query: str) -> str:
    """Return current date and time info."""
    now = datetime.now()
    return (
        f"Current datetime: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Day of week: {now.strftime('%A')}\n"
        f"Day of year: {now.timetuple().tm_yday}\n"
        f"Week number: {now.isocalendar()[1]}"
    )


def tool_knowledge(query: str) -> str:
    """A small hardcoded knowledge base — simulates a real search tool."""
    kb = {
        "proxmox": "Proxmox VE is an open-source virtualization platform based on Debian Linux. It supports KVM VMs and LXC containers.",
        "ollama": "Ollama is a tool for running large language models locally. It exposes a REST API at port 11434.",
        "react pattern": "ReAct (Reason+Act) is an agent pattern where the LLM alternates between Thought, Action, and Observation steps until reaching a Final Answer.",
        "llm": "A Large Language Model (LLM) is a neural network trained on large text datasets to predict and generate human-like text.",
        "agent": "An AI agent is a system that perceives its environment, reasons about it, takes actions, and observes the results in a loop.",
        "python": "Python is a high-level programming language known for readability and a large ecosystem of libraries.",
        "docker": "Docker is a containerization platform that packages applications and their dependencies into portable containers.",
    }
    query_lower = query.lower()
    for key, value in kb.items():
        if key in query_lower:
            return value
    return f"No knowledge found for: '{query}'. Try different keywords."


# Tool registry — maps tool name → function
TOOLS = {
    "calculator": tool_calculator,
    "get_time": tool_get_time,
    "search_knowledge": tool_knowledge,
}

# ─────────────────────────────────────────────
# SYSTEM PROMPT — The ReAct format
# This teaches the LLM HOW to think and act
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a reasoning agent. You solve problems step by step using tools.

You have access to these tools:
- calculator(expression): Evaluates a math expression. Example: calculator(24 * 60)
- get_time(query): Returns current date and time information.
- search_knowledge(query): Searches a knowledge base for information.

STRICT FORMAT — you must follow this exactly:

Thought: <your reasoning about what to do next>
Action: <tool_name>(<input>)

OR if you have the final answer:

Thought: <your final reasoning>
Final Answer: <your complete answer to the user>

RULES:
- Always start with a Thought
- Only call ONE tool per response
- Wait for the Observation before your next Thought
- Never make up tool results — always use the actual Observation
- When you have enough information, give the Final Answer
"""

# ─────────────────────────────────────────────
# LLM CALL
# ─────────────────────────────────────────────

def call_llm(messages: list[dict]) -> str:
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0}  # deterministic for agents
    }
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    return response.json()["message"]["content"]


# ─────────────────────────────────────────────
# PARSER
# Extract Thought, Action, Final Answer from LLM output
# ─────────────────────────────────────────────

def parse_llm_output(text: str) -> dict:
    """
    Parse LLM output into structured parts.
    Returns dict with keys: thought, action, action_input, final_answer
    """
    result = {
        "thought": None,
        "action": None,
        "action_input": None,
        "final_answer": None,
        "raw": text
    }

    # Extract Thought
    thought_match = re.search(r"Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|$)", text, re.DOTALL)
    if thought_match:
        result["thought"] = thought_match.group(1).strip()

    # Extract Final Answer
    final_match = re.search(r"Final Answer:\s*(.+?)$", text, re.DOTALL)
    if final_match:
        result["final_answer"] = final_match.group(1).strip()
        return result  # No action needed if final answer exists

    # Extract Action — matches: tool_name(input)
    action_match = re.search(r"Action:\s*(\w+)\((.+?)\)", text, re.DOTALL)
    if action_match:
        result["action"] = action_match.group(1).strip()
        result["action_input"] = action_match.group(2).strip()

    return result


# ─────────────────────────────────────────────
# TOOL EXECUTOR
# ─────────────────────────────────────────────

def execute_tool(action: str, action_input: str) -> str:
    """Look up and run the tool. Return observation string."""
    if action not in TOOLS:
        return f"Error: Unknown tool '{action}'. Available tools: {list(TOOLS.keys())}"
    try:
        return TOOLS[action](action_input)
    except Exception as e:
        return f"Tool error: {e}"


# ─────────────────────────────────────────────
# THE AGENT LOOP
# This is the core of everything
# ─────────────────────────────────────────────

def run_agent(user_goal: str, max_steps: int = 8) -> str:
    print(f"\n{'='*60}")
    print(f"🎯 Goal: {user_goal}")
    print('='*60)

    # Start conversation with system prompt + user goal
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_goal}
    ]

    for step in range(1, max_steps + 1):
        print(f"\n--- Step {step} ---")

        # ① THINK: ask LLM what to do next
        llm_output = call_llm(messages)
        print(f"\n🧠 LLM Output:\n{llm_output}")

        # ② PARSE: extract structured info from LLM response
        parsed = parse_llm_output(llm_output)

        if parsed["thought"]:
            print(f"\n💭 Thought: {parsed['thought']}")

        # ③ CHECK: did LLM reach a final answer?
        if parsed["final_answer"]:
            print(f"\n✅ Final Answer: {parsed['final_answer']}")
            return parsed["final_answer"]

        # ④ ACT: if no final answer, execute the tool
        if not parsed["action"]:
            print("\n⚠️  No action or final answer found. Asking LLM to clarify...")
            messages.append({"role": "assistant", "content": llm_output})
            messages.append({
                "role": "user",
                "content": "Please continue. Either use a tool with Action: tool_name(input) or provide a Final Answer: ..."
            })
            continue

        print(f"\n🔧 Action: {parsed['action']}({parsed['action_input']})")

        # ⑤ OBSERVE: run the tool and get the result
        observation = execute_tool(parsed["action"], parsed["action_input"])
        print(f"👁️  Observation: {observation}")

        # ⑥ FEED BACK: add LLM output + observation to message history
        # This is how the agent "sees" what happened
        messages.append({"role": "assistant", "content": llm_output})
        messages.append({
            "role": "user",
            "content": f"Observation: {observation}"
        })

    return "Agent stopped: max steps reached without a final answer."


# ─────────────────────────────────────────────
# MAIN — Run 4 test goals
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("🤖 Phase 1.3 — The Agent Loop (ReAct from scratch)\n")

    # Goal 1: Single tool use
    run_agent("What is 347 multiplied by 28?")

    # Goal 2: Multi-step reasoning (must use calculator twice)
    run_agent("What is 25% of the number of minutes in a day?")

    # Goal 3: Knowledge + math combined
    run_agent("What is Ollama and on which port does it run? Also tell me what 11434 divided by 2 is.")

    # Goal 4: Time awareness
    run_agent("What day of the week is it today, and what is the square root of today's day of the year?")