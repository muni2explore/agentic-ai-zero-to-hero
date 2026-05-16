import requests
import json
import time
import os
from dotenv import load_dotenv

load_dotenv()


OLLAMA_URL = os.environ.get('OLLAMA_URL')
MODEL = os.environ.get('MODEL') 


def chat(messages: list[dict], temperature: float = 0.7) -> str:
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature  # 0 = deterministic, 1 = creative
        }
    }
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    return response.json()["message"]["content"]

def run_demo(title: str, messages: list[dict], temperature: float = 0.7):
    print(f"\n{'='*60}")
    print(f"🧪 {title}")
    print('='*60)
    for m in messages:
        if m["role"] != "system":
            print(f"\n[{m['role'].upper()}]\n{m['content']}")
    print(f"\n[ASSISTANT]")
    result = chat(messages, temperature)
    print(result)
    return result

# ─────────────────────────────────────────────
# TECHNIQUE 1: Chain-of-Thought (CoT)
# Force the LLM to reason before answering
# ─────────────────────────────────────────────

def demo_chain_of_thought():
    # Without CoT — LLM just guesses
    no_cot = [
        {"role": "system", "content": "Answer the user's question."},
        {"role": "user", "content": "A bat and ball together cost $1.10. The bat costs $1 more than the ball. How much does the ball cost?"}
    ]
    run_demo("WITHOUT Chain-of-Thought (likely wrong!)", no_cot, temperature=0)

    # With CoT — LLM reasons step by step
    with_cot = [
        {
            "role": "system",
            "content": (
                "You are a careful reasoning assistant. "
                "ALWAYS follow this format:\n\n"
                "THINKING:\n"
                "<reason through the problem step by step here>\n\n"
                "ANSWER:\n"
                "<your final answer here>"
            )
        },
        {"role": "user", "content": "A bat and ball together cost $1.10. The bat costs $1 more than the ball. How much does the ball cost?"}
    ]
    run_demo("WITH Chain-of-Thought (correct reasoning)", with_cot, temperature=0)

# ─────────────────────────────────────────────
# TECHNIQUE 2: Role + Constraint Prompting
# Define who the agent is and what it must/must not do
# ─────────────────────────────────────────────

def demo_role_constraint():
    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior Linux systems engineer specializing in Proxmox and containerization.\n\n"
                "RULES:\n"
                "- Only answer questions related to Linux, Proxmox, Docker, or networking\n"
                "- If asked something outside your domain, say: 'That is outside my expertise as a systems engineer'\n"
                "- Always mention potential risks before giving a destructive command\n"
                "- Format commands in code blocks\n"
                "- Be concise — no unnecessary filler"
            )
        },
        {"role": "user", "content": "How do I delete all stopped Docker containers?"}
    ]
    run_demo("Role + Constraint: Linux Engineer Agent", messages)

    # Now test the constraint
    messages2 = [
        messages[0],  # same system prompt
        {"role": "user", "content": "What is the best recipe for biryani?"}
    ]
    run_demo("Role + Constraint: Out-of-domain question", messages2)


# ─────────────────────────────────────────────
# TECHNIQUE 3: Structured Output (JSON)
# Make the LLM return parseable data — critical for agents
# ─────────────────────────────────────────────

def demo_structured_output():
    messages = [
        {
            "role": "system",
            "content": (
                "You are a task analysis assistant.\n"
                "When given a task description, extract structured information.\n\n"
                "IMPORTANT: Respond ONLY with a valid JSON object. No explanation, no markdown, no code fences.\n\n"
                "JSON format:\n"
                "{\n"
                '  "task_name": "short name",\n'
                '  "priority": "high|medium|low",\n'
                '  "estimated_hours": <number>,\n'
                '  "skills_required": ["skill1", "skill2"],\n'
                '  "subtasks": ["subtask1", "subtask2", "subtask3"]\n'
                "}"
            )
        },
        {
            "role": "user",
            "content": "Build a REST API for an employee payroll system with EPF, ESI, and TDS calculations using Laravel"
        }
    ]

    print(f"\n{'='*60}")
    print("🧪 Structured Output: JSON extraction")
    print('='*60)
    raw = chat(messages, temperature=0)
    print("\n[RAW LLM OUTPUT]")
    print(raw)

    # Try to parse it — this is what your agent code will do
    print("\n[PARSED AS PYTHON DICT]")
    try:
        # Strip markdown fences if the model added them despite instructions
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(cleaned)
        for key, val in parsed.items():
            print(f"  {key}: {val}")
        print("\n✅ Successfully parsed JSON")
    except json.JSONDecodeError as e:
        print(f"  ❌ Parse failed: {e}")
        print("  → This happens — agents need error handling for bad JSON")


# ─────────────────────────────────────────────
# TECHNIQUE 4: Few-Shot Prompting
# Teach the LLM by example — faster than long instructions
# ─────────────────────────────────────────────

def demo_few_shot():
    messages = [
        {
            "role": "system",
            "content": "You classify user intent for a smart home assistant. Respond with only the intent label."
        },
        # Few-shot examples — these are "training" examples in the prompt
        {"role": "user",    "content": "Turn off the lights in the bedroom"},
        {"role": "assistant","content": "CONTROL_LIGHTS"},
        {"role": "user",    "content": "What is the temperature outside?"},
        {"role": "assistant","content": "QUERY_WEATHER"},
        {"role": "user",    "content": "Play some jazz music"},
        {"role": "assistant","content": "PLAY_MUSIC"},
        {"role": "user",    "content": "Set an alarm for 7am tomorrow"},
        {"role": "assistant","content": "SET_ALARM"},
        # Now the real question
        {"role": "user",    "content": "Dim the living room lights to 40%"}
    ]
    run_demo("Few-Shot: Intent Classification", messages, temperature=0)

    # Test with something ambiguous
    messages2 = messages[:-1] + [
        {"role": "user", "content": "I am feeling cold"}
    ]
    run_demo("Few-Shot: Ambiguous input", messages2, temperature=0)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("🧠 Phase 1, Step 1.2 — Prompt Engineering for Agents")
    print("Running 4 techniques...\n")

    print("\n📌 TECHNIQUE 1: Chain-of-Thought")
    demo_chain_of_thought()

    print("\n\n📌 TECHNIQUE 2: Role + Constraint")
    demo_role_constraint()

    print("\n\n📌 TECHNIQUE 3: Structured Output (JSON)")
    demo_structured_output()

    print("\n\n📌 TECHNIQUE 4: Few-Shot Prompting")
    demo_few_shot()

    print("\n\n✅ All techniques done!")
    print("Key takeaway: The system prompt IS your agent's brain configuration.")