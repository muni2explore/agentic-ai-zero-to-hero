"""
Phase 5.1 — LangChain Basics
Shows: LLM wrapper, prompt templates, chains, tools, agents.
Everything maps to what you built in Phases 1-4.
"""

import math
import os
import re
import json
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
import math

# ── LangChain imports ──────────────────────────────────────
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain.agents import create_agent

MODEL = os.environ.get('MODEL') 

load_dotenv()

# ─────────────────────────────────────────────
# SECTION 1: LLM WRAPPER
# Your Phase 1 call_llm() → ChatGroq
# ─────────────────────────────────────────────

# Helper to initialize the model
def get_llm(temperature=0):
    return ChatGroq(
        model=MODEL,
        temperature=temperature,
        # Ensure GROQ_API_KEY is set in your env
        api_key=os.environ.get("GROQ_API_KEY") 
    )

def demo_llm_wrapper():
    print("\n" + "═"*60)
    print("📌 SECTION 1: LLM Wrapper (ChatGroq)")
    print("═"*60)

    llm = get_llm(temperature=0)

    # Direct call — equivalent to your call_llm([{...}])
    print("\n1a. Direct invoke:")
    response = llm.invoke("What is the capital of Tamil Nadu? One sentence.")
    print(f"   Response: {response.content}")
    print(f"   Type: {type(response).__name__}")

    # With messages list — same as your messages[] pattern
    print("\n1b. With message list (same as your messages[]):")
    messages = [
        SystemMessage(content="You answer in exactly one sentence."),
        HumanMessage(content="What is Proxmox used for?"),
    ]
    response = llm.invoke(messages)
    print(f"   Response: {response.content}")

    # Streaming — useful for long outputs
    print("\n1c. Streaming:")
    print("   ", end="", flush=True)
    for chunk in llm.stream("Count from 1 to 5, one number per word."):
        print(chunk.content, end="", flush=True)
    print()


# ─────────────────────────────────────────────
# SECTION 2: PROMPT TEMPLATES
# Your f-string system prompts → ChatPromptTemplate
# ─────────────────────────────────────────────

def demo_prompt_templates():
    print("\n" + "═"*60)
    print("📌 SECTION 2: Prompt Templates")
    print("═"*60)

    llm = get_llm(temperature=0)


    # 2a. Simple template with variables
    print("\n2a. Template with variables:")
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a {role}. Answer in {style} style. "
         "Keep it under 2 sentences."),
        ("human", "{question}"),
    ])

    # LCEL chain: prompt | llm | parser
    # This is the pipe operator — chains steps together
    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({
        "role":     "Linux systems engineer",
        "style":    "concise technical",
        "question": "What is an LXC container?",
    })
    print(f"   {result}")

    # 2b. Few-shot via template
    print("\n2b. Few-shot template:")
    few_shot_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Classify the intent of a home automation command. "
         "Return ONLY the intent label."),
        ("human",  "Turn off bedroom lights"),
        ("ai",     "CONTROL_LIGHTS"),
        ("human",  "What is the temperature?"),
        ("ai",     "QUERY_SENSOR"),
        ("human",  "Play jazz music"),
        ("ai",     "PLAY_MEDIA"),
        ("human",  "{command}"),
    ])

    chain2 = few_shot_prompt | llm | StrOutputParser()
    tests  = [
        "Dim the kitchen lights to 50%",
        "Set an alarm for 6am",
        "Is the front door locked?",
    ]
    for cmd in tests:
        result = chain2.invoke({"command": cmd})
        print(f"   '{cmd}' → {result.strip()}")

    # 2c. Chain of thought template
    print("\n2c. Chain-of-thought template (your Phase 1.2 CoT):")
    cot_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Think step by step.\n\n"
         "Format:\n"
         "THINKING:\n<reason step by step>\n\n"
         "ANSWER:\n<final answer>"),
        ("human", "{problem}"),
    ])

    chain3  = cot_prompt | llm | StrOutputParser()
    problem = "A server has 128GB RAM. 40% is used by the OS. " \
              "How many GB are available for containers?"
    result  = chain3.invoke({"problem": problem})
    print(f"\n   Problem: {problem}")
    print(f"\n   {result}")


# ─────────────────────────────────────────────
# SECTION 3: CHAINS
# Your manual pipeline → LCEL chain
# ─────────────────────────────────────────────

def demo_chains():
    print("\n" + "═"*60)
    print("📌 SECTION 3: Chains (LCEL)")
    print("═"*60)

    llm = get_llm(temperature=0.3)

    parser = StrOutputParser()

    # 3a. Sequential chain — output of one feeds into the next
    print("\n3a. Sequential chain (summarise → translate style):")

    summarise_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarise the following in exactly 2 sentences."),
        ("human",  "{text}"),
    ])

    simplify_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Rewrite this for a 10-year-old. "
         "Use simple words. Max 2 sentences."),
        ("human", "{summary}"),
    ])

    # Chain: text → summarise → simplify
    summarise_chain = summarise_prompt | llm | parser
    simplify_chain  = simplify_prompt | llm | parser

    text = (
        "Proxmox Virtual Environment is an open-source server "
        "virtualisation management platform based on Debian Linux. "
        "It supports two types of virtualisation: KVM-based virtual "
        "machines for full OS virtualisation, and LXC containers for "
        "lightweight process-level isolation. Proxmox includes a "
        "web-based management interface, REST API, and clustering support."
    )

    summary    = summarise_chain.invoke({"text": text})
    simplified = simplify_chain.invoke({"summary": summary})

    print(f"\n   Original ({len(text)} chars)")
    print(f"   Summary:    {summary}")
    print(f"   Simplified: {simplified}")

    # 3b. Branching — classify then route
    print("\n3b. Branching chain (classify → route to specialist):")

    classify_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Classify this query into ONE category: "
         "math | code | general. "
         "Return ONLY the category word."),
        ("human", "{query}"),
    ])

    specialist_prompts = {
        "math": ChatPromptTemplate.from_messages([
            ("system", "You are a math tutor. "
                       "Solve step by step, show working."),
            ("human", "{query}"),
        ]),
        "code": ChatPromptTemplate.from_messages([
            ("system", "You are a senior Python developer. "
                       "Give clean, well-commented code."),
            ("human", "{query}"),
        ]),
        "general": ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Be concise."),
            ("human", "{query}"),
        ]),
    }

    def route_and_answer(query: str) -> str:
        category = (classify_prompt | llm | parser).invoke(
            {"query": query}
        ).strip().lower()
        category = category if category in specialist_prompts else "general"
        prompt   = specialist_prompts[category]
        answer   = (prompt | llm | parser).invoke({"query": query})
        return f"[{category.upper()}] {answer[:200]}"

    test_queries = [
        "What is 15% of 3200?",
        "Write a Python function to flatten a nested list",
        "What is Ollama?",
    ]
    for q in test_queries:
        result = route_and_answer(q)
        print(f"\n   Q: {q}")
        print(f"   A: {result[:150]}")

    # 3c. Memory chain — conversation with history
    print("\n3c. Memory chain (multi-turn with history):")

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. "
                   "Be concise — max 2 sentences per reply."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    chain_with_memory = chat_prompt | llm | parser

    history = []
    turns   = [
        "My name is Muni and I am learning agentic AI.",
        "I am using Ollama on my Proxmox server.",
        "What do you know about me so far?",
    ]

    for turn in turns:
        result = chain_with_memory.invoke({
            "history": history,
            "input":   turn,
        })
        print(f"\n   You: {turn}")
        print(f"   Bot: {result}")
        history.append(HumanMessage(content=turn))
        history.append(AIMessage(content=result))


# ─────────────────────────────────────────────
# SECTION 4: TOOLS + AGENT
# Your Phase 2 ToolRegistry → @tool decorator
# Your Phase 4 ReAct loop   → AgentExecutor
# ─────────────────────────────────────────────

# Define tools using @tool decorator
# Under the hood this is your JSON schema + function pattern

@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    Use for any arithmetic, algebra, or formula calculation.
    Example: calculator("2 ** 10") or calculator("math.sqrt(144)")
    """
    allowed = {k: v for k, v in math.__dict__.items()
               if not k.startswith("_")}
    allowed.update({"abs": abs, "round": round})
    try:
        return str(eval(expression.strip(),
                        {"__builtins__": {}}, allowed))
    except Exception as e:
        return f"Math error: {e}"


@tool
def get_datetime(info_type: str) -> str:
    """
    Get current date/time information.
    info_type options: full, date, time, day, timestamp
    Example: get_datetime("date") returns today's date.
    """
    now = datetime.now()
    return {
        "full":      now.strftime("%Y-%m-%d %H:%M:%S, %A"),
        "date":      now.strftime("%Y-%m-%d"),
        "time":      now.strftime("%H:%M:%S"),
        "day":       now.strftime("%A"),
        "timestamp": str(int(now.timestamp())),
    }.get(info_type, str(now))


@tool
def file_write(filename: str, content: str) -> str:
    """
    Write text content to a file on disk.
    Returns confirmation with character count.
    Example: file_write("notes.txt", "Hello world")
    """
    safe = os.path.basename(filename)
    with open(safe, "w") as f:
        f.write(content)
    return f"Wrote {len(content)} chars to '{safe}'"


@tool
def file_read(filename: str) -> str:
    """
    Read the contents of a file from disk.
    Returns the file content or an error message if not found.
    """
    safe = os.path.basename(filename)
    if not os.path.exists(safe):
        return f"File '{safe}' not found"
    return open(safe).read()


@tool
def text_analyze(text: str, operation: str) -> str:
    """
    Analyze text statistics.
    operation options: word_count, char_count, sentence_count, all
    Example: text_analyze("Hello world", "word_count") returns "2"
    """
    words = text.strip().split()
    sents = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    return {
        "word_count":     str(len(words)),
        "char_count":     str(len(text)),
        "sentence_count": str(len(sents)),
        "all": f"words={len(words)} chars={len(text)} sentences={len(sents)}",
    }.get(operation, f"Unknown operation: {operation}")


def demo_agent():
    print("\n" + "═"*60)
    print("📌 SECTION 4: Tools + ReAct Agent")
    print("═"*60)

    llm = get_llm(temperature=0)
    tools = [calculator, get_datetime, file_write, file_read, text_analyze]

    # In LangChain 1.0+, AgentExecutor + create_react_agent + hub prompts
    # are gone. create_agent builds the same Thought → Action → Observation
    # loop internally (via LangGraph), driven by tool-calling instead of
    # a parsed ReAct text format — no prompt template needed.
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are a helpful assistant with access to tools. "
            "Use them whenever they help answer the question accurately. "
            "Think step by step, and use multiple tools in sequence when needed."
        ),
    )

    goals = [
        # 4a: Single tool
        "What is the square root of 2025?",

        # 4b: Multi-tool chain
        "Get today's date. Calculate how many days are in "
        "10 years (use 365.25 days/year). Write both answers "
        "to 'langchain_demo.txt'.",

        # 4c: Text + file pipeline
        "Analyze this text and get all stats, then write "
        "the stats to 'stats_report.txt': "
        "'LangChain makes it easy to build LLM-powered applications "
        "by providing modular components for prompts, chains, and agents.'",
    ]

    for i, goal in enumerate(goals, 1):
        print(f"\n{'─'*60}")
        print(f"🎯 Goal {i}: {goal[:80]}")
        print('─'*60)
        try:
            # recursion_limit replaces max_iterations (each model call +
            # each tool call is one "step" in the graph, so give it
            # enough headroom for multi-tool goals)
            result = agent.invoke(
                {"messages": [{"role": "user", "content": goal}]},
                config={"recursion_limit": 25},
            )

            # verbose=True used to print Thought/Action/Observation for you;
            # here we replicate that by walking the returned message list
            for msg in result["messages"]:
                role = getattr(msg, "type", msg.__class__.__name__)
                if role == "ai" and getattr(msg, "tool_calls", None):
                    for tc in msg.tool_calls:
                        print(f"  🔧 Action: {tc['name']}({tc['args']})")
                elif role == "tool":
                    print(f"  👁️  Observation: {msg.content}")

            final_message = result["messages"][-1]
            print(f"\n✅ Final: {final_message.content}")
        except Exception as e:
            print(f"❌ Error: {e}")


# ─────────────────────────────────────────────
# SECTION 5: SIDE-BY-SIDE COMPARISON
# Your raw code vs LangChain — same thing
# ─────────────────────────────────────────────

def demo_comparison():
    print("\n" + "═"*60)
    print("📌 SECTION 5: Raw vs LangChain — side by side")
    print("═"*60)

    print("""
  SAME THING — different syntax:

  ┌─────────────────────────────────────────────────────┐
  │ YOUR RAW CODE          │ LANGCHAIN EQUIVALENT        │
  ├────────────────────────┼─────────────────────────────┤
  │ call_llm(messages)     │ llm.invoke(messages)        │
  │ f-string system prompt │ ChatPromptTemplate          │
  │ messages.append(...)   │ MessagesPlaceholder         │
  │ chain A → chain B      │ chain_a | chain_b (LCEL)   │
  │ ToolRegistry + Tool    │ @tool decorator             │
  │ run_react() loop       │ AgentExecutor               │
  │ json.loads(response)   │ JsonOutputParser()          │
  │ max_steps check        │ max_iterations=N            │
  │ parse error recovery   │ handle_parsing_errors=True  │
  └────────────────────────┴─────────────────────────────┘

  LANGCHAIN ADDS:
  ✓ Dozens of pre-built integrations (DBs, APIs, vector stores)
  ✓ LangSmith tracing (observability)
  ✓ Community tool hub
  ✓ Streaming helpers
  ✓ LCEL pipe syntax for clean chaining

  LANGCHAIN COSTS:
  ✗ Extra dependency layer (~30+ packages)
  ✗ Abstractions hide errors (harder to debug)
  ✗ Version churn — APIs change frequently
  ✗ Overkill for simple agents
    """)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    print("🦜 Phase 5.1 — LangChain Basics\n")

    section = sys.argv[1] if len(sys.argv) > 1 else "all"

    if section in ("1", "all"):
        demo_llm_wrapper()

    if section in ("2", "all"):
        demo_prompt_templates()

    if section in ("3", "all"):
        demo_chains()

    if section in ("4", "all"):
        demo_agent()

    if section in ("5", "all"):
        demo_comparison()

    print("\n\n✅ Phase 5.1 complete!")
    print("Run individual sections: python3 agent_13_langchain.py 4")