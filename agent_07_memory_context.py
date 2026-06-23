import requests
import json
import re
import time
import sqlite3
import os
from datetime import datetime
import logging
from typing import Optional
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
# TOKEN ESTIMATOR
# Ollama doesn't return token counts mid-conversation,
# so we estimate: ~4 chars per token (good enough for windowing)
# ─────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)

def estimate_messages_tokens(messages: list[dict]) -> int:
    total = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        total += 4  # role + overhead per message
    return total


# ─────────────────────────────────────────────
# MEMORY MANAGER
# Handles 4 types of memory slots:
#   PINNED   — never dropped (system prompt, original goal)
#   SUMMARY  — compressed digest of dropped messages
#   RECENT   — sliding window of recent messages
#   WORKING  — agent's scratchpad for current task
# ─────────────────────────────────────────────

class MemoryManager:
    def __init__(self, max_tokens: int = 3000, summary_trigger: float = 0.75):
        """
        max_tokens:      context limit before compressing
        summary_trigger: compress when usage exceeds this fraction of max
        """
        self.max_tokens      = max_tokens
        self.summary_trigger = summary_trigger

        self.pinned: list[dict]  = []   # always kept — system + goal
        self.summary: str        = ""   # compressed digest of old messages
        self.recent: list[dict]  = []   # sliding window
        self.working: dict       = {}   # key-value scratchpad

        self.compression_count   = 0
        self.total_msgs_seen     = 0

    # ── Public API ──────────────────────────

    def pin(self, message: dict):
        """Pin a message — it will never be dropped."""
        self.pinned.append(message)

    def add(self, message: dict):
        """Add a message to the sliding window."""
        self.recent.append(message)
        self.total_msgs_seen += 1
        self._maybe_compress()

    def remember(self, key: str, value: str):
        """Store a key fact in the working scratchpad."""
        self.working[key] = value

    def recall(self, key: str) -> Optional[str]:
        """Retrieve a key fact from the scratchpad."""
        return self.working.get(key)

    def get_messages(self) -> list[dict]:
        """
        Build the full messages list to send to the LLM.
        Structure: pinned + [summary injection] + recent
        """
        messages = list(self.pinned)

        if self.summary:
            messages.append({
                "role":    "system",
                "content": f"[MEMORY SUMMARY — earlier conversation compressed]\n{self.summary}"
            })

        if self.working:
            facts = "\n".join(f"  {k}: {v}" for k, v in self.working.items())
            messages.append({
                "role":    "system",
                "content": f"[WORKING MEMORY — key facts from this session]\n{facts}"
            })

        messages.extend(self.recent)
        return messages

    def stats(self) -> dict:
        total_tokens = estimate_messages_tokens(self.get_messages())
        return {
            "pinned_msgs":       len(self.pinned),
            "recent_msgs":       len(self.recent),
            "working_facts":     len(self.working),
            "summary_length":    len(self.summary),
            "compressions":      self.compression_count,
            "total_msgs_seen":   self.total_msgs_seen,
            "estimated_tokens":  total_tokens,
            "token_limit":       self.max_tokens,
            "usage_pct":         round(total_tokens / self.max_tokens * 100, 1),
        }

    # ── Internal ────────────────────────────

    def _maybe_compress(self):
        """Compress recent messages if we're approaching the token limit."""
        current_tokens = estimate_messages_tokens(self.get_messages())
        threshold      = int(self.max_tokens * self.summary_trigger)

        if current_tokens < threshold:
            return

        # Keep the most recent 4 messages untouched
        keep_count    = 4
        to_compress   = self.recent[:-keep_count] if len(self.recent) > keep_count else []
        self.recent   = self.recent[-keep_count:]

        if not to_compress:
            return

        print(f"\n🗜️  Compressing {len(to_compress)} messages into summary...")

        new_digest     = self._summarize(to_compress)
        self.summary   = self._merge_summaries(self.summary, new_digest)
        self.compression_count += 1

        print(f"   Summary updated ({len(self.summary)} chars)")

    def _summarize(self, messages: list[dict]) -> str:
        """Ask the LLM to compress a list of messages into a digest."""
        conversation_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in messages
        )
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a memory compressor. "
                    "Summarize the conversation below into a concise digest. "
                    "Preserve: key facts, decisions made, tool results, important numbers. "
                    "Discard: filler, repeated info, pleasantries. "
                    "Format: bullet points, max 150 words."
                )
            },
            {
                "role": "user",
                "content": f"Compress this:\n\n{conversation_text}"
            }
        ]
        return call_llm(messages=prompt)

    def _merge_summaries(self, old: str, new: str) -> str:
        """Merge an existing summary with a new one."""
        if not old:
            return new
        prompt = [
            {
                "role": "system",
                "content": (
                    "Merge these two memory summaries into one concise summary. "
                    "Keep all unique facts. Remove duplicates. Max 200 words. Bullet points."
                )
            },
            {
                "role": "user",
                "content": f"EXISTING:\n{old}\n\nNEW:\n{new}"
            }
        ]
        return call_llm(messages=prompt)


# ─────────────────────────────────────────────
# SESSION STORE
# Saves/loads conversations from SQLite
# so memory survives script restarts
# ─────────────────────────────────────────────

class SessionStore:
    def __init__(self, db_path: str = "sessions.db"):
        self.conn = sqlite3.connect(db_path)
        self._init_db()

    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id         TEXT PRIMARY KEY,
                created_at TEXT,
                updated_at TEXT,
                summary    TEXT,
                working    TEXT,
                messages   TEXT
            )
        """)
        self.conn.commit()

    def save(self, session_id: str, memory: MemoryManager):
        now = datetime.now().isoformat()
        # Only save last 20 recent messages to avoid huge DB entries
        recent_to_save = memory.recent[-20:]
        self.conn.execute("""
            INSERT INTO sessions (id, created_at, updated_at, summary, working, messages)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                updated_at = excluded.updated_at,
                summary    = excluded.summary,
                working    = excluded.working,
                messages   = excluded.messages
        """, (
            session_id, now, now,
            memory.summary,
            json.dumps(memory.working),
            json.dumps(recent_to_save)
        ))
        self.conn.commit()
        print(f"💾 Session '{session_id}' saved")

    def load(self, session_id: str, memory: MemoryManager) -> bool:
        row = self.conn.execute(
            "SELECT summary, working, messages FROM sessions WHERE id = ?",
            (session_id,)
        ).fetchone()

        if not row:
            return False

        memory.summary = row[0] or ""
        memory.working = json.loads(row[1]) if row[1] else {}
        saved_msgs     = json.loads(row[2]) if row[2] else []
        memory.recent  = saved_msgs
        print(f"📂 Session '{session_id}' loaded "
              f"({len(saved_msgs)} messages, "
              f"{len(memory.working)} facts)")
        return True

    def list_sessions(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
        ).fetchall()
        return [{"id": r[0], "created": r[1], "updated": r[2]} for r in rows]



# ─────────────────────────────────────────────
# SMART CHAT AGENT
# A conversational agent with full memory management
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant with persistent memory.

When the user tells you facts about themselves or makes decisions, acknowledge them.
When referencing earlier information, be explicit: "As you mentioned earlier..."
If you don't remember something, say so honestly.

Be concise but complete in your answers."""


def print_memory_stats(memory: MemoryManager):
    s = memory.stats()
    bar_filled = int(s["usage_pct"] / 5)
    bar        = "█" * bar_filled + "░" * (20 - bar_filled)
    print(f"\n  📊 Memory: [{bar}] {s['usage_pct']}% "
          f"({s['estimated_tokens']}/{s['token_limit']} tokens) | "
          f"recent={s['recent_msgs']} msgs | "
          f"compressions={s['compressions']} | "
          f"facts={s['working_facts']}")


def chat_session(session_id: str, store: SessionStore):
    memory = MemoryManager(max_tokens=3000, summary_trigger=0.75)
    memory.pin({"role": "system", "content": SYSTEM_PROMPT})

    # Try to load existing session
    loaded = store.load(session_id, memory)
    if loaded:
        print(f"\n✅ Resumed session '{session_id}'")
        if memory.summary:
            print(f"📝 Summary from last time:\n{memory.summary}\n")
        if memory.working:
            print(f"🧠 Known facts: {memory.working}")
    else:
        print(f"\n🆕 New session '{session_id}'")

    print("\nType 'quit' to exit, 'memory' to see stats, 'facts' to see working memory\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            store.save(session_id, memory)
            break

        if user_input.lower() == "memory":
            print_memory_stats(memory)
            s = memory.stats()
            if memory.summary:
                print(f"\n  📝 Summary:\n{memory.summary}")
            continue

        if user_input.lower() == "facts":
            if memory.working:
                print("\n  🧠 Working memory facts:")
                for k, v in memory.working.items():
                    print(f"    {k}: {v}")
            else:
                print("  No facts stored yet.")
            continue

        # Add user message
        memory.add({"role": "user", "content": user_input})

        # Auto-extract facts from user message
        _auto_extract_facts(user_input, memory)

        # Get response
        messages  = memory.get_messages()
        reply     = call_llm(messages)

        # Add assistant reply
        memory.add({"role": "assistant", "content": reply})

        print(f"\nAssistant: {reply}")
        print_memory_stats(memory)
        print()

        # Auto-save every 5 messages
        if memory.total_msgs_seen % 5 == 0:
            store.save(session_id, memory)


def _auto_extract_facts(text: str, memory: MemoryManager):
    """
    Simple pattern-based fact extractor.
    In Phase 3.3 we'll replace this with LLM-based extraction.
    """
    patterns = [
        (r"my name is ([A-Za-z]+)",            "user_name"),
        (r"i(?:'m| am) ([A-Za-z]+ developer)", "user_role"),
        (r"i(?:'m| am) working on (.+?)[\.\!]","current_project"),
        (r"i(?:'m| am) from ([A-Za-z]+)",      "user_location"),
        (r"i use ([A-Za-z]+) for",             "preferred_tool"),
        (r"my (?:favourite|favorite) (.+?) is (.+?)[\.\!]", "preference"),
    ]
    text_lower = text.lower()
    for pattern, key in patterns:
        match = re.search(pattern, text_lower)
        if match:
            value = match.group(1)
            memory.remember(key, value)


# ─────────────────────────────────────────────
# DEMO: Automated run to show memory in action
# ─────────────────────────────────────────────

def run_demo(store: SessionStore):
    """
    Simulates a multi-turn conversation that
    triggers compression, then resumes the session.
    """
    print("\n" + "="*60)
    print("🎬 DEMO MODE — automated conversation")
    print("="*60)

    memory = MemoryManager(max_tokens=1500, summary_trigger=0.70)
    memory.pin({"role": "system", "content": SYSTEM_PROMPT})

    # Simulate a long conversation
    exchanges = [
        ("My name is Muni and I build AI agents.",
         None),
        ("I work with Laravel, React, and I run Ollama on my Proxmox server.",
         None),
        ("What is 2 + 2?",
         None),
        ("I am based in Krishnagiri, Tamil Nadu.",
         None),
        ("Tell me about the ReAct agent pattern.",
         None),
        ("What tools have we discussed so far in building agents?",
         None),
        ("My current project is a knowledge-sharing website for my child.",
         None),
        ("What do you remember about me so far?",
         None),
    ]

    for user_msg, _ in exchanges:
        print(f"\n{'─'*40}")
        print(f"👤 User: {user_msg}")

        memory.add({"role": "user", "content": user_msg})
        _auto_extract_facts(user_msg, memory)

        reply = call_llm(memory.get_messages())
        memory.add({"role": "assistant", "content": reply})

        print(f"🤖 Assistant: {reply[:200]}{'...' if len(reply) > 200 else ''}")
        print_memory_stats(memory)

    # Save session
    store.save("demo_session", memory)

    print("\n\n" + "="*60)
    print("🔄 SIMULATING RESTART — loading saved session")
    print("="*60)

    # Reload from DB
    memory2 = MemoryManager(max_tokens=1500)
    memory2.pin({"role": "system", "content": SYSTEM_PROMPT})
    store.load("demo_session", memory2)

    # Ask something that requires memory
    test_question = "What is my name and what project am I working on?"
    print(f"\n👤 User (after restart): {test_question}")
    memory2.add({"role": "user", "content": test_question})
    reply = call_llm(memory2.get_messages())
    print(f"🤖 Assistant: {reply}")
    print("\n✅ Memory survived the restart!")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("🧠 Phase 3, Step 3.1 — In-Context Memory\n")

    store = SessionStore("sessions.db")

    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        run_demo(store)
    else:
        # Interactive mode
        print("Existing sessions:")
        sessions = store.list_sessions()
        if sessions:
            for s in sessions:
                print(f"  - {s['id']} (updated: {s['updated']})")
        else:
            print("  None yet")

        session_id = input("\nEnter session name (or press Enter for 'default'): ").strip()
        if not session_id:
            session_id = "default"

        chat_session(session_id, store)