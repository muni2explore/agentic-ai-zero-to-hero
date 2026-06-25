import requests
import json
import re
import time
import sqlite3
import hashlib
import os
from datetime import datetime, timedelta
from typing import Optional
from contextlib import contextmanager
import logging
import chromadb
from chromadb.utils import embedding_functions
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
# SQLITE LAYER
# Handles: sessions, episodes, structured facts,
# time queries, metadata, relationships
# ─────────────────────────────────────────────

class SQLiteStore:
    def __init__(self, db_path: str = "hybrid_memory.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # dict-like access
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript("""
            -- Sessions: one per conversation
            CREATE TABLE IF NOT EXISTS sessions (
                id          TEXT PRIMARY KEY,
                started_at  TEXT NOT NULL,
                ended_at    TEXT,
                title       TEXT,
                summary     TEXT,
                turn_count  INTEGER DEFAULT 0,
                tags        TEXT DEFAULT '[]'
            );

            -- Episodes: meaningful chunks within sessions
            -- An episode = a topic discussed within a session
            CREATE TABLE IF NOT EXISTS episodes (
                id          TEXT PRIMARY KEY,
                session_id  TEXT NOT NULL,
                started_at  TEXT NOT NULL,
                ended_at    TEXT,
                topic       TEXT,
                summary     TEXT,
                importance  INTEGER DEFAULT 1,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            -- Messages: every single turn
            CREATE TABLE IF NOT EXISTS messages (
                id          TEXT PRIMARY KEY,
                session_id  TEXT NOT NULL,
                episode_id  TEXT,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                timestamp   TEXT NOT NULL,
                token_est   INTEGER DEFAULT 0,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            -- Structured facts with full metadata
            CREATE TABLE IF NOT EXISTS facts (
                id          TEXT PRIMARY KEY,
                content     TEXT NOT NULL,
                category    TEXT DEFAULT 'general',
                source      TEXT DEFAULT 'conversation',
                session_id  TEXT,
                confidence  REAL DEFAULT 1.0,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL,
                access_count INTEGER DEFAULT 0,
                is_active   INTEGER DEFAULT 1
            );

            -- Fact relationships
            CREATE TABLE IF NOT EXISTS fact_relations (
                fact_id_a   TEXT,
                fact_id_b   TEXT,
                relation    TEXT,
                PRIMARY KEY (fact_id_a, fact_id_b)
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id);
            CREATE INDEX IF NOT EXISTS idx_messages_time
                ON messages(timestamp);
            CREATE INDEX IF NOT EXISTS idx_facts_category
                ON facts(category);
            CREATE INDEX IF NOT EXISTS idx_episodes_session
                ON episodes(session_id);
        """)
        self.conn.commit()

    @contextmanager
    def transaction(self):
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # ── Sessions ────────────────────────────

    def create_session(self, session_id: str, title: str = "") -> str:
        with self.transaction() as db:
            db.execute("""
                INSERT OR IGNORE INTO sessions (id, started_at, title)
                VALUES (?, ?, ?)
            """, (session_id, datetime.now().isoformat(), title))
        return session_id

    def end_session(self, session_id: str, summary: str = ""):
        with self.transaction() as db:
            db.execute("""
                UPDATE sessions
                SET ended_at = ?, summary = ?
                WHERE id = ?
            """, (datetime.now().isoformat(), summary, session_id))

    def get_session(self, session_id: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_sessions(self, limit: int = 10) -> list[dict]:
        rows = self.conn.execute("""
            SELECT s.*, COUNT(m.id) as message_count
            FROM sessions s
            LEFT JOIN messages m ON s.id = m.session_id
            GROUP BY s.id
            ORDER BY s.started_at DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]

    # ── Episodes ────────────────────────────

    def create_episode(self, session_id: str, topic: str) -> str:
        ep_id = hashlib.md5(
            f"{session_id}{topic}{time.time()}".encode()
        ).hexdigest()[:12]
        with self.transaction() as db:
            db.execute("""
                INSERT INTO episodes (id, session_id, started_at, topic)
                VALUES (?, ?, ?, ?)
            """, (ep_id, session_id, datetime.now().isoformat(), topic))
        return ep_id

    def close_episode(self, episode_id: str, summary: str,
                      importance: int = 1):
        with self.transaction() as db:
            db.execute("""
                UPDATE episodes
                SET ended_at = ?, summary = ?, importance = ?
                WHERE id = ?
            """, (datetime.now().isoformat(), summary, importance, episode_id))

    def get_recent_episodes(self, limit: int = 5,
                            session_id: str = None) -> list[dict]:
        query  = "SELECT * FROM episodes WHERE 1=1"
        params = []
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        query += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)
        return [dict(r) for r in self.conn.execute(query, params).fetchall()]

    # ── Messages ────────────────────────────

    def store_message(self, session_id: str, role: str,
                      content: str, episode_id: str = None) -> str:
        msg_id = hashlib.md5(
            f"{session_id}{role}{content}{time.time()}".encode()
        ).hexdigest()
        token_est = max(1, len(content) // 4)
        with self.transaction() as db:
            db.execute("""
                INSERT INTO messages
                    (id, session_id, episode_id, role, content, timestamp, token_est)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                msg_id, session_id, episode_id,
                role, content, datetime.now().isoformat(), token_est
            ))
            db.execute("""
                UPDATE sessions SET turn_count = turn_count + 1
                WHERE id = ?
            """, (session_id,))
        return msg_id

    def get_messages(self, session_id: str,
                     limit: int = 50) -> list[dict]:
        rows = self.conn.execute("""
            SELECT * FROM messages
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (session_id, limit)).fetchall()
        return [dict(r) for r in reversed(rows)]

    def get_messages_since(self, hours: float) -> list[dict]:
        since = (datetime.now() - timedelta(hours=hours)).isoformat()
        rows  = self.conn.execute("""
            SELECT * FROM messages
            WHERE timestamp >= ?
            ORDER BY timestamp ASC
        """, (since,)).fetchall()
        return [dict(r) for r in rows]

    # ── Facts ────────────────────────────────

    def store_fact(self, content: str, category: str = "general",
                   source: str = "conversation",
                   session_id: str = None,
                   confidence: float = 1.0) -> str:
        fact_id = hashlib.md5(content.encode()).hexdigest()[:16]
        now     = datetime.now().isoformat()
        with self.transaction() as db:
            db.execute("""
                INSERT INTO facts
                    (id, content, category, source, session_id,
                     confidence, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    updated_at   = excluded.updated_at,
                    confidence   = MAX(facts.confidence, excluded.confidence),
                    access_count = facts.access_count + 1
            """, (fact_id, content, category, source,
                  session_id, confidence, now, now))
        return fact_id

    def get_facts_by_category(self, category: str) -> list[dict]:
        rows = self.conn.execute("""
            SELECT * FROM facts
            WHERE category = ? AND is_active = 1
            ORDER BY confidence DESC, access_count DESC
        """, (category,)).fetchall()
        return [dict(r) for r in rows]

    def update_fact_access(self, fact_id: str):
        with self.transaction() as db:
            db.execute("""
                UPDATE facts
                SET access_count = access_count + 1,
                    updated_at   = ?
                WHERE id = ?
            """, (datetime.now().isoformat(), fact_id))

    def get_all_facts(self, active_only: bool = True) -> list[dict]:
        query = "SELECT * FROM facts"
        if active_only:
            query += " WHERE is_active = 1"
        query += " ORDER BY category, confidence DESC"
        return [dict(r) for r in self.conn.execute(query).fetchall()]

    # ── Time-based queries ────────────────────

    def get_timeline(self, days: int = 7) -> list[dict]:
        """What happened across sessions in the last N days."""
        since = (datetime.now() - timedelta(days=days)).isoformat()
        rows  = self.conn.execute("""
            SELECT
                s.id        as session_id,
                s.started_at,
                s.title,
                s.summary,
                s.turn_count,
                COUNT(DISTINCT e.id) as episode_count
            FROM sessions s
            LEFT JOIN episodes e ON s.id = e.session_id
            WHERE s.started_at >= ?
            GROUP BY s.id
            ORDER BY s.started_at DESC
        """, (since,)).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        def count(table, where=""):
            return self.conn.execute(
                f"SELECT COUNT(*) FROM {table} {where}"
            ).fetchone()[0]
        return {
            "sessions":  count("sessions"),
            "episodes":  count("episodes"),
            "messages":  count("messages"),
            "facts":     count("facts", "WHERE is_active=1"),
        }


# ─────────────────────────────────────────────
# VECTOR LAYER
# Handles: semantic search across all memory types
# ─────────────────────────────────────────────

class VectorStore:
    def __init__(self, persist_dir: str = "./chroma_hybrid"):
        self.client   = chromadb.PersistentClient(path=persist_dir)
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.memories = self.client.get_or_create_collection(
            name="all_memories",
            embedding_function=self.embed_fn,
        )

    def upsert(self, doc_id: str, content: str, metadata: dict):
        """Store or update a memory in the vector store."""
        try:
            self.memories.upsert(
                documents=[content],
                ids=[doc_id],
                metadatas=[metadata]
            )
        except Exception as e:
            print(f"   ⚠️  Vector upsert failed: {e}")

    def search(self, query: str, n: int = 5,
               where: dict = None,
               min_score: float = 0.25) -> list[dict]:
        """Semantic search with optional metadata filter."""
        count = self.memories.count()
        if count == 0:
            return []
        try:
            results = self.memories.query(
                query_texts=[query],
                n_results=min(n, count),
                where=where,
            )
            if not results["documents"][0]:
                return []
            out = []
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                score = round(1 - dist, 3)
                if score >= min_score:
                    out.append({
                        "content":  doc,
                        "metadata": meta,
                        "score":    score,
                    })
            return sorted(out, key=lambda x: x["score"], reverse=True)
        except Exception:
            return []

    def count(self) -> int:
        return self.memories.count()


# ─────────────────────────────────────────────
# HYBRID MEMORY SYSTEM
# Unified API over SQLite + Vector
# Single source of truth for the agent
# ─────────────────────────────────────────────

class HybridMemory:
    def __init__(self,
                 db_path:     str = "hybrid_memory.db",
                 vector_dir:  str = "./chroma_hybrid"):
        print("🔧 Initialising Hybrid Memory...")
        self.sql    = SQLiteStore(db_path)
        self.vector = VectorStore(vector_dir)
        self._session_id:  Optional[str] = None
        self._episode_id:  Optional[str] = None
        print(f"   SQLite : {self.sql.stats()}")
        print(f"   Vectors: {self.vector.count()} entries")

    # ── Session management ────────────────────

    def start_session(self, session_id: str, title: str = "") -> str:
        self._session_id = session_id
        self._episode_id = None
        self.sql.create_session(session_id, title)
        return session_id

    def end_session(self, summary: str = ""):
        if self._session_id:
            if self._episode_id:
                self._close_current_episode()
            self.sql.end_session(self._session_id, summary)

    def new_episode(self, topic: str):
        """Mark the start of a new topic within the session."""
        if self._episode_id:
            self._close_current_episode()
        if self._session_id:
            self._episode_id = self.sql.create_episode(
                self._session_id, topic
            )
            print(f"\n📖 New episode: '{topic}'")

    def _close_current_episode(self):
        if self._episode_id:
            # Get recent messages for this episode and summarise
            msgs = self.sql.get_messages(self._session_id, limit=20)
            ep_msgs = [m for m in msgs if m.get("episode_id") == self._episode_id]
            if ep_msgs:
                summary = self._summarise_episode(ep_msgs)
                importance = self._rate_importance(summary)
                self.sql.close_episode(
                    self._episode_id, summary, importance
                )
                # Store episode summary in vector memory too
                self.vector.upsert(
                    doc_id=f"ep_{self._episode_id}",
                    content=summary,
                    metadata={
                        "type":       "episode",
                        "session_id": self._session_id,
                        "topic":      "",
                        "timestamp":  datetime.now().isoformat(),
                    }
                )
            self._episode_id = None

    # ── Core memory operations ────────────────

    def remember(self, role: str, content: str):
        """Store a message in both SQLite and vector store."""
        if not self._session_id:
            return

        # SQLite
        msg_id = self.sql.store_message(
            self._session_id, role, content, self._episode_id
        )

        # Vector — only store substantive messages
        if len(content.split()) >= 5:
            self.vector.upsert(
                doc_id=f"msg_{msg_id}",
                content=content,
                metadata={
                    "type":       "message",
                    "role":       role,
                    "session_id": self._session_id or "",
                    "episode_id": self._episode_id or "",
                    "timestamp":  datetime.now().isoformat(),
                }
            )

    def learn_fact(self, content: str, category: str = "general",
                   confidence: float = 1.0):
        """Store a fact in both layers."""
        fact_id = self.sql.store_fact(
            content, category,
            session_id=self._session_id,
            confidence=confidence
        )
        self.vector.upsert(
            doc_id=f"fact_{fact_id}",
            content=content,
            metadata={
                "type":      "fact",
                "category":  category,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def ingest_document(self, content: str, title: str,
                        doc_type: str = "text",
                        chunk_size: int = 300):
        """Chunk and store a document for RAG."""
        chunks  = self._chunk(content, chunk_size)
        for i, chunk in enumerate(chunks):
            chunk_id = hashlib.md5(
                f"{title}{i}".encode()
            ).hexdigest()[:16]
            self.vector.upsert(
                doc_id=f"doc_{chunk_id}",
                content=chunk,
                metadata={
                    "type":        "document",
                    "title":       title,
                    "doc_type":    doc_type,
                    "chunk_index": i,
                    "timestamp":   datetime.now().isoformat(),
                }
            )
        print(f"   📄 '{title}' → {len(chunks)} chunks")

    # ── Retrieval ────────────────────────────

    def recall(self, query: str, n: int = 6,
               memory_types: list = None) -> dict:
        """
        Unified recall — searches vector store,
        optionally filtered by memory type.
        Also enriches with SQLite metadata.
        """
        where = None
        if memory_types:
            if len(memory_types) == 1:
                where = {"type": memory_types[0]}

        raw = self.vector.search(query, n=n, where=where)

        # Enrich with time context
        results = {"semantic": raw}

        # Add episodic context — what happened recently
        results["recent_episodes"] = self.sql.get_recent_episodes(limit=3)

        # Add structured facts for this category
        results["structured_facts"] = self.sql.get_all_facts()[:10]

        return results

    def recall_timeline(self, days: int = 7) -> list[dict]:
        """What happened in the last N days."""
        return self.sql.get_timeline(days)

    def recall_session(self, session_id: str) -> dict:
        """Full replay of a past session."""
        session  = self.sql.get_session(session_id)
        messages = self.sql.get_messages(session_id)
        episodes = self.sql.get_recent_episodes(session_id=session_id)
        return {
            "session":  session,
            "messages": messages,
            "episodes": episodes,
        }

    def build_context(self, query: str, min_score: float = 0.3) -> str:
        """
        Build the full memory context string to inject into the LLM prompt.
        Combines semantic search + recent episodes + key facts.
        """
        sections = []

        # 1. Semantic results
        semantic = self.vector.search(query, n=5, min_score=min_score)
        if semantic:
            lines = []
            for r in semantic:
                t    = r["metadata"].get("type", "?")
                ts   = r["metadata"].get("timestamp", "")[:10]
                lines.append(
                    f"  [{t}|{ts}|score={r['score']}] {r['content'][:120]}"
                )
            sections.append("SEMANTIC RECALL:\n" + "\n".join(lines))

        # 2. Recent episodes
        episodes = self.sql.get_recent_episodes(limit=3)
        if episodes:
            lines = []
            for ep in episodes:
                if ep.get("summary"):
                    lines.append(
                        f"  [{ep['started_at'][:10]}] {ep['topic']}: {ep['summary'][:100]}"
                    )
            if lines:
                sections.append("RECENT EPISODES:\n" + "\n".join(lines))

        # 3. High-confidence facts
        facts = self.sql.get_all_facts()
        if facts:
            top_facts = [f for f in facts if f["confidence"] >= 0.8][:5]
            if top_facts:
                lines = [f"  [{f['category']}] {f['content']}" for f in top_facts]
                sections.append("KEY FACTS:\n" + "\n".join(lines))

        if not sections:
            return ""
        return "━━━ MEMORY CONTEXT ━━━\n" + "\n\n".join(sections) + "\n━━━"

    # ── Intelligence helpers ──────────────────

    def _summarise_episode(self, messages: list[dict]) -> str:
        text = "\n".join(
            f"{m['role'].upper()}: {m['content'][:200]}"
            for m in messages
        )
        prompt = [{
            "role": "system",
            "content": (
                "Summarise this conversation episode in 2-3 bullet points. "
                "Focus on: decisions made, facts revealed, outcomes. "
                "Be concise. No preamble."
            )
        }, {
            "role": "user",
            "content": text
        }]
        try:
            return call_llm(messages=prompt)
        except Exception:
            return "Episode summary unavailable."

    def _rate_importance(self, summary: str) -> int:
        """Rate episode importance 1-5 using LLM."""
        prompt = [{
            "role": "system",
            "content": (
                "Rate the importance of this memory for future recall. "
                "Respond with ONLY a single digit 1-5. "
                "5=critical decision/fact, 1=small talk."
            )
        }, {"role": "user", "content": summary}]
        try:
            raw = call_llm(messages=prompt)
            match = re.search(r'[1-5]', raw)
            return int(match.group()) if match else 1
        except Exception:
            return 1

    def _chunk(self, text: str, size: int) -> list[str]:
        words   = text.split()
        overlap = size // 5
        chunks, i = [], 0
        while i < len(words):
            chunks.append(" ".join(words[i:i + size]))
            i += size - overlap
        return chunks or [text]

    def stats(self) -> dict:
        return {
            "sql":    self.sql.stats(),
            "vector": self.vector.count(),
        }


# ─────────────────────────────────────────────
# FACT EXTRACTOR (LLM-based)
# ─────────────────────────────────────────────

def extract_facts(text: str) -> list[dict]:
    prompt = [{
        "role": "system",
        "content": (
            "Extract factual statements from the text. "
            "Return ONLY a JSON array: "
            '[{"fact":"...","category":"personal|technical|project|preference|other","confidence":0.9}]. '
            "Confidence: 1.0=certain, 0.7=likely, 0.5=uncertain. "
            "If no facts, return []. No markdown."
        )
    }, {"role": "user", "content": text}]
    raw = call_llm(messages=prompt)
    cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip()).strip()
    try:
        result = json.loads(cleaned)
        return result if isinstance(result, list) else []
    except Exception:
        return []


def detect_topic_shift(prev_messages: list[dict],
                       new_message: str) -> Optional[str]:
    """
    Ask LLM if the conversation has shifted to a new topic.
    Returns new topic name or None.
    """
    if len(prev_messages) < 2:
        return new_message[:40]

    recent = "\n".join(
        f"{m['role']}: {m['content'][:100]}"
        for m in prev_messages[-4:]
    )
    prompt = [{
        "role": "system",
        "content": (
            "Detect if the new message shifts to a significantly different topic "
            "than the recent conversation. "
            'Respond with JSON: {"shifted": true/false, "new_topic": "brief topic name"}. '
            "Only shifted=true for clear topic changes. No markdown."
        )
    }, {
        "role": "user",
        "content": f"RECENT:\n{recent}\n\nNEW MESSAGE: {new_message}"
    }]
    raw = call_llm(messages=prompt)
    cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip()).strip()
    try:
        parsed = json.loads(cleaned)
        if parsed.get("shifted"):
            return parsed.get("new_topic", "New Topic")
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────
# HYBRID AGENT — full chat loop
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant with a sophisticated long-term memory system.

You are provided MEMORY CONTEXT before each response — this contains:
- Semantically relevant past conversations
- Recent episode summaries
- Key facts about the user

Use this memory naturally. Reference past context when relevant.
If memory contradicts what you know, trust the memory — it's ground truth about this user.
Be concise but warm. When using memory, say "Based on what you've shared..." or "I recall that..."
"""


def print_stats(memory: HybridMemory):
    s = memory.stats()
    sq = s["sql"]
    print(f"\n  📊 sessions={sq['sessions']} | "
          f"episodes={sq['episodes']} | "
          f"messages={sq['messages']} | "
          f"facts={sq['facts']} | "
          f"vectors={s['vector']}")


def run_chat(session_id: str, memory: HybridMemory):
    print(f"\n{'='*60}")
    print(f"💬 Session: {session_id}")
    print_stats(memory)
    print("\nCommands: quit | stats | timeline | sessions | "
          "recall <query> | episode <topic> | ingest")
    print('='*60)

    memory.start_session(
        session_id,
        title=f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    memory.new_episode("Opening")

    short_term = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        # ── Commands ──────────────────────────

        if user_input.lower() == "quit":
            summary = _generate_session_summary(short_term)
            memory.end_session(summary)
            print(f"\n💾 Session ended. Summary: {summary[:100]}...")
            break

        if user_input.lower() == "stats":
            print_stats(memory)
            continue

        if user_input.lower() == "timeline":
            timeline = memory.recall_timeline(days=30)
            print("\n📅 Timeline (last 30 days):")
            if not timeline:
                print("  No sessions yet.")
            for entry in timeline:
                print(f"  {entry['started_at'][:10]} | "
                      f"{entry['turn_count']} turns | "
                      f"{entry['title'] or 'Untitled'}")
                if entry.get("summary"):
                    print(f"    → {entry['summary'][:80]}")
            continue

        if user_input.lower() == "sessions":
            sessions = memory.sql.list_sessions(10)
            print("\n📚 Past sessions:")
            for s in sessions:
                print(f"  [{s['id']}] {s['started_at'][:16]} | "
                      f"{s['turn_count']} turns | "
                      f"{s['title'] or 'Untitled'}")
            continue

        if user_input.lower().startswith("recall "):
            query = user_input[7:].strip()
            print(f"\n🔍 Recalling: '{query}'")
            results = memory.vector.search(query, n=5, min_score=0.2)
            if not results:
                print("  Nothing found.")
            for r in results:
                t  = r["metadata"].get("type", "?")
                ts = r["metadata"].get("timestamp", "")[:10]
                print(f"  [{t}|{ts}|{r['score']:.2f}] {r['content'][:100]}")
            continue

        if user_input.lower().startswith("episode "):
            topic = user_input[8:].strip()
            memory.new_episode(topic)
            continue

        if user_input.lower() == "ingest":
            _ingest_knowledge(memory)
            continue

        # ── Topic shift detection ──────────────

        user_messages = [m for m in short_term if m["role"] == "user"]
        new_topic = detect_topic_shift(user_messages[-4:], user_input) \
                    if len(user_messages) >= 3 else None
        if new_topic:
            memory.new_episode(new_topic)

        # ── Store + extract facts ──────────────

        memory.remember("user", user_input)

        facts = extract_facts(user_input)
        for item in facts:
            if isinstance(item, dict) and item.get("fact"):
                memory.learn_fact(
                    item["fact"],
                    item.get("category", "general"),
                    item.get("confidence", 0.8),
                )
                print(f"   💡 [{item.get('category','?')}|"
                      f"conf={item.get('confidence',0.8)}] {item['fact'][:60]}")

        # ── Build context + respond ────────────

        mem_context = memory.build_context(user_input)

        messages = list(short_term)
        if mem_context:
            messages.append({"role": "system", "content": mem_context})
        messages.append({"role": "user", "content": user_input})

        reply = call_llm(messages)

        memory.remember("assistant", reply)

        short_term.append({"role": "user",      "content": user_input})
        short_term.append({"role": "assistant",  "content": reply})
        if len(short_term) > 15:
            short_term = [short_term[0]] + short_term[-14:]

        print(f"\nAssistant: {reply}")


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _generate_session_summary(messages: list[dict]) -> str:
    text = "\n".join(
        f"{m['role']}: {m['content'][:150]}"
        for m in messages[1:]  # skip system
    )
    prompt = [{
        "role": "system",
        "content": (
            "Write a 2-sentence summary of this conversation session. "
            "Include: main topics, key facts revealed, decisions made."
        )
    }, {"role": "user", "content": text}]
    try:
        return call_llm(messages=prompt)
    except Exception:
        return "Session summary unavailable."


def _ingest_knowledge(memory: HybridMemory):
    docs = [
        ("Proxmox Setup", "technical", """
Proxmox 8.1 homelab. Container 100: ollama-lxc with RTX A2000 GPU passthrough,
runs llama3.2 and mistral on port 11434. GPU issue: broken nvidia-uvm device
node resolved with udev rules. Container 102: kali-linux with WebSploit Labs,
19 Docker containers for ethical hacking practice. WireGuard VPN for remote access.
        """),
        ("Active Projects", "project", """
1. Agentic AI learning — building from scratch with Ollama on Proxmox.
2. Child knowledge-sharing site — Payload CMS, Next.js, Vercel, YouTube API.
   Channel: youtube.com/@hello-magi. Young Scientist program content.
3. Will Sparrow Technologies — software, electronics, PCB services.
4. AEM multi-tenant project — HTL, Sling Models, SCSS, RTE customisation.
5. Laravel payroll system — EPF, ESI, TDS for FY 2025-26, PostgreSQL.
        """),
        ("Tech Stack", "preference", """
Backend: Laravel (primary), FastAPI. Frontend: React, Inertia.js.
Databases: PostgreSQL, SQLite, Redis, ChromaDB. Infrastructure: Proxmox,
Docker, Kubernetes (kubeadm + Flannel), Terraform. Hardware: Arduino, ESP32.
AI: Ollama, LangGraph, ChromaDB. Systems: Go (built container runtime,
mini-k8s, TCP proxy). Learning style: build from scratch before frameworks.
        """),
    ]
    print("\n📥 Ingesting knowledge base...")
    for title, doc_type, content in docs:
        memory.ingest_document(content.strip(), title, doc_type)

    base_facts = [
        ("Muni is a full-stack developer in Krishnagiri, Tamil Nadu", "personal",    1.0),
        ("Muni runs Ollama on Proxmox LXC with RTX A2000",           "technical",   1.0),
        ("Muni learns by building systems from scratch",              "preference",  1.0),
        ("Muni's child has a YouTube channel @hello-magi",           "personal",    1.0),
        ("Muni works with Will Sparrow Technologies",                 "project",     1.0),
    ]
    for content, category, confidence in base_facts:
        memory.learn_fact(content, category, confidence)
    print(f"   ✅ Knowledge base ready")


# ─────────────────────────────────────────────
# DEMO — automated run showing all features
# ─────────────────────────────────────────────

def run_demo(memory: HybridMemory):
    print("\n" + "="*60)
    print("🎬 HYBRID MEMORY DEMO")
    print("="*60)

    # Ingest knowledge
    _ingest_knowledge(memory)

    # Session 1 — teach the agent things
    print("\n\n── SESSION 1: Teaching ──")
    memory.start_session("demo_s1", "Demo session 1")
    memory.new_episode("Introduction")

    exchanges_s1 = [
        "My name is Muni and I am learning agentic AI",
        "I prefer to learn by building things from scratch rather than using tutorials",
        "I am currently on Phase 3 of the roadmap — memory systems",
        "My Proxmox server has an RTX A2000 GPU",
    ]
    short_term = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in exchanges_s1:
        print(f"\n👤 {msg}")
        memory.remember("user", msg)
        facts = extract_facts(msg)
        for f in facts:
            if f.get("fact"):
                memory.learn_fact(f["fact"], f.get("category","general"),
                                  f.get("confidence", 0.8))
        ctx = memory.build_context(msg)
        msgs = list(short_term)
        if ctx:
            msgs.append({"role": "system", "content": ctx})
        msgs.append({"role": "user", "content": msg})
        reply = call_llm(msgs)
        memory.remember("assistant", reply)
        short_term.append({"role": "user", "content": msg})
        short_term.append({"role": "assistant", "content": reply})
        print(f"🤖 {reply[:150]}...")

    memory.end_session("Muni introduced themselves and shared learning preferences.")

    # Session 2 — test recall across sessions
    print("\n\n── SESSION 2: Recall across sessions ──")
    memory.start_session("demo_s2", "Demo session 2")
    memory.new_episode("Recall test")

    test_queries = [
        "What do you remember about me?",
        "What phase of the agentic AI roadmap am I on?",
        "What GPU does my Proxmox server have?",
        "What is my learning style?",
    ]
    short_term2 = [{"role": "system", "content": SYSTEM_PROMPT}]
    for query in test_queries:
        print(f"\n❓ {query}")
        ctx = memory.build_context(query)
        msgs = list(short_term2)
        if ctx:
            msgs.append({"role": "system", "content": ctx})
        msgs.append({"role": "user", "content": query})
        reply = call_llm(msgs)
        print(f"🤖 {reply[:200]}...")
        short_term2.append({"role": "user", "content": query})
        short_term2.append({"role": "assistant", "content": reply})

    memory.end_session("Tested cross-session memory recall successfully.")

    # Show timeline
    print("\n\n── TIMELINE ──")
    for entry in memory.recall_timeline(days=1):
        print(f"  {entry['started_at'][:16]} | "
              f"{entry['turn_count']} turns | {entry['title']}")

    print("\n\n✅ Hybrid memory demo complete!")
    print_stats(memory)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("🧠 Phase 3, Step 3.3 — Hybrid Memory (SQLite + Vector + Episodic)\n")

    memory = HybridMemory(
        db_path    = "hybrid_memory.db",
        vector_dir = "./chroma_hybrid"
    )

    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        run_demo(memory)
    else:
        session_id = sys.argv[1] if len(sys.argv) > 1 else \
                     f"session_{datetime.now().strftime('%Y%m%d_%H%M')}"
        run_chat(session_id, memory)