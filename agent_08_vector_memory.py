import requests
import json
import re
import time
import hashlib
import os
from datetime import datetime
from typing import Optional
import logging
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
# VECTOR MEMORY STORE
# Uses ChromaDB with local sentence-transformer embeddings.
# No OpenAI API needed — runs entirely on your Proxmox box.
# ─────────────────────────────────────────────

import chromadb
from chromadb.utils import embedding_functions

class VectorMemory:
    def __init__(self, persist_dir: str = "./chroma_memory"):
        """
        persist_dir: where ChromaDB stores its data on disk.
        Data survives restarts automatically.
        """
        print("🔧 Initialising ChromaDB...")

        self.client = chromadb.PersistentClient(path=persist_dir)

        # Local embedding model — no API key, runs on CPU
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"  # 80MB, fast, good quality
        )

        # Three separate collections — different types of memory
        self.conversations = self.client.get_or_create_collection(
            name="conversations",
            embedding_function=self.embed_fn,
            metadata={"description": "raw conversation turns"}
        )
        self.facts = self.client.get_or_create_collection(
            name="facts",
            embedding_function=self.embed_fn,
            metadata={"description": "extracted key facts about the user"}
        )
        self.documents = self.client.get_or_create_collection(
            name="documents",
            embedding_function=self.embed_fn,
            metadata={"description": "ingested documents and code snippets"}
        )

        print(f"   conversations: {self.conversations.count()} entries")
        print(f"   facts:         {self.facts.count()} entries")
        print(f"   documents:     {self.documents.count()} entries")

    # ── STORE ────────────────────────────────

    def store_conversation(self, role: str, content: str,
                           session_id: str = "default") -> str:
        """Store a single conversation turn."""
        doc_id = hashlib.md5(
            f"{session_id}{role}{content}{time.time()}".encode()
        ).hexdigest()

        self.conversations.add(
            documents=[content],
            ids=[doc_id],
            metadatas=[{
                "role":       role,
                "session_id": session_id,
                "timestamp":  datetime.now().isoformat(),
                "preview":    content[:80],
            }]
        )
        return doc_id

    def store_fact(self, fact: str, category: str = "general",
                   source: str = "conversation") -> str:
        """Store an extracted fact for long-term recall."""
        doc_id = hashlib.md5(f"{fact}{time.time()}".encode()).hexdigest()

        self.facts.add(
            documents=[fact],
            ids=[doc_id],
            metadatas=[{
                "category":  category,
                "source":    source,
                "timestamp": datetime.now().isoformat(),
            }]
        )
        return doc_id

    def store_document(self, content: str, title: str,
                       doc_type: str = "text",
                       chunk_size: int = 400) -> list[str]:
        """
        Store a document as chunks for RAG retrieval.
        Long documents are split so each chunk is independently retrievable.
        """
        chunks  = self._chunk_text(content, chunk_size)
        ids     = []

        for i, chunk in enumerate(chunks):
            doc_id = hashlib.md5(
                f"{title}{i}{chunk[:50]}".encode()
            ).hexdigest()

            self.documents.add(
                documents=[chunk],
                ids=[doc_id],
                metadatas=[{
                    "title":       title,
                    "doc_type":    doc_type,
                    "chunk_index": i,
                    "total_chunks":len(chunks),
                    "timestamp":   datetime.now().isoformat(),
                }]
            )
            ids.append(doc_id)

        return ids

    # ── RETRIEVE ─────────────────────────────

    def search_conversations(self, query: str, n: int = 5,
                             session_id: str = None) -> list[dict]:
        """Find past conversation turns semantically similar to query."""
        where = {"session_id": session_id} if session_id else None
        try:
            results = self.conversations.query(
                query_texts=[query],
                n_results=min(n, max(1, self.conversations.count())),
                where=where,
            )
            return self._format_results(results)
        except Exception:
            return []

    def search_facts(self, query: str, n: int = 5,
                     category: str = None) -> list[dict]:
        """Find stored facts relevant to the query."""
        where = {"category": category} if category else None
        try:
            results = self.facts.query(
                query_texts=[query],
                n_results=min(n, max(1, self.facts.count())),
                where=where,
            )
            return self._format_results(results)
        except Exception:
            return []

    def search_documents(self, query: str, n: int = 4,
                         doc_type: str = None) -> list[dict]:
        """RAG: retrieve document chunks relevant to the query."""
        where = {"doc_type": doc_type} if doc_type else None
        try:
            results = self.documents.query(
                query_texts=[query],
                n_results=min(n, max(1, self.documents.count())),
                where=where,
            )
            return self._format_results(results)
        except Exception:
            return []

    def search_all(self, query: str, n_each: int = 3) -> dict:
        """Search across all collections and return merged results."""
        return {
            "conversations": self.search_conversations(query, n_each),
            "facts":         self.search_facts(query, n_each),
            "documents":     self.search_documents(query, n_each),
        }

    # ── HELPERS ──────────────────────────────

    def _format_results(self, raw: dict) -> list[dict]:
        """Convert ChromaDB result format into clean list of dicts."""
        if not raw["documents"] or not raw["documents"][0]:
            return []
        results = []
        for doc, meta, dist in zip(
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0]
        ):
            results.append({
                "content":   doc,
                "metadata":  meta,
                "score":     round(1 - dist, 3),  # convert distance → similarity
            })
        # Sort by similarity score descending
        return sorted(results, key=lambda x: x["score"], reverse=True)

    def _chunk_text(self, text: str, chunk_size: int) -> list[str]:
        """Split text into overlapping chunks for better retrieval."""
        words   = text.split()
        chunks  = []
        overlap = chunk_size // 5  # 20% overlap between chunks

        i = 0
        while i < len(words):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            i += chunk_size - overlap

        return chunks or [text]

    def stats(self) -> dict:
        return {
            "conversations": self.conversations.count(),
            "facts":         self.facts.count(),
            "documents":     self.documents.count(),
        }


# ─────────────────────────────────────────────
# FACT EXTRACTOR
# Uses LLM to extract structured facts from conversation turns.
# More accurate than regex (Phase 3.1 approach).
# ─────────────────────────────────────────────

def extract_facts_llm(text: str) -> list[dict]:
    """
    Ask LLM to extract facts from a message.
    Returns list of {"fact": "...", "category": "..."} dicts.
    """
    prompt = [
        {
            "role": "system",
            "content": (
                "Extract factual statements from the user message. "
                "Return ONLY a JSON array. Each item: "
                '{"fact": "complete fact sentence", "category": "personal|technical|preference|project|other"}. '
                "If no facts, return []. No explanation, no markdown."
            )
        },
        {"role": "user", "content": text}
    ]
    raw = call_llm(messages=prompt)

    cleaned = re.sub(r'^```(?:json)?\s*', '', raw.strip())
    cleaned = re.sub(r'\s*```$', '', cleaned).strip()
    try:
        items = json.loads(cleaned)
        return items if isinstance(items, list) else []
    except json.JSONDecodeError:
        return []


# ─────────────────────────────────────────────
# RAG CONTEXT BUILDER
# Builds relevant context from vector memory
# to inject into the LLM prompt before answering
# ─────────────────────────────────────────────

def build_rag_context(query: str, memory: VectorMemory,
                      min_score: float = 0.3) -> str:
    """
    Retrieve relevant memories and format them as context.
    Only includes results above min_score threshold.
    """
    results = memory.search_all(query, n_each=3)
    sections = []

    # Relevant facts
    good_facts = [r for r in results["facts"] if r["score"] >= min_score]
    if good_facts:
        facts_text = "\n".join(
            f"  • {r['content']} (score={r['score']})"
            for r in good_facts
        )
        sections.append(f"RELEVANT FACTS:\n{facts_text}")

    # Relevant past conversations
    good_convs = [r for r in results["conversations"] if r["score"] >= min_score]
    if good_convs:
        convs_text = "\n".join(
            f"  [{r['metadata']['role']}]: {r['content'][:100]}"
            for r in good_convs
        )
        sections.append(f"RELEVANT PAST CONTEXT:\n{convs_text}")

    # Relevant document chunks
    good_docs = [r for r in results["documents"] if r["score"] >= min_score]
    if good_docs:
        docs_text = "\n".join(
            f"  [{r['metadata']['title']}]: {r['content'][:150]}"
            for r in good_docs
        )
        sections.append(f"RELEVANT DOCUMENTS:\n{docs_text}")

    if not sections:
        return ""

    return "--- MEMORY CONTEXT ---\n" + "\n\n".join(sections) + "\n---"


# ─────────────────────────────────────────────
# RAG AGENT
# Combines vector memory retrieval with LLM response
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant with semantic long-term memory.

Before answering, you are given MEMORY CONTEXT — retrieved facts and past conversations relevant to the current question.

RULES:
- Use memory context to personalise and inform your answers
- If memory context answers the question directly, use it
- If memory has no relevant info, answer from your own knowledge
- When using a memory, reference it naturally: "Based on what you've shared..."
- Never hallucinate facts — only state what's in memory or clearly known
"""


def chat_with_rag(session_id: str, memory: VectorMemory):
    """Interactive chat with full vector memory integration."""

    print(f"\n{'='*60}")
    print(f"🧠 RAG Chat — session: {session_id}")
    s = memory.stats()
    print(f"   Memory: {s['conversations']} convs | "
          f"{s['facts']} facts | {s['documents']} docs")
    print("Commands: 'quit', 'stats', 'search <query>', 'ingest'")
    print('='*60)

    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        # ── Commands ──
        if user_input.lower() == "quit":
            break

        if user_input.lower() == "stats":
            s = memory.stats()
            print(f"\n📊 Vector Memory Stats:")
            print(f"   Conversations stored: {s['conversations']}")
            print(f"   Facts stored:         {s['facts']}")
            print(f"   Document chunks:      {s['documents']}")
            continue

        if user_input.lower().startswith("search "):
            query = user_input[7:].strip()
            print(f"\n🔍 Searching memory for: '{query}'")
            results = memory.search_all(query)
            for collection, items in results.items():
                if items:
                    print(f"\n  [{collection.upper()}]")
                    for r in items:
                        print(f"    score={r['score']:.3f} | {r['content'][:100]}")
            continue

        if user_input.lower() == "ingest":
            _ingest_demo_documents(memory)
            continue

        # ── Normal chat with RAG ──

        # 1. Store user message in vector memory
        memory.store_conversation(
            role="user", content=user_input, session_id=session_id
        )

        # 2. Extract and store facts from user message
        facts = extract_facts_llm(user_input)
        for item in facts:
            if isinstance(item, dict) and "fact" in item:
                memory.store_fact(
                    fact=item["fact"],
                    category=item.get("category", "general"),
                    source="conversation"
                )
                print(f"   💾 Stored fact [{item.get('category','?')}]: {item['fact'][:60]}")

        # 3. Build RAG context — retrieve relevant memories
        rag_context = build_rag_context(user_input, memory)

        # 4. Inject RAG context into conversation
        messages_with_context = list(conversation)
        if rag_context:
            messages_with_context.append({
                "role":    "system",
                "content": rag_context
            })
        messages_with_context.append({
            "role": "user", "content": user_input
        })

        # 5. Get LLM response
        reply = call_llm(messages=messages_with_context)

        # 6. Store assistant reply in vector memory
        memory.store_conversation(
            role="assistant", content=reply, session_id=session_id
        )

        # 7. Update short-term conversation context (last 6 turns only)
        conversation.append({"role": "user",      "content": user_input})
        conversation.append({"role": "assistant",  "content": reply})
        if len(conversation) > 13:  # system + 6 turns
            conversation = [conversation[0]] + conversation[-12:]

        print(f"\nAssistant: {reply}")


# ─────────────────────────────────────────────
# DEMO DOCUMENT INGESTION
# Simulates ingesting your own knowledge base
# ─────────────────────────────────────────────

def _ingest_demo_documents(memory: VectorMemory):
    """Ingest sample documents to demonstrate RAG retrieval."""

    docs = [
        {
            "title": "Proxmox Setup Notes",
            "type":  "technical",
            "content": """
Proxmox VE setup notes for homelab:
- Host: Proxmox 8.1 on bare metal with RTX A2000 GPU
- Container 100: ollama-lxc — runs Ollama with GPU passthrough
  - Model: llama3.2, mistral
  - Port: 11434
  - GPU issue resolved: broken /dev/nvidia-uvm device node fixed by
    adding udev rules and restarting the LXC
- Container 101: webserver — nginx + PHP 8.2
- Container 102: kali-linux — for ethical hacking labs
- WebSploit Labs installed with 19 Docker containers
- WireGuard VPN configured for remote access
- Network: 192.168.1.0/24, Proxmox at 192.168.1.10
"""
        },
        {
            "title": "Current Projects",
            "type":  "project",
            "content": """
Active projects:
1. Agentic AI learning roadmap — building agents from scratch with Ollama
2. Knowledge-sharing website for child — Payload CMS + Next.js + Vercel
   - YouTube Data API integration for @hello-magi channel
   - Young Scientist program content
3. Will Sparrow Technologies Pvt. Ltd.
   - Software development, electronics, PCB services
4. AEM multi-tenant project — component development, RTE customisation
   - Tech: Adobe Experience Manager, HTL, Sling Models, SCSS
5. Payroll management system — Laravel + Inertia.js + React + PostgreSQL
   - Indian payroll: EPF, ESI, TDS calculations for FY 2025-26
"""
        },
        {
            "title": "Tech Stack Preferences",
            "type":  "preference",
            "content": """
Preferred technologies:
- Backend:     Laravel (primary), FastAPI (Python services)
- Frontend:    React, Inertia.js
- Database:    PostgreSQL, SQLite, Redis
- DevOps:      Proxmox, Docker, Kubernetes (kubeadm + Flannel)
- Hardware:    Arduino, ESP32 for IoT projects
- AI/ML:       Ollama (local LLMs), LangGraph, ChromaDB
- Languages:   PHP, Python, JavaScript, Go (systems work)
- Learning:    Assembly (x86-64), OS internals, networking fundamentals
"""
        },
    ]

    print("\n📥 Ingesting documents into vector memory...")
    for doc in docs:
        ids = memory.store_document(
            content=doc["content"],
            title=doc["title"],
            doc_type=doc["type"]
        )
        print(f"   ✅ '{doc['title']}' → {len(ids)} chunk(s)")

    print(f"   Total document chunks: {memory.documents.count()}")


# ─────────────────────────────────────────────
# STANDALONE RAG DEMO
# Shows the full retrieve → augment → generate pipeline
# without interactive input
# ─────────────────────────────────────────────

def run_rag_demo(memory: VectorMemory):
    print("\n" + "="*60)
    print("🎬 RAG DEMO — automated pipeline walkthrough")
    print("="*60)

    # Step 1: Ingest documents
    _ingest_demo_documents(memory)

    # Step 2: Store some facts directly
    demo_facts = [
        ("Muni is a full-stack developer based in Krishnagiri, Tamil Nadu", "personal"),
        ("Muni runs Ollama on Proxmox LXC with an RTX A2000 GPU",          "technical"),
        ("Muni is learning agentic AI by building systems from scratch",     "project"),
        ("Muni's child has a YouTube channel at youtube.com/@hello-magi",   "personal"),
        ("Muni prefers PostgreSQL for production databases",                 "preference"),
    ]
    print("\n💾 Storing facts...")
    for fact, category in demo_facts:
        memory.store_fact(fact, category)
        print(f"   [{category}] {fact[:60]}")

    # Step 3: Test queries
    queries = [
        "What GPU does Muni use and which container runs Ollama?",
        "What projects is Muni currently working on?",
        "What is Muni's preferred backend framework?",
        "Tell me about the child's YouTube channel",
        "What ethical hacking tools are set up?",
    ]

    print("\n" + "─"*60)
    print("🔍 Testing RAG retrieval + generation")
    print("─"*60)

    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]

    for query in queries:
        print(f"\n❓ Query: {query}")

        # Retrieve
        rag_context = build_rag_context(query, memory, min_score=0.25)

        if rag_context:
            print("   📎 Context retrieved (top matches):")
            # Show which sources were found
            results = memory.search_all(query, n_each=2)
            for col, items in results.items():
                for r in items:
                    if r["score"] >= 0.25:
                        print(f"      [{col}] score={r['score']:.2f} | "
                              f"{r['content'][:70]}...")
        else:
            print("   ⚠️  No relevant context found")

        # Generate
        messages = list(conversation)
        if rag_context:
            messages.append({"role": "system", "content": rag_context})
        messages.append({"role": "user", "content": query})

        reply = call_llm(messages)
        print(f"\n   🤖 {reply[:250]}{'...' if len(reply) > 250 else ''}")

        # Store in conversation context
        conversation.append({"role": "user",      "content": query})
        conversation.append({"role": "assistant",  "content": reply})
        if len(conversation) > 9:
            conversation = [conversation[0]] + conversation[-8:]

    print("\n\n✅ RAG demo complete!")
    s = memory.stats()
    print(f"📊 Final memory: {s['conversations']} convs | "
          f"{s['facts']} facts | {s['documents']} doc chunks")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("🧠 Phase 3, Step 3.2 — Vector Store Memory (ChromaDB + RAG)\n")

    memory = VectorMemory(persist_dir="./chroma_memory")

    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        run_rag_demo(memory)
    else:
        session_id = sys.argv[1] if len(sys.argv) > 1 else "default"
        chat_with_rag(session_id, memory)