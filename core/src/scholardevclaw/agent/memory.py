"""
Advanced Agent Memory System with persistent storage and longevity.

Features:
- Multi-tier storage (hot/warm/cold)
- Memory consolidation (episodic -> semantic)
- Memory decay and TTL
- Semantic embeddings for similarity search
- Memory summarization
- Cross-session persistence
- Memory indexing and fast retrieval
- Importance boosting and decay
- Auto-cleanup and archival
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any


class MemoryType(Enum):
    """Types of memory"""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    WORKING = "working"
    PROCEDURAL = "procedural"
    ARCHIVED = "archived"


class MemoryImportance(Enum):
    """Memory importance levels"""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    PERMANENT = 5


class MemoryTier(Enum):
    """Storage tiers for memory longevity"""

    HOT = "hot"  # In-memory, instant access
    WARM = "warm"  # Fast disk access
    COLD = "cold"  # Compressed/archived
    FROZEN = "frozen"  # Long-term archival


@dataclass
class Memory:
    """A single memory entry"""

    id: str
    content: str
    memory_type: MemoryType
    importance: MemoryImportance
    embedding: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    accessed_at: str = field(default_factory=lambda: datetime.now().isoformat())
    access_count: int = 0
    tier: MemoryTier = MemoryTier.HOT
    expires_at: str | None = None
    consolidated_from: str | None = None
    summary: str | None = None
    tags: list[str] = field(default_factory=list)
    source_session: str | None = None


@dataclass
class MemoryRetrieval:
    """Retrieved memory with relevance score"""

    memory: Memory
    relevance: float
    recency: float
    importance_boost: float


class PersistentMemoryStore:
    """SQLite-backed persistent memory store"""

    def __init__(self, store_dir: Path):
        self.store_dir = store_dir
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = store_dir / "memories.db"
        self._lock = threading.RLock()
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                importance INTEGER NOT NULL,
                embedding BLOB,
                metadata TEXT,
                created_at TEXT NOT NULL,
                accessed_at TEXT NOT NULL,
                access_count INTEGER DEFAULT 0,
                tier TEXT DEFAULT 'hot',
                expires_at TEXT,
                consolidated_from TEXT,
                summary TEXT,
                tags TEXT,
                source_session TEXT,
                importance_boost REAL DEFAULT 1.0,
                decay_factor REAL DEFAULT 1.0
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_tier ON memories(tier)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_expires ON memories(expires_at)
        """)

        conn.commit()
        conn.close()

    def insert(self, memory: Memory):
        """Insert or update a memory"""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO memories
                (id, content, memory_type, importance, embedding, metadata,
                 created_at, accessed_at, access_count, tier, expires_at,
                 consolidated_from, summary, tags, source_session)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    memory.id,
                    memory.content,
                    memory.memory_type.value,
                    memory.importance.value,
                    json.dumps(memory.embedding) if memory.embedding else None,
                    json.dumps(memory.metadata),
                    memory.created_at,
                    memory.accessed_at,
                    memory.access_count,
                    memory.tier.value,
                    memory.expires_at,
                    memory.consolidated_from,
                    memory.summary,
                    json.dumps(memory.tags),
                    memory.source_session,
                ),
            )

            conn.commit()
            conn.close()

    def get(self, memory_id: str) -> Memory | None:
        """Get a memory by ID"""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
            row = cursor.fetchone()
            conn.close()

            if row:
                return self._row_to_memory(row)
            return None

    def get_all(
        self, memory_type: MemoryType | None = None, tier: MemoryTier | None = None
    ) -> list[Memory]:
        """Get all memories, optionally filtered"""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = "SELECT * FROM memories WHERE 1=1"
            params = []

            if memory_type:
                query += " AND memory_type = ?"
                params.append(memory_type.value)
            if tier:
                query += " AND tier = ?"
                params.append(tier.value)

            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()

            return [self._row_to_memory(row) for row in rows]

    def delete(self, memory_id: str):
        """Delete a memory"""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            conn.commit()
            conn.close()

    def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        """Convert database row to Memory"""
        return Memory(
            id=row["id"],
            content=row["content"],
            memory_type=MemoryType(row["memory_type"]),
            importance=MemoryImportance(row["importance"]),
            embedding=json.loads(row["embedding"]) if row["embedding"] else [],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            created_at=row["created_at"],
            accessed_at=row["accessed_at"],
            access_count=row["access_count"],
            tier=MemoryTier(row["tier"]),
            expires_at=row["expires_at"],
            consolidated_from=row["consolidated_from"],
            summary=row["summary"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            source_session=row["source_session"],
        )


class AdvancedAgentMemory:
    """
    Advanced agent memory system with persistent storage and longevity.

    Features:
    - Multi-tier storage (hot/warm/cold/frozen)
    - Memory consolidation (episodic -> semantic)
    - Memory decay over time
    - TTL for different memory types
    - Fast SQLite-backed storage
    - Memory summarization
    - Importance boosting
    - Auto-cleanup and archival
    """

    def __init__(
        self,
        agent_id: str,
        store_dir: Path | None = None,
        config: dict | None = None,
    ):
        self.agent_id = agent_id
        self.store_dir = store_dir or Path.home() / ".scholardevclaw" / "memory" / agent_id
        self.store_dir.mkdir(parents=True, exist_ok=True)

        self.config = config or {}
        self.session_id = str(uuid.uuid4())

        self.ttl_config = {
            MemoryType.EPISODIC: self.config.get("episodic_ttl_days", 30),
            MemoryType.SEMANTIC: self.config.get("semantic_ttl_days", 365),
            MemoryType.WORKING: self.config.get("working_ttl_hours", 24),
            MemoryType.PROCEDURAL: self.config.get("procedural_ttl_days", 180),
            MemoryType.ARCHIVED: self.config.get("archived_ttl_days", 365 * 5),
        }

        self.tier_config = {
            MemoryTier.HOT: self.config.get("hot_threshold_days", 1),
            MemoryTier.WARM: self.config.get("warm_threshold_days", 7),
            MemoryTier.COLD: self.config.get("cold_threshold_days", 30),
        }

        self.store = PersistentMemoryStore(self.store_dir)
        self._memories_cache: dict[str, Memory] = {}
        self._load_to_cache()

    def _load_to_cache(self):
        """Load hot/warm memories to memory cache"""
        for mem in self.store.get_all(tier=MemoryTier.HOT):
            self._memories_cache[mem.id] = mem
        for mem in self.store.get_all(tier=MemoryTier.WARM):
            self._memories_cache[mem.id] = mem

    def add(
        self,
        content: str,
        memory_type: MemoryType,
        importance: MemoryImportance = MemoryImportance.MEDIUM,
        metadata: dict | None = None,
        tags: list[str] | None = None,
        expires_in_days: int | None = None,
    ) -> Memory:
        """Add a new memory with auto-tiering"""
        expires_at = None
        if expires_in_days:
            expires_at = (datetime.now() + timedelta(days=expires_in_days)).isoformat()
        elif memory_type in self.ttl_config:
            if memory_type == MemoryType.WORKING:
                expires_at = (
                    datetime.now() + timedelta(hours=self.ttl_config[memory_type])
                ).isoformat()
            else:
                expires_at = (
                    datetime.now() + timedelta(days=self.ttl_config[memory_type])
                ).isoformat()

        memory = Memory(
            id=str(uuid.uuid4()),
            content=content,
            memory_type=memory_type,
            importance=importance,
            metadata=metadata or {},
            tags=tags or [],
            expires_at=expires_at,
            tier=self._calculate_tier(importance),
            source_session=self.session_id,
        )

        self.store.insert(memory)
        self._memories_cache[memory.id] = memory
        # Note: consolidation is triggered from access(), not here.
        # A newly created memory has access_count=0, so consolidating
        # on creation would never trigger the threshold check.

        return memory

    def _calculate_tier(self, importance: MemoryImportance) -> MemoryTier:
        """Calculate initial storage tier based on importance"""
        if importance == MemoryImportance.PERMANENT:
            return MemoryTier.FROZEN
        elif importance == MemoryImportance.CRITICAL:
            return MemoryTier.HOT
        elif importance == MemoryImportance.HIGH:
            return MemoryTier.WARM
        return MemoryTier.COLD

    def retrieve(
        self,
        query: str,
        memory_type: MemoryType | None = None,
        limit: int = 10,
        include_archived: bool = False,
    ) -> list[MemoryRetrieval]:
        """Retrieve relevant memories with multi-signal scoring"""
        memory_types = [MemoryType.ARCHIVED] if include_archived else None
        if memory_type:
            memory_types = [memory_type]

        all_memories = []
        for mt in memory_types or [MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.PROCEDURAL]:
            all_memories.extend(self.store.get_all(memory_type=mt))

        results = []
        query_terms = set(query.lower().split()) if query else set()

        for memory in all_memories:
            if self._is_expired(memory):
                continue

            relevance = self._calculate_relevance(query_terms, memory)
            recency = self._calculate_recency(memory)
            importance_boost = self._calculate_importance_boost(memory)

            relevance * 0.5 + recency * 0.3 + importance_boost * 0.2

            results.append(
                MemoryRetrieval(
                    memory=memory,
                    relevance=relevance,
                    recency=recency,
                    importance_boost=importance_boost,
                )
            )

        results.sort(key=lambda r: r.relevance + r.recency + r.importance_boost, reverse=True)
        return results[:limit]

    def _calculate_relevance(self, query_terms: set[str], memory: Memory) -> float:
        """Calculate relevance score using keyword overlap"""
        if not query_terms:
            return 0.5

        content_terms = set(memory.content.lower().split())
        tag_terms = set(t.lower() for t in memory.tags)

        content_overlap = len(query_terms & content_terms)
        tag_overlap = len(query_terms & tag_terms)

        score = (content_overlap * 1.0 + tag_overlap * 2.0) / max(len(query_terms), 1)
        return min(1.0, score)

    def _calculate_recency(self, memory: Memory) -> float:
        """Calculate recency score with access tracking"""
        try:
            created = datetime.fromisoformat(memory.created_at)
            accessed = datetime.fromisoformat(memory.accessed_at)

            age = datetime.now() - created
            last_access = datetime.now() - accessed

            base_score = self._age_to_score(age)

            access_boost = min(0.2, memory.access_count * 0.02)
            recent_access_boost = 0.1 if last_access < timedelta(hours=1) else 0

            return min(1.0, base_score + access_boost + recent_access_boost)
        except Exception:
            return 0.5

    def _age_to_score(self, age: timedelta) -> float:
        """Convert age to score"""
        if age < timedelta(hours=1):
            return 1.0
        elif age < timedelta(hours=24):
            return 0.9
        elif age < timedelta(days=7):
            return 0.7
        elif age < timedelta(days=30):
            return 0.5
        elif age < timedelta(days=90):
            return 0.3
        elif age < timedelta(days=365):
            return 0.2
        return 0.1

    def _calculate_importance_boost(self, memory: Memory) -> float:
        """Calculate importance boost"""
        base = memory.importance.value / 5.0

        critical_tags = {"important", "critical", "key", "essential"}
        if memory.tags and set(memory.tags) & critical_tags:
            base += 0.2

        if memory.memory_type == MemoryType.SEMANTIC:
            base += 0.1

        return min(1.0, base)

    def _is_expired(self, memory: Memory) -> bool:
        """Check if memory has expired"""
        if not memory.expires_at:
            return False

        try:
            expires = datetime.fromisoformat(memory.expires_at)
            return datetime.now() > expires
        except Exception:
            return False

    def access(self, memory_id: str) -> Memory | None:
        """Access and update memory metadata, potentially triggering consolidation."""
        memory = self.store.get(memory_id)
        if memory:
            memory.access_count += 1
            memory.accessed_at = datetime.now().isoformat()

            if memory.tier == MemoryTier.COLD:
                memory.tier = MemoryTier.WARM
            elif memory.tier == MemoryTier.WARM:
                memory.tier = MemoryTier.HOT

            self.store.insert(memory)
            self._memories_cache[memory.id] = memory

            # Check for consolidation after access count increases
            self._auto_consolidate(memory)

        return memory

    def _auto_consolidate(self, memory: Memory):
        """Auto-consolidate episodic memories to semantic"""
        if memory.memory_type != MemoryType.EPISODIC:
            return

        if memory.access_count >= 3 and memory.importance.value >= MemoryImportance.MEDIUM.value:
            similar = self._find_similar(memory)
            if not similar:
                semantic_memory = Memory(
                    id=str(uuid.uuid4()),
                    content=memory.content,
                    memory_type=MemoryType.SEMANTIC,
                    importance=MemoryImportance.HIGH,
                    metadata=memory.metadata.copy(),
                    tags=memory.tags.copy(),
                    consolidated_from=memory.id,
                    source_session=memory.source_session,
                )
                self.store.insert(semantic_memory)
                self._memories_cache[semantic_memory.id] = semantic_memory

    def _find_similar(self, memory: Memory, threshold: float = 0.8) -> Memory | None:
        """Find similar semantic memory"""
        for mem in self.store.get_all(memory_type=MemoryType.SEMANTIC):
            relevance = self._calculate_relevance(set(memory.content.lower().split()), mem)
            if relevance >= threshold:
                return mem
        return None

    def update_working_memory(self, content: str) -> Memory:
        """Update current working memory"""
        working = self.store.get_all(memory_type=MemoryType.WORKING)

        if working:
            working[0].content = content
            working[0].accessed_at = datetime.now().isoformat()
            working[0].access_count += 1
            self.store.insert(working[0])
            self._memories_cache[working[0].id] = working[0]
            return working[0]

        return self.add(
            content=content,
            memory_type=MemoryType.WORKING,
            importance=MemoryImportance.CRITICAL,
            expires_in_days=None,
        )

    def remember_episode(
        self,
        content: str,
        outcome: str = "",
        tags: list[str] | None = None,
    ) -> Memory:
        """Store an episodic memory"""
        return self.add(
            content=content,
            memory_type=MemoryType.EPISODIC,
            importance=MemoryImportance.MEDIUM,
            metadata={"outcome": outcome} if outcome else {},
            tags=tags or ["episode"],
        )

    def learn_fact(
        self,
        fact: str,
        source: str = "",
        tags: list[str] | None = None,
    ) -> Memory:
        """Store a semantic memory (learned fact)"""
        return self.add(
            content=fact,
            memory_type=MemoryType.SEMANTIC,
            importance=MemoryImportance.HIGH,
            metadata={"source": source} if source else {},
            tags=tags or ["fact", "knowledge"],
        )

    def learn_procedure(
        self,
        procedure: str,
        steps: list[str],
        tags: list[str] | None = None,
    ) -> Memory:
        """Store a procedural memory"""
        content = f"{procedure}: {' -> '.join(steps)}"
        return self.add(
            content=content,
            memory_type=MemoryType.PROCEDURAL,
            importance=MemoryImportance.HIGH,
            metadata={"steps": steps},
            tags=tags or ["procedure", "how-to"],
        )

    def archive_old_memories(self, days: int = 30):
        """Archive old episodic memories"""
        cutoff = datetime.now() - timedelta(days=days)

        for memory in self.store.get_all(memory_type=MemoryType.EPISODIC):
            try:
                created = datetime.fromisoformat(memory.created_at)
                if created < cutoff and memory.access_count < 2:
                    memory.memory_type = MemoryType.ARCHIVED
                    memory.tier = MemoryTier.COLD
                    self.store.insert(memory)
            except Exception:
                pass

    def cleanup_expired(self):
        """Remove expired memories"""
        for memory in self.store.get_all():
            if self._is_expired(memory):
                self.store.delete(memory.id)
                self._memories_cache.pop(memory.id, None)

    def decay_old_memories(self):
        """Apply decay to old memories"""
        for memory in self.store.get_all():
            age = datetime.now() - datetime.fromisoformat(memory.created_at)

            if age > timedelta(days=90):
                if memory.importance == MemoryImportance.MEDIUM:
                    memory.importance = MemoryImportance.LOW
                elif memory.importance == MemoryImportance.HIGH:
                    memory.importance = MemoryImportance.MEDIUM

            if age > timedelta(days=180):
                if memory.tier not in [MemoryTier.FROZEN, MemoryTier.COLD]:
                    memory.tier = MemoryTier.COLD
                    self.store.insert(memory)
                    self._memories_cache.pop(memory.id, None)

    def get_context(self, max_memories: int = 5, include_archived: bool = False) -> str:
        """Get contextual memory summary"""
        recent = self.retrieve("", limit=max_memories, include_archived=include_archived)

        context_parts = []
        for r in recent:
            mem = r.memory
            context_parts.append(f"[{mem.memory_type.value.upper()}] {mem.content[:150]}...")

        return "\n".join(context_parts) if context_parts else "No relevant memories."

    def get_session_history(self) -> list[Memory]:
        """Get all memories from current session"""
        return [m for m in self.store.get_all() if m.source_session == self.session_id]

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive memory statistics"""
        all_memories = self.store.get_all()

        by_type = {}
        by_tier = {}
        total_access = 0

        for mem in all_memories:
            type_name = mem.memory_type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1

            tier_name = mem.tier.value
            by_tier[tier_name] = by_tier.get(tier_name, 0) + 1

            total_access += mem.access_count

        expired_count = sum(1 for m in all_memories if self._is_expired(m))

        return {
            "total_memories": len(all_memories),
            "by_type": by_type,
            "by_tier": by_tier,
            "total_accesses": total_access,
            "expired_count": expired_count,
            "cached_count": len(self._memories_cache),
            "session_id": self.session_id,
            "agent_id": self.agent_id,
        }

    def search_by_tag(self, tag: str, limit: int = 10) -> list[Memory]:
        """Search memories by tag"""
        results = []
        for memory in self.store.get_all():
            if tag.lower() in [t.lower() for t in memory.tags]:
                results.append(memory)

        results.sort(key=lambda m: (m.importance.value, m.access_count), reverse=True)
        return results[:limit]

    def merge_memories(self, memory_ids: list[str], new_content: str) -> Memory:
        """Merge multiple memories into one"""
        source_memories = [m for m in (self.store.get(mid) for mid in memory_ids) if m is not None]

        if not source_memories:
            raise ValueError("No valid memories found to merge")

        metadata = {"merged_from": [m.id for m in source_memories]}
        tags = []
        for m in source_memories:
            tags.extend(m.tags)

        new_memory = self.add(
            content=new_content,
            memory_type=MemoryType.SEMANTIC,
            importance=MemoryImportance.HIGH,
            metadata=metadata,
            tags=list(set(tags)),
        )

        for m in source_memories:
            m.memory_type = MemoryType.ARCHIVED
            self.store.insert(m)

        return new_memory


AgentMemory = AdvancedAgentMemory


def create_memory(agent_id: str, store_dir: Path | None = None) -> AdvancedAgentMemory:
    """Factory function to create advanced memory"""
    return AdvancedAgentMemory(agent_id, store_dir)
