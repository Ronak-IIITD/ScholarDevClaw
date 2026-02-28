"""
Agent memory system for long-term context management.

Provides:
- Episodic memory (past interactions)
- Semantic memory (learned facts)
- Working memory (current context)
- Memory retrieval and relevance scoring
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any


class MemoryType(Enum):
    """Types of memory"""

    EPISODIC = "episodic"  # Past interactions
    SEMANTIC = "semantic"  # Learned facts
    WORKING = "working"  # Current context
    PROCEDURAL = "procedural"  # How to do things


class MemoryImportance(Enum):
    """Memory importance levels"""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


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
    access_count: int = 0
    last_accessed: str = ""


@dataclass
class MemoryRetrieval:
    """Retrieved memory with relevance score"""

    memory: Memory
    relevance: float
    recency: float


class AgentMemory:
    """Agent memory system"""

    def __init__(self, agent_id: str, store_dir: Path | None = None):
        self.agent_id = agent_id
        self.store_dir = store_dir or Path.home() / ".scholardevclaw" / "memory" / agent_id
        self.store_dir.mkdir(parents=True, exist_ok=True)

        self.memories: dict[str, Memory] = {}
        self._load()

    def _load(self):
        """Load memories from disk"""
        for mem_file in self.store_dir.glob("*.json"):
            try:
                data = json.loads(mem_file.read_text())
                self.memories[data["id"]] = Memory(**data)
            except Exception:
                pass

    def _save(self, memory: Memory):
        """Save memory to disk"""
        (self.store_dir / f"{memory.id}.json").write_text(json.dumps(memory.__dict__, default=str))

    def add(
        self,
        content: str,
        memory_type: MemoryType,
        importance: MemoryImportance = MemoryImportance.MEDIUM,
        metadata: dict | None = None,
    ) -> Memory:
        """Add a new memory"""
        memory = Memory(
            id=str(uuid.uuid4()),
            content=content,
            memory_type=memory_type,
            importance=importance,
            metadata=metadata or {},
        )

        self.memories[memory.id] = memory
        self._save(memory)

        return memory

    def retrieve(
        self,
        query: str,
        memory_type: MemoryType | None = None,
        limit: int = 10,
    ) -> list[MemoryRetrieval]:
        """Retrieve relevant memories"""
        results = []

        for memory in self.memories.values():
            if memory_type and memory.memory_type != memory_type:
                continue

            relevance = self._calculate_relevance(query, memory.content)
            recency = self._calculate_recency(memory.created_at)

            combined = (relevance * 0.6) + (recency * 0.3) + (memory.importance.value / 4 * 0.1)

            results.append(
                MemoryRetrieval(
                    memory=memory,
                    relevance=combined,
                    recency=recency,
                )
            )

        results.sort(key=lambda r: r.relevance, reverse=True)
        return results[:limit]

    def _calculate_relevance(self, query: str, content: str) -> float:
        """Calculate keyword-based relevance"""
        query_terms = set(query.lower().split())
        content_terms = set(content.lower().split())

        if not query_terms:
            return 0.0

        overlap = len(query_terms & content_terms)
        return min(1.0, overlap / len(query_terms))

    def _calculate_recency(self, created_at: str) -> float:
        """Calculate recency score (0-1)"""
        try:
            created = datetime.fromisoformat(created_at)
            age = datetime.now() - created

            if age < timedelta(hours=1):
                return 1.0
            elif age < timedelta(days=1):
                return 0.8
            elif age < timedelta(days=7):
                return 0.6
            elif age < timedelta(days=30):
                return 0.4
            else:
                return 0.2
        except Exception:
            return 0.5

    def get_context(self, max_memories: int = 5) -> str:
        """Get contextual memory summary"""
        recent = self.retrieve("", limit=max_memories)

        context_parts = []
        for r in recent:
            context_parts.append(f"- {r.memory.content[:200]}")

        return "\n".join(context_parts) if context_parts else "No relevant memories found."

    def remember_episode(self, content: str, outcome: str = "") -> Memory:
        """Store an episodic memory from interaction"""
        return self.add(
            content=content,
            memory_type=MemoryType.EPISODIC,
            importance=MemoryImportance.MEDIUM,
            metadata={"outcome": outcome},
        )

    def learn_fact(self, fact: str, source: str = "") -> Memory:
        """Store a semantic memory (learned fact)"""
        return self.add(
            content=fact,
            memory_type=MemoryType.SEMANTIC,
            importance=MemoryImportance.HIGH,
            metadata={"source": source},
        )

    def update_working_memory(self, content: str) -> Memory:
        """Update current working memory"""
        working = [m for m in self.memories.values() if m.memory_type == MemoryType.WORKING]

        if working:
            working[0].content = content
            working[0].created_at = datetime.now().isoformat()
            self._save(working[0])
            return working[0]

        return self.add(
            content=content,
            memory_type=MemoryType.WORKING,
            importance=MemoryImportance.CRITICAL,
        )

    def forget_old(self, days: int = 30):
        """Remove old low-importance memories"""
        to_remove = []

        for memory in self.memories.values():
            if memory.importance == MemoryImportance.LOW:
                age = datetime.now() - datetime.fromisoformat(memory.created_at)
                if age > timedelta(days=days):
                    to_remove.append(memory.id)

        for mem_id in to_remove:
            del self.memories[mem_id]
            (self.store_dir / f"{mem_id}.json").unlink(missing_ok=True)

    def get_stats(self) -> dict[str, Any]:
        """Get memory statistics"""
        by_type = {}
        for mem in self.memories.values():
            type_name = mem.memory_type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1

        return {
            "total_memories": len(self.memories),
            "by_type": by_type,
            "agent_id": self.agent_id,
        }
