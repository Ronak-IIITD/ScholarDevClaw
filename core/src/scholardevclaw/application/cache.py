"""Pipeline stage caching for ScholarDevClaw.

Provides persistent caching for expensive pipeline stages:
- Repository analysis (tree-sitter parsing)
- Research search results
- Mapping results
- Patch generation

Cache is stored in ~/.cache/scholardevclaw/ with content-addressable keys.
"""

from __future__ import annotations

import hashlib
import pickle
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import orjson

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "scholardevclaw"
CACHE_VERSION = 2


@dataclass
class CacheEntry:
    """A single cache entry with metadata."""

    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    expires_at: float | None = None
    metadata: dict = field(default_factory=dict)

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "value": self.value,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CacheEntry:
        return cls(
            key=data["key"],
            value=data["value"],
            created_at=data.get("created_at", time.time()),
            expires_at=data.get("expires_at"),
            metadata=data.get("metadata", {}),
        )


class PipelineCache:
    """Content-addressable cache for pipeline stages."""

    def __init__(self, cache_dir: Path | None = None, ttl_seconds: int = 86400 * 7):
        """
        Args:
            cache_dir: Cache directory (default: ~/.cache/scholardevclaw)
            ttl_seconds: Default TTL in seconds (default: 7 days)
        """
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.ttl_seconds = ttl_seconds
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: dict[str, CacheEntry] = {}
        self._load_index()

    def _load_index(self) -> None:
        """Load cache index from disk (uses orjson for faster deserialization)."""
        index_path = self.cache_dir / "index.json"
        if not index_path.exists():
            return
        try:
            with open(index_path, "rb") as f:
                index = orjson.loads(f.read())
            for key, entry_data in index.items():
                entry = CacheEntry.from_dict(entry_data)
                if not entry.is_expired():
                    self._memory_cache[key] = entry
        except Exception:
            # Corrupted index - start fresh
            self._memory_cache = {}

    def _save_index(self) -> None:
        """Save cache index to disk (uses orjson for 5-10x faster serialization)."""
        index_path = self.cache_dir / "index.json"
        try:
            index = {key: entry.to_dict() for key, entry in self._memory_cache.items()}
            with open(index_path, "wb") as f:
                f.write(orjson.dumps(index))
        except Exception:
            pass  # Best effort

    def _make_key(self, stage: str, *args: Any, **kwargs: Any) -> str:
        """Create a content-addressable cache key."""
        key_parts = orjson.dumps(
            {"stage": stage, "args": args, "kwargs": kwargs},
            option=orjson.OPT_SORT_KEYS,
        )
        return hashlib.sha256(key_parts).hexdigest()[:32]

    def _get_cache_path(self, key: str) -> Path:
        """Get the file path for a cache key."""
        return self.cache_dir / f"{key}.cache"

    def get(self, stage: str, *args: Any, **kwargs: Any) -> Any | None:
        """Get a cached value."""
        key = self._make_key(stage, *args, **kwargs)

        # Check memory cache first
        if key in self._memory_cache:
            entry = self._memory_cache[key]
            if not entry.is_expired():
                return entry.value
            else:
                del self._memory_cache[key]

        # Check disk cache
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    entry = pickle.load(f)
                if not entry.is_expired():
                    self._memory_cache[key] = entry
                    return entry.value
                else:
                    cache_path.unlink(missing_ok=True)
            except Exception:
                cache_path.unlink(missing_ok=True)

        return None

    def set(
        self,
        stage: str,
        value: Any,
        *args: Any,
        ttl: int | None = None,
        metadata: dict | None = None,
        **kwargs: Any,
    ) -> None:
        """Set a cached value."""
        key = self._make_key(stage, *args, **kwargs)
        expires_at = time.time() + (ttl or self.ttl_seconds)

        entry = CacheEntry(
            key=key,
            value=value,
            expires_at=expires_at,
            metadata=metadata or {},
        )

        self._memory_cache[key] = entry

        # Write to disk
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(entry, f)
        except Exception:
            pass

        self._save_index()

    def invalidate(self, stage: str, *args: Any, **kwargs: Any) -> bool:
        """Invalidate a specific cache entry."""
        key = self._make_key(stage, *args, **kwargs)
        removed = False

        if key in self._memory_cache:
            del self._memory_cache[key]
            removed = True

        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            cache_path.unlink(missing_ok=True)
            removed = True

        if removed:
            self._save_index()

        return removed

    def invalidate_stage(self, stage: str) -> int:
        """Invalidate all entries for a stage."""
        keys_to_remove = [k for k in self._memory_cache if k.startswith(f"{stage}:")]
        for key in keys_to_remove:
            del self._memory_cache[key]
            cache_path = self._get_cache_path(key)
            cache_path.unlink(missing_ok=True)

        if keys_to_remove:
            self._save_index()

        return len(keys_to_remove)

    def clear(self) -> None:
        """Clear all cache entries."""
        self._memory_cache.clear()
        for cache_file in self.cache_dir.glob("*.cache"):
            cache_file.unlink(missing_ok=True)
        (self.cache_dir / "index.json").unlink(missing_ok=True)

    def stats(self) -> dict:
        """Get cache statistics."""
        total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.cache"))
        return {
            "entries": len(self._memory_cache),
            "total_size_bytes": total_size,
            "cache_dir": str(self.cache_dir),
        }


# Global cache instance
_global_cache: PipelineCache | None = None


def get_pipeline_cache() -> PipelineCache:
    """Get the global pipeline cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = PipelineCache()
    return _global_cache


def cached_stage(stage: str, ttl: int | None = None, key_args: Callable | None = None):
    """
    Decorator to cache a pipeline stage function.

    Args:
        stage: Stage name for cache namespace
        ttl: Time-to-live in seconds
        key_args: Optional function to extract cache key from args
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            cache = get_pipeline_cache()

            # Build cache key
            if key_args:
                cache_key_args = key_args(*args, **kwargs)
            else:
                cache_key_args = args[1:] if args else ()  # Skip self/cls

            cached = cache.get(stage, *cache_key_args, **kwargs)
            if cached is not None:
                return cached

            result = func(*args, **kwargs)
            cache.set(stage, result, *cache_key_args, ttl=ttl, **kwargs)
            return result

        return wrapper

    return decorator


def invalidate_stage(stage: str, *args: Any, **kwargs: Any) -> bool:
    """Invalidate a cached stage."""
    return get_pipeline_cache().invalidate(stage, *args, **kwargs)


def invalidate_all_stages(stage: str) -> int:
    """Invalidate all entries for a stage."""
    return get_pipeline_cache().invalidate_stage(stage)


def clear_pipeline_cache() -> None:
    """Clear the entire pipeline cache."""
    get_pipeline_cache().clear()


def get_cache_stats() -> dict:
    """Get cache statistics."""
    return get_pipeline_cache().stats()
