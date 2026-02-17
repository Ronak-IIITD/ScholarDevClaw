from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CacheEntry:
    key: str
    value: Any
    created_at: float
    expires_at: float | None = None
    hits: int = 0


class Cache:
    """In-memory cache with TTL and size limits."""

    def __init__(
        self,
        max_size: int = 100,
        default_ttl: float = 3600.0,
    ):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: dict[str, CacheEntry] = {}

    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from args and kwargs."""
        data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def get(self, key: str) -> Any | None:
        """Get value from cache if exists and not expired."""
        if key not in self._cache:
            return None

        entry = self._cache[key]
        current_time = time.time()

        if entry.expires_at and current_time > entry.expires_at:
            del self._cache[key]
            return None

        entry.hits += 1
        return entry.value

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Set value in cache with optional TTL."""
        current_time = time.time()

        if len(self._cache) >= self.max_size:
            self._evict_lru()

        expires_at = None
        if ttl is None:
            ttl = self.default_ttl

        if ttl > 0:
            expires_at = current_time + ttl

        self._cache[key] = CacheEntry(
            key=key,
            value=value,
            created_at=current_time,
            expires_at=expires_at,
        )

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return

        lru_key = min(self._cache.keys(), key=lambda k: self._cache[k].hits)
        del self._cache[lru_key]

    def invalidate(self, key: str) -> None:
        """Invalidate a cache entry."""
        if key in self._cache:
            del self._cache[key]

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_hits = sum(e.hits for e in self._cache.values())
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "total_hits": total_hits,
        }


class FileCache:
    """File-based cache for persistence across sessions."""

    def __init__(self, cache_dir: Path | None = None):
        if cache_dir:
            self.cache_dir = cache_dir
        else:
            self.cache_dir = Path.home() / ".scholardevclaw" / "cache"

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, key: str, max_age: float = 3600.0) -> Any | None:
        """Get value from file cache if exists and not expired."""
        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            return None

        try:
            mtime = cache_path.stat().st_mtime
            current_time = time.time()

            if current_time - mtime > max_age:
                cache_path.unlink()
                return None

            with open(cache_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def set(self, key: str, value: Any) -> None:
        """Set value in file cache."""
        cache_path = self._get_cache_path(key)

        try:
            with open(cache_path, "w") as f:
                json.dump(value, f, default=str)
        except OSError:
            pass

    def invalidate(self, key: str) -> None:
        """Invalidate a cache entry."""
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            cache_path.unlink()

    def clear(self) -> None:
        """Clear all file cache entries."""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()


def cached(
    cache: Cache,
    ttl: float = 3600.0,
    key_func: callable | None = None,
):
    """Decorator for caching function results."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{cache._generate_key(*args, **kwargs)}"

            cached_value = cache.get(cache_key)
            if cached_value is not None:
                return cached_value

            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            return result

        return wrapper

    return decorator


class ResultCache:
    """Specialized cache for analysis results with repo hashing."""

    def __init__(self, cache_dir: Path | None = None):
        self.file_cache = FileCache(cache_dir)
        self.memory_cache = Cache(max_size=50, default_ttl=1800.0)

    def _hash_repo(self, repo_path: str) -> str:
        """Generate stable hash for repository."""
        path = Path(repo_path).resolve()
        data = f"{path}:{path.stat().st_mtime}"
        return hashlib.sha256(data.encode()).hexdigest()[:12]

    def get_analysis(self, repo_path: str) -> Any | None:
        """Get cached analysis for repository."""
        cache_key = f"analysis:{self._hash_repo(repo_path)}"

        cached = self.memory_cache.get(cache_key)
        if cached is not None:
            return cached

        cached = self.file_cache.get(cache_key, max_age=1800.0)
        if cached is not None:
            self.memory_cache.set(cache_key, cached)
            return cached

        return None

    def set_analysis(self, repo_path: str, result: Any) -> None:
        """Cache analysis result for repository."""
        cache_key = f"analysis:{self._hash_repo(repo_path)}"

        self.memory_cache.set(cache_key, result)
        self.file_cache.set(cache_key, result)

    def invalidate_analysis(self, repo_path: str) -> None:
        """Invalidate cached analysis for repository."""
        cache_key = f"analysis:{self._hash_repo(repo_path)}"

        self.memory_cache.invalidate(cache_key)
        self.file_cache.invalidate(cache_key)
