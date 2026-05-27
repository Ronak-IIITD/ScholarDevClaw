"""Tests for utils/cache.py"""

import time
from pathlib import Path

from scholardevclaw.utils.cache import Cache, FileCache, ResultCache, cached


class TestCache:
    def test_get_missing_key(self):
        c = Cache(max_size=10)
        assert c.get("nonexistent") is None

    def test_set_and_get(self):
        c = Cache(max_size=10)
        c.set("key1", "value1")
        assert c.get("key1") == "value1"

    def test_get_expired_entry(self):
        c = Cache(max_size=10, default_ttl=0.01)
        c.set("key1", "value1", ttl=0.01)
        time.sleep(0.02)
        assert c.get("key1") is None

    def test_set_with_custom_ttl(self):
        c = Cache(max_size=10)
        c.set("key1", "value1", ttl=60.0)
        assert c.get("key1") == "value1"

    def test_set_with_negative_ttl_no_expiry(self):
        c = Cache(max_size=10)
        c.set("key1", "value1", ttl=-1)
        assert c.get("key1") == "value1"

    def test_evict_lru_on_overflow(self):
        c = Cache(max_size=2)
        c.set("a", 1)
        c.set("b", 2)
        c.set("c", 3)
        assert c.get("a") is None
        assert c.get("b") is not None or c.get("c") is not None

    def test_invalidate(self):
        c = Cache(max_size=10)
        c.set("key1", "value1")
        c.invalidate("key1")
        assert c.get("key1") is None

    def test_clear(self):
        c = Cache(max_size=10)
        c.set("a", 1)
        c.set("b", 2)
        c.clear()
        assert c.get("a") is None
        assert c.get("b") is None

    def test_stats(self):
        c = Cache(max_size=100)
        c.set("a", 1)
        c.get("a")
        c.get("a")
        stats = c.stats()
        assert stats["size"] == 1
        assert stats["max_size"] == 100
        assert stats["total_hits"] == 2

    def test_stats_empty(self):
        c = Cache(max_size=50)
        stats = c.stats()
        assert stats["size"] == 0
        assert stats["max_size"] == 50
        assert stats["total_hits"] == 0

    def test_evict_lru_empty_cache(self):
        c = Cache(max_size=10)
        c._evict_lru()
        assert c.stats()["size"] == 0

    def test_invalidate_nonexistent(self):
        c = Cache(max_size=10)
        c.invalidate("nonexistent")
        assert c.stats()["size"] == 0


class TestFileCache:
    def test_set_and_get(self, tmp_path):
        fc = FileCache(tmp_path)
        fc.set("test_key", {"data": 123})
        assert fc.get("test_key") == {"data": 123}

    def test_get_expired(self, tmp_path):
        fc = FileCache(tmp_path)
        fc.set("test_key", "value")
        assert fc.get("test_key", max_age=0) is None

    def test_get_nonexistent(self, tmp_path):
        fc = FileCache(tmp_path)
        assert fc.get("nonexistent") is None

    def test_invalidate(self, tmp_path):
        fc = FileCache(tmp_path)
        fc.set("test_key", "value")
        fc.invalidate("test_key")
        assert fc.get("test_key") is None

    def test_clear(self, tmp_path):
        fc = FileCache(tmp_path)
        fc.set("a", 1)
        fc.set("b", 2)
        fc.clear()
        assert fc.get("a") is None
        assert fc.get("b") is None

    def test_cache_dir_created(self, tmp_path):
        sub = tmp_path / "cache_dir"
        FileCache(sub)
        assert sub.exists()

    def test_default_cache_dir(self, monkeypatch):
        tmp = Path("/tmp/test_cache_home")
        monkeypatch.setattr(Path, "home", lambda: tmp)
        fc = FileCache()
        assert "scholardevclaw" in str(fc.cache_dir)


class TestCachedDecorator:
    def test_caches_result(self):
        cache = Cache(max_size=10)
        call_count = 0

        @cached(cache, ttl=60.0)
        def compute(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        assert compute(5) == 10
        assert call_count == 1
        assert compute(5) == 10
        assert call_count == 1

    def test_different_args_different_cache(self):
        cache = Cache(max_size=10)
        call_count = 0

        @cached(cache, ttl=60.0)
        def compute(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        assert compute(5) == 10
        assert compute(10) == 20
        assert call_count == 2

    def test_custom_key_func(self):
        cache = Cache(max_size=10)
        call_count = 0

        @cached(cache, ttl=60.0, key_func=lambda *a, **kw: "always_same")
        def compute(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        assert compute(5) == 10
        assert compute(10) == 10
        assert call_count == 1


class TestResultCache:
    def test_get_analysis_miss(self, tmp_path):
        rc = ResultCache(tmp_path)
        # Repo path exists but has no cached analysis
        result = rc.get_analysis(str(tmp_path))
        assert result is None

    def test_set_and_get_analysis(self, tmp_path, monkeypatch):
        rc = ResultCache(tmp_path)

        mock_path = tmp_path / "repo"
        mock_path.mkdir()
        monkeypatch.setattr(Path, "resolve", lambda self, orig=mock_path: mock_path)

        rc.set_analysis(str(mock_path), {"result": 42})
        result = rc.get_analysis(str(mock_path))
        assert result is not None
        assert result["result"] == 42

    def test_invalidate_analysis(self, tmp_path, monkeypatch):
        rc = ResultCache(tmp_path)

        mock_path = tmp_path / "repo"
        mock_path.mkdir()
        monkeypatch.setattr(Path, "resolve", lambda self, orig=mock_path: mock_path)

        rc.set_analysis(str(mock_path), {"result": 42})
        rc.invalidate_analysis(str(mock_path))
        assert rc.get_analysis(str(mock_path)) is None
