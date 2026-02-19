import time
import threading
from unittest.mock import Mock, patch, MagicMock

import pytest

from scholardevclaw.utils.connection_pool import (
    PoolConfig,
    ConnectionStats,
    ConnectionPool,
    HTTPConnectionPool,
    AsyncHTTPConnectionPool,
)


class TestPoolConfig:
    def test_default_config(self):
        config = PoolConfig()
        assert config.max_connections == 10
        assert config.max_keepalive == 5
        assert config.keepalive_expiry == 30.0
        assert config.timeout == 30.0

    def test_custom_config(self):
        config = PoolConfig(
            max_connections=20,
            max_keepalive=10,
            timeout=60.0,
        )
        assert config.max_connections == 20
        assert config.max_keepalive == 10
        assert config.timeout == 60.0


class TestConnectionStats:
    def test_initial_stats(self):
        stats = ConnectionStats()
        assert stats.total_created == 0
        assert stats.total_reused == 0
        assert stats.total_closed == 0
        assert stats.active_connections == 0

    def test_record_create(self):
        stats = ConnectionStats()
        stats.record_create()
        assert stats.total_created == 1
        assert stats.active_connections == 1

    def test_record_reuse(self):
        stats = ConnectionStats()
        stats.record_reuse()
        assert stats.total_reused == 1
        assert stats.pool_hits == 1

    def test_record_close(self):
        stats = ConnectionStats()
        stats.record_create()
        stats.record_close()
        assert stats.total_closed == 1
        assert stats.active_connections == 0

    def test_record_close_never_negative(self):
        stats = ConnectionStats()
        stats.record_close()
        assert stats.active_connections == 0

    def test_record_miss(self):
        stats = ConnectionStats()
        stats.record_miss()
        assert stats.pool_misses == 1

    def test_record_error(self):
        stats = ConnectionStats()
        stats.record_error()
        assert stats.errors == 1


class TestConnectionPool:
    def test_basic_acquire_release(self):
        created = []

        def factory():
            obj = {"id": len(created)}
            created.append(obj)
            return obj

        pool = ConnectionPool(factory, max_size=2)

        with pool.acquire() as conn:
            assert conn["id"] == 0
            assert len(created) == 1

        assert pool.size == 1
        assert pool.stats.total_created == 1
        assert pool.stats.total_reused == 0

    def test_connection_reuse(self):
        created = []

        def factory():
            obj = {"id": len(created)}
            created.append(obj)
            return obj

        pool = ConnectionPool(factory, max_size=2)

        with pool.acquire() as conn1:
            pass

        with pool.acquire() as conn2:
            pass

        assert len(created) == 1
        assert conn1 is conn2
        assert pool.stats.total_reused == 1

    def test_pool_size_limit(self):
        created = []

        def factory():
            obj = {"id": len(created)}
            created.append(obj)
            return obj

        pool = ConnectionPool(factory, max_size=2)

        with pool.acquire() as conn1:
            pass

        with pool.acquire() as conn2:
            pass

        assert pool.size <= 2

    def test_validation(self):
        valid_connections = [True, True, False]

        def factory():
            return {"id": len(valid_connections)}

        def validator(conn):
            idx = conn["id"]
            if idx < len(valid_connections):
                return valid_connections[idx]
            return True

        pool = ConnectionPool(factory, validator=validator, max_size=2)

        with pool.acquire() as conn1:
            pass

        with pool.acquire() as conn2:
            pass

        assert pool.stats.total_created >= 1

    def test_idle_timeout(self):
        def factory():
            return {"id": time.time()}

        pool = ConnectionPool(factory, max_size=2, idle_timeout=0.1)

        with pool.acquire() as conn1:
            conn1_id = conn1["id"]

        time.sleep(0.15)

        with pool.acquire() as conn2:
            pass

        assert conn2["id"] != conn1_id

    def test_clear(self):
        closed = []

        def factory():
            return {"id": 0}

        def closer(conn):
            closed.append(conn)

        pool = ConnectionPool(factory, closer=closer, max_size=5)

        with pool.acquire() as conn1:
            pass

        with pool.acquire() as conn2:
            pass

        assert pool.size == 1

        pool.clear()
        assert pool.size == 0

    def test_thread_safety(self):
        acquired = []

        def factory():
            return {"id": id(object())}

        pool = ConnectionPool(factory, max_size=10)

        def worker():
            with pool.acquire() as conn:
                acquired.append(conn["id"])
                time.sleep(0.001)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(acquired) == 20


class TestHTTPConnectionPool:
    def test_get_singleton(self):
        HTTPConnectionPool.clear_all()
        pool1 = HTTPConnectionPool.get_pool("https://api.example.com")
        pool2 = HTTPConnectionPool.get_pool("https://api.example.com")
        assert pool1 is pool2

    def test_different_urls_different_pools(self):
        HTTPConnectionPool.clear_all()
        pool1 = HTTPConnectionPool.get_pool("https://api1.example.com")
        pool2 = HTTPConnectionPool.get_pool("https://api2.example.com")
        assert pool1 is not pool2

    @patch("httpx.Client")
    def test_client_lazy_creation(self, mock_client_class):
        HTTPConnectionPool.clear_all()
        pool = HTTPConnectionPool("https://api.example.com")

        assert pool._client is None

        _ = pool.client

        mock_client_class.assert_called_once()

    @patch("httpx.Client")
    def test_request_stats(self, mock_client_class):
        HTTPConnectionPool.clear_all()
        mock_client = MagicMock()
        mock_client.request.return_value = MagicMock(status_code=200)
        mock_client_class.return_value = mock_client

        pool = HTTPConnectionPool("https://api.example.com")
        pool.get("/test")

        assert pool.stats.total_created == 1
        assert pool.stats.total_reused == 1

    @patch("httpx.Client")
    def test_error_stats(self, mock_client_class):
        HTTPConnectionPool.clear_all()
        mock_client = MagicMock()
        mock_client.request.side_effect = Exception("Network error")
        mock_client_class.return_value = mock_client

        pool = HTTPConnectionPool("https://api.example.com")

        with pytest.raises(Exception):
            pool.get("/test")

        assert pool.stats.errors == 1

    @patch("httpx.Client")
    def test_close(self, mock_client_class):
        HTTPConnectionPool.clear_all()
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        pool = HTTPConnectionPool("https://api.example.com")
        _ = pool.client
        pool.close()

        mock_client.close.assert_called_once()


class TestAsyncHTTPConnectionPool:
    def test_get_singleton(self):
        AsyncHTTPConnectionPool.clear_all()
        pool1 = AsyncHTTPConnectionPool.get_pool("https://api.example.com")
        pool2 = AsyncHTTPConnectionPool.get_pool("https://api.example.com")
        assert pool1 is pool2

    def test_stats_tracking(self):
        AsyncHTTPConnectionPool.clear_all()
        pool = AsyncHTTPConnectionPool("https://api.example.com")
        assert pool.stats.total_created == 0
