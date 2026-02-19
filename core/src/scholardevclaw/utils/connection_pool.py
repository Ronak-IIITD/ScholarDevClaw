from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar
from contextlib import contextmanager
import httpx


T = TypeVar("T")


@dataclass
class PoolConfig:
    max_connections: int = 10
    max_keepalive: int = 5
    keepalive_expiry: float = 30.0
    timeout: float = 30.0
    connect_timeout: float = 5.0
    read_timeout: float = 30.0
    write_timeout: float = 30.0
    pool_timeout: float = 5.0


@dataclass
class ConnectionStats:
    total_created: int = 0
    total_reused: int = 0
    total_closed: int = 0
    active_connections: int = 0
    pool_hits: int = 0
    pool_misses: int = 0
    errors: int = 0

    def record_create(self) -> None:
        self.total_created += 1
        self.active_connections += 1

    def record_reuse(self) -> None:
        self.total_reused += 1
        self.pool_hits += 1

    def record_close(self) -> None:
        self.total_closed += 1
        self.active_connections = max(0, self.active_connections - 1)

    def record_miss(self) -> None:
        self.pool_misses += 1

    def record_error(self) -> None:
        self.errors += 1


class ConnectionPool(Generic[T]):
    def __init__(
        self,
        factory: Callable[[], T],
        validator: Callable[[T], bool] | None = None,
        closer: Callable[[T], None] | None = None,
        max_size: int = 10,
        idle_timeout: float = 300.0,
    ):
        self._factory = factory
        self._validator = validator or (lambda _: True)
        self._closer = closer or (lambda _: None)
        self._max_size = max_size
        self._idle_timeout = idle_timeout
        self._pool: list[tuple[T, float]] = []
        self._lock = threading.RLock()
        self._stats = ConnectionStats()

    @contextmanager
    def acquire(self):
        conn = self._get()
        try:
            yield conn
        finally:
            self._return(conn)

    def _get(self) -> T:
        with self._lock:
            now = time.time()
            while self._pool:
                conn, created_at = self._pool.pop()
                if now - created_at > self._idle_timeout:
                    self._close_connection(conn)
                    continue
                if self._validator(conn):
                    self._stats.record_reuse()
                    return conn
                else:
                    self._close_connection(conn)

            self._stats.record_miss()
            self._stats.record_create()
            return self._factory()

    def _return(self, conn: T) -> None:
        with self._lock:
            if len(self._pool) < self._max_size and self._validator(conn):
                self._pool.append((conn, time.time()))
            else:
                self._close_connection(conn)

    def _close_connection(self, conn: T) -> None:
        try:
            self._closer(conn)
        except Exception:
            pass
        self._stats.record_close()

    def clear(self) -> None:
        with self._lock:
            while self._pool:
                conn, _ = self._pool.pop()
                self._close_connection(conn)

    @property
    def stats(self) -> ConnectionStats:
        return self._stats

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._pool)


class HTTPConnectionPool:
    _instances: dict[str, "HTTPConnectionPool"] = {}
    _lock = threading.RLock()

    def __init__(self, base_url: str, config: PoolConfig | None = None):
        self._base_url = base_url
        self._config = config or PoolConfig()
        self._stats = ConnectionStats()
        self._client: httpx.Client | None = None
        self._client_lock = threading.RLock()

    @classmethod
    def get_pool(cls, base_url: str, config: PoolConfig | None = None) -> "HTTPConnectionPool":
        with cls._lock:
            if base_url not in cls._instances:
                cls._instances[base_url] = cls(base_url, config)
            return cls._instances[base_url]

    @classmethod
    def clear_all(cls) -> None:
        with cls._lock:
            for pool in cls._instances.values():
                pool.close()
            cls._instances.clear()

    @property
    def client(self) -> httpx.Client:
        with self._client_lock:
            if self._client is None:
                limits = httpx.Limits(
                    max_connections=self._config.max_connections,
                    max_keepalive_connections=self._config.max_keepalive,
                    keepalive_expiry=self._config.keepalive_expiry,
                )
                timeout = httpx.Timeout(
                    connect=self._config.connect_timeout,
                    read=self._config.read_timeout,
                    write=self._config.write_timeout,
                    pool=self._config.pool_timeout,
                )
                self._client = httpx.Client(
                    base_url=self._base_url,
                    limits=limits,
                    timeout=timeout,
                )
                self._stats.record_create()
            return self._client

    def request(self, method: str, path: str, **kwargs) -> httpx.Response:
        try:
            response = self.client.request(method, path, **kwargs)
            self._stats.record_reuse()
            return response
        except Exception:
            self._stats.record_error()
            raise

    def get(self, path: str, **kwargs) -> httpx.Response:
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs) -> httpx.Response:
        return self.request("POST", path, **kwargs)

    def put(self, path: str, **kwargs) -> httpx.Response:
        return self.request("PUT", path, **kwargs)

    def delete(self, path: str, **kwargs) -> httpx.Response:
        return self.request("DELETE", path, **kwargs)

    def close(self) -> None:
        with self._client_lock:
            if self._client is not None:
                self._client.close()
                self._client = None
                self._stats.record_close()

    @property
    def stats(self) -> ConnectionStats:
        return self._stats


class AsyncHTTPConnectionPool:
    _instances: dict[str, "AsyncHTTPConnectionPool"] = {}
    _lock = threading.RLock()

    def __init__(self, base_url: str, config: PoolConfig | None = None):
        self._base_url = base_url
        self._config = config or PoolConfig()
        self._stats = ConnectionStats()
        self._client: httpx.AsyncClient | None = None
        self._client_lock = threading.RLock()

    @classmethod
    def get_pool(cls, base_url: str, config: PoolConfig | None = None) -> "AsyncHTTPConnectionPool":
        with cls._lock:
            if base_url not in cls._instances:
                cls._instances[base_url] = cls(base_url, config)
            return cls._instances[base_url]

    @classmethod
    def clear_all(cls) -> None:
        with cls._lock:
            cls._instances.clear()

    @property
    def client(self) -> httpx.AsyncClient:
        with self._client_lock:
            if self._client is None:
                limits = httpx.Limits(
                    max_connections=self._config.max_connections,
                    max_keepalive_connections=self._config.max_keepalive,
                    keepalive_expiry=self._config.keepalive_expiry,
                )
                timeout = httpx.Timeout(
                    connect=self._config.connect_timeout,
                    read=self._config.read_timeout,
                    write=self._config.write_timeout,
                    pool=self._config.pool_timeout,
                )
                self._client = httpx.AsyncClient(
                    base_url=self._base_url,
                    limits=limits,
                    timeout=timeout,
                )
                self._stats.record_create()
            return self._client

    async def request(self, method: str, path: str, **kwargs) -> httpx.Response:
        try:
            response = await self.client.request(method, path, **kwargs)
            self._stats.record_reuse()
            return response
        except Exception:
            self._stats.record_error()
            raise

    async def get(self, path: str, **kwargs) -> httpx.Response:
        return await self.request("GET", path, **kwargs)

    async def post(self, path: str, **kwargs) -> httpx.Response:
        return await self.request("POST", path, **kwargs)

    async def put(self, path: str, **kwargs) -> httpx.Response:
        return await self.request("PUT", path, **kwargs)

    async def delete(self, path: str, **kwargs) -> httpx.Response:
        return await self.request("DELETE", path, **kwargs)

    async def close(self) -> None:
        with self._client_lock:
            if self._client is not None:
                await self._client.aclose()
                self._client = None
                self._stats.record_close()

    @property
    def stats(self) -> ConnectionStats:
        return self._stats
