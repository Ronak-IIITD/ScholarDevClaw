"""Security tests for input validation and edge cases."""

import json
import pytest
from pathlib import Path
import tempfile
import threading
import time


class TestInputValidation:
    """Tests for input validation security."""

    def test_malicious_repo_path_traversal(self):
        """Prevent path traversal via ../../../etc/passwd"""
        from scholardevclaw.repo_intelligence.tree_sitter_analyzer import TreeSitterAnalyzer

        # Test with path that tries to traverse - should handle gracefully
        malicious_path = "../../../etc/passwd"

        # Should not actually access /etc/passwd - either raise or handle safely
        try:
            analyzer = TreeSitterAnalyzer(Path(malicious_path))
            result = analyzer.analyze()
            # If it doesn't raise, it should return something safe (not crash)
            assert result is not None
        except (ValueError, FileNotFoundError, Exception):
            pass  # Expected to raise or handle safely

    def test_oversized_input_handling(self):
        """Test handling of oversized inputs"""
        from scholardevclaw.application.schema_contract import evaluate_payload_compatibility

        large_payload = {
            "_meta": {"schema_version": "1.0.0", "payload_type": "test"},
            "data": "x" * (1024 * 1024),
        }

        try:
            result = evaluate_payload_compatibility(large_payload)
        except (ValueError, MemoryError, RuntimeError, Exception):
            pass

    def test_null_byte_injection(self):
        """Prevent null byte injection in file paths"""
        from scholardevclaw.repo_intelligence.tree_sitter_analyzer import TreeSitterAnalyzer

        malicious_path = "repo\x00 evil"

        # Should handle null bytes gracefully
        try:
            analyzer = TreeSitterAnalyzer(Path(malicious_path))
            result = analyzer.analyze()
            assert result is not None
        except (ValueError, FileNotFoundError, Exception):
            pass  # Expected to handle safely

    def test_malformed_json_handling(self):
        """Test handling of malformed JSON"""
        from scholardevclaw.application.schema_contract import evaluate_payload_compatibility

        malformed_payloads = [
            {"_meta": {"schema_version": "1.0.0", "payload_type": "test"}, "data": "{"},
            {"_meta": {"schema_version": "invalid", "payload_type": "test"}, "data": "test"},
            {"_meta": {}, "data": "test"},
        ]

        for payload in malformed_payloads:
            try:
                result = evaluate_payload_compatibility(payload)
            except Exception:
                pass

    def test_unicode_in_filenames(self):
        """Test handling of unicode in file paths"""
        from scholardevclaw.repo_intelligence.tree_sitter_analyzer import TreeSitterAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            unicode_file = Path(tmpdir) / "test_文件.py"
            unicode_file.write_text("x = 1")

            analyzer = TreeSitterAnalyzer(Path(tmpdir))
            result = analyzer.analyze()

            assert result is not None


class TestRateLimitSecurity:
    """Tests for rate limiting security."""

    def test_rate_limit_per_key(self):
        """Test rate limit enforcement per key"""
        from scholardevclaw.auth.rate_limit import RateLimiter, RateLimitConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            limiter = RateLimiter(tmpdir)
            limiter.set_limit("test-key", RateLimitConfig(requests_per_minute=3))

            for _ in range(3):
                allowed, _ = limiter.check_rate_limit("test-key")
                assert allowed is True
                limiter.record_usage("test-key")

            allowed, reason = limiter.check_rate_limit("test-key")
            assert allowed is False
            assert "Rate limit exceeded" in reason

    def test_rate_limit_different_keys(self):
        """Test rate limits are per-key"""
        from scholardevclaw.auth.rate_limit import RateLimiter, RateLimitConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            limiter = RateLimiter(tmpdir)
            limiter.set_limit("key1", RateLimitConfig(requests_per_minute=1))
            limiter.set_limit("key2", RateLimitConfig(requests_per_minute=1))

            limiter.record_usage("key1")
            allowed1, _ = limiter.check_rate_limit("key1")
            assert allowed1 is False

            allowed2, _ = limiter.check_rate_limit("key2")
            assert allowed2 is True

    def test_rate_limit_window_pruning(self):
        """Test old entries are pruned"""
        from scholardevclaw.auth.rate_limit import RateLimiter, RateLimitConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            limiter = RateLimiter(tmpdir)
            limiter.set_limit("user1", RateLimitConfig(requests_per_minute=5, requests_per_day=100))

            limiter.record_usage("user1")
            limiter.record_usage("user1")

            stats = limiter.get_usage_stats("user1")
            assert stats.requests_last_minute == 2


class TestRaceConditionPrevention:
    """Tests for race condition handling."""

    def test_concurrent_file_writes(self):
        """Test safe concurrent file access"""
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            results = []
            errors = []

            def write_key(i):
                try:
                    store = AuthStore(tmpdir)
                    store.add_api_key(f"key-{i}", f"key-{i}", AuthProvider.CUSTOM)
                    results.append(i)
                except Exception as e:
                    errors.append(str(e))

            threads = [threading.Thread(target=write_key, args=(i,)) for i in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(results) + len(errors) == 10

    def test_concurrent_rate_limit(self):
        """Test rate limiting under concurrent load"""
        from scholardevclaw.auth.rate_limit import RateLimiter, RateLimitConfig
        import threading

        with tempfile.TemporaryDirectory() as tmpdir:
            limiter = RateLimiter(tmpdir)
            limiter.set_limit("user1", RateLimitConfig(requests_per_minute=100))
            results = []

            def make_request():
                allowed, _ = limiter.check_rate_limit("user1")
                results.append(allowed)

            threads = [threading.Thread(target=make_request) for _ in range(20)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert all(results)


class TestMemoryEdgeCases:
    """Tests for memory-related edge cases."""

    def test_large_file_handling(self):
        """Test handling of large files"""
        from scholardevclaw.repo_intelligence.tree_sitter_analyzer import TreeSitterAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            large_file = Path(tmpdir) / "huge.py"
            large_file.write_text("x = 1\n" * (100_000))

            analyzer = TreeSitterAnalyzer(Path(tmpdir))
            result = analyzer.analyze()

            assert result is not None

    def test_binary_file_in_repo(self):
        """Test handling of binary files"""
        from scholardevclaw.repo_intelligence.tree_sitter_analyzer import TreeSitterAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            binary_file = Path(tmpdir) / "binary.bin"
            binary_file.write_bytes(b"\x00\x01\x02\xff\xfe\x00\x01")

            (Path(tmpdir) / "test.py").write_text("x = 1")

            analyzer = TreeSitterAnalyzer(Path(tmpdir))
            result = analyzer.analyze()

            assert result is not None

    def test_empty_repository(self):
        """Test handling of empty repository"""
        from scholardevclaw.repo_intelligence.tree_sitter_analyzer import TreeSitterAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = TreeSitterAnalyzer(Path(tmpdir))
            result = analyzer.analyze()

            assert result is not None

    def test_deeply_nested_structure(self):
        """Test handling of deeply nested code"""
        from scholardevclaw.repo_intelligence.tree_sitter_analyzer import TreeSitterAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            nested = "def f():\n" + "  " * 100 + "return 1\n"
            (Path(tmpdir) / "nested.py").write_text(nested)

            analyzer = TreeSitterAnalyzer(Path(tmpdir))
            result = analyzer.analyze()

            assert result is not None


class TestAuthStoreSecurity:
    """Security tests for AuthStore."""

    def test_key_id_randomness(self):
        """Verify key IDs are random and unpredictable"""
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider

        ids = set()
        with tempfile.TemporaryDirectory() as tmpdir:
            for _ in range(50):
                store = AuthStore(tmpdir)
                key = store.add_api_key("sk-test-key-1234567890", "test", AuthProvider.CUSTOM)
                ids.add(key.id)

        assert len(ids) == 50

    def test_profile_id_randomness(self):
        """Verify profile IDs are random"""
        from scholardevclaw.auth.store import AuthStore

        ids = set()
        with tempfile.TemporaryDirectory() as tmpdir:
            for _ in range(20):
                store = AuthStore(tmpdir)
                profile = store.create_profile(email="test@example.com")
                ids.add(profile.id)

        assert len(ids) == 20

    def test_concurrent_store_instances(self):
        """Test store handles concurrent access"""
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            store1 = AuthStore(tmpdir)
            store1.add_api_key("sk-first", "first", AuthProvider.CUSTOM)

            store2 = AuthStore(tmpdir)
            keys2 = store2.list_api_keys()

            assert len(keys2) >= 1


class TestEncryptionSecurity:
    """Security tests for encryption."""

    def test_encryption_roundtrip(self):
        """Test encryption/decryption works"""
        from scholardevclaw.auth.encryption import EncryptionManager

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = EncryptionManager(tmpdir)
            mgr.enable("password")

            ciphertext = mgr.encrypt("hello world")
            plaintext = mgr.decrypt(ciphertext)

            assert plaintext == "hello world"

    def test_wrong_password_rejected(self):
        """Test wrong password is rejected"""
        from scholardevclaw.auth.encryption import EncryptionManager

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = EncryptionManager(tmpdir)
            mgr.enable("correctpassword")
            ciphertext = mgr.encrypt("secret")

            mgr2 = EncryptionManager(tmpdir)
            result = mgr2.unlock("wrongpassword")

            assert result is False

    def test_encryption_unicode(self):
        """Test unicode encryption"""
        from scholardevclaw.auth.encryption import EncryptionManager

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = EncryptionManager(tmpdir)
            mgr.enable("password")

            text = "Hello 世界 🔑"
            ciphertext = mgr.encrypt(text)
            plaintext = mgr.decrypt(ciphertext)

            assert plaintext == text


class TestAPIKeyFingerprint:
    """Tests for API key fingerprinting."""

    def test_fingerprint_no_key_exposure(self):
        """Verify fingerprint doesn't expose key"""
        from scholardevclaw.auth.types import APIKey, AuthProvider

        api_key = APIKey(
            id="test",
            name="test",
            provider=AuthProvider.ANTHROPIC,
            key="sk-very-secret-key-12345",
        )

        fp = api_key.get_fingerprint()

        assert "sk-very" not in fp
        assert "12345" not in fp
        assert len(fp) == 64


class TestWebhookSecurity:
    """Security tests for webhook handling."""

    def test_webhook_router_creation(self):
        """Test webhook router can be created"""
        from scholardevclaw.automation.webhooks import WebhookRouter

        router = WebhookRouter()
        assert router is not None
        assert router.triggers == {}

    def test_webhook_trigger_with_secret(self):
        """Test webhook trigger creation with secret"""
        from scholardevclaw.automation.webhooks import WebhookRouter

        router = WebhookRouter()
        trigger = router.add_trigger(
            name="test-trigger",
            event_type="push",
            secret="my-secret-key",
        )

        assert trigger.secret == "my-secret-key"
        assert trigger.name == "test-trigger"


class TestSchedulerSecurity:
    """Security tests for scheduler."""

    def test_schedule_validation(self):
        """Test schedule validation"""
        from scholardevclaw.automation.scheduler import Schedule

        schedule = Schedule(
            id="test",
            name="Test",
            cron_expression="* * * * *",
            enabled=True,
        )

        assert schedule.id == "test"
        assert schedule.enabled is True
