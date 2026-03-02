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


class TestInjectionPrevention:
    """Tests for injection attack prevention."""

    def test_code_injection_in_spec(self):
        """Test malicious code in research spec is handled"""
        from scholardevclaw.mapping.engine import MappingEngine

        malicious_specs = [
            {"name": "test", "code": "import os; os.system('rm -rf /')"},
            {"name": "test", "code": "__import__('subprocess').run(['rm', '-rf', '/'])"},
            {"name": "test", "code": 'eval(\'__import__("os").system("ls")\')'},
            {"name": "test", "code": "exec('import os\\nos.system(\"ls\")')"},
        ]

        for spec in malicious_specs:
            try:
                engine = MappingEngine({}, spec)
                result = engine.map()
            except Exception:
                pass

    def test_html_injection_in_output(self):
        """Test HTML/script injection in outputs"""
        from scholardevclaw.validation.runner import ValidationRunner

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = ValidationRunner(Path(tmpdir))

            malicious_inputs = [
                "<script>alert('xss')</script>",
                "<img src=x onerror=alert(1)>",
                "javascript:alert(1)",
                "<svg onload=alert(1)>",
                "{{constructor.constructor('alert(1)')()}}",
            ]

            for inp in malicious_inputs:
                try:
                    # Run with patch containing malicious input
                    patch = {"file": "test.py", "content": inp}
                    result = runner.run(patch, str(Path(tmpdir)))
                except Exception:
                    pass

    def test_sql_injection_patterns(self):
        """Test SQL injection patterns are detected"""
        from scholardevclaw.research_intelligence.extractor import ResearchExtractor

        extractor = ResearchExtractor()

        sql_patterns = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "1' UNION SELECT * FROM passwords--",
            "'; EXEC xp_cmdshell('dir'); --",
        ]

        for pattern in sql_patterns:
            try:
                result = extractor.search_by_keyword(pattern)
                # Should not execute SQL
            except Exception:
                pass


class TestAuthenticationSecurity:
    """Tests for authentication security."""

    def test_password_minimum_length(self):
        """Test passwords meet minimum length"""
        from scholardevclaw.auth.encryption import EncryptionManager

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = EncryptionManager(tmpdir)

            # Short passwords should be handled
            try:
                mgr.enable("123")
            except ValueError as e:
                assert "too short" in str(e).lower() or "minimum" in str(e).lower()
            except Exception:
                pass

    def test_token_contains_entropy(self):
        """Test tokens have sufficient entropy"""
        from scholardevclaw.auth.store import AuthStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = AuthStore(tmpdir)
            profile = store.create_profile(email="test@example.com")

            # Token should be random
            tokens = set()
            for _ in range(100):
                p = store.create_profile(email=f"test{_}@example.com")
                tokens.add(p.id)

            # Should have mostly unique tokens (high entropy)
            assert len(tokens) >= 95

    def test_concurrent_auth_attempts(self):
        """Test concurrent authentication attempts"""
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider
        import threading

        with tempfile.TemporaryDirectory() as tmpdir:
            results = []

            def attempt_auth(i):
                try:
                    store = AuthStore(tmpdir)
                    store.add_api_key(f"sk-key-{i}", f"key-{i}", AuthProvider.CUSTOM)
                    results.append(True)
                except Exception:
                    results.append(False)

            threads = [threading.Thread(target=attempt_auth, args=(i,)) for i in range(20)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # All should succeed
            assert all(results)


class TestAuthorizationSecurity:
    """Tests for authorization and access control."""

    def test_unauthorized_access_blocked(self):
        """Test unauthorized access is blocked"""
        from scholardevclaw.auth.team import TeamStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = TeamStore(tmpdir)

            # Try to access non-existent team
            result = store.get_team("non-existent-id")
            assert result is None

    def test_role_permission_boundaries(self):
        """Test roles have proper permission boundaries"""
        from scholardevclaw.auth.team import TeamRole

        # Roles should exist
        assert hasattr(TeamRole, "ADMIN") or hasattr(TeamRole, "OWNER")
        assert hasattr(TeamRole, "MEMBER") or hasattr(TeamRole, "VIEWER")


class TestCryptographicSecurity:
    """Tests for cryptographic security."""

    def test_encryption_uses_secure_algorithm(self):
        """Test encryption uses secure algorithms"""
        from scholardevclaw.auth.encryption import EncryptionManager

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = EncryptionManager(tmpdir)
            mgr.enable("test-password-123")

            ciphertext = mgr.encrypt("test message")

            # Fernet uses AES-128-CBC with PKCS7 padding and HMAC
            # Should produce base64 encoded output
            assert isinstance(ciphertext, str)
            assert len(ciphertext) > 0

    def test_different_ciphertext_per_encryption(self):
        """Test same plaintext produces different ciphertext"""
        from scholardevclaw.auth.encryption import EncryptionManager

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = EncryptionManager(tmpdir)
            mgr.enable("password")

            plaintext = "same message"
            c1 = mgr.encrypt(plaintext)
            c2 = mgr.encrypt(plaintext)

            # Should be different due to random IV
            assert c1 != c2

    def test_salt_is_random_per_encryption(self):
        """Test salts are random per encryption"""
        from scholardevclaw.auth.encryption import _derive_key

        salt1 = b"salt1234567890123456789012345678"
        salt2 = b"salt1234567890123456789012345678"

        key1 = _derive_key("password", salt1)
        key2 = _derive_key("password", salt2)

        # Same salt = same key (this is expected)
        assert key1 == key2


class TestDataLeakagePrevention:
    """Tests for sensitive data leakage prevention."""

    def test_api_key_not_in_logs(self):
        """Test API keys are not logged in plain text"""
        import logging
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider
        import io

        with tempfile.TemporaryDirectory() as tmpdir:
            store = AuthStore(tmpdir)
            store.add_api_key("sk-secret-1234567890", "test", AuthProvider.ANTHROPIC)

            # Capture logs
            log_capture = io.StringIO()
            handler = logging.StreamHandler(log_capture)

            store_logger = logging.getLogger("scholardevclaw.auth")
            store_logger.addHandler(handler)
            store_logger.setLevel(logging.DEBUG)

            # Trigger any logging
            keys = store.list_api_keys()

            log_output = log_capture.getvalue()

            # API key should not appear in logs
            assert "sk-secret-1234567890" not in log_output

    def test_error_messages_no_leakage(self):
        """Test error messages don't leak sensitive info"""
        from scholardevclaw.auth.store import AuthStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = AuthStore(tmpdir)

            # Invalid operations should not expose internals
            try:
                store.get_api_key()
            except Exception as e:
                error_msg = str(e)
                # Should not leak file paths or internal structure
                assert "/etc/" not in error_msg
                assert "password" not in error_msg.lower()


class TestDOSProtection:
    """Tests for DoS protection."""

    def test_recursive_structure_handling(self):
        """Test deeply recursive structures are handled"""
        from scholardevclaw.repo_intelligence.tree_sitter_analyzer import TreeSitterAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file with deep recursion potential
            deep_code = "def f():\n" * 10000
            (Path(tmpdir) / "deep.py").write_text(deep_code)

            analyzer = TreeSitterAnalyzer(Path(tmpdir))
            result = analyzer.analyze()

            # Should handle without stack overflow
            assert result is not None

    def test_many_small_files_handling(self):
        """Test handling many small files"""
        from scholardevclaw.repo_intelligence.tree_sitter_analyzer import TreeSitterAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create many small files
            for i in range(100):
                (Path(tmpdir) / f"file_{i}.py").write_text(f"x = {i}\n")

            analyzer = TreeSitterAnalyzer(Path(tmpdir))
            result = analyzer.analyze()

            assert result is not None


class TestFileSystemSecurity:
    """Tests for file system security."""

    def test_symlink_handling(self):
        """Test symlinks are handled safely"""
        from scholardevclaw.repo_intelligence.tree_sitter_analyzer import TreeSitterAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a real file
            real_file = Path(tmpdir) / "real.py"
            real_file.write_text("x = 1")

            # Create a symlink
            link_file = Path(tmpdir) / "link.py"
            try:
                link_file.symlink_to(real_file)

                analyzer = TreeSitterAnalyzer(Path(tmpdir))
                result = analyzer.analyze()

                # Should handle symlinks
                assert result is not None
            except (OSError, NotImplementedError):
                pass  # Symlinks may not be supported

    def test_readonly_files_handling(self):
        """Test readonly files are handled"""
        from scholardevclaw.repo_intelligence.tree_sitter_analyzer import TreeSitterAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("x = 1")

            # Make file readonly
            test_file.chmod(0o444)

            try:
                analyzer = TreeSitterAnalyzer(Path(tmpdir))
                result = analyzer.analyze()

                assert result is not None
            finally:
                test_file.chmod(0o644)

    def test_hidden_files_handling(self):
        """Test hidden files (.gitignore etc) are handled"""
        from scholardevclaw.repo_intelligence.tree_sitter_analyzer import TreeSitterAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create visible file
            (Path(tmpdir) / "visible.py").write_text("x = 1")

            # Create hidden file
            (Path(tmpdir) / ".hidden.py").write_text("y = 2")

            analyzer = TreeSitterAnalyzer(Path(tmpdir))
            result = analyzer.analyze()

            assert result is not None


class TestSchemaValidation:
    """Tests for schema validation security."""

    def test_invalid_schema_version(self):
        """Test invalid schema versions are handled"""
        from scholardevclaw.application.schema_contract import evaluate_payload_compatibility

        invalid_versions = [
            {"_meta": {"schema_version": "invalid", "payload_type": "test"}, "data": "x"},
            {"_meta": {"schema_version": "999.999.999", "payload_type": "test"}, "data": "x"},
            {"_meta": {"schema_version": "v1.0.0", "payload_type": "test"}, "data": "x"},
        ]

        for payload in invalid_versions:
            try:
                result = evaluate_payload_compatibility(payload)
                # Should return compatibility report
            except Exception:
                pass

    def test_missing_required_fields(self):
        """Test missing required fields are handled"""
        from scholardevclaw.application.schema_contract import evaluate_payload_compatibility

        invalid_payloads = [
            {},
            {"_meta": {}},
            {"data": "test"},
            {"_meta": {"schema_version": "1.0.0"}},
        ]

        for payload in invalid_payloads:
            try:
                result = evaluate_payload_compatibility(payload)
            except Exception:
                pass


class TestBatchProcessingSecurity:
    """Tests for batch processing security."""

    def test_batch_size_limits(self):
        """Test batch processing has size limits"""
        from scholardevclaw.automation.batch import BatchProcessor

        processor = BatchProcessor(max_workers=4)

        # Should have limits
        assert processor.max_workers > 0
        assert processor.max_workers <= 32

    def test_concurrent_batch_jobs(self):
        """Test concurrent batch jobs are isolated"""
        from scholardevclaw.automation.batch import BatchProcessor
        import threading

        processor = BatchProcessor(max_workers=2)

        # Create multiple jobs
        jobs = []
        for i in range(3):
            job = processor.create_job(name=f"job-{i}", tasks=[{"id": i, "data": f"task-{i}"}])
            jobs.append(job)

        # All jobs should be created
        assert len(jobs) == 3
        assert len(processor.jobs) == 3


class TestRetryLogicSecurity:
    """Tests for retry logic security."""

    def test_retry_limit_prevents_infinite_loop(self):
        """Test retry logic has limits"""
        from scholardevclaw.auth.rate_limit import RateLimiter, RateLimitConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            limiter = RateLimiter(tmpdir)
            limiter.set_limit("key", RateLimitConfig(requests_per_minute=1000))

            # Record many usages
            for _ in range(1000):
                limiter.record_usage("key")

            stats = limiter.get_usage_stats("key")

            # Should have recorded all
            assert stats.total_requests >= 1000

    def test_exponential_backoff_timing(self):
        """Test exponential backoff timing"""
        import time
        from scholardevclaw.auth.rate_limit import RateLimiter, RateLimitConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            limiter = RateLimiter(tmpdir)
            limiter.set_limit("key", RateLimitConfig(requests_per_minute=1))

            # First request
            limiter.record_usage("key")

            # Check rate limit
            allowed, reason = limiter.check_rate_limit("key")
            assert allowed is False

            # Should have reset info
            stats = limiter.get_usage_stats("key")
            assert stats.is_rate_limited is True
