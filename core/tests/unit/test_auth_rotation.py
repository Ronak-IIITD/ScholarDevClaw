"""Tests for secret rotation automation."""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scholardevclaw.auth.rotation import (
    RotationPolicy,
    RotationResult,
    RotationProvider,
    RotationScheduler,
    get_rotation_provider,
    AnthropicRotationProvider,
    OpenAIRotationProvider,
)


class TestRotationPolicy:
    def test_creation(self):
        policy = RotationPolicy(
            id="policy_1",
            key_id="key_1",
            provider="anthropic",
            rotation_days=90,
        )
        assert policy.rotation_days == 90
        assert policy.enabled is True

    def test_to_dict_roundtrip(self):
        policy = RotationPolicy(
            id="policy_1",
            key_id="key_1",
            provider="anthropic",
            rotation_days=90,
        )
        data = policy.to_dict()
        restored = RotationPolicy.from_dict(data)
        assert restored.id == policy.id
        assert restored.rotation_days == policy.rotation_days


class TestRotationResult:
    def test_success(self):
        result = RotationResult(
            success=True,
            key_id="key_1",
            new_key_id="key_2",
            message="Rotated successfully",
        )
        assert result.success is True
        assert result.new_key_id == "key_2"

    def test_failure(self):
        result = RotationResult(
            success=False,
            key_id="key_1",
            error="Failed to rotate",
        )
        assert result.success is False
        assert result.error == "Failed to rotate"


class TestGetRotationProvider:
    def test_anthropic_provider(self):
        provider = get_rotation_provider("anthropic", "api_key")
        assert isinstance(provider, AnthropicRotationProvider)

    def test_openai_provider(self):
        provider = get_rotation_provider("openai", "api_key")
        assert isinstance(provider, OpenAIRotationProvider)

    def test_unknown_provider(self):
        with pytest.raises(ValueError):
            get_rotation_provider("unknown", "api_key")


class TestRotationScheduler:
    @pytest.fixture
    def store_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_create_policy(self, store_dir):
        scheduler = RotationScheduler(store_dir)
        policy = scheduler.create_policy("key_1", "anthropic", rotation_days=90)
        assert policy.key_id == "key_1"
        assert policy.rotation_days == 90

    def test_get_policy(self, store_dir):
        scheduler = RotationScheduler(store_dir)
        created = scheduler.create_policy("key_1", "anthropic", rotation_days=90)
        retrieved = scheduler.get_policy(created.id)
        assert retrieved is not None
        assert retrieved.rotation_days == 90

    def test_update_policy(self, store_dir):
        scheduler = RotationScheduler(store_dir)
        policy = scheduler.create_policy("key_1", "anthropic", rotation_days=90)
        policy.rotation_days = 30
        scheduler.update_policy(policy)

        retrieved = scheduler.get_policy(policy.id)
        assert retrieved is not None
        assert retrieved.rotation_days == 30

    def test_delete_policy(self, store_dir):
        scheduler = RotationScheduler(store_dir)
        policy = scheduler.create_policy("key_1", "anthropic", rotation_days=90)
        assert scheduler.delete_policy(policy.id) is True
        assert scheduler.get_policy(policy.id) is None

    def test_list_policies(self, store_dir):
        scheduler = RotationScheduler(store_dir)
        scheduler.create_policy("key_1", "anthropic", rotation_days=90)
        scheduler.create_policy("key_2", "openai", rotation_days=60)
        policies = scheduler.list_policies()
        assert len(policies) == 2

    def test_get_policy_for_key(self, store_dir):
        scheduler = RotationScheduler(store_dir)
        scheduler.create_policy("key_1", "anthropic", rotation_days=90)
        policy = scheduler.get_policy_for_key("key_1")
        assert policy is not None
        assert policy.key_id == "key_1"

    def test_get_policy_for_key_not_found(self, store_dir):
        scheduler = RotationScheduler(store_dir)
        policy = scheduler.get_policy_for_key("nonexistent")
        assert policy is None


class TestAnthropicRotationProvider:
    def test_creation(self):
        provider = AnthropicRotationProvider("api_key")
        assert provider.api_key == "api_key"


class TestOpenAIRotationProvider:
    def test_creation(self):
        provider = OpenAIRotationProvider("api_key")
        assert provider.api_key == "api_key"
