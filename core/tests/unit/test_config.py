"""Tests for utils/config.py"""

from scholardevclaw.utils.config import Settings, settings


class TestSettings:
    def test_default_values(self):
        s = Settings()
        assert s.repo_intelligence is True
        assert s.research_intelligence is True
        assert s.mapping_engine is True
        assert s.patch_generation is True
        assert s.validation is True
        assert s.benchmark_timeout == 300
        assert s.max_retries == 2
        assert s.log_level == "INFO"

    def test_env_prefix(self):
        assert Settings.Config.env_prefix == "SC_"

    def test_module_settings_instance(self):
        assert isinstance(settings, Settings)

    def test_log_level_override(self):
        s = Settings(log_level="DEBUG")
        assert s.log_level == "DEBUG"

    def test_benchmark_timeout_override(self):
        s = Settings(benchmark_timeout=600)
        assert s.benchmark_timeout == 600

    def test_max_retries_override(self):
        s = Settings(max_retries=5)
        assert s.max_retries == 5

    def test_feature_flag_overrides(self):
        s = Settings(repo_intelligence=False, validation=False)
        assert s.repo_intelligence is False
        assert s.validation is False
