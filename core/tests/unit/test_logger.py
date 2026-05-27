"""Tests for utils/logger.py"""

import logging

from scholardevclaw.utils.logger import logger, setup_logger


class TestSetupLogger:
    def test_returns_logger_instance(self):
        log = setup_logger("test_logger")
        assert isinstance(log, logging.Logger)
        assert log.name == "test_logger"

    def test_logger_has_handler(self):
        log = setup_logger("test_handler")
        assert len(log.handlers) > 0

    def test_level_from_string(self):
        log = setup_logger("test_level_debug", level="DEBUG")
        assert log.level == logging.DEBUG

    def test_level_case_insensitive(self):
        log = setup_logger("test_level_info", level="info")
        assert log.level == logging.INFO

    def test_default_level(self):
        log = setup_logger("test_default_level")
        assert log.level == logging.INFO

    def test_logger_reuses_handlers(self):
        log1 = setup_logger("test_reuse")
        log2 = setup_logger("test_reuse")
        assert len(log2.handlers) == len(log1.handlers)

    def test_module_level_logger(self):
        assert isinstance(logger, logging.Logger)
        assert logger.name == "scholardevclaw"
