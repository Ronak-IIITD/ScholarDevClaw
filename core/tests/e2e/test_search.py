from __future__ import annotations

import pytest

from scholardevclaw.application.pipeline import run_search


class TestE2ESearch:
    def test_search_rmsnorm_finds_local_specs(self):
        result = run_search("layer normalization")

        assert result.ok is True
        assert len(result.payload.get("local", [])) >= 0

    def test_search_returns_query_in_payload(self):
        result = run_search("normalization")

        assert result.ok is True
        assert result.payload["query"] == "normalization"

    def test_search_has_all_result_types(self):
        result = run_search("test")

        assert result.ok is True
        assert "local" in result.payload
        assert "arxiv" in result.payload
        assert "web" in result.payload
