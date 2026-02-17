from __future__ import annotations

import pytest

from scholardevclaw.application.pipeline import run_specs


class TestE2ESpecs:
    def test_specs_returns_spec_list(self):
        result = run_specs()

        assert result.ok is True
        assert "spec_names" in result.payload
        assert len(result.payload["spec_names"]) > 0

    def test_specs_includes_rmsnorm(self):
        result = run_specs()

        assert result.ok is True
        assert "rmsnorm" in result.payload["spec_names"]

    def test_specs_returns_categories(self):
        result = run_specs()

        assert result.ok is True
        assert "categories" in result.payload
        assert len(result.payload["categories"]) > 0

    def test_specs_detailed_includes_details(self):
        result = run_specs(detailed=True)

        assert result.ok is True
        assert "details" in result.payload
        assert "rmsnorm" in result.payload["details"]

    def test_specs_by_category_groups_by_category(self):
        result = run_specs(by_category=True)

        assert result.ok is True
        assert result.payload["view"] == "categories"
