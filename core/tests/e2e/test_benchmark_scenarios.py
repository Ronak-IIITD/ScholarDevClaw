from __future__ import annotations

import json
from pathlib import Path


def test_benchmark_scenarios_manifest_is_well_formed():
    root = Path(__file__).resolve().parent
    manifest_path = root / "benchmarks" / "scenarios.json"
    assert manifest_path.exists()

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data.get("version")

    scenarios = data.get("scenarios")
    assert isinstance(scenarios, list)
    assert len(scenarios) >= 3

    ids = {s.get("id") for s in scenarios}
    assert None not in ids
    assert len(ids) == len(scenarios)

    for scenario in scenarios:
        assert scenario.get("spec")
        assert scenario.get("repo")
        assertions = scenario.get("assertions")
        assert isinstance(assertions, list)
        assert assertions
