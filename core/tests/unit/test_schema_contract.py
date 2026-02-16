from __future__ import annotations

from scholardevclaw.application.schema_contract import evaluate_payload_compatibility


def test_schema_contract_compatible_same_major_newer_minor_warns() -> None:
    payload = {
        "_meta": {
            "schema_version": "1.2.0",
            "payload_type": "validation",
        }
    }

    report = evaluate_payload_compatibility(payload, expected_types={"validation"})

    assert report.is_compatible is True
    assert report.issues == []
    assert report.warnings
    assert any("newer" in item.lower() for item in report.warnings)


def test_schema_contract_major_mismatch_is_incompatible() -> None:
    payload = {
        "_meta": {
            "schema_version": "2.0.0",
            "payload_type": "integration",
        }
    }

    report = evaluate_payload_compatibility(payload, expected_types={"integration"})

    assert report.is_compatible is False
    assert report.issues
    assert any("major" in item.lower() for item in report.issues)
    assert report.notes


def test_schema_contract_type_mismatch_is_incompatible() -> None:
    payload = {
        "_meta": {
            "schema_version": "1.0.0",
            "payload_type": "validation",
        }
    }

    report = evaluate_payload_compatibility(payload, expected_types={"integration"})

    assert report.is_compatible is False
    assert any("payload_type" in item for item in report.issues)
