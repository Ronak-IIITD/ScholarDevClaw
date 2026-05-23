from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .runner import DEFAULT_REPORT_PATH

DEFAULT_SUMMARY_PATH = DEFAULT_REPORT_PATH.with_name("benchmark_summary.md")


def build_markdown_summary(payload: dict[str, Any]) -> str:
    aggregate = payload.get("aggregate", {})
    lines = [
        "# ScholarDevClaw Benchmark Summary",
        "",
        f"- Generated at: `{payload.get('generated_at', 'unknown')}`",
        f"- Total cases: `{aggregate.get('total_cases', 0)}`",
        f"- Supported cases: `{aggregate.get('supported_cases', 0)}`",
        f"- Unsupported cases: `{aggregate.get('unsupported_cases', 0)}`",
        f"- Aggregate score: `{aggregate.get('aggregate_score', 0)}`",
        f"- Supported score: `{aggregate.get('supported_score', 0)}`",
        "",
        "| Case | Spec | Status | Score | Candidate | Notes |",
        "|------|------|--------|-------|-----------|-------|",
    ]

    for result in payload.get("results", []):
        notes = str(result.get("metadata", {}).get("notes", "")).replace("\n", " ").strip()
        lines.append(
            "| {case} | {spec} | {status} | {score} | {candidate} | {notes} |".format(
                case=result.get("id", ""),
                spec=result.get("pipeline_spec", "n/a"),
                status=result.get("status", ""),
                score=result.get("score", 0),
                candidate=result.get("candidate_file", "n/a"),
                notes=notes or "-",
            )
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render benchmark_report.json as Markdown.")
    parser.add_argument(
        "--input", default=str(DEFAULT_REPORT_PATH), help="Path to benchmark_report.json"
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_SUMMARY_PATH),
        help="Path to write benchmark_summary.md",
    )
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text())
    summary = build_markdown_summary(payload)
    Path(args.output).write_text(summary)
    print(Path(args.output).resolve())


if __name__ == "__main__":
    main()
