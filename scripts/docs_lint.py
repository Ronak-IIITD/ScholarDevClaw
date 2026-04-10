#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC_FILES = [ROOT / "README.md", ROOT / "GUIDE.md", ROOT / "DEPLOYMENT.md"]
CANONICAL_INSTALL_URL = "https://ronak-iiitd.github.io/ScholarDevClaw/install.sh"
REQUIRED_COMMANDS = [
    "analyze",
    "search",
    "suggest",
    "map",
    "generate",
    "validate",
    "integrate",
    "specs",
    "tui",
]


def _lint_doc(path: Path) -> list[str]:
    content = path.read_text(encoding="utf-8")
    errors: list[str] = []

    if "install.sh" in content and CANONICAL_INSTALL_URL not in content:
        errors.append(
            f"{path}: non-canonical install URL detected (expected {CANONICAL_INSTALL_URL})"
        )

    return errors


def main() -> int:
    missing = [p for p in DOC_FILES if not p.exists()]
    if missing:
        for p in missing:
            print(f"Missing doc file: {p}", file=sys.stderr)
        return 1

    all_errors: list[str] = []
    merged = ""
    for doc in DOC_FILES:
        merged += "\n" + doc.read_text(encoding="utf-8")
        all_errors.extend(_lint_doc(doc))

    for cmd in REQUIRED_COMMANDS:
        pattern = rf"\bscholardevclaw\s+{re.escape(cmd)}\b"
        if re.search(pattern, merged) is None:
            all_errors.append(f"missing command example across docs set: '{cmd}'")

    if all_errors:
        print("Documentation lint failed:")
        for err in all_errors:
            print(f" - {err}")
        return 1

    print("Documentation lint passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
