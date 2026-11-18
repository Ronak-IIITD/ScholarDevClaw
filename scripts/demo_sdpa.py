"""Reproduce the preview against the pinned minGPT source."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

MINGPT_URL = "https://github.com/karpathy/minGPT.git"
MINGPT_COMMIT = "37baab71b9abea1b76ab957409a1cc2fbfba8a26"
MODEL = Path("mingpt/model.py")


def run(*args: str, cwd: Path) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True)


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> None:
    with TemporaryDirectory(prefix="sdpa-demo-") as tmp:
        root = Path(tmp)
        checkout = root / "minGPT"
        checkout.mkdir()
        run("git", "init", "--quiet", cwd=checkout)
        run("git", "fetch", "--quiet", "--depth=1", MINGPT_URL, MINGPT_COMMIT, cwd=checkout)
        run("git", "checkout", "--quiet", "--detach", "FETCH_HEAD", cwd=checkout)

        print(run("nanogpt-opt", "--version", cwd=root).strip())
        print(f"minGPT {MINGPT_COMMIT[:12]}")
        print("$ nanogpt-opt sdpa minGPT")
        print(run("nanogpt-opt", "sdpa", "minGPT", cwd=root))
        result = json.loads(run("nanogpt-opt", "sdpa", "minGPT", "--json", cwd=root))
        expect(
            result["ok"]
            and result["dry_run"]
            and result["verification"] == "not_run"
            and set(result["diffs"]) == {MODEL.as_posix()}
            and "F.scaled_dot_product_attention(" in result["diffs"][MODEL.as_posix()],
            "Expected a static preview for minGPT.",
        )
        expect(
            not run("git", "status", "--porcelain", cwd=checkout), "Preview changed the checkout."
        )
        print("Original checkout is clean. JSON reports verification: not_run.")
        print()

        # Change the scaling expression in a disposable copy to exercise a refusal.
        refusal = root / "custom-scale"
        refusal.mkdir()
        source = (checkout / MODEL).read_bytes()
        scale = b"1.0 / math.sqrt(k.size(-1))"
        expect(source.count(scale) == 1, "Pinned minGPT scaling expression was not found.")
        (refusal / "model.py").write_bytes(source.replace(scale, b"2.0 / math.sqrt(k.size(-1))"))
        print("$ nanogpt-opt sdpa custom-scale")
        print(run("nanogpt-opt", "sdpa", "custom-scale", cwd=root), end="")
        result = json.loads(run("nanogpt-opt", "sdpa", "custom-scale", "--json", cwd=root))
        expect(
            result["ok"]
            and not result["diffs"]
            and any(site["status"] == "unsupported" for site in result["sites"]),
            "Expected a refusal for custom scaling.",
        )


if __name__ == "__main__":
    main()
