from __future__ import annotations

import pytest

from scholardevclaw.security.path_policy import enforce_allowed_repo_path


def test_enforce_allowed_repo_path_allows_descendant(monkeypatch, tmp_path):
    allowed = tmp_path / "allowed"
    repo = allowed / "repo"
    repo.mkdir(parents=True)
    monkeypatch.setenv("SCHOLARDEVCLAW_ALLOWED_REPO_DIRS", str(allowed))

    resolved = enforce_allowed_repo_path(repo.resolve())

    assert resolved == repo.resolve()


def test_enforce_allowed_repo_path_blocks_outside(monkeypatch, tmp_path):
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    monkeypatch.setenv("SCHOLARDEVCLAW_ALLOWED_REPO_DIRS", str(allowed))

    with pytest.raises(PermissionError):
        enforce_allowed_repo_path(outside.resolve())


def test_enforce_allowed_repo_path_normalized_traversal(monkeypatch, tmp_path):
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    monkeypatch.setenv("SCHOLARDEVCLAW_ALLOWED_REPO_DIRS", str(allowed))

    traversal_like = (allowed / ".." / "outside").resolve()
    assert traversal_like == outside.resolve()

    with pytest.raises(PermissionError):
        enforce_allowed_repo_path(traversal_like)
