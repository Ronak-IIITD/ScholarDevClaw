"""Tests for utils/atomic_write.py"""

import json
import os
import stat
from pathlib import Path

from scholardevclaw.utils.atomic_write import (
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
)


class TestAtomicWriteText:
    def test_writes_content(self, tmp_path):
        f = tmp_path / "test.txt"
        atomic_write_text(f, "hello world")
        assert f.read_text() == "hello world"

    def test_default_permissions(self, tmp_path):
        f = tmp_path / "test.txt"
        atomic_write_text(f, "content")
        mode = os.stat(f).st_mode & 0o777
        assert mode == stat.S_IRUSR | stat.S_IWUSR

    def test_custom_permissions(self, tmp_path):
        f = tmp_path / "test.txt"
        atomic_write_text(f, "content", mode=0o644)
        mode = os.stat(f).st_mode & 0o777
        assert mode == 0o644

    def test_overwrites_existing(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("old")
        atomic_write_text(f, "new")
        assert f.read_text() == "new"

    def test_accepts_string_path(self, tmp_path):
        f = str(tmp_path / "str_path.txt")
        atomic_write_text(f, "content")
        assert Path(f).read_text() == "content"

    def test_tmp_file_cleaned_on_success(self, tmp_path):
        f = tmp_path / "test.txt"
        atomic_write_text(f, "content")
        tmp_files = list(tmp_path.glob(".tmp_*"))
        assert len(tmp_files) == 0

    def test_empty_content(self, tmp_path):
        f = tmp_path / "empty.txt"
        atomic_write_text(f, "")
        assert f.read_text() == ""


class TestAtomicWriteJson:
    def test_writes_json(self, tmp_path):
        f = tmp_path / "data.json"
        atomic_write_json(f, {"key": "value"})
        assert json.loads(f.read_text()) == {"key": "value"}

    def test_indent_parameter(self, tmp_path):
        f = tmp_path / "data.json"
        atomic_write_json(f, {"nested": {"a": 1}}, indent=2)
        data = json.loads(f.read_text())
        assert data == {"nested": {"a": 1}}

    def test_empty_dict(self, tmp_path):
        f = tmp_path / "data.json"
        atomic_write_json(f, {})
        assert json.loads(f.read_text()) == {}


class TestAtomicWriteJsonl:
    def test_writes_jsonl(self, tmp_path):
        f = tmp_path / "data.jsonl"
        atomic_write_jsonl(f, ['{"a": 1}', '{"b": 2}'])
        lines = f.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"a": 1}

    def test_empty_list(self, tmp_path):
        f = tmp_path / "data.jsonl"
        atomic_write_jsonl(f, [])
        assert f.read_text() == ""

    def test_trailing_newline(self, tmp_path):
        f = tmp_path / "data.jsonl"
        atomic_write_jsonl(f, ['{"x": 1}'])
        assert f.read_text().endswith("\n")
