"""Tests for utils/parallel.py"""

from scholardevclaw.utils.parallel import (
    LazyFileScanner,
    ParallelGit,
    count_files_fast,
    find_files_by_ext,
    parallel_map,
)


class TestParallelMap:
    def test_sequential_for_empty(self):
        assert parallel_map(lambda x: x * 2, []) == []

    def test_single_item(self):
        assert parallel_map(lambda x: x * 2, [5]) == [10]

    def test_multiple_items_thread(self):
        result = parallel_map(lambda x: x * 2, [1, 2, 3])
        assert result == [2, 4, 6]

    def test_preserves_order(self):
        result = parallel_map(str, [3, 1, 2])
        assert result == ["3", "1", "2"]

    def test_with_max_workers(self):
        result = parallel_map(lambda x: x + 1, [1, 2, 3], max_workers=2)
        assert result == [2, 3, 4]


class TestLazyFileScanner:
    def test_scan_py_files(self, tmp_path):
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.py").write_text("")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "c.py").write_text("")
        (tmp_path / "d.txt").write_text("")

        scanner = LazyFileScanner(tmp_path, pattern="*.py")
        files = scanner.files()
        assert len(files) == 3
        assert all(f.suffix == ".py" for f in files)

    def test_count(self, tmp_path):
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.py").write_text("")
        scanner = LazyFileScanner(tmp_path, pattern="*.py")
        assert scanner.count() == 2

    def test_iter_chunks(self, tmp_path):
        for i in range(5):
            (tmp_path / f"{i}.py").write_text("")
        scanner = LazyFileScanner(tmp_path, pattern="*.py")
        chunks = list(scanner.iter_chunks(chunk_size=2))
        assert len(chunks) > 1
        assert all(len(c) <= 2 for c in chunks)

    def test_caches_results(self, tmp_path):
        (tmp_path / "a.py").write_text("")
        scanner = LazyFileScanner(tmp_path, pattern="*.py")
        first = scanner.files()
        (tmp_path / "b.py").write_text("")
        second = scanner.files()
        assert len(first) == len(second)

    def test_empty_directory(self, tmp_path):
        scanner = LazyFileScanner(tmp_path, pattern="*.py")
        assert scanner.files() == []


class TestParallelGit:
    def test_check_status_clean(self, tmp_path):
        import subprocess

        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"], cwd=tmp_path, capture_output=True
        )
        subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path, capture_output=True)
        (tmp_path / "file.txt").write_text("content")
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True)

        result = ParallelGit.check_status(tmp_path)
        assert result["available"] is True
        assert result["is_clean"] is True

    def test_check_status_dirty(self, tmp_path):
        import subprocess

        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"], cwd=tmp_path, capture_output=True
        )
        subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path, capture_output=True)
        (tmp_path / "file.txt").write_text("content")
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True)
        (tmp_path / "file.txt").write_text("modified")

        result = ParallelGit.check_status(tmp_path)
        assert result["available"] is True
        assert result["is_clean"] is False
        assert len(result["changed_files"]) > 0

    def test_check_status_not_git_repo(self, tmp_path):
        result = ParallelGit.check_status(tmp_path)
        assert result["available"] is False

    def test_get_branches(self, tmp_path):
        import subprocess

        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"], cwd=tmp_path, capture_output=True
        )
        subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path, capture_output=True)
        (tmp_path / "f.txt").write_text("x")
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True)

        branches = ParallelGit.get_branches(tmp_path)
        assert len(branches) >= 1

    def test_get_branches_not_git_repo(self, tmp_path):
        branches = ParallelGit.get_branches(tmp_path)
        assert branches == []


class TestCountFilesFast:
    def test_counts_py_files(self, tmp_path):
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.py").write_text("")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "c.py").write_text("")
        assert count_files_fast(tmp_path, "*.py") == 3

    def test_empty(self, tmp_path):
        assert count_files_fast(tmp_path, "*.py") == 0


class TestFindFilesByExt:
    def test_finds_by_extension(self, tmp_path):
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.py").write_text("")
        (tmp_path / "c.txt").write_text("")
        result = find_files_by_ext(tmp_path, ["py", "txt"])
        assert len(result["*.py"]) == 2
        assert len(result["*.txt"]) == 1

    def test_empty_extensions(self, tmp_path):
        result = find_files_by_ext(tmp_path, [])
        assert result == {}

    def test_prefixed_extension(self, tmp_path):
        (tmp_path / "a.py").write_text("")
        result = find_files_by_ext(tmp_path, ["*.py"])
        assert len(result["*.py"]) == 1
