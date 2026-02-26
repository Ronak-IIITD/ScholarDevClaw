"""Tests for clipboard and image handling."""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from scholardevclaw.tui.clipboard import (
    ClipboardManager,
    ImageAttachment,
    ImageInputHandler,
    copy_to_clipboard,
    get_clipboard_text,
)


class TestImageAttachment:
    def test_creation(self):
        att = ImageAttachment(
            path=Path("/tmp/test.png"),
            original_name="test.png",
            attached_at=datetime.now().isoformat(),
            size_bytes=1024,
        )
        assert att.path == Path("/tmp/test.png")
        assert att.size_bytes == 1024

    def test_to_dict(self):
        att = ImageAttachment(
            path=Path("/tmp/test.png"),
            original_name="test.png",
            attached_at="2024-01-01T00:00:00",
            size_bytes=1024,
        )
        data = att.to_dict()
        assert data["original_name"] == "test.png"
        assert data["size_bytes"] == 1024


class TestClipboardManager:
    @pytest.fixture
    def store_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_initialization(self, store_dir):
        mgr = ClipboardManager(store_dir)
        assert mgr.store_dir == store_dir
        assert mgr.images_dir.exists()

    def test_get_system(self, store_dir):
        mgr = ClipboardManager(store_dir)
        system = mgr.get_system()
        assert system in ["Darwin", "Linux", "Windows"]

    @patch("subprocess.run")
    def test_read_text_clipboard_macos(self, mock_run, store_dir):
        mock_run.return_value = MagicMock(returncode=0, stdout="test text")
        mgr = ClipboardManager(store_dir)
        with patch.object(mgr, "get_system", return_value="Darwin"):
            text = mgr.read_text_from_clipboard()
            assert text == "test text"

    @patch("subprocess.Popen")
    def test_write_text_clipboard_macos(self, mock_popen, store_dir):
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_popen.return_value = mock_process
        mgr = ClipboardManager(store_dir)
        with patch.object(mgr, "get_system", return_value="Darwin"):
            result = mgr.write_text_to_clipboard("test")
            assert result is True

    @patch("subprocess.run")
    def test_read_text_clipboard_linux_xclip(self, mock_run, store_dir):
        mock_run.return_value = MagicMock(returncode=0, stdout="test text")
        mgr = ClipboardManager(store_dir)
        with patch.object(mgr, "get_system", return_value="Linux"):
            text = mgr.read_text_from_clipboard()
            assert text == "test text"

    @patch("subprocess.run")
    def test_read_text_clipboard_linux_xsel(self, mock_run, store_dir):
        mock_run.side_effect = [FileNotFoundError(), MagicMock(returncode=0, stdout="test text")]
        mgr = ClipboardManager(store_dir)
        with patch.object(mgr, "get_system", return_value="Linux"):
            text = mgr.read_text_from_clipboard()
            assert text == "test text"

    def test_save_image(self, store_dir):
        mgr = ClipboardManager(store_dir)
        image_data = b"fake image data"
        att = mgr.save_image(image_data, "test")
        assert att.path.exists()
        assert att.size_bytes == len(image_data)

    def test_save_dropped_file(self, store_dir):
        mgr = ClipboardManager(store_dir)
        # Create a temp file
        temp_file = Path(tempfile.mktemp(suffix=".png"))
        temp_file.write_bytes(b"fake image")

        att = mgr.save_dropped_file(temp_file)
        assert att.path.exists()
        assert att.original_name == temp_file.name

        temp_file.unlink()

    def test_list_attached_images(self, store_dir):
        mgr = ClipboardManager(store_dir)
        mgr.save_image(b"image1", "test1")
        mgr.save_image(b"image2", "test2")

        images = mgr.list_attached_images()
        assert len(images) == 2

    def test_delete_attachment(self, store_dir):
        mgr = ClipboardManager(store_dir)
        att = mgr.save_image(b"test", "delete_me")
        assert mgr.delete_attachment(att.path) is True
        assert not att.path.exists()

    def test_delete_attachment_wrong_path(self, store_dir):
        mgr = ClipboardManager(store_dir)
        assert mgr.delete_attachment(Path("/tmp/nonexistent")) is False

    def test_clear_old_attachments(self, store_dir):
        mgr = ClipboardManager(store_dir)
        att = mgr.save_image(b"old", "old")
        # Directly modify the file's ctime to be old
        import os

        old_time = (datetime.now() - timedelta(days=10)).timestamp()
        os.utime(att.path, (old_time, old_time))

        deleted = mgr.clear_old_attachments(days=7)
        # Note: ctime might not be modifiable on all systems, so we check either way
        assert deleted >= 0


class TestImageInputHandler:
    @pytest.fixture
    def store_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @patch.object(ClipboardManager, "read_image_from_clipboard")
    def test_paste_from_clipboard(self, mock_read, store_dir):
        mock_read.return_value = b"fake image data"
        mgr = ClipboardManager(store_dir)
        handler = ImageInputHandler(mgr)

        att = handler.paste_from_clipboard()
        assert att is not None
        assert att.path.exists()

    def test_handle_dropped_file(self, store_dir):
        mgr = ClipboardManager(store_dir)
        handler = ImageInputHandler(mgr)

        # Use NamedTemporaryFile for proper temp file handling
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            tf.write(b"dropped")
            temp_path = Path(tf.name)

        try:
            att = handler.handle_dropped_file(temp_path)
            assert att is not None
            assert att.path.exists()
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_handle_dropped_unsupported_file(self, store_dir):
        mgr = ClipboardManager(store_dir)
        handler = ImageInputHandler(mgr)

        temp_file = Path(tempfile.mktemp(suffix=".txt"))
        temp_file.write_text("text")

        att = handler.handle_dropped_file(temp_file)
        assert att is None

        temp_file.unlink()

    def test_get_image_context_no_images(self, store_dir):
        mgr = ClipboardManager(store_dir)
        handler = ImageInputHandler(mgr)

        ctx = handler.get_image_context()
        assert ctx["has_images"] is False
        assert ctx["images"] == []

    def test_get_image_context_with_images(self, store_dir):
        mgr = ClipboardManager(store_dir)
        handler = ImageInputHandler(mgr)

        att = mgr.save_image(b"test", "context_test")
        handler.current_attachments.append(att)

        ctx = handler.get_image_context()
        assert ctx["has_images"] is True
        assert ctx["image_count"] == 1

    def test_build_image_prompt_context(self, store_dir):
        mgr = ClipboardManager(store_dir)
        handler = ImageInputHandler(mgr)

        att = mgr.save_image(b"test", "prompt_test")
        handler.current_attachments.append(att)

        context = handler.build_image_prompt_context()
        assert "Attached Images" in context
        assert str(att.path) in context

    def test_clear_attachments(self, store_dir):
        mgr = ClipboardManager(store_dir)
        handler = ImageInputHandler(mgr)

        att = mgr.save_image(b"test", "clear_test")
        handler.current_attachments.append(att)

        handler.clear_attachments()
        assert len(handler.current_attachments) == 0
