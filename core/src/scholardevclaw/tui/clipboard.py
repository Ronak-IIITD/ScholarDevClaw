"""Clipboard and image handling for TUI.

Supports:
- Paste images from clipboard (Ctrl+V / Cmd+V)
- Drag and drop images into TUI
- Copy text output to clipboard
- Save pasted images to temp files for AI agent access
"""

from __future__ import annotations

import base64
import json
import os
import platform
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ImageAttachment:
    """An image attached to input."""

    path: Path
    original_name: str
    attached_at: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "original_name": self.original_name,
            "attached_at": self.attached_at,
            "size_bytes": self.size_bytes,
        }


class ClipboardManager:
    """Handle clipboard operations for text and images."""

    def __init__(self, store_dir: str | Path | None = None):
        if store_dir:
            self.store_dir = Path(store_dir)
        else:
            self.store_dir = Path.home() / ".scholardevclaw"
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.store_dir / "attached_images"
        self.images_dir.mkdir(exist_ok=True)

    def get_system(self) -> str:
        """Get the current operating system."""
        return platform.system()

    def read_text_from_clipboard(self) -> str | None:
        """Read text from system clipboard."""
        system = self.get_system()

        try:
            if system == "Darwin":  # macOS
                result = subprocess.run(
                    ["pbpaste"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                return result.stdout if result.returncode == 0 else None

            elif system == "Linux":
                # Try xclip first, then xsel
                for cmd in [
                    ["xclip", "-selection", "clipboard", "-o"],
                    ["xsel", "--clipboard", "--output"],
                ]:
                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                        if result.returncode == 0:
                            return result.stdout
                    except FileNotFoundError:
                        continue
                return None

            elif system == "Windows":
                result = subprocess.run(
                    ["powershell", "-Command", "Get-Clipboard"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                return result.stdout if result.returncode == 0 else None

        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None

        return None

    def write_text_to_clipboard(self, text: str) -> bool:
        """Write text to system clipboard."""
        system = self.get_system()

        try:
            if system == "Darwin":  # macOS
                process = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
                process.communicate(text.encode())
                return process.returncode == 0

            elif system == "Linux":
                for cmd in [
                    ["xclip", "-selection", "clipboard"],
                    ["xsel", "--clipboard", "--input"],
                ]:
                    try:
                        process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
                        process.communicate(text.encode())
                        if process.returncode == 0:
                            return True
                    except FileNotFoundError:
                        continue
                return False

            elif system == "Windows":
                # SECURITY: Pass text via stdin to avoid command injection
                process = subprocess.Popen(
                    ["powershell", "-Command", "Set-Clipboard -Value ($input | Out-String)"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                process.communicate(text.encode())
                return process.returncode == 0

        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

        return False

    def read_image_from_clipboard(self) -> bytes | None:
        """Read image data from system clipboard."""
        system = self.get_system()

        try:
            if system == "Darwin":  # macOS
                # Use osascript to get image from clipboard
                # SECURITY: Use tempfile instead of hardcoded /tmp path to avoid symlink attacks
                fd, temp_path_str = tempfile.mkstemp(suffix=".png", prefix="sdclaw_clip_")
                os.close(fd)
                temp_path = Path(temp_path_str)
                try:
                    script = f"""
                    try
                        set theData to the clipboard as «class PNGf»
                        return do shell script "echo " & quoted form of (do shell script "python3 -c 'import base64,sys; print(base64.b64encode(stdin.read()).decode())'") & " | base64 -d > {temp_path_str} && cat {temp_path_str}"
                    on error
                        return ""
                    end try
                    """
                    # Simpler approach: save clipboard to temp file
                    result = subprocess.run(
                        ["osascript", "-e", "set theData to the clipboard as «class PNGf»"],
                        capture_output=True,
                        timeout=5,
                    )
                    if result.returncode != 0:
                        # Try JPEG
                        result = subprocess.run(
                            ["osascript", "-e", "set theData to the clipboard as «class JPEG»"],
                            capture_output=True,
                            timeout=5,
                        )
                    # Read from temp file
                    if temp_path.exists() and temp_path.stat().st_size > 0:
                        data = temp_path.read_bytes()
                        return data
                    return None
                finally:
                    # Cleanup temp file
                    try:
                        temp_path.unlink(missing_ok=True)
                    except Exception:
                        pass

            elif system == "Linux":
                # Try xclip to get image
                try:
                    result = subprocess.run(
                        ["xclip", "-selection", "clipboard", "-t", "image/png", "-o"],
                        capture_output=True,
                        timeout=5,
                    )
                    if result.returncode == 0 and result.stdout:
                        return result.stdout
                except FileNotFoundError:
                    pass
                return None

            elif system == "Windows":
                # PowerShell to get image from clipboard
                script = """
                Add-Type -AssemblyName System.Windows.Forms
                if ([System.Windows.Forms.Clipboard]::ContainsImage()) {
                    $img = [System.Windows.Forms.Clipboard]::GetImage()
                    $ms = New-Object System.IO.MemoryStream
                    $img.Save($ms, [System.Drawing.Imaging.ImageFormat]::Png)
                    [Convert]::ToBase64String($ms.ToArray())
                }
                """
                result = subprocess.run(
                    ["powershell", "-Command", script],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return base64.b64decode(result.stdout.strip())

        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            return None

        return None

    def save_image(self, image_data: bytes, prefix: str = "pasted") -> ImageAttachment:
        """Save image data to file and return attachment info."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.png"
        filepath = self.images_dir / filename

        filepath.write_bytes(image_data)

        return ImageAttachment(
            path=filepath,
            original_name=filename,
            attached_at=datetime.now().isoformat(),
            size_bytes=len(image_data),
        )

    def save_dropped_file(self, source_path: Path) -> ImageAttachment:
        """Save a dropped file and return attachment info."""
        if not source_path.exists():
            raise FileNotFoundError(f"File not found: {source_path}")

        # SECURITY: Reject symlinks to prevent symlink-following attacks
        if source_path.is_symlink():
            raise ValueError(f"Symlinks are not allowed: {source_path}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = source_path.suffix or ".png"
        filename = f"dropped_{timestamp}{ext}"
        dest_path = self.images_dir / filename

        shutil.copy2(source_path, dest_path)

        return ImageAttachment(
            path=dest_path,
            original_name=source_path.name,
            attached_at=datetime.now().isoformat(),
            size_bytes=dest_path.stat().st_size,
        )

    def list_attached_images(self) -> list[ImageAttachment]:
        """List all attached images."""
        attachments = []
        for f in self.images_dir.glob("*"):
            if f.is_file():
                attachments.append(
                    ImageAttachment(
                        path=f,
                        original_name=f.name,
                        attached_at=datetime.fromtimestamp(f.stat().st_ctime).isoformat(),
                        size_bytes=f.stat().st_size,
                    )
                )
        return sorted(attachments, key=lambda x: x.attached_at, reverse=True)

    def delete_attachment(self, path: Path) -> bool:
        """Delete an attached image."""
        try:
            if path.exists() and path.parent == self.images_dir:
                path.unlink()
                return True
        except Exception:
            pass
        return False

    def clear_old_attachments(self, days: int = 7) -> int:
        """Clear attachments older than specified days."""
        cutoff = datetime.now().timestamp() - (days * 86400)
        deleted = 0

        for f in self.images_dir.glob("*"):
            if f.is_file() and f.stat().st_ctime < cutoff:
                try:
                    f.unlink()
                    deleted += 1
                except Exception:
                    pass

        return deleted


class ImageInputHandler:
    """Handle image input for AI agent.

    Provides:
    - Image path resolution for AI context
    - Image metadata for prompts
    - Integration with agent workflow
    """

    def __init__(self, clipboard_manager: ClipboardManager | None = None):
        self.clipboard = clipboard_manager or ClipboardManager()
        self.current_attachments: list[ImageAttachment] = []

    def paste_from_clipboard(self) -> ImageAttachment | None:
        """Paste image from clipboard if available."""
        image_data = self.clipboard.read_image_from_clipboard()
        if image_data:
            attachment = self.clipboard.save_image(image_data, "clipboard_paste")
            self.current_attachments.append(attachment)
            return attachment
        return None

    def handle_dropped_file(self, path: Path) -> ImageAttachment | None:
        """Handle a dropped file."""
        supported_formats = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff"}

        if path.suffix.lower() in supported_formats:
            attachment = self.clipboard.save_dropped_file(path)
            self.current_attachments.append(attachment)
            return attachment

        return None

    def add_attachment(self, attachment: ImageAttachment) -> None:
        """Add an attachment manually."""
        self.current_attachments.append(attachment)

    def clear_attachments(self) -> None:
        """Clear current attachments."""
        self.current_attachments = []

    def get_image_context(self) -> dict[str, Any]:
        """Get image context for AI agent."""
        if not self.current_attachments:
            return {"has_images": False, "images": []}

        return {
            "has_images": True,
            "image_count": len(self.current_attachments),
            "images": [a.to_dict() for a in self.current_attachments],
            "image_paths": [str(a.path) for a in self.current_attachments],
        }

    def build_image_prompt_context(self) -> str:
        """Build a prompt context string about attached images."""
        if not self.current_attachments:
            return ""

        lines = ["\n📎 **Attached Images:**"]
        for i, att in enumerate(self.current_attachments, 1):
            lines.append(f"  {i}. `{att.path}` ({att.size_bytes:,} bytes)")
        lines.append("")

        return "\n".join(lines)


def copy_to_clipboard(text: str) -> bool:
    """Convenience function to copy text to clipboard."""
    manager = ClipboardManager()
    return manager.write_text_to_clipboard(text)


def get_clipboard_text() -> str | None:
    """Convenience function to get text from clipboard."""
    manager = ClipboardManager()
    return manager.read_text_from_clipboard()
