"""Hardware key support for secure credential storage.

Supports YubiKey via PIV and PKCS#11 for hardware-backed key storage.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class HardwareKeyInfo:
    """Information about a hardware key."""

    key_id: str
    key_type: str  # "yubikey", "pkcs11"
    serial: str | None
    label: str
    has_pin: bool
    slot: int | None = None


class HardwareKeyManager:
    """Manage hardware security keys."""

    def __init__(self, store_dir: str | Path | None = None):
        self.store_dir = Path(store_dir) if store_dir else Path.home() / ".scholardevclaw"
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def is_yubikey_available(self) -> bool:
        """Check if a YubiKey is connected."""
        try:
            result = subprocess.run(
                ["ykinfo", "-a"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def is_pkcs11_available(self) -> bool:
        """Check if PKCS#11 is available."""
        return os.environ.get("PKCS11_MODULE") is not None

    def list_yubikey_slots(self) -> list[HardwareKeyInfo]:
        """List available YubiKey PIV slots."""
        if not self.is_yubikey_available():
            return []

        try:
            result = subprocess.run(
                ["ykman", "piv", "info"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return []

            slots = []
            lines = result.stdout.split("\n")
            for i, line in enumerate(lines):
                if "Slot" in line and "-" in line:
                    slot_num = int(line.split(":")[0].replace("Slot", "").strip())
                    slots.append(
                        HardwareKeyInfo(
                            key_id=f"yubikey_slot_{slot_num}",
                            key_type="yubikey",
                            serial=None,
                            label=line.strip(),
                            has_pin=True,
                            slot=slot_num,
                        )
                    )
            return slots
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            return []

    def generate_yubikey_key(self, slot: int = 9, pin: str | None = None) -> HardwareKeyInfo:
        """Generate a new key in YubiKey PIV slot."""
        if not self.is_yubikey_available():
            raise RuntimeError("YubiKey not available")

        # Generate key with ykman
        cmd = ["ykman", "piv", "keys", "generate", str(slot), "-"]
        try:
            result = subprocess.run(
                cmd,
                input=pin or "",
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Failed to generate key: {result.stderr}")

            return HardwareKeyInfo(
                key_id=f"yubikey_slot_{slot}",
                key_type="yubikey",
                serial=None,
                label=f"PIV Slot {slot}",
                has_pin=pin is not None,
                slot=slot,
            )
        except FileNotFoundError:
            raise RuntimeError("ykman not installed. Install with: pip install yubikey-manager")

    def sign_with_yubikey(self, data: bytes, slot: int = 9, pin: str | None = None) -> bytes:
        """Sign data with YubiKey."""
        if not self.is_yubikey_available():
            raise RuntimeError("YubiKey not available")

        cmd = ["ykman", "piv", "sign", str(slot), "-"]
        try:
            result = subprocess.run(
                cmd,
                input=data,
                capture_output=True,
                timeout=30,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Failed to sign: {result.stderr}")
            return result.stdout
        except FileNotFoundError:
            raise RuntimeError("ykman not installed")

    def export_yubikey_pubkey(self, slot: int = 9) -> str:
        """Export public key from YubiKey."""
        if not self.is_yubikey_available():
            raise RuntimeError("YubiKey not available")

        cmd = ["ykman", "piv", "keys", "export", str(slot), "-"]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=30,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Failed to export: {result.stderr}")
            return result.stdout.decode()
        except FileNotFoundError:
            raise RuntimeError("ykman not installed")

    def encrypt_with_pkcs11(self, data: bytes, key_label: str) -> bytes:
        """Encrypt data using PKCS#11 module."""
        pkcs11_module = os.environ.get("PKCS11_MODULE")
        if not pkcs11_module:
            raise RuntimeError("PKCS11_MODULE environment variable not set")

        try:
            from pkcs11 import PKCS11  # type: ignore[import-untyped]

            pkcs11 = PKCS11(lib=pkcs11_module)
            pkcs11.initialize()
            slot = pkcs11.get_slot(token_label=key_label)
            key = slot.get_objects()[0]

            return key.encrypt(data)
        except ImportError:
            raise RuntimeError("pkcs11 package not installed. pip install pkcs11")

    def decrypt_with_pkcs11(self, encrypted_data: bytes, key_label: str) -> bytes:
        """Decrypt data using PKCS#11 module."""
        pkcs11_module = os.environ.get("PKCS11_MODULE")
        if not pkcs11_module:
            raise RuntimeError("PKCS11_MODULE environment variable not set")

        try:
            from pkcs11 import PKCS11  # type: ignore[import-untyped]

            pkcs11 = PKCS11(lib=pkcs11_module)
            pkcs11.initialize()
            slot = pkcs11.get_slot(token_label=key_label)
            key = slot.get_objects()[0]

            return key.decrypt(encrypted_data)
        except ImportError:
            raise RuntimeError("pkcs11 package not installed")

    def store_key_reference(
        self,
        key_id: str,
        key_type: str,
        label: str,
        slot: int | None = None,
    ) -> None:
        """Store a reference to a hardware key in config."""
        import json

        hw_keys_file = self.store_dir / "hardware_keys.json"
        keys = {}
        if hw_keys_file.exists():
            try:
                keys = json.loads(hw_keys_file.read_text())
            except json.JSONDecodeError:
                keys = {}

        keys[key_id] = {
            "type": key_type,
            "label": label,
            "slot": slot,
            "added_at": str(Path("/").stat().st_ctime),
        }

        hw_keys_file.write_text(json.dumps(keys, indent=2))

    def list_key_references(self) -> dict[str, dict[str, Any]]:
        """List stored hardware key references."""
        import json

        hw_keys_file = self.store_dir / "hardware_keys.json"
        if not hw_keys_file.exists():
            return {}
        try:
            return json.loads(hw_keys_file.read_text())
        except json.JSONDecodeError:
            return {}

    def remove_key_reference(self, key_id: str) -> bool:
        """Remove a hardware key reference."""
        import json

        hw_keys_file = self.store_dir / "hardware_keys.json"
        if not hw_keys_file.exists():
            return False

        try:
            keys = json.loads(hw_keys_file.read_text())
            if key_id in keys:
                del keys[key_id]
                hw_keys_file.write_text(json.dumps(keys, indent=2))
                return True
        except json.JSONDecodeError:
            pass
        return False


class YubiKeyPIV:
    """High-level YubiKey PIV operations."""

    def __init__(self, pin: str | None = None):
        self.pin = pin

    def authenticate(self, pin: str) -> bool:
        """Verify PIN and authenticate to PIV."""
        if not HardwareKeyManager().is_yubikey_available():
            raise RuntimeError("YubiKey not available")

        result = subprocess.run(
            ["ykman", "piv", "verify", "9", pin],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0

    def generate_csr(self, slot: int, cn: str, country: str = "US") -> str:
        """Generate a CSR for the key in a slot."""
        if not HardwareKeyManager().is_yubikey_available():
            raise RuntimeError("YubiKey not available")

        # Use openssl to generate CSR from public key
        # This is a simplified version - real implementation would be more complex
        raise NotImplementedError("CSR generation requires openssl integration")
