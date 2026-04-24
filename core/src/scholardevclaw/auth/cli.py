from __future__ import annotations

import getpass
import json
import sys
from datetime import datetime
from pathlib import Path

from .store import AuthStore
from .types import AuthProvider


def cmd_auth(args):
    """Manage authentication and API keys"""
    store = AuthStore()

    action = args.auth_action
    dispatch = {
        "login": _cmd_login,
        "logout": _cmd_logout,
        "status": _cmd_status,
        "list": _cmd_list,
        "add": _cmd_add,
        "remove": _cmd_remove,
        "default": _cmd_default,
        "setup": _cmd_setup,
        "rotate": _cmd_rotate,
        "audit": _cmd_audit,
        "export": _cmd_export,
        "import": _cmd_import,
        "encrypt": _cmd_encrypt,
        "profiles": _cmd_profiles,
        "usage": _cmd_usage,
        "expiry": _cmd_expiry,
        "validate": _cmd_validate,
    }

    handler = dispatch.get(action)
    if handler:
        return handler(args, store)
    else:
        print(
            "Unknown auth action. Use: login, logout, status, list, add, remove, "
            "default, setup, rotate, audit, export, import, encrypt, profiles, usage, expiry, validate"
        )
        sys.exit(1)


# ------------------------------------------------------------------
# Setup wizard
# ------------------------------------------------------------------


def _cmd_setup(args, store: AuthStore):
    """Interactive setup wizard"""
    print("\n🔑 ScholarDevClaw Setup")
    print("=" * 50)

    status = store.get_status()

    if status.has_api_key:
        print("\n✅ Already authenticated!")
        print(f"   Active keys: {status.active_keys}")
        if status.user_email:
            print(f"   Email: {status.user_email}")
        print("\nRun 'scholardevclaw auth status' for more details.")
        return

    print("\nNo API key found. Let's set one up!")
    print("\nChoose your LLM provider:")
    print("  1.  Anthropic (Claude) — Default")
    print("  2.  OpenAI (GPT)")
    print("  3.  Ollama (Local — no key needed)")
    print("  4.  Groq")
    print("  5.  Mistral AI")
    print("  6.  DeepSeek")
    print("  7.  Cohere")
    print("  8.  OpenRouter")
    print("  9.  Together AI")
    print("  10. Fireworks AI")
    print("  11. GitHub Copilot")
    print("  12. Azure OpenAI")
    print("  13. GitHub (identity only)")
    print("  14. Custom")

    try:
        choice = input("\nProvider [1-14] (default: 1): ").strip() or "1"
    except EOFError:
        choice = "1"

    provider_map = {
        "1": AuthProvider.ANTHROPIC,
        "2": AuthProvider.OPENAI,
        "3": AuthProvider.OLLAMA,
        "4": AuthProvider.GROQ,
        "5": AuthProvider.MISTRAL,
        "6": AuthProvider.DEEPSEEK,
        "7": AuthProvider.COHERE,
        "8": AuthProvider.OPENROUTER,
        "9": AuthProvider.TOGETHER,
        "10": AuthProvider.FIREWORKS,
        "11": AuthProvider.GITHUB_COPILOT,
        "12": AuthProvider.AZURE_OPENAI,
        "13": AuthProvider.GITHUB,
        "14": AuthProvider.CUSTOM,
    }
    provider = provider_map.get(choice, AuthProvider.ANTHROPIC)

    env_var = provider.env_var_name

    import os

    env_key = os.environ.get(env_var)

    if env_key:
        print(f"\n✓ Found {env_var} in environment")
        try:
            use_env = input("Use this key? [Y/n]: ").strip().lower()
        except EOFError:
            use_env = "y"

        if use_env != "n":
            api_key = env_key
        else:
            api_key = _prompt_for_key(provider)
    elif provider == AuthProvider.OLLAMA:
        print("\n✓ Ollama runs locally — no API key needed.")
        print(f"  Make sure Ollama is running at {provider.default_base_url}")
        api_key = "ollama-local"
    else:
        print(f"\n💡 Tip: You can also set {env_var} environment variable")
        api_key = _prompt_for_key(provider)

    name = f"{provider.value}-key"

    try:
        store.add_api_key(api_key, name, provider, set_default=True)
        print("\n✅ API key added successfully!")
        print(f"   Provider: {provider.display_name}")
        print(f"   Key: {api_key[:4]}...{api_key[-2:]}")
        if provider.default_base_url:
            print(f"   Base URL: {provider.default_base_url}")

        email = input("\nEmail (optional, for sync): ").strip()
        if email:
            name_input = input("Your name (optional): ").strip()
            store.create_profile(email=email, name=name_input or None)
            print("✅ Profile created!")

        print("\n🎉 Setup complete! You're ready to use ScholarDevClaw.")
        print("\nQuick start:")
        print("  scholardevclaw agent")
        print("  scholardevclaw analyze ./my-project")

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


def _prompt_for_key(provider: AuthProvider) -> str:
    """Prompt user for API key"""
    hints = {
        AuthProvider.ANTHROPIC: "Get your key at: https://console.anthropic.com",
        AuthProvider.OPENAI: "Get your key at: https://platform.openai.com/api-keys",
        AuthProvider.GITHUB: "Get a token at: https://github.com/settings/tokens",
        AuthProvider.GITHUB_COPILOT: "Use your GitHub token (ghp_... or github_pat_...)",
        AuthProvider.GROQ: "Get your key at: https://console.groq.com/keys",
        AuthProvider.MISTRAL: "Get your key at: https://console.mistral.ai/api-keys",
        AuthProvider.DEEPSEEK: "Get your key at: https://platform.deepseek.com/api_keys",
        AuthProvider.COHERE: "Get your key at: https://dashboard.cohere.com/api-keys",
        AuthProvider.OPENROUTER: "Get your key at: https://openrouter.ai/keys",
        AuthProvider.TOGETHER: "Get your key at: https://api.together.xyz/settings/api-keys",
        AuthProvider.FIREWORKS: "Get your key at: https://fireworks.ai/api-keys",
        AuthProvider.AZURE_OPENAI: "Get your key from Azure Portal > OpenAI resource > Keys",
        AuthProvider.CUSTOM: "Enter your custom API key",
    }

    print(f"\n{hints.get(provider, 'Enter your API key')}")
    if provider.key_format_hint:
        print(f"  Format: {provider.key_format_hint}")
    api_key = getpass.getpass("API Key: ").strip()

    if not api_key:
        print("❌ API key cannot be empty", file=sys.stderr)
        sys.exit(1)

    return api_key


# ------------------------------------------------------------------
# Login / Logout
# ------------------------------------------------------------------


def _cmd_login(args, store: AuthStore):
    """Login with API key"""
    if args.provider:
        try:
            provider = AuthProvider(args.provider.lower())
        except ValueError:
            print(f"Unknown provider: {args.provider}", file=sys.stderr)
            print(
                "Supported: anthropic, openai, ollama, groq, mistral, deepseek, cohere, "
                "openrouter, together, fireworks, github_copilot, azure_openai, github, google, custom"
            )
            sys.exit(1)
    else:
        provider = AuthProvider.ANTHROPIC

    if args.key:
        import logging

        logging.getLogger(__name__).warning(
            "Passing API keys via --key CLI argument is insecure (visible in shell history "
            "and process listings). Prefer interactive prompt or environment variables."
        )
        print(
            "⚠️  Warning: --key argument is visible in shell history. Prefer interactive prompt.",
            file=sys.stderr,
        )
        api_key = args.key
    else:
        api_key = getpass.getpass("API Key: ").strip()

    if not api_key:
        print("API key is required", file=sys.stderr)
        sys.exit(1)

    name = args.name or f"{provider.value}-key"

    try:
        added = store.add_api_key(api_key, name, provider, set_default=True)
        print("✅ Logged in successfully!")
        print(f"   Provider: {provider.value}")
        print(f"   Key ID: {added.id}")
        print(f"   Key: {added.mask()}")

        if args.output_json:
            print(json.dumps(added.to_safe_dict(), indent=2))

    except Exception as e:
        print(f"❌ Login failed: {e}", file=sys.stderr)
        sys.exit(1)


def _cmd_logout(args, store: AuthStore):
    """Logout and clear credentials"""
    if not args.force:
        print("⚠️  This will remove all saved API keys and profile data.")
        try:
            confirm = input("Continue? [y/N]: ").strip().lower()
        except EOFError:
            confirm = "n"

        if confirm != "y":
            print("Cancelled.")
            return

    if store.logout():
        print("✅ Logged out successfully. All credentials removed.")
    else:
        print("ℹ️  No credentials to remove.")


# ------------------------------------------------------------------
# Status / List
# ------------------------------------------------------------------


def _cmd_status(args, store: AuthStore):
    """Show authentication status"""
    status = store.get_status()

    if args.output_json:
        print(json.dumps(status.to_dict(), indent=2))
        return

    print("\n🔑 Authentication Status")
    print("=" * 50)

    if not status.is_authenticated:
        print("\n❌ Not authenticated")
        print("\nRun 'scholardevclaw auth setup' to get started.")
        return

    print("\n✅ Authenticated")
    print(f"   API Keys: {status.active_keys} active / {status.key_count} total")

    if status.user_email:
        print(f"   Email: {status.user_email}")
    if status.user_name:
        print(f"   Name: {status.user_name}")
    if status.provider:
        print(f"   Default Provider: {status.provider}")
    print(f"   Tier: {status.subscription_tier}")

    # Encryption status
    if store.is_encryption_enabled():
        print("   🔒 Encryption: enabled")
    else:
        print("   🔓 Encryption: disabled")

    profile = store.get_profile()
    if profile:
        print(f"\n   Profile ID: {profile.id}")
        if profile.created_at:
            print(f"   Created: {profile.created_at[:10]}")

    keys = store.list_api_keys()
    if keys:
        print("\n📋 API Keys:")
        for key in keys:
            status_icon = "✓" if key.is_valid() else "✗"
            default = " (default)" if key.id == store.get_config().default_key_id else ""
            scope_str = f" [{key.scope.value}]"
            expiry = ""
            if key.expires_at:
                try:
                    exp_dt = datetime.fromisoformat(key.expires_at)
                    days_left = (exp_dt - datetime.now()).days
                    if days_left < 0:
                        expiry = " ⚠️ EXPIRED"
                    elif days_left <= 7:
                        expiry = f" ⚠️ expires in {days_left}d"
                except ValueError:
                    pass
            print(f"   [{status_icon}] {key.name}: {key.mask()}{default}{scope_str}{expiry}")

    # Expiring keys warning
    expiring = store.get_expiring_keys(within_days=7)
    if expiring:
        print(f"\n⚠️  {len(expiring)} key(s) expiring within 7 days!")

    # Rotation recommendations
    needing_rotation = store.get_keys_needing_rotation(days=90)
    if needing_rotation:
        print(f"\n🔄 {len(needing_rotation)} key(s) older than 90 days — consider rotating")


def _cmd_list(args, store: AuthStore):
    """List all API keys"""
    keys = store.list_api_keys()
    config = store.get_config()

    if args.output_json:
        print(json.dumps([k.to_safe_dict() for k in keys], indent=2))
        return

    if not keys:
        print("\n❌ No API keys found")
        print("Run 'scholardevclaw auth login' to add one.")
        return

    print("\n📋 API Keys")
    print("=" * 50)

    for key in keys:
        status = "✓ active" if key.is_valid() else "✗ inactive"
        default = " [DEFAULT]" if key.id == config.default_key_id else ""
        print(f"\n  ID: {key.id}{default}")
        print(f"  Name: {key.name}")
        print(f"  Provider: {key.provider.value}")
        print(f"  Key: {key.mask()}")
        print(f"  Status: {status}")
        print(f"  Scope: {key.scope.value}")
        print(f"  Created: {key.created_at[:10]}")
        if key.last_used:
            print(f"  Last Used: {key.last_used[:10]}")
        if key.expires_at:
            print(f"  Expires: {key.expires_at[:10]}")
        if key.rotation_recommended_at:
            print(f"  🔄 Rotation recommended since: {key.rotation_recommended_at[:10]}")


# ------------------------------------------------------------------
# Add / Remove / Default
# ------------------------------------------------------------------


def _cmd_add(args, store: AuthStore):
    """Add a new API key"""
    if not args.key:
        args.key = getpass.getpass("API Key: ").strip()

    if not args.key:
        print("API key is required", file=sys.stderr)
        sys.exit(1)

    provider = AuthProvider(args.provider) if args.provider else AuthProvider.CUSTOM
    name = args.name or f"{provider.value}-key"

    added = store.add_api_key(args.key, name, provider, set_default=args.default)

    print(f"✅ API key added: {added.id}")
    print(f"   Key: {added.mask()}")

    if args.output_json:
        print(json.dumps(added.to_safe_dict(), indent=2))


def _cmd_remove(args, store: AuthStore):
    """Remove an API key"""
    if not args.key_id:
        print("Key ID is required. Use: auth remove <key-id>", file=sys.stderr)
        sys.exit(1)

    if store.remove_api_key(args.key_id):
        print(f"✅ API key removed: {args.key_id}")
    else:
        print(f"❌ Key not found: {args.key_id}", file=sys.stderr)
        sys.exit(1)


def _cmd_default(args, store: AuthStore):
    """Set the default API key"""
    if not args.key_id:
        print("Key ID is required. Use: auth default <key-id>", file=sys.stderr)
        sys.exit(1)

    if store.set_default_key(args.key_id):
        print(f"✅ Default key set: {args.key_id}")
    else:
        print(f"❌ Key not found: {args.key_id}", file=sys.stderr)
        sys.exit(1)


# ------------------------------------------------------------------
# Key rotation
# ------------------------------------------------------------------


def _cmd_rotate(args, store: AuthStore):
    """Rotate an API key"""
    key_id = getattr(args, "key_id", None)
    if not key_id:
        print("Key ID is required. Use: auth rotate <key-id>", file=sys.stderr)
        sys.exit(1)

    new_key = getattr(args, "new_key", None)
    if not new_key:
        new_key = getpass.getpass("New API Key: ").strip()

    if not new_key:
        print("New API key is required", file=sys.stderr)
        sys.exit(1)

    reason = getattr(args, "reason", None) or "Manual rotation"

    rotated = store.rotate_api_key(key_id, new_key, reason=reason)
    if rotated:
        print("✅ Key rotated successfully!")
        print(f"   New Key ID: {rotated.id}")
        print(f"   Key: {rotated.mask()}")
        print(f"   Rotations: {len(rotated.rotation_history)}")
    else:
        print(f"❌ Key not found: {key_id}", file=sys.stderr)
        sys.exit(1)


# ------------------------------------------------------------------
# Audit log
# ------------------------------------------------------------------


def _cmd_audit(args, store: AuthStore):
    """View audit log"""
    if not store._audit:
        print("Audit logging is disabled.", file=sys.stderr)
        sys.exit(1)

    limit = getattr(args, "limit", 20) or 20
    key_id = getattr(args, "key_id", None)

    events = store._audit.get_events(key_id=key_id, limit=limit)

    output_json = getattr(args, "output_json", False)
    if output_json:
        print(json.dumps([e.to_dict() for e in events], indent=2))
        return

    if not events:
        print("\nNo audit events found.")
        return

    print(f"\n📜 Audit Log (last {len(events)} events)")
    print("=" * 60)

    for event in events:
        ts = event.timestamp[:19]
        icon = "✅" if event.success else "❌"
        fp = f" [{event.key_fingerprint[:8]}]" if event.key_fingerprint else ""
        provider = f" ({event.provider})" if event.provider else ""
        print(f"  {icon} {ts} | {event.event_type.value}{provider}{fp}")
        if event.details:
            for k, v in event.details.items():
                if v is not None:
                    print(f"     {k}: {v}")


# ------------------------------------------------------------------
# Export / Import
# ------------------------------------------------------------------


def _cmd_export(args, store: AuthStore):
    """Export credentials"""
    fmt = getattr(args, "format", "json") or "json"
    output = getattr(args, "output", None)
    # SECURITY: Default to redacted (no plaintext keys) unless explicitly requested
    include_keys = getattr(args, "include_keys", False)

    if fmt == "env":
        include_all = bool(getattr(args, "include_all", False))
        content = store.export_env(include_all=include_all)
    else:
        content = store.export_json(include_keys=include_keys)

    if output:
        Path(output).write_text(content)
        print(f"✅ Exported to {output}")
    else:
        print(content)


def _cmd_import(args, store: AuthStore):
    """Import credentials"""
    source = getattr(args, "source", None)
    fmt = getattr(args, "format", "auto") or "auto"

    if not source:
        print("Source file is required. Use: auth import <file>", file=sys.stderr)
        sys.exit(1)

    source_path = Path(source)
    if not source_path.exists():
        print(f"❌ File not found: {source}", file=sys.stderr)
        sys.exit(1)

    content = source_path.read_text()

    # Auto-detect format
    if fmt == "auto":
        if source.endswith(".csv"):
            fmt = "1password"
        elif source.endswith(".env") or source.startswith(".env"):
            fmt = "env"
        else:
            fmt = "json"

    if fmt == "env":
        count, errors = store.import_keys_from_env(content)
    elif fmt == "1password":
        count, errors = store.import_keys_from_1password(content)
    else:
        count, errors = store.import_keys_from_json(content)

    print(f"✅ Imported {count} key(s)")
    if errors:
        for err in errors:
            print(f"  ⚠️  {err}")


# ------------------------------------------------------------------
# Encryption
# ------------------------------------------------------------------


def _cmd_encrypt(args, store: AuthStore):
    """Manage encryption at rest"""
    action = getattr(args, "encrypt_action", "status")

    if action == "enable":
        password = getpass.getpass("Master password: ").strip()
        if not password:
            print("Password is required", file=sys.stderr)
            sys.exit(1)

        # SECURITY: Enforce minimum password strength
        if len(password) < 12:
            print("Password must be at least 12 characters long", file=sys.stderr)
            sys.exit(1)
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        if not (has_upper and has_lower and has_digit):
            print(
                "Password must contain at least one uppercase letter, one lowercase letter, and one digit",
                file=sys.stderr,
            )
            sys.exit(1)

        confirm = getpass.getpass("Confirm password: ").strip()
        if password != confirm:
            print("Passwords do not match", file=sys.stderr)
            sys.exit(1)

        try:
            store.enable_encryption(password)
            print("✅ Encryption enabled. All credentials are now encrypted at rest.")
            print("⚠️  Remember your master password — it cannot be recovered!")
        except RuntimeError as e:
            print(f"❌ {e}", file=sys.stderr)
            sys.exit(1)

    elif action == "disable":
        password = getpass.getpass("Master password: ").strip()
        if store.disable_encryption(password):
            print("✅ Encryption disabled. Credentials stored in plaintext.")
        else:
            print("❌ Wrong password or encryption not enabled.", file=sys.stderr)
            sys.exit(1)

    elif action == "status":
        if store.is_encryption_enabled():
            print("🔒 Encryption: enabled")
        else:
            print("🔓 Encryption: disabled")
            print("Enable with: scholardevclaw auth encrypt enable")

    else:
        print("Usage: auth encrypt [enable|disable|status]")


# ------------------------------------------------------------------
# Multi-profile / workspace
# ------------------------------------------------------------------


def _cmd_profiles(args, store: AuthStore):
    """Manage credential profiles (workspaces)"""
    action = getattr(args, "profile_action", "list")

    if action == "list":
        profiles = store.list_profiles()
        if not profiles:
            print("\nNo saved profiles.")
            print("Save current config: scholardevclaw auth profiles save <name>")
            return
        print("\n📁 Saved Profiles:")
        for p in profiles:
            print(f"  - {p}")

    elif action == "save":
        name = getattr(args, "profile_name", None)
        if not name:
            print("Profile name required.", file=sys.stderr)
            sys.exit(1)
        store.save_profile_as(name)
        print(f"✅ Profile saved: {name}")

    elif action == "load":
        name = getattr(args, "profile_name", None)
        if not name:
            print("Profile name required.", file=sys.stderr)
            sys.exit(1)
        if store.load_profile(name):
            print(f"✅ Switched to profile: {name}")
        else:
            print(f"❌ Profile not found: {name}", file=sys.stderr)
            sys.exit(1)

    elif action == "delete":
        name = getattr(args, "profile_name", None)
        if not name:
            print("Profile name required.", file=sys.stderr)
            sys.exit(1)
        if store.delete_profile(name):
            print(f"✅ Profile deleted: {name}")
        else:
            print(f"❌ Profile not found: {name}", file=sys.stderr)
            sys.exit(1)

    else:
        print("Usage: auth profiles [list|save|load|delete] <name>")


# ------------------------------------------------------------------
# Usage / rate limiting
# ------------------------------------------------------------------


def _cmd_usage(args, store: AuthStore):
    """View key usage statistics"""
    key_id = getattr(args, "key_id", None)
    output_json = getattr(args, "output_json", False)

    usage = store.get_key_usage(key_id)

    if output_json:
        print(json.dumps(usage, indent=2))
        return

    if not usage:
        print("\nNo usage data available.")
        return

    if key_id:
        # Single key stats
        print(f"\n📊 Usage Stats for {key_id}")
        print("=" * 40)
        for k, v in usage.items():
            print(f"  {k}: {v}")
    else:
        # All keys
        print("\n📊 Usage Stats")
        print("=" * 50)
        for kid, stats in usage.items():
            print(f"\n  Key: {kid}")
            print(f"    Total: {stats['total_requests']}")
            print(f"    Last minute: {stats['requests_last_minute']}")
            print(f"    Last hour: {stats['requests_last_hour']}")
            if stats.get("is_rate_limited"):
                print("    ⚠️  RATE LIMITED")


# ------------------------------------------------------------------
# Expiry management
# ------------------------------------------------------------------


def _cmd_expiry(args, store: AuthStore):
    """Manage key expiration"""
    action = getattr(args, "expiry_action", "check")

    if action == "check":
        expiring = store.get_expiring_keys(within_days=30)
        if not expiring:
            print("\n✅ No keys expiring within 30 days.")
            return

        print(f"\n⚠️  {len(expiring)} key(s) expiring soon:")
        for key in expiring:
            days_left = 0
            if key.expires_at:
                try:
                    exp_dt = datetime.fromisoformat(key.expires_at)
                    days_left = (exp_dt - datetime.now()).days
                except ValueError:
                    pass
            print(f"  - {key.name} ({key.id}): {days_left} days left")

    elif action == "set":
        key_id = getattr(args, "key_id", None)
        expires = getattr(args, "expires_at", None)
        if not key_id or not expires:
            print("Usage: auth expiry set <key-id> <date>", file=sys.stderr)
            sys.exit(1)
        try:
            store.set_key_expiry(key_id, expires)
            print(f"✅ Expiry set for {key_id}: {expires}")
        except ValueError as e:
            print(f"❌ {e}", file=sys.stderr)
            sys.exit(1)

    elif action == "deactivate":
        deactivated = store.deactivate_expired_keys()
        if deactivated:
            print(f"✅ Deactivated {len(deactivated)} expired key(s):")
            for key in deactivated:
                print(f"  - {key.name} ({key.id})")
        else:
            print("✅ No expired keys to deactivate.")

    else:
        print("Usage: auth expiry [check|set|deactivate]")


# ------------------------------------------------------------------
# Validate API keys
# ------------------------------------------------------------------


def _cmd_validate(args, store: AuthStore):
    """Validate API keys by making a test request"""
    import os

    # Get provider from args or environment, default to openrouter
    provider_arg = getattr(args, "provider", None)
    provider_name = (
        provider_arg or os.environ.get("SCHOLARDEVCLAW_API_PROVIDER", "") or "openrouter"
    )

    # Get API key from args --api-key flag or environment variable
    api_key_arg = getattr(args, "api_key", None)
    env_var_name = f"{provider_name.upper()}_API_KEY"
    api_key = api_key_arg or os.environ.get(env_var_name, "")

    if not api_key:
        print(f"Error: No API key found for provider '{provider_name}'", file=sys.stderr)
        print(f"Set API key in env var: {env_var_name}")
        print(
            "Or pass directly: scholardevclaw auth validate --provider <provider> --api-key <key>"
        )
        sys.exit(1)

    print(f"Validating API key for {provider_name}...")

    # Get model from args or use provider default
    model_arg = getattr(args, "model", None)

    try:
        from scholardevclaw.auth.types import AuthProvider
        from scholardevclaw.llm.client import DEFAULT_MODELS, LLMClient

        auth_provider = AuthProvider(provider_name)
        model = model_arg or DEFAULT_MODELS.get(auth_provider, "")

        # OpenRouter requires HTTP-Referer and X-Title for free tier
        if auth_provider == AuthProvider.OPENROUTER:
            # Build headers with OpenRouter-specific additions
            openrouter_extra = {
                "HTTP-Referer": "https://scholardevclaw.dev",
                "X-Title": "ScholarDevClaw",
            }
            client = LLMClient(
                auth_provider,
                api_key=api_key,
                model=model,
                base_url="https://openrouter.ai/api/v1",
            )
            # Patch the _build_headers to include OpenRouter extras
            original_build_headers = client._build_headers

            def _openrouter_headers(key: str) -> dict:
                h = original_build_headers(key)
                h.update(openrouter_extra)
                return h

            client._build_headers = _openrouter_headers
        else:
            client = LLMClient.from_provider(provider_name, api_key=api_key, model=model)

        # Make a simple test request
        response = client.chat("Say 'OK' if you receive this.", max_tokens=10)

        if response:
            print("✅ API key is valid!")
            print(f"   Model: {response.model}")
            print(f"   Provider: {response.provider}")
            print(f"   Latency: {response.latency_ms:.0f}ms")
        else:
            print("❌ API key validation failed (empty response)", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        print(f"❌ API key validation failed: {e}", file=sys.stderr)
        sys.exit(1)


# ------------------------------------------------------------------
# List available providers
# ------------------------------------------------------------------


def _cmd_list_providers(args, store: AuthStore):
    """List all available LLM providers"""
    from scholardevclaw.auth.types import AuthProvider

    print("Available LLM providers:")
    print("=" * 50)

    # Get LLM providers only
    llm_providers = [p for p in AuthProvider if p.is_llm_provider]

    for provider in sorted(llm_providers, key=lambda p: p.value):
        # Get default model for this provider
        from scholardevclaw.llm.client import DEFAULT_MODELS

        default_model = DEFAULT_MODELS.get(provider, "")

        # Get environment variable name
        env_var = provider.env_var_name

        # Check if key is configured
        import os

        configured = bool(os.environ.get(env_var, ""))

        status = "✓ configured" if configured else "✗ not set"
        print(f"  {provider.value:<15} {status}")
        if default_model:
            print(f"    Default model: {default_model}")
        print(f"    Env var: {env_var}")
        print()
