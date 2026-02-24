from __future__ import annotations

import getpass
import sys
from pathlib import Path

from .store import AuthStore
from .types import AuthProvider


def cmd_auth(args):
    """Manage authentication and API keys"""
    store = AuthStore()

    if args.auth_action == "login":
        return _cmd_login(args, store)
    elif args.auth_action == "logout":
        return _cmd_logout(args, store)
    elif args.auth_action == "status":
        return _cmd_status(args, store)
    elif args.auth_action == "list":
        return _cmd_list(args, store)
    elif args.auth_action == "add":
        return _cmd_add(args, store)
    elif args.auth_action == "remove":
        return _cmd_remove(args, store)
    elif args.auth_action == "default":
        return _cmd_default(args, store)
    elif args.auth_action == "setup":
        return _cmd_setup(args, store)
    else:
        print("Unknown auth action. Use: login, logout, status, list, add, remove, default, setup")
        sys.exit(1)


def _cmd_setup(args, store: AuthStore):
    """Interactive setup wizard"""
    print("\n🔑 ScholarDevClaw Setup")
    print("=" * 50)

    status = store.get_status()

    if status.has_api_key:
        print(f"\n✅ Already authenticated!")
        print(f"   Active keys: {status.active_keys}")
        if status.user_email:
            print(f"   Email: {status.user_email}")
        print("\nRun 'scholardevclaw auth status' for more details.")
        return

    print("\nNo API key found. Let's set one up!")
    print("\nChoose your provider:")
    print("  1. Anthropic (Claude) - Default")
    print("  2. OpenAI (GPT)")
    print("  3. GitHub")
    print("  4. Custom")

    try:
        choice = input("\nProvider [1-4] (default: 1): ").strip() or "1"
    except EOFError:
        choice = "1"

    provider_map = {
        "1": AuthProvider.ANTHROPIC,
        "2": AuthProvider.OPENAI,
        "3": AuthProvider.GITHUB,
        "4": AuthProvider.CUSTOM,
    }
    provider = provider_map.get(choice, AuthProvider.ANTHROPIC)

    env_hints = {
        AuthProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
        AuthProvider.OPENAI: "OPENAI_API_KEY",
        AuthProvider.GITHUB: "GITHUB_TOKEN",
        AuthProvider.CUSTOM: "SCHOLARDEVCLAW_API_KEY",
    }

    env_var = env_hints.get(provider, "SCHOLARDEVCLAW_API_KEY")

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
    else:
        print(f"\n💡 Tip: You can also set {env_var} environment variable")
        api_key = _prompt_for_key(provider)

    name = f"{provider.value}-key"

    try:
        store.add_api_key(api_key, name, provider, set_default=True)
        print(f"\n✅ API key added successfully!")
        print(f"   Provider: {provider.value}")
        print(f"   Key: {api_key[:8]}...{api_key[-4:]}")

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
        AuthProvider.CUSTOM: "Enter your custom API key",
    }

    print(f"\n{hints.get(provider, 'Enter your API key')}")
    api_key = getpass.getpass("API Key: ").strip()

    if not api_key:
        print("❌ API key cannot be empty", file=sys.stderr)
        sys.exit(1)

    return api_key


def _cmd_login(args, store: AuthStore):
    """Login with API key"""
    if args.provider:
        try:
            provider = AuthProvider(args.provider.lower())
        except ValueError:
            print(f"Unknown provider: {args.provider}", file=sys.stderr)
            print("Supported: anthropic, openai, github, google, custom")
            sys.exit(1)
    else:
        provider = AuthProvider.ANTHROPIC

    if args.key:
        api_key = args.key
    else:
        api_key = getpass.getpass("API Key: ").strip()

    if not api_key:
        print("API key is required", file=sys.stderr)
        sys.exit(1)

    name = args.name or f"{provider.value}-key"

    try:
        added = store.add_api_key(api_key, name, provider, set_default=True)
        print(f"✅ Logged in successfully!")
        print(f"   Provider: {provider.value}")
        print(f"   Key ID: {added.id}")
        print(f"   Key: {added.mask()}")

        if args.output_json:
            import json

            print(json.dumps(added.to_dict(), indent=2))

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


def _cmd_status(args, store: AuthStore):
    """Show authentication status"""
    status = store.get_status()

    if args.output_json:
        import json

        print(json.dumps(status.to_dict(), indent=2))
        return

    print("\n🔑 Authentication Status")
    print("=" * 50)

    if not status.is_authenticated:
        print("\n❌ Not authenticated")
        print("\nRun 'scholardevclaw auth setup' to get started.")
        return

    print(f"\n✅ Authenticated")
    print(f"   API Keys: {status.active_keys} active / {status.key_count} total")

    if status.user_email:
        print(f"   Email: {status.user_email}")
    if status.user_name:
        print(f"   Name: {status.user_name}")
    if status.provider:
        print(f"   Default Provider: {status.provider}")
    print(f"   Tier: {status.subscription_tier}")

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
            print(f"   [{status_icon}] {key.name}: {key.mask()}{default}")


def _cmd_list(args, store: AuthStore):
    """List all API keys"""
    keys = store.list_api_keys()
    config = store.get_config()

    if args.output_json:
        import json

        print(json.dumps([k.to_dict() for k in keys], indent=2))
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
        print(f"  Created: {key.created_at[:10]}")
        if key.last_used:
            print(f"  Last Used: {key.last_used[:10]}")


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
        import json

        print(json.dumps(added.to_dict(), indent=2))


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
