from __future__ import annotations

import inspect

from scholardevclaw.tui.screens import ProviderSetupScreen


def test_provider_setup_submitted_handlers_are_field_scoped():
    source = inspect.getsource(ProviderSetupScreen)

    assert '@on(Input.Submitted, "#setup-provider")' in source
    assert '@on(Input.Submitted, "#setup-model")' in source
    assert '@on(Input.Submitted, "#setup-key")' in source
    assert "@on(Input.Submitted)\n    def on_input_submitted" not in source
