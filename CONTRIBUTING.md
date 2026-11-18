# Contributing to ScholarDevClaw

The developer preview focuses on one source migration from minGPT-style causal
attention to native PyTorch SDPA. Small fixtures that document what the tool
supports or why it skips a pattern are a good first contribution.

## Run the focused checks

From the repository root, use Python 3.12:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e './core[dev]'
(cd core && python -m pytest -q tests/unit/test_sdpa_preview.py)
(cd core && ruff check src tests && ruff format --check src tests)
python scripts/demo_sdpa.py
```

The demo needs Git and fetches the pinned minGPT revision. The static preview
tests run with the development extra. Install `./core[dev,torch]` to run the
fixture's CPU output and gradient check as well.

## Add an attention pattern fixture

1. Start with the source fixture in
   [test_sdpa_preview.py](core/tests/unit/test_sdpa_preview.py). Minimize it to
   the code that distinguishes your case and keep its imports and bindings.
2. State whether the tool should emit a diff or skip the site. For a rewrite,
   assert the key edit and check that a second preview emits no diff. For a
   refusal, assert the reason and that the source bytes match the input.
3. If the example comes from an upstream repo, include a link to the exact
   commit and file. Explain how custom masks, scaling, or dropout affect it.
4. Run the focused checks and include the test command and outcome in the PR.

Use the [attention pattern issue template](https://github.com/Ronak-IIITD/ScholarDevClaw/issues/new?template=attention_pattern.md)
for a new example. For changes elsewhere in the project, see
[AGENTS.md](AGENTS.md) for subsystem guidance and
[UPDATES.md](UPDATES.md) for recent changes.
