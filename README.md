# ScholarDevClaw

Preview a source migration from minGPT-style causal attention to PyTorch's
`scaled_dot_product_attention` (SDPA). `nanogpt-opt` shows a unified diff,
reports why it skipped unfamiliar patterns, and leaves the checkout untouched.

## Install the developer preview

Use Python 3.10–3.12 and Git. The preview runs without PyTorch, a GPU, Node.js,
or API credentials. The proposed code uses PyTorch 2.0 or later.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install https://github.com/Ronak-IIITD/ScholarDevClaw/releases/download/preview-v2.1.0a1/scholardevclaw-2.1.0a1-py3-none-any.whl
nanogpt-opt --version
```

The [GitHub prerelease](https://github.com/Ronak-IIITD/ScholarDevClaw/releases/tag/preview-v2.1.0a1)
also includes the source archive and SHA-256 checksums. To install from a
checkout, run `python -m pip install ./core` in its root.

## Reproduce the demo

```bash
git clone --branch preview-v2.1.0a1 https://github.com/Ronak-IIITD/ScholarDevClaw.git
cd ScholarDevClaw
python scripts/demo_sdpa.py
```

The script fetches minGPT at
[`37baab71b9ab`](https://github.com/karpathy/minGPT/blob/37baab71b9abea1b76ab957409a1cc2fbfba8a26/mingpt/model.py)
into a temporary directory. It shows the proposed diff, checks that the checkout
stayed clean, and shows what happens when a copy uses custom attention scaling.
![Terminal demo showing the minGPT diff and custom-scale refusal](docs/sdpa-preview.gif)

To try a checkout of your own:

```bash
nanogpt-opt sdpa /path/to/repo
nanogpt-opt sdpa /path/to/repo --json
```

## Supported patterns

| Attention pattern | Preview result |
|---|---|
| Pinned minGPT's manual causal attention | Proposes an SDPA call using the existing mask and attention dropout mode |
| Already calls native SDPA, as in nanoGPT | Reports `already_migrated` and emits no diff for the site |
| Custom scaling, masks, or ambiguous bindings | Reports `unsupported` with a reason and emits no diff for the site |

The matcher expects minGPT's forward layout, packed Q/K/V projections, standard
scaling, lower-triangular mask buffer, and `nn.Dropout`. It keeps parameter names
and the mask buffer. The command performs static analysis and reports
`verification: "not_run"`. Runtime numerical checks and applying SDPA patches
are future milestones.

## Contributing

Start with the [contributor guide](CONTRIBUTING.md) or
[submit an attention pattern](https://github.com/Ronak-IIITD/ScholarDevClaw/issues/new?template=attention_pattern.md)
with a small source fixture and the expected result.

The existing `analyze`, `suggest`, `apply`, and `validate` commands cover the
earlier transformer patch workflow. See the [nanoGPT demo recording](demo-nanogpt.mp4)
and the [legacy CLI guide](GUIDE.md) for those workflows and the wider project.

[MIT License](LICENSE)
