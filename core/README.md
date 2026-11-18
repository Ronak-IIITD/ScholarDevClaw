# ScholarDevClaw

ScholarDevClaw provides `nanogpt-opt sdpa`, a static preview for migrating
minGPT-style causal attention to PyTorch's native SDPA. It shows a unified diff
and explains skipped patterns. The target checkout stays untouched.

The GitHub developer preview runs on Python 3.10–3.12 without PyTorch or API
credentials. The proposed code targets PyTorch 2.0 or later. The command reports
`verification: "not_run"` for its static preview.

- [Download the wheel or source archive](https://github.com/Ronak-IIITD/ScholarDevClaw/releases/tag/preview-v2.1.0a1)
- [Reproduce the pinned minGPT demo](https://github.com/Ronak-IIITD/ScholarDevClaw#reproduce-the-demo)
- [Contribute an attention fixture](https://github.com/Ronak-IIITD/ScholarDevClaw/blob/main/CONTRIBUTING.md)

The distribution includes the existing `scholardevclaw` CLI. See the
[CLI guide](https://github.com/Ronak-IIITD/ScholarDevClaw/blob/main/GUIDE.md) for
that interface.
