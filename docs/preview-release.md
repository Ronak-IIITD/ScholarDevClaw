ScholarDevClaw 2.1.0a1 is a developer preview of a source migration from
minGPT-style causal attention to native PyTorch SDPA. The command shows a diff,
explains which sites it skipped, and leaves target files untouched.

With Python 3.10–3.12, activate a fresh virtual environment, download the wheel
below, then run:

```bash
python -m pip install ./scholardevclaw-2.1.0a1-py3-none-any.whl
nanogpt-opt --version
nanogpt-opt sdpa /path/to/minGPT
```

The source archive can also be installed with `python -m pip install`.
`SHA256SUMS` lists checksums for both archives. Release CI installs each
archive in a fresh environment and runs the pinned minGPT demo without
PyTorch. PyTorch 2.0 or later is required to run the proposed code.

The preview checks the minGPT attention source pattern and reports
`verification: "not_run"`. Applying SDPA changes and checking numerical
equivalence are planned separately.

[Install and reproduce the demo](https://github.com/Ronak-IIITD/ScholarDevClaw/blob/preview-v2.1.0a1/README.md)
