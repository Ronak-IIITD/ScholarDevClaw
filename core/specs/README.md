# nanogpt-opt v1 — Frozen Specs (Step 1)

Frozen from `src/scholardevclaw/research_intelligence/extractor.py:PAPER_SPECS`.
Only these 10 are supported. Everything else in `PAPER_SPECS` is out of scope for v1.

- `rmsnorm`, `preln_transformer`, `qknorm` — normalization
- `swiglu` — FFN activation (covers GEGLU pattern via same MLP target)
- `flashattention2`, `grouped_query_attention` — attention
- `rope`, `alibi` — position encoding
- `lion` — optimizer (replaces Adam/AdamW)
- `cosine_warmup` — scheduler

## Rules
- Do NOT add new specs here without a matching libcst transformer in `patch_generation/generator.py` + an eval case on `test_repos/nanogpt`.
- Do NOT edit these JSONs to change `target_patterns` without updating `mapping/engine.py` aliases + tests.
- Source of truth for v1 is these files, loaded via `src/scholardevclaw/nano_specs.py`. `extractor.py` remains legacy.

See `manifest.json` for version + scope.
