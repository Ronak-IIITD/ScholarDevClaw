# ScholarDevClaw Benchmark Summary

- Generated at: `2026-05-14T06:42:12.918101+00:00`
- Total cases: `10`
- Supported cases: `7`
- Unsupported cases: `3`
- Aggregate score: `0.15`
- Supported score: `0.214`

| Case | Spec | Status | Score | Candidate | Notes |
|------|------|--------|-------|-----------|-------|
| rmsnorm | rmsnorm | partial | 0.5 | rmsnorm.py | Directly supported by the current extractor. |
| rope | rope | mismatch | 0.0 | model.py | Directly supported by the current extractor. |
| swiglu | swiglu | partial | 0.5 | swiglu.py | Directly supported by the current extractor. |
| flashattention | flashattention | mismatch | 0.0 | model.py | Directly supported by the current extractor. |
| lora | None | unsupported_spec | 0.0 | None | Tracked by the hardening doc but not implemented as a runtime spec yet. |
| layernorm | None | unsupported_spec | 0.0 | None | Tracked by the hardening doc but not implemented as a runtime spec yet. |
| gelu | None | unsupported_spec | 0.0 | None | Tracked by the hardening doc but not implemented as a runtime spec yet. |
| grouped_query_attention | grouped_query_attention | partial | 0.5 | model.py | Current runtime spec name differs from the hardening doc shorthand. |
| alibi | alibi | mismatch | 0.0 | model.py | Directly supported by the current extractor. |
| cosine_lr_schedule | cosine_warmup | missing_candidate | 0.0 | None | Current runtime spec name differs from the hardening doc shorthand. |
