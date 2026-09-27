# ScholarDevClaw Benchmark Summary

- Generated at: `2026-09-27T08:03:34.555562+00:00`
- Total cases: `10`
- Supported cases: `10`
- Unsupported cases: `0`
- Aggregate score: `0.725`
- Supported score: `0.725`
- Verified cases: `3` (candidates that executed)
- Verified score: `1.0` (imported/executed candidates only)

| Case | Spec | Status | Score | Candidate | Notes |
|------|------|--------|-------|-----------|-------|
| rmsnorm | rmsnorm | ast_matched | 0.75 | rmsnorm.py | Directly supported by the current extractor. |
| rope | rope | partial | 0.5 | rotary_positional_embedding.py | Directly supported by the current extractor. |
| swiglu | swiglu | ast_matched | 0.75 | swiglu.py | Directly supported by the current extractor. |
| flashattention | flashattention | partial | 0.5 | flash_attention.py | Directly supported by the current extractor. |
| lora | lora | partial | 0.5 | lora.py | Runtime spec added during hardening to cover parameter-efficient fine-tuning. |
| layernorm | layernorm | matched | 1.0 | layernorm.py | Runtime spec added during hardening to close the normalization benchmark gap. |
| gelu | gelu | matched | 1.0 | gelu.py | Runtime spec added during hardening to close the activation benchmark gap. |
| grouped_query_attention | grouped_query_attention | partial | 0.5 | grouped_query_attention.py | Current runtime spec name differs from the hardening doc shorthand. |
| alibi | alibi | ast_matched | 0.75 | alibi_positional_bias.py | Directly supported by the current extractor. |
| cosine_lr_schedule | cosine_warmup | matched | 1.0 | cosine_warmup_schedule.py | Current runtime spec name differs from the hardening doc shorthand. |
