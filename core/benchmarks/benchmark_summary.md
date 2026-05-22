# ScholarDevClaw Benchmark Summary

- Generated at: `2026-05-22T11:46:59.449835+00:00`
- Total cases: `10`
- Supported cases: `10`
- Unsupported cases: `0`
- Aggregate score: `1.0`
- Supported score: `1.0`

| Case | Spec | Status | Score | Candidate | Notes |
|------|------|--------|-------|-----------|-------|
| rmsnorm | rmsnorm | matched | 1.0 | rmsnorm.py | Directly supported by the current extractor. |
| rope | rope | matched | 1.0 | rotary_positional_embedding.py | Directly supported by the current extractor. |
| swiglu | swiglu | matched | 1.0 | swiglu.py | Directly supported by the current extractor. |
| flashattention | flashattention | matched | 1.0 | flash_attention.py | Directly supported by the current extractor. |
| lora | lora | matched | 1.0 | lora.py | Runtime spec added during hardening to cover parameter-efficient fine-tuning. |
| layernorm | layernorm | matched | 1.0 | layernorm.py | Runtime spec added during hardening to close the normalization benchmark gap. |
| gelu | gelu | matched | 1.0 | gelu.py | Runtime spec added during hardening to close the activation benchmark gap. |
| grouped_query_attention | grouped_query_attention | matched | 1.0 | grouped_query_attention.py | Current runtime spec name differs from the hardening doc shorthand. |
| alibi | alibi | matched | 1.0 | alibi_positional_bias.py | Directly supported by the current extractor. |
| cosine_lr_schedule | cosine_warmup | matched | 1.0 | cosine_warmup_schedule.py | Current runtime spec name differs from the hardening doc shorthand. |
