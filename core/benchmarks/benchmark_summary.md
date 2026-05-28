# ScholarDevClaw Benchmark Summary

- Generated at: `2026-05-28T09:06:19.537448+00:00`
- Total cases: `10`
- Supported cases: `10`
- Unsupported cases: `0`
- Aggregate score: `0.8`
- Supported score: `0.8`

| Case | Spec | Status | Score | Candidate | Notes |
|------|------|--------|-------|-----------|-------|
| rmsnorm | rmsnorm | matched | 1.0 | rmsnorm.py | Directly supported by the current extractor. |
| rope | rope | partial | 0.5 | rotary_positional_embedding.py | Directly supported by the current extractor. |
| swiglu | swiglu | matched | 1.0 | swiglu.py | Directly supported by the current extractor. |
| flashattention | flashattention | partial | 0.5 | flash_attention.py | Directly supported by the current extractor. |
| lora | lora | partial | 0.5 | lora.py | Runtime spec added during hardening to cover parameter-efficient fine-tuning. |
| layernorm | layernorm | matched | 1.0 | layernorm.py | Runtime spec added during hardening to close the normalization benchmark gap. |
| gelu | gelu | matched | 1.0 | gelu.py | Runtime spec added during hardening to close the activation benchmark gap. |
| grouped_query_attention | grouped_query_attention | partial | 0.5 | grouped_query_attention.py | Current runtime spec name differs from the hardening doc shorthand. |
| alibi | alibi | matched | 1.0 | alibi_positional_bias.py | Directly supported by the current extractor. |
| cosine_lr_schedule | cosine_warmup | matched | 1.0 | cosine_warmup_schedule.py | Current runtime spec name differs from the hardening doc shorthand. |
