# ScholarDevClaw Benchmark Summary

- Generated at: `2026-05-14T07:27:42.472316+00:00`
- Total cases: `10`
- Supported cases: `10`
- Unsupported cases: `0`
- Aggregate score: `0.45`
- Supported score: `0.45`

| Case | Spec | Status | Score | Candidate | Notes |
|------|------|--------|-------|-----------|-------|
| rmsnorm | rmsnorm | partial | 0.5 | rmsnorm.py | Directly supported by the current extractor. |
| rope | rope | mismatch | 0.0 | model.py | Directly supported by the current extractor. |
| swiglu | swiglu | partial | 0.5 | swiglu.py | Directly supported by the current extractor. |
| flashattention | flashattention | mismatch | 0.0 | model.py | Directly supported by the current extractor. |
| lora | lora | matched | 1.0 | lora.py | Runtime spec added during hardening to cover parameter-efficient fine-tuning. |
| layernorm | layernorm | matched | 1.0 | layernorm.py | Runtime spec added during hardening to close the normalization benchmark gap. |
| gelu | gelu | matched | 1.0 | gelu.py | Runtime spec added during hardening to close the activation benchmark gap. |
| grouped_query_attention | grouped_query_attention | partial | 0.5 | grouped_query_attention.py | Current runtime spec name differs from the hardening doc shorthand. |
| alibi | alibi | mismatch | 0.0 | model.py | Directly supported by the current extractor. |
| cosine_lr_schedule | cosine_warmup | mismatch | 0.0 | cosine_warmup_schedule.py | Current runtime spec name differs from the hardening doc shorthand. |
