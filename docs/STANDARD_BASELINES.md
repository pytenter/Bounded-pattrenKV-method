# Standard Baseline Methods

This repository contains several historical and exploratory runners. For paper-v2
reproduction runs, the canonical baseline/method set is exactly:

```text
fp16
kivi_paper_g128
patternkv_paper
```

Treat these method names as the source of truth when asking GPT/Codex to run,
compare, summarize, or continue experiments in this repository.

## Canonical Methods

| Method name | Role | Required configuration |
| --- | --- | --- |
| `fp16` | Full precision reference baseline | FP16 model/KV cache; no KV quantization. |
| `kivi_paper_g128` | KIVI paper-aligned INT2 baseline | `k_bits=2`, `v_bits=2`, `group_size=128`, `residual_length=128`, K per-channel, V per-token, official KIVI backend. |
| `patternkv_paper` | PatternKV paper-v2 INT2 method | `k_bits=2`, `v_bits=2`, `group_size=128`, `residual_length=128`, `num_k_base=32`, `num_v_base=32`, pattern group `128`, post-RoPE pattern selection. |

## Non-Canonical Legacy Names

The following names exist in older scripts or debug code, but they are not the
standard baseline set for paper-v2 reproduction:

```text
kivi
kivi_official
kivi_original_g32
patternkv
```

Use them only when reproducing an explicitly labeled legacy/debug experiment.

## Current GSM8K Full Run Audit

The latest complete GSM8K full run is:

```text
results/paper_repro_v2/gsm8k_full_2048
reports/paper_repro_v2/gsm8k_full_2048/summary_4gpu_sequence.md
```

It was checked against the canonical baseline set above:

| Method | Result files | Unique GSM8K problem IDs | Config check |
| --- | ---: | ---: | --- |
| `fp16` | 1319 | 1319 | PASS |
| `kivi_paper_g128` | 1319 | 1319 | PASS |
| `patternkv_paper` | 1319 | 1319 | PASS |

The recorded `quantization_config` fields in the GSM8K result JSON files match
the required standard configurations for all three methods. The older directory
`results/paper_repro_v2/gsm8k_full` is not the latest complete three-method run.

