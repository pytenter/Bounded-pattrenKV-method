# Revised AIME24 Wave 1A Full Run Manifest

## Git and Environment

- Branch: `exp/aime-int2-wave1-v100-8gpu`
- HEAD: `1caeff1237877e2bf1be283d57657f28ecc872db`
- Python: `/home/qinch2023/miniconda3/envs/patternkv-v100/bin/python`
- Torch: `2.4.1+cu118`
- CUDA runtime: `11.8`
- Model path: `/home/qinch2023/modelscope_models/DeepSeek-R1-Distill-Llama-8B`
- Launch timestamp: `2026-08-07T03:29:59.350537+00:00`

## Task Manifest

- Path: `configs/aime24_wave1_selected_tasks.json`
- SHA256: `ed3ff618c8072787a7b1687fef368c5c8d2c04801baf33fe850fca3b24a7af2e`
- Task count: `12`
- Cohort: fixed 12-task paired diagnostic cohort, not a full AIME accuracy headline benchmark.

## Resolved Generation Config

```json
{
  "batch_size": 1,
  "configs": [
    "pattern_legacy_chunked_k2v2_r128",
    "pattern_rolling_k2v2_s0_r128",
    "pattern_rolling_k2v2_s64_r256",
    "pattern_rolling_k4v2_s0_r128",
    "pattern_rolling_k2v4_s0_r128",
    "kivi_legacy_chunked_k2v2_r128",
    "kivi_rolling_k2v2_s0_r128",
    "kivi_rolling_k2v2_s64_r256"
  ],
  "do_sample": true,
  "dtype": "float16",
  "manifest_hash": "ed3ff618c8072787a7b1687fef368c5c8d2c04801baf33fe850fca3b24a7af2e",
  "max_model_len": 131072,
  "max_new_tokens": 32768,
  "model": "/home/qinch2023/modelscope_models/DeepSeek-R1-Distill-Llama-8B",
  "seed": 42,
  "task_count": 12,
  "temperature": 0.6,
  "top_p": 0.95
}
```

FORMAL_GENERATION_CONFIG_HASH=`a7d6b2f8bab37893b6331c66b3e5eb6a`

## GPU Mapping

| GPU | config | method | cache mode | sink | recent | residual | K bits | V bits | group | role |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | `pattern_legacy_chunked_k2v2_r128` | `patternkv` | `legacy_tuple_chunked` | 0 | 0 | 128 | 2 | 2 | 128 | PatternKV legacy baseline |
| 1 | `pattern_rolling_k2v2_s0_r128` | `patternkv` | `segmented_rolling` | 0 | 128 | 128 | 2 | 2 | 128 | rolling-recent intervention |
| 2 | `pattern_rolling_k2v2_s64_r256` | `patternkv` | `segmented_rolling` | 64 | 256 | 128 | 2 | 2 | 128 | Sink+Recent combined protection |
| 3 | `pattern_rolling_k4v2_s0_r128` | `patternkv` | `segmented_rolling` | 0 | 128 | 128 | 4 | 2 | 128 | Key precision intervention |
| 4 | `pattern_rolling_k2v4_s0_r128` | `patternkv` | `segmented_rolling` | 0 | 128 | 128 | 2 | 4 | 128 | Value precision intervention |
| 5 | `kivi_legacy_chunked_k2v2_r128` | `kivi_official` | `legacy_tuple_chunked` | 0 | 0 | 128 | 2 | 2 | 128 | KIVI legacy baseline |
| 6 | `kivi_rolling_k2v2_s0_r128` | `kivi_official` | `segmented_rolling` | 0 | 128 | 128 | 2 | 2 | 128 | KIVI rolling control |
| 7 | `kivi_rolling_k2v2_s64_r256` | `kivi_official` | `segmented_rolling` | 64 | 256 | 128 | 2 | 2 | 128 | KIVI Sink+Recent control |

## Output Locations

- Results: `results/aime24_int2_wave1_v100_8gpu_revised_full/wave1a`
- Run logs/status: `run/aime24_int2_wave1_v100_8gpu_revised_full`
- Reports: `reports/aime24_int2_wave1_v100_8gpu/revised_wave1a_full`
