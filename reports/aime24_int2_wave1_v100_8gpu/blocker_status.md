# AIME24 INT2 Wave1 Blocker Status

Date: 2026-08-06

## Completed

- Created branch `exp/aime-int2-wave1-v100-8gpu`.
- Located local model at `/home/qinch2023/modelscope_models/DeepSeek-R1-Distill-Llama-8B`.
- Verified runtime with `/home/qinch2023/miniconda3/envs/patternkv-v100/bin/python`.
- Generated fixed 12-task selected manifest at `configs/aime24_wave1_selected_tasks.json`.
- Generated fixed 8-task calibration manifest at `configs/aime24_wave1_calibration_tasks.json`.
- Added Wave 1 runner at `scripts/run_aime24_int2_wave1_8gpu.sh`.
- Added cache schema audit at `reports/aime24_int2_wave1_v100_8gpu/cache_schema_before.md`.
- Added segmented cache infrastructure with permanent FP16 sink, packed history, pending history, and rolling FP16 recent.
- Added semantic tests for prefill partition, rolling decode, K2/K4/V2/V4 packed cache, assignment alignment, long rolling, and serialization.
- Disabled mixed-Key placeholder configs for Wave 1A.

## Wave 1A Status

Wave 1A is now scoped to six real uniform-bitwidth configs:

```text
kivi_k2v2_s0_r128
pattern_k2v2_s0_r128
kivi_k2v2_s64_r256
pattern_k2v2_s64_r256
pattern_k4v2_s0_r128
pattern_k2v4_s0_r128
```

Runner modes:

```bash
bash scripts/run_aime24_int2_wave1_8gpu.sh wave1a-smoke
bash scripts/run_aime24_int2_wave1_8gpu.sh wave1a-long-smoke
bash scripts/run_aime24_int2_wave1_8gpu.sh wave1a-full
bash scripts/run_aime24_int2_wave1_8gpu.sh status
bash scripts/run_aime24_int2_wave1_8gpu.sh summarize-wave1a
```

The script prints:

```text
Wave 1A uses 6 of 8 GPUs
```

## Wave 1B Blocker

The following mixed-Key configs are `blocked_wave1b` and must not be launched or summarized as completed Wave 1A methods:

```text
pattern_magnitude_kmix_v2_s0_r128
pattern_queryaware_kmix_v2_s0_r128
```

Their mask files are deterministic placeholders and have been renamed with `PLACEHOLDER_NOT_FOR_RESULTS`. They are not real calibration masks and are not evidence for mixed-key INT2/INT4 claims.

## Remaining

- Run `wave1a-smoke` with `PATTERNKV_CACHE_VALIDATE=1`.
- Run `wave1a-long-smoke` with `max_new_tokens=4096`.
- Generate `reports/aime24_int2_wave1_v100_8gpu/wave1a_long_smoke.md`.
- Commit infrastructure after tests and long smoke pass.
- Run six-config Wave 1A full only after long smoke passes.
- Generate Wave 1A summary artifacts under `reports/aime24_int2_wave1_v100_8gpu/wave1a/`.
