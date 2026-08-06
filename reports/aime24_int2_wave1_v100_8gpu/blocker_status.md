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
- Restored segmented PatternKV dynamic centroid pack-window updates, V gate metadata, sample reset, dynamic stats, and bitwidth accounting.
- Added dynamic centroid, V gate, runtime reset, and synthetic legacy/segmented equivalence tests.
- Ran isolated Wave 1A smoke and long-smoke validation for the restored dynamic centroid path without overwriting existing Wave 1A result files.
- Added dual-path PatternKV cache switch and model-level legacy/segmented validation harness.
- Fixed equivalence tasks to `aime24:p12:s0:seed12042` and `aime24:p14:s0:seed14042`.
- Ran production Level 2 teacher-forcing through 4096 generated tokens for both fixed samples.

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

- Resolve or explicitly re-scope the Level 2 structural mismatch: at `aime24:p12:s0:seed12042`, checkpoint `128`, layer `0`, legacy has packed `128` tokens and one dynamic centroid update, while segmented has packed `0`, pending `64`, recent `128`, and no dynamic update.
- Complete teacher-forcing legacy-vs-segmented reference backend only after the structural cadence mismatch is addressed or accepted as intentionally out-of-scope.
- Complete greedy 1024-token legacy-vs-segmented generation equivalence only after Level 2 structure passes.
- Keep `FULL_RUN_APPROVED=false` until strict Level 2/3 equivalence evidence exists.
- Run six-config Wave 1A full only after `reports/aime24_int2_wave1_v100_8gpu/patternkv_legacy_segmented_equivalence.md` ends with `FULL_RUN_APPROVED=true`.
- Generate Wave 1A summary artifacts under `reports/aime24_int2_wave1_v100_8gpu/wave1a/`.
