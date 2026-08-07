# Revised Wave 1A Launch Readiness

## 1. Current Git State

- Branch: `exp/aime-int2-wave1-v100-8gpu`
- Starting HEAD: `e9dc6f0163510a5cc4eab70c300a805e1d4af8a3`

## 2. Test Results

- `70 passed` for the requested PatternKV test set.

## 3. Chunked Structure Equivalence

- CHUNKED_STRUCTURE_EQUIVALENT=true

## 4. Full Model-Level Reference Result

- CHUNKED_REFERENCE_ALGORITHM_STATUS=passed
- CHUNKED_REFERENCE_ALGORITHM_EQUIVALENT=true
- Worst logits cosine: 0.9999936819076538
- Max NLL difference: 0.0
- Attention/KV diagnostic tensor metrics: not collected in reference_v3.

## 5. Production Teacher-Forcing Result

- CHUNKED_PRODUCTION_NUMERIC_EQUIVALENT=true

## 6. Greedy Result

- CHUNKED_GREEDY_TRAJECTORY_EQUIVALENT=true

## 7. Rolling Smoke Result

- ROLLING_VARIANT_SMOKE_PASS=true
- Records: 8
- Errors: 0

## 8. Rolling Long-Smoke Result

- ROLLING_VARIANT_LONG_SMOKE_PASS=true
- Records: 8
- Errors: 0

## 9. Mixed-Key Blocker

- Mixed-Key remains blocked and is not part of Wave 1A.

## 10. Fixed Task Manifest Hash

- `configs/aime24_wave1_selected_tasks.json`: `ed3ff618c8072787a7b1687fef368c5c8d2c04801baf33fe850fca3b24a7af2e`

## 11. Generation Config Hash

- Smoke and long-smoke config hashes are recorded in result JSON files.
- Formal generation parameters were not changed.

## 12. GPU Mapping

| GPU | config | cache mode | sink | recent | residual/chunk | K bits | V bits | group size | role |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | `pattern_legacy_chunked_k2v2_r128` | `legacy_tuple_chunked` | 0 | 0 | 128 | 2 | 2 | 128 | baseline |
| 1 | `pattern_rolling_k2v2_s0_r128` | `segmented_rolling` | 0 | 128 | 128 | 2 | 2 | 128 | intervention |
| 2 | `pattern_rolling_k2v2_s64_r256` | `segmented_rolling` | 64 | 256 | 128 | 2 | 2 | 128 | intervention |
| 3 | `pattern_rolling_k4v2_s0_r128` | `segmented_rolling` | 0 | 128 | 128 | 4 | 2 | 128 | intervention |
| 4 | `pattern_rolling_k2v4_s0_r128` | `segmented_rolling` | 0 | 128 | 128 | 2 | 4 | 128 | intervention |
| 5 | `kivi_legacy_chunked_k2v2_r128` | `legacy_tuple_chunked` | 0 | 0 | 128 | 2 | 2 | 128 | baseline |
| 6 | `kivi_rolling_k2v2_s0_r128` | `segmented_rolling` | 0 | 128 | 128 | 2 | 2 | 128 | intervention |
| 7 | `kivi_rolling_k2v2_s64_r256` | `segmented_rolling` | 64 | 256 | 128 | 2 | 2 | 128 | intervention |

## 13. Full-Run Decision

- FULL_RUN_APPROVED=false
- Reason: attention/KV scalar diagnostic rows were not collected by reference_v3, so launch approval remains conservative.

## 14. Exact Launch Command

```bash
bash scripts/run_aime24_int2_wave1_8gpu.sh revised-wave1a-full
```

Do not execute this command until FULL_RUN_APPROVED=true is explicitly recorded.
