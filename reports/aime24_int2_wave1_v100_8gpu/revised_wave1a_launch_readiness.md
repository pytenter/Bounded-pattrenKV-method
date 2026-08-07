# Revised Wave 1A Launch Readiness

## Current Git State

- Branch: `exp/aime-int2-wave1-v100-8gpu`
- Starting HEAD for final audit: `42294b01b9f31743ec82885c560b769bf3bc7f6f`

## Test Results

- `75 passed` for the requested PatternKV test set with reference v4 metric tests.

## Final Approval Table

| Metric | Threshold | Worst result | Pass |
| --- | ---: | ---: | --- |
| Structure mismatch | `0` | `0` | PASS |
| K assignment mismatch | `0` | `0` | PASS |
| V assignment mismatch | `0` | `0` | PASS |
| V gate mismatch | `0` | `0` | PASS |
| Reconstructed K cosine | `>=0.9999` | `0.999982059002` | PASS |
| Reconstructed V cosine | `>=0.9999` | `0.999967694283` | PASS |
| Attention score cosine | `>=0.9999` | `0.999998807907` | PASS |
| Attention symmetric KL | `<=1e-4` | `0` | PASS |
| Attention output cosine | `>=0.9999` | `0.999998986721` | PASS |
| Post-o-proj cosine | `>=0.9999` | `0.999999284744` | PASS |
| Logits cosine | `>=0.9999` | `0.999993681908` | PASS |
| Top-1 agreement | `100%` | `100%` | PASS |
| Max teacher NLL abs difference | `<=0.01` | `0` | PASS |
| Rolling smoke errors | `0` | `0` | PASS |
| Rolling long-smoke errors | `0` | `0` | PASS |

## Reference v4

- Directory: `reports/aime24_int2_wave1_v100_8gpu/equivalence_chunked_level2/reference_v4`
- First mismatch: `none`
- First metric failure: `none`
- REFERENCE_METRIC_PASS=true

## Prior Required Evidence

- CHUNKED_PRODUCTION_NUMERIC_EQUIVALENT=true
- CHUNKED_GREEDY_TRAJECTORY_EQUIVALENT=true
- ROLLING_VARIANT_SMOKE_PASS=true
- ROLLING_VARIANT_LONG_SMOKE_PASS=true
- Mixed-Key remains blocked outside Wave 1A.

## GPU Mapping

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

## Full-Run Decision

- FULL_RUN_APPROVED=true
- LAUNCH_READINESS_STATUS=approved

## Exact Launch Command

```bash
bash scripts/run_aime24_int2_wave1_8gpu.sh revised-wave1a-full
```

This audit did not execute the full run.
