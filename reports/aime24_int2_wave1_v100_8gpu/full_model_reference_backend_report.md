# Full Model Reference Backend Report

Status: passed.

Reference backend: `reference`
Comparison: `legacy_tuple_chunked` vs `segmented_chunked`
Latest output directory: `reports/aime24_int2_wave1_v100_8gpu/equivalence_chunked_level2/reference_v4`

## Reference v4 Results

- Tasks: p12_s0 and p14_s0
- Checkpoints: 128, 256, 512, 1024, 2048, 4096
- Metric layers: 0, 7, 15, 23, 31
- First mismatches: 0
- Metric failures: 0
- Worst reconstructed K cosine: `0.999982059002`
- Worst reconstructed V cosine: `0.999967694283`
- Worst attention score cosine: `0.999998807907`
- Worst attention symmetric KL: `0`
- Worst attention output cosine: `0.999998986721`
- Worst post-o-proj cosine: `0.999999284744`
- Worst logits cosine: `0.999993681908`
- Max NLL difference: `0`

## Status Flags

```text
CHUNKED_STRUCTURE_EQUIVALENT=true
CHUNKED_REFERENCE_ALGORITHM_STATUS=passed
CHUNKED_REFERENCE_ALGORITHM_EQUIVALENT=true
FULL_RUN_APPROVED=true
LAUNCH_READINESS_STATUS=approved
```
