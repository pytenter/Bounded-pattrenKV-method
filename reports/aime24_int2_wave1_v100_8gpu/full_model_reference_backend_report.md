# Full Model Reference Backend Report

Status: passed.

Reference backend: `reference`
Comparison: `legacy_tuple_chunked` vs `segmented_chunked`
Output directory: `reports/aime24_int2_wave1_v100_8gpu/equivalence_chunked_level2/reference_v3`

## Results

- Tasks: p12_s0 and p14_s0
- Checkpoints: 128, 256, 512, 1024, 2048, 4096
- Checkpoint rows: 12
- Layer rows: 384
- First mismatches: 0
- Worst logits cosine: 0.9999936819076538
- Max NLL difference: 0.0
- Assignment/gate disagreement rows with failures: 0

## Diagnostics

- `cache_structure.csv`, `centroid_metrics.csv`, `assignment_metrics.csv`, `v_gate_metrics.csv`, and `logits_metrics.csv` are populated from the model-level run.
- `kv_reconstruction_metrics.csv` and `attention_metrics.csv` contain explicit `not_collected_in_reference_v3` status rows; v3 did not persist raw reconstructed KV or attention tensors.

```text
CHUNKED_STRUCTURE_EQUIVALENT=true
CHUNKED_REFERENCE_ALGORITHM_STATUS=passed
CHUNKED_REFERENCE_ALGORITHM_EQUIVALENT=true
CHUNKED_PRODUCTION_NUMERIC_EQUIVALENT=true
CHUNKED_GREEDY_TRAJECTORY_EQUIVALENT=true
ROLLING_VARIANT_SMOKE_PASS=true
ROLLING_VARIANT_LONG_SMOKE_PASS=true
FULL_RUN_APPROVED=false
```
