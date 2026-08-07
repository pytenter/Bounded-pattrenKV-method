# Revised Wave 1A Manifest

Status: not approved for launch.

The intended diagnostic configs are:

| GPU | config | cache mode | sink | recent | residual/chunk | K/V bits | role |
| ---: | --- | --- | ---: | ---: | ---: | --- | --- |
| 0 | `pattern_legacy_chunked_k2v2_r128` | `legacy_tuple_chunked` | 0 | 0 | 128 | K2/V2 | legacy baseline |
| 1 | `pattern_rolling_k2v2_s0_r128` | `segmented_rolling` | 0 | 128 | 128 | K2/V2 | rolling variant |
| 2 | `pattern_rolling_k2v2_s64_r256` | `segmented_rolling` | 64 | 256 | 128 | K2/V2 | sink+recent variant |
| 3 | `pattern_rolling_k4v2_s0_r128` | `segmented_rolling` | 0 | 128 | 128 | K4/V2 | key precision variant |
| 4 | `pattern_rolling_k2v4_s0_r128` | `segmented_rolling` | 0 | 128 | 128 | K2/V4 | value precision variant |
| 5 | `kivi_legacy_chunked_k2v2_r128` | `legacy_tuple_chunked` | 0 | 0 | 128 | K2/V2 | KIVI legacy baseline |
| 6 | `kivi_rolling_k2v2_s0_r128` | `segmented_rolling` | 0 | 128 | 128 | K2/V2 | KIVI rolling variant |
| 7 | `kivi_rolling_k2v2_s64_r256` | `segmented_rolling` | 64 | 256 | 128 | K2/V2 | KIVI sink+recent variant |

Launch remains blocked until:

```text
CHUNKED_CONTAINER_EQUIVALENT=true
ROLLING_VARIANT_LONG_SMOKE_PASS=true
FULL_RUN_APPROVED=true
```

## 2026-08-07 Reference And Rolling Update

- Full model-level reference backend ran for p12/p14 through 4096 checkpoints with `LEVEL2_REFERENCE_PASS=true` and `first_mismatch_count=0`.
- Rolling smoke and long-smoke were rerun on current final code with 8 records each, 0 runtime errors, and packed/assignment/gate alignment preserved.
- `FULL_RUN_APPROVED=false` remains conservative because reference_v3 did not collect raw KV reconstruction and attention tensor scalar diagnostics.
