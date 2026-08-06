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
