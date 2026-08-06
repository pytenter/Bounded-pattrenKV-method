# PatternKV Legacy/Segmented Equivalence Report

## Current Modes

```text
legacy_tuple_chunked
segmented_chunked
segmented_rolling
```

`legacy_tuple_chunked` and `segmented_chunked` are the container-equivalence pair. `segmented_rolling` is the CoT cache-cadence variant and is not expected to be structurally equivalent to legacy chunk flushing.

## Original Failed Assumption

The earlier comparison between legacy chunked residual buffering and rolling recent protection failed at generated token `128` because the two modes intentionally maintain different FP16 regions.

```text
legacy packed=128
rolling segmented packed=0, pending=64, recent=128
```

This is now recorded as a design distinction, not a rolling implementation bug.

## Chunked Equivalence v2

The first production mismatch was reproduced:

```text
aime24:p12:s0:seed12042 checkpoint=256 layer=1
k_assignment_disagreement_rate=0.0009765625
count/denominator=2/2048
```

Trace showed the raw K window and centroid bank were already slightly different before assignment, so the root cause was not assignment tie-breaking. The source was production numeric drift from using a different V attention execution path in `segmented_chunked` than legacy.

The fix makes `segmented_chunked` use the same fused V attention call form as legacy when packed history and FP16 chunk buffer are both present.

## Results

```text
tests=62 passed
CHUNKED_STRUCTURE_EQUIVALENT=true
CHUNKED_REFERENCE_ALGORITHM_EQUIVALENT=false
CHUNKED_PRODUCTION_NUMERIC_EQUIVALENT=true
CHUNKED_GREEDY_TRAJECTORY_EQUIVALENT=true
ROLLING_VARIANT_SMOKE_PASS=false
ROLLING_VARIANT_LONG_SMOKE_PASS=false
FULL_RUN_APPROVED=false
```

## Reports

```text
reports/aime24_int2_wave1_v100_8gpu/assignment_mismatch_root_cause.md
reports/aime24_int2_wave1_v100_8gpu/assignment_mismatch_repro/repro_summary.md
reports/aime24_int2_wave1_v100_8gpu/equivalence_chunked_level2/production_v2/teacher_forcing_summary.md
reports/aime24_int2_wave1_v100_8gpu/equivalence_chunked_level3/greedy_v2/greedy_summary.md
```

## Full-Run Decision

Revised Wave 1A full remains blocked because the full independent reference backend and current-HEAD rolling smoke/long-smoke are not complete.
