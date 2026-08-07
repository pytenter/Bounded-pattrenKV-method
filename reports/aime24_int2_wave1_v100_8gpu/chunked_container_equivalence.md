# Chunked Container Equivalence

## Scope

This report compares `legacy_tuple_chunked` against `segmented_chunked`.

The comparison no longer treats `segmented_rolling` as a legacy-equivalent cache. The chunked baseline is:

```text
[packed history]
[FP16 residual chunk buffer]
```

The segmented chunked container stores the same state as:

```text
sink_fp16 = empty
packed_history
pending_fp16 = chunk buffer
recent_fp16 = empty
```

## Implementation

- Added `cache_mode` with explicit values `segmented_chunked` and `segmented_rolling`.
- Added `patternkv_cache_mode` with `legacy_tuple_chunked`, `segmented_chunked`, and `segmented_rolling`.
- Added chunked prefill cadence: complete chunks are packed, remainder remains in the FP16 chunk buffer.
- Added chunked decode cadence: current token is appended to the FP16 chunk buffer, attention uses the FP16 chunk for the current step, and full chunks are flushed before serializing the cache for the next step.
- Kept rolling recent semantics unchanged.

## Test Result

```text
50 passed
```

## Level 1

```text
CHUNKED_LEVEL1_PASS=true
```

## Level 2 Production

Output directory:

```text
reports/aime24_int2_wave1_v100_8gpu/equivalence_chunked_level2/production
```

Summary:

```text
CHUNKED_LEVEL2_STRUCTURE_PASS=true
CHUNKED_LEVEL2_PRODUCTION_PASS=false
```

The structural mismatch from the earlier rolling comparison is fixed for chunked mode. All checkpoint/layer structural counts match:

```text
packed tokens
chunk tokens
centroid counts
centroid update counts
assignment tokens
V gate tokens
```

The remaining failure is numerical/algorithmic state divergence after the first chunk:

```text
first mismatch: aime24:p12:s0:seed12042, checkpoint 256, layer 1
mismatch type: k_assignment_disagreement_rate
rate: 0.0009765625
```

Top-1 logits agreement stayed true at all recorded checkpoints, but production logits cosine fell below the requested threshold at later checkpoints.

## Level 3 Greedy

Output directory:

```text
reports/aime24_int2_wave1_v100_8gpu/equivalence_chunked_level3
```

Result:

```text
CHUNKED_LEVEL3_GREEDY_PASS=false
```

First divergence:

```text
aime24:p12:s0:seed12042 -> token 355
aime24:p14:s0:seed14042 -> token 288
```

## Decision

```text
CHUNKED_CONTAINER_EQUIVALENT=false
FULL_RUN_APPROVED=false
```

The segmented chunked container now reproduces the legacy chunked cadence structurally, but strict model-level equivalence is not yet proven because production assignment/gate and greedy generation diverge.

## 2026-08-07 Reference And Rolling Update

- Full model-level reference backend ran for p12/p14 through 4096 checkpoints with `LEVEL2_REFERENCE_PASS=true` and `first_mismatch_count=0`.
- Rolling smoke and long-smoke were rerun on current final code with 8 records each, 0 runtime errors, and packed/assignment/gate alignment preserved.
- `FULL_RUN_APPROVED=false` remains conservative because reference_v3 did not collect raw KV reconstruction and attention tensor scalar diagnostics.
