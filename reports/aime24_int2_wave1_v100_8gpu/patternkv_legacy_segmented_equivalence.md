# PatternKV Legacy/Segmented Equivalence Report

## 1. Original Failed Assumption

The previous validation compared:

```text
legacy tuple residual_length=128
vs
segmented rolling recent_length=128
```

That comparison was invalid. At generated token `128` for `aime24:p12:s0:seed12042`, the observed state was:

```text
legacy packed=128
rolling segmented packed=0, pending=64, recent=128
```

This is an expected semantic difference between chunked residual buffering and stable rolling-recent protection, not sufficient evidence of an implementation bug in the rolling variant.

## 2. Legacy Chunked Semantics

`legacy_tuple_chunked` uses:

```text
[packed history]
[FP16 residual chunk buffer]
```

The residual buffer accumulates until `residual_length`, then the full chunk is assigned, gated, quantized, packed, and cleared. For total token count `T`:

```text
packed=floor(T/residual_length)*residual_length
chunk=T%residual_length
```

## 3. Rolling Recent Semantics

`segmented_rolling` uses:

```text
[sink_fp16]
[packed history]
[pending_fp16]
[rolling_recent_fp16]
```

It always protects the latest `recent_length` non-sink tokens in FP16. It is a CoT cache-cadence variant, not a legacy-equivalent baseline.

## 4. Why They Are Not Equivalent

Chunked buffering periodically clears FP16 history. Rolling recent preserves a stable suffix. With short prompts, the first chunked flush can happen while rolling mode still protects the same tokens in recent.

## 5. Segmented Chunked Implementation

Added explicit cache modes:

```text
legacy_tuple_chunked
segmented_chunked
segmented_rolling
```

`segmented_chunked` uses the segmented container while preserving legacy cadence:

```text
sink_fp16 = empty
packed_history
pending_fp16 = chunk buffer
recent_fp16 = empty
```

Decode now follows legacy ordering: append current token to the FP16 chunk buffer, compute current attention using that FP16 buffer, then flush a full chunk before serializing cache for the next step.

## 6. Chunked Container Equivalence

Reports:

```text
reports/aime24_int2_wave1_v100_8gpu/equivalence_chunked_level1/summary.md
reports/aime24_int2_wave1_v100_8gpu/equivalence_chunked_level2/production/teacher_forcing_summary.md
reports/aime24_int2_wave1_v100_8gpu/chunked_container_equivalence.md
```

Level 1 synthetic:

```text
CHUNKED_LEVEL1_PASS=true
```

Level 2 production structure:

```text
CHUNKED_LEVEL2_STRUCTURE_PASS=true
```

The prior structural cadence mismatch is fixed for `legacy_tuple_chunked` vs `segmented_chunked`.

## 7. Greedy Equivalence

Level 3 greedy was run for the two fixed equivalence tasks:

```text
aime24:p12:s0:seed12042 -> first divergence token 355
aime24:p14:s0:seed14042 -> first divergence token 288
```

Result:

```text
CHUNKED_LEVEL3_GREEDY_PASS=false
```

## 8. Rolling Variant Status

The rolling implementation remains present and semantically distinct:

```text
ROLLING_VARIANT_IMPLEMENTED=true
```

The previous isolated rolling smoke and long-smoke evidence remains historical, but this round did not rerun the revised three-mode smoke matrix.

```text
ROLLING_VARIANT_SMOKE_PASS=false
ROLLING_VARIANT_LONG_SMOKE_PASS=false
```

## 9. Revised Wave 1A Design

Revised manifest:

```text
reports/aime24_int2_wave1_v100_8gpu/revised_wave1a_manifest.md
```

Future diagnostic names must distinguish baseline and variant, for example:

```text
pattern_legacy_chunked_k2v2_r128
pattern_rolling_k2v2_s0_r128
pattern_rolling_k2v2_s64_r256
pattern_rolling_k4v2_s0_r128
pattern_rolling_k2v4_s0_r128
```

## 10. Full-Run Decision

Full is not approved. Although chunked structural cadence is now equivalent, strict production assignment/gate equivalence and greedy equivalence did not pass.

```text
CHUNKED_LEVEL1_PASS=true
CHUNKED_LEVEL2_STRUCTURE_PASS=true
CHUNKED_LEVEL2_REFERENCE_PASS=false
CHUNKED_LEVEL2_PRODUCTION_PASS=false
CHUNKED_LEVEL3_GREEDY_PASS=false
CHUNKED_CONTAINER_EQUIVALENT=false
ROLLING_VARIANT_IMPLEMENTED=true
ROLLING_VARIANT_SMOKE_PASS=false
ROLLING_VARIANT_LONG_SMOKE_PASS=false
FULL_RUN_APPROVED=false
```
