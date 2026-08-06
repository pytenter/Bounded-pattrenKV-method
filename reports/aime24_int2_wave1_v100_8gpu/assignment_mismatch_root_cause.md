# PatternKV Assignment Mismatch Root Cause

## 1. Original mismatch

```text
sample: aime24:p12:s0:seed12042
checkpoint: 256
layer: 1
metric: k_assignment_disagreement_rate
rate: 0.0009765625
count/denominator: 2/2048
```

## 2. Exact disagreement count

The reproduced mismatch count is `2` differing K assignments out of `2048` compared assignment slots.

## 3. Global token alignment

`legacy_tuple_chunked` and `segmented_chunked` were structurally aligned at checkpoint 256:

```text
packed_k_tokens: 256 == 256
chunk_k_tokens: 64 == 64
centroid_count: 34 == 34
assignment_tokens: 256 == 256
```

The comparison denominator is `8 KV heads x 256 packed tokens = 2048`.

## 4. Input-window equivalence

Trace at the second packed chunk showed that inputs were already different before the assignment argmin:

```json
{
  "chunk_assignment_denominator": 1024,
  "chunk_assignment_diff_count": 2,
  "first_difference": {
    "batch_index": 0,
    "centroid_count": 34,
    "centroid_dtype": "torch.float16",
    "global_token_index": 252,
    "input_dtype": "torch.float16",
    "kv_head": 6,
    "legacy": {
      "assignment": 33,
      "top2_fp16_distances": [
        7.203125,
        7.20703125
      ],
      "top2_fp16_indices": [
        33,
        30
      ],
      "top2_fp16_margin": 0.00390625,
      "top2_fp32_distances": [
        7.205078125,
        7.2099609375
      ],
      "top2_fp32_indices": [
        33,
        30
      ],
      "top2_fp32_margin": 0.0048828125
    },
    "local_chunk_token_index": 124,
    "segmented": {
      "assignment": 30,
      "top2_fp16_distances": [
        7.20703125,
        7.2109375
      ],
      "top2_fp16_indices": [
        30,
        33
      ],
      "top2_fp16_margin": 0.00390625,
      "top2_fp32_distances": [
        7.208984375,
        7.2099609375
      ],
      "top2_fp32_indices": [
        33,
        30
      ],
      "top2_fp32_margin": 0.0009765625
    }
  },
  "k_centroid_exact_equal": false,
  "k_centroid_max_abs_error": 0.00390625,
  "k_centroid_relative_l2_error": 5.706316005671397e-05,
  "k_window_exact_equal": false,
  "k_window_max_abs_error": 0.0078125,
  "k_window_relative_l2_error": 0.0002711507841013372
}

```

The first differing local assignment in the second chunk was at KV head `6`, local chunk token `124`, global packed token `252`.

## 5. Centroid-bank equivalence

The centroid bank was also already slightly different before assignment:

```text
k_centroid_exact_equal=false
k_centroid_max_abs_error=0.00390625
k_centroid_relative_l2_error=5.706316005671397e-05
```

This rules out a pure argmin tie-breaking or block-size-only cause.

## 6. FP16/FP32 top-2 distances

For the first differing assignment:

```text
legacy FP16 top2: 33 @ 7.203125, 30 @ 7.20703125, margin 0.00390625
legacy FP32 top2: 33 @ 7.205078125, 30 @ 7.2099609375, margin 0.0048828125
segmented FP16 top2: 30 @ 7.20703125, 33 @ 7.2109375, margin 0.00390625
segmented FP32 top2: 33 @ 7.208984375, 30 @ 7.2099609375, margin 0.0009765625
```

The mismatch is not an exact tie. It is downstream of input/centroid drift.

## 7. Tie margin

Margins were small, but the primary evidence is that the compared raw K window and centroid bank were not exact-equal before assignment.

## 8. Block-size analysis

Independent reference tests now cover full-tensor assignment and block sizes `1, 2, 4, 32, 256`. The segmented block implementation matches full reference on exact tie and cross-block cases.

## 9. Tie-breaking analysis

The reference evaluator documents PyTorch `argmin` tie behavior as selecting the lowest index. Existing block traversal with `<` preserves the earlier lower-index winner across ascending centroid blocks.

## 10. V gate analysis

V assignment/gate mismatches appeared together with K mismatches in the old production path. After the production fix, Level 2 production v2 has zero first mismatches through 4096 tokens.

## 11. Root-cause classification

```text
kernel_numeric_difference
```

The segmented chunked path was structurally equivalent, but its production V attention combined packed and FP16 chunk contributions with a different execution path from legacy. The resulting small hidden-state drift changed the next chunk's raw K window and centroid bank before assignment.

## 12. Code fix

For `segmented_chunked`, V attention now uses the same fused production call shape as legacy when both quantized history and FP16 chunk buffer are present:

```text
cuda_attn_v_fused_with_base(..., attn_f=attn_weights_f, v_full=chunk_buffer)
```

This avoids splitting packed and FP16 contributions into separate operations in chunked legacy-equivalence mode. Rolling mode is unchanged.

## 13. Level 2 reference v2

A standalone reference evaluator was added in `bench/patternkv_equivalence_reference.py`, with tests for assignment, tie-breaking, block-size equivalence, V gate, dequant helpers, attention, and logits metrics.

A full model-level reference backend that bypasses all production PatternKV kernels was not completed in this round.

```text
CHUNKED_REFERENCE_ALGORITHM_EQUIVALENT=false
```

## 14. Level 2 production v2

```text
report: reports/aime24_int2_wave1_v100_8gpu/equivalence_chunked_level2/production_v2/teacher_forcing_summary.md
CHUNKED_PRODUCTION_NUMERIC_EQUIVALENT=true
first_mismatch_count=0
```

All recorded checkpoints have top-1 agreement and zero logits max absolute error in the saved metrics.

## 15. Greedy v2

```text
report: reports/aime24_int2_wave1_v100_8gpu/equivalence_chunked_level3/greedy_v2/greedy_summary.md
CHUNKED_GREEDY_TRAJECTORY_EQUIVALENT=true
p12 first divergence: null
p14 first divergence: null
```

Both fixed samples produced exact 1024-token matches.

## 16. Rolling regression

Rolling regression smoke/long-smoke was not rerun in this round.

```text
ROLLING_VARIANT_SMOKE_PASS=false
ROLLING_VARIANT_LONG_SMOKE_PASS=false
```

## 17. Remaining caveats

- Full model-level reference backend remains incomplete.
- Rolling mode still needs current-HEAD smoke and long-smoke after the cache mode split.
- Revised Wave 1A full was not started.

## 18. Full-run decision

```text
FULL_RUN_APPROVED=false
```
