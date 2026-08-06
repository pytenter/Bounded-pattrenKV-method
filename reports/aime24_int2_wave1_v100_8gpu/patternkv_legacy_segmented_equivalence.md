# PatternKV Legacy/Segmented Equivalence Report

Date: 2026-08-06

## Scope

This report covers the restoration of PatternKV dynamic centroid semantics in the segmented cache path and the validation completed before any Wave 1A full run.

## Legacy Dynamic Centroid Algorithm

The legacy non-segmented PatternKV path:

- Builds initial K centroids per layer and KV head with k-means over batch-mean post-RoPE K states, default `num_k_base=32`.
- Builds initial V centroids per layer and KV head with k-means over batch-mean V states, default `num_v_base=32`.
- Stores K/V centroid banks as `[num_kv_heads, num_centroids, head_dim]` in activation dtype, normally FP16.
- On decode, appends one Chebyshev center per KV head when the FP16 residual window reaches `residual_length`.
- Uses Chebyshev center `(window_min + window_max) / 2` over `[H_kv, B * window, D]`.
- Reassigns the whole current window after appending the centroid bank.
- Uses min-max residual range `max(x - c) - min(x - c)` for dynamic K and V assignment.
- Computes K residual after assignment and always packs K residual.
- Computes V gate with `rho = R(V - centroid) / R(V)` and the statistical threshold from the legacy code.
- Packs raw V for gate false and V residual for gate true; fused V attention adds centroids only for gate true.

## Segmented Recovery

The segmented cache now keeps the required physical layout:

```text
[sink_fp16]
[packed_pattern_history]
[pending_fp16_history]
[recent_fp16]
```

Recovered behavior:

- Prefill passes legacy K/V centroid banks, K assignments, V assignment indices, and V gate masks into `build_cache_from_prefill()`.
- Pattern-specific pending flush subtracts K centroids before packing packed K history.
- Pattern-specific pending flush applies V assignment and V gate before packing packed V history.
- Decode dynamic centroid updates trigger only inside `flush_pending()` when pending has a legal pack window.
- The dynamic pack window is exactly the pending prefix that is about to enter packed history.
- K/V dynamic centroid banks append one Chebyshev center per KV head per packed window.
- K/V assignments and V gate masks are appended with packed tensors.
- `validate_cache()` checks K assignment, V assignment, and V gate token counts against packed history tokens.
- `reconstruct_full_k()` and `reconstruct_full_v()` restore packed pattern history with centroid/gate semantics.
- Segmented decode attention now uses centroid-aware K scores and fused V centroid/gate restoration for packed history.

## V Gate Logic

For each V token:

- `gate=True`: packed tensor stores quantized residual `V - centroid`; attention reconstructs `centroid + dequantized_residual`.
- `gate=False`: packed tensor stores raw-quantized V; attention does not add centroid.
- The cache stores this explicitly as `v_pattern_mask`, with `v_assignments` retained as a backward-compatible alias.

## Sample Reset

Added `reset_patternkv_runtime_state(model)` and call it before each PatternKV AIME task. It clears per-layer runtime centroid banks on attention modules without reloading model weights.

Added `collect_patternkv_dynamic_stats(model, past_key_values)` and records this per result:

```json
{
  "patternkv_dynamic_stats": {
    "initial_k_centroids_per_layer": [],
    "final_k_centroids_per_layer": [],
    "initial_v_centroids_per_layer": [],
    "final_v_centroids_per_layer": [],
    "k_centroid_updates_per_layer": [],
    "v_centroid_updates_per_layer": [],
    "k_assignment_tokens_per_layer": [],
    "v_assignment_tokens_per_layer": [],
    "v_pattern_selected_tokens_per_layer": [],
    "v_pattern_rejected_tokens_per_layer": [],
    "packed_k_tokens_per_layer": [],
    "packed_v_tokens_per_layer": []
  }
}
```

## Unit Tests

Command run:

```bash
/home/qinch2023/miniconda3/envs/patternkv-v100/bin/python -m pytest \
  tests/test_segmented_cache_semantics.py \
  tests/test_patternkv_dynamic_centroids.py \
  tests/test_patternkv_legacy_segmented_equivalence.py \
  tests/test_patternkv_runtime_reset.py \
  tests/test_patternkv_v_gate.py \
  tests/test_sink_recent_cache.py \
  tests/test_effective_bitwidth.py \
  -q
```

Result:

```text
30 passed
```

Covered:

- Dynamic K centroid shape, dtype, per-head independence, append, assignment range, residual path.
- V centroid gather, gate true reconstruction, gate false raw path, mask length.
- Dynamic cadence with `sink=2`, `recent=4`, `group_size=4`.
- Assignment alignment after every pack.
- Sample reset and deterministic cache serialization.
- Serialization of dynamic centroid, assignment, and V gate mask.
- Synthetic legacy Chebyshev/min-max equivalence.
- Bitwidth accounting for dynamic K/V centroids, K/V assignments, and V gate.

## Synthetic Tensor Equivalence

Completed Level 1 synthetic checks:

- Chebyshev center equals `(min + max) / 2` per KV head.
- Dynamic min-max assignment matches a direct legacy reference implementation.
- Assignment count equals packed token count.
- K/V reconstruct paths include centroid/gate semantics.
- S=0/R=4 synthetic segmented rolling behavior matches the legacy residual-window cadence.

## Teacher-Forcing Comparison

Not completed in this run.

Required Level 2 checkpoints at token positions `128, 256, 512, 1024, 2048, 4096` were not generated. Therefore logits cosine, top-1 agreement, NLL difference, and reconstructed K/V relative MSE against the legacy runtime are not yet available.

## Greedy Generation Comparison

Not completed in this run.

The required two-task `max_new_tokens=1024`, `do_sample=false` legacy-vs-segmented generated-token comparison was not run. Therefore exact token sequence equality and first divergence position are not yet available.

## Sampling Sanity

Completed practical sampling sanity on isolated output directories, without overwriting existing Wave 1A results.

Smoke command:

```bash
RESULT_DIR=results/aime24_int2_wave1_v100_8gpu_dynamic_centroid \
LOG_DIR=run/aime24_int2_wave1_v100_8gpu_dynamic_centroid \
REPORT_DIR=reports/aime24_int2_wave1_v100_8gpu/dynamic_centroid \
bash scripts/run_aime24_int2_wave1_8gpu.sh wave1a-smoke
```

Smoke result:

```text
records=12
errors=0
PatternKV assignment alignment failures=0
GPU residual processes=0
```

Long-smoke command:

```bash
RESULT_DIR=results/aime24_int2_wave1_v100_8gpu_dynamic_centroid_long \
LOG_DIR=run/aime24_int2_wave1_v100_8gpu_dynamic_centroid_long \
REPORT_DIR=reports/aime24_int2_wave1_v100_8gpu/dynamic_centroid_long \
bash scripts/run_aime24_int2_wave1_8gpu.sh wave1a-long-smoke
```

Long-smoke result:

```text
records=12
errors=0
cache validation errors=0
CUDA illegal memory access=0
PatternKV assignment alignment failures=0
GPU residual processes=0
```

PatternKV long-smoke packed/assignment checks:

```text
pattern_k2v2_s0_r128: packed=3968, K assignment=3968, V assignment=3968
pattern_k2v2_s64_r256: packed=2560/3840, K assignment matches, V assignment matches
pattern_k2v4_s0_r128: packed=3968, K assignment=3968, V assignment=3968
pattern_k4v2_s0_r128: packed=3968, K assignment=3968, V assignment=3968
```

## Remaining Differences

The implementation now restores the segmented dynamic centroid and V gate mechanics, but strict legacy equivalence is not fully proven yet:

- No teacher-forcing legacy-vs-segmented logits comparison has been run.
- No greedy legacy-vs-segmented generation comparison has been run.
- No full Level 2/3 report exists for first divergence analysis.
- A temporary legacy and segmented dual-run harness still needs to be added or documented for the model-level equivalence checks.

## Full Run Decision

Wave 1A full is not approved in this report. The dynamic path passes unit tests and smoke/long-smoke sanity, but the requested strict teacher-forcing and greedy legacy equivalence layers are incomplete.

Final method naming:

```text
PatternKV_paper_segmented_candidate
```

```text
FULL_RUN_APPROVED=false
```
