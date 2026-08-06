# PatternKV Dynamic Centroid Audit

Repository state at audit start:

- Branch: `exp/aime-int2-wave1-v100-8gpu`
- HEAD: `76d4e3c393741b7d7599dd94cb749e5617d4cdfa`
- Starting command note: `/home/qinch2023` is not the Git root; the active repository is `/home/qinch2023/Bounded-pattrenKV-method`.
- Existing uncommitted files were observed and left untouched: `bench/aime_utils.py`, `reports/paper_repro_v2/aime24/smoke_test_report.md`, existing `artifacts/`, `core`, AIME25 dataset files, existing Wave 1A reports, insight runner manifests, model download scripts, and existing smoke JSON.

Sources audited:

- Legacy commit: `ce4ef0749680aebe016aeaa06e6e22ff9d711167`
- Legacy branch check: `exp/patternkv-insight-wave-a-4gpu`
- Current segmented HEAD: `76d4e3c393741b7d7599dd94cb749e5617d4cdfa`
- Key files: `models/llama_patternkv.py`, `models/segmented_cache.py`, `insight/hook_metrics.py`, `bench/bench_aime24_patternkv.py`, `bench/paper_config.py`

## Executive Summary

The legacy PatternKV algorithm is implemented in `models/llama_patternkv.py` in the non-segmented tuple cache path. It builds initial K and V centroid banks during prefill using per-KV-head k-means, then dynamically appends one Chebyshev center per KV head whenever the FP16 residual decode window reaches `residual_length`. After appending, it reassigns the whole just-full window against the expanded centroid bank, subtracts gathered centroids, quantizes residuals, appends assignments/masks, and clears the FP16 full window.

The current segmented path preserves the desired physical cache layout:

```text
[sink_fp16]
[packed_pattern_history]
[pending_fp16_history]
[recent_fp16]
```

However, it does not currently reproduce PatternKV paper semantics for packed history. `build_cache_from_prefill()` and `append_decode()` call `flush_pending()`, which packs raw K/V tensors with ordinary quantization. The PatternKV wrapper then fills missing K assignments, V assignment indices, and V masks with zeros. It does not initialize segmented centroid banks from prefill, does not subtract centroid residuals before packing, does not apply V threshold/gate, does not append dynamic centroids on pack windows, and does not reconstruct packed V by adding centroids only for gate-selected tokens.

## Legacy Algorithm Answers

### 1. Old prefill K centroid generation

In legacy prefill, after RoPE is applied to `key_states`, the code computes:

```python
key_states_means = key_states.mean(dim=0, keepdim=True)
Xmk = key_states_means.permute(1, 0, 2, 3).reshape(n_kv, seq_len, hd).to(torch.float32)
Xk = key_states.permute(1, 0, 2, 3).reshape(n_kv, bz * seq_len, hd).to(torch.float32)
assign_k, k_centroids = batched_kmeans_fast_compiled(Xmk, k=num_k_base, iters=30, tol=1e-4, seed=0)
assign_k = batched_assign_compiled(Xk, k_centroids)
self.k_base = k_centroids.to(key_states.dtype)
```

Important details:

- Initial K centroids are trained on the batch-mean K stream, not directly on all batch samples.
- The training input shape is `[H_kv, seq_len, head_dim]` because `B` has been averaged out.
- The final K assignment is computed for all tokens from `Xk` with shape `[H_kv, B * seq_len, head_dim]`.
- Initial centroid dtype is FP32 during k-means, then stored as the model activation dtype, usually FP16.

### 2. Old prefill V centroid generation

V follows the same k-means initialization pattern, using post-projection `value_states`:

```python
value_states_means = value_states.mean(dim=0, keepdim=True)
Xm = value_states_means.permute(1, 0, 2, 3).reshape(n_kv, seq_len, hd).to(torch.float32)
X = value_states.permute(1, 0, 2, 3).reshape(n_kv, bz * seq_len, hd).to(torch.float32)
assign, centroids = batched_kmeans_fast_compiled(Xm, k=num_v_base, iters=30, tol=1e-4, seed=0)
assign = batched_assign_compiled(X, centroids)
v_assignments_idx_all = assign.view(n_kv, bz, seq_len).permute(1, 0, 2).contiguous().to(torch.long)
self.v_centroids = centroids.to(value_states.dtype)
```

Then only the quantized prefill region gets V assignment indices and a V gate mask. The trailing FP16 residual region does not get persistent assignments.

### 3. K/V initial centroid counts

- K initial centroid count: `config.num_k_base`, default `32`.
- V initial centroid count: `config.num_v_base`, default `32`.
- For the AIME paper config these are forced to `32` in `bench/paper_config.py`.

### 4. Centroid ownership granularity

Centroids are maintained per attention layer and per KV head:

- Each `LlamaFlashAttention_PatternKV` module owns `self.k_base` and `self.v_centroids`.
- The model sets `layer.self_attn.layer_idx = layer_idx`.
- Shapes are `[num_key_value_heads, num_centroids, head_dim]`.
- They are not global, not per query head, and not created after GQA repeat.

### 5. Decode trigger for new centroid

Legacy non-segmented decode uses FP16 rolling buffers named `key_states_full` and `value_states_full`.

- K dynamic update triggers when `key_states_full.shape[-2] == residual_length`.
- V dynamic update triggers when `value_full_length == residual_length`.
- With one-token decode, this happens exactly when a full residual window has accumulated.
- After update/pack, the full window is set to `None`, so the next update occurs after another `residual_length` decoded tokens have accumulated.

### 6. Window used for new centroid

The new centroid is computed from the just-full FP16 window:

- K window: `key_states_full`, shape `[B, H_kv, residual_length, D]`.
- V window: `value_states_full`, shape `[B, H_kv, residual_length, D]`.
- The window is converted to `[H_kv, B * residual_length, D]`.

For segmented semantics, this maps to the legal window that has left `recent` and is about to move from `pending_fp16_history` into `packed_pattern_history`.

### 7. K dynamic centroid computation

K dynamic centroid is not k-means. It is one Chebyshev center per KV head:

```python
Xw = key_states_full.permute(1, 0, 2, 3).reshape(H, B * Lr, D).contiguous()
cur_centroid = ((Xw.amin(dim=1, keepdim=True) + Xw.amax(dim=1, keepdim=True)) * 0.5)
```

The resulting shape is `[H_kv, 1, D]`.

### 8. V dynamic centroid computation

V dynamic centroid is also one Chebyshev center per KV head, computed by `_append_v_centroid_from_window()`:

```python
Xw = Vw.permute(1, 0, 2, 3).reshape(H, B * Lr, D).contiguous()
cur = ((Xw.amin(dim=1, keepdim=True) + Xw.amax(dim=1, keepdim=True)) * 0.5)
self.v_centroids = torch.cat([self.v_centroids, cur.to(self.v_centroids.dtype)], dim=1)
```

### 9. How new centroid is appended

- K: `self.k_base = torch.cat([self.k_base, cur_centroid], dim=1)`.
- V: `self.v_centroids = torch.cat([self.v_centroids, cur], dim=1)`.
- The appended dimension is the centroid-bank dimension, not token, head, or feature dimension.

### 10. Current window reassignment

After appending the dynamic centroid, legacy code reassigns every token in the current full window against the expanded bank:

- K uses `_assign_minmax_hnk(Xw, self.k_base)`, returning `[H_kv, B * Lr]`.
- V uses `_nearest_v_centroid(value_states_full, self.v_centroids)`, returning `[B, H_kv, Lr]`.
- Assignment is done after the bank has been expanded, so tokens may choose old or newly appended centroids.

### 11. Assignment distance

Both K and V dynamic assignment use min-max residual range, not L2:

```python
distance(x, c) = max(x - c) - min(x - c)
```

The initial prefill assignment uses `batched_assign_compiled()`. It should be checked in code before refactoring, but the dynamic path audited here explicitly uses min-max range.

### 12. Residual timing

Residual is computed after assignment:

1. Append/generate centroid bank.
2. Assign each token to nearest centroid.
3. Gather per-token centroid.
4. Compute residual.
5. Quantize and pack residual.

K residual:

```python
key_states_full = key_states_full - k_base_per_pos
```

V residual/gated raw path:

```python
value_states_full_adj = value_states_full - v_mask_w.unsqueeze(-1) * cent_w
```

### 13. Centroid and residual dtype

- Initial centroids are generated in FP32 k-means and stored as `key_states.dtype` / `value_states.dtype`, usually FP16.
- Dynamic K centroid is cast to `self.k_base.dtype`.
- Dynamic V centroid is cast to `self.v_centroids.dtype`.
- Residual tensors are computed in the current activation dtype, usually FP16, then quantized.

### 14. Assignment dtype

- K assignment tensors are `torch.long`.
- V assignment indices are `torch.long`.
- V gate mask is saved as `torch.uint8`.
- Dynamic min-max distance computation uses the dtype of the input tensors unless explicitly converted by caller.

### 15. Centroid bank growth

The legacy centroid bank grows without an explicit cap:

- K gains one centroid per KV head per full decode residual window.
- V gains one centroid per KV head per full decode residual window.
- No eviction or max-bank bound was found in the audited code.

### 16. K/V dynamic update symmetry

K and V are similar but not identical:

- Both append a Chebyshev center per full window.
- Both reassign the just-full window using min-max range.
- K always subtracts the assigned centroid and quantizes residual.
- V applies a threshold/gate. Gate-selected tokens subtract the centroid before quantization; gate-rejected tokens are quantized raw.

### 17. Meaning of `iters=0`, Chebyshev center, and threshold gate

- `iters=0` in `batched_kmeans_fast()` means random-sample initialization only, no Lloyd iterations. In the audited prefill path, the actual call uses `iters=30`, so the initial centroid bank is real k-means after random initialization.
- Chebyshev center here means per-feature midpoint `(max + min) / 2` over the current window for each KV head. It minimizes the coordinate-wise max range objective used by the min-max residual distance heuristic.
- The V threshold gate computes `rho = R(V - centroid) / R(V)`, where `R(x) = max(x) - min(x)` over head_dim. It accepts pattern residualization when:

```text
1 - rho^2 >= (2 * z_0.95 / sqrt(5 * head_dim)) * sqrt(1 + rho^4)
```

`True` means use `centroid + quantized_residual` on reconstruction; `False` means quantize raw V and do not add a centroid during attention.

### 18. Current segmented path missing steps

Current segmented PatternKV is not yet semantically equivalent to legacy PatternKV:

1. `build_cache_from_prefill()` constructs segmented history and calls `flush_pending()` without PatternKV centroid inputs.
2. `flush_pending()` quantizes raw K and raw V; it never computes or subtracts K centroid residuals.
3. `flush_pending()` never computes V nearest centroid, V gate, or gated V residual.
4. `models/llama_patternkv.py` fills missing segmented assignments with zeros after prefill/decode, which creates token counts but not meaningful assignments.
5. Segmented cache has `k_centroids` and `v_centroids` fields, but current prefill does not populate them from `self.k_base` / `self.v_centroids`.
6. Segmented decode `append_decode()` packs pending windows immediately using raw quantization; it never appends dynamic K/V Chebyshev centroids.
7. The packed-K attention path uses `cuda_bmm_fA_qB_outer()` without centroid addition, while legacy PatternKV uses `cuda_bmm_fA_qB_outer_with_base()`.
8. The packed-V attention path uses `cuda_bmm_fA_qB_outer()` without V assignment/gate centroid addition, while legacy PatternKV uses `cuda_attn_v_fused_with_base()`.
9. `reconstruct_full_k()` and `reconstruct_full_v()` dequantize raw packed tensors only; they do not add K centroids or gated V centroids.
10. `validate_cache()` checks assignment token count alignment, but does not validate centroid shapes, assignment ranges, or V gate length.
11. Serialization does not currently distinguish V assignment index from V gate by name; schema has `v_assignments` and `v_assignment_idx`, but no explicit `v_pattern_mask` alias/field.
12. Bitwidth accounting includes assignment and mask tensor bytes, but does not report theoretical compact assignment bits versus actual tensor dtype bits, nor dynamic centroid bank payload by K/V separately.
13. No explicit sample-level reset function was found. Legacy state lives on attention modules (`self.k_base`, `self.v_centroids`) and can leak across repeated AIME tasks unless reset or overwritten by a clean prefill path.

## Required Recovery Semantics

The segmented recovery should make pattern assignment and residual quantization operate only on packed pattern history:

```text
sink_fp16: no persistent assignment, no persistent gate
packed_pattern_history: centroid assignment, residual quantization, packed tensors
pending_fp16_history: no persistent assignment until flushed
recent_fp16: no persistent assignment, latest R non-sink tokens
```

Dynamic centroid update should be driven by `flush_pending()` or an equivalent PatternKV-specific flush function, because this is the moment when a legal window leaves FP16 history and enters packed history.

For each legal pack window:

1. Take K/V window from pending prefix.
2. Append one Chebyshev centroid per KV head for K and V.
3. Reassign the same window against the updated bank.
4. Gather centroids.
5. Compute K residual unconditionally.
6. Compute V gate using the threshold formula; subtract V centroid only where gate is true.
7. Quantize/pack residual or raw adjusted tensor.
8. Append K assignment, V assignment index, and V gate mask with exactly the packed token count.
9. Delete the packed prefix from pending.
10. Increment per-layer or per-cache update counters.

## Current Blocker Status

`FULL_RUN_APPROVED=false`

The segmented PatternKV path cannot be called `PatternKV_paper_segmented` yet. It still lacks the dynamic centroid, assignment, V gate, fused reconstruction, reset, validation, tests, and equivalence evidence required before Wave 1A full.
