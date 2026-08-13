# B=1 Assumption Audit

Scope: exact blockers and adjacent assumptions for the frozen mixed V2/V4 runtime. Classification: `TRIVIAL` means wrapper-only plumbing, `MODERATE` means local ABI changes without new storage model, `STRUCTURAL` means the current representation cannot encode true continuous batching safely.

## Findings

| Class | Site | Evidence | Blocker | Severity |
|---|---|---|---|---|
| A. Python wrapper | `models/segmented_cache.py:1149` | `_cat_mixed_packed_v` raises when `v_adjusted.shape[0] != 1` | prevents B>1 mixed cache packing | STRUCTURAL |
| A. Python wrapper | `models/segmented_cache.py:1151` | `mask = precision_mask[0].bool()` | selects compact V2/V4 payload by request 0 only | STRUCTURAL |
| A. Python wrapper | `models/segmented_cache.py:1205` | `reconstruct_packed_v` raises for `v_precision_mask.shape[0] != 1` | reference/materialization path is B=1-only for mixed V | MODERATE |
| A. Python wrapper | `models/segmented_cache.py:1215` | scatter back with `~mask[0]` and `mask[0]` | row-0 logical reconstruction | STRUCTURAL |
| D. CUDA binding | `quant/matmul.py:1190` | `_cuda_attn_v_mixed_fused_with_base_impl` raises for `B != 1` | mixed fused Value attention entry is hard-blocked | STRUCTURAL |
| D. CUDA binding | `quant/matmul.py:1202` | `mask = precision_mask[0].bool()` | mapping prep assumes one precision vector | STRUCTURAL |
| D. CUDA binding | `quant/matmul.py:1223` / `1278` | payload length compared to global `v2_tokens`/`v4_tokens` | cannot represent per-request compact lengths | STRUCTURAL |
| B. Cache representation | `models/segmented_cache.py:793` | capacity install derives `v2_pattern_mask/v4_pattern_mask` from `mask[0]` | synthetic adoption also row-0 only | STRUCTURAL |
| B. Cache representation | `models/segmented_cache.py:1833` | validates `selected == packed_v4_tokens` as a scalar | payload count is global, not per request | STRUCTURAL |
| C. Selector state | `models/segmented_cache.py:1092` | selector returns `[B,T]` and reads `v_causal_importance[:, range]` | selector itself is batch-shaped | TRIVIAL |
| C. Selector top-k | `models/segmented_cache.py:1077` | Python loop over B for deterministic tie handling | acceptable for offline selector, not serving-hot production | MODERATE |
| E. CUDA kernel | `quant/csrc/gemv_cuda.cu:975` | kernels map `batch_idx -> b,hq,kv` | many base kernels are batch-aware for dense equal-length streams | TRIVIAL for dense, STRUCTURAL for ragged mixed |
| F. Memory layout | `quant/matmul.py:1226` / `1281` | attention weights are compacted with boolean masks before kernels | creates temporary dense V2/V4 attention tensors | MODERATE for B=1, STRUCTURAL for B>1 ragged |
| G. Variable stream length | current `packed_v` / `packed_v4` | physical V2/V4 token counts are single scalars (`Tv2`, `Tv4`) | incompatible with heterogeneous request masks | STRUCTURAL |

## Blocker Categories

- Python wrapper: current code explicitly rejects B>1 before pack and fused attention.
- Cache representation: current compact V2/V4 streams have one physical length per cache object; they do not carry per-request offsets/counts/ranks.
- Selector state: causal importance and selector scores are not the primary blocker; they already carry B.
- CUDA binding: wrapper ABI has no metadata for ragged V2/V4 streams, only dense equal-length payloads.
- CUDA kernel: existing V2/V4 kernels can process dense `[B,H,T,D]`-like compact streams, but cannot discover request-local logical-to-physical mapping.
- Memory layout: compact streams are compute-friendly for B=1 but not request-aware.
- Variable stream length: the fundamental serving blocker.

## Final Judgment

`CONCURRENCY_RUNTIME_BLOCKED` is structural. It is not enough to loop over `precision_mask[b]` or remove the guards: each request has different V2/V4 rank maps, and the production operator needs an ABI that exposes those maps to the kernel without Python per-request dispatch.
