# Current Runtime Tensor ABI

Scope: audit of the frozen `ASYMMETRIC_KV_RUNTIME` as implemented in `models/segmented_cache.py`, `models/llama_patternkv.py`, `quant/matmul.py`, and `quant/csrc/*`. No implementation changes were made.

## Runtime Shape Conventions

- Logical cache order is `sink | packed history | pending | recent` for rolling mode and `packed history | pending` for chunked mode.
- K packed history is tight and compute-oriented: quantized K residuals are stored in a single logical stream.
- V packed history is split for mixed precision: base V2 and selected V4 are independent affine quantization streams, plus logical precision metadata.
- Token dimensions:
  - FP16 K/V regions: dim 2 in `[B, Hkv, T, D]`.
  - packed K payload: logical token axis is dim 3 after transpose/packing.
  - packed V payload and V metadata: logical/compact token axis is dim 2, except `v_precision_mask` uses dim 1.

## Tensor ABI Table

| Object | Shape | Dtype | Logical meaning | Token dimension | Current B semantics | Request-specific? | Variable length? | Contiguous? | Capacity-backed? | Append strategy | CUDA reader assumption |
|---|---:|---|---|---:|---|---|---|---|---|---|---|
| `packed_k` | `[B,Hkv,D,ceil(Tk/pack2)]` for K INT2 | `int32` | packed K residuals after subtracting K centroid | dim 3 physical packed tokens | batch dimension exists and QK wrapper flattens `B*H`; no hard B=1 guard in production tight path | yes | no within one cache object; `Tk == packed_k_tokens` shared across B | contiguous for tight path | not V capacity; K has tight and strided experimental paths | `_cat_packed_k` / capacity helpers depending path | `cuda_bmm_fA_qB_outer_with_base` reshapes to `[B*Hkv,N/pack,D]`; assumes dense same `Tk` for all B |
| `packed_v` | `[B,Hkv,Tv2,ceil(D/pack2)]` | `int32` | V2 adjusted Value stream, independently affine-quantized | dim 2 compact V2 tokens | B is present but mixed packer requires `B==1`; `Tv2` is derived from `precision_mask[0]` | yes | yes, varies by request under real batching | contiguous or capacity buffer logical view | yes for `fixed_capacity` / `chunked_capacity` | `_cat_mixed_packed_v` splits `v_adjusted[:, :, ~mask, :]` | fused V reader expects dense compact stream length equal to global V2 count |
| `packed_v4` | `[B,Hkv,Tv4,ceil(D/pack4)]` | `int32` | V4 adjusted Value stream, independently affine-quantized | dim 2 compact V4 tokens | B is present but current code only permits B=1 | yes | yes, varies by request | contiguous or capacity buffer logical view | yes | `_cat_mixed_packed_v` splits `v_adjusted[:, :, mask, :]` | mixed wrapper launches V2/V4 kernels separately after compact attention gather |
| `packed_v_scale` | `[B,Hkv,Tv2,D/group]` | usually `float16` | per-token, per-head, per-group affine scale for V2 stream | dim 2 | B=1-only in mixed cache | yes | yes | same as `packed_v` | yes | appended with `packed_v` | reader assumes same compact token order as `packed_v` |
| `packed_v_zero` | `[B,Hkv,Tv2,D/group]` | usually `float16` | per-token affine zero/min for V2 stream | dim 2 | B=1-only in mixed cache | yes | yes | same as `packed_v` | yes | appended with `packed_v` | reader assumes same compact token order as `packed_v` |
| `packed_v4_scale` | `[B,Hkv,Tv4,D/group]` | usually `float16` | per-token, per-head, per-group affine scale for V4 stream | dim 2 | B=1-only | yes | yes | same as `packed_v4` | yes | appended with `packed_v4` | reader assumes same compact token order as `packed_v4` |
| `packed_v4_zero` | `[B,Hkv,Tv4,D/group]` | usually `float16` | per-token affine zero/min for V4 stream | dim 2 | B=1-only | yes | yes | same as `packed_v4` | yes | appended with `packed_v4` | reader assumes same compact token order as `packed_v4` |
| `v_precision_mask` | `[B,Tpacked]` | `uint8` / bool | logical token precision: `1 => V4`, `0 => V2` | dim 1 | stored as `[B,T]`, but all split/gather code uses row 0 | yes | yes | contiguous or capacity buffer | yes | `_cat_v_precision_mask` | mixed wrapper requires exact `[B,total_tokens]` then immediately uses `precision_mask[0]` |
| `v_pattern_mask` | `[B,Hkv,Tpacked]` | `uint8` | logical token Pattern gate for restoring selected centroid contribution | dim 2 | supports B-shaped storage | yes | no for current dense logical history | contiguous or capacity buffer | yes | `_cat_assignment` or `_cat_v_metadata` | non-capacity mixed path gathers with `low_mask/high_mask` from row 0 |
| `v_assignment_idx` | `[B,Hkv,Tpacked]` | `int32` mostly | logical token V centroid id | dim 2 | supports B-shaped storage | yes | no for current dense logical history | contiguous or capacity buffer | yes | `_cat_assignment` or `_cat_v_metadata` | same as `v_pattern_mask` |
| `v2_pattern_mask` | `[B,Hkv,Tv2]` | `uint8` | compact V2 Pattern gates | dim 2 | produced from row-0 precision split; not valid for heterogeneous B | yes | yes | contiguous or capacity buffer | yes | `_cat_v_metadata` after V2 split | capacity mixed reader consumes it directly |
| `v2_assignment_idx` | `[B,Hkv,Tv2]` | `int32` | compact V2 centroid ids | dim 2 | produced from row-0 precision split | yes | yes | contiguous or capacity buffer | yes | `_cat_v_metadata` | capacity mixed reader consumes it directly |
| `v4_pattern_mask` | `[B,Hkv,Tv4]` | `uint8` | compact V4 Pattern gates | dim 2 | produced from row-0 precision split | yes | yes | contiguous or capacity buffer | yes | `_cat_v_metadata` | capacity mixed reader consumes it directly |
| `v4_assignment_idx` | `[B,Hkv,Tv4]` | `int32` | compact V4 centroid ids | dim 2 | produced from row-0 precision split | yes | yes | contiguous or capacity buffer | yes | `_cat_v_metadata` | capacity mixed reader consumes it directly |
| `centroids` (`k_centroids`, `v_centroids`) | `[Hkv,M,D]` | model dtype / `float16` in kernels | centroid banks shared across requests in a layer cache object | none | no batch dimension; derived from batch windows during prefill/dynamic update | shared layer/cache state, not per request in current wrapper | grows by centroid count | contiguous after cat | no | `_append_dynamic_centroids` cats one centroid per pack window | CUDA expects `[Hkv,M,D]`; assignment carries request identity |
| `v_causal_importance` | `[B,total_tokens]` | `float32` | accumulated historical attention mass for causal selector | dim 1 | batch-shaped and request-specific | yes | grows with total_tokens | contiguous | no | resized/copy in `update_value_causal_importance` | selector reads `[B, absolute_start:absolute_start+tokens]`; storage B is ok before mixed pack |

## Key Evidence

- `PatternQuantizedKVCache` declares the mixed V2/V4 tensors and selector state in `models/segmented_cache.py:45`.
- `_cat_mixed_packed_v` rejects `v_adjusted.shape[0] != 1` and splits with `precision_mask[0]` in `models/segmented_cache.py:1139`.
- `reconstruct_packed_v` repeats the B=1 guard and row-0 scattering in `models/segmented_cache.py:1201`.
- `select_value_precision_mask` is batch-shaped before storage: it returns `[B,T]` and uses per-request scores in `models/segmented_cache.py:1092`.
- `patternkv_mixed_value_attention` forwards compact V2/V4 streams to `cuda_attn_v_mixed_fused_with_base` in `models/llama_patternkv.py:112`.
- `_cuda_attn_v_mixed_fused_with_base_impl` rejects `B != 1`, then derives `low_mask/high_mask` from `precision_mask[0]` in `quant/matmul.py:1165`.

## Audit Conclusion

The current ABI is internally consistent for one request, but it is not a serving-native batch ABI. The structural issue is the coupling between a logical `[B,T]` precision mask and two compact physical streams whose lengths are derived from only one request. Once requests select different V4 positions, compact stream lengths and logical-to-physical ranks diverge per request. A B>1 design must therefore add page/request-aware offset metadata or a page table ABI; deleting B=1 guards would corrupt request B's V2/V4 mapping.
