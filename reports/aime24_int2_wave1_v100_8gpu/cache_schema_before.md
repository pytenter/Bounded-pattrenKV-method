# Cache Schema Before Wave 1A Sink/Recent Rewrite

Repository state audited: `ce4ef0749680aebe016aeaa06e6e22ff9d711167` on branch `exp/aime-int2-wave1-v100-8gpu`.

## KIVI (`models/llama_kivi.py`)

### Active attention classes

`LlamaAttention_KIVI` and `LlamaFlashAttention_KIVI` both use the same cache tuple schema. AIME runner sets `config.use_flash = True`, so `LlamaFlashAttention_KIVI.forward` is the active experiment path. The non-flash class has the same tuple indexing and split logic.

### Current `past_key_value` tuple

Returned as:

```python
(
    key_states_quant_trans,
    key_states_full,
    key_scale_trans,
    key_mn_trans,
    value_states_quant,
    value_states_full,
    value_scale,
    value_mn,
    kv_seq_len,
)
```

| index | name | meaning | shape | dtype | token dimension |
| ---: | --- | --- | --- | --- | --- |
| 0 | `key_states_quant_trans` | packed quantized key history, after transposing K to `[B,H_kv,D,T]` before pack | `[B,H_kv,D,T_quant/(32/k_bits)]` | `torch.int32` | packed last dim; unpacked tokens are last dim times `32/k_bits` |
| 1 | `key_states_full` | unquantized key residual window | `[B,H_kv,T_full,D]` or `None` | model dtype, normally FP16 | dim 2 |
| 2 | `key_scale_trans` | affine scales for quantized key | `[B,H_kv,D,T_quant/group_size]` | model dtype | last dim groups over key token axis |
| 3 | `key_mn_trans` | affine minimum/zero for quantized key | `[B,H_kv,D,T_quant/group_size]` | model dtype | last dim groups over key token axis |
| 4 | `value_states_quant` | packed quantized value history | `[B,H_kv,T_quant,D/(32/v_bits)]` | `torch.int32` | dim 2 is token count |
| 5 | `value_states_full` | unquantized value residual window | `[B,H_kv,T_full,D]` | model dtype, normally FP16 | dim 2 |
| 6 | `value_scale` | affine scales for quantized value | `[B,H_kv,T_quant,D/group_size]` | model dtype | dim 2 token count |
| 7 | `value_mn` | affine minimum/zero for quantized value | `[B,H_kv,T_quant,D/group_size]` | model dtype | dim 2 token count |
| 8 | `kv_seq_len` | total cached tokens visible to attention | Python int | int | scalar |

### Prefill initialization

Forward computes Q/K/V, applies RoPE, then attends over full FP16 prefill K/V. Cache construction happens after attention.

Key prefill split:

- If `T < residual_length`: all K remains in `key_states_full`, no key pack.
- If `T % residual_length != 0`: prefix `K[:, :, :-(T % residual_length), :]` is packed; suffix `K[:, :, -(T % residual_length):, :]` is full.
- If `T % residual_length == 0`: all K is packed and `key_states_full = None`.
- Packing calls `triton_quantize_and_pack_along_last_dim(key_states_quant.transpose(2, 3), group_size, k_bits)`, so packed token dimension is the last dim of transposed K.

Value prefill split:

- If `T <= residual_length`: all V remains in `value_states_full`, no value pack.
- Else prefix `V[:, :, :-residual_length, :]` is packed; suffix `V[:, :, -residual_length:, :]` is full.
- Packing calls `triton_quantize_and_pack_along_last_dim(value_states_quant, group_size, v_bits)`, so value token dimension remains dim 2 and head_dim is packed along dim 3.

This means Key and Value do not use the same residual rule at prefill. Key preserves `T % residual_length` full tokens, while Value always preserves exactly `residual_length` full tokens when `T > residual_length`.

### Decode append and quantization

On each decode call, `kv_seq_len` is increased by `past_key_value[-1]` before attention.

Key decode:

1. Existing packed key, if any, is consumed by `cuda_bmm_fA_qB_outer(group_size, query_states, key_states_quant_trans, key_scale_trans, key_mn_trans, k_bits)`.
2. New `key_states` is appended to `key_states_full`.
3. Attention scores are concatenated as `[quantized_key_scores, full_key_scores]`.
4. If `key_states_full.shape[-2] == residual_length`, the entire full key window is quantized and appended to packed K. `key_states_full` is set to `None`.

Value decode:

1. New `value_states` is appended to `value_states_full`.
2. Attention values are consumed as quantized prefix plus full suffix. Quantized V is multiplied by the prefix attention weights; full V is multiplied by suffix weights.
3. If `value_states_full.shape[-2] > residual_length`, the oldest one full value token `value_states_full[:, :, :1, :]` is quantized and appended to packed V; the remaining full window is kept.

### Attention order

Current KIVI order is:

```text
quantized_history -> full_residual
```

for both Key and Value. There is no permanent FP16 sink segment and no separate pending segment. `sink_length` and `recent_length` are accepted by the bench config but are not used by model attention/cache logic; `residual_length` remains the actual residual window parameter.

## PatternKV (`models/llama_patternkv.py`)

### Active attention classes

`LlamaAttention_PatternKV` has an older non-flash path with the same 9-element KIVI-like tuple. AIME runner sets `config.use_flash = True`, so `LlamaFlashAttention_PatternKV.forward` is the active experiment path.

### Current flash `past_key_value` tuple

Returned as:

```python
(
    key_states_quant_trans,
    key_states_full,
    key_scale_trans,
    key_mn_trans,
    value_states_quant,
    value_states_full,
    value_scale,
    value_mn,
    kv_seq_len,
    assignments,
    v_assignments,
    v_assignments_idx,
)
```

| index | name | meaning | shape | dtype | token dimension |
| ---: | --- | --- | --- | --- | --- |
| 0 | `key_states_quant_trans` | packed quantized key residuals after subtracting selected key centroid | `[B,H_kv,D,T_quant/(32/k_bits)]` | `torch.int32` | packed last dim |
| 1 | `key_states_full` | full key residual window not yet pattern-quantized | `[B,H_kv,T_full,D]` or `None` | model dtype | dim 2 |
| 2 | `key_scale_trans` | affine scales for packed key residuals | `[B,H_kv,D,T_quant/group_size]` | model dtype | last dim groups over token axis |
| 3 | `key_mn_trans` | affine minimum/zero for packed key residuals | `[B,H_kv,D,T_quant/group_size]` | model dtype | last dim groups over token axis |
| 4 | `value_states_quant` | packed quantized value residuals after optional centroid subtraction | `[B,H_kv,T_quant,D/(32/v_bits)]` | `torch.int32` | dim 2 |
| 5 | `value_states_full` | full value residual window not yet pattern-quantized | `[B,H_kv,T_full,D]` or `None` | model dtype | dim 2 |
| 6 | `value_scale` | affine scales for packed value residuals | `[B,H_kv,T_quant,D/group_size]` | model dtype | dim 2 token count |
| 7 | `value_mn` | affine minimum/zero for packed value residuals | `[B,H_kv,T_quant,D/group_size]` | model dtype | dim 2 token count |
| 8 | `kv_seq_len` | total cached tokens visible to attention | Python int | int | scalar |
| 9 | `assignments` | key centroid id for packed key tokens | `[B,H_kv,T_key_quant]` | `torch.long` | dim 2 |
| 10 | `v_assignments` | value residualization mask for packed value tokens | `[B,H_kv,T_value_quant]` | `torch.uint8` | dim 2 |
| 11 | `v_assignments_idx` | value centroid id for packed value tokens | `[B,H_kv,T_value_quant]` | `torch.long` | dim 2 |

Centroids are not stored in the tuple. They live on the attention module as `self.k_base` (`[H_kv,M,D]`) and `self.v_centroids` (`[H_kv,M,D]`).

### Prefill initialization

Forward computes Q/K/V, applies RoPE, and runs full FP16 flash attention for the prefill.

Pattern setup before splitting:

- Key centroids are initialized from the mean key sequence via `batched_kmeans_fast_compiled`, then every prefill key token is assigned to `self.k_base` with `batched_assign_compiled`.
- Value centroids are initialized similarly and every prefill value token gets `v_assignments_idx_all`.

Key prefill split:

- Same residual block split as KIVI Key.
- If a quantized prefix exists, `assignments` is sliced to the same quantized prefix, selected centroids are gathered, key residuals are computed as `key_states_quant - k_base_per_pos`, and residuals are packed with `triton_quantize_and_pack_along_last_dim(key_states_quant.transpose(2, 3), group_size, k_bits)`.
- If `T < residual_length`, `assignments = None` and all key tokens remain full.

Value prefill split:

- If `T <= residual_length`: no packed V, all V remains full, and assignment/mask tensors are `None`.
- Else it computes `qlen = -(T % residual_length)`. If `T % residual_length == 0`, all V is quantized and full V is `None`; otherwise the prefix up to `qlen` is quantized and the suffix is full.
- The value prefix gets centroid ids and a residualization mask; the quantized payload is `value_states_quant - mask * centroid`.

This means PatternKV also has no sink segment; prefill assignments can cover the whole packed prefix, including tokens that should become permanent sink under Wave 1A.

### Decode append and quantization

Key decode:

1. Packed key history is consumed by `cuda_bmm_fA_qB_outer_with_base`, which adds centroid contributions using `self.k_base` and `assignments`.
2. New key is appended to `key_states_full`.
3. Scores are concatenated as `[pattern_quantized_key_scores, full_key_scores]`.
4. When `key_states_full.shape[-2] == residual_length`, a new Chebyshev centroid is appended to `self.k_base`, the whole full key window is assigned, residualized, packed, and appended. The corresponding assignments are appended.

Value decode:

1. New value is appended to `value_states_full`.
2. Packed V is consumed by `cuda_attn_v_fused_with_base` using `value_states_quant`, affine metadata, `self.v_centroids`, `v_assignments`, and `v_assignments_idx`; full suffix is handled by the same fused call via `attn_f`/`v_full`.
3. When `value_full_length == residual_length`, a new value centroid is appended, the whole full value window is assigned/masked/residualized/packed, and `value_states_full` is set to `None`.

### Attention order

Current PatternKV order is:

```text
pattern_quantized_history -> full_residual
```

for both Key and Value. There is no permanent FP16 sink, no rolling recent window independent of sink, and no pending history segment. `sink_length` and `recent_length` are accepted by config but the model logic uses only `residual_length`.

## Quant kernels (`quant/`)

### Packing

`triton_quantize_and_pack_along_last_dim(data, group_size, bit)` requires:

- 4D input.
- Last dimension divisible by `group_size`.
- Last dimension divisible by `32 / bit` for packing.
- `bit` values supported by Python pack/unpack helpers are `[2, 4, 8]`; Triton path accepts the passed bit and the CUDA GEMV wrapper asserts `bits in [2, 4]`.

K packing transposes K to `[B,H_kv,D,T]`, so K group/pack constraints apply to token length. V packing keeps `[B,H_kv,T,D]`, so V group/pack constraints apply to `head_dim`.

### Attention kernels

`cuda_bmm_fA_qB_outer` accepts `bits in [2,4]`, so K2/K4 QK and V2/V4 AV are supported by the wrapper. Pattern fused kernels also calculate `pack_factor = 32 / bit` in CUDA code and include explicit 2-bit and 4-bit code paths.

## Bench and runner

`bench/bench_aime24_patternkv.py` passes `sink_length` and `recent_length` into model config, but also sets `args.residual_length = args.recent_length`; current model code then uses residual length as the only cache window. `cache_storage_summary` currently reports `quantized_tokens` and `fp16_residual_tokens` via old residual split and does not distinguish sink, packed history, pending history, or recent.

`scripts/run_aime24_int2_wave1_8gpu.sh` currently defines 8 configs, including two mixed-key configs backed by placeholder masks. Smoke/full launch all 8 configs. Summary comparisons also include the mixed-key configs.
