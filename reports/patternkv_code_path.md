# PatternKV Code Path Report

Target hardware for this reproduction stage: NVIDIA RTX 3090 / SM86.

## 1. KV Cache Handoff

- File path: `models/llama_patternkv.py`
- Class/function: `LlamaForCausalLM_PatternKV.forward`, `LlamaModel_PatternKV.forward`, `LlamaDecoderLayer_PatternKV.forward`, `LlamaFlashAttention_PatternKV.forward`
- Role: Hugging Face generation calls `prepare_inputs_for_generation`, then model/layer/attention forward pass. PatternKV owns the legacy `past_key_values` tuple returned by each attention layer.
- Input tensor shape: `input_ids [B, T]`; attention hidden states `[B, q_len, hidden_size]`; projected Q `[B, num_heads, q_len, head_dim]`; K/V `[B, num_kv_heads, q_len, head_dim]`.
- Output tensor shape: `past_key_value` per layer is a 12-field tuple:
  `key_states_quant_trans`, `key_states_full`, `key_scale_trans`, `key_mn_trans`, `value_states_quant`, `value_states_full`, `value_scale`, `value_mn`, `kv_seq_len`, `assignments`, `v_assignments`, `v_assignments_idx`.
- Prefill: yes, when `past_key_value is None`.
- Decode: yes, when `past_key_value is not None`.
- Packed KV path: yes. Historical K/V cache is stored as packed `torch.int32` tensors plus scale/min metadata; recent tokens remain FP16 in the residual window.

## 2. Key Quantization Entry

- File path: `models/llama_patternkv.py`
- Class/function: `LlamaFlashAttention_PatternKV.forward`
- Role: K residuals are quantized and packed after subtracting the assigned K pattern/centroid.
- Input tensor shape: K before residualization `[B, num_kv_heads, T_quant, head_dim]`; after transpose for pack `[B, num_kv_heads, head_dim, T_quant]`.
- Output tensor shape: `key_states_quant_trans [B, num_kv_heads, head_dim, T_quant / (32 / k_bits)]`; `key_scale_trans/key_mn_trans [B, num_kv_heads, head_dim, T_quant / group_size]`.
- Prefill: yes. Prefill path gathers `k_base_per_pos` using `assignments`, subtracts it, then calls `triton_quantize_and_pack_along_last_dim`.
- Decode: yes. When `key_states_full.shape[-2] == residual_length`, the full residual window is assigned, residualized, and packed.
- Packed KV path: yes, packed K is `torch.int32`.

## 3. Value Quantization Entry

- File path: `models/llama_patternkv.py`
- Class/function: `LlamaFlashAttention_PatternKV.forward`
- Role: V values are assigned to V patterns, masked by threshold, residualized conditionally, and packed.
- Input tensor shape: V before residualization `[B, num_kv_heads, T_quant, head_dim]`.
- Output tensor shape: `value_states_quant [B, num_kv_heads, T_quant, head_dim / (32 / v_bits)]`; `value_scale/value_mn [B, num_kv_heads, T_quant, head_dim / group_size]`.
- Prefill: yes. Prefill path uses `v_assignments_idx_all`, gathers `v_cent_per_pos_q`, computes `v_mask_q`, subtracts masked centroids, then packs.
- Decode: yes. When `value_full_length == residual_length`, the full V residual window is assigned/masked/residualized/packed.
- Packed KV path: yes, packed V is `torch.int32`.

## 4. Packed INT32 Cache Storage Format

- File path: `quant/new_pack.py`
- Function: `triton_quantize_and_pack_along_last_dim`, `_minmax_along_last_dim`, `_pack_along_last_dim`
- Role: Computes group min/max, converts FP16 values to integer codes, and packs multiple low-bit values into one int32.
- Input tensor shape: generic `[B, H, D, T]` for K-transposed path or `[B, H, T, D]` for V path, where the last dimension is divisible by `group_size`.
- Output tensor shape: packed code `[B, H, D, T/(32/bits)]` or `[B, H, T, D/(32/bits)]`; scale/min `[B, H, D, T/group_size]` or `[B, H, T, D/group_size]`.
- Prefill: yes.
- Decode: yes.
- Packed KV path: yes. `code.dtype == torch.int32`; `scale/min.dtype == input dtype`, normally `torch.float16`.

## 5. Pattern Initialization Logic

- File path: `models/llama_patternkv.py`
- Functions: `batched_kmeans_fast`, `batched_assign`, call sites in `LlamaFlashAttention_PatternKV.forward`
- Role: Initializes K and V pattern centroids using PyTorch batched KMeans helpers. The imported `cuml.KMeans` is optional and not called in the active path.
- Input tensor shape: K/V samples reshaped to `[num_kv_heads, B*T, head_dim]`; mean samples for KMeans initialization use `[num_kv_heads, T, head_dim]` when `B=1`.
- Output tensor shape: `k_base [num_kv_heads, num_k_base, head_dim]`; `v_centroids [num_kv_heads, num_v_base, head_dim]`; assignments `[B, num_kv_heads, T]`.
- Prefill: yes.
- Decode: no for initial KMeans; decode appends Chebyshev centers.
- Packed KV path: yes, assignments are used to residualize K/V before packing.

## 6. Pattern Update Trigger

- File path: `models/llama_patternkv.py`
- Functions: `_chebyshev_center_per_head`, `_append_v_centroid_from_window`, `LlamaFlashAttention_PatternKV.forward`
- Role: Appends one new per-head Chebyshev center when the FP16 residual window reaches `residual_length`.
- Input tensor shape: residual window `[B, num_kv_heads, residual_length, head_dim]`, reshaped to `[num_kv_heads, B*residual_length, head_dim]`.
- Output tensor shape: new centroid `[num_kv_heads, 1, head_dim]`; appended K/V pattern tensors `[num_kv_heads, old_patterns+1, head_dim]`.
- Prefill: no for updates.
- Decode: yes.
- Packed KV path: yes. After update, the full residual window is reassigned/residualized/packed and removed from the FP16 window.

## 7. V Mask Generation and Use

- File path: `models/llama_patternkv.py`
- Functions: `_threshold_and_mask_given_base`, `_v_threshold_and_mask`, `_gather_centroids`, `_nearest_v_centroid`
- Role: Computes range contraction ratio `R(v - centroid) / R(v)`, generates a boolean mask, stores it as `torch.uint8`, and uses it during fused attention to add back pattern contribution only for masked positions.
- Input tensor shape: `v_states [B, num_kv_heads, L, head_dim]`; gathered centroids `[B, num_kv_heads, L, head_dim]`.
- Output tensor shape: `rho [B, num_kv_heads, L, 1]`; `mask [B, num_kv_heads, L]`; cached mask `v_assignments` as `torch.uint8`.
- Prefill: yes.
- Decode: yes.
- Packed KV path: yes. V residual is packed as int32 and mask/index metadata are passed into `cuda_attn_v_fused_with_base`.

## 8. Prefill and Decode PatternKV Entry

- File path: `models/llama_patternkv.py`
- Function: `LlamaFlashAttention_PatternKV.forward`
- Role: Prefill computes flash attention output using full Q/K/V, then initializes PatternKV cache. Decode consumes packed historical cache and FP16 residual cache.
- Input tensor shape: prefill `q_len = prompt length`; decode `q_len = 1`.
- Output tensor shape: attention output `[B, q_len, hidden_size]`; updated `past_key_value` tuple.
- Prefill: yes.
- Decode: yes.
- Packed KV path: prefill creates packed cache for future decode; decode uses custom CUDA fused kernels for packed historical K/V.

## 9. Recent High-Precision Token Retention

- File path: `models/llama_patternkv.py`
- Function: `LlamaFlashAttention_PatternKV.forward`
- Role: Keeps the most recent residual window in FP16 as `key_states_full` and `value_states_full`.
- Input tensor shape: appended decode token K/V `[B, num_kv_heads, 1, head_dim]`.
- Output tensor shape: residual cache `[B, num_kv_heads, <=residual_length, head_dim]`.
- Prefill: yes, prompt tail shorter than residual block remains FP16.
- Decode: yes, grows by one token until update/pack trigger.
- Packed KV path: recent window is intentionally not packed until reaching `residual_length`.

## 10. Quantization Granularity and Metadata

- File path: `quant/new_pack.py`
- Function: `triton_quantize_and_pack_along_last_dim`
- Role: Performs min/max quantization by group along the last dimension.
- Input tensor shape: `[B, H, D, T]` or `[B, H, T, D]`.
- Output tensor shape: packed int32 code plus `scale` and `mn`.
- Prefill: yes.
- Decode: yes.
- Packed KV path: yes.
- Granularity:
  - K: quantizes K after transpose, so groups are contiguous token positions per `(batch, kv_head, head_dim)`.
  - V: quantizes V along `head_dim`, so groups are contiguous hidden dimensions per `(batch, kv_head, token)`.
  - `group_size=128` in official smoke configuration.

## 11. CUDA Fused Packed Attention

- File path: `quant/matmul.py`, `quant/csrc/pybind.cpp`, `quant/csrc/gemv_cuda.cu`
- Functions/classes: `cuda_bmm_fA_qB_outer_with_base`, `cuda_attn_v_fused_with_base`, `gemv_forward_cuda_outer_dim_with_base`, `attn_v_forward_cuda_outer_dim_with_base`
- Role: Custom CUDA computes QK over packed K residuals plus K pattern compensation, and AV over packed V residuals plus V centroid/mask compensation.
- Input tensor shape:
  - QK: `fA [B, num_heads, 1, head_dim]`, packed K `[B, num_kv_heads, head_dim, N/(32/k_bits)]`, scale/min `[B, num_kv_heads, head_dim, N/group_size]`, centroids `[num_kv_heads, patterns, head_dim]`, assignments `[B, num_kv_heads, N]`.
  - AV: attention weights `[B, num_heads, 1, N]`, packed V `[B, num_kv_heads, N, head_dim/(32/v_bits)]`, scale/min `[B, num_kv_heads, N, head_dim/group_size]`, V centroids `[num_kv_heads, patterns, head_dim]`, mask/index `[B, num_kv_heads, N]`.
- Output tensor shape: QK logits `[B, num_heads, 1, N]`; attention output `[B, num_heads, 1, head_dim]`.
- Prefill: no for fused packed attention; prefill uses flash attention output before constructing cache.
- Decode: yes.
- Packed KV path: yes, this is the primary packed decode path.
