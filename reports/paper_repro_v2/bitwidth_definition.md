# Bitwidth Definition

## KIVI Paper G128

- Method name: `kivi_paper_g128`
- Payload: INT2 for K and V.
- Group size: 128.
- Affine metadata: FP16 scale plus FP16 min per group.
- Quantized-region theoretical bits per scalar: `2 + 32 / 128 = 2.25`.
- Residual tokens stay FP16 and are averaged by each sample's actual residual length, not by a hard-coded full 128-token assumption.

## PatternKV Paper

- Method name: `patternkv_paper`
- Payload: INT2 for K and V residual/centroid quantized tensors.
- Residual affine quant group: 128.
- Initial pattern count: 32.
- Dynamic pattern period `G_pattern`: 128 decode tokens.
- Pattern selection position: post-RoPE K/V states.

## Stored Fields

Each LongBench output row records `paper_config_snapshot` and `cache_bitwidth_stats`, including:

- `total_cached_tokens`
- `quantized_tokens`
- `fp16_residual_tokens`
- `packed_payload_bytes`
- `scale_min_bytes`
- `fp16_residual_bytes`
- `assignment_dtype` and `assignment_bytes`
- `mask_dtype` and `mask_bytes`
- `centroid_bytes`
- `initial_pattern_count`
- `dynamic_pattern_count_k`
- `dynamic_pattern_count_v`
- `quantized_region_theoretical_bits_per_scalar`
- `python_tensor_storage_bytes`
