# Qwen3 Compressed Backend Design

Status: implementation mapping before coding.

## Native Qwen3 Geometry

- Model class: `Qwen3ForCausalLM`.
- Attention class: `Qwen3Attention`.
- Model path: `/home/qinch2023/modelscope_models/Qwen3-8B`.
- Config identity: `model_type=qwen3`, `architectures=[Qwen3ForCausalLM]`.
- Layers: 36.
- Q heads: 32.
- KV heads: 8.
- GQA groups: 4 Q heads per KV head.
- Hidden size: 4096.
- Head dim: 128.
- Q/K norm: Qwen3 applies `q_norm` after Q projection and `k_norm` after K projection.
- RoPE: applied to normalized Q/K before cache append/update.
- Scaling: Qwen3 native attention uses the module scaling value; for this model it is equivalent to `1 / sqrt(head_dim)`.
- Mask semantics: additive causal attention mask shaped `[B, 1, Q, T]` after cache growth.
- Cache position / position embeddings: Qwen3 model computes position embeddings and passes `(cos, sin)` into attention; compressed backend must consume these native values rather than recomputing Llama positions.
- Output projection: segmented attention output is reshaped back to `[B, Q, hidden]` then passed through Qwen3 `o_proj`.
- Actual dtype: formal timing is intended to use FP16 for matched FP16 and CAUSAL surrounding compute. Config declares `bfloat16`, but no formal timing is allowed until gates pass.

## Stage Mapping

| Qwen3 native stage | Compressed PatternKV stage |
| --- | --- |
| Q projection | native `q_proj`, then `q_norm` |
| K projection | native `k_proj`, then `k_norm` |
| V projection | native `v_proj` |
| RoPE | native Qwen3 `apply_rotary_pos_emb` |
| Cache prefill | `build_cache_from_prefill` with frozen CAUSAL config |
| Cache decode append | `append_decode`, no full reconstruction |
| Historical QK | INT2 packed K via frozen `cuda_bmm_fA_qB_outer[_with_base]` or PyTorch compressed fallback |
| Sink/pending/recent QK | FP16 `patternkv_request_invariant_qk_scores` |
| Softmax | `request_invariant_segmented_attention_softmax` |
| Historical V | mixed V2/V4 compressed reader; page-local fallback allowed, full historical reconstruction forbidden |
| Sink/pending/recent V | FP16 request-invariant value attention |
| Importance update | `update_value_causal_importance` using segmented attention probabilities |
| Output projection | native Qwen3 `o_proj` |

## Backend Split

- `qwen3_patternkv_reference`: existing semantic oracle. It reconstructs full historical K/V and is not valid for system timing.
- `qwen3_patternkv_compressed`: new target backend. It must not call `reconstruct_full_k` or `reconstruct_full_v` in decode.

## Scope Boundary

This task closes backend correctness/readiness only. It does not run formal batch/context/long-decode performance matrices and does not make speedup claims.
