
# KIVI GQA Call Chain Audit

| 阶段 | Q shape | K shape | V shape | 预期 head 数 | 实际 head 数 | 是否正确 |
|---|---|---|---|---:|---:|---|
| prefill input | `[B,32,q_len,128]` | `[B,8,q_len,128]` | `[B,8,q_len,128]` | Q=32, KV=8 | Q=32, KV=8 | 正确 |
| persistent cache | n/a | `[B,8,seq,128]` or packed `[B,8,128,N/16]` | `[B,8,seq,128]` or packed `[B,8,N,8]` | 8 | 8 | 正确 |
| dequantized temporary K/V | kernel internal | kernel internal | kernel internal | Q=32 mapped to KV=8 | kernel accepts `nh, nh_kv` | 正确 |
| residual K/V | n/a | `[B,8,Lr,128]` | `[B,8,Lr,128]` | 8 persistent | 8 | 正确 |
| QK matmul 前 | `[B,32,1,128]` | temporary `[B,32,L,128]` for residual K | n/a | 32 | 32 | 正确 after helper |
| AV matmul 前 | attention `[B,32,1,L]` | n/a | temporary `[B,32,L,128]` for residual V | 32 | 32 | 正确 after helper |
| attention output | n/a | n/a | n/a | 32 | `[B,32,1,128]` | 正确 |

## Findings

- `LlamaFlashAttention_KIVI` prefill uses flash-attn with native GQA tensors and then stores persistent K/V as 8 KV heads.
- Decode QK residual path already applied `repeat_kv`, while the AV residual-only path did not.
- The quantized K and V kernels call `cuda_bmm_fA_qB_outer`, whose wrapper passes both `nh` and `nh_kv` and asserts `nh % nh_kv == 0`; it is designed for GQA mapping.
- The minimal defect was therefore in the residual FP16 V branch, with the same helper applied to base attention for consistency.
