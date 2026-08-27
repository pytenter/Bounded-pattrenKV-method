# V100 Compatibility Audit

V100 compute capability is sm70. BF16 is not used. FlashAttention is absent in the audited environment, so the compatible FP16 attention backend is eager/native attention. Formal CAUSAL timing is stopped because the Qwen compressed-domain backend is not ready, not because of FlashAttention fallback.
