# RMSNorm Implementation

HF `LlamaRMSNorm` computes FP32 variance over hidden dim and multiplies by fp16 weight. Production fix uses fixed hidden-dim chunk reduction. Hidden size: `4096`. Eps: `1e-05`.
