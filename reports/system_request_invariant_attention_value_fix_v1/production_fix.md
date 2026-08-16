# Production Fix

Changed `quant/csrc/gemv_cuda.cu/.h` and `quant/page_batch.py` so fused page mixed-V reduction receives request-local `seq_lens` and only reduces valid packed tokens for each row. Added `request_invariant_full_value_attention` in `models/segmented_cache.py` and routed sink/pending/recent full precision tails through it from `models/llama_patternkv.py`.
