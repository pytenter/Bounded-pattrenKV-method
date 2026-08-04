
# KIVI GQA Fix Design

1. Original location: `models/llama_kivi.py`, original line 380, `LlamaFlashAttention_KIVI.forward`, decode AV residual branch.
2. Cause: attention probabilities had 32 query heads, but residual V cache was persisted as 8 KV heads and was not temporarily repeated before `torch.matmul`.
3. Mapping: `Hq=32`, `Hkv=8`, `num_key_value_groups=4`; query heads `0..3 -> kv0`, `4..7 -> kv1`, ..., `28..31 -> kv7`.
4. Fix: added `repeat_kv_for_gqa` helper and routed residual K/V attention matmul through it with explicit head-count assertions.
5. Persistent cache remains `[B,8,L,128]` for full residual tensors and `[B,8,...]` for packed quantized tensors.
6. The fix uses temporary repeat only for ordinary PyTorch residual attention matmul; quantized packed cache is not expanded.
7. Quantized cache uses the existing CUDA wrapper that accepts `nh` and `nh_kv`; residual cache uses temporary repeat before QK/AV matmul.
8. Both K and V paths are covered; the immediate crash was V, but base residual K/V paths were made consistent.
9. MHA `n_rep=1` is covered by regression test and remains identity.
10. Performance: correctness PASS. Memory efficiency is PARTIAL because residual K/V matmul temporarily materializes 32-head tensors for the residual window; persistent cache storage does not inflate 4x. Later optimization can move this to native grouped matmul or kernel-level `kv_head_id = query_head_id // groups`.
