# V100 Page Kernel Blocker

The packed-history GPU4 probe reached the compressed mixed-V path, but the page-pool operator failed because `patternkv_gemv` does not export `attn_v_forward_cuda_page_mixed_pool` in this environment.

Classification: PAGE_MIXED_POOL_SYMBOL_MISSING

Mitigation for correctness closure: select the existing legacy compressed mixed-V CUDA reader by default (`QWEN3_COMPRESSED_V_BACKEND=legacy_cuda`). This keeps historical V in compressed V2/V4 streams and does not call `reconstruct_full_v`.
