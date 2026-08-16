# Atomic Audit

`rg` found CUDA `atomicAdd` uses in `quant/csrc/gemv_cuda.cu` at lines 1596, 1632, 1637, 1879, 2065, and 2305. The page mixed pool value kernel entry is `page_mixed_pool_value_kernel` at line 2698 and `attn_v_forward_cuda_page_mixed_pool` at line 2797; no `atomicAdd` occurrence was found inside that page mixed pool region in this audit. Python/Triton `tl.atomic_add` was not found in the inspected value reduction path.
