# Strided K Reader Design

- Added experimental API `cuda_bmm_fA_qB_outer_with_base_strided_k`.
- Added C++ binding `gemv_forward_cuda_outer_dim_with_base_strided_k`.
- The production default path is unchanged.
- Only historical compressed K addressing changed; K quantization, centroid restoration, beta/scale/zero math, and QK accumulation are copied from the production kernel.
- Inputs are narrow logical views over capacity storage. Shape controls logical length; `Tensor.stride()` controls physical pitch.
- Sink/recent/pending FP16 regions are out of scope and unchanged.
- No page lookup, page table, CUDA VMM, vLLM, SGLang, GQA redesign, selector tuning, or centroid tuning.
- STRIDED_K_KERNEL_ITERATES_ONLY_LOGICAL_TOKENS=YES.
