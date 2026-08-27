# Request-local CUDA K/V Interface Audit

- Branch: `sys/qwen3-8b-v100-system-generalization-v1`
- Start HEAD: `5dd03dcf2dec1e91aed52498299fa1d6a73329e0`
- CUDA build: rebuilt `quant/patternkv_gemv.cpython-310-x86_64-linux-gnu.so` in the current worktree with CUDA 11.8 `nvcc` from `/home/qinch2023/miniconda3/envs/patternkv-v100/bin/nvcc`, `TORCH_CUDA_ARCH_LIST=7.0`.
- Import audit: Python imports `patternkv_gemv` from the current worktree `quant/`, not the sibling AIME worktree.
- K/QK CUDA reader: `gemv_forward_cuda_outer_dim_with_base` accepts centroids as `[nh_kv,M,IC]` or request-local `[B,nh_kv,M,IC]`; kernel indexes with `B_centroids` and request id `b`.
- V CUDA reader: `attn_v_forward_cuda_outer_dim_with_base` and debug wrapper accept `[nh_kv,M,OC]` or `[B,nh_kv,M,OC]`; kernel indexes with `Bcent` and request id `b`.
- Python wrappers: `cuda_attn_v_fused_with_base` accepts 3D/4D V centroids; mixed-V compaction gathers V2/V4 payload and metadata per request instead of using a B==1-only mask.
- Cache validation: request-local `request_packed_v_tokens` and `request_packed_v4_tokens` are initialized for true batch and checked per request, preventing global-sum false failures.
- Prohibited paths: no Python request loop or serial request dispatch added; counters for B2/B4 show `serial_request_forward_dispatches=0` and `serial_attention_dispatches=0`.
