# V100 Compressed Backend Compatibility

- GPU SKU: Tesla V100-SXM2-32GB.
- Compute capability: sm70.
- Driver/CUDA as reported by `nvidia-smi`: driver 525.60.13, CUDA 12.0.
- Runtime env used for development: `/home/qinch2023/miniconda3/envs/patternkv-v100/bin/python`.
- PyTorch: 2.4.1+cu118.
- CUDA runtime: 11.8.
- Triton: 3.0.0.
- FlashAttention: absent.

## Component Classification

| Component | Classification | Note |
| --- | --- | --- |
| Qwen3 native model import | PYTORCH_FALLBACK_AVAILABLE | Requires Transformers 4.51 runtime; target worktree uses server-side vendor path if local vendor is absent. |
| segmented cache | V100_READY | Pure PyTorch plus existing CUDA helpers. |
| segmented softmax | PYTORCH_FALLBACK_AVAILABLE | Existing helper has PyTorch fallback if fixed split CUDA unsupported. |
| INT2 K QK CUDA helper | V100_RECOMPILE_REQUIRED_OR_PYTORCH_FALLBACK | Correctness may use compressed PyTorch fallback if CUDA extension is unavailable on sm70. |
| mixed V2/V4 fused page reader | V100_RECOMPILE_REQUIRED_OR_PYTORCH_FALLBACK | Correctness may use page-local compressed fallback; full historical V reconstruction remains forbidden. |
| FP16 sink/pending/recent tail | V100_READY | Native PyTorch FP16 matmul. |
| FlashAttention2 | V100_UNSUPPORTED_IN_ENV | Not installed and not required for correctness closure. |

A PyTorch fallback is acceptable only when it remains compressed-domain and does not concatenate full historical K/V or V.
