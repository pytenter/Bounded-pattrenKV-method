# VarN Source Audit

- source repo: `/data/zypan/kvarn-repro/repos/KVarN`
- source remote: `https://github.com/huawei-csl/KVarN.git`
- source commit: `7586257f1c632e63187bfacbbe21ccb51540f7b3`
- source commit pinned: `True`
- dirty worktree: `True`

## Canonical Semantics

- Pinned KVarN source is read via Git object content at the canonical commit, not via the dirty local worktree.
- This diagnostic ports only canonical VarN/Sinkhorn scaling semantics into PatternKV; Hadamard and KVarN kernels stay disabled.
- K uses post-RoPE residual tiles with canonical [D, group] axes.
- V uses post-projection adjusted residual tiles with canonical [group, D] axes.
- Decode restores logical K/V by applying inverse VarN scales after low-bit dequantization and before Pattern base reconstruction.

## Source Files

- `vllm/model_executor/layers/quantization/kvarn/sinkhorn.py`
- `vllm/model_executor/layers/quantization/kvarn/config.py`
- `vllm/v1/attention/backends/kvarn_attn.py`
- `vllm/v1/attention/ops/triton_kvarn_sinkhorn.py`
- `vllm/v1/attention/ops/triton_kvarn_decode.py`

## Local Dirty KVarN Files

- `M vllm/config/cache.py`
- `M vllm/model_executor/layers/quantization/kvarn/config.py`
- `M vllm/model_executor/layers/quantization/kvarn/debug.py`
- `M vllm/utils/torch_utils.py`
- `M vllm/v1/attention/backends/kvarn_attn.py`
- `M vllm/v1/attention/ops/triton_kvarn_decode.py`

`VARN_SOURCE_VALID=True`
`VARN_EQUIVALENCE_VALID` is established by canonical reference equivalence and round-trip tests/preflight.
