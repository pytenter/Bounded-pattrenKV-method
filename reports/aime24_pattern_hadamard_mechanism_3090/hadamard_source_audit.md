# Hadamard Source Audit

- source repo: `/data/zypan/kvarn-repro/repos/KVarN`
- source remote: `https://github.com/huawei-csl/KVarN.git`
- source commit: `7586257f1c632e63187bfacbbe21ccb51540f7b3`
- source commit pinned: `True`
- dirty worktree: `True`

## Canonical Semantics

- Pinned KVarN describes an orthonormal Sylvester-Hadamard rotation along the channel/head dimension.
- The small diagnostic ports only this rotation, not VarN, Sinkhorn, metadata fusion, or KVarN kernels.
- The transform is deterministic, uses no random sign vector, no permutation, and includes `1/sqrt(d)` scaling.
- K uses post-RoPE states; V uses post-projection value states. Both are rotated before quantized cache storage.
- During decode, packed K is consumed in the rotated frame by rotating Q; packed V contribution is un-rotated after the value matmul.

## Source Files

- `vllm/model_executor/layers/quantization/kvarn/config.py`
- `vllm/model_executor/layers/quantization/kvarn/mla_probe.py`
- `vllm/model_executor/layers/quantization/kvarn/mla_quant.py`

## Local Dirty KVarN Files

- `M vllm/config/cache.py`
- `M vllm/model_executor/layers/quantization/kvarn/config.py`
- `M vllm/model_executor/layers/quantization/kvarn/debug.py`
- `M vllm/utils/torch_utils.py`
- `M vllm/v1/attention/backends/kvarn_attn.py`
- `M vllm/v1/attention/ops/triton_kvarn_decode.py`

`HADAMARD_SOURCE_VALID=True`
`HADAMARD_EQUIVALENCE_VALID` is established by orthogonality and round-trip tests/preflight, not by enabling a separate FP16 model path.
