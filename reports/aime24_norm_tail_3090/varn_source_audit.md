# VarN Source Audit

## Source

- `VARN_SOURCE_FOUND=True`
- source repo: `/data/zypan/kvarn-repro/repos/KVarN`
- source remote URL: `https://github.com/huawei-csl/KVarN.git`
- source branch: `origin/main`
- local branch: `repro/kvarn-h-varn-ablation`
- source commit: `7586257f1c632e63187bfacbbe21ccb51540f7b3`
- dirty worktree: `True`

## Files

- `vllm/model_executor/layers/quantization/kvarn/sinkhorn.py`
- `vllm/model_executor/layers/quantization/kvarn/config.py`
- `vllm/v1/attention/backends/kvarn_attn.py`
- `vllm/v1/attention/ops/triton_kvarn_sinkhorn.py`
- `vllm/v1/attention/ops/triton_kvarn_decode.py`

## Semantics

- Formula: `balanced = tile / s_col / s_row`.
- Algorithm: iterative log-domain alternating column/row standard-deviation normalization with best-so-far imbalance selection.
- Applies to both K and V cache tiles.
- Application point: after fp16 K/V are emitted by attention and before low-bit quantized tile storage.
- K orientation: `[D, group]`; V orientation: `[group, D]`.
- Scale axes are per-token and per-channel depending on K/V tile orientation.
- No offline calibration requirement was found in the audited source.
- Decode restores by reading stored metadata scales together with asymmetric RTN scale/zero-point.

## Hadamard / VarN-Only Gate

- `HADAMARD_REQUIRED_BY_CANONICAL_PIPELINE=True`
- `CANONICAL_VARN_ONLY_SUPPORT=False`
- `LOCAL_UNCOMMITTED_VARN_ONLY_SUPPORT=True`

Dirty local KVarN files:

- `M vllm/config/cache.py`
- `M vllm/model_executor/layers/quantization/kvarn/config.py`
- `M vllm/model_executor/layers/quantization/kvarn/debug.py`
- `M vllm/utils/torch_utils.py`
- `M vllm/v1/attention/backends/kvarn_attn.py`
- `M vllm/v1/attention/ops/triton_kvarn_decode.py`

## Gate

- `VARN_SOURCE_COMMIT_PINNED=True`
- `VARN_SEMANTICS_AUDITED=True`
- `VARN_CAN_RUN_WITHOUT_UNRELATED_KVARN_COMPONENTS=False`
- `VARN_SOURCE_VALID=False`
- `DO_NOT_IMPLEMENT_VARN=True`

Conclusion: the pinned canonical source contains VarN as part of the KVarN pipeline, but the audited canonical config does not expose a clean VarN-only intervention independent of Hadamard and the rest of KVarN. Local dirty files appear to add VarN-only switches, so they are useful evidence for a future source-freeze step but are not treated as canonical here.
