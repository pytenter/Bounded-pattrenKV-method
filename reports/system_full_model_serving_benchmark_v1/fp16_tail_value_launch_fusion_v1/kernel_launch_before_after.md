# Kernel Launch Before After

## Formal Full-Model Profiler

Formal GPU1 profiler attribution is blocked by external compute contamination.

## Non-Formal Direct Tail Probe

Environment: `CUDA_VISIBLE_DEVICES=2`, direct synthetic FP16 tail Value sequence, C2048-like tail lengths `sink=16`, `pending=112`, `recent=128`, `layers=32`, `decode=8`.

Results:

- Old direct FP16 tail CUDA kernels/token: `1472.0`
- Fused direct FP16 tail CUDA kernels/token: `32.0`
- Old direct FP16 tail GPU ms/token: `3.3549025000002954`
- Fused direct FP16 tail GPU ms/token: `0.4791142499999999`
- Old direct elapsed ms/token under profiler: `42.06208419799805`
- Fused direct elapsed ms/token under profiler: `0.6927000284194946`
- Old tiny-kernel fraction: `1.0`
- Fused tiny-kernel fraction: `1.0`

This is structural kernel-count evidence for the tail operator only, not formal serving TPOT.

## Prior Formal Baseline Evidence

Prior forensic baseline:

- Causal total kernels/token: `12599`
- Causal attention kernels/token: `4768`
- FP16 tail Value kernel calls/token: `2560`
- FP16 tail Value GPU compute ms/token: `4.818`
- FP16 tail Value launch/gap ms/token: `37.342`
