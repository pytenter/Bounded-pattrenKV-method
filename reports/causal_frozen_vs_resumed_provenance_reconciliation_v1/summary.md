# Summary

The sharp CAUSAL discrepancy is reconciled without changing the CAUSAL algorithm. Direct same-GPU A/B shows 50a remains in the frozen fast regime when run under the frozen environment, so a real post-freeze CAUSAL runtime regression is not supported. The capacity regression is reproduced as a memory lifecycle/protocol drift: the current paper wrapper omits `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, and C4096/B8 fails only under that omission.

Primary classification: `CAUSAL_FROZEN_VS_RESUMED_RECONCILED_NO_REGRESSION`. Capacity classification: `CAUSAL_CAPACITY_MEMORY_LIFECYCLE_DRIFT_CONFIRMED`.
