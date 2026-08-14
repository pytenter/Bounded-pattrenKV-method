# Risk Analysis

Implementation risk is low because P2 reuses the existing batch-invariant V2 projection kernel and only changes prefill K/V projection dispatch.

Remaining risks:

- Structural P2 cache exactness fields are still `null`; no claim is made for K/V centroid, assignment, mask, or packed cache exactness in this report.
- ctx2048 and ctx4096 timing/numerics were not sampled in this run.
- Layerwise propagation was not sampled because full hidden-state capture increased memory pressure on the 24GB RTX 3090.
- Fused page operator preservation counters were not re-audited here and remain `null`.

Production risk is controlled by keeping default behavior unchanged and requiring explicit `PATTERNKV_PREFILL_PROJ_MODE=bi_kv` for P2.
