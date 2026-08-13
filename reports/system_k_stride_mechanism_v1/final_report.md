# Final Report

Classification: `K_STRIDE_REGRESSION_PARTIALLY_SUPPORTED`

Dominant mechanism: `ADDRESS_ARITHMETIC_DOMINATED`

Confidence: `MEDIUM`

## Decision

Keep the asymmetric runtime architecture:

- K: tight QK-optimized layout
- V: capacity-managed stride-aware layout
- Architecture name: `ASYMMETRIC_KV_RUNTIME`

Do not re-enable K capacity in this branch. The S5A-3 performance gate stopped that path.

## Evidence Summary

- MEASURED: 32K tight K `247.808 us`, strided K `331.264 us`, overhead `33.68%`.
- MEASURED: physical capacity scan hypothesis is rejected by 8K pitch sensitivity.
- STATIC_CODE_EVIDENCE: strided K uses generic stride address equations for packed K, scale, zero, and assignment.
- STATIC_CODE_EVIDENCE: tight K keeps a fixed-token channel tile contiguous; strided capacity K spaces adjacent channels by `cap_packs`.
- STATIC_CODE_EVIDENCE: SASS integer address arithmetic grows substantially in strided K.
- STATIC_CODE_EVIDENCE: register pressure and occupancy-loss hypotheses are rejected by cuobjdump resource usage.

## Caveat

NCU/NSYS were unavailable in PATH, so memory-sector and stall-reason counters were not measured. Coalescing remains a plausible but unmeasured contributor. The strongest directly supported mechanism is address arithmetic plus tight-layout coupling.

NEXT_TASK: `V_ONLY_CAPACITY_FINAL_SYSTEM_BENCHMARK`
