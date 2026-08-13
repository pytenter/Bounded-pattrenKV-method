# Memory Behavior Comparison

MEASURED CUDA Event reference from S5A-3:

| Context | Tight K | Strided K | Overhead |
| --- | ---: | ---: | ---: |
| 8K | 74.752 us | 99.328 us | 32.88% |
| 16K | 124.928 us | 182.272 us | 45.90% |
| 24K | 186.368 us | 271.360 us | 45.60% |
| 32K | 247.808 us | 331.264 us | 33.68% |

Capacity pitch sensitivity at logical 8K:

| Logical T | Capacity | Strided K latency |
| ---: | ---: | ---: |
| 8192 | 8192 | 95.232 us |
| 8192 | 16384 | 95.232 us |
| 8192 | 32768 | 95.232 us |

PHYSICAL_CAPACITY_SCAN_HYPOTHESIS=REJECTED.

Reason: keeping logical T fixed at 8192 while changing capacity from 8192 to 32768 keeps latency around 95-97 us. The kernel is not looping over physical capacity.

Memory coalescing hypothesis: INCONCLUSIVE but plausible.

STATIC_CODE_EVIDENCE: tight K stores the 128-channel tile contiguously for a fixed packed token; strided capacity storage spaces adjacent channel loads by `cap_packs`. Without NCU memory-sector counters, this remains a hypothesis rather than a measured conclusion.

SASS global load instruction count: tight K `126`, strided K `142`. This is a static instruction count, not DRAM bytes.
