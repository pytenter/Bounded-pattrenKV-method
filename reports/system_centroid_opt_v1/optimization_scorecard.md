# S2B-2A Optimization Scorecard

| Candidate | Normal V2@32K | Skewed V2@32K | Mixed-V@32K | E2E TPOT@32K | Correctness | Decision |
|---|---:|---:|---:|---:|---|---|
| Candidate A warp aggregation | 1442.8160190582275 us, regression | 2697.216033935547 us, regression | not run after kernel regression | not run | PASS | REGRESSION |
| Candidate B per-warp histogram | 309.2480003833771 us, 1.440x | 551.9359707832336 us, 2.197x | 827.3919820785522 us, 1.226x | 121.64556884765625 ms, 1.040x | PASS | WIN |
| Candidate C hybrid | not measured | not measured | not measured | not measured | not measured | NOT_IMPLEMENTED |

## Decision

`CENTROID_HISTOGRAM_OPTIMIZATION_SUPPORTED`

Candidate B is retained in production. Candidate A reduced logical atomics but regressed due to warp matching/shuffle overhead. Candidate C was not implemented because Candidate B already passed the full gate.
