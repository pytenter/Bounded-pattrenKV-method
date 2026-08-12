# Amdahl Table Headroom

Post-histogram decomposition uses `FULL - NO_TABLE_CONTRIBUTION`, so the table component is an approximate ablation cost.

- V2 16K: FULL 225.280 us, NO_TABLE 150.528 us, table 74.752 us, fraction 33.18%.
- V2 32K: FULL 448.512 us, NO_TABLE 295.936 us, table 152.576 us, fraction 34.02%.
- V4 32K guard: FULL 164.864 us, NO_TABLE 104.448 us, table 60.416 us, fraction 36.65%.

If the V2 32K table component became zero-cost, the isolated V2 upper-bound speedup would be about `1.516x`. For mixed-V 32K, using V2+V4 table estimates against same-run baseline mixed-V gives about `19.87%` table headroom and a rough upper-bound speedup of `1.248x`.

Observed Candidate B: V2 32K `1.460x`, mixed-V 32K `1.053x`, E2E median TPOT 32K `1.001x`.
