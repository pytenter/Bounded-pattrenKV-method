# Trace Methodology

The runner uses the same Request A/B and token construction protocol as S6-B.2.5. It compares independent B1 A/B against true B2 [A,B] at explicit PREFILL and DECODE1 snapshots. Layer0 K pipeline tensors are recomputed diagnostically under `PATTERNKV_K_ASSIGNMENT_TRACE=1`; production behavior is unchanged when trace is disabled.
