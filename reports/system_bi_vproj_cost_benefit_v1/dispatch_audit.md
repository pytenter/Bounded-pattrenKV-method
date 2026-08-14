# Dispatch Audit

Prefill projection dispatch is controlled by `PATTERNKV_PREFILL_PROJ_MODE`; legacy `PATTERNKV_BATCH_INVARIANT_KPROJ=1` maps to P1.

Observed ctx512 counters:

- P0 B2/B4: normal prefill K/V calls > 0, BI prefill K/V calls = 0.
- P1 B2/B4: BI prefill K calls > 0, normal prefill V calls > 0, BI prefill V calls = 0.
- P2 B2/B4: BI prefill K/V calls > 0, normal prefill K/V calls = 0.
- P2 decode-one: normal decode K/V calls > 0, BI decode K/V calls = 0.
- Serial request dispatches and fallback calls: 0.
