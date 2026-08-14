# Final Recommendation

Classification: `BI_VPROJ_LOW_COST_OPTIONAL`

Recommendation: keep production default at P1 (`bi_k`) and expose P2 (`bi_kv`) only as an explicit deterministic/strict prefill projection mode. The measured ctx512 cost is effectively flat versus P1, but the full-model first-token logit drift reduction is small rather than decisive.

Key ctx512 measurements:

- B2 prefill median: P1 2243.34 ms, P2 2248.62 ms, overhead 0.24%.
- B4 prefill median: P1 4141.40 ms, P2 4140.66 ms, overhead -0.02%.
- B2 logit drift ratio P1/P2: 1.11x.
- B4 logit drift ratio P1/P2: 1.10x.
- Argmax match rate: P1 1.0, P2 1.0.

Next task: `DESIGN_PREFILL_PROJECTION_MODE_POLICY`.

Notes: ctx2048/ctx4096, layerwise hidden propagation, and structural P2 cache exactness remain unsampled in this run and are left as `null` in `final_gate.json`.
