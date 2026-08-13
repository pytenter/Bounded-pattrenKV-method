# Integration Audit

- Growing historical Value streams: packed V2, V2 scale/zero, packed V4, V4 scale/zero, precision mask, V pattern mask, V assignment/index, compact V2/V4 pattern/index metadata.
- Historical K streams still use the existing growing contiguous path in this phase.
- QK reader consumes packed K, K scale/zero, and K assignments; it still assumes tight contiguous layout.
- V2 reader consumes compact packed V2, V2 scale/zero, compact V pattern mask, and compact V assignment/index.
- V4 reader consumes compact packed V4, V4 scale/zero, compact V pattern mask, and compact V assignment/index.
- Selector and packing consume pending FP16 V, centroids, causal importance, and emit frozen causal_v4 25% precision identities.
- Flush cadence remains 128 tokens.
- Sink/recent remain unchanged.
