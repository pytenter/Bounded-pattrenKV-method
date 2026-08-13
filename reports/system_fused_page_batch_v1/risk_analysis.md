# Risk Analysis

- The kernel is correctness-first and maps one block per output scalar, so it may need optimization for serial-reference parity.
- `v4_prefix_counts` is retained for unambiguous MVP rank lookup; a bitmap+popcount path can reduce metadata later.
- The fused operator currently covers fixed-length B in `{1,2,4}` plus partial final pages, not ragged serving integration.
