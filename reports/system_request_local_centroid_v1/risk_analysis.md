# Risk Analysis

Primary residual risk is that full active centroid views are materialized as `[B,H,M,D]` for B>1, which is correctness-oriented for the MVP. It does not copy inactive slots and does not alter K payload, V page ABI, quantization, selector, or fused Value arithmetic.
