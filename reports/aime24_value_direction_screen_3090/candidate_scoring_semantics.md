# Candidate Scoring Semantics

For each Value token x KV-head vector and each existing centroid candidate, the scorer applies the production threshold/mask rule, simulates the same INT2 affine quantize/dequantize along head_dim, restores the centroid when masked, and computes the objective against the final reconstructed Value `v_hat(c)`.

- BASE: minmax residual range `amax(v-c)-amin(v-c)`.
- V-DIR: `1-cos(v, v_hat(c))`, with NRE fallback for near-zero vectors.
- V-HYBRID: `NRE(v, v_hat(c)) + DIR(v, v_hat(c))`, `lambda_dir=1.0`.

The candidate bank, candidate order, dynamic centroid creation, K path, packing format, scale/min, bits, Sink16, Recent128, and cache segmentation are unchanged.
