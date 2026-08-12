# Current Reader Audit

- Production contiguous Value reader remains `attn_v_forward_cuda_outer_dim_with_base`.
- It expects already contiguous compact V payload, scale, zero, mask, and assignment tensors.
- S3-2 adds an experimental V2 page pointer-table reader only; QK, GQA, selectors, quantization, masks, assignments, centroids, sink/recent, residuals, and group size are unchanged.
