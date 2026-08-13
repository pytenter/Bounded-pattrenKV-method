# Optimization Options

- Do not redesign PAGE_CENTRIC_DUAL_STREAM.
- Move page scheduling and metadata lookup away from Python `.item()`.
- Replace page-local Value reconstruction with compressed-domain V2/V4 page reads.
- Fuse per-page restore/index/matmul/accumulate into a small number of batched GPU launches.
