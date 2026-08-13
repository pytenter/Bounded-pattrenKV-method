# Kernel Mapping

- Launches per decode: `1`.
- Blocks: `(B*nh, head_dim, 1)`.
- Threads: `256`.
- Token precision is resolved through `metadata_page_table` and `v4_prefix_counts`.
- Compact stream offsets come from `v2_page_offsets` and `v4_page_offsets`.
