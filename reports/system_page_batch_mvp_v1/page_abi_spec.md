# Page ABI Spec

- `PAGE_SIZE=128`.
- Each request owns logical pages; page tables map request-local logical pages to physical V2/V4/metadata pages.
- V2 and V4 payloads have independent affine scale/zero streams.
- `precision_bitmap[num_pages,4]` stores 128 logical precision bits per page.
- `v4_prefix_counts[num_pages,129]` is correctness-MVP metadata for logical-to-compact rank.
