# Batch Metadata Spec

| field | shape | role |
| --- | --- | --- |
| `seq_lens` | `[B]` | fixed-length MVP sequence lengths |
| `request_indptr` | `[B+1]` | request-to-logical-page offsets |
| `num_pages` | `[B]` | logical page count per request |
| `v2_page_table` / `v4_page_table` | `[B,num_pages]` | request-local logical page to compact physical page |
| `metadata_page_table` | `[B,num_pages]` | request-local logical page to metadata row |
| `precision_bitmap` | `[total_pages,4]` | 128 logical precision bits per page |
| `v2_counts` / `v4_counts` | `[total_pages]` | page-local compact stream counts |
| `valid_tokens` | `[total_pages]` | excludes final-page padding from attention |
| `v4_prefix_counts` | `[total_pages,129]` | correctness-MVP rank metadata |
