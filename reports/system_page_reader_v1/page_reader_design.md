# Page Reader Design

- `PATTERNKV_PAGE_V_READER=contiguous|paged_v2`; default is `contiguous`.
- `paged_v2` consumes device pointer tables for fixed pages and reads logical token `t` as `(page_id=t/page_size, page_offset=t%page_size)`.
- Historical pages are not concatenated or materialized in the reader wrapper.
