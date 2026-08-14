# Classification

`PATTERNKV_RAGGED_DECODE1_SEMANTIC_SUPPORTED`

Root cause: `RAGGED_SEGMENT_OFFSET_DIVERGENCE`

Before the fix, decode overflow from recent was appended at the physical pending tail for every row. In ragged [384,513], row A had a padded pending tail, so the new valid token landed after the padding while K/V masks assumed a valid prefix.
