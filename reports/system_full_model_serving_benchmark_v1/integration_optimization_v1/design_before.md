# Design Before

The serving harness assembled a PatternKV batch before every decode step and split every returned layer cache back into per-request caches after the step. For B=1 this still rebuilt all 32 layer cache objects and performed row-slice materialization even though membership and row order were unchanged.
