# Masking Design

The segmented PatternKV attention path builds a vectorized `[B, K]` validity mask over sink, packed, pending, and recent segments and applies `masked_fill(-inf)` to QK scores before softmax.
