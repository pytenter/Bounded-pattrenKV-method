# Ragged Shape Contract

Current production cache serializes `total_tokens` as a scalar and the model builds batch-global decode positions. Ragged requires per-request `total_tokens`, `position_ids`, packed lengths, page ranges, and valid K/V lengths.
