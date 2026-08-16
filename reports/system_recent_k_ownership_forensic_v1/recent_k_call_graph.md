# Recent-K Call Graph

- `models/llama_patternkv.py:1129`: decode K projection is reshaped to `[B,Hkv,1,D]` and RoPE is applied.
- `models/llama_patternkv.py:1162`: segmented rolling decode calls `append_decode(cache, key_states, value_states)`.
- `models/segmented_cache.py:2469`: `append_decode_rolling` appends current K to `cache.recent_k` using active batch row layout.
- `models/segmented_cache.py:2486-2494`: overflow is rolled with `_roll_ragged_recent_overflow` for ragged request-local valid lengths.
- `models/segmented_cache.py:2424-2465`: `_roll_ragged_recent_overflow` rebuilds per-row recent/pending from valid logical prefixes.
