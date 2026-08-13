# Capacity Backend Design

- `PATTERNKV_CACHE_GROWTH_BACKEND=baseline|fixed_capacity|chunked_capacity`.
- Default remains `baseline`.
- Fixed capacity default tokens: `32768`.
- Chunked grow size: `4096`.
- Capacity append writes only new slots with `copy_`; historical Value torch.cat is avoided.
