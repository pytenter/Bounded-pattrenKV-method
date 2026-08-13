# Capacity Design

- `ContiguousCapacityBuffer` supports fixed capacity and grow-by-chunk modes.
- Append writes only new slots using `copy_` into preallocated storage while capacity allows.
- Fixed mode raises on overflow; chunked mode grows by `ceil(required/chunk)*chunk` and copies old valid region only on growth.
- The backend switch is `PATTERNKV_CACHE_GROWTH_BACKEND=baseline|fixed_capacity|chunked_capacity`, default `baseline`.
