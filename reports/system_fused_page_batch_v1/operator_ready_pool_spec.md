# Operator Ready Pool Spec

See `pool_layout.json`. Physical pages are flattened into contiguous GPU pools with `int32` page offsets. Metadata remains request-local and device resident. Empty stream pages use offset `-1` and page table id `-1`.
