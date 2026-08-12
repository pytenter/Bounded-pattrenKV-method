# Page Size Audit

`PAGE_SIZE chosen: 128`

`PAGE_SIZE_128_COMPATIBLE = YES`

Reasons:

- Residual length is frozen at 128.
- Group size is frozen at 128.
- Current flush/quantization cadence naturally packs 128-token windows.
- Metadata (`v_precision_mask`, Pattern mask, assignment/index) is produced for the same 128-token quantization window.
- V2/V4 compact streams preserve compact order because the logical precision mask is paged independently from compact payload pages.
- Recent128 maps naturally to a fixed-capacity ring.

Potential downsides:

- Last-page internal fragmentation can be up to 127 slots per stream.
- Existing CUDA readers still expect giant contiguous tensors, so page-native CUDA is required for end-to-end benefit.
- Python object/list overhead grows with number of pages until descriptors are moved closer to a CUDA-facing representation.
