# Page Table Design

- `DevicePageTable` stores CUDA page `data_ptr()` values in an int64 CUDA tensor.
- The table refreshes only when the tuple of page pointers changes; in-place page content updates reuse the existing table.
