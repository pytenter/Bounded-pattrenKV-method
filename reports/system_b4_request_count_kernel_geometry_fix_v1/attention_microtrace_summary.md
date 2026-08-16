# B4 Attention Microtrace

{
  "B2_KERNEL_GEOMETRY": {
    "attention_width": 256,
    "batch": 2,
    "grid_blocks": [
      64,
      128,
      1
    ],
    "heads": 32,
    "num_pages": [
      1,
      2
    ],
    "pages_per_request": 2,
    "seq_lens": [
      128,
      256
    ],
    "threads": [
      256,
      1,
      1
    ],
    "v2_tokens": 288,
    "v4_tokens": 96
  },
  "B3_KERNEL_GEOMETRY": {
    "attention_width": 512,
    "batch": 4,
    "grid_blocks": [
      128,
      128,
      1
    ],
    "heads": 32,
    "num_pages": [
      1,
      2,
      3,
      4
    ],
    "pages_per_request": 4,
    "seq_lens": [
      128,
      256,
      384,
      512
    ],
    "threads": [
      256,
      1,
      1
    ],
    "v2_tokens": 960,
    "v4_tokens": 320
  },
  "FIRST_BAD_ATTENTION_SUBCOMPONENT": null
}
