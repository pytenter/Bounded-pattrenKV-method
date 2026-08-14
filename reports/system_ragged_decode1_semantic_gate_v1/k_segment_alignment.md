# K Segment Alignment

{
  "assignment_shape": [
    2,
    8,
    256
  ],
  "packed_k_shape": [
    2,
    8,
    128,
    16
  ],
  "page_indptr": [
    0,
    1,
    3
  ],
  "page_seq_lens": [
    128,
    256
  ],
  "request_packed_k_tokens": [
    128,
    256
  ],
  "request_total_tokens": [
    384,
    513
  ],
  "segment_valid_lengths": {
    "packed": [
      128,
      256
    ],
    "pending": [
      112,
      113
    ],
    "recent": [
      128,
      128
    ],
    "sink": [
      16,
      16
    ]
  }
}

Root cause fixed: ragged decode append now compacts each row from logical valid recent/pending prefixes before rolling overflow, preserving valid-prefix semantics for short rows with padded physical tails.
