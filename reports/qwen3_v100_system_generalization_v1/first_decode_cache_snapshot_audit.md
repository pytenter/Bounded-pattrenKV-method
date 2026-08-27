# first_decode_cache_snapshot_audit

```json
{
  "k_assignments": {
    "device": "cuda:0",
    "dtype": "torch.int64",
    "max": 31.0,
    "mean": 11.458984375,
    "min": 1.0,
    "present": true,
    "shape": [
      1,
      8,
      256
    ],
    "std": 7.452325344085693,
    "sum": 23468.0
  },
  "k_centroids": {
    "device": "cuda:0",
    "dtype": "torch.float16",
    "max": 31.59375,
    "mean": 0.050454653799533844,
    "min": -22.484375,
    "present": true,
    "shape": [
      8,
      32,
      128
    ],
    "std": 1.8687036037445068,
    "sum": 1653.298095703125
  },
  "layer": 7,
  "packed_k_tokens": 256,
  "packed_v4_tokens": 64,
  "packed_v_tokens": 256,
  "segment_order": [
    "sink",
    "packed",
    "pending",
    "recent"
  ],
  "stats": {
    "cache_mode": "segmented_rolling",
    "chunk_length": 128,
    "chunk_tokens": 0,
    "k_assignment_tokens": 256,
    "packed_history_tokens": 256,
    "pending_history_tokens": 113,
    "recent_tokens": 128,
    "sink_tokens": 16,
    "total_tokens": 513,
    "v_assignment_tokens": 256,
    "v_pattern_mask_tokens": 256
  },
  "v2_count": 192,
  "v4_count": 64,
  "v_assignment_idx": {
    "device": "cuda:0",
    "dtype": "torch.int32",
    "max": 27.0,
    "mean": 12.6923828125,
    "min": 0.0,
    "present": true,
    "shape": [
      1,
      8,
      256
    ],
    "std": 5.658417224884033,
    "sum": 25994.0
  },
  "v_centroids": {
    "device": "cuda:0",
    "dtype": "torch.float16",
    "max": 1.400390625,
    "mean": -0.0038887797854840755,
    "min": -1.6796875,
    "present": true,
    "shape": [
      8,
      32,
      128
    ],
    "std": 0.1824648380279541,
    "sum": -127.42753601074219
  },
  "v_precision_mask": {
    "device": "cuda:0",
    "dtype": "torch.uint8",
    "max": 1.0,
    "mean": 0.25,
    "min": 0.0,
    "present": true,
    "shape": [
      1,
      256
    ],
    "std": 0.4338609278202057,
    "sum": 64.0
  },
  "value_parts": [
    {
      "length": 16,
      "name": "sink"
    },
    {
      "length": 256,
      "name": "packed"
    },
    {
      "length": 113,
      "name": "pending"
    },
    {
      "length": 128,
      "name": "recent"
    }
  ]
}
```
