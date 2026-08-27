# qk_oracle_comparison

```json
{
  "classification": "PASS",
  "overall_scaled": {
    "cosine": 0.9999998807907104,
    "dtype_a": "torch.float16",
    "dtype_b": "torch.float16",
    "max_abs": 0.03125,
    "mean_abs": 0.0004855337319895625,
    "present": true,
    "rel_l2": 0.0002510194026399404,
    "shape_a": [
      1,
      32,
      1,
      513
    ],
    "shape_b": [
      1,
      32,
      1,
      513
    ]
  },
  "overall_unscaled": {
    "cosine": 0.9999997615814209,
    "dtype_a": "torch.float16",
    "dtype_b": "torch.float16",
    "max_abs": 0.25,
    "mean_abs": 0.005510931834578514,
    "present": true,
    "rel_l2": 0.00022792837989982218,
    "shape_a": [
      1,
      32,
      1,
      513
    ],
    "shape_b": [
      1,
      32,
      1,
      513
    ]
  },
  "segments": [
    {
      "metrics": {
        "cosine": 0.9999999403953552,
        "dtype_a": "torch.float16",
        "dtype_b": "torch.float16",
        "max_abs": 0.0,
        "mean_abs": 0.0,
        "present": true,
        "rel_l2": 0.0,
        "shape_a": [
          1,
          32,
          1,
          16
        ],
        "shape_b": [
          1,
          32,
          1,
          16
        ]
      },
      "name": "sink"
    },
    {
      "metrics": {
        "cosine": 0.9999998807907104,
        "dtype_a": "torch.float16",
        "dtype_b": "torch.float16",
        "max_abs": 0.25,
        "mean_abs": 0.01104339025914669,
        "present": true,
        "rel_l2": 0.00033744462416507304,
        "shape_a": [
          1,
          32,
          1,
          256
        ],
        "shape_b": [
          1,
          32,
          1,
          256
        ]
      },
      "name": "packed"
    },
    {
      "metrics": {
        "cosine": 0.9999999403953552,
        "dtype_a": "torch.float16",
        "dtype_b": "torch.float16",
        "max_abs": 0.0,
        "mean_abs": 0.0,
        "present": true,
        "rel_l2": 0.0,
        "shape_a": [
          1,
          32,
          1,
          113
        ],
        "shape_b": [
          1,
          32,
          1,
          113
        ]
      },
      "name": "pending"
    },
    {
      "metrics": {
        "cosine": 0.9999999403953552,
        "dtype_a": "torch.float16",
        "dtype_b": "torch.float16",
        "max_abs": 0.0,
        "mean_abs": 0.0,
        "present": true,
        "rel_l2": 0.0,
        "shape_a": [
          1,
          32,
          1,
          128
        ],
        "shape_b": [
          1,
          32,
          1,
          128
        ]
      },
      "name": "recent"
    }
  ]
}
```
