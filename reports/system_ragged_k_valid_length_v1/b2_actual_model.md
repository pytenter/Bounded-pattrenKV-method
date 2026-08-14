# B2 Actual Model

{
  "b2_context_lengths": [
    384,
    513
  ],
  "b2_packed_k_valid_lengths": [
    128,
    256
  ],
  "b2_position_ids": [
    [
      384
    ],
    [
      513
    ]
  ],
  "decode_probe": {
    "attempted": true,
    "classification": "DECODE1_PASS",
    "error": null,
    "passed": true
  },
  "logit_comparison": {
    "A": {
      "cosine": 0.9681556224822998,
      "inf": 0,
      "max_abs": 5.0,
      "nan": 0,
      "relative_l2": 0.25869569182395935,
      "top1_equal": true,
      "top1_margin": 10.96875,
      "top5_overlap": 2
    },
    "B": {
      "cosine": 0.9999997615814209,
      "inf": 0,
      "max_abs": 0.015625,
      "nan": 0,
      "relative_l2": 0.0008930010953918099,
      "top1_equal": true,
      "top1_margin": 9.0703125,
      "top5_overlap": 5
    }
  },
  "physical_k_workspace_length": 256
}
