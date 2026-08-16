# Step1 Layer0 Full BI Trace

{
  "layer0_attention_output": {
    "exact_equal": false,
    "max_abs": 3.814697265625e-06,
    "mismatch_count": 21,
    "rel_l2": 8.557456567359623e-06,
    "shape": [
      1,
      1,
      4096
    ]
  },
  "layer0_attention_pre_o_proj": {
    "exact_equal": false,
    "max_abs": 4.76837158203125e-07,
    "mismatch_count": 2,
    "rel_l2": 8.517238256899873e-07,
    "shape": [
      1,
      1,
      4096
    ]
  },
  "layer0_hidden_in": {
    "exact_equal": true,
    "max_abs": 0.0,
    "mismatch_count": 0,
    "rel_l2": 0.0,
    "shape": [
      1,
      1,
      4096
    ]
  },
  "layer0_hidden_out": {
    "exact_equal": false,
    "max_abs": 0.0001220703125,
    "mismatch_count": 402,
    "rel_l2": 0.00016997924831230193,
    "shape": [
      1,
      1,
      4096
    ]
  },
  "layer0_k_proj": {
    "exact_equal": true,
    "max_abs": 0.0,
    "mismatch_count": 0,
    "rel_l2": 0.0,
    "shape": [
      1,
      1,
      1024
    ]
  },
  "layer0_k_rope": {
    "exact_equal": true,
    "max_abs": 0.0,
    "mismatch_count": 0,
    "rel_l2": 0.0,
    "shape": [
      1,
      8,
      1,
      128
    ]
  },
  "layer0_mlp_norm": {
    "exact_equal": false,
    "max_abs": 0.0001220703125,
    "mismatch_count": 5,
    "rel_l2": 3.007010673172772e-05,
    "shape": [
      1,
      1,
      4096
    ]
  },
  "layer0_mlp_output": {
    "exact_equal": false,
    "max_abs": 6.103515625e-05,
    "mismatch_count": 859,
    "rel_l2": 0.0001745388435665518,
    "shape": [
      1,
      1,
      4096
    ]
  },
  "layer0_norm": {
    "exact_equal": true,
    "max_abs": 0.0,
    "mismatch_count": 0,
    "rel_l2": 0.0,
    "shape": [
      1,
      1,
      4096
    ]
  },
  "layer0_post_attention_residual": {
    "exact_equal": false,
    "max_abs": 1.52587890625e-05,
    "mismatch_count": 5,
    "rel_l2": 2.287360439368058e-05,
    "shape": [
      1,
      1,
      4096
    ]
  },
  "layer0_q_proj": {
    "exact_equal": true,
    "max_abs": 0.0,
    "mismatch_count": 0,
    "rel_l2": 0.0,
    "shape": [
      1,
      1,
      4096
    ]
  },
  "layer0_q_rope": {
    "exact_equal": true,
    "max_abs": 0.0,
    "mismatch_count": 0,
    "rel_l2": 0.0,
    "shape": [
      1,
      32,
      1,
      128
    ]
  },
  "layer0_v_proj": {
    "exact_equal": true,
    "max_abs": 0.0,
    "mismatch_count": 0,
    "rel_l2": 0.0,
    "shape": [
      1,
      1,
      1024
    ]
  }
}
