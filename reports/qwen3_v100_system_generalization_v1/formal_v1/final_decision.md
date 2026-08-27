# Final Decision

```json
{
  "FINAL_CLASSIFICATION": "QWEN_V100_RUNTIME_OVERHEAD_REMAINS_MATERIAL",
  "QWEN_V100_FULL_MODEL_DECODE_THROUGHPUT_ADVANTAGE": "NOT_SUPPORTED",
  "QWEN_V100_NEAR_FP16_DECODE_EFFICIENCY": "NOT_SUPPORTED",
  "batch_scaling": [
    {
      "CAUSAL_CV": 0.0016489269440294846,
      "CAUSAL_TPOT_mean_ms": 287.15916951497394,
      "CAUSAL_TPOT_median_ms": 287.1822204589844,
      "CAUSAL_over_FP16_TPOT": 7.032309595660624,
      "CAUSAL_over_FP16_throughput": 0.1422002042585417,
      "CAUSAL_tok_s": 3.482395545828352,
      "CAUSAL_valid_runs": 3,
      "FP16_CV": 0.002989418563713255,
      "FP16_TPOT_mean_ms": 40.83426157633463,
      "FP16_TPOT_median_ms": 40.80614471435547,
      "FP16_tok_s": 24.489384976528054,
      "FP16_valid_runs": 3,
      "batch": 1,
      "protocol_status": "PASS"
    },
    {
      "CAUSAL_CV": 0.0022629401094999063,
      "CAUSAL_TPOT_mean_ms": 147.4843292236328,
      "CAUSAL_TPOT_median_ms": 147.5458526611328,
      "CAUSAL_over_FP16_TPOT": 5.866244409376036,
      "CAUSAL_over_FP16_throughput": 0.17046587961242932,
      "CAUSAL_tok_s": 6.780404543839651,
      "CAUSAL_valid_runs": 3,
      "FP16_CV": 0.0036546632179752935,
      "FP16_TPOT_mean_ms": 25.141183853149414,
      "FP16_TPOT_median_ms": 25.137983322143555,
      "FP16_tok_s": 39.77572848745777,
      "FP16_valid_runs": 3,
      "batch": 2,
      "protocol_status": "PASS"
    },
    {
      "CAUSAL_CV": 0.0016236576791551997,
      "CAUSAL_TPOT_mean_ms": 73.64261372884114,
      "CAUSAL_TPOT_median_ms": 73.70524597167969,
      "CAUSAL_over_FP16_TPOT": 4.846051477381783,
      "CAUSAL_over_FP16_throughput": 0.2063526064995063,
      "CAUSAL_tok_s": 13.57911823748099,
      "CAUSAL_valid_runs": 3,
      "FP16_CV": 0.0030984550262556834,
      "FP16_TPOT_mean_ms": 15.196415901184082,
      "FP16_TPOT_median_ms": 15.204992294311523,
      "FP16_tok_s": 65.80541175530767,
      "FP16_valid_runs": 3,
      "batch": 4,
      "protocol_status": "PASS"
    },
    {
      "CAUSAL_CV": 0.009207080109467261,
      "CAUSAL_TPOT_mean_ms": 37.46820322672526,
      "CAUSAL_TPOT_median_ms": 37.662384033203125,
      "CAUSAL_over_FP16_TPOT": 3.358169924547378,
      "CAUSAL_over_FP16_throughput": 0.29779630321013345,
      "CAUSAL_tok_s": 26.690813340067635,
      "CAUSAL_valid_runs": 3,
      "FP16_CV": 0.003059319050122674,
      "FP16_TPOT_mean_ms": 11.157327969868978,
      "FP16_TPOT_median_ms": 11.140992164611816,
      "FP16_tok_s": 89.62775243463598,
      "FP16_valid_runs": 3,
      "batch": 8,
      "protocol_status": "PASS"
    }
  ],
  "context_scaling": [
    {
      "CAUSAL_CV": 0.1359526757825726,
      "CAUSAL_TPOT_mean_ms": 303.0286051432292,
      "CAUSAL_TPOT_median_ms": 279.81964111328125,
      "CAUSAL_over_FP16_TPOT": 6.645190000546258,
      "CAUSAL_over_FP16_throughput": 0.15030575857521783,
      "CAUSAL_tok_s": 3.338160781850362,
      "CAUSAL_valid_runs": 3,
      "FP16_CV": 0.14290088662320366,
      "FP16_TPOT_mean_ms": 45.60119501749674,
      "FP16_TPOT_median_ms": 42.31513595581055,
      "FP16_tok_s": 22.209134323884467,
      "FP16_valid_runs": 3,
      "context": 2048,
      "protocol_status": "PASS"
    },
    {
      "CAUSAL_CV": 0.0016783551196673844,
      "CAUSAL_TPOT_mean_ms": 342.6211446126302,
      "CAUSAL_TPOT_median_ms": 342.8703918457031,
      "CAUSAL_over_FP16_TPOT": 6.737850003847189,
      "CAUSAL_over_FP16_throughput": 0.14841504961402788,
      "CAUSAL_tok_s": 2.9186811593819963,
      "CAUSAL_valid_runs": 3,
      "FP16_CV": 0.0023047902195345166,
      "FP16_TPOT_mean_ms": 50.85021845499674,
      "FP16_TPOT_median_ms": 50.797054290771484,
      "FP16_tok_s": 19.665668454596727,
      "FP16_valid_runs": 3,
      "context": 4096,
      "protocol_status": "PASS"
    },
    {
      "CAUSAL_CV": null,
      "CAUSAL_TPOT_mean_ms": null,
      "CAUSAL_TPOT_median_ms": null,
      "CAUSAL_over_FP16_TPOT": null,
      "CAUSAL_over_FP16_throughput": null,
      "CAUSAL_tok_s": null,
      "CAUSAL_valid_runs": 0,
      "FP16_CV": 0.004399488436711046,
      "FP16_TPOT_mean_ms": 70.4093017578125,
      "FP16_TPOT_median_ms": 70.27852630615234,
      "FP16_tok_s": 14.202851745520709,
      "FP16_valid_runs": 3,
      "context": 8192,
      "protocol_status": "INVALID"
    }
  ],
  "formal_protocol_gate": {
    "invalid_runs": [
      {
        "CAUSAL_valid_runs": 0,
        "FP16_valid_runs": 3,
        "context": 8192,
        "protocol_status": "INVALID",
        "table": "context_scaling"
      }
    ],
    "status": "PASS_WITH_REPORTED_INVALID_POINTS"
  },
  "fp16_backend_selected": "sdpa",
  "gpu7_anchor": [
    {
      "CAUSAL_CV": 0.0005528470956809372,
      "CAUSAL_TPOT_mean_ms": 286.47532145182294,
      "CAUSAL_TPOT_median_ms": 286.47064208984375,
      "CAUSAL_over_FP16_TPOT": 5.444319331764898,
      "CAUSAL_over_FP16_throughput": 0.1836776052536274,
      "CAUSAL_tok_s": 3.4907027896404226,
      "CAUSAL_valid_runs": 3,
      "FP16_CV": 0.000999811263731026,
      "FP16_TPOT_mean_ms": 52.61912536621094,
      "FP16_TPOT_median_ms": 52.63298416137695,
      "FP16_tok_s": 19.00450947637497,
      "FP16_valid_runs": 3,
      "decode": 256,
      "protocol_status": "PASS"
    }
  ],
  "limitations": [
    "Tesla V100-SXM2-32GB only",
    "Qwen3-8B only",
    "fixed-batch formal protocol",
    "ragged Qwen true-batch smoke not dynamically closed in this experiment",
    "decode-only timing",
    "no new peak-memory evaluation",
    "no capacity/OOM evaluation",
    "legacy compressed CUDA backend",
    "cross-environment comparison to RTX3090/Llama is confounded"
  ],
  "long_decode": [
    {
      "CAUSAL_CV": 0.0021550570107069992,
      "CAUSAL_TPOT_mean_ms": 293.7997334798177,
      "CAUSAL_TPOT_median_ms": 293.547607421875,
      "CAUSAL_over_FP16_TPOT": 6.8835825283418,
      "CAUSAL_over_FP16_throughput": 0.14520334026482312,
      "CAUSAL_tok_s": 3.403689584728611,
      "CAUSAL_valid_runs": 3,
      "FP16_CV": 0.027025874852451188,
      "FP16_TPOT_mean_ms": 42.68122482299805,
      "FP16_TPOT_median_ms": 42.49304962158203,
      "FP16_tok_s": 23.440849077720472,
      "FP16_valid_runs": 3,
      "decode": 256,
      "protocol_status": "PASS"
    },
    {
      "CAUSAL_CV": 0.00021694492267649836,
      "CAUSAL_TPOT_mean_ms": 295.2067057291667,
      "CAUSAL_TPOT_median_ms": 295.2247009277344,
      "CAUSAL_over_FP16_TPOT": 7.264160105893533,
      "CAUSAL_over_FP16_throughput": 0.13766216641960036,
      "CAUSAL_tok_s": 3.387457032554041,
      "CAUSAL_valid_runs": 3,
      "FP16_CV": 0.00014482800197252105,
      "FP16_TPOT_mean_ms": 40.6387939453125,
      "FP16_TPOT_median_ms": 40.6384162902832,
      "FP16_tok_s": 24.607029808240288,
      "FP16_valid_runs": 3,
      "decode": 512,
      "protocol_status": "PASS"
    },
    {
      "CAUSAL_CV": 0.0021224201976756006,
      "CAUSAL_TPOT_mean_ms": 305.26373291015625,
      "CAUSAL_TPOT_median_ms": 305.2127685546875,
      "CAUSAL_over_FP16_TPOT": 7.298899688654804,
      "CAUSAL_over_FP16_throughput": 0.13700725480186884,
      "CAUSAL_tok_s": 3.2758657336870507,
      "CAUSAL_valid_runs": 3,
      "FP16_CV": 0.0010905679287978307,
      "FP16_TPOT_mean_ms": 41.8232536315918,
      "FP16_TPOT_median_ms": 41.82558059692383,
      "FP16_tok_s": 23.910162556168274,
      "FP16_valid_runs": 3,
      "decode": 1024,
      "protocol_status": "PASS"
    }
  ],
  "mean_relative_throughput": 0.1706009514729054,
  "status": "PASS_WITH_REPORTED_INVALID_POINTS"
}
```
