# Same-Budget Random Evidence

Status: SUPPORTED_WITH_SCOPE.

`RANDOM_V4_25` and `CAUSAL_V4_25` use the same formal bit budget:

```json
{
  "CAUSAL_V4_25": {
    "effective_kv_bits_per_element_max": 2.50048828125,
    "effective_kv_bits_per_element_mean": 2.50048828125,
    "effective_kv_bits_per_element_min": 2.50048828125,
    "realized_v4_fraction_max": 0.25,
    "realized_v4_fraction_mean": 0.25,
    "realized_v4_fraction_min": 0.25
  },
  "FP16": {
    "effective_kv_bits_per_element_max": 16.0,
    "effective_kv_bits_per_element_mean": 16.0,
    "effective_kv_bits_per_element_min": 16.0,
    "realized_v4_fraction_max": null,
    "realized_v4_fraction_mean": null,
    "realized_v4_fraction_min": null
  },
  "PATTERN_BASE": {
    "effective_kv_bits_per_element_max": 2.25,
    "effective_kv_bits_per_element_mean": 2.25,
    "effective_kv_bits_per_element_min": 2.25,
    "realized_v4_fraction_max": null,
    "realized_v4_fraction_mean": null,
    "realized_v4_fraction_min": null
  },
  "RANDOM_V4_25": {
    "effective_kv_bits_per_element_max": 2.50048828125,
    "effective_kv_bits_per_element_mean": 2.50048828125,
    "effective_kv_bits_per_element_min": 2.50048828125,
    "realized_v4_fraction_max": 0.25,
    "realized_v4_fraction_mean": 0.25,
    "realized_v4_fraction_min": 0.25
  },
  "SAME_BIT_CONTROL_VALID": true,
  "same_bit_delta": 0.0
}
```

AIME24 aggregate: Random-25% `36/90` vs CAUSAL-V4@25% `45/90`.
The paired bootstrap CI for CAUSAL - Random crosses zero, so this is an aggregate advantage and same-budget control, not a 95% significance claim.
