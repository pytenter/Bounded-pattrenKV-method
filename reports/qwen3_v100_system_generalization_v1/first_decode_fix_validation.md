# first_decode_fix_validation

```json
{
  "actual_vs_none_after_fix_rel_l2": 0.0,
  "first_divergent_component_after_fix": "layer_output",
  "first_divergent_layer_after_fix": 7,
  "first_step_logits_max_abs_after": 0.234375,
  "first_step_logits_rel_l2_after": 0.014264937490224838,
  "first_step_logits_rel_l2_before": 0.40075185894966125,
  "first_step_top1_parity_after": true,
  "mask_future_slot_min": -65504.0,
  "mask_shape": [
    1,
    1,
    1,
    514
  ],
  "mask_zero_count": 513,
  "qk_classification_after_fix": "PASS",
  "root_cause": "Qwen3 decode causal_mask can be one slot longer than the PatternKV logical cache after append. The compressed path right-aligned the mask, dropping the first sink token and retaining the masked future slot. The fix keeps the logical cache prefix and drops future slots.",
  "root_cause_classification": "ATTENTION_MASK_FUTURE_SLOT_ALIGNMENT_DRIFT",
  "softmax_classification_after_fix": "PASS",
  "status": "PASS_SINGLE_STEP_TOP1_AND_DRIFT_REDUCED",
  "value_full_reference_vs_compressed_rel_l2_after_fix": 0.00045475171646103263
}
```
