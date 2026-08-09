# Mechanism-Guided Value Objective Screen

## Executive Summary

- Prior from Experiment 6: `VALUE_DOMINATED`.
- V-DIR and V-HYBRID are clean objective-rescoring candidates over the existing V centroid bank.
- V-CAUSAL-ATTN is not a meaningful same-candidate objective under current PatternKV V granularity because per-token scalar weights cancel from the independent per-token argmin.
- `FORMAL_VALUE_OBJECTIVE_RUN_APPROVED=false`.

## Gate

```json
{
  "baseline_reproduction_valid": true,
  "cache_semantics_valid": true,
  "causal_attention_no_leakage": true,
  "formal_value_objective_run_approved": false,
  "k_path_identical_across_value_configs": true,
  "no_nan_inf": true,
  "pseudo_importance_causal_valid": true,
  "reference_alignment_valid": true,
  "static_importance_matched_path_valid": false,
  "stop_reason": "V-CAUSAL-ATTN cannot produce a meaningful objective-rescoring intervention without changing the current per-token V decision granularity; static pack-time causal importance is not exposed by the matched Experiment 6 static path.",
  "v_causal_attn_effective_under_current_granularity": false,
  "value_candidate_set_invariant": true,
  "value_objective_hook_compatible": true,
  "value_objective_hook_compatible_scope": [
    "v_dir",
    "v_hybrid"
  ]
}
```

## Decision

No 8-GPU formal screen was launched. The correct next intervention is to first expose a coupled Value degree of freedom, such as tile-level assignment or selective Value precision, if causal attention weighting is to have an actual decision to influence.

## Safety

No VarN, Hadamard, Sink sweep, Recent sweep, K precision/objective, full AIME24, or AIME25 run was started.
