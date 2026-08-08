# AIME24 Pseudo-Decode Preflight Validation Report

## 1. Executive Summary

- REFERENCE_TRAJECTORIES_VALID: `True`
- FP16_ZERO_ACCUMULATION_CONTROL_PASS: `False`
- STATIC_INDEPENDENCE_PASS: `True`
- PSEUDO_FEEDBACK_PASS: `True`
- PSEUDO_PRODUCTION_PARITY_PASS: `True`
- OBSERVER_NONINVASIVE: `True`
- PAPER_CONFIG_PREFLIGHT_PASS: `True`
- FORMAL_RUN_APPROVED: `False`

## 2. Git/Experiment Origin

- Source commit: `232e3b08d10919ca24932ad0a0135e46119ecfd5`
- Branch: `exp/aime-pseudodecode-3090-8gpu`

## 3. Portable Generation Semantics

- Portable generation hash: `86648d12304ce11890c1a8f64bf5a896`

## 4. FP16 Reference Generation

- Expected: `12`
- Actual: `12`
- Missing: `0`
- Duplicates: `0`
- Runtime errors: `0`

## 5. Reference Trajectory Manifest

See `reference_trajectories_manifest.json` and `.md`.

## 6. Checkpoint Availability

See `checkpoint_availability.csv`.

## 7. FP16 Same-Path Numerical Repeat Baseline

```json
{
  "attention_output_relative_diff": 0.0,
  "checkpoint": 128,
  "derived_kl_tolerance": 1e-07,
  "derived_logit_tolerance": 1e-05,
  "derived_relative_tolerance": 1e-05,
  "hidden_relative_diff": 0.0,
  "logit_max_abs_diff": 0.0
}
```

## 8. FP16 Static vs Pseudo Zero-Gap

Gate: `False`

## 9. Static Definition

Each static checkpoint is rebuilt from a fresh prefix by calling the production model path with clean state.

## 10. Static Independence Validation

Gate: `True`

## 11. Pattern STATIC State Reset

Pattern state is reset through repository runtime reset hooks before each static build.

## 12. KIVI STATIC State Reset

KIVI state is recreated by fresh model replay for each static build.

## 13. Pseudo-Decode Definition

Pseudo preflight teacher-forces the frozen reference tokens through cached production forward calls.

## 14. Quantized Feedback Validation

Gate: `True`

## 15. Production Cache Consumption Evidence

See `preflight_gate_summary.json`.

## 16. Pattern S0 Production Parity

`True`

## 17. Pattern S16 Production Parity

`True`

## 18. KIVI S0 Production Parity

`True`

## 19. KIVI S16 Production Parity

`True`

## 20. Paper Config Smoke

Gate: `True`

## 21. Observer OFF vs ON

Gate: `True`

## 22. Gate Decisions

```json
{
  "formal_run_approved": false,
  "fp16_zero_accumulation_control_pass": false,
  "generation_config_valid": true,
  "model_identity_valid": true,
  "observer_noninvasive": true,
  "paper_config_preflight_pass": true,
  "preflight_complete": false,
  "pseudo_feedback_pass": true,
  "pseudo_production_parity_pass": true,
  "reference_trajectories_valid": true,
  "source_commit_valid": true,
  "static_independence_pass": true,
  "task_manifest_valid": true,
  "tokenizer_identity_valid": true
}
```

## 23. Remaining Risks

This preflight only validates short prefixes and does not run the formal 72 pseudo trajectories or full static checkpoint matrix.

## 24. Formal Run Readiness

`FORMAL_RUN_APPROVED=False`. The current prompt still forbids starting the formal long-run.

## 25. Reproducibility

Reference token artifacts are stored under `artifacts/aime24_pseudodecode_3090/reference_tokens/`.
