# AIME24 Pseudo-Decode Preflight Failure Report

## Summary

Preflight did not approve the formal run.

- Failure class: `NUMERICAL_TOLERANCE_ISSUE`
- Failed gate: `FP16_ZERO_ACCUMULATION_CONTROL_PASS=false`
- Formal run approved: `false`

All other preflight gates completed and passed in this run:

- `REFERENCE_TRAJECTORIES_VALID=true`
- `STATIC_INDEPENDENCE_PASS=true`
- `PSEUDO_FEEDBACK_PASS=true`
- `PSEUDO_PRODUCTION_PARITY_PASS=true`
- `OBSERVER_NONINVASIVE=true`
- `PAPER_CONFIG_PREFLIGHT_PASS=true`

## FP16 Zero-Gap Observations

The same-path FP16 pseudo repeat baseline was exactly zero at checkpoint 128:

- logit max abs diff: `0.0`
- hidden relative diff: `0.0`
- attention-output relative diff: `0.0`

Static-vs-pseudo FP16 replay was close but nonzero:

| checkpoint | hidden cosine | hidden relative L2 | next-token KL | top1 agreement | logit max abs diff |
| ---: | ---: | ---: | ---: | --- | ---: |
| 128 | `0.999998152256012` | `0.00175051752012223` | `1.8878992705140263e-05` | `true` | `0.015625` |
| 512 | `0.999996542930603` | `0.0024958793073892593` | `1.8352326947024267e-07` | `true` | `0.01953125` |
| 1024 | `0.9999971985816956` | `0.0023145321756601334` | `-7.11540906195296e-08` | `true` | `0.021484375` |

Because the prompt requires tolerance to be derived rather than relaxed ad hoc, this run keeps the zero-gap gate failed.

## Interpretation

The evidence suggests a cached-vs-full-prefix numerical path difference in FP16 replay rather than a task, model identity, tokenizer, or production-cache failure. The top-1 token agrees at all checked zero-gap checkpoints, and hidden cosine remains above `0.999996`, but the measured relative hidden/attention-output difference exceeds the current derived tolerance from same-path repeat noise.

## Required Follow-Up

Before formal long-run approval, define a repository-level FP16 cached-vs-full-prefix numerical baseline or an accepted tolerance protocol for Ampere RTX3090. Do not start the formal 72-trajectory pseudo-decode run until `FP16_ZERO_ACCUMULATION_CONTROL_PASS=true` under that protocol.

## Resolution Status

The original `FP16_ZERO_ACCUMULATION_CONTROL_PASS=false` remains preserved under the original protocol. The follow-up resolution is documented in `execution_path_baseline_resolution.md`, which defines the observed gap as an FP16 execution-path numerical baseline and switches formal quantization degradation to matched-path FP16 controls.

The updated formal gate excludes the legacy cross-path zero-gap condition, includes `FP16_EXECUTION_PATH_BASELINE_CHARACTERIZED`, `EXECUTION_PATH_BEHAVIOR_ACCEPTABLE`, `MATCHED_PATH_CONTROL_VALID`, and `MATCH_ALIGNMENT_VALID`, and now records `PREFLIGHT_COMPLETE=true` plus `FORMAL_RUN_APPROVED=true`. A formal long-run still requires a separate user instruction.
