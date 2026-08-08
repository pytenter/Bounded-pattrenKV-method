# Execution-Path Baseline Resolution

## 1. Executive Summary

The original FP16 zero-gap gate remains false. The observed full-prefix vs cached FP16 difference is treated as an execution-path numerical baseline, not quantization error. Formal quantization metrics now use matched-path FP16 controls.

## 2. Why the Original Zero-Gap Gate Failed

The original gate compared `FP16_static` and `FP16_pseudo` directly and assumed those execution paths should be near-identical. RTX3090 measurements showed a small, stable nonzero path difference.

## 3. Historical Zero-Gap Results

- `FP16_ZERO_ACCUMULATION_CONTROL_PASS=false` is preserved.

## 4. Same-Path Numerical Repeat

Same-path FP16 pseudo repeat remained deterministic in the prior preflight.

## 5. Full-Prefix vs Cached FP16 Execution

See `fp16_execution_path_baseline.csv`.

## 6. Multi-Task FP16 Execution-Path Baseline

- Baseline task count: `12`
- Checkpoints: `[128, 512, 1024, 2048, 4096]`

| checkpoint | n | median hidden relative L2 | max hidden relative L2 | median hidden cosine | median raw KL | median clamped KL | top1 disagreements |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 12 | `0.0020590152125805616` | `0.0027257332112640142` | `0.9999977946281433` | `9.85565134214994e-06` | `9.85565134214994e-06` | `0.0` |
| 512 | 12 | `0.0022946062963455915` | `0.0027414835058152676` | `0.999997079372406` | `1.6218278204860326e-06` | `1.6218278204860326e-06` | `0.0` |
| 1024 | 12 | `0.0018882196745835245` | `0.0025944013614207506` | `0.9999979138374329` | `1.5269413822238675e-07` | `1.5269413822238675e-07` | `0.0` |
| 2048 | 12 | `0.0021554502891376615` | `0.004808769561350346` | `0.9999974071979523` | `5.00247399060072e-08` | `5.00247399060072e-08` | `0.0` |
| 4096 | 12 | `0.0021809630561619997` | `0.003068686928600073` | `0.9999974370002747` | `8.903035464413733e-08` | `8.903035464413733e-08` | `0.0` |

## 7. Checkpoint Growth Analysis

- Runaway growth detected: `False`
- Behavior acceptable: `True`

## 8. Why Post-Hoc Tolerance Relaxation Was Rejected

No tolerance was relaxed to flip the old gate. The protocol now avoids cross-path comparison for quantization degradation.

## 9. Matched-Path Control Design

`STATIC: D(Q_static, FP16_static)` and `PSEUDO: D(Q_pseudo, FP16_pseudo)`.

## 10. Static Matched FP16 Definition

Fresh full-prefix FP16 replay is matched only with static quantized replay.

## 11. Pseudo Matched FP16 Definition

Cached teacher-forced FP16 replay is matched only with pseudo quantized replay.

## 12. Corrected Accumulation Metric

`E_acc = D(Q_pseudo, FP16_pseudo) - D(Q_static, FP16_static)`.

## 13. Metric-by-Metric Definitions

Hidden cosine is converted to loss; top1 is converted to disagreement; KL/JS are clamped at zero for roundoff while raw baseline values are retained.

## 14. Token/Checkpoint Alignment

Generated-token checkpoint, prompt offset, absolute position, next-token target, and trajectory SHA are recorded in `matched_path_control_audit.csv`.

## 15. Mini-Validation on Pattern S0/S16

See `matched_path_mini_validation.csv`.

## 16. Mini-Validation on KIVI S0/S16

See `matched_path_mini_validation.csv`.

## 17. Matched-Control Zero Conditions

- FP16 self-degradation zero: `True`

## 18. Updated Formal Gate

- FP16_EXECUTION_PATH_BASELINE_CHARACTERIZED: `True`
- EXECUTION_PATH_BEHAVIOR_ACCEPTABLE: `True`
- MATCHED_PATH_CONTROL_VALID: `True`
- MATCH_ALIGNMENT_VALID: `True`

## 19. Remaining Risks

This is still a protocol validation, not the formal 12-task x 6-config accumulation run.

## 20. Formal Run Readiness

`FORMAL_RUN_APPROVED=True`. Do not start the formal run without a separate instruction.

## 21. Reproducibility

All inputs are the frozen reference token artifacts committed under `artifacts/aime24_pseudodecode_3090/reference_tokens/`.
