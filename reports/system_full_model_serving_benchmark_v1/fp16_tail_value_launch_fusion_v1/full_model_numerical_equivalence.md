# Full Model Numerical Equivalence

## Multi-Step Gate

`top1_match_all_steps = true`

| step | old top1 | fused top1 | logits relL2 | logits max abs |
|---:|---:|---:|---:|---:|
| 0 | 3249 | 3249 | 0.0008482704870402813 | 0.017578125 |
| 1 | 6303 | 6303 | 0.0006340397521853447 | 0.015625 |
| 2 | 47544 | 47544 | 0.0011096152011305094 | 0.0234375 |
| 3 | 374 | 374 | 0.0008293600403703749 | 0.017578125 |
| 4 | 70003 | 70003 | 0.0008906206348910928 | 0.0166015625 |
| 5 | 304 | 304 | 0.0021007393952459097 | 0.05859375 |
| 6 | 1403 | 1403 | 0.022798627614974976 | 0.5859375 |
| 7 | 64694 | 64694 | 0.001730669871903956 | 0.0390625 |

## Layer0 Trace Metrics

- Tail output max_abs: `3.0517578125e-05`
- Tail output relL2: `0.00038778127054683864`
- Attention pre-o-proj max_abs: `3.0517578125e-05`
- Attention pre-o-proj relL2: `0.0003747809096239507`
- Post-o-proj max_abs: `3.0517578125e-05`
- Post-o-proj relL2: `0.0003278797084931284`

## Conclusion

Full-model top1 semantics are preserved for the tested C2048 B1 decode sequence. Logit drift is nonzero but did not change top1 at any tested step.
