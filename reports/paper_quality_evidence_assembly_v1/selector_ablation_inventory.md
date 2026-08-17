# Selector Ablation Inventory

| Variant | Implementation Exists | Test Exists | Quality Result Exists | Canonical Result Exists | Status |
| --- | --- | --- | --- | --- | --- |
| Random-25% | yes | yes | yes | yes | DIRECT_CONTROL |
| CAUSAL-25% | yes | yes | yes | yes | PRIMARY_METHOD |
| Importance-Only-25% | no separate selector found | no | no | no | MISSING_P0 |
| Error-Reduction-Only-25% | no separate selector found | no | no | no | MISSING_P0 |
| Oracle/Future | yes, forensic only | yes | forensic only | no task-quality canonical | SUPPLEMENTARY_NOT_DEPLOYABLE |

Selector implementation currently normalizes to `base_v2`, `all_v2`, `all_v4`, `random_v4`, `causal_v4`, and `oracle_v4`. Search hits sampled:

- No separate importance-only/error-only selector ablation artifacts found.
