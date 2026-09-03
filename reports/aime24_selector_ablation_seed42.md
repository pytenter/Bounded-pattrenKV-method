# AIME24 Selector Ablation Seed42

This report summarizes the formal seed42 run for the AIME24 selector ablation.

Source result directories:

- `results/aime24_selector_ablation/importance_only/formal/seed42`
- `results/aime24_selector_ablation/error_only/formal/seed42`

## Summary

| method | samples | correct | accuracy | avg generated tokens | median generated tokens | min tokens | max tokens | length stops | parser errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `error_only25` | 30 | 15 | 50.00% | 12932.3 | 11387.5 | 2647 | 32768 | 2 | 0 |
| `importance_only25` | 30 | 14 | 46.67% | 13509.9 | 11806.5 | 2747 | 32768 | 3 | 2 |

## Disagreements

| problem_id | importance_only25 | importance answer | error_only25 | error answer | gold |
|---:|---|---:|---|---:|---:|
| 10 | correct | 104 | wrong | 873 | 104 |
| 14 | wrong | 2 | correct | 480 | 480 |
| 16 | wrong | 312 | correct | 468 | 468 |
| 20 | correct | 211 | wrong | 256 | 211 |
| 27 | wrong | 200 | correct | 699 | 699 |

## Notes

- `error_only25` is ahead by one problem on seed42: 15/30 vs. 14/30.
- `importance_only25` has one additional length stop and two parser errors, both on length-stopped samples.
- Full per-problem details are in `reports/aime24_selector_ablation_seed42.csv`.
