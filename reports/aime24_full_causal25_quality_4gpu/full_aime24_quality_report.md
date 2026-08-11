# Full AIME24 Task-Quality Validation

- Repository: `pytenter/Bounded-pattrenKV-method`
- Branch: `exp/aime24-full-causal25-quality-4gpu`
- Parent: `83c46ed1252a32ca42dcb81e172bd3e4c0a060a0`
- Completed generations: `360/360`
- Classification: `SUPPORTED`

## Accuracy

| method | total | mean | std |
|---|---:|---:|---:|
| FP16 | 45/90 | 0.5 | 0.05773502691896256 |
| PATTERN_BASE | 32/90 | 0.3555555555555555 | 0.01924500897298752 |
| RANDOM_V4_25 | 36/90 | 0.39999999999999997 | 0.057735026918962595 |
| CAUSAL_V4_25 | 45/90 | 0.5 | 0.033333333333333326 |

## Accuracy By Seed

| method | seed | correct/30 | accuracy |
|---|---:|---:|---:|
| FP16 | 42 | 16/30 | 0.5333333333333333 |
| FP16 | 43 | 13/30 | 0.43333333333333335 |
| FP16 | 44 | 16/30 | 0.5333333333333333 |
| PATTERN_BASE | 42 | 11/30 | 0.36666666666666664 |
| PATTERN_BASE | 43 | 10/30 | 0.3333333333333333 |
| PATTERN_BASE | 44 | 11/30 | 0.36666666666666664 |
| RANDOM_V4_25 | 42 | 11/30 | 0.36666666666666664 |
| RANDOM_V4_25 | 43 | 11/30 | 0.36666666666666664 |
| RANDOM_V4_25 | 44 | 14/30 | 0.4666666666666667 |
| CAUSAL_V4_25 | 42 | 16/30 | 0.5333333333333333 |
| CAUSAL_V4_25 | 43 | 15/30 | 0.5 |
| CAUSAL_V4_25 | 44 | 14/30 | 0.4666666666666667 |

## Paired Bootstrap

- Unit: `question`
- Resamples: `10000`
- CAUSAL - RANDOM: `{'mean_delta': 0.10012444444444443, 'median_delta': 0.09999999999999999, 'ci95_low': -0.011111111111111108, 'ci95_high': 0.21111111111111105}`
- CAUSAL - BASE: `{'mean_delta': 0.14438888888888887, 'median_delta': 0.14444444444444443, 'ci95_low': 0.04444444444444444, 'ci95_high': 0.24444444444444444}`

## Notes

- Full raw generations are stored locally under ignored `raw_generations/`; committed records are compact.
- Runtime is recorded as a secondary implementation metric only.
- No AIME25/GPQA run is started by this experiment.
