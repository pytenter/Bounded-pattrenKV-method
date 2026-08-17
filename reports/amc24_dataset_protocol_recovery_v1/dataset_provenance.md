# Dataset Provenance

## Citation Chain

PatternKV cites AMC as Li et al. 2024a.

The full PatternKV bibliography entry for Li et al. 2024a identifies NuminaMath:

```text
Li, J., Beeching, E., Tunstall, L., Lipkin, B., Soletskyi, R.,
Huang, S., Rasul, K., Yu, L., Jiang, A. Q., Shen, Z., et al.
Numinamath: The largest public dataset in ai4maths with
860k pairs of competition math problems and solutions.
Hugging Face repository, 13(9):9, 2024a.
```

## Upstream Dataset Inspected

- Name: `AI-MO/NuminaMath-CoT`
- Hugging Face revision: `9d8d210c9f6a36c8f3cd84045668c9b7800ef517`
- Config: `default`
- Splits: `train`, `test`
- License: Apache-2.0
- Features:
  - `source`
  - `problem`
  - `solution`
  - `messages`

The dataset card reports:

- `train`: 859494 examples;
- `test`: 100 examples;
- source breakdown includes `amc_aime`: 4072 examples;
- source breakdown includes `synthetic_amc`: 62111 examples.

## Why This Does Not Freeze AMC24

The public dataset metadata does not expose:

- `competition`;
- `year`;
- `contest`;
- `AMC24`;
- `AMC23`;
- `AIME24`;
- answer label;
- final answer field separate from the solution text;
- source URL;
- source row ID stable across transformations;
- duplicate-removal policy for AMC24;
- train/test selection rule for AMC24.

The only primary-source field that can be filtered is `source`, such as `source == "amc_aime"`. That field groups AMC and AIME material across unknown contests and years, and cannot identify AMC24.

## Related Dataset Checked

`AI-MO/NuminaMath-TIR` was checked because it is derived from NuminaMath-CoT, but it is not the cited PatternKV dataset and its card states it selected approximately 70k numerical-output problems from NuminaMath-CoT. It does not resolve PatternKV AMC24 row identity.

## Classification

```text
PATTERNKV_AMC_SOURCE = PARTIALLY_VERIFIED
```

Verified:

- PatternKV AMC citation points to Li et al. 2024a.
- Li et al. 2024a is NuminaMath.
- NuminaMath-CoT revision can be pinned.

Unresolved:

- whether PatternKV used NuminaMath-CoT directly, a private filtered subset, or a derived evaluation file;
- exact AMC24 row-selection rule;
- exact problem count;
- exact source row IDs;
- exact ground truth format.

Because exact rows and ground truth are unresolved, no local canonical `datasets/amc24/amc24.jsonl` was created.
