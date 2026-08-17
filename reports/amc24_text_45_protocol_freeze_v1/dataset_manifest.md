# Dataset Manifest

Machine-readable source: `datasets/amc24_text_45/manifest.json`.

## Canonical Dataset

- Path: `datasets/amc24_text_45/amc24_text_45.jsonl`
- Rows: 45
- AMC12A rows: 22
- AMC12B rows: 23
- Canonical JSONL SHA256: `59a7450d9e480a41aa0d9db6dc2d89d16b1188cdf9a1ea8fd12e19dd2033c4b9`
- Upstream revision: `47b35303156a75cdfc6fcca694db66905d5b2033`

## Row Identity

Problem IDs are frozen as:

```text
12A_01 ... 12A_25 excluding 12A_14, 12A_18, 12A_22
12B_01 ... 12B_25 excluding 12B_07, 12B_19
```

Each row records:

```text
benchmark
problem_id
competition
year
problem_number
problem
choices
answer
answer_format
source
source_row_id
source_revision
metadata
```

`choices` is frozen as an empty list because the public source does not provide options.
