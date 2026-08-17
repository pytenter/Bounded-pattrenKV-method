# Dataset Source Audit

## Public Source

- Provider: Hugging Face user `rawsh`
- Dataset: `rawsh/2024_AMC12`
- Repository: `https://huggingface.co/datasets/rawsh/2024_AMC12`
- Revision: `47b35303156a75cdfc6fcca694db66905d5b2033`
- File: `amc12-2024.jsonl`
- File SHA256 from pinned download: `3e020a1c03a42d9b846892ca92b4c7dc55490492f08cdc00df3c0379d2556a58`
- Dataset metadata format: JSON
- License field: not specified in HF API; README states MAA copyright for problems

The source README lists:

- `2024 AMC 12A` source page on Art of Problem Solving;
- `2024 AMC 12B` source page on Art of Problem Solving;
- removed figure problems: 12A problems 14, 18, 22; 12B problems 7, 19.

## Source Schema

Each upstream row contains:

```text
exam
problem_number
problem
answer
```

The public artifact does not contain multiple-choice options. Therefore AMC24-Text-45 scoring freezes the upstream `answer` string as ground truth, not an A/B/C/D/E choice label.

## Verification

The pinned upstream file contains exactly 45 JSONL rows:

- AMC12A: 22 rows
- AMC12B: 23 rows

This matches 50 original contest problem slots minus five README-declared figure exclusions.
