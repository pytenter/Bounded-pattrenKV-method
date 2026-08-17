# AMC24 Dataset Audit

## Verdict

`BLOCKED_DATASET_IDENTITY_UNRESOLVED`.

No AMC24 dataset file, dataset manifest, download script, benchmark runner, result directory, problem-ID list, or ground-truth mapping was found in the repository working tree, in all accessible local branches/remotes/tags, or in the local `/data/zypan` dataset search.

## Searches Performed

- Working tree text search: `AMC24`, `AMC 2024`, `AMC_2024`, `amc24`, `AMC`, `Avg@8`, and `Maj@8`.
- File-name search under `datasets`, `data`, `results`, `reports`, `scripts`, `bench`, and `tests`.
- Ref-wide `git grep` over local heads, remote refs, and tags.
- Local filesystem search under `/data/zypan` for AMC/AMC24 data artifacts.

The only AMC24 statements found are the prior quality-audit inventories marking AMC24 as missing:

- `reports/paper_quality_evidence_assembly_v1/long_cot_benchmark_inventory.md`
- `reports/paper_quality_evidence_assembly_v1/tables/long_cot_benchmark_matrix.md`

## Missing Canonical Inputs

The following must exist and be frozen before any AMC24 generation is valid:

1. Dataset source and immutable revision/checksum.
2. Benchmark identity and split, including whether it is one AMC contest or a defined collection.
3. Exact problem count and stable problem IDs.
4. Ground-truth representation and option-to-answer mapping.
5. Evidence that this is the intended PatternKV-comparable AMC protocol.

The project must not construct a dataset by combining AMC10A, AMC10B, AMC12A, or AMC12B without an existing canonical protocol defining that composition.
