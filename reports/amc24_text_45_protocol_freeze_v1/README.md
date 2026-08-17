# AMC24-Text-45 Protocol Freeze V1

Status: `AMC24_TEXT_45_PROTOCOL_FREEZE_V1_SUPPORTED`.

This package freezes an independent public benchmark:

```text
benchmark_id = amc24_text_45
display_name = AMC24-Text
description = 2024 AMC 12A/12B text-only subset (45 problems)
```

It is not an exact reproduction of PatternKV's unpublished AMC24 protocol. PatternKV paper AMC24 values are retained only as `REFERENCE_ONLY`.

No model was loaded and no GPU generation was run.

Canonical artifacts:

- `datasets/amc24_text_45/amc24_text_45.jsonl`
- `datasets/amc24_text_45/manifest.json`
- `datasets/amc24_text_45/protocol.json`
- `datasets/amc24_text_45/README.md`

The protocol is frozen before any AMC24-Text method result observation. Any incompatible future change requires a protocol v2 and invalidates v1 generations for cross-protocol aggregation.
