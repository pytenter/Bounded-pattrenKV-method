# AMC24 Dataset Protocol Recovery V1

Status: `AMC24_DATASET_PROTOCOL_RECOVERY_V1_BLOCKED`.

This audit recovered the PatternKV paper-level AMC citation chain and metric definitions, but did not recover a canonical AMC24 dataset identity with exact rows and ground truth.

No AMC24 generation was run. No model was loaded. No GPU, CUDA, serving, memory, capacity, kernel, or runtime benchmark was started.

## Outcome

- PatternKV paper source was identified as arXiv `2510.05176v1`, "PatternKV: Flattening KV Representation Expands Quantization Headroom".
- PatternKV Table 2 includes `AIME25`, `AIME24`, `AMC24`, and `AMC23`.
- PatternKV explicitly states eight independent responses per problem.
- PatternKV explicitly defines `Avg@8` as per-sample accuracy averaged over eight responses.
- PatternKV explicitly defines `Maj@8` as problem-level accuracy under majority voting across eight responses.
- PatternKV cites AMC as Li et al. 2024a.
- Li et al. 2024a resolves to NuminaMath on Hugging Face.
- NuminaMath-CoT was pinned to Hugging Face revision `9d8d210c9f6a36c8f3cd84045668c9b7800ef517`.
- The public NuminaMath-CoT schema exposes only `source`, `problem`, `solution`, and `messages`; it does not expose competition, year, contest, answer label, source row identifier, or AMC24 split fields.
- The public PatternKV repository at `HCOOOH/PatternKV` commit `aba09a82e14732f6a0ed1f2b133925ff368d0538` contains no AMC/AIME/Long-CoT benchmark scripts or dataset scripts.

## Final Gate

The recovery is blocked because the essential scientific identity is not recoverable from available primary sources:

- exact AMC24 row-selection rule is unresolved;
- exact AMC24 problem count is unresolved;
- exact AMC24 source row IDs are unresolved;
- ground-truth answer format is unresolved;
- PatternKV AMC prompt and sampling implementation are not published;
- majority tie policy is not published.

The next valid action is continued provenance recovery, not AMC24 GPU generation.
