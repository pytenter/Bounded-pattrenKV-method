# PatternKV / Long-CoT Protocol Audit

## Verdict

`BLOCKED_PROTOCOL_UNRESOLVED`.

The repository contains a reproducible DeepSeek-R1 AIME24 protocol, but no AMC24-specific protocol. Reusing AIME24 sampling parameters alone would not establish AMC24 comparability because AMC24 dataset identity, answer semantics, and aggregation definitions are absent.

## Existing Reusable Evidence

`reports/paper_repro_v2/aime24/experiment_protocol.md` records the existing DeepSeek-R1 long-CoT sampling configuration:

- tokenizer chat template with no system prompt;
- `force_think_prefix=true`;
- `do_sample=true`;
- temperature `0.6`;
- top-p `0.95`;
- maximum generation `32768`.

The same report explicitly says that PatternKV did not publish an exact AIME prompt string. It cannot therefore serve as proof of an AMC24 prompt or answer protocol.

## Unresolved AMC24 Protocol Fields

- AMC24 benchmark identity and source.
- Prompt wording and final-answer instruction.
- Chat-template rendering audit on AMC24 inputs.
- Maximum generation length justification.
- Eight-response seed list and response-index policy.
- Avg@8 definition for the chosen benchmark.
- Maj@8 majority/tie behavior.
- Any original PatternKV AMC-specific protocol.

## Required Unblock Artifact

Add a reviewed AMC24 protocol manifest that fixes all fields above before a runner or dataset is introduced. The manifest must be committed before Gate 2.
