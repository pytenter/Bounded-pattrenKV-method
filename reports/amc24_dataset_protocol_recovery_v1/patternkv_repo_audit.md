# PatternKV Repository Audit

## Repository

- URL: `https://github.com/HCOOOH/PatternKV`
- Local remote name: `origin`
- Inspected commit: `aba09a82e14732f6a0ed1f2b133925ff368d0538`
- Remote heads/tags: only `refs/heads/main`; no tags observed.

## Search Scope

The repository tree and commit history available from `origin/main` were searched for:

```text
AMC
AMC23
AMC24
AIME
Numina
Avg@8
Maj@8
majority
temperature
top_p
do_sample
max_new_tokens
DeepSeek
R1
GSM8K
```

## Findings

The public repository does not contain:

- AMC24 dataset preparation code;
- AMC24 runner;
- AMC24 parser;
- AIME/AMC Long-CoT evaluator;
- Avg@8 or Maj@8 implementation;
- prompt template for AMC;
- sampling configuration for AMC;
- seed policy for eight responses.

The only relevant generation example is `example.py`, which demonstrates generic `model.generate(..., max_new_tokens=512)`. It is not an AMC protocol source.

## Classification

`PATTERNKV_REPO_AMC_PROTOCOL = UNRESOLVED`

The official code is insufficient to recover the AMC24 dataset/protocol.
