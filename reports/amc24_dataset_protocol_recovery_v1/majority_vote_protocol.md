# Majority Vote Protocol

## Paper-Recovered Semantics

PatternKV verifies:

```text
8 independent responses / problem
Avg@8 = per-sample accuracy averaged over eight responses
Maj@8 = problem-level accuracy under majority voting across eight responses
```

For `N` canonical problems, the paper-supported Avg@8 interpretation is:

```text
Avg@8 = correct responses / (N x 8)
```

The paper-supported Maj@8 interpretation is:

```text
extract one final answer from each response
perform majority voting over the eight extracted answers
score one prediction per problem
Maj@8 = correct majority predictions / N
```

## Tie Policy

`TIE_POLICY_SOURCE = UNRESOLVED`.

PatternKV does not publish majority tie handling. Since 8 is even, this is a material scoring detail.

## Existing Project Convention

The AIME helper `bench/aime_utils.py` returns no answer on a tie:

```text
valid answers -> counts -> top vote count
if exactly one winner: answer = winner
if multiple winners: answer = None and tie = true
```

This can be preregistered for AMC24 only after dataset identity is established and before GPU results are generated.

## Classification

```text
AVG8_SOURCE_STATUS = PAPER_VERIFIED
MAJ8_SOURCE_STATUS = PAPER_VERIFIED_PARTIAL
TIE_POLICY_SOURCE_STATUS = UNRESOLVED
```
