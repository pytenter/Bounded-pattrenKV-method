# Prompt Protocol

## PatternKV Source Status

`UNRESOLVED`.

The PatternKV paper and official repository do not publish an AMC24-specific prompt template, chat template rendering rule, assistant prefix, or reasoning instruction.

## Project Existing Protocol

The existing project AIME24 protocol uses:

```text
{problem}

Please reason step by step, and put your final answer within \boxed{}.
```

with tokenizer chat template, no system prompt, and `force_think_prefix=true`.

This is reusable project evidence only after AMC24 dataset identity is established. It is not evidence of the PatternKV AMC prompt.

## Newline Bug Audit

No AMC24 rendered prompt or tokenized prompt can be audited because no canonical AMC24 rows exist.

The future AMC24 protocol must explicitly record:

```text
manual assistant newline = true / false
duplicate newline = absent by construction
```

## Classification

```text
PROMPT_PROTOCOL_STATUS = UNRESOLVED
```
