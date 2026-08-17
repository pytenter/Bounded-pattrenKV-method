# Prompt Protocol

Machine-readable source: `datasets/amc24_text_45/protocol.json`.

## Frozen Prompt

The user prompt template is:

```text
{problem}

Please reason step by step, and put your final answer within \boxed{}.
```

This reuses the canonical project AIME24 Long-CoT prompt shape from `bench/bench_aime24_patternkv.py`.

## Chat Template

```text
tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

System prompt: none.

Assistant suffix:

```text
<think>\n
```

## Choices

The public dataset does not provide choices, so the prompt contains problem text only and asks for a boxed final answer. Scoring compares the parsed final answer to the upstream `answer` string.

## Newline Rule

No manual blank line is appended after the chat template. The only assistant suffix is exactly `<think>\n`, appended once.
