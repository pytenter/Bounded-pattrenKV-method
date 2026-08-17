# Sampling Protocol

## Paper-Recovered Fields

PatternKV paper verifies:

```text
responses_per_problem = 8
responses are independent
```

## Project Existing AIME24 Protocol

The local AIME24 reproduction uses:

```text
do_sample = true
temperature = 0.6
top_p = 0.95
top_k = unresolved / not set
repetition_penalty = 1.0
max_new_tokens = 32768
max_model_len = model-context dependent
EOS handling = tokenizer/model EOS IDs normalized, stop_reason records eos/length
```

This is `PROJECT_CANONICAL_REUSED` only if a future AMC24 protocol explicitly chooses to reuse the AIME24 project protocol before seeing AMC24 results.

## PatternKV AMC Fields Not Recovered

The PatternKV paper/repository do not publish:

- temperature;
- top-p;
- top-k;
- min-p;
- repetition penalty;
- max new tokens;
- max model length;
- stop tokens;
- seed list;
- invalid parse behavior.

## Classification

```text
SAMPLING_PROTOCOL_STATUS = INSUFFICIENT_FOR_SUPPORTED_AMC24_RUN
```

The missing seed list alone would not block a future project run if preregistered. The missing dataset identity and answer format do block.
