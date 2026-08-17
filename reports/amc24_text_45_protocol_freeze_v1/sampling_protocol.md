# Sampling Protocol

Machine-readable source: `datasets/amc24_text_45/protocol.json`.

## Frozen Generation Settings

| Field | Value | Source |
| --- | --- | --- |
| `do_sample` | `true` | project canonical AIME24 reuse |
| `temperature` | `0.6` | project canonical AIME24 reuse |
| `top_p` | `0.95` | project canonical AIME24 reuse |
| `top_k` | `null` | project preregistered, not set |
| `repetition_penalty` | `1.0` | project canonical AIME24 reuse |
| `num_return_sequences` | `1` | project canonical AIME24 reuse |
| `max_new_tokens` | `32768` | project canonical AIME24 reuse |
| `max_model_len` | validate model/tokenizer context before generation | project protocol |
| `stop_policy` | normalized tokenizer/model EOS IDs; record `eos` or `length` | project canonical AIME24 reuse |

These settings are frozen for all four methods before any AMC24-Text result exists.
