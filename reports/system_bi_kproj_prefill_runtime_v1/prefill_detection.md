# Prefill Detection

Initial prefill is detected by an empty layer cache (`past_key_value is None`). Decode is detected by a non-empty PatternKV segmented cache, independent of `q_len`.
