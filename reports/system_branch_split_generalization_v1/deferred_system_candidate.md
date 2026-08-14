# Deferred System Candidate

`DEFERRED_SYSTEM_CANDIDATE = PREFILL_LAST_TOKEN_LM_HEAD_OPTIMIZATION`

The reverted generalization commit included a production change in `models/llama_patternkv.py` that projected only `hidden_states[:, -1:, :]` through the LM head during cached inference prefill when:

- `labels is None`
- `use_cache == True`
- not training
- sequence length is greater than 1
- `PATTERNKV_FULL_PREFILL_LOGITS != 1`

This can reduce prefill logits memory substantially, but changes CausalLM forward output-shape/API behavior. It is preserved on `exp/causal-v4-25-generalization-v1` and should be evaluated later as an isolated system-performance commit with a semantic gate.
