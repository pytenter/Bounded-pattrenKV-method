# Final Summary

- V100 micro FPR: `0.01527572135350836`
- V100 micro FNR: `0.14719587789295316`
- 4090 micro FPR: `0.014677032623701227`
- 4090 micro FNR: `0.14716749831655646`
- task macro uses equal-weight task averages; layer_head macro uses equal-weight task-layer-head averages; sample macro is not collected.
- top FNR tasks V100: `passage_retrieval_en, dureader, passage_retrieval_zh, hotpotqa, samsum, gsm8k`
- top FNR tasks 4090: `passage_retrieval_en, passage_retrieval_zh, dureader, hotpotqa, samsum, gsm8k`
- stable high FNR candidates: `0`
- stable high FNR low-support candidates: `361`
- FN opportunity: `not_collected`
- FP penalty: `not_collected`
- gate score / rho: `False` / `False`
- offline sweep readiness: `aggregate_only`
- pre-registered benefit-aware gate: `not_supported`
- recall-aware gate: `not_supported`
- next step: `Do not implement a new recall-aware gate yet; the current cross-hardware evidence does not support it.`
