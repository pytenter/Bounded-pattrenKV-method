# Paper Quality Story

INT2 quantization error enters persistent KV state and is recursively reused by later autoregressive steps. The matched pseudo-decode evidence supports error accumulation over the tested AIME24 trajectories, and the routing/value forensic shows value-path propagation dominated the tested regime.

The primary quality result is AIME24: CAUSAL-V4@25% reaches `45/90` and matches FP16 aggregate accuracy under the three-seed protocol, while Pattern Base reaches `32/90` and same-budget Random reaches `36/90`. The CAUSAL-vs-Base paired bootstrap CI is positive; the CAUSAL-vs-Random CI crosses zero and should be described as an aggregate advantage only.

General reasoning evidence exists on GSM8K: CAUSAL reaches `1041/1319` (`78.9234%`), above FP16, PatternKV, and KIVI in aggregate. Long-context evidence exists on the 8K-capped LongBench 21x50 setup: CAUSAL averages `42.4657`, above PatternKV and KIVI but below FP16.

Efficiency evidence supports CAUSAL-V4@25% at approximately `2.50048828125` effective bit/KV element under project payload-and-metadata accounting. The 25% budget is a useful operating point in the current forensic budget curve, not a universal optimum.

Genuine remaining gaps are selector component quality ablations, AIME25 generalization, and second-backbone CAUSAL quality evidence.
