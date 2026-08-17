# AIME24 Selector Ablation Status

| Method | Selector | Correct | Accuracy | Status |
| --- | --- | --- | --- | --- |
| FP16 | reference | 45/90 | 50.00% | CANONICAL |
| Pattern Base | base_v2 | 32/90 | 35.56% | CANONICAL |
| Random-25% | random_v4 | 36/90 | 40.00% | CANONICAL |
| Importance-Only-25% | not implemented as separate selector | NOT_RUN | NOT_RUN | MISSING |
| Error-Reduction-Only-25% | not implemented as separate selector | NOT_RUN | NOT_RUN | MISSING |
| CAUSAL-25% | causal_v4 | 45/90 | 50.00% | CANONICAL |
