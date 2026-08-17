# GPU Budget Estimate

| P0 Item | Generation Count | Parallelization Candidate | Relative Cost |
| --- | --- | --- | --- |
| Selector ablation: 2 missing methods x 30 x 3 | 180 | AIME24 shard by method/seed/problem | LOW |
| AIME25: 4 methods x 30 x 8 | 960 | Shard by method and seed across available GPUs | HIGH |
| Second-backbone Qwen AIME24: 4 methods x 30 x 3 | 360 | Shard by method/seed/problem | MEDIUM |
| Total P0 | 1500 | Staged execution after offline plan approval | HIGH |
