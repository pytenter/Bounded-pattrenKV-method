# VarN Metadata Analysis

K metadata: s_col [1, group] per token-in-tile, s_row [D, 1] per channel; RTN scale/zero are absorbed into the per-channel axis.

V metadata: s_col [1, D] per channel, s_row [group, 1] per token-in-tile; RTN scale/zero are absorbed into the per-token axis.

For `head_dim=128`, `group=128`, fp16 scale metadata contributes:

```text
K scale/zero/second-scale bytes per tile = 768
V scale/zero/second-scale bytes per tile = 768
metadata bits per K/V element = 0.75
```

Calibration required: `False`.
