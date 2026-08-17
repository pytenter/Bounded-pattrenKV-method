# Long-CoT Error Mechanism

| Mechanism Claim | Status | Evidence |
| --- | --- | --- |
| Persistent KV quantization error exists | SUPPORTED_WITH_SCOPE | Matched pseudo/static formal audit over checkpoints 128-4096. |
| Autoregressive recursion accumulates error | SUPPORTED_WITH_SCOPE | Pattern S16 pseudo degradation grows beyond static degradation after 512 tokens. |
| Early error acts as accumulation seed | SUPPLEMENTARY | Sink16 reduces AUC for Pattern/KIVI in tested cohort. |
| Universal context behavior | NOT_SUPPORTED | Extended 8192/16384 matched static rows were hardware-limited and excluded. |
