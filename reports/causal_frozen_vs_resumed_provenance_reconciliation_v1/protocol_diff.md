# Protocol Diff

- Decode timing: matched in stored rows and reproduction; all relevant rows have zero timed prefill/refill/membership changes.
- Prefill: before timed decode in both protocols.
- Refill: zero in both protocols.
- Active batch: enabled in both protocols.
- Subprocess: one worker process per formal point in both protocols.
- Selective prefill: enabled in both protocols.
- Mixed V backend: `fused_page` for CAUSAL in both protocols.
- Cache mode: segmented rolling in both protocols.
- Memory lifecycle: different. Frozen runner sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; current paper wrapper does not. This difference reproduces the current B8 OOM.
- Long decode scope: frozen stored long decode is C2048/B1/D256; current paper comparison long decode is C4096/B1/D256, so those two long-decode rows are not a matched context comparison.
