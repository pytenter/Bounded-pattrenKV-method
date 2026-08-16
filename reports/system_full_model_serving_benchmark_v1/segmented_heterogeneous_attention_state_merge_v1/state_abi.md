# State ABI

- `STATE_O = o: [B,Hq,Q,D], torch.float32`.
- `STATE_M = m: [B,Hq,Q], torch.float32`.
- `STATE_L = l: [B,Hq,Q], torch.float32`.
- `FINAL_OUTPUT = finalize(o,m,l).to(query_states.dtype): [B,Hq,Q,D]`.
- Empty segments use `m=-inf`, `l=0`, `o=0`.
- Row-local empty segments are guarded to avoid NaN propagation.
- Merge formula is the exact online-softmax state merge:
  - `m=max(m_a,m_b)`
  - `alpha=exp(m_a-m)`
  - `beta=exp(m_b-m)`
  - `l=alpha*l_a+beta*l_b`
  - `o=alpha*o_a+beta*o_b`

The ABI is request-safe and reorder-safe because it contains only dense active-row tensors and uses existing request-local cache metadata for segment validity.

